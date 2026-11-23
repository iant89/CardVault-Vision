"""Low-level image operations for the scan engine.

All operations are numpy/PIL only - no OpenCV - so generation runs on the same
minimal dependency set as the dataset engine (important: generation happens on
CPU workers, and a 100k-image run should not need a GPU container).

Colour convention: images are ``float32`` RGB in ``[0, 1]`` inside the pipeline.
Conversion to ``uint8`` happens exactly once, at the very end, after JPEG
compression - so quantisation error is not multiplied by later operations.

Every operation is written to be **content-aware**: blur is applied with a
synthetic PSF, perspective with a true homography, noise with a per-channel
model. Adding blur does not just downscale a crop.
"""

from __future__ import annotations

import io
import math

import numpy as np

# ---------------------------------------------------------------------------
# Colour / dtype conversion
# ---------------------------------------------------------------------------

def to_float(image: np.ndarray) -> np.ndarray:
    """uint8/uint16/float -> float32 in [0, 1]."""
    if image.dtype == np.float32:
        return image
    if image.dtype == np.uint8:
        return (image.astype(np.float32) / 255.0)
    if image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0
    arr = image.astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    return arr


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)


def to_gray(image: np.ndarray, weights: tuple[float, float, float] = (0.299, 0.587, 0.114)
            ) -> np.ndarray:
    if image.ndim == 2:
        return image
    r, g, b = weights
    return (image[..., 0] * r + image[..., 1] * g + image[..., 2] * b).astype(np.float32)


def rgb_to_hsv(image: np.ndarray) -> np.ndarray:
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc
    v = maxc
    s = np.where(maxc > 1e-8, delta / np.maximum(maxc, 1e-8), 0.0)
    h = np.zeros_like(maxc)
    mask = delta > 1e-8
    rmask = mask & (maxc == r)
    gmask = mask & (maxc == g) & ~rmask
    bmask = mask & ~rmask & ~gmask
    h[rmask] = ((g[rmask] - b[rmask]) / delta[rmask]) % 6.0
    h[gmask] = (b[gmask] - r[gmask]) / delta[gmask] + 2.0
    h[bmask] = (r[bmask] - g[bmask]) / delta[bmask] + 4.0
    return np.stack([h / 6.0, s, v], axis=-1).astype(np.float32)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0] * 6.0, hsv[..., 1], hsv[..., 2]
    i = np.floor(h)
    f = h - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i.astype(np.int32) % 6
    r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [v, q, p, p, t, v])
    g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [t, v, v, q, p, p])
    b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p, p, t, v, v, q])
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Resampling / geometry
# ---------------------------------------------------------------------------

def resize(image: np.ndarray, size: tuple[int, int], resample: str = "lanczos") -> np.ndarray:
    """Resize to ``(width, height)``.

    Accepts float or uint8; returns the same scale (float stays [0,1]).
    """
    from PIL import Image

    width, height = int(max(1, round(size[0]))), int(max(1, round(size[1])))
    filters = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    is_float = image.dtype == np.float32
    arr = to_uint8(image) if is_float else image
    # PIL cannot represent a HxWx1 array: drop the singleton channel and restore
    # it afterwards so callers can keep using consistent shapes.
    squeezed = arr.ndim == 3 and arr.shape[2] == 1
    if squeezed:
        arr = arr[..., 0]
    if arr.ndim == 3 and arr.shape[2] not in (3, 4):
        # Composite anything unusual into RGB rather than failing.
        arr = np.repeat(arr[..., :1], 3, axis=2)
    pil = Image.fromarray(arr)
    out = np.asarray(pil.resize((width, height), filters[resample]))
    if squeezed:
        out = out[..., None]
    return to_float(out) if is_float else out


def homography_from_corners(
    src: np.ndarray, dst: np.ndarray
) -> np.ndarray:
    """Solve the 3x3 homography mapping four ``src`` points to four ``dst`` points.

    Standard 8x8 DLT solve; raises if the configuration is degenerate (three
    collinear points), which is exactly the case that would produce a
    self-intersecting card quad.
    """
    src = np.asarray(src, dtype=np.float64).reshape(4, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(4, 2)
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.array(rows, dtype=np.float64)
    # Smallest singular vector is the solution (||h|| = 1).
    try:
        _, _, vt = np.linalg.svd(A)
    except np.linalg.LinAlgError as exc:  # pragma: no cover
        raise ValueError("Degenerate point configuration for homography") from exc
    h = vt[-1]
    if not np.all(np.isfinite(h)) or abs(h[8]) < 1e-12:
        raise ValueError("Degenerate homography")
    return (h / h[8]).reshape(3, 3)


def apply_homography(
    image: np.ndarray,
    matrix: np.ndarray,
    out_size: tuple[int, int],
    *,
    background: np.ndarray | None = None,
    resample: str = "bilinear",
    border_fill: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp ``image`` by ``matrix`` (mapping image coords -> output coords).

    Returns ``(warped, valid_mask)`` where ``valid_mask`` is 1 where the source
    image actually contributed (not background). Callers use the mask to
    composite over a scene and to decide whether the card is still visible.

    Implemented as an inverse-mapped bilinear sample. That is what gives us
    correct perspective for both the card pixels and the alpha channel in one
    step, avoiding PIL's square-only perspective transform.
    """
    width, height = int(out_size[0]), int(out_size[1])
    src = to_float(image) if image.dtype != np.float32 else image
    src_h, src_w = src.shape[:2]
    if background is None:
        if border_fill is None:
            background = np.zeros((height, width, 3), dtype=np.float32)
        else:
            background = np.full((height, width, 3), border_fill, dtype=np.float32)
    else:
        background = to_float(background)

    try:
        inverse = np.linalg.inv(np.asarray(matrix, dtype=np.float64))
    except np.linalg.LinAlgError as exc:  # pragma: no cover
        raise ValueError("Singular homography") from exc

    ys, xs = np.meshgrid(np.arange(height, dtype=np.float64) + 0.5,
                         np.arange(width, dtype=np.float64) + 0.5, indexing="ij")
    denom = inverse[2, 0] * xs + inverse[2, 1] * ys + inverse[2, 2]
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    us = (inverse[0, 0] * xs + inverse[0, 1] * ys + inverse[0, 2]) / denom
    vs = (inverse[1, 0] * xs + inverse[1, 1] * ys + inverse[1, 2]) / denom

    valid = (us >= 0) & (us <= src_w - 1) & (vs >= 0) & (vs <= src_h - 1)
    us_c = np.clip(us, 0, src_w - 1)
    vs_c = np.clip(vs, 0, src_h - 1)

    if resample == "nearest":
        ui = np.rint(us_c).astype(np.int32)
        vi = np.rint(vs_c).astype(np.int32)
        sampled = src[vi, ui]
    else:
        x0 = np.floor(us_c).astype(np.int32)
        y0 = np.floor(vs_c).astype(np.int32)
        x1 = np.minimum(x0 + 1, src_w - 1)
        y1 = np.minimum(y0 + 1, src_h - 1)
        fx = (us_c - x0)[..., None].astype(np.float32)
        fy = (vs_c - y0)[..., None].astype(np.float32)
        top = src[y0, x0] * (1 - fx) + src[y0, x1] * fx
        bottom = src[y1, x0] * (1 - fx) + src[y1, x1] * fx
        sampled = (top * (1 - fy) + bottom * fy).astype(np.float32)

    mask = valid[..., None].astype(np.float32)
    warped = sampled * mask + background * (1.0 - mask)
    return warped.astype(np.float32), valid.astype(np.uint8)


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to an ``(N, 2)`` array of points."""
    pts = np.asarray(points, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1))
    hom = np.hstack([pts, ones]) @ np.asarray(matrix, dtype=np.float64).T
    w = np.where(np.abs(hom[:, 2:3]) < 1e-12, 1e-12, hom[:, 2:3])
    return (hom[:, :2] / w)


def rotate_point(x: float, y: float, angle: float, cx: float, cy: float) -> tuple[float, float]:
    """Rotate a point by ``angle`` radians about ``(cx, cy)``."""
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    dx, dy = x - cx, y - cy
    return cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a


# ---------------------------------------------------------------------------
# Blur / sharpening
# ---------------------------------------------------------------------------

def gaussian_kernel(sigma: float, radius: float | None = None) -> np.ndarray:
    """Normalised 2-D Gaussian kernel."""
    if sigma <= 0.05:
        return np.array([[1.0]], dtype=np.float32)
    radius = radius if radius is not None else max(1.0, 3.0 * sigma)
    r = int(math.ceil(radius))
    xs = np.arange(-r, r + 1, dtype=np.float32)
    kernel_1d = np.exp(-(xs**2) / (2.0 * sigma * sigma))
    kernel_1d /= kernel_1d.sum()
    return np.outer(kernel_1d, kernel_1d).astype(np.float32)


def _is_radially_symmetric(kernel: np.ndarray, tolerance: float = 0.08) -> bool:
    """Whether a kernel is (near) invariant under 90-degree rotation.

    Gaussians and discs are; motion kernels are not. This is the cheap test that
    decides whether the box approximation is allowed.
    """
    if kernel.shape[0] != kernel.shape[1]:
        return False
    scale = float(np.abs(kernel).max()) or 1.0
    for candidate in (kernel.T, np.fliplr(kernel), np.flipud(kernel)):
        if np.abs(candidate - kernel).max() / scale > tolerance:
            return False
    return True


def _kernel_sigma(kernel: np.ndarray) -> float:
    """Estimate the Gaussian sigma of a normalised kernel (0 if not Gaussian-ish)."""
    total = float(kernel.sum())
    if total <= 0:
        return 0.0
    norm = kernel.astype(np.float64) / total
    ys, xs = np.mgrid[0:kernel.shape[0], 0:kernel.shape[1]]
    cx = (xs * norm).sum()
    cy = (ys * norm).sum()
    var = (((xs - cx) ** 2 + (ys - cy) ** 2) * norm).sum() / 2.0
    return float(np.sqrt(max(var, 0.0)))


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Fast Gaussian blur using box-pass approximation for large radii."""
    if sigma <= 0.05:
        return image
    # Map sigma to an equivalent box radius (3 passes): r ~ sigma * sqrt(12/3 + 1)/2
    radius = int(max(1, round(sigma * 1.1)))
    # Keep 2-D input 2-D: the single-channel box-blur path is markedly faster.
    return box_blur(np.asarray(image, dtype=np.float32), radius, passes=3)


def convolve(image: np.ndarray, kernel: np.ndarray, *, mode: str = "reflect") -> np.ndarray:
    """Separable-aware 2-D convolution.

    Kernels that are outer products of 1-D vectors are applied as two 1-D
    passes, which is ~2r times cheaper for large blur radii - the difference
    between a 100k-image run taking minutes and hours.
    """
    if kernel.size == 1:
        return image
    # Large *radially symmetric* kernels (Gaussians, defocus discs) are the common
    # case, and the box-pass approximation is both faster and visually equivalent.
    # Asymmetric kernels - motion PSFs in particular - must go through the exact
    # convolution below or motion blur would lose its direction.
    if max(kernel.shape) >= 7 and _is_radially_symmetric(kernel):
        sigma = _kernel_sigma(kernel)
        if sigma > 0:
            return gaussian_blur(image, sigma)
    arr = image
    squeeze = False
    if arr.ndim == 2:
        arr = arr[..., None]
        squeeze = True
    arr = arr.astype(np.float32, copy=False)

    row = kernel[kernel.shape[0] // 2]
    col = kernel[:, kernel.shape[1] // 2]
    outer = np.outer(col, row)
    is_separable = np.allclose(outer, kernel, atol=1e-6 * (np.abs(kernel).max() + 1e-9))
    try:
        if is_separable:
            arr = _convolve_1d(arr, col / col.sum(), axis=0, mode=mode)
            arr = _convolve_1d(arr, row / row.sum(), axis=1, mode=mode)
        else:
            arr = _convolve_2d(arr, kernel, mode=mode)
    except Exception:  # pragma: no cover - fallback to PIL for odd paths
        from PIL import Image, ImageFilter

        blurred = Image.fromarray(to_uint8(arr.squeeze() if squeeze else arr)).filter(
            ImageFilter.GaussianBlur(radius=max(0.5, kernel.shape[0] / 6))
        )
        arr = to_float(np.asarray(blurred))
        if not squeeze and arr.ndim == 2:
            arr = arr[..., None]
    return arr.squeeze(-1) if squeeze else arr


def _convolve_1d(arr: np.ndarray, weights: np.ndarray, *, axis: int, mode: str) -> np.ndarray:
    r = len(weights) // 2
    pad = [(0, 0)] * arr.ndim
    pad[axis] = (r, r)
    padded = np.pad(arr, pad, mode=mode)
    out = np.zeros_like(arr, dtype=np.float32)
    n = arr.shape[axis]
    for i, weight in enumerate(weights):
        if weight == 0:
            continue
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(i, i + n)
        out += weight * padded[tuple(sl)]
    return out


def _cumsum_box_blur(arr: np.ndarray, radius: int) -> np.ndarray:
    """Box blur via a summed-area table: O(1) per pixel regardless of radius.

    This is the workhorse behind every large blur in the engine. A 37x37 disc on a
    1 MP frame is a few milliseconds, where a direct gather convolution would be
    seconds and would allocate gigabytes.

    The single-channel case (2-D array, the common one - masks and grayscale
    fields) takes a dedicated path: it runs one 2-D cumsum and slices the table
    instead of gathering with four index arrays, which is several times faster.
    """
    if radius < 1:
        return arr
    size = 2 * radius + 1
    area = size * size

    if arr.ndim == 2:
        height, width = arr.shape
        # Work on a zero-padded copy: padding with zeros *after* the cumsum (the
        # previous approach) forced a second full-size allocation and copy, which
        # dominated this function on megapixel frames.
        integral = np.zeros((height + 2 * radius + 1, width + 2 * radius + 1),
                            dtype=np.float32)
        integral[radius + 1:radius + 1 + height, radius + 1:radius + 1 + width] = arr
        np.cumsum(integral, axis=0, out=integral)
        np.cumsum(integral, axis=1, out=integral)
        # Integral image convention: cumulative sums inclusive of the current cell.
        total = (integral[size:size + height, size:size + width]
                 - integral[:height, size:size + width]
                 - integral[size:size + height, :width]
                 + integral[:height, :width])
        return (total / area).astype(np.float32)

    height, width = arr.shape[:2]
    padded = np.pad(arr.astype(np.float32, copy=False),
                    ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    integral = np.cumsum(np.cumsum(padded, axis=0, dtype=np.float32), axis=1,
                         dtype=np.float32)
    integral = np.pad(integral, ((1, 0), (1, 0), (0, 0)), mode="constant")
    total = (integral[size:size + height, size:size + width]
             - integral[:height, size:size + width]
             - integral[size:size + height, :width]
             + integral[:height, :width])
    return (total / area).astype(np.float32)


def box_blur(arr: np.ndarray, radius: int, passes: int = 3) -> np.ndarray:
    """Approximate a Gaussian with successive box blurs (standard technique).

    Three box passes converge to a Gaussian to within a few percent, and the cost
    is independent of the radius - which is what makes large defocus affordable.
    """
    out = arr.astype(np.float32, copy=False)
    for _ in range(max(1, passes)):
        out = _cumsum_box_blur(out, radius)
    return out


def _convolve_2d(arr: np.ndarray, kernel: np.ndarray, *, mode: str) -> np.ndarray:
    """General 2-D correlation for small, non-separable kernels.

    Used for sparse, high-frequency kernels (motion PSFs, custom filters) where a
    box approximation would be wrong. Large kernels are routed to
    :func:`box_blur` by :func:`gaussian_kernel`/:func:`disc_kernel` callers, so
    this only ever sees modest sizes.

    Borders replicate (``mode='edge'``), matching PIL's behaviour.
    """
    kh, kw = kernel.shape
    ry, rx = kh // 2, kw // 2
    padded = np.pad(arr.astype(np.float32, copy=False), ((ry, ry), (rx, rx), (0, 0)),
                    mode="edge")
    h, w = arr.shape[:2]
    out = np.zeros((h, w, arr.shape[2]), dtype=np.float32)
    for i in range(kh):
        row = kernel[i]
        nonzero = np.nonzero(row)[0]
        if nonzero.size == 0:
            continue
        row_slice = padded[i:i + h]
        for j in nonzero:
            weight = float(row[j])
            out += weight * row_slice[:, j:j + w]
    return out


def motion_blur_kernel(length: float, angle: float, *, taper: bool = True) -> np.ndarray:
    """A linear-motion point-spread function.

    A real hand-shake blur is a *line* segment in the PSF, not a circle; using a
    Gaussian for motion blur teaches the model the wrong degradation.
    """
    length = max(1.0, float(length))
    size = int(math.ceil(length)) | 1  # odd
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    theta = float(angle)
    steps = max(int(length * 4), 8)
    for i in range(steps + 1):
        t = i / steps - 0.5
        x = center + t * length * math.cos(theta)
        y = center + t * length * math.sin(theta)
        weight = 1.0 - (2.0 * abs(t)) ** 2 if taper else 1.0
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < size and 0 <= yi < size:
            kernel[yi, xi] += max(weight, 0.05)
    total = kernel.sum()
    return kernel / total if total > 0 else np.array([[1.0]], dtype=np.float32)


def disc_kernel(diameter: float) -> np.ndarray:
    """Defocus PSF: a uniform disc (bokeh), not a Gaussian."""
    diameter = max(1.0, float(diameter))
    size = int(math.ceil(diameter)) | 1
    center = size / 2.0 - 0.5
    ys, xs = np.mgrid[0:size, 0:size]
    dist = np.sqrt((xs - center) ** 2 + (ys - center) ** 2)
    kernel = (dist <= diameter / 2.0).astype(np.float32)
    total = kernel.sum()
    return kernel / total if total > 0 else np.array([[1.0]], dtype=np.float32)


def unsharp_mask(image: np.ndarray, amount: float = 0.6, radius: float = 1.2,
                 threshold: float = 0.0) -> np.ndarray:
    """Camera-style digital sharpening.

    Applied after capture to mimic the in-camera/ISP sharpening that real
    phone JPEGs always carry - without it, synthetic images are systematically
    softer than real ones.
    """
    blurred = convolve(image, gaussian_kernel(radius))
    detail = image - blurred
    if threshold > 0:
        detail = np.where(np.abs(detail) >= threshold, detail, 0.0)
    return np.clip(image + amount * detail, 0.0, 1.0)


def denoise(image: np.ndarray, strength: float = 0.4) -> np.ndarray:
    """Mild edge-preserving-ish smoothing (ISP noise reduction approximation)."""
    if strength <= 0:
        return image
    blurred = convolve(image, gaussian_kernel(0.8))
    return np.clip(image * (1 - strength) + blurred * strength, 0.0, 1.0).astype(np.float32)


def depth_blur(image: np.ndarray, focus_depth: float, aperture: float,
               *, plane_depth: float = 0.5) -> np.ndarray:
    """Radially varying blur approximating shallow depth of field.

    Focus falls off with distance *from the card plane*, which for a
    top-down-ish shot means the background blurs more than the card. We build the
    blur as a weighted blend of two blur levels driven by a smooth radial mask.
    """
    sharp = image
    far_sigma = max(0.0, aperture * 6.0 * abs(plane_depth - focus_depth))
    if far_sigma < 0.3:
        return image
    blurred = convolve(image, gaussian_kernel(min(far_sigma, 8.0)))
    h, w = image.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    dist = np.sqrt(((xs - cx) / max(w, 1)) ** 2 + ((ys - cy) / max(h, 1)) ** 2)
    mask = np.clip((dist - 0.15) / 0.45, 0.0, 1.0)[..., None]
    return (sharp * (1 - mask) + blurred * mask).astype(np.float32)


# ---------------------------------------------------------------------------
# Sensor artefacts
# ---------------------------------------------------------------------------

def add_sensor_noise(
    image: np.ndarray,
    *,
    sigma: float = 0.0,
    read_noise: float = 0.0,
    chroma_sigma: float = 0.0,
    shot_scale: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add physically-motivated sensor noise.

    * ``shot_scale`` - photon shot noise, whose magnitude scales with the square
      root of signal (visible in shadows, invisible in highlights);
    * ``sigma`` - uniform luminance noise (ISO grain);
    * ``chroma_sigma`` - independent per-channel noise (high-ISO colour noise);
    * ``read_noise`` - additive baseline floor.
    """
    if sigma <= 0 and chroma_sigma <= 0 and read_noise <= 0 and shot_scale <= 0:
        return image
    rng = rng or np.random.default_rng(0)
    h, w = image.shape[:2]
    # One (h, w, 3) standard-normal draw, then scale it per term. Drawing four
    # separate arrays of different shapes costs more than the arithmetic itself.
    white = rng.standard_normal((h, w, 3)).astype(np.float32)
    noise = np.zeros_like(image)
    luminance_sigma = float(np.hypot(sigma, read_noise))
    if luminance_sigma > 0:
        # Luminance and read noise are both achromatic: share one channel slice.
        noise += white[:, :, :1] * luminance_sigma
    if chroma_sigma > 0:
        noise += white * chroma_sigma
    if shot_scale > 0:
        # Shot noise is signal-dependent: sqrt of the photon count.
        signal = np.maximum(np.clip(image, 0.0, 1.0), 1e-4)
        noise += white * shot_scale * np.sqrt(signal)
    return np.clip(image + noise, 0.0, 1.0).astype(np.float32)


def chromatic_aberration(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Radial R/B scaling - lens dispersion.

    Stronger at the frame edges, which is why cheap phone lenses show coloured
    fringes on the card border.
    """
    if strength <= 0.01:
        return image
    h, w = image.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    scale = 1.0 + 0.0015 * strength
    # Build the three coordinate maps, then do a single gather across channels.
    # Looping per channel triples the (dominant) floating-point work.
    factors = np.array([scale, 1.0, 2.0 - scale], dtype=np.float32)
    sx = cx + (xs[..., None] - cx) * factors          # H, W, 3
    sy = cy + (ys[..., None] - cy) * factors
    out = _bilinear_sample_channels(image, sx, sy)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def lens_distortion(image: np.ndarray, k1: float = 0.0, k2: float = 0.0) -> np.ndarray:
    """Barrel/pincushion radial distortion (Brown-Conrady, first two terms).

    Positive ``k1`` gives pincushion, negative gives barrel - the common case for
    wide phone lenses.
    """
    if abs(k1) < 1e-4 and abs(k2) < 1e-5:
        return image
    h, w = image.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    nx = (xs - cx) / max(cx, 1e-6)
    ny = (ys - cy) / max(cy, 1e-6)
    r2 = nx * nx + ny * ny
    factor = 1.0 + k1 * r2 + k2 * r2 * r2
    # The distortion is radial, so all three channels sample the same location.
    sx = cx + nx * factor * cx
    sy = cy + ny * factor * cy
    out = _bilinear_sample_channels(image, sx, sy)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _bilinear_sample_channels(image: np.ndarray, xs: np.ndarray,
                              ys: np.ndarray) -> np.ndarray:
    """Bilinear gather of a 3-channel image at per-channel coordinates.

    ``xs``/``ys`` are ``(H, W)`` (the same location for every channel) or
    ``(H, W, C)`` (per-channel locations, as chromatic aberration needs).

    All channels are gathered in one pass. The channel index is broadcast into the
    fancy index explicitly; without it numpy would append a trailing axis and
    return ``(H, W, C, C)``.
    """
    height, width, channels = image.shape
    if xs.ndim == 2:
        xs = np.repeat(xs[..., None], channels, axis=2)
        ys = np.repeat(ys[..., None], channels, axis=2)

    xs = np.clip(xs, 0, width - 1)
    ys = np.clip(ys, 0, height - 1)
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = (xs - x0).astype(np.float32)
    fy = (ys - y0).astype(np.float32)
    channel = np.arange(channels, dtype=np.int32).reshape(1, 1, channels)

    top = image[y0, x0, channel] * (1 - fx) + image[y0, x1, channel] * fx
    bottom = image[y1, x0, channel] * (1 - fx) + image[y1, x1, channel] * fx
    return (top * (1 - fy) + bottom * fy).astype(np.float32)


def _bilinear_sample(channel: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = channel.shape
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = xs - x0
    fy = ys - y0
    top = channel[y0, x0] * (1 - fx) + channel[y0, x1] * fx
    bottom = channel[y1, x0] * (1 - fx) + channel[y1, x1] * fx
    return (top * (1 - fy) + bottom * fy).astype(np.float32)


def jpeg_compress(image: np.ndarray, quality: int, *, subsampling: int = 0) -> np.ndarray:
    """Round-trip through a real JPEG encoder.

    Doing the actual encode/decode matters: block artefacts are a large part of
    what makes a real phone capture hard to read, and approximating them with
    noise does not reproduce 8x8 block structure.
    """
    from PIL import Image

    quality = int(np.clip(quality, 1, 100))
    arr = to_uint8(image)
    buffer = io.BytesIO()
    Image.fromarray(arr).save(buffer, format="JPEG", quality=quality, subsampling=subsampling)
    buffer.seek(0)
    with Image.open(buffer) as img:
        return to_float(np.asarray(img.convert("RGB")))


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Gamma adjustment on linear-ish values (``gamma<1`` brightens)."""
    if abs(gamma - 1.0) < 1e-3:
        return image
    return np.clip(np.power(np.clip(image, 0, 1), gamma), 0.0, 1.0).astype(np.float32)


def adjust_exposure(image: np.ndarray, stops: float) -> np.ndarray:
    """Exposure in photographic stops (2**stops gain)."""
    return np.clip(image * (2.0 ** stops), 0.0, 1.0).astype(np.float32)


def adjust_contrast(image: np.ndarray, factor: float, pivot: float = 0.5) -> np.ndarray:
    return np.clip((image - pivot) * factor + pivot, 0.0, 1.0).astype(np.float32)


def adjust_saturation(image: np.ndarray, factor: float) -> np.ndarray:
    hsv = rgb_to_hsv(image)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0.0, 1.0)
    return hsv_to_rgb(hsv)


_REFERENCE_KELVIN = 6500.0


def _blackbody_rgb(kelvin: float) -> np.ndarray:
    """Approximate the RGB of a blackbody radiator (Tanner Helland fit)."""
    temp = max(kelvin, 1000.0) / 100.0
    if temp <= 66:
        red = 255.0
        green = 99.4708025861 * math.log(temp) - 161.1195681661
        blue = 0.0 if temp <= 19 else 138.5177312231 * math.log(temp - 10) - 305.0447927307
    else:
        red = 329.698727446 * ((temp - 60) ** -0.1332047592)
        green = 288.1221695283 * ((temp - 60) ** -0.0755148492)
        blue = 255.0
    return np.array([red, green, blue], dtype=np.float32) / 255.0


def colour_temperature(image: np.ndarray, kelvin: float, *, strength: float = 1.0) -> np.ndarray:
    """Apply a light-source colour cast at ``kelvin``, normalised to 6500K.

    2700 K (warm tungsten) lifts red and drops blue; 6500 K is neutral; 9000 K
    (open shade / overcast) does the reverse. Relative to the 6500 K reference
    the gains are identity at neutral, so this never shifts a neutral image.
    """
    if abs(kelvin - _REFERENCE_KELVIN) < 50.0 or strength <= 0:
        return image
    gains = _blackbody_rgb(kelvin) / np.maximum(_blackbody_rgb(_REFERENCE_KELVIN), 1e-6)
    gains = gains / max(float(gains.mean()), 1e-6)
    gains = 1.0 + (gains - 1.0) * float(strength)
    return np.clip(image * gains.astype(np.float32), 0.0, 1.0).astype(np.float32)


def white_balance_shift(image: np.ndarray, r_gain: float, b_gain: float) -> np.ndarray:
    """Explicit per-channel WB gains (the physical basis of colour casts)."""
    gains = np.array([r_gain, 1.0 - 0.5 * (r_gain - 1.0) - 0.5 * (b_gain - 1.0), b_gain],
                     dtype=np.float32)
    gains = np.clip(gains, 0.2, 3.0)
    return np.clip(image * gains, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------

def alpha_composite(foreground: np.ndarray, background: np.ndarray,
                    alpha: np.ndarray | float) -> np.ndarray:
    """``fg*alpha + bg*(1-alpha)`` with a broadcastable alpha."""
    a = np.asarray(alpha, dtype=np.float32)
    if a.ndim == 2:
        a = a[..., None]
    return np.clip(foreground * a + background * (1.0 - a), 0.0, 1.0).astype(np.float32)


def vignette(image: np.ndarray, strength: float = 0.15, *, power: float = 2.2) -> np.ndarray:
    """Radial darkening from the lens/lighting falloff."""
    if strength <= 0:
        return image
    h, w = image.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    dist = np.sqrt(((xs - cx) / max(cx, 1)) ** 2 + ((ys - cy) / max(cy, 1)) ** 2)
    mask = 1.0 - strength * np.clip(dist, 0, 1.2) ** power
    return np.clip(image * np.maximum(mask, 0.0)[..., None], 0.0, 1.0).astype(np.float32)


def gradient_field(
    width: int,
    height: int,
    *,
    angle: float,
    falloff: float = 1.0,
    center: float = 0.5,
) -> np.ndarray:
    """Smooth directional illumination field in [0, 1].

    Used for "light comes from one side" (phone flash, window, desk lamp) and
    for uneven ambient light across a page.
    """
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    nx = xs / max(width - 1, 1)
    ny = ys / max(height - 1, 1)
    proj = nx * math.cos(angle) + ny * math.sin(angle)
    field = proj - proj.min()
    span = field.max() if field.max() > 1e-6 else 1.0
    field = field / span
    field = 1.0 - np.clip(np.abs(field - center) * 2.0 * falloff, 0.0, 1.0)
    return field.astype(np.float32)


def blur_map(image: np.ndarray, sigma_map: np.ndarray) -> np.ndarray:
    """Apply spatially varying blur (used by shadows and depth of field)."""
    flat = float(np.mean(sigma_map))
    lo = convolve(image, gaussian_kernel(max(0.5, float(np.min(sigma_map)))))
    hi = convolve(image, gaussian_kernel(max(1.0, float(np.percentile(sigma_map, 90)))))
    weight = np.clip((sigma_map - sigma_map.min()) /
                     max(sigma_map.max() - sigma_map.min(), 1e-6), 0.0, 1.0)[..., None]
    blended = lo * (1 - weight) + hi * weight
    if not np.isfinite(blended).all():  # pragma: no cover
        return convolve(image, gaussian_kernel(max(0.5, flat)))
    return blended.astype(np.float32)


def soft_light(image: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Multiply by an illumination field (broadcast over channels)."""
    f = np.asarray(field, dtype=np.float32)
    if f.ndim == 2:
        f = f[..., None]
    return np.clip(image * f, 0.0, 1.0).astype(np.float32)


def screen_blend(base: np.ndarray, light: np.ndarray) -> np.ndarray:
    """Screen blend - the correct maths for adding light (glare, reflections)."""
    return np.clip(1.0 - (1.0 - base) * (1.0 - np.clip(light, 0.0, 1.0)), 0.0, 1.0).astype(
        np.float32)


def blend_modes(base: np.ndarray, top: np.ndarray, mode: str = "overlay") -> np.ndarray:
    """Photoshop-style blend modes used for foil and holo effects."""
    if mode == "normal":
        return top
    if mode == "screen":
        return screen_blend(base, top)
    if mode == "multiply":
        return (base * top).astype(np.float32)
    if mode == "add":
        return np.clip(base + top, 0.0, 1.0).astype(np.float32)
    if mode == "overlay":
        return np.where(base < 0.5, 2 * base * top,
                        1 - 2 * (1 - base) * (1 - top)).astype(np.float32)
    if mode == "soft_light":
        return np.clip(base + (2 * top - 1) * base * (1 - base), 0.0, 1.0).astype(np.float32)
    if mode == "color_dodge":
        return np.clip(base / np.maximum(1.0 - top, 1e-3), 0.0, 1.0).astype(np.float32)
    raise ValueError(f"Unknown blend mode {mode!r}")


def value_noise(width: int, height: int, scale: int, rng: np.random.Generator,
                octaves: int = 1) -> np.ndarray:
    """Smooth 2-D value noise in [0, 1] (cheap Perlin-like texture).

    Used for wood grain, carpet fibre, uneven illumination and card wear. Built
    by upsampling low-resolution noise and summing octaves.
    """
    out = np.zeros((height, width), dtype=np.float32)
    total_amplitude = 0.0
    for octave in range(max(1, octaves)):
        grid_w = max(2, int(width / max(scale / (2 ** octave), 1)))
        grid_h = max(2, int(height / max(scale / (2 ** octave), 1)))
        coarse = rng.random((grid_h, grid_w)).astype(np.float32)
        upscaled = resize(coarse[..., None], (width, height))[..., 0]
        amplitude = 1.0 / (2 ** octave)
        out += upscaled * amplitude
        total_amplitude += amplitude
    out /= total_amplitude
    lo, hi = float(out.min()), float(out.max())
    return ((out - lo) / (hi - lo)) if hi - lo > 1e-6 else np.full_like(out, 0.5)


def fbm(width: int, height: int, rng: np.random.Generator, *, base_scale: int = 32,
        octaves: int = 4) -> np.ndarray:
    """Fractal Brownian motion - organic texture (wood, stone, cloth)."""
    return value_noise(width, height, base_scale, rng, octaves=octaves)


def _radial_distance(width: int, height: int, cx: float | None = None,
                     cy: float | None = None) -> np.ndarray:
    """Radial distance from a centre, vectorised with two 1-D broadcasts.

    Building full ``mgrid`` arrays (the obvious approach) allocates two float
    arrays per call; this allocates one result and is the hot path for every
    vignette, flash falloff and glare disc.
    """
    cx = (width - 1) / 2.0 if cx is None else float(cx)
    cy = (height - 1) / 2.0 if cy is None else float(cy)
    xs = (np.arange(width, dtype=np.float32) - np.float32(cx))[None, :]
    ys = (np.arange(height, dtype=np.float32) - np.float32(cy))[:, None]
    return np.hypot(ys, xs)


def radial_mask(width: int, height: int, cx: float, cy: float, radius: float,
                softness: float = 0.3) -> np.ndarray:
    """Smooth elliptical mask (1 at centre, 0 outside), for glare/reflection."""
    dist = _radial_distance(width, height, cx=cx, cy=cy) / max(radius, 1e-3)
    inner = max(0.0, 1.0 - softness)
    mask = np.clip((1.0 - dist) / max(1.0 - inner, 1e-3), 0.0, 1.0)
    return (mask * np.sqrt(mask)).astype(np.float32)  # mask ** 1.5


def linear_mask(width: int, height: int, angle: float, *, softness: float = 0.6,
                offset: float = 0.0) -> np.ndarray:
    """Directional gradient mask in [0, 1] (specular streaks, shadows)."""
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    nx = (xs / max(width - 1, 1)) - 0.5
    ny = (ys / max(height - 1, 1)) - 0.5
    proj = nx * math.cos(angle) + ny * math.sin(angle) + 0.5 + offset
    lo, hi = 0.5 - softness / 2, 0.5 + softness / 2
    return np.clip((proj - lo) / max(hi - lo, 1e-3), 0.0, 1.0).astype(np.float32)


def scratch_texture(width: int, height: int, rng: np.random.Generator, count: int = 6,
                    max_length: float = 0.35) -> np.ndarray:
    """Thin bright/dark lines approximating surface scratches and print lines."""
    canvas = np.zeros((height, width), dtype=np.float32)
    for _ in range(count):
        y0 = rng.uniform(0, height)
        x0 = rng.uniform(0, width)
        angle = rng.uniform(0, math.pi)
        length = rng.uniform(0.2, 1.0) * max_length * max(width, height)
        thickness = rng.choice([1, 1, 2])
        steps = int(length * 2)
        for i in range(steps):
            t = i / max(steps - 1, 1)
            x = int(x0 + t * length * math.cos(angle))
            y = int(y0 + t * length * math.sin(angle))
            if 0 <= x < width and 0 <= y < height:
                canvas[max(0, y - thickness):y + thickness + 1,
                       max(0, x - thickness):x + thickness + 1] = rng.uniform(0.2, 1.0)
    return canvas
