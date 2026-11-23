"""Camera simulation (Phase 2.3).

Models the image *formation chain* of a real capture, in the order the physics
happens:

    scene light -> lens (distortion, aberration, focus) -> sensor (resolution,
    noise) -> ISP (denoise, sharpen, white balance) -> encoding (JPEG)

Getting this order right is what separates a synthetic camera from "random
augmentation": sharpening after noise produces noise-amplified edges, JPEG after
sharpening produces ringing, and downsampling after blur produces realistic soft
text rather than aliased text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .imaging import (
    add_sensor_noise,
    chromatic_aberration,
    convolve,
    denoise,
    disc_kernel,
    gaussian_kernel,
    jpeg_compress,
    lens_distortion,
    motion_blur_kernel,
    resize,
    to_float,
    to_uint8,
    unsharp_mask,
)


@dataclass
class CameraConfig:
    """Resolved capture-chain parameters for one image."""

    # Sensor / resolution
    target_width: int = 1240
    target_height: int = 1754
    resolution_scale: float = 1.0     # renders at scale, then resamples to target

    # Lens
    lens_k1: float = 0.0
    lens_k2: float = 0.0
    chromatic: float = 0.0

    # Focus / motion
    defocus_sigma: float = 0.0
    motion_length: float = 0.0
    motion_angle: float = 0.0
    focus_error: float = 0.0          # 0..1, fraction of the card out of focus

    # Sensor
    iso: int = 100
    luminance_noise: float = 0.0
    chroma_noise: float = 0.0
    shot_noise: float = 0.0
    read_noise: float = 0.0

    # ISP
    sharpen_amount: float = 0.0
    denoise_strength: float = 0.0
    white_balance_gain: tuple[float, float] = (1.0, 1.0)

    # Encoding
    jpeg_quality: int = 90
    jpeg_subsampling: int = 0

    def to_dict(self) -> dict:
        return {
            "resolution": [self.target_width, self.target_height],
            "iso": self.iso,
            "lens": {"k1": round(self.lens_k1, 5), "k2": round(self.lens_k2, 6),
                     "chromatic": round(self.chromatic, 4)},
            "blur": {"defocus_sigma": round(self.defocus_sigma, 3),
                     "motion_length": round(self.motion_length, 3),
                     "motion_angle": round(self.motion_angle, 2),
                     "focus_error": round(self.focus_error, 3)},
            "noise": {"luminance": round(self.luminance_noise, 5),
                      "chroma": round(self.chroma_noise, 5),
                      "shot": round(self.shot_noise, 5),
                      "read": round(self.read_noise, 5)},
            "isp": {"sharpen": round(self.sharpen_amount, 3),
                    "denoise": round(self.denoise_strength, 3)},
            "jpeg_quality": self.jpeg_quality,
            "jpeg_subsampling": self.jpeg_subsampling,
        }


@dataclass
class CameraResult:
    """The captured frame plus the degradations applied."""

    image: np.ndarray
    config: CameraConfig
    effects: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"config": self.config.to_dict(), "effects": self.effects}


#: Pixels-per-card-width targets for common capture devices. A card is 63 mm
#: wide, so 600 px/width ~ 240 dpi. These set the *detail* floor.
RESOLUTION_PROFILES = {
    "phone_modern": (900, 1400),
    "phone_normal": (650, 1000),
    "phone_old": (420, 700),
    "phone_low_end": (260, 460),
    "scanner": (1400, 2200),
    "child": (300, 700),
}


def iso_noise_model(iso: float) -> tuple[float, float, float, float]:
    """Map ISO to ``(luminance, chroma, shot, read)`` noise magnitudes.

    Empirically ISO 100 is near-clean; ISO 3200 is visibly grainy with colour
    blotches. The shot-noise term grows faster than the read-noise floor, which
    reproduces the "shadow noise" users complain about.
    """
    ratio = max(iso, 50.0) / 100.0
    luminance = 0.006 * math.sqrt(ratio)
    chroma = 0.004 * ratio ** 0.8
    shot = 0.010 * math.sqrt(ratio)
    read = 0.0025 * ratio ** 0.6
    return luminance, chroma, shot, read


def sample_camera(
    rng: np.random.Generator,
    *,
    resolution: tuple[int, int],
    mode: str = "normal",
    device: str = "phone_normal",
    base_iso: int = 100,
) -> CameraConfig:
    """Draw a camera configuration for the given device class and mode.

    ``device`` selects the resolution/noise envelope (a scanner is clean and
    high-resolution; a "child camera" is shaky and low-resolution), and ``mode``
    scales how often severe degradations occur.
    """
    scale = {"clean": 0.2, "normal": 1.0, "poor": 1.9, "extreme": 3.2}.get(mode, 1.0)
    width, height = resolution

    iso = int(np.clip(rng.lognormal(math.log(max(base_iso, 50)), 0.45 * scale), 50, 12800))
    luminance, chroma, shot, read = iso_noise_model(iso)

    config = CameraConfig(
        target_width=width,
        target_height=height,
        resolution_scale=float(rng.uniform(0.75, 1.35)),
        iso=iso,
        luminance_noise=luminance,
        chroma_noise=chroma,
        shot_noise=shot,
        read_noise=read,
    )

    # Lens: phones have mild barrel distortion, scanners almost none.
    if device == "scanner":
        config.lens_k1 = float(rng.normal(0, 0.01))
        config.chromatic = float(abs(rng.normal(0, 0.15)))
    else:
        config.lens_k1 = float(rng.normal(-0.03, 0.04) * scale)
        config.lens_k2 = float(rng.normal(0, 0.01) * scale)
        config.chromatic = float(abs(rng.normal(0.5, 0.5) * scale))

    # Focus error and blur. Division of labour: defocus is optical, motion is
    # hand shake. Both get more likely and stronger with `scale`.
    focus_fail_chance = {"clean": 0.02, "normal": 0.18, "poor": 0.45, "extreme": 0.75}[mode]
    if rng.random() < focus_fail_chance:
        config.defocus_sigma = float(rng.uniform(0.6, 2.2) * scale)
        config.focus_error = float(rng.uniform(0.1, 1.0))
    elif rng.random() < 0.5:
        config.defocus_sigma = float(abs(rng.normal(0.25, 0.2)))

    shake_chance = {"clean": 0.03, "normal": 0.22, "poor": 0.5, "extreme": 0.8}[mode]
    if device == "child":
        shake_chance = min(0.95, shake_chance + 0.25)
    if rng.random() < shake_chance:
        config.motion_length = float(rng.uniform(1.5, 12.0) * scale)
        config.motion_angle = float(rng.uniform(0, 2 * math.pi))

    # ISP behaviour: cheap cameras denoise hard and sharpen hard.
    config.denoise_strength = float(np.clip(
        abs(rng.normal(0.25 if device != "scanner" else 0.05, 0.15)), 0.0, 0.8))
    config.sharpen_amount = float(np.clip(
        rng.normal(0.7 if device != "phone_low_end" else 1.3, 0.3), 0.0, 2.5))
    if rng.random() < 0.3:
        config.white_balance_gain = (float(rng.uniform(0.95, 1.05)),
                                     float(rng.uniform(0.95, 1.05)))

    # JPEG quality: modern phones sit near 90; tired/low-end ones near 60.
    quality_center = {"phone_modern": 92, "phone_normal": 88, "phone_old": 78,
                      "phone_low_end": 60, "scanner": 96, "child": 72}.get(device, 88)
    config.jpeg_quality = int(np.clip(rng.normal(quality_center, 6 * scale), 25, 98))
    config.jpeg_subsampling = 2 if config.jpeg_quality < 80 else (1 if config.jpeg_quality < 90
                                                                 else 0)
    return config


def apply_optics(image: np.ndarray, config: CameraConfig,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Lens distortion, chromatic aberration and focus blur."""
    out = to_float(image) if image.dtype != np.float32 else image
    if abs(config.lens_k1) > 1e-4 or abs(config.lens_k2) > 1e-5:
        out = lens_distortion(out, config.lens_k1, config.lens_k2)

    # Defocus first (optical), then motion (also optical but directional).
    if config.defocus_sigma > 0.05:
        # Mix Gaussian against a disc: real defocus has bokeh character.
        gauss = convolve(out, gaussian_kernel(config.defocus_sigma))
        if config.defocus_sigma > 1.5:
            disc = convolve(out, disc_kernel(config.defocus_sigma * 2.4))
            out = 0.5 * gauss + 0.5 * disc
        else:
            out = gauss
    if config.motion_length > 0.5:
        kernel = motion_blur_kernel(config.motion_length, config.motion_angle)
        out = convolve(out, kernel)

    if config.chromatic > 0.02:
        out = chromatic_aberration(out, config.chromatic)

    if config.white_balance_gain != (1.0, 1.0):
        from .imaging import white_balance_shift

        out = white_balance_shift(out, *config.white_balance_gain)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def apply_sensor(image: np.ndarray, config: CameraConfig,
                 rng: np.random.Generator) -> np.ndarray:
    """Noise, resolution loss and ISP post-processing."""
    out = image
    # Resolution: render was at ``resolution_scale``; sample down to target.
    if abs(config.resolution_scale - 1.0) > 0.02:
        scaled_w = max(1, int(round(config.target_width * config.resolution_scale)))
        scaled_h = max(1, int(round(config.target_height * config.resolution_scale)))
        out = resize(out, (scaled_w, scaled_h), resample="bilinear")
    if out.shape[1] != config.target_width or out.shape[0] != config.target_height:
        out = resize(out, (config.target_width, config.target_height),
                     resample="bilinear" if config.resolution_scale > 1 else "nearest")
        # Use area-like sampling to emulate pixel binning:
        out = resize(out, (config.target_width, config.target_height), resample="bilinear")

    out = add_sensor_noise(
        out,
        sigma=config.luminance_noise,
        read_noise=config.read_noise,
        chroma_sigma=config.chroma_noise,
        shot_scale=config.shot_noise,
        rng=rng,
    )
    if config.denoise_strength > 0.02:
        out = denoise(out, config.denoise_strength)
    if config.sharpen_amount > 0.02:
        out = unsharp_mask(out, amount=config.sharpen_amount, radius=1.0 + 0.5 * config.iso / 3200)
    return out


def apply_encoding(image: np.ndarray, config: CameraConfig) -> np.ndarray:
    """JPEG compression, returned as a float array (still 0..1).

    We keep it as float after decoding so the pipeline can continue adding
    artefacts; the true uint8 quantisation happens once at the very end.
    """
    if config.jpeg_quality >= 100:
        return to_float(to_uint8(image))
    return jpeg_compress(image, config.jpeg_quality, subsampling=config.jpeg_subsampling)


def capture(
    scene: np.ndarray,
    config: CameraConfig,
    rng: np.random.Generator,
    *,
    apply_jpeg: bool = True,
) -> CameraResult:
    """Run the complete capture chain over a rendered scene."""
    out = apply_optics(scene, config, rng)
    out = apply_sensor(out, config, rng)
    if apply_jpeg:
        out = apply_encoding(out, config)
    effects = {
        "iso": config.iso,
        "jpeg_quality": config.jpeg_quality,
        "blurred": bool(config.defocus_sigma > 0.05 or config.motion_length > 0.5),
        "distorted": bool(abs(config.lens_k1) > 1e-3),
        "noisy": bool(config.luminance_noise > 0.01),
        "sharpened": bool(config.sharpen_amount > 0.1),
    }
    return CameraResult(image=np.clip(out, 0.0, 1.0).astype(np.float32),
                        config=config, effects=effects)


def estimate_sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian - the standard cheap focus metric.

    Reported on the conventional **0-255 intensity scale** (not 0-1), so the
    numbers are comparable with published thresholds and with OpenCV-based tools.
    Roughly: < 50 is visibly soft, < 10 is unusable for small text, > 400 is
    crisp.

    Used by quality control to reject images where the text is unrecoverable,
    and by hard-example mining to label blur severity.
    """
    arr = to_float(image)
    gray = arr.mean(axis=-1) if arr.ndim == 3 else arr
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    gray = gray * 255.0
    laplacian = (
        4 * gray[1:-1, 1:-1]
        - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
    )
    return float(laplacian.var())


def estimate_noise(image: np.ndarray) -> float:
    """Rough noise estimate from a high-pass residual."""
    arr = to_float(image)
    gray = arr.mean(axis=-1) if arr.ndim == 3 else arr
    smooth = convolve(gray, gaussian_kernel(1.0))
    return float(np.std(gray - smooth))
