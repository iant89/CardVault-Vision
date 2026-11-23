"""Card geometry simulation (Phase 2.2).

Turns a flat, clean card image into a *placed card* inside a camera frame:

    clean card  --(rigid pose + perspective)-->  quadrilateral in the frame

Everything here is projective geometry, so the ground truth follows
analytically rather than approximately: a field whose card-space box is ``roi``
is warped by the *same* homography as the card, so its annotation box is exact
up to resampling - not a heuristic estimate.

Design notes
------------
* The card's target quadrilateral is generated in the frame, and the homography
  that maps the source card rectangle onto it is solved by DLT. This is a true
  projective transform (perspective), not an affine skew.
* A physically-plausible camera model is used for the "tilt" component: the card
  lies on a plane, the camera is at a distance, and we rotate the plane about the
  camera axes. That produces the correct foreshortening law (far edge smaller)
  instead of arbitrary corner jitter.
* ``GeometryResult`` carries the homography *and* the inverse, because
  compositing needs forward mapping while annotation needs inverse mapping for
  clipping in card space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..errors import GenerationError
from ..logging_utils import get_logger
from .imaging import homography_from_corners, resize, to_float, to_uint8, transform_points

_LOG = get_logger("synthetic.geometry")

#: ISO/IEC 7810 ID-1 - the ratio every major trading card follows (63 x 88 mm).
CARD_ASPECT = 88.0 / 63.0  # height / width


@dataclass
class CardPose:
    """How the card sits in 3-D relative to the camera.

    Angles are in degrees. ``distance`` is measured in card widths, which makes
    the perspective strength independent of the rendered pixel size.
    """

    roll: float = 0.0        # in-plane rotation (card rotated on the table)
    pitch: float = 0.0       # top-away/bottom-away tilt
    yaw: float = 0.0         # left/right turn
    distance: float = 4.0    # camera distance in card widths
    scale: float = 1.0       # card width as a fraction of the frame width
    scale_max_height: float = 0.96  # cap on card height as a fraction of frame height

    def to_dict(self) -> dict:
        return {
            "roll": round(self.roll, 3),
            "pitch": round(self.pitch, 3),
            "yaw": round(self.yaw, 3),
            "distance": round(self.distance, 3),
            "scale": round(self.scale, 3),
        }


@dataclass
class GeometryResult:
    """The outcome of placing a card in a frame."""

    homography: np.ndarray           # card space (px) -> frame (px)
    inverse: np.ndarray              # frame -> card space
    corners: np.ndarray              # 4x2 card corners in the frame (TL,TR,BR,BL)
    frame_size: tuple[int, int]      # (width, height)
    card_size: tuple[int, int]       # (width, height) of the source card
    pose: CardPose
    cropped: bool = False
    bend: float = 0.0
    edge_warp: float = 0.0
    visibility: float = 1.0

    def to_dict(self) -> dict:
        return {
            "frame": {"width": self.frame_size[0], "height": self.frame_size[1]},
            "card": {"width": self.card_size[0], "height": self.card_size[1]},
            "pose": self.pose.to_dict(),
            "corners": [[round(float(x), 2), round(float(y), 2)] for x, y in self.corners],
            "cropped": bool(self.cropped),
            "bend": round(float(self.bend), 4),
            "edge_warp": round(float(self.edge_warp), 4),
            "visibility": round(float(self.visibility), 4),
            "homography": [[round(float(v), 8) for v in row] for row in self.homography],
        }

    def warp_points(self, points: np.ndarray) -> np.ndarray:
        """Map card-space points to the frame."""
        return transform_points(points, self.homography)

    def warp_boxes(self, boxes: list[tuple[float, float, float, float]]) -> list[np.ndarray]:
        """Map card-space boxes to frame-space quadrilaterals.

        Returns one ``(4, 2)`` array per input box. A box under perspective is a
        quadrilateral, so callers take the axis-aligned envelope when they need a
        rectangle (that is what OCR training crops want) or keep the quad when
        they need an exact mask.
        """
        out = []
        for x0, y0, x1, y1 in boxes:
            corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
            out.append(self.warp_points(corners))
        return out

    def envelope(self, quad: np.ndarray, *, clamp: bool = True) -> tuple[float, float, float, float]:
        """Axis-aligned envelope of a warped box, optionally clamped to the frame."""
        x0, y0 = quad.min(axis=0)
        x1, y1 = quad.max(axis=0)
        if clamp:
            x0 = max(0.0, x0)
            y0 = max(0.0, y0)
            x1 = min(float(self.frame_size[0]), x1)
            y1 = min(float(self.frame_size[1]), y1)
        return float(x0), float(y0), float(x1), float(y1)


def card_project(
    card_size: tuple[int, int],
    frame_size: tuple[int, int],
    pose: CardPose,
    *,
    offset: tuple[float, float] = (0.0, 0.0),
    edge_warp: float = 0.0,
    rng: np.random.Generator | None = None,
) -> GeometryResult:
    """Project a card through a pinhole camera at the given pose.

    The card is placed on the plane ``z = 0`` centred at the origin, expressed in
    units of card widths, then rotated by (roll, pitch, yaw) and translated to
    ``z = distance``. Perspective division by the resulting depth gives the
    correct foreshortening.

    ``offset`` shifts the card in normalised frame units (so
    ``(0.3, 0.0)`` pushes it noticeably off-centre); ``edge_warp`` bends the
    card edges slightly, simulating a card that is not perfectly flat.
    """
    width_px, height_px = int(card_size[0]), int(card_size[1])
    frame_w, frame_h = int(frame_size[0]), int(frame_size[1])
    if min(width_px, height_px, frame_w, frame_h) <= 0:
        raise GenerationError("Card and frame sizes must be positive")

    # Card corners in normalised plane coords, centred on the origin.
    aspect = height_px / width_px
    half_w = 0.5
    half_h = aspect / 2.0
    corners_3d = np.array(
        [[-half_w, -half_h, 0.0], [half_w, -half_h, 0.0],
         [half_w, half_h, 0.0], [-half_w, half_h, 0.0]],
        dtype=np.float64,
    )

    # Rotations: yaw (Y), pitch (X), roll (Z), applied in that order.
    rx, ry, rz = (math.radians(pose.pitch), math.radians(pose.yaw),
                  math.radians(pose.roll))
    rot_x = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)],
                      [0, math.sin(rx), math.cos(rx)]])
    rot_y = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0],
                      [-math.sin(ry), 0, math.cos(ry)]])
    rot_z = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0],
                      [0, 0, 1]])
    rotation = rot_z @ rot_x @ rot_y

    pts = corners_3d @ rotation.T
    pts[:, 2] += max(pose.distance, 0.5)

    # Project through the pinhole first. The rotation applied above is what
    # produces the perspective foreshortening; the scale is applied afterwards so
    # ``pose.scale`` means exactly "card width as a fraction of the frame width".
    proj = np.empty((4, 2), dtype=np.float64)
    proj[:, 0] = pts[:, 0] / pts[:, 2]
    proj[:, 1] = pts[:, 1] / pts[:, 2]

    # Solve for the scale factor that makes the *rotated* card fill the requested
    # fraction of the frame, so tilt does not change apparent size.
    span_x = proj[:, 0].max() - proj[:, 0].min()
    span_y = proj[:, 1].max() - proj[:, 1].min()
    if span_x <= 1e-9 or span_y <= 1e-9:
        raise GenerationError("Degenerate card projection (zero span)")
    max_h = (pose.scale_max_height or 0.96) * frame_h
    total = pose.scale * frame_w
    if total * (span_y / span_x) > max_h:
        total = max_h * (span_x / span_y)
    total = max(total, 8.0)

    proj[:, 0] = (proj[:, 0] - (proj[:, 0].max() + proj[:, 0].min()) / 2.0) / span_x * total
    proj[:, 1] = (proj[:, 1] - (proj[:, 1].max() + proj[:, 1].min()) / 2.0) / span_x * total
    proj[:, 0] += frame_w / 2.0 + offset[0] * frame_w
    proj[:, 1] += frame_h / 2.0 + offset[1] * frame_h

    # Slight, smooth edge bending: displace only the midpoints of the corners
    # via a small radial function, keeping the quad convex.
    if edge_warp > 0 and rng is not None:
        centre = proj.mean(axis=0)
        for i in range(4):
            direction = proj[i] - centre
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                continue
            wobble = edge_warp * rng.uniform(0.6, 1.4)
            proj[i] += direction / norm * wobble * 0.01 * min(frame_w, frame_h)

    if not _is_sane_quad(proj):
        raise GenerationError("Degenerate card projection (self-intersecting quad)")

    src = np.array([[0.0, 0.0], [float(width_px), 0.0],
                    [float(width_px), float(height_px)], [0.0, float(height_px)]],
                   dtype=np.float64)
    homography = homography_from_corners(src, proj)
    try:
        inverse = np.linalg.inv(homography)
    except np.linalg.LinAlgError as exc:  # pragma: no cover
        raise GenerationError("Non-invertible card homography") from exc

    from ..datasets.schema import Quad, quad_visible_fraction

    visibility = quad_visible_fraction(Quad(tuple((float(x), float(y)) for x, y in proj)),
                                       frame_w, frame_h)
    return GeometryResult(
        homography=homography,
        inverse=inverse,
        corners=proj,
        frame_size=(frame_w, frame_h),
        card_size=(width_px, height_px),
        pose=pose,
        cropped=visibility < 0.999,
        bend=0.0,
        edge_warp=edge_warp,
        visibility=visibility,
    )


def _is_sane_quad(points: np.ndarray) -> bool:
    """Reject self-intersecting or collinear corner sets."""
    if not np.all(np.isfinite(points)):
        return False
    area = 0.0
    for i in range(4):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % 4]
        area += x0 * y1 - x1 * y0
    if abs(area) / 2.0 < 1.0:
        return False
    # All cross products must share a sign for a convex quad.
    signs = []
    for i in range(4):
        a, b, c = points[i], points[(i + 1) % 4], points[(i + 2) % 4]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) > 1e-9:
            signs.append(cross > 0)
    return len(set(signs)) == 1


def sample_pose(
    rng: np.random.Generator,
    *,
    max_roll: float = 12.0,
    max_pitch: float = 18.0,
    max_yaw: float = 18.0,
    distance_range: tuple[float, float] = (2.2, 6.0),
    scale_range: tuple[float, float] = (0.55, 0.95),
) -> CardPose:
    """Draw a plausible pose.

    Roll is nearly uniform (users tilt the card freely), while pitch/yaw are
    drawn from a triangular-ish distribution - most captures are close to
    perpendicular, and extreme angles are rarer but must still occur. That shape
    matters: a uniform distribution would over-represent unusable captures and
    waste generation budget.
    """
    roll = float(rng.uniform(-max_roll, max_roll))
    pitch = float(rng.triangular(-max_pitch, 0.0, max_pitch))
    yaw = float(rng.triangular(-max_yaw, 0.0, max_yaw))
    distance = float(rng.uniform(*distance_range))
    scale = float(rng.uniform(*scale_range))
    return CardPose(roll=roll, pitch=pitch, yaw=yaw, distance=distance, scale=scale)


def rasterize_card(
    card: np.ndarray,
    result: GeometryResult,
    *,
    supersample: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Render the placed card into the frame as ``(rgb, alpha)``, both full-frame.

    Performance and quality both matter here, so the implementation is specific:

    * The warp is solved for the card's **bounding box** only, never the whole
      frame. A card covering a fifth of a 12 MP frame then costs a fifth of the
      work - this is the difference between ~2 s and ~50 ms per sample.
    * Rendering happens at ``supersample``x and is box-filtered down. Real camera
      pixels integrate light over their area; a hard-edged geometric warp looks
      artificially sharp and teaches the model wrong edge statistics.
    * Only the covered region is copied out, and the alpha channel records exactly
      where the card contributed.
    """
    from PIL import Image

    frame_w, frame_h = result.frame_size
    card_h, card_w = card.shape[:2]
    factor = max(1, int(supersample))

    # Bounding box of the card in the frame.
    corners = np.asarray(result.corners, dtype=np.float64)
    x0 = int(np.floor(corners[:, 0].min()))
    y0 = int(np.floor(corners[:, 1].min()))
    x1 = int(np.ceil(corners[:, 0].max()))
    y1 = int(np.ceil(corners[:, 1].max()))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(frame_w, x1), min(frame_h, y1)
    rgb = np.zeros((frame_h, frame_w, 3), dtype=np.float32)
    alpha = np.zeros((frame_h, frame_w), dtype=np.float32)
    if x1 <= x0 or y1 <= y0:
        return rgb, alpha

    box_w, box_h = x1 - x0, y1 - y0

    # Coefficients of the *inverse* warp into card space, from destination
    # (subsampled) coordinates. PIL's PERSPECTIVE maps output -> input, which is
    # exactly what we need, and it operates on float32 natively.
    inverse = np.asarray(result.inverse, dtype=np.float64)
    # Compose: dst_sub -> frame -> card.
    #   frame = dst_sub / factor + (x0, y0)
    #   card  = inverse @ frame
    a = inverse
    # Scale terms for x_sub and y_sub, and the translation for (x0, y0).
    coeffs = np.array([
        [a[0, 0] / factor, a[0, 1] / factor, a[0, 0] * x0 + a[0, 1] * y0 + a[0, 2]],
        [a[1, 0] / factor, a[1, 1] / factor, a[1, 0] * x0 + a[1, 1] * y0 + a[1, 2]],
        [a[2, 0] / factor, a[2, 1] / factor, a[2, 0] * x0 + a[2, 1] * y0 + a[2, 2]],
    ], dtype=np.float32)

    sub_w, sub_h = max(1, box_w * factor), max(1, box_h * factor)
    source = card if card.dtype == np.float32 else card.astype(np.float32) / 255.0
    source = np.clip(source, 0.0, 1.0).astype(np.float32)
    if source.ndim == 2:
        source = np.repeat(source[..., None], 3, axis=2)
    source_u8 = to_uint8(source)

    # One inverse-map transform for all channels: warp an RGBA image where the
    # alpha channel is a constant 255, then read the resulting alpha as coverage.
    # PIL's PERSPECTIVE takes the eight (h11..h32) coefficients of the
    # output -> input map, which is exactly our inverse homography.
    coefficients = tuple(float(v) for v in coeffs.reshape(9)[:8])
    rgba = np.concatenate([source_u8, np.full((card_h, card_w, 1), 255, np.uint8)], axis=2)
    warped_rgba = np.asarray(
        Image.fromarray(rgba).transform(
            (sub_w, sub_h), Image.Transform.PERSPECTIVE, coefficients,
            resample=Image.Resampling.BILINEAR,
        )
    )
    warped = to_float(warped_rgba[..., :3])
    sampled_alpha = warped_rgba[..., 3].astype(np.float32) / 255.0

    # Resample down from the supersampled render (box-like averaging).
    if factor > 1:
        warped = resize(warped, (box_w, box_h), resample="bilinear")
        sampled_alpha = resize(sampled_alpha[..., None], (box_w, box_h),
                               resample="bilinear")[..., 0]

    inside = sampled_alpha > 0.5
    rgb[y0:y1, x0:x1] = np.where(inside[..., None], np.clip(warped, 0, 1), 0.0)
    alpha[y0:y1, x0:x1] = np.clip(sampled_alpha, 0.0, 1.0)
    return rgb, alpha


def warp_card_to_frame(
    card: np.ndarray,
    result: GeometryResult,
    *,
    supersample: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Backwards-compatible alias for :func:`rasterize_card`."""
    return rasterize_card(card, result, supersample=supersample)


def warp_field_boxes(
    result: GeometryResult,
    fields: dict[str, tuple[float, float, float, float]],
    *,
    clamp: bool = True,
) -> dict[str, tuple[float, float, float, float]]:
    """Warp card-space field boxes into frame-space envelopes."""
    names = list(fields)
    quads = result.warp_boxes([fields[name] for name in names])
    out = {}
    for name, quad in zip(names, quads):
        out[name] = result.envelope(quad, clamp=clamp)
    return out


def perspective_severity(result: GeometryResult) -> float:
    """Scalar 0..1 describing how strong the perspective distortion is.

    Used for targeted generation ("give me more perspective examples") and for
    hard-example classification. Measured as the ratio of the shortest to the
    longest edge of the card quad - a rectangular card seen head-on scores 1.0.
    """
    quad = result.corners
    edges = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
    longest = max(edges)
    if longest <= 1e-6:
        return 1.0
    return float(1.0 - min(edges) / longest)


def card_scale_in_frame(result: GeometryResult) -> float:
    """Area of the card as a fraction of the frame."""
    from ..datasets.schema import Quad

    quad = Quad(tuple((float(x), float(y)) for x, y in result.corners))
    frame_area = result.frame_size[0] * result.frame_size[1]
    return float(min(1.0, quad.area() / frame_area))
