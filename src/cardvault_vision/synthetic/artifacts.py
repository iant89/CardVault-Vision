"""Artifact simulation and quality measurement.

Some degradations are camera effects (handled in :mod:`camera`), some are surface
effects (handled in :mod:`materials`). This module covers the rest - the ones
that come from the *environment* or the *process* rather than the optics:

* moiré / interference from photographing fine print or a fabric lattice,
* jello/rolling shutter skew,
* sensor dust, hot pixels and stuck pixels,
* "scanner line" banding from a dirty flatbed,
* double-JPEG (an image shared over messaging apps),
* blur inconsistent with focus (nothing here - that belongs in camera).

It also provides the **quality measurements** that let the engine reject
unusable generations and grade degradation severity, which is the input to hard
example mining.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .camera import estimate_noise, estimate_sharpness
from .imaging import (
    convolve,
    gaussian_kernel,
    jpeg_compress,
    rgb_to_hsv,
    to_float,
)

# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

def moire_pattern(image: np.ndarray, rng: np.random.Generator,
                  *, strength: float = 0.25) -> np.ndarray:
    """Interference pattern from photographing fine periodic detail.

    Holographic card art is a fine grating, so moiré is not exotic here - it is
    one of the main reasons holo cards OCR badly.
    """
    if strength <= 0.01:
        return image
    height, width = image.shape[:2]
    angle = rng.uniform(0, math.pi)
    freq = rng.uniform(0.15, 0.6)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    proj = xs * math.cos(angle) + ys * math.sin(angle)
    wave = np.sin(proj * freq * 2 * math.pi * 0.1) * np.sin(
        (xs * math.sin(angle) - ys * math.cos(angle)) * freq * 2 * math.pi * 0.07)
    mask = 0.5 + 0.5 * wave
    scale = 1.0 + (mask - 0.5)[..., None] * strength
    return np.clip(image * scale, 0.0, 1.0).astype(np.float32)


def rolling_shutter(image: np.ndarray, rng: np.random.Generator,
                    *, max_shift: float = 0.012) -> np.ndarray:
    """Horizontal band skew from a rolling-shutter sensor during motion."""
    height, width = image.shape[:2]
    shift = rng.uniform(-max_shift, max_shift)
    if abs(shift) < 1e-4:
        return image
    out = np.empty_like(image)
    for row in range(height):
        amount = shift * width * (row / max(height - 1, 1))
        out[row] = np.roll(image[row], int(round(amount)), axis=0)
    return out


def sensor_dead_pixels(image: np.ndarray, rng: np.random.Generator,
                       *, hot: int = 12, stuck: int = 12) -> np.ndarray:
    """Hot (bright) and stuck (single-channel) pixels - visible at high ISO."""
    out = image.copy()
    height, width = out.shape[:2]
    for _ in range(hot):
        y, x = int(rng.integers(0, height)), int(rng.integers(0, width))
        out[y, x] = np.clip(out[y, x] + rng.uniform(0.4, 1.0), 0.0, 1.0)
    for _ in range(stuck):
        y, x = int(rng.integers(0, height)), int(rng.integers(0, width))
        channel = int(rng.integers(0, 3))
        out[y, x, channel] = rng.uniform(0.7, 1.0)
    return out


def dust_spots(image: np.ndarray, rng: np.random.Generator,
               *, count: int = 4) -> np.ndarray:
    """Out-of-focus dust on the sensor/lens: soft dark blobs."""
    out = image
    height, width = out.shape[:2]
    for _ in range(count):
        cx, cy = rng.uniform(0, width), rng.uniform(0, height)
        radius = min(width, height) * rng.uniform(0.01, 0.05)
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        blob = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * radius ** 2))
        out = out * (1.0 - 0.28 * blob[..., None] * rng.uniform(0.4, 1.0))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def scanner_banding(image: np.ndarray, rng: np.random.Generator,
                    *, strength: float = 0.03) -> np.ndarray:
    """Vertical streaks from a dirty or failing flatbed scanner lamp."""
    height, width = image.shape[:2]
    profile = 1.0 + rng.normal(0, strength, size=width).astype(np.float32)
    profile = convolve(profile.reshape(1, -1), gaussian_kernel(2.0)).reshape(-1)
    return np.clip(image * profile.reshape(1, -1, 1), 0.0, 1.0).astype(np.float32)


def double_compression(image: np.ndarray, rng: np.random.Generator,
                       *, first: int = 80, second: int = 60) -> np.ndarray:
    """Re-compression, as happens when a photo is sent through a chat app.

    Double-JPEG is measurably harder for OCR than single compression because the
    first pass's block structure is re-quantised rather than refined.
    """
    once = jpeg_compress(image, first)
    return jpeg_compress(once, second)


def banding_from_8bit(image: np.ndarray, rng: np.random.Generator,
                      *, levels: int = 24) -> np.ndarray:
    """Posterisation - cheap sensors/aggressive processing on gradients."""
    return np.clip(np.round(image * (levels - 1)) / (levels - 1), 0.0, 1.0).astype(np.float32)


def gamma_clipping(image: np.ndarray, rng: np.random.Generator,
                   *, black: float = 0.02, white: float = 0.98) -> np.ndarray:
    """Crush blacks / blow highlights, as an aggressive tone curve does."""
    out = (image - black) / max(white - black, 1e-3)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Quality measurement
# ---------------------------------------------------------------------------

@dataclass
class QualityMetrics:
    """Per-image measurements used for QC and hard-example mining."""

    sharpness: float
    noise: float
    mean_luma: float
    contrast: float
    clipped_fraction: float
    crushed_fraction: float
    glare_fraction: float
    card_visibility: float
    card_area_fraction: float

    def to_dict(self) -> dict:
        return {
            "sharpness": round(self.sharpness, 3),
            "noise": round(self.noise, 4),
            "mean_luma": round(self.mean_luma, 4),
            "contrast": round(self.contrast, 4),
            "clipped_fraction": round(self.clipped_fraction, 5),
            "crushed_fraction": round(self.crushed_fraction, 5),
            "glare_fraction": round(self.glare_fraction, 4),
            "card_visibility": round(self.card_visibility, 4),
            "card_area_fraction": round(self.card_area_fraction, 4),
        }

    @property
    def is_usable(self) -> bool:
        """Coarse usability gate.

        Deliberately lenient: the engine's job is to produce hard examples, and
        rejecting everything difficult would defeat the purpose. These bounds
        only catch images with essentially no information left (or none of the
        card left).
        """
        return (
            self.card_visibility >= 0.25
            and self.sharpness >= 8.0        # 0-255 Laplacian variance scale
            and self.mean_luma >= 0.03
            and not (self.clipped_fraction > 0.85 and self.contrast < 0.02)
        )


def measure_quality(
    image: np.ndarray,
    *,
    card_visibility: float = 1.0,
    card_area_fraction: float = 0.4,
) -> QualityMetrics:
    """Compute the standard quality metrics for a rendered capture."""
    arr = to_float(image)
    gray = arr.mean(axis=-1) if arr.ndim == 3 else arr
    hsv = rgb_to_hsv(arr) if arr.ndim == 3 else None
    glare = 0.0
    if hsv is not None:
        glare = float(((hsv[..., 2] > 0.92) & (hsv[..., 1] < 0.15)).mean())
    return QualityMetrics(
        sharpness=estimate_sharpness(arr),
        noise=estimate_noise(arr),
        mean_luma=float(gray.mean()),
        contrast=float(gray.std()),
        clipped_fraction=float((gray >= 0.995).mean()),
        crushed_fraction=float((gray <= 0.005).mean()),
        glare_fraction=glare,
        card_visibility=float(card_visibility),
        card_area_fraction=float(card_area_fraction),
    )


def field_legibility(image: np.ndarray, box: tuple[float, float, float, float]) -> float:
    """A cheap proxy for "can the text in this box still be read".

    Combines local contrast (text needs edges) and local sharpness (edges need to
    survive blur). This is intentionally a *proxy*: the real arbiter is the
    trained model, which is what Phase 9's mining loop measures. Being cheap lets
    us run it on every generated sample.
    """
    arr = to_float(image)
    gray = arr.mean(axis=-1) if arr.ndim == 3 else arr
    height, width = gray.shape
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return 0.0
    region = gray[y0:y1, x0:x1]
    contrast = float(region.std())
    sharpness = estimate_sharpness(region)
    # Normalise both into 0..1 with soft saturating curves, then combine.
    # ``contrast`` is on the 0-1 scale; ``sharpness`` is on the 0-255 scale.
    c_score = min(1.0, contrast / 0.14)
    s_score = min(1.0, math.log1p(max(sharpness, 0.0)) / math.log1p(2500.0))
    return float(max(0.0, min(1.0, 0.55 * c_score + 0.45 * s_score)))


def apply_artifacts(
    image: np.ndarray,
    rng: np.random.Generator,
    *,
    level: float = 0.0,
    allow_double_jpeg: bool = True,
    scanner_mode: bool = False,
) -> tuple[np.ndarray, dict]:
    """Apply a random selection of artifacts scaled by ``level`` (0..1+)."""
    out = image
    applied: dict = {}
    if level <= 0.01:
        return out, applied

    if rng.random() < 0.25 * level:
        strength = rng.uniform(0.05, 0.3) * level
        out = moire_pattern(out, rng, strength=strength)
        applied["moire"] = round(float(strength), 4)
    if rng.random() < 0.12 * level:
        out = rolling_shutter(out, rng, max_shift=0.008 * level)
        applied["rolling_shutter"] = True
    if rng.random() < 0.2 * level:
        out = sensor_dead_pixels(out, rng, hot=int(6 * level), stuck=int(6 * level))
        applied["dead_pixels"] = True
    if rng.random() < 0.25 * level:
        out = dust_spots(out, rng, count=int(1 + 3 * level))
        applied["dust"] = True
    if scanner_mode and rng.random() < 0.35:
        strength = rng.uniform(0.01, 0.04) * level
        out = scanner_banding(out, rng, strength=strength)
        applied["scanner_banding"] = round(float(strength), 4)
    if allow_double_jpeg and rng.random() < 0.2 * level:
        first = int(rng.uniform(60, 90))
        second = int(rng.uniform(40, first))
        out = double_compression(out, rng, first=first, second=second)
        applied["double_jpeg"] = [first, second]
    if rng.random() < 0.1 * level:
        out = banding_from_8bit(out, rng, levels=int(rng.uniform(16, 48)))
        applied["posterised"] = True
    return np.clip(out, 0.0, 1.0).astype(np.float32), applied
