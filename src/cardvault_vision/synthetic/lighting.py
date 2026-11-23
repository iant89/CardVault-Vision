"""Lighting simulation (Phase 2.4).

Lighting is applied in two places, deliberately:

1. **On the card** (here, and in :mod:`materials`) - directional light, uneven
   illumination, over/under-exposure, colour temperature. This is the light the
   model must be invariant to.
2. **On the scene** (shadows cast onto the background) - hard/soft shadows,
   phone shadows, hand shadows. These are the cues that tell the model "there is
   a person holding this", and they must not be confused with card content.

The order of operations mirrors a real capture: illuminate the surface, then
let the sensor clip (overexposure) and lift (underexposure).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .imaging import (
    adjust_contrast,
    adjust_exposure,
    adjust_saturation,
    apply_gamma,
    colour_temperature,
    convolve,
    fbm,
    gaussian_kernel,
    gradient_field,
    radial_mask,
    soft_light,
    to_float,
    vignette,
)
from .materials import LightSource


@dataclass
class LightingConfig:
    """Resolved lighting parameters for one capture."""

    brightness: float = 0.0          # exposure in stops
    contrast: float = 1.0
    gamma: float = 1.0
    saturation: float = 1.0
    kelvin: float = 5500.0
    ambient: float = 0.35            # fraction of diffuse ambient light
    key: LightSource | None = None
    fill: LightSource | None = None
    uneven: float = 0.0              # 0..1 strength of illumination gradient
    uneven_scale: float = 0.35       # spatial frequency of that gradient
    vignette: float = 0.0
    overexposed: bool = False
    underexposed: bool = False
    tint: tuple[float, float] = (1.0, 1.0)   # extra r/b white-balance gains
    noise_from_iso: float = 0.0      # how much shadow noise the exposure implies

    def to_dict(self) -> dict:
        out = {
            "brightness": round(self.brightness, 3),
            "contrast": round(self.contrast, 3),
            "gamma": round(self.gamma, 3),
            "saturation": round(self.saturation, 3),
            "kelvin": round(self.kelvin, 1),
            "ambient": round(self.ambient, 3),
            "uneven": round(self.uneven, 3),
            "vignette": round(self.vignette, 3),
            "overexposed": self.overexposed,
            "underexposed": self.underexposed,
            "tint": [round(self.tint[0], 3), round(self.tint[1], 3)],
            "noise_from_iso": round(self.noise_from_iso, 3),
        }
        if self.key is not None:
            out["key"] = self.key.to_dict()
        if self.fill is not None:
            out["fill"] = self.fill.to_dict()
        return out


@dataclass
class LightingResult:
    """The illuminated card plus the illumination fields used."""

    card: np.ndarray
    illumination: np.ndarray       # HxW field actually applied (2-D)
    config: LightingConfig
    shadow: np.ndarray | None = None
    effects: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "effects": self.effects,
            "mean_illumination": round(float(self.illumination.mean()), 4),
            "illumination_range": [round(float(self.illumination.min()), 4),
                                   round(float(self.illumination.max()), 4)],
        }


def illumination_field(
    width: int,
    height: int,
    config: LightingConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build the multiplicative illumination field for a card surface.

    Combines a directional gradient (light from one side), an optional low-
    frequency unevenness (a hand or lamp partially blocking the light) and a
    vignette term. Values hover around 1.0 so it modulates rather than darkens.
    """
    field = np.ones((height, width), dtype=np.float32)

    if config.key is not None:
        directional = gradient_field(width, height,
                                     angle=math.radians(config.key.azimuth),
                                     falloff=0.55 + 1.2 * config.key.hardness)
        strength = 0.55 * config.key.intensity
        field = field * (1.0 - strength) + directional * strength * 1.0
    if config.fill is not None:
        fill = gradient_field(width, height, angle=math.radians(config.fill.azimuth + 180),
                              falloff=0.4)
        strength = 0.3 * config.fill.intensity
        field = field * (1.0 - strength) + fill * strength

    if config.uneven > 0:
        blob = fbm(width, height, rng, base_scale=max(8, int(min(width, height) *
                                                            config.uneven_scale)),
                   octaves=3)
        field = field * (1.0 - config.uneven * 0.6) + blob * (config.uneven * 0.6)

    if config.vignette > 0:
        field = vignette(field[..., None], config.vignette)[..., 0]

    # Keep the mean near 1 so exposure stays interpretable.
    mean = float(field.mean())
    if mean > 1e-6:
        field = field / mean
    return np.clip(field, 0.05, 3.0).astype(np.float32)


def apply_lighting(
    card: np.ndarray,
    config: LightingConfig,
    rng: np.random.Generator,
) -> LightingResult:
    """Apply exposure, colour, illumination and clipping to a card image."""
    card = to_float(card) if card.dtype != np.float32 else card
    height, width = card.shape[:2]

    illumination = illumination_field(width, height, config, rng)
    out = soft_light(card, illumination * (0.6 + 0.4 * config.ambient * 2))

    # Colour temperature of the light source, then any residual WB error.
    out = colour_temperature(out, config.kelvin)
    if config.tint != (1.0, 1.0):
        from .imaging import white_balance_shift

        out = white_balance_shift(out, config.tint[0], config.tint[1])

    out = adjust_exposure(out, config.brightness)
    out = apply_gamma(out, config.gamma)
    out = adjust_contrast(out, config.contrast)
    if abs(config.saturation - 1.0) > 1e-3:
        out = adjust_saturation(out, config.saturation)

    # Sensor clipping happens last: an overexposed name field becomes flat white
    # and the text information is genuinely destroyed (not merely hidden).
    if config.overexposed:
        clip = 0.72 + 0.2 * rng.uniform()
        out = np.clip(out / max(clip, 0.3), 0.0, 1.0)
    if config.underexposed:
        out = out * rng.uniform(0.18, 0.4)

    return LightingResult(
        card=np.clip(out, 0.0, 1.0).astype(np.float32),
        illumination=illumination,
        config=config,
        effects={
            "mean_brightness": round(float(out.mean()), 4),
            "clipped_fraction": round(float((out.max(axis=-1) >= 0.999).mean()), 5),
            "shadow_fraction": round(float((out.mean(axis=-1) < 0.08).mean()), 5),
        },
    )


# ---------------------------------------------------------------------------
# Shadows
# ---------------------------------------------------------------------------

@dataclass
class ShadowResult:
    """A shadow to be composited onto the background beneath the card."""

    mask: np.ndarray          # HxW float 0..1 (1 = full shadow)
    darkness: float
    softness: float
    kind: str

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "darkness": round(float(self.darkness), 3),
            "softness": round(float(self.softness), 3),
            "coverage": round(float(self.mask.mean()), 4),
        }


def cast_shadow(
    card_alpha: np.ndarray,
    *,
    rng: np.random.Generator,
    light: LightSource | None = None,
    hardness: float = 0.5,
    offset: tuple[float, float] | None = None,
    darkness: float | None = None,
) -> ShadowResult:
    """Cast a shadow of the card itself onto the surface.

    Hard light -> tight, dark shadow (phone flash close to the card).
    Soft light -> wide, faint shadow (window light). The shadow is the card's own
    alpha, offset away from the light and blurred according to ``hardness``.
    """
    height, width = card_alpha.shape[:2]
    alpha = np.clip(card_alpha.astype(np.float32), 0.0, 1.0)

    if offset is None:
        azimuth = light.azimuth if light is not None else rng.uniform(0, 360)
        magnitude = (0.02 + 0.05 * hardness) * min(width, height)
        offset = (math.cos(math.radians(azimuth)) * magnitude,
                  math.sin(math.radians(azimuth)) * magnitude)
    shift_x = int(round(offset[0]))
    shift_y = int(round(offset[1]))
    shadow = np.zeros_like(alpha)
    src_y0, src_y1 = max(0, -shift_y), min(height, height - shift_y)
    src_x0, src_x1 = max(0, -shift_x), min(width, width - shift_x)
    if src_x1 > src_x0 and src_y1 > src_y0:
        shadow[src_y0 + shift_y:src_y1 + shift_y, src_x0 + shift_x:src_x1 + shift_x] = \
            alpha[src_y0:src_y1, src_x0:src_x1]

    # A tighter, darker component plus a wide, faint penumbra.
    tight_sigma = max(1.0, (1.0 - hardness) * min(width, height) * 0.012)
    wide_sigma = tight_sigma * (2.5 + 4.0 * (1.0 - hardness))
    penumbra = convolve(shadow, gaussian_kernel(wide_sigma))
    core = convolve(shadow, gaussian_kernel(tight_sigma))
    mask = np.clip(0.65 * penumbra + 0.5 * core, 0.0, 1.0)
    if darkness is None:
        darkness = 0.25 + 0.45 * hardness
    return ShadowResult(mask=mask.astype(np.float32), darkness=float(darkness),
                        softness=float(1.0 - hardness), kind="card")


def phone_shadow(
    width: int,
    height: int,
    rng: np.random.Generator,
    *,
    card_box: tuple[float, float, float, float] | None = None,
) -> ShadowResult:
    """The shadow of the *phone* the user is holding over the card.

    Unmistakable in real datasets: a large hard-edged dark region covering a big
    part of the frame, often with the phone's rounded corners visible. It is a
    top cause of unreadable name fields, so it must be synthesised.
    """
    mask = np.zeros((height, width), dtype=np.float32)
    # Phone occupies a large slab entering from an edge.
    edge = int(rng.integers(0, 4))
    coverage = rng.uniform(0.25, 0.6)
    if edge == 0:
        mask[: int(height * coverage), :] = 1.0
    elif edge == 1:
        mask[:, : int(width * coverage)] = 1.0
    elif edge == 2:
        mask[int(height * (1 - coverage)):, :] = 1.0
    else:
        mask[:, int(width * (1 - coverage)):] = 1.0
    # Rounded corners at the boundary.
    mask = convolve(mask, gaussian_kernel(max(2.0, min(width, height) * 0.012)))
    return ShadowResult(mask=np.clip(mask, 0, 1).astype(np.float32),
                        darkness=float(rng.uniform(0.45, 0.8)),
                        softness=0.15, kind="phone")


def hand_shadow(
    width: int,
    height: int,
    rng: np.random.Generator,
) -> ShadowResult:
    """A soft shadow from the hand holding the phone."""
    cx = rng.uniform(-0.1, 0.4) * width
    cy = rng.uniform(0.0, 0.5) * height
    mask = radial_mask(width, height, cx, cy, min(width, height) * rng.uniform(0.35, 0.7),
                       softness=0.95)
    mask = convolve(mask, gaussian_kernel(min(width, height) * 0.04))
    return ShadowResult(mask=np.clip(mask, 0, 1).astype(np.float32),
                        darkness=float(rng.uniform(0.2, 0.45)),
                        softness=0.8, kind="hand")


def apply_shadow(
    background: np.ndarray,
    shadow: ShadowResult,
    *,
    colour: tuple[float, float, float] = (0.02, 0.02, 0.03),
) -> np.ndarray:
    """Darken the background where the shadow falls."""
    bg = to_float(background) if background.dtype != np.float32 else background
    alpha = np.clip(shadow.mask * shadow.darkness, 0.0, 1.0)[..., None]
    # Multiply-darken, then tint the darkest part very slightly towards the
    # shadow colour so it does not read as pure black.
    darkened = bg * (1.0 - alpha)
    tint = np.array(colour, dtype=np.float32).reshape(1, 1, 3)
    return np.clip(darkened * (1.0 - 0.35 * alpha) + tint * 0.35 * alpha, 0.0, 1.0).astype(
        np.float32)


def sample_lighting(rng: np.random.Generator, *, mode: str = "normal",
                    ambient_light: bool = False,
                    overexpose_chance: float = 0.05,
                    underexpose_chance: float = 0.05) -> LightingConfig:
    """Draw a plausible lighting configuration.

    ``mode`` scales the degradation: ``clean`` keeps exposure and colour nearly
    ideal, ``poor`` widens every distribution, ``extreme`` allows near-black
    frames and strong casts. ``ambient_light`` switches to a flat, shadowless
    look (scanner / light box).
    """
    scale = {"clean": 0.25, "normal": 1.0, "poor": 1.8, "extreme": 3.0}.get(mode, 1.0)

    if ambient_light:
        config = LightingConfig(
            brightness=float(rng.normal(0.05, 0.1 * scale)),
            contrast=float(rng.uniform(0.95, 1.05)),
            kelvin=float(rng.uniform(5800, 6800)),
            ambient=0.9,
            uneven=float(abs(rng.normal(0, 0.05 * scale))),
            vignette=float(abs(rng.normal(0, 0.04 * scale))),
        )
    else:
        azimuth = float(rng.uniform(0, 360))
        elevation = float(rng.uniform(15, 85))
        intensity = float(rng.uniform(0.5, 1.4))
        hardness = float(rng.uniform(0.1, 0.95))
        key = LightSource(azimuth=azimuth, elevation=elevation, intensity=intensity,
                          hardness=hardness, kelvin=float(rng.uniform(2700, 7500)),
                          width=float(rng.uniform(0.2, 0.9)))
        fill = None
        if rng.random() < 0.6:
            fill = LightSource(azimuth=azimuth + rng.uniform(90, 270),
                               elevation=float(rng.uniform(10, 60)),
                               intensity=float(rng.uniform(0.1, 0.6)),
                               hardness=float(rng.uniform(0.0, 0.5)),
                               kelvin=key.kelvin + float(rng.normal(0, 400)))
        config = LightingConfig(
            brightness=float(rng.normal(0.0, 0.45 * scale)),
            contrast=float(np.clip(rng.normal(1.0, 0.12 * scale), 0.6, 1.5)),
            gamma=float(np.clip(rng.normal(1.0, 0.08 * scale), 0.75, 1.3)),
            saturation=float(np.clip(rng.normal(1.0, 0.1 * scale), 0.6, 1.4)),
            kelvin=key.kelvin,
            ambient=float(np.clip(rng.normal(0.35, 0.12 * scale), 0.05, 0.9)),
            key=key,
            fill=fill,
            uneven=float(abs(rng.normal(0, 0.18 * scale))) if rng.random() < 0.7 else 0.0,
            uneven_scale=float(rng.uniform(0.2, 0.6)),
            vignette=float(abs(rng.normal(0, 0.12 * scale))),
            tint=(1.0, 1.0),
        )
        if rng.random() < 0.25 * scale:
            config.tint = (float(rng.uniform(0.92, 1.08)), float(rng.uniform(0.92, 1.08)))

    over_chance = overexpose_chance * scale
    under_chance = underexpose_chance * scale
    config.overexposed = bool(rng.random() < over_chance)
    config.underexposed = bool(rng.random() < under_chance) and not config.overexposed
    if config.overexposed:
        config.brightness += 0.9
    if config.underexposed:
        config.brightness -= 0.9
    return config


def exposure_severity(card: np.ndarray) -> dict:
    """Measure how badly exposed an image is (used by quality control)."""
    arr = to_float(card)
    gray = arr.mean(axis=-1) if arr.ndim == 3 else arr
    return {
        "mean": float(gray.mean()),
        "clipped_highlights": float((gray >= 0.995).mean()),
        "crushed_shadows": float((gray <= 0.02).mean()),
        "dynamic_range": float(np.percentile(gray, 99) - np.percentile(gray, 1)),
    }
