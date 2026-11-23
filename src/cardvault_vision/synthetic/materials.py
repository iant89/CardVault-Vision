"""Card surface simulation (Phase 2.5) and the surface-light interaction.

The roadmap is explicit: *"The system should understand that different card
finishes behave differently."* They do, and the difference is not cosmetic - it
is the difference between a readable name field and a white blob.

Physical model used here
------------------------
A card is a layered surface:

    substrate (paper/cardstock)  ->  ink/art layer  ->  finish coat (gloss/foil)

Each finish is described by three parameters:

* ``specular`` - how much of the incident light reflects directly (glare).
  Glossy > foil > matte.
* ``diffuse`` - how much is scattered (base brightness under the same light).
* ``roughness`` - how *spread out* the highlight is. A matte surface has a wide,
  weak highlight; a chrome surface has a tiny, blinding one.

For holographic/foil finishes there is an additional **diffraction** term: the
reflected colour depends on viewing angle, which is why the same card shows a
rainbow that moves as you tilt it. We model this as an angle-dependent,
spectrally-varying specular term driven by a periodic grating pattern.

This module produces a *surface light map* (RGB, additive) plus an optional
occlusion term, which the pipeline screen-blends over the card. Because the
glare is added as light rather than painted as pixels, it correctly overexposes
text to white (destroying it) exactly as it does in reality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .imaging import (
    blend_modes,
    fbm,
    gaussian_kernel,
    convolve,
    linear_mask,
    radial_mask,
    scratch_texture,
    screen_blend,
    soft_light,
    value_noise,
)


@dataclass(frozen=True)
class FinishProfile:
    """Physical surface description of one card finish."""

    name: str
    specular: float        # 0..1 direct reflection strength
    diffuse: float         # 0..1 scattering strength
    roughness: float       # 0..1 (0 = mirror, 1 = fully matte)
    diffraction: float = 0.0  # 0..1 holo/rainbow angular colour shift
    metallic: float = 0.0     # 0..1 chrome-like tinting of the reflection
    texture: float = 0.0      # 0..1 visible surface grain

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "specular": self.specular,
            "diffuse": self.diffuse,
            "roughness": self.roughness,
            "diffraction": self.diffraction,
            "metallic": self.metallic,
            "texture": self.texture,
        }


#: Finishes from the schema, each with a distinct physical signature.
FINISH_PROFILES: dict[str, FinishProfile] = {
    "normal": FinishProfile("normal", specular=0.18, diffuse=0.95, roughness=0.65, texture=0.05),
    "matte": FinishProfile("matte", specular=0.06, diffuse=1.0, roughness=0.92, texture=0.09),
    "glossy": FinishProfile("glossy", specular=0.55, diffuse=0.88, roughness=0.20, metallic=0.05),
    "holo": FinishProfile("holo", specular=0.62, diffuse=0.85, roughness=0.18,
                          diffraction=0.85, metallic=0.35, texture=0.03),
    "reverse_holo": FinishProfile("reverse_holo", specular=0.58, diffuse=0.82, roughness=0.22,
                                 diffraction=0.65, metallic=0.25, texture=0.03),
    "foil": FinishProfile("foil", specular=0.72, diffuse=0.80, roughness=0.14,
                          diffraction=0.45, metallic=0.75, texture=0.02),
    "etched_foil": FinishProfile("etched_foil", specular=0.70, diffuse=0.78, roughness=0.16,
                                diffraction=0.55, metallic=0.70, texture=0.04),
    "chrome": FinishProfile("chrome", specular=0.92, diffuse=0.55, roughness=0.05,
                            metallic=0.95, diffraction=0.10),
    "refractor": FinishProfile("refractor", specular=0.68, diffuse=0.84, roughness=0.12,
                               diffraction=0.95, metallic=0.45, texture=0.02),
}

DEFAULT_FINISH = "normal"


def finish_profile(name: str | None) -> FinishProfile:
    """Look up a finish, falling back to ``normal`` for unknown names."""
    return FINISH_PROFILES.get((name or DEFAULT_FINISH).lower(), FINISH_PROFILES[DEFAULT_FINISH])


@dataclass
class LightSource:
    """A light in the scene, in card-relative coordinates.

    ``direction`` is the unit vector from the card towards the light; ``azimuth``
    and ``elevation`` are its angular description (degrees) which is the form
    profiles are authored in. ``hardness`` 0 = fully diffuse, 1 = point source.
    """

    azimuth: float = 45.0
    elevation: float = 60.0
    intensity: float = 1.0
    hardness: float = 0.5
    kelvin: float = 5500.0
    width: float = 0.5      # angular size (affects highlight extent)

    def to_dict(self) -> dict:
        return {
            "azimuth": round(self.azimuth, 2),
            "elevation": round(self.elevation, 2),
            "intensity": round(self.intensity, 3),
            "hardness": round(self.hardness, 3),
            "kelvin": round(self.kelvin, 1),
            "width": round(self.width, 3),
        }

    @property
    def direction(self) -> np.ndarray:
        az, el = math.radians(self.azimuth), math.radians(self.elevation)
        return np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az),
                         math.sin(el)], dtype=np.float64)


@dataclass
class SurfaceLightResult:
    """Additive light contribution, multiplicative shading, and composited card."""

    light_map: np.ndarray            # HxWx3 additive RGB (glare/holo/foil)
    shading: np.ndarray              # HxW multiplicative illumination in ~[0, 1.5]
    highlight_mask: np.ndarray       # HxW float, 1 where specular dominates
    composited: np.ndarray           # HxWx3 the card with the light applied
    effects: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "effects": self.effects,
            "mean_shading": round(float(self.shading.mean()), 4),
            "highlight_fraction": round(float((self.highlight_mask > 0.5).mean()), 4),
        }


# ---------------------------------------------------------------------------
# Individual effects
# ---------------------------------------------------------------------------

def specular_highlight(
    width: int,
    height: int,
    finish: FinishProfile,
    light: LightSource,
    rng: np.random.Generator,
    *,
    centre: tuple[float, float] | None = None,
    view_tilt: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """A broad specular highlight plus a small hot core.

    Returns ``(intensity HxW, colour HxWx3)``. The highlight position moves with
    the light azimuth and with card tilt (``view_tilt``), which is exactly the
    behaviour a model must learn to become robust: the bright spot is *not*
    always in the same place.
    """
    if finish.specular <= 0.01 or light.intensity <= 0.01:
        return np.zeros((height, width), dtype=np.float32), np.zeros((height, width, 3),
                                                                    dtype=np.float32)

    # Where the highlight falls: opposite the light azimuth, shifted by tilt.
    if centre is None:
        cx = width * (0.5 + 0.42 * math.cos(math.radians(light.azimuth)) * (1 - view_tilt))
        cy = height * (0.5 + 0.42 * math.sin(math.radians(light.azimuth)) * (1 - view_tilt))
    else:
        cx, cy = centre[0] * width, centre[1] * height

    # Roughness widens and dims the highlight; width parameter scales it.
    # A specular highlight is *small*. Sizing it as a fraction of the card makes
    # the glare read as a reflection rather than a painted white blob; a rough
    # matte surface spreads it wider but weaker.
    base_radius = max(3.0, min(width, height) * (0.035 + 0.085 * finish.roughness) *
                      (0.55 + 0.6 * light.width))
    core_radius = base_radius * max(0.22, 1.0 - 0.7 * light.hardness)

    broad = radial_mask(width, height, cx, cy, base_radius, softness=0.85)
    core = radial_mask(width, height, cx, cy, core_radius, softness=0.5)
    intensity = np.clip(
        (0.45 * broad + 0.8 * core) * finish.specular * light.intensity, 0.0, 1.0
    ).astype(np.float32)

    # Anisotropic stretch along the light direction reads as a "streak".
    streak = linear_mask(width, height, math.radians(light.azimuth + 90),
                         softness=max(0.12, 0.5 - 0.3 * light.hardness))
    intensity = np.clip(intensity * (0.5 + 0.7 * streak), 0.0, 1.0).astype(np.float32)

    colour = _light_colour(light.kelvin)
    if finish.metallic > 0:
        # Metal tints the reflection towards the surface's own hue.
        tint = np.array([1.0, 0.97, 0.88], dtype=np.float32) if finish.metallic > 0.5 else \
            np.array([0.95, 0.98, 1.0], dtype=np.float32)
        colour = colour * (1 - finish.metallic) + colour * tint * finish.metallic
    return intensity, np.broadcast_to(colour, (height, width, 3)).astype(np.float32)


def _light_colour(kelvin: float) -> np.ndarray:
    from .imaging import _blackbody_rgb

    rgb = _blackbody_rgb(kelvin)
    return (rgb / max(float(rgb.mean()), 1e-6)).astype(np.float32)


def diffraction_pattern(
    width: int,
    height: int,
    finish: FinishProfile,
    light: LightSource,
    rng: np.random.Generator,
    *,
    grating_angle: float | None = None,
    scale: float | None = None,
) -> np.ndarray:
    """Rainbow interference pattern for holographic/foil finishes.

    Built as a sinusoid whose phase depends on position *and* on viewing angle,
    so tilting the card slides the rainbow - the defining visual property of a
    holo card. Three hue-shifted band patterns are combined so the result has
    real colour variation rather than a greyscale stripe.
    """
    if finish.diffraction <= 0.01:
        return np.zeros((height, width, 3), dtype=np.float32)

    angle = grating_angle if grating_angle is not None else rng.uniform(0, math.pi)
    period = max(3.0, (scale or 1.0) * rng.uniform(14.0, 46.0) / max(light.intensity, 0.4))
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    proj = xs * math.cos(angle) + ys * math.sin(angle)
    phase = proj / period * 2 * math.pi

    # Viewing-angle dependence: elevation modulates the phase offset.
    offset = math.radians(light.elevation) * rng.uniform(0.5, 1.5)
    bands = np.stack([
        np.sin(phase * 1.0 + offset),
        np.sin(phase * 1.13 + offset * 1.7 + 2.094),   # 120 deg
        np.sin(phase * 0.87 + offset * 1.3 + 4.188),   # 240 deg
    ], axis=-1).astype(np.float32)

    # Squash to 0..1 and colour-shift so it reads as a rainbow, not a sine.
    bands = 0.5 + 0.5 * bands
    hue_shift = (0.5 + 0.5 * np.sin(xs / max(width, 1) * 2 * math.pi +
                                   math.radians(light.azimuth))) [..., None]
    bands = np.clip(bands * (0.55 + 0.75 * hue_shift), 0.0, 1.0)

    # Confine the effect to bands/patches - real holo cards are not uniformly lit.
    patch = value_noise(width, height, max(8, min(width, height) // 6), rng, octaves=2)
    patch = np.clip((patch - 0.35) / 0.55, 0.0, 1.0)[..., None]
    return (bands * patch * finish.diffraction * light.intensity).astype(np.float32)


def foil_reflection(
    width: int,
    height: int,
    finish: FinishProfile,
    light: LightSource,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sharp, displaced mirror-like reflection for foil/chrome surfaces."""
    if finish.metallic <= 0.05:
        return np.zeros((height, width, 3), dtype=np.float32)
    count = 1 if finish.metallic < 0.7 else int(rng.integers(1, 3))
    lightmap = np.zeros((height, width, 3), dtype=np.float32)
    for _ in range(count):
        cx = rng.uniform(0.15, 0.85)
        cy = rng.uniform(0.1, 0.6)
        radius = max(6.0, min(width, height) * rng.uniform(0.06, 0.22))
        mask = radial_mask(width, height, cx * width, cy * height, radius, softness=0.6)
        tint = _light_colour(rng.uniform(4200, 11000)) * finish.metallic
        lightmap += (mask[..., None] * tint * 0.55).astype(np.float32)
    return np.clip(lightmap, 0.0, 1.0)


def refraction_band(
    width: int,
    height: int,
    rng: np.random.Generator,
    *,
    strength: float = 0.25,
) -> np.ndarray:
    """A soft prismatic band - refractive effects on thick/embossed card stock."""
    if strength <= 0.01:
        return np.zeros((height, width, 3), dtype=np.float32)
    angle = rng.uniform(0, math.pi)
    mask = linear_mask(width, height, angle, softness=rng.uniform(0.05, 0.25),
                       offset=rng.uniform(-0.3, 0.3))
    centre_falloff = linear_mask(width, height, angle + math.pi, softness=0.9)
    mask = (mask * centre_falloff)[..., None]
    rainbow = np.stack([
        np.full((height, width), 1.0, np.float32),
        np.full((height, width), 0.92, np.float32),
        np.full((height, width), 0.78, np.float32),
    ], axis=-1)
    return (mask * rainbow * strength * rng.uniform(0.4, 1.0)).astype(np.float32)


def surface_texture(
    width: int,
    height: int,
    finish: FinishProfile,
    rng: np.random.Generator,
    *,
    wear: float = 0.0,
) -> np.ndarray:
    """Multiplicative micro-texture: paper grain, print lines, scratches, wear.

    Returned as a field in ~[0.85, 1.05] so it modulates the card rather than
    replacing it. ``wear`` adds scratches and edge whitening, which real
    collections show and which confuse edge-based text detection.
    """
    field = np.ones((height, width), dtype=np.float32)
    if finish.texture > 0:
        grain = value_noise(width, height, max(2, min(width, height) // 40), rng, octaves=2)
        field *= 1.0 + (grain - 0.5) * 0.10 * finish.texture
    if wear > 0:
        scratches = scratch_texture(width, height, rng,
                                    count=int(2 + wear * 10),
                                    max_length=0.15 + 0.4 * wear)
        field += scratches * 0.35 * wear
        # Edge whitening: cards wear white at the borders.
        edge = _edge_mask(width, height, band_frac=0.03)
        field += edge * wear * 0.18
        specks = (rng.random((height, width)) < 0.0008 * wear * 10).astype(np.float32)
        field *= 1.0 - convolve(specks, gaussian_kernel(0.8)) * 0.25
    return np.clip(field, 0.55, 1.35).astype(np.float32)


def _edge_mask(width: int, height: int, band_frac: float = 0.03) -> np.ndarray:
    """1 near the card border, 0 in the middle."""
    band = max(1, int(min(width, height) * band_frac))
    mask = np.zeros((height, width), dtype=np.float32)
    mask[:band, :] = 1.0
    mask[-band:, :] = 1.0
    mask[:, :band] = 1.0
    mask[:, -band:] = 1.0
    return convolve(mask, gaussian_kernel(max(1.0, band / 3.0)))


# ---------------------------------------------------------------------------
# Composite surface lighting
# ---------------------------------------------------------------------------

def apply_surface_lighting(
    card: np.ndarray,
    finish: FinishProfile,
    lights: list[LightSource],
    rng: np.random.Generator,
    *,
    view_tilt: float = 0.0,
    ambient: float = 0.35,
    holo_scale: float = 1.0,
    wear: float = 0.0,
) -> SurfaceLightResult:
    """Compute and apply the full surface response for a card image.

    Order matters and follows the physics: texture/wear modulate the surface,
    diffuse ambient+key shading scales it, then specular/diffraction/refraction
    light is *added* on top (screen blend), because those are emitted light, not
    reflected albedo.
    """
    from .imaging import to_float

    card = to_float(card) if card.dtype != np.float32 else card
    height, width = card.shape[:2]

    texture = surface_texture(width, height, finish, rng, wear=wear)
    textured = card * texture[..., None]

    # Diffuse shading: sum of cosine-weighted contributions per light.
    shading = np.full((height, width), ambient, dtype=np.float32)
    for light in lights:
        falloff = 1.0 - 0.55 * min(abs(light.elevation) / 90.0, 1.0)
        field = linear_mask(width, height, math.radians(light.azimuth),
                            softness=0.6 + 0.4 * (1 - light.hardness))
        shading += (field * falloff * light.intensity * finish.diffuse *
                    (1.0 - 0.45 * finish.metallic))
    shading = np.clip(shading, 0.0, 2.2).astype(np.float32)
    shaded = np.clip(textured * shading[..., None], 0.0, 1.0).astype(np.float32)

    # Additive light.
    light_map = np.zeros((height, width, 3), dtype=np.float32)
    highlight = np.zeros((height, width), dtype=np.float32)
    key = max(lights, key=lambda item: item.intensity) if lights else None
    if key is not None:
        intensity, colour = specular_highlight(width, height, finish, key, rng,
                                               view_tilt=view_tilt)
        highlight = intensity
        light_map += intensity[..., None] * colour
        if finish.diffraction > 0:
            light_map += diffraction_pattern(width, height, finish, key, rng,
                                             scale=holo_scale)
        light_map += foil_reflection(width, height, finish, key, rng)
        light_map += refraction_band(width, height, rng,
                                     strength=0.18 * finish.diffraction + 0.1 * finish.metallic)

    if finish.diffraction > 0:
        # Holo sheen: a low-level iridescence across the whole card, tilted with
        # the view.
        sheen = diffraction_pattern(width, height, finish, LightSource(
            azimuth=rng.uniform(0, 360), elevation=rng.uniform(10, 80),
            intensity=0.45, kelvin=6500), rng, scale=holo_scale * 2.2)
        light_map += sheen * 0.55

    light_map = np.clip(light_map, 0.0, 1.5).astype(np.float32)
    result = screen_blend(shaded, light_map)

    return SurfaceLightResult(
        light_map=light_map,
        shading=shading,
        highlight_mask=highlight,
        effects={
            "finish": finish.name,
            "lights": [light.to_dict() for light in lights],
            "specular": finish.specular,
            "diffuse": finish.diffuse,
            "diffraction": finish.diffraction,
            "metallic": finish.metallic,
            "wear": round(wear, 4),
            "highlight_area": round(float((highlight > 0.5).mean()), 5),
            "glare_over_text": round(float(np.clip(highlight - 0.55, 0, 1).max()), 4),
            "ambient": round(ambient, 4),
        },
        composited=result,
    )


def glare_from_environment(
    width: int,
    height: int,
    rng: np.random.Generator,
    *,
    strength: float = 0.5,
    window: bool = True,
) -> np.ndarray:
    """A big, soft reflection of a bright environment (window, ceiling light).

    This is the "phone shadow's cousin": the dominant glare source in indoor
    card photos is overhead lighting reflected off a glossy sleeve.
    """
    if strength <= 0.01:
        return np.zeros((height, width, 3), dtype=np.float32)
    count = int(rng.integers(1, 3))
    out = np.zeros((height, width, 3), dtype=np.float32)
    for _ in range(count):
        if window:
            # Rectangular window reflection, slightly rotated.
            bw = rng.uniform(0.25, 0.7) * width
            bh = rng.uniform(0.15, 0.5) * height
            cx = rng.uniform(0.2, 0.8) * width
            cy = rng.uniform(0.15, 0.7) * height
            ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
            dx = np.abs(xs - cx) / max(bw / 2, 1)
            dy = np.abs(ys - cy) / max(bh / 2, 1)
            mask = np.clip(1.0 - np.maximum(dx, dy), 0.0, 1.0) ** 1.4
        else:
            mask = radial_mask(width, height, rng.uniform(0.2, 0.8) * width,
                               rng.uniform(0.2, 0.8) * height,
                               min(width, height) * rng.uniform(0.1, 0.3), softness=0.8)
        mask = convolve(mask.astype(np.float32), gaussian_kernel(min(width, height) * 0.03 + 1))
        tint = _light_colour(rng.uniform(4500, 9000))
        out += (mask[..., None] * tint * strength * rng.uniform(0.5, 1.0)).astype(np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def reflected_card_smudge(
    card: np.ndarray,
    rng: np.random.Generator,
    *,
    strength: float = 0.25,
) -> np.ndarray:
    """A blurred, mirrored ghost of the card itself.

    Glossy cards in a sleeve reflect their own artwork back at an offset. This is
    a very common and very confusing artefact for OCR - the model sees two
    overlapping drawings of the same text.
    """
    from .imaging import resize, to_float

    card = to_float(card)
    height, width = card.shape[:2]
    small = resize(card, (max(2, width // 6), max(2, height // 6)), resample="bilinear")
    blur = convolve(small, gaussian_kernel(2.0))
    ghost = resize(blur, (width, height), resample="bilinear")
    shift_x = int(rng.uniform(-0.1, 0.1) * width)
    shift_y = int(rng.uniform(-0.1, 0.1) * height)
    ghost = np.roll(ghost, (shift_y, shift_x), axis=(0, 1))
    return (ghost * strength).astype(np.float32)
