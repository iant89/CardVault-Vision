"""Environment / background simulation (Phase 2.6).

Backgrounds are procedurally generated so the engine has no external asset
dependency - a 100k-image run on a fresh machine works immediately. Each material
is built from noise and geometry rather than tiled photos, which also avoids the
risk of the model memorising one specific desk.

Materials: wood, desk (laminate), carpet, play mat, binder page, card sleeve,
paper, neutral backdrop, plus optional household clutter.

Every background is generated to sit *behind* the card and to be plausible at the
card's scale: grain size scales with the frame, not with pixels, so the same
material looks right whether the frame is 600 px or 2400 px wide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .imaging import (
    convolve,
    fbm,
    gaussian_kernel,
    hsv_to_rgb,
    linear_mask,
    radial_mask,
    screen_blend,
    soft_light,
    value_noise,
    vignette,
)

BACKGROUND_TYPES = (
    "wood",
    "desk",
    "carpet",
    "play_mat",
    "binder",
    "sleeve",
    "paper",
    "neutral",
    "tablecloth",
    "clutter",
)


@dataclass
class BackgroundResult:
    image: np.ndarray
    kind: str
    effects: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "effects": self.effects}


def _tint(base: np.ndarray, colour: tuple[float, float, float], strength: float = 1.0
          ) -> np.ndarray:
    tint = np.array(colour, dtype=np.float32).reshape(1, 1, 3)
    return np.clip(base[..., None] * tint * strength, 0.0, 1.0).astype(np.float32)


def _wood_grain(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Wood grain: anisotropic rings modulated by fractal streaks."""
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    angle = rng.uniform(0, math.pi)
    # Stretch coordinates along the grain direction.
    xr = xs * math.cos(angle) + ys * math.sin(angle)
    yr = -xs * math.sin(angle) + ys * math.cos(angle)
    # Grain runs *along* the board: low spatial frequency across the width, with
    # slight warping so it does not read as a printed stripe pattern.
    freq = rng.uniform(0.0015, 0.006) * (1000.0 / max(width, 1))
    rings = 0.5 + 0.5 * np.sin(xr * freq * 2 * math.pi +
                              np.sin(yr * freq * 0.15) * rng.uniform(1.0, 4.0))
    streaks = fbm(width, height, rng, base_scale=max(4, int(width / 24)), octaves=4)
    grain = np.clip(0.65 * rings + 0.45 * streaks, 0.0, 1.0)
    return grain.astype(np.float32)


def _carpet(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Fibrous texture: fine directional noise plus large-scale shading."""
    fibre = value_noise(width, height, max(2, width // 220), rng, octaves=3)
    coarse = fbm(width, height, rng, base_scale=max(16, width // 8), octaves=3)
    return np.clip(0.55 * fibre + 0.55 * coarse, 0.0, 1.0).astype(np.float32)


def _checker(width: int, height: int, rng: np.random.Generator,
             cell: int) -> np.ndarray:
    ys, xs = np.mgrid[0:height, 0:width]
    board = (((xs // max(cell, 1)) + (ys // max(cell, 1))) % 2).astype(np.float32)
    return board


def generate_background(
    rng: np.random.Generator,
    size: tuple[int, int],
    *,
    kind: str | None = None,
    dark: bool | None = None,
) -> BackgroundResult:
    """Generate one background image.

    ``kind=None`` chooses a plausible material at random; ``dark`` biases the
    luminance, which matters because a dark card on a dark desk is a genuinely
    hard case that should appear in the data.
    """
    width, height = int(size[0]), int(size[1])
    if width < 8 or height < 8:
        raise ValueError("Background must be at least 8x8")

    chosen = kind if kind in BACKGROUND_TYPES else BACKGROUND_TYPES[
        int(rng.integers(0, len(BACKGROUND_TYPES)))]

    if dark is None:
        dark = bool(rng.random() < 0.35)

    if chosen == "wood":
        grain = _wood_grain(width, height, rng)
        hue = rng.uniform(0.05, 0.10)
        sat = rng.uniform(0.35, 0.6)
        val = rng.uniform(0.22, 0.55)
        base = _from_hsv(grain, hue, sat, val)
        effects = {"material": "wood", "grain_scale": round(float(rng.uniform(0.5, 2.0)), 2)}
    elif chosen == "desk":
        grid = _checker(width, height, rng, cell=int(rng.uniform(0.15, 0.4) * width))
        noise = fbm(width, height, rng, base_scale=max(8, width // 30), octaves=2)
        base = _from_hsv(np.clip(0.85 * noise + 0.15 * grid, 0, 1),
                         rng.uniform(0.55, 0.65), rng.uniform(0.05, 0.2),
                         rng.uniform(0.25, 0.5))
        effects = {"material": "laminate"}
    elif chosen == "carpet":
        texture = _carpet(width, height, rng)
        base = _from_hsv(texture, rng.uniform(0, 1), rng.uniform(0.1, 0.5),
                         rng.uniform(0.2, 0.6))
        effects = {"material": "carpet"}
    elif chosen == "play_mat":
        cell = max(8, int(width * rng.uniform(0.08, 0.16)))
        base = _checker(width, height, rng, cell).astype(np.float32)
        base = base[..., None] * np.array([0.12, 0.28, 0.2], np.float32) + \
            (1 - base[..., None]) * np.array([0.06, 0.15, 0.12], np.float32)
        base = convolve(base, gaussian_kernel(0.8))
        effects = {"material": "play_mat", "cell_px": cell}
    elif chosen in {"binder", "sleeve"}:
        # A transparent plastic sheet over a dark backing, with a bright
        # specular sheet highlight - the classic "card in a page" capture.
        backing = _from_hsv(fbm(width, height, rng, base_scale=max(8, width // 20), octaves=3),
                            rng.uniform(0.55, 0.7), rng.uniform(0.05, 0.25),
                            rng.uniform(0.1, 0.3))
        gradient = linear_mask(width, height, rng.uniform(0, math.pi), softness=0.7)
        sheen = screen_blend(backing, (gradient[..., None] * rng.uniform(0.05, 0.25)))
        base = sheen
        effects = {"material": chosen, "plastic_sheen": True}
    elif chosen == "paper":
        base = _from_hsv(np.full((height, width), 0.5, np.float32), 0.1,
                         rng.uniform(0.0, 0.05), rng.uniform(0.7, 0.95))
        effects = {"material": "paper"}
    elif chosen == "tablecloth":
        weave = value_noise(width, height, max(2, width // 300), rng, octaves=2)
        stripes = 0.5 + 0.5 * np.sin(np.mgrid[0:height, 0:width][1] /
                                     max(1, width / rng.uniform(6, 20)) * 2 * math.pi)
        pattern = np.clip(0.6 * weave + 0.4 * stripes, 0, 1)
        base = _from_hsv(pattern, rng.uniform(0, 1), rng.uniform(0.1, 0.4),
                         rng.uniform(0.35, 0.8))
        effects = {"material": "tablecloth"}
    elif chosen == "clutter":
        base = _from_hsv(fbm(width, height, rng, base_scale=max(16, width // 12), octaves=4),
                         rng.uniform(0, 1), rng.uniform(0.05, 0.35),
                         rng.uniform(0.15, 0.5))
        base = _add_clutter_objects(base, rng)
        effects = {"material": "clutter"}
    else:  # neutral
        base = _from_hsv(np.full((height, width), 0.5, np.float32), 0.0, 0.0,
                         rng.uniform(0.15, 0.75))
        effects = {"material": "neutral"}

    if dark:
        base = np.clip(base * rng.uniform(0.35, 0.7), 0.0, 1.0)
    # Global surface shading + slight vignette keeps backgrounds from looking
    # procedurally flat.
    light_dir = rng.uniform(0, 2 * math.pi)
    base = soft_light(base, 0.85 + 0.3 * linear_mask(width, height, light_dir, softness=0.9))
    base = vignette(base, strength=float(abs(rng.normal(0.1, 0.06))))
    effects["dark"] = dark
    return BackgroundResult(image=np.clip(base, 0, 1).astype(np.float32), kind=chosen,
                            effects=effects)


def _from_hsv(field: np.ndarray, hue: float, sat: float, val: float) -> np.ndarray:
    height, width = field.shape
    hsv = np.empty((height, width, 3), dtype=np.float32)
    hsv[..., 0] = (hue + 0.04 * (field - 0.5)) % 1.0
    hsv[..., 1] = np.clip(sat * (0.8 + 0.4 * field), 0, 1)
    hsv[..., 2] = np.clip(val * (0.55 + 0.9 * field), 0, 1)
    return hsv_to_rgb(hsv)


def _add_clutter_objects(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Scatter simple objects (pens, coins, other cards) around the frame."""
    from PIL import Image, ImageDraw

    height, width = image.shape[:2]
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    count = int(rng.integers(1, 4))
    for _ in range(count):
        kind = rng.choice(["rect", "ellipse"])
        cx, cy = rng.uniform(0.05, 0.95) * width, rng.uniform(0.05, 0.95) * height
        half_w = rng.uniform(0.03, 0.16) * width
        half_h = rng.uniform(0.02, 0.1) * height
        colour = tuple(int(c * 255) for c in rng.uniform(0.05, 0.7, size=3))
        box = [cx - half_w, cy - half_h, cx + half_w, cy + half_h]
        if kind == "rect":
            draw.rectangle(box, fill=colour)
        else:
            draw.ellipse(box, fill=colour)
    objects = np.asarray(canvas, dtype=np.float32) / 255.0
    mask = (objects.max(axis=-1, keepdims=True) > 0.01).astype(np.float32)
    mask = convolve(mask, gaussian_kernel(1.0))
    blended = image * (1 - mask) + objects * mask
    return np.clip(blended, 0.0, 1.0).astype(np.float32)


def add_ambient_light_pool(
    background: np.ndarray,
    rng: np.random.Generator,
    *,
    centre: tuple[float, float] | None = None,
    strength: float | None = None,
) -> np.ndarray:
    """A pool of light on the surface (desk lamp / phone flash spill).

    Real captures have a bright region around the card; without it the card's
    brightness does not match its surroundings and models can learn that
    "bright rectangle = card" instead of learning the card.
    """
    height, width = background.shape[:2]
    if centre is None:
        centre = (rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7))
    radius = min(width, height) * rng.uniform(0.45, 0.85)
    pool = radial_mask(width, height, centre[0] * width, centre[1] * height, radius,
                       softness=0.95)
    amount = strength if strength is not None else rng.uniform(0.15, 0.5)
    return np.clip(background + pool[..., None] * amount * rng.uniform(0.6, 1.0), 0.0, 1.0
                   ).astype(np.float32)
