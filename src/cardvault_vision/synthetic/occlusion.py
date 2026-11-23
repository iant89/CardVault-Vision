"""Occlusion simulation (Phase 2.7).

Occlusions are the hardest thing to fake convincingly and the most damaging when
ignored: a thumb across the collector number is the difference between a correct
match and a wrong one.

Every occluder returns an **alpha mask** (not a pre-blended image), so the
pipeline can:
  1. composite it over the card,
  2. subtract it from field annotations (a field that is half-covered is still a
     valid training label, but a field that is *entirely* covered is not),
  3. record the occluded area, so quality control can reject samples where the
     name or number is no longer legible.

Occluder types: fingers, hands, card sleeves, other cards, objects, and the
edge obstruction caused by the card sliding out of frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .imaging import (
    blend_modes,
    convolve,
    fbm,
    gaussian_kernel,
    radial_mask,
    resize,
    to_float,
)

OCCLUDER_TYPES = ("finger", "hand", "sleeve", "card", "object", "edge")


@dataclass
class Occluder:
    """One object that covers part of the scene."""

    mask: np.ndarray          # HxW float 0..1
    rgb: np.ndarray           # HxWx3 float 0..1
    kind: str
    affects_card: bool = True
    effects: dict = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        return float((self.mask > 0.5).mean())

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "coverage": round(self.coverage, 4),
            "affects_card": self.affects_card,
            "effects": self.effects,
        }

    def composite_over(self, image: np.ndarray) -> np.ndarray:
        alpha = np.clip(self.mask, 0.0, 1.0)[..., None]
        return np.clip(image * (1 - alpha) + self.rgb * alpha, 0.0, 1.0).astype(np.float32)


def _skin_colour(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Base skin tone and a shadow tone, drawn across a realistic range."""
    tone = rng.uniform(0.0, 1.0)
    # Light -> deep, using a simple perceptual ramp.
    base = np.array([0.93 - 0.45 * tone, 0.76 - 0.36 * tone, 0.65 - 0.30 * tone],
                    dtype=np.float32)
    base += rng.normal(0, 0.02, size=3).astype(np.float32)
    shadow = np.clip(base * rng.uniform(0.55, 0.75), 0, 1)
    return np.clip(base, 0.05, 1.0), shadow


def make_finger(
    size: tuple[int, int],
    rng: np.random.Generator,
    *,
    anchor: tuple[float, float] | None = None,
    angle: float | None = None,
    length: float | None = None,
    width: float | None = None,
) -> Occluder:
    """A single finger entering the frame from an edge.

    Modelled as a rounded capsule with a soft edge, a slightly darker rim and a
    nail highlight - enough structure that the model cannot trivially learn
    "flat colour blob = finger", while staying cheap to render.
    """
    width_px, height_px = int(size[0]), int(size[1])
    base, shadow = _skin_colour(rng)

    # Fingers enter from a frame edge and point roughly towards the card.
    if anchor is None:
        edge = int(rng.integers(0, 4))
        along = rng.uniform(0.15, 0.85)
        anchor = [(along, 0.0), (0.0, along), (along, 1.0), (1.0, along)][edge]
    ax, ay = anchor[0] * width_px, anchor[1] * height_px
    if angle is None:
        angle = math.atan2(height_px * 0.5 - ay, width_px * 0.5 - ax) + rng.normal(0, 0.25)

    finger_len = (length or rng.uniform(0.18, 0.45)) * min(width_px, height_px) * 2.2
    finger_w = (width or rng.uniform(0.055, 0.11)) * min(width_px, height_px)

    ys, xs = np.mgrid[0:height_px, 0:width_px].astype(np.float32)
    dx, dy = xs - ax, ys - ay
    # Distance along / across the finger axis.
    along = dx * math.cos(angle) + dy * math.sin(angle)
    across = -dx * math.sin(angle) + dy * math.cos(angle)
    taper = np.clip(1.0 - np.maximum(along, 0) / max(finger_len, 1e-3), 0.35, 1.0)
    radius = finger_w / 2 * taper
    dist = np.abs(across) / np.maximum(radius, 1e-3)
    capsule = np.clip(1.0 - (dist - 0.72) / 0.28, 0.0, 1.0)
    # Round the fingertip.
    tip = np.clip(1.0 - (np.hypot(along - finger_len * 0.92, across) /
                         max(finger_w / 2, 1e-3) - 0.6) / 0.4, 0.0, 1.0)
    mask = np.where(along > finger_len * 0.92, np.minimum(capsule, tip), capsule)

    # Shading: darker at the sides, brighter down the middle.
    shade = np.clip(1.0 - 0.35 * np.abs(across) / np.maximum(radius, 1e-3), 0.4, 1.0)
    rgb = base.reshape(1, 1, 3) * shade[..., None]
    # Nail highlight near the tip.
    nail = radial_mask(width_px, height_px,
                       ax + math.cos(angle) * finger_len * 0.88,
                       ay + math.sin(angle) * finger_len * 0.88,
                       finger_w * 0.28, softness=0.7) * (along > finger_len * 0.7)
    rgb = rgb + nail[..., None] * 0.18
    # Soft contact edge, but not so soft it looks like a blur.
    mask = convolve(mask.astype(np.float32), gaussian_kernel(max(0.8, finger_w * 0.03)))
    return Occluder(mask=np.clip(mask, 0, 1).astype(np.float32),
                    rgb=np.clip(rgb, 0, 1).astype(np.float32), kind="finger",
                    effects={"angle": round(float(angle), 3),
                             "length": round(float(finger_len), 1),
                             "width": round(float(finger_w), 1)})


def make_hand(size: tuple[int, int], rng: np.random.Generator) -> Occluder:
    """A hand (palm plus fingers) holding the card.

    Composed from a soft palm blob and 3-5 fingers, so the silhouette has the
    irregular boundary that real hands have.
    """
    width_px, height_px = int(size[0]), int(size[1])
    base, _ = _skin_colour(rng)
    mask = np.zeros((height_px, width_px), dtype=np.float32)

    palm_cx = rng.uniform(0.0, 1.0) * width_px
    palm_cy = rng.uniform(0.85, 1.15) * height_px
    palm_r = min(width_px, height_px) * rng.uniform(0.35, 0.55)
    palm = radial_mask(width_px, height_px, palm_cx, palm_cy, palm_r, softness=0.75)
    mask = np.maximum(mask, palm)

    fingers = int(rng.integers(3, 6))
    base_angle = -math.pi / 2 + rng.normal(0, 0.3)
    effects = {"fingers": fingers}
    for i in range(fingers):
        offset = (i - (fingers - 1) / 2) * rng.uniform(0.12, 0.2)
        finger = make_finger(
            size, rng,
            anchor=(palm_cx / width_px + offset * 0.5, palm_cy / height_px),
            angle=base_angle + offset * 0.8,
            length=rng.uniform(0.2, 0.4),
            width=rng.uniform(0.05, 0.09),
        )
        mask = np.maximum(mask, finger.mask)

    mask = convolve(mask, gaussian_kernel(max(1.0, min(width_px, height_px) * 0.004)))
    shade = 0.8 + 0.35 * fbm(width_px, height_px, rng, base_scale=max(8, width_px // 16),
                             octaves=2)
    rgb = base.reshape(1, 1, 3) * shade[..., None]
    return Occluder(mask=np.clip(mask, 0, 1).astype(np.float32),
                    rgb=np.clip(rgb, 0, 1).astype(np.float32), kind="hand", effects=effects)


def make_card_overlap(size: tuple[int, int], rng: np.random.Generator,
                      other_card: np.ndarray | None = None) -> Occluder:
    """Another card lying on top of / next to the target card.

    If a second source card image is supplied it is used as the occluder art
    (much better than a blank rectangle); otherwise a generic card-like
    rectangle is drawn. Real users photograph a stack of cards, so overlapping
    cards with *readable* text are a key source of OCR false positives.
    """
    width_px, height_px = int(size[0]), int(size[1])
    mask = np.zeros((height_px, width_px), dtype=np.float32)
    angle = rng.uniform(0, math.pi)
    card_w = width_px * rng.uniform(0.55, 0.9)
    card_h = card_w * (88.0 / 63.0)
    cx = rng.uniform(0.15, 0.85) * width_px
    cy = rng.uniform(0.15, 0.85) * height_px

    ys, xs = np.mgrid[0:height_px, 0:width_px].astype(np.float32)
    dx, dy = xs - cx, ys - cy
    lx = dx * math.cos(angle) + dy * math.sin(angle)
    ly = -dx * math.sin(angle) + dy * math.cos(angle)
    inside = (np.abs(lx) <= card_w / 2) & (np.abs(ly) <= card_h / 2)
    mask = inside.astype(np.float32)

    if other_card is not None:
        art = resize(to_float(other_card), (max(2, int(card_w)), max(2, int(card_h))))
        canvas = np.full((height_px, width_px, 3), 0.5, dtype=np.float32)
        # Rotate-then-paste is expensive; approximate by resizing the whole card
        # into the frame region the overlay occupies.
        y0 = max(0, int(cy - card_h / 2))
        y1 = min(height_px, int(cy + card_h / 2))
        x0 = max(0, int(cx - card_w / 2))
        x1 = min(width_px, int(cx + card_w / 2))
        if y1 > y0 and x1 > x0:
            patch = resize(art, (x1 - x0, y1 - y0))
            canvas[y0:y1, x0:x1] = patch
        rgb = canvas
    else:
        colour = np.array([rng.uniform(0.1, 0.9) for _ in range(3)], dtype=np.float32)
        rgb = np.broadcast_to(colour, (height_px, width_px, 3)).astype(np.float32).copy()
    mask = convolve(mask, gaussian_kernel(0.7))
    return Occluder(mask=np.clip(mask, 0, 1).astype(np.float32), rgb=rgb, kind="card",
                    effects={"angle": round(float(angle), 3)})


def make_sleeve(size: tuple[int, int], rng: np.random.Generator) -> Occluder:
    """A plastic sleeve over the card: milky haze plus a specular sheet edge.

    Crucially this occluder is mostly *transparent*: the card is still readable
    but contrast drops and a bright reflected band appears. Rendering it as
    alpha (rather than opaque) is what makes it a useful training example.
    """
    width_px, height_px = int(size[0]), int(size[1])
    ys, xs = np.mgrid[0:height_px, 0:width_px].astype(np.float32)
    # Sleeve slightly larger than the card, shifted randomly.
    cx = width_px * (0.5 + rng.normal(0, 0.07))
    cy = height_px * (0.5 + rng.normal(0, 0.07))
    half_w = width_px * rng.uniform(0.52, 0.62)
    half_h = height_px * rng.uniform(0.52, 0.62)
    inside = (np.abs(xs - cx) <= half_w) & (np.abs(ys - cy) <= half_h)
    haze = inside.astype(np.float32) * rng.uniform(0.05, 0.16)
    # Bright reflected band. The alpha is low, so it lightens without hiding.
    band_angle = rng.uniform(0, math.pi)
    proj = (xs - cx) * math.cos(band_angle) + (ys - cy) * math.sin(band_angle)
    band = np.exp(-0.5 * (proj / (min(width_px, height_px) * rng.uniform(0.04, 0.1))) ** 2)
    band *= inside
    mask = np.clip(haze + band * rng.uniform(0.12, 0.3), 0.0, 0.55)
    rgb = np.full((height_px, width_px, 3), rng.uniform(0.9, 1.0), dtype=np.float32)
    return Occluder(mask=mask.astype(np.float32), rgb=rgb, kind="sleeve",
                    effects={"haze": round(float(haze.max()), 4),
                             "band_alpha": round(float(mask.max()), 4)})


# Plausible real-world object palettes. A random RGB triple produces a
# saturated primary that no household object has, and a model trained on those
# learns "saturated blue disc" instead of "something is in front of the card".
_OBJECT_PALETTES = {
    "metal": [(0.62, 0.63, 0.66), (0.72, 0.66, 0.48), (0.45, 0.46, 0.50)],
    "plastic": [(0.10, 0.11, 0.13), (0.86, 0.86, 0.84), (0.28, 0.30, 0.34)],
    "wood": [(0.52, 0.36, 0.20), (0.38, 0.26, 0.15)],
    "paper": [(0.90, 0.89, 0.85), (0.86, 0.84, 0.80)],
    "fabric": [(0.22, 0.26, 0.36), (0.30, 0.22, 0.18), (0.16, 0.16, 0.16)],
}


def make_object(size: tuple[int, int], rng: np.random.Generator) -> Occluder:
    """A random household object (coin, pen, dice, remote).

    Objects are rendered with a plausible material rather than a random colour,
    and they receive a simple Lambert-ish shading ramp: real objects are not flat.
    A flat fill would be an adversarial cue the model could latch onto.
    """
    width_px, height_px = int(size[0]), int(size[1])
    material = str(rng.choice(list(_OBJECT_PALETTES)))
    colour = np.array(_OBJECT_PALETTES[material][int(rng.integers(0, len(_OBJECT_PALETTES[material])))],
                      dtype=np.float32)
    colour = np.clip(colour * rng.uniform(0.85, 1.12), 0.0, 1.0)
    kind = str(rng.choice(["ellipse", "bar", "blob"]))
    cx, cy = rng.uniform(0.1, 0.9) * width_px, rng.uniform(0.1, 0.9) * height_px
    if kind == "ellipse":
        radius = min(width_px, height_px) * rng.uniform(0.05, 0.16)
        mask = radial_mask(width_px, height_px, cx, cy, radius, softness=0.2)
    elif kind == "bar":
        length = min(width_px, height_px) * rng.uniform(0.3, 0.7)
        thickness = min(width_px, height_px) * rng.uniform(0.02, 0.05)
        angle = rng.uniform(0, math.pi)
        ys, xs = np.mgrid[0:height_px, 0:width_px].astype(np.float32)
        dx, dy = xs - cx, ys - cy
        along = np.abs(dx * math.cos(angle) + dy * math.sin(angle))
        across = np.abs(-dx * math.sin(angle) + dy * math.cos(angle))
        mask = ((along < length / 2) & (across < thickness / 2)).astype(np.float32)
    else:
        blob = fbm(width_px, height_px, rng, base_scale=max(8, width_px // 10), octaves=3)
        mask = (blob > rng.uniform(0.45, 0.62)).astype(np.float32) * \
            radial_mask(width_px, height_px, cx, cy,
                        min(width_px, height_px) * rng.uniform(0.15, 0.3), softness=0.4)
    mask = convolve(mask, gaussian_kernel(1.0))

    # Shading: lit from one side, so the object has a gradient and a soft rim.
    ys, xs = np.mgrid[0:height_px, 0:width_px].astype(np.float32)
    light = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1)], dtype=np.float32)
    norm = float(np.hypot(*light)) or 1.0
    light = (light / norm) * min(width_px, height_px) * 0.25
    shade = 0.72 + 0.28 * np.clip(
        1.0 - np.hypot(xs - (cx + light[0]), ys - (cy + light[1])) /
        (min(width_px, height_px) * 0.6), 0.0, 1.0)
    rgb = (colour[None, None, :] * shade[..., None]).astype(np.float32)
    return Occluder(mask=np.clip(mask, 0, 1).astype(np.float32), rgb=rgb, kind="object",
                    effects={"shape": kind, "material": material})


def make_edge_obstruction(size: tuple[int, int], rng: np.random.Generator) -> Occluder:
    """A dark band along one frame edge, as if the card is partly under a book."""
    width_px, height_px = int(size[0]), int(size[1])
    mask = np.zeros((height_px, width_px), dtype=np.float32)
    edge = int(rng.integers(0, 4))
    frac = rng.uniform(0.05, 0.18)
    if edge == 0:
        mask[: int(height_px * frac), :] = 1.0
    elif edge == 1:
        mask[:, : int(width_px * frac)] = 1.0
    elif edge == 2:
        mask[int(height_px * (1 - frac)):, :] = 1.0
    else:
        mask[:, int(width_px * (1 - frac)):] = 1.0
    mask = convolve(mask, gaussian_kernel(max(1.0, min(width_px, height_px) * 0.01)))
    colour = np.array([rng.uniform(0.02, 0.2) for _ in range(3)], dtype=np.float32)
    rgb = np.broadcast_to(colour, (height_px, width_px, 3)).astype(np.float32).copy()
    return Occluder(mask=np.clip(mask, 0, 1).astype(np.float32), rgb=rgb, kind="edge",
                    effects={"edge": edge, "fraction": round(float(frac), 3)})


def sample_occluders(
    size: tuple[int, int],
    rng: np.random.Generator,
    *,
    count: int | None = None,
    allow_hand: bool = True,
    other_card: np.ndarray | None = None,
    forced_kind: str | None = None,
) -> list[Occluder]:
    """Draw 0..N occluders for a scene.

    ``count=None`` samples from a distribution where 0 occluders is common
    (most captures are clean) but 1-2 is not rare - important, because a dataset
    containing only clean cards teaches a model that cannot cope at all.
    """
    if forced_kind is not None:
        builders = {"finger": make_finger, "hand": make_hand, "object": make_object,
                    "edge": make_edge_obstruction}
        if forced_kind == "card":
            return [make_card_overlap(size, rng, other_card)]
        if forced_kind == "sleeve":
            return [make_sleeve(size, rng)]
        builder = builders.get(forced_kind)
        return [builder(size, rng)] if builder else []

    if count is None:
        count = int(rng.choice([0, 0, 0, 1, 1, 2, 3]))
    occluders: list[Occluder] = []
    for _ in range(count):
        choices = ["finger", "sleeve", "card", "object", "edge"]
        weights = np.array([0.42, 0.16, 0.16, 0.16, 0.10])
        if allow_hand:
            choices.append("hand")
            weights = np.append(weights, 0.14)
        kind = str(rng.choice(choices, p=weights / weights.sum()))
        if kind == "finger":
            occluders.append(make_finger(size, rng))
        elif kind == "hand":
            occluders.append(make_hand(size, rng))
        elif kind == "sleeve":
            occluders.append(make_sleeve(size, rng))
        elif kind == "card":
            occluders.append(make_card_overlap(size, rng, other_card))
        elif kind == "object":
            occluders.append(make_object(size, rng))
        else:
            occluders.append(make_edge_obstruction(size, rng))
    return occluders


def occlusion_coverage(mask: np.ndarray, box: tuple[float, float, float, float]) -> float:
    """Fraction of a field's box covered by an occlusion mask.

    Used to decide whether a field is still trainable and to power the
    ``--problem occlusion`` targeted mode.
    """
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    height, width = mask.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return 1.0  # a box entirely outside the frame is fully unusable
    region = mask[y0:y1, x0:x1]
    return float((region > 0.5).mean())
