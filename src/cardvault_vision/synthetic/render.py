"""Scene rendering: composes background, card, light and occluders into a frame.

This is the single place where layers are combined, in the order a real
photograph accumulates light:

    1. background (environment material + ambient light pool)
    2. cast shadows (card shadow, phone shadow, hand shadow)
    3. the card itself (surface texture, material lighting, illumination)
    4. occluders in front of the card
    5. frame-level effects (vignette, glare bleed)

Keeping composition in one module makes the layer order testable and lets the
generator, the preview tool and the debug dumper share one code path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .background import BackgroundResult, add_ambient_light_pool
from .imaging import resize, screen_blend, vignette
from .lighting import ShadowResult, apply_shadow
from .occlusion import Occluder


@dataclass
class SceneLayers:
    """Everything that goes into a frame, before flattening."""

    background: BackgroundResult
    card_rgb: np.ndarray          # HxWx3 card colours, already lit
    card_alpha: np.ndarray        # HxW card coverage in the frame
    shadows: list[ShadowResult] = field(default_factory=list)
    occluders: list[Occluder] = field(default_factory=list)
    ambient_pool: bool = True
    frame_vignette: float = 0.0
    glare_bleed: np.ndarray | None = None


def render_scene(layers: SceneLayers, rng: np.random.Generator) -> np.ndarray:
    """Flatten the layers into a single ``HxWx3`` float image.

    The card is composited with its own alpha *after* shadows, so the card is
    never darkened by its own shadow, while occluders go on top of everything.
    """
    background = layers.background.image
    height, width = background.shape[:2]

    if layers.ambient_pool:
        background = add_ambient_light_pool(background, rng)

    for shadow in layers.shadows:
        background = apply_shadow(background, shadow)

    if layers.glare_bleed is not None:
        bleed = layers.glare_bleed
        if bleed.shape[:2] != (height, width):
            # Glare is authored at card resolution; resample to the frame.
            bleed = resize(bleed, (width, height), resample="bilinear")
        background = screen_blend(background, bleed)

    # Card over background.
    alpha = np.clip(layers.card_alpha, 0.0, 1.0)
    if alpha.ndim == 2:
        alpha = alpha[..., None]
    scene = background * (1.0 - alpha) + layers.card_rgb * alpha
    scene = np.clip(scene, 0.0, 1.0).astype(np.float32)

    # Occluders in front.
    for occluder in layers.occluders:
        scene = occluder.composite_over(scene)

    if layers.frame_vignette > 0:
        scene = vignette(scene, layers.frame_vignette)

    return np.clip(scene, 0.0, 1.0).astype(np.float32)


def card_alpha_mask(
    alpha: np.ndarray,
    occluders: list[Occluder],
) -> np.ndarray:
    """Card coverage after occluders, used for visibility accounting.

    A finger over the card means the *card* is still there, but the model cannot
    see it. Quality control cares about the visible part, so this is what we
    measure against.
    """
    visible = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    for occluder in occluders:
        if occluder.affects_card:
            visible = visible * (1.0 - np.clip(occluder.mask, 0.0, 1.0))
    return visible


def occlusion_mask(occluders: list[Occluder], size: tuple[int, int]) -> np.ndarray:
    """Union of all occluder masks (used for annotation adjustment)."""
    width, height = size
    combined = np.zeros((height, width), dtype=np.float32)
    for occluder in occluders:
        if occluder.mask.shape[:2] != (height, width):
            mask = resize(occluder.mask[..., None], (width, height))[..., 0]
        else:
            mask = occluder.mask
        combined = np.maximum(combined, np.clip(mask, 0.0, 1.0))
    return combined


def downscale_scene(scene: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resample a rendered scene to the capture resolution."""
    if (scene.shape[1], scene.shape[0]) == tuple(size):
        return scene
    return resize(scene, size, resample="bilinear")
