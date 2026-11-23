"""Per-sample generation: one source card -> one simulated camera capture.

This is the orchestration point of the engine. Each stage draws from its own
independent random stream (see :mod:`cardvault_vision.rng`), so:

* the same ``(source, config, seed, index)`` always yields the same image;
* adding a new lighting parameter cannot shift the numbers consumed by geometry;
* samples can be generated in any order, on any worker, and still match.

The pipeline also records **automatic ground truth** (Phase 2.10): every field's
text and box, the card quad, and the full simulation parameters - enough to
reproduce or attribute the sample later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..datasets.schema import (
    Annotation,
    BBox,
    CardKnowledge,
    FieldAnnotation,
    Quad,
    quad_visible_fraction,
)
from ..errors import GenerationError
from ..logging_utils import get_logger
from ..rng import RngStreams
from .artifacts import apply_artifacts, field_legibility, measure_quality
from .background import BackgroundResult, generate_background
from .camera import (
    RESOLUTION_PROFILES,
    apply_encoding,
    apply_optics,
    apply_sensor,
    sample_camera,
)
from .geometry import (
    card_project,
    sample_pose,
    warp_card_to_frame,
    warp_field_boxes,
)
from .imaging import to_float, to_uint8
from .lighting import (
    apply_lighting,
    cast_shadow,
    hand_shadow,
    phone_shadow,
    sample_lighting,
)
from .materials import (
    LightSource,
    apply_surface_lighting,
    finish_profile,
    glare_from_environment,
)
from .occlusion import occlusion_coverage, sample_occluders
from .render import SceneLayers, card_alpha_mask, downscale_scene, render_scene
from .text import draw_card_text

_LOG = get_logger("synthetic.generator")

#: Distribution helpers shared with :mod:`profiles`.


def _draw(rng: np.random.Generator, spec, default):
    """Resolve a profile distribution spec into one value.

    Supported specs: ``{"type": "uniform", "low", "high"}``,
    ``{"type": "normal", "mean", "sigma", "clamp"}``,
    ``{"type": "half_normal", "sigma", "clamp"}``,
    ``{"type": "choice", "options", "weights"}``, a plain ``[low, high]`` pair,
    or a scalar (returned as-is). An unknown spec raises rather than silently
    defaulting - a typo in a profile should be loud.
    """
    if spec is None:
        return default
    if isinstance(spec, (int, float, str, bool)):
        return spec
    if isinstance(spec, (list, tuple)):
        if len(spec) == 2 and all(isinstance(v, (int, float)) for v in spec):
            low, high = spec
            return float(rng.uniform(low, high))
        raise GenerationError(f"Ambiguous distribution spec: {spec!r}")
    if not isinstance(spec, dict):
        raise GenerationError(f"Unsupported distribution spec: {spec!r}")

    kind = spec.get("type", "uniform")
    clamp = spec.get("clamp")
    if kind == "uniform":
        value = float(rng.uniform(float(spec["low"]), float(spec["high"])))
    elif kind == "normal":
        value = float(rng.normal(float(spec.get("mean", 0.0)), float(spec.get("sigma", 1.0))))
    elif kind == "half_normal":
        value = float(abs(rng.normal(0.0, float(spec.get("sigma", 1.0)))))
    elif kind == "log_normal":
        value = float(rng.lognormal(float(spec.get("mean", 0.0)),
                                    float(spec.get("sigma", 1.0))))
    elif kind == "choice":
        options = list(spec["options"])
        weights = spec.get("weights")
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)
            weights = weights / weights.sum()
        index = int(rng.choice(len(options), p=weights))
        value = options[index]
    elif kind == "constant":
        value = spec.get("value")
    elif kind == "bernoulli":
        value = bool(rng.random() < float(spec.get("p", 0.5)))
    else:
        raise GenerationError(f"Unknown distribution type {kind!r}")
    if clamp and isinstance(value, (int, float)):
        low, high = clamp
        value = float(np.clip(value, low, high))
    return value


@dataclass
class SourceCard:
    """A clean card image plus its ground truth, ready to be simulated."""

    card_id: str
    image: np.ndarray                    # HxWx3 uint8 or float, the clean card
    card_type: str
    fields: dict[str, str] = field(default_factory=dict)       # name -> text
    field_rois: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    knowledge: CardKnowledge | None = None
    finish: str | None = None
    layout: str | None = None

    @property
    def size(self) -> tuple[int, int]:
        return int(self.image.shape[1]), int(self.image.shape[0])

    def effective_rois(self) -> dict[str, tuple[float, float, float, float]]:
        """Field boxes in card pixels.

        Uses explicit ROIs when provided; otherwise falls back to the layout
        default for the card type, so a source image without ROIs still produces
        valid annotations.
        """
        from ..layouts import resolve_layout

        if self.field_rois:
            out = {}
            width, height = self.size
            for name, roi in self.field_rois.items():
                # Accept normalised (0..1) or pixel ROIs.
                if max(roi) <= 1.5:
                    out[name] = (roi[0] * width, roi[1] * height,
                                 roi[2] * width, roi[3] * height)
                else:
                    out[name] = tuple(float(v) for v in roi)
            return out
        layout = resolve_layout(self.card_type, self.layout)
        return layout.rois_in_pixels(self.size, names=list(self.fields))


@dataclass
class GeneratedSample:
    """One simulated capture and its automatic ground truth."""

    image: np.ndarray                    # uint8 HxWx3
    annotation: Annotation
    profile: str
    seed: int
    index: int
    quality: dict = field(default_factory=dict)
    generation: dict = field(default_factory=dict)

    @property
    def image_id(self) -> str:
        return self.annotation.image_id

    @property
    def usable(self) -> bool:
        return bool(self.quality.get("is_usable", True))

    def to_pil(self):
        from PIL import Image

        return Image.fromarray(self.image)


# ---------------------------------------------------------------------------
# The generation pipeline
# ---------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    """Fully-resolved configuration for generating one sample.

    Built from a :class:`~cardvault_vision.synthetic.profiles.CaptureProfile`
    with mode/problem overrides already applied.
    """

    profile: dict
    mode: str = "normal"
    problems: list[str] = field(default_factory=list)
    min_visibility: float = 0.45
    min_legibility: float = 0.12
    render_scale: float = 1.0
    draw_missing_text: bool = False
    jpeg_on_final: bool = True
    metadata: dict = field(default_factory=dict)


def generate_sample(
    card: SourceCard,
    config: GenerationConfig,
    *,
    seed: int,
    index: int,
    image_id: str | None = None,
    other_card: np.ndarray | None = None,
) -> GeneratedSample:
    """Generate one simulated camera capture of ``card``."""
    streams = RngStreams(master_seed=seed, sample_index=index)
    profile = config.profile
    mode = config.mode
    problems = set(config.problems)

    # ---- 1. target resolution & render scale ---------------------------
    cam_cfg = profile["camera"]
    # Prefer the profile's authored resolution (a scan is a scan); fall back to the
    # per-device resolution table when the profile leaves it to the device.
    authored = profile.get("target_size")
    if isinstance(authored, (list, tuple)) and len(authored) == 2 and \
            float(profile.get("mode", "normal") != "custom"):
        base_w = int(authored[0])
    else:
        resolution_name = _draw(streams.get("sample"), cam_cfg.get("resolution"),
                                "phone_normal")
        base_w = RESOLUTION_PROFILES.get(str(resolution_name), (900, 1400))[0]
    target_w = max(96, int(round(base_w * config.render_scale)))

    # Frame aspect can differ from card aspect: users shoot 4:3, 16:9, square.
    aspect_choice = float(streams.get("sample").choice([4 / 3, 3 / 4, 1.0, 16 / 9],
                                                       p=[0.6, 0.15, 0.15, 0.10]))
    frame_w = target_w
    # max() already guarantees the floor, so no "if frame_h < 64" repair is
    # needed (the one that used to be here referenced an undefined target_h and
    # could only ever raise NameError).
    frame_h = max(64, int(round(frame_w / aspect_choice)))
    # Cap the render size for sane latency on CPU.
    render_w, render_h = frame_w, frame_h

    geometry_cfg = profile["geometry"]
    lighting_cfg = profile["lighting"]
    env_cfg = profile["environment"]
    occlusion_cfg = profile["occlusion"]
    artifact_cfg = profile["artifacts"]
    surface_cfg = profile["surface"]

    # ---- 2. geometry ----------------------------------------------------
    geo_rng = streams.get("geometry")
    max_pitch = float(geometry_cfg.get("max_pitch", 12.0))
    max_yaw = float(geometry_cfg.get("max_yaw", 12.0))
    max_roll = float(geometry_cfg.get("max_roll", 8.0))
    if "perspective" in problems:
        max_pitch = max(max_pitch, 40.0)
        max_yaw = max(max_yaw, 40.0)
        max_roll = max(max_roll, 25.0)
    scale_spec = geometry_cfg.get("scale")
    if isinstance(scale_spec, dict) and {"low", "high"} <= set(scale_spec):
        scale_range = (float(scale_spec["low"]), float(scale_spec["high"]))
    elif isinstance(scale_spec, (list, tuple)) and len(scale_spec) == 2:
        scale_range = (float(scale_spec[0]), float(scale_spec[1]))
    else:
        scale_range = (0.55, 0.92)
    pose = sample_pose(
        geo_rng,
        max_roll=max_roll,
        max_pitch=max_pitch,
        max_yaw=max_yaw,
        scale_range=scale_range,
        distance_range=(2.2, 6.0),
    )
    offset_sigma = float(geometry_cfg.get("offset_sigma", 0.06))
    if "tiny_text" in problems:
        # A smaller card means smaller text: the honest way to make this hard.
        pose.scale *= 0.72
        offset_sigma = max(offset_sigma, 0.1)
    offset = (float(geo_rng.normal(0, offset_sigma)),
              float(geo_rng.normal(0, offset_sigma)))
    crop_chance = float(geometry_cfg.get("crop_chance", 0.05))
    if "occlusion" in problems:
        crop_chance = max(crop_chance, 0.2)
    if geo_rng.random() < crop_chance:
        offset = (offset[0] + float(geo_rng.choice([-1, 1]) * geo_rng.uniform(0.1, 0.25)),
                  offset[1] + float(geo_rng.choice([-1, 1]) * geo_rng.uniform(0.1, 0.25)))
    edge_warp = float(_draw(geo_rng, geometry_cfg.get("edge_warp"), 0.0))

    try:
        geom = card_project(
            card.size, (render_w, render_h), pose,
            offset=offset, edge_warp=edge_warp, rng=geo_rng,
        )
    except GenerationError:
        raise

    # ---- 3. card surface, before it is placed ---------------------------
    card_rgb = to_float(card.image) if card.image.dtype != np.uint8 else to_float(card.image)
    finish_name = card.finish
    if finish_name is None:
        weights = surface_cfg.get("finish_weights") or {"normal": 1.0}
        names = list(weights)
        probs = np.array([weights[n] for n in names], dtype=np.float64)
        finish_name = str(names[int(streams.get("material").choice(len(names),
                                                                 p=probs / probs.sum()))])
    finish = finish_profile(finish_name)

    if config.draw_missing_text and card.fields:
        drawn, rois = draw_card_text(
            to_uint8(card_rgb), card.fields,
            rois=card.effective_rois() if card.field_rois else None,
            rng=streams.get("source"),
        )
        card_rgb = to_float(drawn)
        if rois and not card.field_rois:
            card = SourceCard(
                card_id=card.card_id, image=card.image, card_type=card.card_type,
                fields=card.fields, field_rois=rois, knowledge=card.knowledge,
                finish=card.finish, layout=card.layout,
            )

    mat_rng = streams.get("material")
    light_rng = streams.get("lighting")

    # ---- 4. lighting ----------------------------------------------------
    light_mode = mode
    if "low_light" in problems:
        profile = _with_lighting_override(profile, {
            "brightness": {"type": "normal", "mean": -1.2, "sigma": 0.4,
                           "clamp": [-2.8, -0.4]},
            "underexpose_chance": 0.5,
            "kelvin": {"type": "uniform", "low": 2400, "high": 3600},
        })
        lighting_cfg = profile["lighting"]
        light_mode = "poor"

    lighting = sample_lighting(
        light_rng,
        mode=light_mode,
        ambient_light=str(profile["camera"].get("device")) == "scanner",
        overexpose_chance=float(lighting_cfg.get("overexpose_chance", 0.05)),
        underexpose_chance=float(lighting_cfg.get("underexpose_chance", 0.05)),
    )
    # Profile-driven refinements (sample_lighting gives the mode baseline).
    lighting.kelvin = float(_draw(light_rng, lighting_cfg.get("kelvin"), lighting.kelvin))
    lighting.brightness += float(_draw(light_rng, lighting_cfg.get("brightness"), 0.0))
    lighting.contrast = float(_draw(light_rng, lighting_cfg.get("contrast"), lighting.contrast))
    lighting.gamma = float(_draw(light_rng, lighting_cfg.get("gamma"), lighting.gamma))
    lighting.ambient = float(_draw(light_rng, lighting_cfg.get("ambient"), lighting.ambient))
    lighting.uneven = float(_draw(light_rng, lighting_cfg.get("uneven"), lighting.uneven))
    lighting.vignette = float(_draw(light_rng, lighting_cfg.get("vignette"), lighting.vignette))
    if lighting.key is not None:
        lighting.key.hardness = float(_draw(light_rng, lighting_cfg.get("key_hardness"),
                                            lighting.key.hardness))
        lighting.key.intensity = float(_draw(light_rng, lighting_cfg.get("key_intensity"),
                                             lighting.key.intensity))
        lighting.key.kelvin = lighting.kelvin
    if lighting.fill is not None and light_rng.random() > float(
            lighting_cfg.get("fill_chance", 0.6)):
        lighting.fill = None

    lights = [light for light in (lighting.key, lighting.fill) if light is not None]
    if not lights:
        lights = [LightSource(azimuth=45, elevation=60, intensity=0.6, hardness=0.3,
                              kelvin=lighting.kelvin)]

    glare_chance = float(surface_cfg.get("glare_chance", 0.45))
    if "glare" in problems:
        glare_chance = 1.0
    wear = float(_draw(mat_rng, surface_cfg.get("wear"), 0.0))
    surface = apply_surface_lighting(
        card_rgb, finish, lights, mat_rng,
        view_tilt=abs(pose.pitch) / 45.0,
        ambient=lighting.ambient,
        wear=wear,
    )
    glare_strength = float(_draw(mat_rng, lighting_cfg.get("glare_strength"), 0.35))
    if "glare" in problems:
        glare_strength = max(glare_strength, 0.75)
    if mat_rng.random() < glare_chance:
        from .imaging import screen_blend

        glare = glare_from_environment(
            surface.composited.shape[1], surface.composited.shape[0], mat_rng,
            strength=glare_strength)
        lit_card = screen_blend(surface.composited, glare)
    else:
        glare = None
        lit_card = surface.composited

    # ---- 5. illumination + exposure -------------------------------------
    lit = apply_lighting(lit_card, lighting, light_rng)
    card_lit = lit.card

    # ---- 6. place the card in the frame ---------------------------------
    placed, alpha = warp_card_to_frame(card_lit, geom, supersample=int(
        profile.get("supersample", 2)))

    # ---- 7. background + shadows ---------------------------------------
    env_rng = streams.get("environment")
    backgrounds = env_cfg.get("backgrounds")
    bg_kind = None
    if isinstance(backgrounds, dict):
        bg_kind = _draw(env_rng, backgrounds, None)
    elif isinstance(backgrounds, (list, tuple)):
        bg_kind = str(env_rng.choice(list(backgrounds)))
    dark = None
    if not isinstance(env_cfg.get("dark_chance"), bool):
        dark = bool(env_rng.random() < float(env_cfg.get("dark_chance", 0.35)))
    background = generate_background(env_rng, (render_w, render_h), kind=bg_kind, dark=dark)

    shadows = []
    if env_rng.random() < float(lighting_cfg.get("shadow_chance", 0.8)):
        key = lighting.key or lights[0]
        shadows.append(cast_shadow(alpha, rng=env_rng, light=key,
                                   hardness=key.hardness))
    phone_shadow_chance = float(lighting_cfg.get("phone_shadow_chance", 0.06))
    if "occlusion" in problems:
        phone_shadow_chance = max(phone_shadow_chance, 0.3)
    if env_rng.random() < phone_shadow_chance:
        shadows.append(phone_shadow(render_w, render_h, env_rng))
    if env_rng.random() < float(lighting_cfg.get("hand_shadow_chance", 0.2)):
        shadows.append(hand_shadow(render_w, render_h, env_rng))

    # ---- 8. occluders ---------------------------------------------------
    occ_rng = streams.get("occlusion")
    count_weights = list(occlusion_cfg.get("count_weights", [0.62, 0.24, 0.1, 0.04]))
    forced_kind = None
    if "occlusion" in problems:
        count_weights = [0.05, 0.2, 0.4, 0.35]
        forced_kind = None
    count_probs = np.array(count_weights, dtype=np.float64)
    count_probs = count_probs / count_probs.sum()
    occ_count = int(occ_rng.choice(len(count_probs), p=count_probs))
    occluders = sample_occluders(
        (render_w, render_h), occ_rng, count=occ_count,
        allow_hand=bool(occlusion_cfg.get("allow_hand", True)),
        other_card=other_card, forced_kind=forced_kind,
    )

    layers = SceneLayers(
        background=background,
        card_rgb=placed,
        card_alpha=alpha,
        shadows=shadows,
        occluders=occluders,
        ambient_pool=bool(env_cfg.get("ambient_pool", True)),
        frame_vignette=float(lighting.vignette) * 0.5,
        glare_bleed=glare * 0.25 if glare is not None else None,
    )
    scene = render_scene(layers, env_rng)

    # ---- 9. camera + artifacts -----------------------------------------
    cam_rng = streams.get("camera")
    camera = sample_camera(
        cam_rng, resolution=(render_w, render_h), mode=mode,
        device=str(cam_cfg.get("device", "phone_normal")),
        base_iso=int(cam_cfg.get("base_iso", 160)),
    )
    camera.lens_k1 = float(_draw(cam_rng, cam_cfg.get("lens_distortion"), camera.lens_k1))
    camera.chromatic = float(_draw(cam_rng, cam_cfg.get("chromatic"), camera.chromatic))
    camera.sharpen_amount = float(_draw(cam_rng, cam_cfg.get("sharpen"),
                                        camera.sharpen_amount))
    camera.denoise_strength = float(_draw(cam_rng, cam_cfg.get("denoise"),
                                          camera.denoise_strength))
    if "blur" in problems:
        camera.defocus_sigma = max(camera.defocus_sigma, float(cam_rng.uniform(1.5, 3.5)))
        camera.motion_length = max(camera.motion_length, float(cam_rng.uniform(4.0, 10.0)))
        camera.motion_angle = float(cam_rng.uniform(0, 2 * math.pi))
    quality_level = float(artifact_cfg.get("level", 0.35))
    if "compression" in problems:
        camera.jpeg_quality = int(min(camera.jpeg_quality, 28))
        quality_level = max(quality_level, 0.8)

    captured = apply_optics(scene, camera, cam_rng)
    captured = apply_sensor(captured, camera, cam_rng)
    captured, applied_artifacts = apply_artifacts(
        captured, cam_rng, level=quality_level,
        allow_double_jpeg=bool(artifact_cfg.get("double_jpeg_chance", 0.1)),
        scanner_mode=bool(artifact_cfg.get("scanner_mode", False)),
    )
    if config.jpeg_on_final:
        captured = apply_encoding(captured, camera)

    image_u8 = to_uint8(np.clip(captured, 0.0, 1.0))

    # ---- 10. ground truth ----------------------------------------------
    annotation, quality = _build_annotation(
        card=card, geom=geom, image=image_u8, occluders=occluders,
        config=config, seed=seed, index=index, image_id=image_id,
        profile_name=str(profile.get("name", "custom")),
        finish_name=finish.name,
        camera=camera, lighting=lighting, surface=surface,
        geometry_pose=pose, applied_artifacts=applied_artifacts,
        problems=sorted(problems),
    )
    if not quality["is_usable"]:
        raise _UnusableSample(quality)

    return GeneratedSample(
        image=image_u8,
        annotation=annotation,
        profile=str(profile.get("name", "custom")),
        seed=seed,
        index=index,
        quality=quality,
        generation=annotation.generation,
    )


class _UnusableSample(GenerationError):
    """Raised internally when a sample fails quality control."""

    def __init__(self, quality: dict):
        super().__init__("Generated sample is not usable")
        self.quality = quality


def _with_lighting_override(profile: dict, overrides: dict) -> dict:
    from ..config import deep_merge

    return deep_merge(profile, {"lighting": overrides})


def _build_annotation(
    *,
    card: SourceCard,
    geom,
    image: np.ndarray,
    occluders,
    config: GenerationConfig,
    seed: int,
    index: int,
    image_id: str | None,
    profile_name: str,
    finish_name: str,
    camera,
    lighting,
    surface,
    geometry_pose,
    applied_artifacts: dict,
    problems: list[str],
) -> tuple[Annotation, dict]:
    """Create the annotation with exact field boxes and full provenance."""
    height, width = image.shape[:2]
    frame_size = (width, height)

    # Card boundary, clipped to the frame for sanity but stored unclipped too.
    card_quad = Quad(tuple((float(x), float(y)) for x, y in geom.corners))
    visibility = quad_visible_fraction(card_quad, width, height)
    from .render import occlusion_mask

    occ_mask = occlusion_mask(occluders, frame_size) if occluders else None

    rois = card.effective_rois()
    warped = warp_field_boxes(geom, rois, clamp=True)
    fields: dict[str, FieldAnnotation] = {}
    legibility: dict[str, float] = {}
    occluded: dict[str, float] = {}
    for name, text in card.fields.items():
        if name not in warped:
            continue
        x0, y0, x1, y1 = warped[name]
        if x1 - x0 < 1.5 or y1 - y0 < 1.5:
            continue
        box = BBox(x0, y0, x1, y1)
        # A field entirely outside the frame carries no information.
        if box.intersection_area(BBox(0, 0, width, height)) <= 1.0:
            continue
        card_roi = rois[name]
        cw, ch = card.size
        fields[name] = FieldAnnotation(
            text=text,
            bbox=box,
            roi=BBox(card_roi[0] / cw, card_roi[1] / ch, card_roi[2] / cw, card_roi[3] / ch),
        )
        legibility[name] = field_legibility(image, (x0, y0, x1, y1))
        if occ_mask is not None:
            occluded[name] = occlusion_coverage(occ_mask, (x0, y0, x1, y1))

    card_area_fraction = geom.visibility if visibility > 1 else visibility
    quality_metrics = measure_quality(image, card_visibility=visibility,
                                      card_area_fraction=card_area_fraction)
    min_legibility = min(legibility.values()) if legibility else 0.0
    max_occlusion = max(occluded.values()) if occluded else 0.0
    quality = quality_metrics.to_dict()
    quality.update({
        "legibility": {k: round(v, 4) for k, v in legibility.items()},
        "occlusion": {k: round(v, 4) for k, v in occluded.items()},
        "min_legibility": round(min_legibility, 4),
        "max_field_occlusion": round(max_occlusion, 4),
        "is_usable": bool(quality_metrics.is_usable
                          and visibility >= config.min_visibility
                          and min_legibility >= config.min_legibility),
    })

    generation = {
        "engine": "cardvault-vision-synthetic",
        "profile": profile_name,
        "mode": config.mode,
        "problems": problems,
        "seed": seed,
        "index": index,
        "stream_seeds": RngStreams(master_seed=seed, sample_index=index).seeds(),
        "geometry": geom.to_dict(),
        "camera": camera.to_dict(),
        "lighting": lighting.to_dict(),
        "surface": surface.to_dict(),
        "artifacts": applied_artifacts,
        "finish": finish_name,
    }
    annotation = Annotation(
        image_id=image_id or f"synthetic_{index:08d}",
        card_id=card.card_id,
        card_type=card.card_type,
        fields=fields,
        card_quad=card_quad,
        width=width,
        height=height,
        knowledge=card.knowledge,
        generation=generation,
    )
    return annotation, quality
