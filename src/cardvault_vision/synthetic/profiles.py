"""Camera capture profiles (Phase 2.8).

A profile is a *probability distribution over capture conditions*, not a fixed
configuration. Each one describes how a class of real user holds a phone: the
jitter of a child, the flat control of a scanner, the badly-lit kitchen table.

Profiles are plain dicts (so they can be overridden by YAML) validated into a
:class:`CaptureProfile`. Every field is a *distribution description* that the
generator resolves with its own random stream.

Adding ``CUSTOM`` support means a user can start from a profile and override any
distribution, e.g.::

    synthetic:
      profile: phone_bad_lighting
      geometry:
        max_pitch: 35          # allow worse angles
      lighting:
        brightness:
          sigma: 0.8           # dimmer, more variable
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..config import deep_merge
from ..errors import ConfigError


def _uniform(low: float, high: float) -> dict:
    return {"type": "uniform", "low": low, "high": high}


def _normal(mu: float, sigma: float, clamp: tuple[float, float] | None = None) -> dict:
    out = {"type": "normal", "mean": mu, "sigma": sigma}
    if clamp:
        out["clamp"] = list(clamp)
    return out


def _choice(options: list, weights: list[float] | None = None) -> dict:
    out: dict[str, Any] = {"type": "choice", "options": options}
    if weights:
        out["weights"] = weights
    return out


def _empty_geometry() -> dict:
    return {
        "max_roll": 8.0,
        "max_pitch": 12.0,
        "max_yaw": 12.0,
        "distance": list(_range_pair(_uniform(2.4, 5.5))),
        "scale": _uniform(0.6, 0.92),
        "offset_sigma": 0.06,
        "edge_warp": _uniform(0.0, 0.6),
        "crop_chance": 0.05,
        "min_visibility": 0.55,
    }


def _range_pair(d: dict) -> tuple[float, float]:
    return (d["low"], d["high"])


def _empty_lighting() -> dict:
    return {
        "brightness": _normal(0.0, 0.35, (-1.6, 1.2)),
        "contrast": _normal(1.0, 0.1, (0.6, 1.5)),
        "gamma": _normal(1.0, 0.06, (0.8, 1.25)),
        "kelvin": _uniform(3200, 7500),
        "ambient": _normal(0.35, 0.1, (0.05, 0.95)),
        "uneven": {"type": "half_normal", "sigma": 0.15, "clamp": [0.0, 0.8]},
        "vignette": {"type": "half_normal", "sigma": 0.1, "clamp": [0.0, 0.5]},
        "key_hardness": _uniform(0.1, 0.9),
        "key_intensity": _uniform(0.5, 1.2),
        "fill_chance": 0.6,
        "overexpose_chance": 0.04,
        "underexpose_chance": 0.05,
        "shadow_chance": 0.8,
        "phone_shadow_chance": 0.06,
        "hand_shadow_chance": 0.2,
        "glare_chance": 0.45,
        "glare_strength": _uniform(0.15, 0.7),
    }


def _empty_camera() -> dict:
    return {
        "device": "phone_normal",
        "base_iso": 160,
        "resolution": _choice(["phone_modern", "phone_normal", "phone_normal", "phone_old"]),
        "focus_fail_chance": 0.18,
        "shake_chance": 0.2,
        "jpeg_quality": list((70, 95)),
        "jpeg_quality_dist": _normal(88, 6, (30, 98)),
        "lens_distortion": _normal(-0.03, 0.04, (-0.25, 0.15)),
        "chromatic": {"type": "half_normal", "sigma": 0.5, "clamp": [0.0, 3.0]},
        "sharpen": _normal(0.7, 0.3, (0.0, 2.5)),
        "denoise": _normal(0.2, 0.15, (0.0, 0.8)),
    }


def _empty_environment() -> dict:
    return {
        "backgrounds": _choice(list(_BACKGROUND_WEIGHTS)),
        "background_weights": list(_BACKGROUND_WEIGHTS.values()),
        "dark_chance": 0.35,
        "ambient_pool": True,
        "clutter_chance": 0.12,
    }


_BACKGROUND_WEIGHTS = {
    "wood": 0.22,
    "desk": 0.18,
    "carpet": 0.10,
    "play_mat": 0.12,
    "binder": 0.06,
    "sleeve": 0.06,
    "paper": 0.08,
    "neutral": 0.10,
    "tablecloth": 0.04,
    "clutter": 0.04,
}


def _empty_occlusion() -> dict:
    return {
        "count_weights": [0.62, 0.24, 0.10, 0.04],
        "kinds": _choice(["finger", "sleeve", "card", "object", "edge", "hand"]),
        "kind_weights": [0.42, 0.16, 0.16, 0.16, 0.10, 0.14],
        "allow_hand": True,
    }


def _empty_artifacts() -> dict:
    return {
        "level": 0.35,
        "double_jpeg_chance": 0.1,
        "scanner_mode": False,
    }


def _empty_surface() -> dict:
    return {
        "finish_weights": {
            "normal": 0.45,
            "glossy": 0.20,
            "holo": 0.12,
            "reverse_holo": 0.08,
            "foil": 0.06,
            "etched_foil": 0.03,
            "chrome": 0.02,
            "refractor": 0.03,
            "matte": 0.01,
        },
        "wear": {"type": "half_normal", "sigma": 0.25, "clamp": [0.0, 1.0]},
        "glare_chance": 0.45,
        "sleeve_glare_chance": 0.2,
    }


def default_profile_dict() -> dict:
    """The complete, fully-specified ``normal`` profile."""
    return {
        "name": "custom",  # replaced by CaptureProfile.get/create with the real name
        "mode": "normal",
        "target_size": [1240, 1754],
        "supersample": 2,
        "geometry": _empty_geometry(),
        "lighting": _empty_lighting(),
        "camera": _empty_camera(),
        "environment": _empty_environment(),
        "occlusion": _empty_occlusion(),
        "artifacts": _empty_artifacts(),
        "surface": _empty_surface(),
    }


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

def _phone_normal() -> dict:
    return default_profile_dict()


def _phone_bad_lighting() -> dict:
    cfg = default_profile_dict()
    cfg["name"] = "phone_bad_lighting"
    cfg["mode"] = "poor"
    cfg["lighting"] = {
        **cfg["lighting"],
        "brightness": _normal(-0.7, 0.6, (-2.5, 0.8)),
        "contrast": _normal(0.85, 0.15, (0.5, 1.3)),
        "kelvin": _uniform(2400, 4200),
        "ambient": _normal(0.2, 0.08, (0.03, 0.6)),
        "uneven": {"type": "half_normal", "sigma": 0.3, "clamp": [0.0, 0.95]},
        "vignette": {"type": "half_normal", "sigma": 0.2, "clamp": [0.0, 0.6]},
        "underexpose_chance": 0.28,
        "overexpose_chance": 0.05,
        "phone_shadow_chance": 0.22,
        "glare_chance": 0.6,
    }
    cfg["camera"] = {
        **cfg["camera"],
        "base_iso": 1250,
        "device": "phone_old",
        "resolution": _choice(["phone_normal", "phone_old", "phone_old"]),
        "focus_fail_chance": 0.32,
        "shake_chance": 0.34,
        "denoise": _normal(0.45, 0.2, (0.0, 0.9)),
        "sharpen": _normal(1.0, 0.4, (0.0, 2.5)),
    }
    cfg["artifacts"] = {**cfg["artifacts"], "level": 0.7, "double_jpeg_chance": 0.2}
    return cfg


def _phone_low_end() -> dict:
    cfg = default_profile_dict()
    cfg["name"] = "phone_low_end"
    cfg["mode"] = "poor"
    cfg["target_size"] = [720, 1018]
    cfg["camera"] = {
        **cfg["camera"],
        "device": "phone_low_end",
        "base_iso": 400,
        "resolution": _choice(["phone_old", "phone_low_end", "phone_low_end"]),
        "focus_fail_chance": 0.4,
        "shake_chance": 0.3,
        "jpeg_quality": (40, 80),
        "jpeg_quality_dist": _normal(62, 10, (20, 90)),
        "lens_distortion": _normal(-0.06, 0.06, (-0.35, 0.1)),
        "chromatic": {"type": "half_normal", "sigma": 1.1, "clamp": [0.0, 4.0]},
        "denoise": _normal(0.55, 0.2, (0.0, 0.95)),
        "sharpen": _normal(1.3, 0.5, (0.0, 3.0)),
    }
    cfg["artifacts"] = {**cfg["artifacts"], "level": 0.85, "double_jpeg_chance": 0.3}
    cfg["lighting"] = {**cfg["lighting"], "underexpose_chance": 0.18,
                       "glare_chance": 0.55}
    return cfg


def _child_camera() -> dict:
    cfg = default_profile_dict()
    cfg["name"] = "child_camera"
    cfg["mode"] = "poor"
    cfg["geometry"] = {
        **cfg["geometry"],
        "max_roll": 35.0,
        "max_pitch": 38.0,
        "max_yaw": 38.0,
        "offset_sigma": 0.16,
        "crop_chance": 0.3,
        "min_visibility": 0.35,
        "scale": _uniform(0.4, 0.95),
    }
    cfg["camera"] = {
        **cfg["camera"],
        "device": "child",
        "base_iso": 600,
        "resolution": _choice(["phone_old", "phone_low_end", "child"]),
        "focus_fail_chance": 0.55,
        "shake_chance": 0.65,
        "jpeg_quality_dist": _normal(72, 12, (20, 95)),
    }
    cfg["lighting"] = {
        **cfg["lighting"],
        "brightness": _normal(-0.2, 0.8, (-2.8, 1.4)),
        "kelvin": _uniform(2600, 8000),
        "uneven": {"type": "half_normal", "sigma": 0.3, "clamp": [0.0, 0.95]},
        "overexpose_chance": 0.16,
        "underexpose_chance": 0.2,
        "phone_shadow_chance": 0.35,
        "hand_shadow_chance": 0.45,
    }
    cfg["occlusion"] = {
        **cfg["occlusion"],
        "count_weights": [0.3, 0.34, 0.24, 0.12],
    }
    cfg["artifacts"] = {**cfg["artifacts"], "level": 0.9, "double_jpeg_chance": 0.35}
    return cfg


def _scanner() -> dict:
    cfg = default_profile_dict()
    cfg["name"] = "scanner"
    cfg["mode"] = "clean"
    cfg["target_size"] = [1476, 2079]   # ~600 dpi for a 63 x 88 mm card
    cfg["supersample"] = 2
    cfg["geometry"] = {
        **cfg["geometry"],
        "max_roll": 1.2,
        "max_pitch": 1.5,
        "max_yaw": 1.5,
        "scale": _uniform(0.88, 0.96),
        "offset_sigma": 0.015,
        "edge_warp": _uniform(0.0, 0.05),
        "crop_chance": 0.02,
        "min_visibility": 0.9,
    }
    cfg["lighting"] = {
        **cfg["lighting"],
        "brightness": _normal(0.05, 0.08, (-0.2, 0.3)),
        "contrast": _normal(1.02, 0.03, (0.9, 1.15)),
        "kelvin": _uniform(5800, 6800),
        "ambient": _normal(0.95, 0.03, (0.85, 1.0)),
        "uneven": {"type": "half_normal", "sigma": 0.03, "clamp": [0.0, 0.15]},
        "vignette": {"type": "half_normal", "sigma": 0.02, "clamp": [0.0, 0.1]},
        "key_hardness": _uniform(0.0, 0.15),
        "key_intensity": _uniform(0.1, 0.3),
        "fill_chance": 1.0,
        "overexpose_chance": 0.01,
        "underexpose_chance": 0.005,
        "shadow_chance": 0.0,
        "phone_shadow_chance": 0.0,
        "hand_shadow_chance": 0.0,
        "glare_chance": 0.12,
        "glare_strength": _uniform(0.05, 0.25),
    }
    cfg["camera"] = {
        **cfg["camera"],
        "device": "scanner",
        "base_iso": 100,
        "resolution": _choice(["scanner"]),
        "focus_fail_chance": 0.01,
        "shake_chance": 0.01,
        "jpeg_quality": (92, 98),
        "jpeg_quality_dist": _normal(96, 2, (85, 100)),
        "lens_distortion": _normal(0.0, 0.005, (-0.02, 0.02)),
        "chromatic": {"type": "half_normal", "sigma": 0.05, "clamp": [0.0, 0.3]},
        "sharpen": _normal(0.2, 0.15, (0.0, 0.8)),
        "denoise": _normal(0.05, 0.05, (0.0, 0.3)),
    }
    cfg["environment"] = {
        **cfg["environment"],
        "backgrounds": _choice(["neutral", "paper", "desk"]),
        "background_weights": [0.6, 0.25, 0.15],
        "dark_chance": 0.05,
        "ambient_pool": False,
        "clutter_chance": 0.0,
    }
    cfg["occlusion"] = {**cfg["occlusion"], "count_weights": [0.9, 0.1, 0.0, 0.0]}
    cfg["artifacts"] = {"level": 0.12, "double_jpeg_chance": 0.0, "scanner_mode": True}
    cfg["surface"] = {**cfg["surface"], "glare_chance": 0.1,
                      "wear": {"type": "half_normal", "sigma": 0.12, "clamp": [0.0, 0.6]}}
    return cfg


def _extreme() -> dict:
    cfg = default_profile_dict()
    cfg["name"] = "extreme"
    cfg["mode"] = "extreme"
    cfg["geometry"] = {
        **cfg["geometry"],
        "max_roll": 40.0,
        "max_pitch": 45.0,
        "max_yaw": 45.0,
        "offset_sigma": 0.2,
        "crop_chance": 0.4,
        "min_visibility": 0.3,
        "scale": _uniform(0.35, 0.95),
    }
    cfg["lighting"] = {
        **cfg["lighting"],
        "brightness": _normal(-0.4, 1.0, (-3.2, 1.6)),
        "contrast": _normal(0.8, 0.25, (0.35, 1.6)),
        "kelvin": _uniform(2000, 9500),
        "ambient": _normal(0.22, 0.15, (0.02, 0.9)),
        "uneven": {"type": "half_normal", "sigma": 0.4, "clamp": [0.0, 1.0]},
        "vignette": {"type": "half_normal", "sigma": 0.25, "clamp": [0.0, 0.7]},
        "overexpose_chance": 0.2,
        "underexpose_chance": 0.3,
        "phone_shadow_chance": 0.35,
        "hand_shadow_chance": 0.4,
        "glare_chance": 0.85,
        "glare_strength": _uniform(0.35, 1.0),
    }
    cfg["camera"] = {
        **cfg["camera"],
        "device": "phone_low_end",
        "base_iso": 2400,
        "resolution": _choice(["phone_old", "phone_low_end", "child"]),
        "focus_fail_chance": 0.7,
        "shake_chance": 0.7,
        "jpeg_quality_dist": _normal(48, 15, (15, 85)),
        "lens_distortion": _normal(-0.1, 0.08, (-0.4, 0.1)),
        "chromatic": {"type": "half_normal", "sigma": 1.6, "clamp": [0.0, 5.0]},
        "denoise": _normal(0.6, 0.2, (0.0, 0.95)),
        "sharpen": _normal(1.5, 0.5, (0.0, 3.0)),
    }
    cfg["occlusion"] = {**cfg["occlusion"], "count_weights": [0.2, 0.3, 0.3, 0.2]}
    cfg["artifacts"] = {**cfg["artifacts"], "level": 1.2, "double_jpeg_chance": 0.45}
    cfg["surface"] = {**cfg["surface"], "glare_chance": 0.85,
                      "finish_weights": {
                          "normal": 0.2, "glossy": 0.2, "holo": 0.2, "reverse_holo": 0.1,
                          "foil": 0.1, "etched_foil": 0.05, "chrome": 0.05,
                          "refractor": 0.08, "matte": 0.02}}
    return cfg


def _clean() -> dict:
    cfg = default_profile_dict()
    cfg["name"] = "clean"
    cfg["mode"] = "clean"
    cfg["geometry"] = {
        **cfg["geometry"],
        "max_roll": 2.0,
        "max_pitch": 2.5,
        "max_yaw": 2.5,
        "scale": _uniform(0.82, 0.94),
        "offset_sigma": 0.02,
        "edge_warp": _uniform(0.0, 0.1),
        "crop_chance": 0.01,
        "min_visibility": 0.85,
    }
    cfg["lighting"] = {
        **cfg["lighting"],
        "brightness": _normal(0.05, 0.15, (-0.4, 0.4)),
        "contrast": _normal(1.0, 0.04, (0.9, 1.1)),
        "kelvin": _uniform(5500, 6500),
        "ambient": _normal(0.7, 0.1, (0.4, 0.95)),
        "uneven": {"type": "half_normal", "sigma": 0.05, "clamp": [0.0, 0.2]},
        "vignette": {"type": "half_normal", "sigma": 0.03, "clamp": [0.0, 0.12]},
        "overexpose_chance": 0.005,
        "underexpose_chance": 0.005,
        "phone_shadow_chance": 0.0,
        "hand_shadow_chance": 0.02,
        "glare_chance": 0.1,
        "glare_strength": _uniform(0.05, 0.25),
    }
    cfg["camera"] = {
        **cfg["camera"],
        "device": "phone_modern",
        "base_iso": 100,
        "resolution": _choice(["phone_modern"]),
        "focus_fail_chance": 0.02,
        "shake_chance": 0.02,
        "jpeg_quality_dist": _normal(94, 3, (80, 100)),
        "denoise": _normal(0.1, 0.08, (0.0, 0.4)),
        "sharpen": _normal(0.4, 0.2, (0.0, 1.2)),
    }
    cfg["occlusion"] = {**cfg["occlusion"], "count_weights": [0.95, 0.05, 0.0, 0.0]}
    cfg["artifacts"] = {"level": 0.08, "double_jpeg_chance": 0.0, "scanner_mode": False}
    cfg["environment"] = {**cfg["environment"], "dark_chance": 0.1, "clutter_chance": 0.0}
    return cfg


BUILTIN_PROFILES: dict[str, dict] = {
    "phone_normal": _phone_normal(),
    "phone": _phone_normal(),
    "phone_bad_lighting": _phone_bad_lighting(),
    "phone_low_end": _phone_low_end(),
    "child_camera": _child_camera(),
    "child": _child_camera(),
    "scanner": _scanner(),
    "extreme": _extreme(),
    "clean": _clean(),
    "normal": _phone_normal(),
}

PROFILE_NAMES = tuple(sorted({
    "phone_normal", "phone_bad_lighting", "phone_low_end", "child_camera",
    "scanner", "extreme", "clean", "custom",
}))

#: Presets for the synthetic dataset modes (Phase 2.11).
MODE_PRESETS = {
    "clean": {"profile": "clean", "overrides": {}},
    "normal": {"profile": "phone_normal", "overrides": {}},
    "poor": {"profile": "phone_bad_lighting", "overrides": {}},
    "extreme": {"profile": "extreme", "overrides": {}},
    "targeted": {"profile": "phone_normal", "overrides": {}},
}

#: "Mix" presets combine profiles with weights, so a single run can cover the
#: whole distribution a production model will see.
MIX_PRESETS: dict[str, dict[str, float]] = {
    "realistic": {"phone_normal": 0.55, "phone_bad_lighting": 0.2,
                  "phone_low_end": 0.12, "child_camera": 0.05, "scanner": 0.08},
    "all": {"clean": 0.1, "phone_normal": 0.3, "phone_bad_lighting": 0.2,
            "phone_low_end": 0.15, "child_camera": 0.1, "scanner": 0.05, "extreme": 0.1},
}


@dataclass
class CaptureProfile:
    """A resolved, validated capture profile.

    Wraps the plain dict so callers get dot access and validation, while the dict
    keeps YAML overrides trivial and the whole thing JSON-serialisable for the
    annotation.
    """

    data: dict = field(default_factory=default_profile_dict)

    @classmethod
    def get(cls, name: str) -> "CaptureProfile":
        """Load a built-in profile by name."""
        key = name.lower()
        if key not in BUILTIN_PROFILES:
            raise ConfigError(
                f"Unknown profile {name!r}; expected one of {sorted(BUILTIN_PROFILES)}"
            )
        data = deep_merge(default_profile_dict(), BUILTIN_PROFILES[key])
        # The stored name is the canonical key, not whatever the base defaults to.
        data["name"] = key
        return cls(data)

    @classmethod
    def custom(cls, base: str = "phone_normal", overrides: Mapping[str, Any] | None = None
               ) -> "CaptureProfile":
        """Start from a built-in profile and deep-merge overrides onto it."""
        profile = cls.get(base)
        if overrides:
            profile.data = deep_merge(profile.data, overrides)
            profile.data["name"] = overrides.get("name", f"custom:{base}")
        profile.validate()
        return profile

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CaptureProfile":
        merged = deep_merge(default_profile_dict(), dict(data))
        profile = cls(merged)
        profile.validate()
        return profile

    def validate(self) -> None:
        """Check internal consistency; raise :class:`ConfigError` on nonsense."""
        data = self.data
        if not isinstance(data.get("target_size"), (list, tuple)) or \
                len(data["target_size"]) != 2:
            raise ConfigError("profile.target_size must be [width, height]")
        width, height = data["target_size"]
        if not (64 <= int(width) <= 12000 and 64 <= int(height) <= 12000):
            raise ConfigError(f"profile.target_size {width}x{height} is out of range")
        for section in ("geometry", "lighting", "camera", "environment", "occlusion",
                        "artifacts", "surface"):
            if section not in data:
                raise ConfigError(f"profile is missing section {section!r}")
        geom = data["geometry"]
        for key in ("max_roll", "max_pitch", "max_yaw"):
            if float(geom.get(key, 0)) < 0 or float(geom.get(key, 0)) > 90:
                raise ConfigError(f"profile.geometry.{key} must be within [0, 90] degrees")
        if not 0.0 < float(geom.get("min_visibility", 0.5)) <= 1.0:
            raise ConfigError("profile.geometry.min_visibility must be in (0, 1]")
        weights = data["surface"].get("finish_weights", {})
        if weights and abs(sum(weights.values())) <= 0:
            raise ConfigError("profile.surface.finish_weights must not be all zero")
        if int(data.get("supersample", 2)) < 1:
            raise ConfigError("profile.supersample must be >= 1")

    @property
    def name(self) -> str:
        return str(self.data.get("name", "custom"))

    @property
    def mode(self) -> str:
        return str(self.data.get("mode", "normal"))

    @property
    def target_size(self) -> tuple[int, int]:
        width, height = self.data["target_size"]
        return int(width), int(height)

    @property
    def supersample(self) -> int:
        return int(self.data.get("supersample", 2))

    def section(self, name: str) -> dict:
        return dict(self.data.get(name, {}))

    def to_dict(self) -> dict:
        return dict(self.data)


def resolve_profiles(spec: str | list[str] | Mapping[str, float] | None) -> dict[str, float]:
    """Turn a CLI/profile spec into a ``{profile_name: weight}`` mapping.

    Accepts ``"phone"``, ``"realistic"`` (a mix preset), ``"phone_normal:0.7,scanner:0.3"``
    or a mapping.
    """
    if spec is None:
        return {"phone_normal": 1.0}
    if isinstance(spec, Mapping):
        weights = {str(k): float(v) for k, v in spec.items()}
    else:
        text = str(spec).strip()
        if text.lower() in MIX_PRESETS:
            weights = dict(MIX_PRESETS[text.lower()])
        elif "," in text:
            weights = {}
            for part in text.split(","):
                part = part.strip()
                if not part:
                    continue
                if ":" in part:
                    name, weight = part.split(":", 1)
                    weights[name.strip()] = float(weight)
                else:
                    weights[part] = 1.0
        elif ":" in text:
            name, weight = text.split(":", 1)
            weights = {name.strip(): float(weight)}
        else:
            weights = {text: 1.0}
    for name in weights:
        if name.lower() not in BUILTIN_PROFILES and name.lower() not in MIX_PRESETS:
            raise ConfigError(
                f"Unknown profile {name!r}; available: {sorted(BUILTIN_PROFILES)} "
                f"or mixes {sorted(MIX_PRESETS)}"
            )
    total = sum(weights.values())
    if total <= 0:
        raise ConfigError("Profile weights must sum to a positive value")
    return {name: weight / total for name, weight in weights.items()}
