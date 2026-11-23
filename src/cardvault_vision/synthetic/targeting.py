"""Targeted synthetic generation (Phase 2.12).

The engine can be told *what* to make hard:

    vision synthetic generate --problem glare --count 10000

Each problem is expressed as an override on the resolved profile plus a handful of
scalar knobs the generator reads. Keeping the targeting logic here (rather than
scattered through the pipeline) means the set of supported problems, and how each
one is realised, is auditable in one place - and it is the same vocabulary hard
example mining uses (Phase 9) when it labels a real failure.

| problem      | realised by                                               |
|--------------|-----------------------------------------------------------|
| perspective  | wider pitch/yaw/roll ranges, larger offsets               |
| glare        | glare always on, strongest strength, holo-biased finishes |
| blur         | forced defocus/motion blur, slower shutter                |
| low_light    | negative exposure bias, high ISO, warm colour cast        |
| occlusion    | more occluders, more phone shadows                        |
| tiny_text    | smaller card in frame, lower resolution                    |
| compression  | very low JPEG quality, double compression, banding        |
"""

from __future__ import annotations

from ..config import deep_merge
from ..errors import ConfigError

PROBLEM_TYPES = (
    "perspective",
    "glare",
    "blur",
    "low_light",
    "occlusion",
    "tiny_text",
    "compression",
)

#: Human-readable description, used by the CLI and docs generation.
PROBLEM_DESCRIPTIONS = {
    "perspective": "Extreme card angle and off-centre framing",
    "glare": "Strong specular glare, holo/foil finishes",
    "blur": "Defocus and motion blur beyond the recovery threshold",
    "low_light": "Dim, warm, high-ISO indoor capture",
    "occlusion": "Fingers, hands, other cards and phone shadows",
    "tiny_text": "Card occupies a small part of the frame",
    "compression": "Heavy JPEG artefacts and double compression",
}

#: Severity levels supported by the CLI (``--problem glare:2``).
SEVERITY_RANGE = (1, 3)


def _severity(value: float) -> float:
    """Map a 1..3 severity onto the 0.5..1.5 multiplier used below."""
    low, high = SEVERITY_RANGE
    value = min(max(value, low), high)
    return 0.5 + (value - low) / (high - low)


def problem_overrides(problem: str, severity: float = 2.0) -> dict:
    """Return the profile overrides that realise ``problem`` at ``severity``."""
    if problem not in PROBLEM_TYPES:
        raise ConfigError(
            f"Unknown problem {problem!r}; expected one of {PROBLEM_TYPES}"
        )
    scale = _severity(severity)

    if problem == "perspective":
        return {
            "geometry": {
                "max_roll": 20.0 * scale,
                "max_pitch": 38.0 * scale,
                "max_yaw": 38.0 * scale,
                "offset_sigma": 0.09 * scale,
                "crop_chance": min(0.45, 0.12 * scale),
            }
        }
    if problem == "glare":
        return {
            "surface": {
                "glare_chance": 1.0,
                "finish_weights": {
                    "normal": 0.05, "glossy": 0.15, "holo": 0.3,
                    "reverse_holo": 0.15, "foil": 0.15, "etched_foil": 0.05,
                    "chrome": 0.05, "refractor": 0.08, "matte": 0.02,
                },
            },
            "lighting": {
                "glare_chance": 1.0,
                "glare_strength": [0.6, 1.0],
                "key_intensity": [1.0, 1.6],
                "key_hardness": [0.6, 1.0],
            },
        }
    if problem == "blur":
        return {
            "camera": {
                "focus_fail_chance": min(0.95, 0.55 * scale),
                "shake_chance": min(0.95, 0.5 * scale),
            }
        }
    if problem == "low_light":
        return {
            "lighting": {
                "brightness": {"type": "normal", "mean": -0.9 * scale, "sigma": 0.35,
                               "clamp": [-2.8, -0.2]},
                "underexpose_chance": min(0.6, 0.35 * scale),
                "kelvin": {"type": "uniform", "low": 2400, "high": 3600},
                "ambient": {"type": "normal", "mean": 0.18, "sigma": 0.08,
                            "clamp": [0.03, 0.5]},
            },
            "camera": {"base_iso": int(1600 * scale), "denoise": [0.4, 0.8]},
        }
    if problem == "occlusion":
        return {
            "occlusion": {"count_weights": [0.05, 0.2, 0.4, 0.35]},
            "lighting": {"phone_shadow_chance": min(0.5, 0.25 * scale),
                         "hand_shadow_chance": min(0.6, 0.3 * scale)},
            "geometry": {"crop_chance": min(0.35, 0.12 * scale)},
        }
    if problem == "tiny_text":
        return {
            "geometry": {"scale": [0.28, 0.55], "offset_sigma": 0.1 * scale},
            "camera": {
                "resolution": {"type": "choice",
                               "options": ["phone_old", "phone_low_end"]},
                "device": "phone_low_end",
            },
        }
    # compression
    return {
        "camera": {
            "jpeg_quality_dist": {"type": "normal", "mean": 26, "sigma": 8,
                                  "clamp": [10, 45]},
        },
        "artifacts": {"level": 1.2, "double_jpeg_chance": 0.8},
    }


def apply_problem(profile: dict, problems: list[str] | None,
                  severities: dict[str, float] | None = None) -> dict:
    """Deep-merge the overrides for each problem into a profile."""
    if not problems:
        return dict(profile)
    merged = dict(profile)
    for problem in problems:
        severity = (severities or {}).get(problem, 2.0)
        merged = deep_merge(merged, problem_overrides(problem, severity))
    merged["targeted"] = list(problems)
    return merged


def parse_problem_spec(spec: str) -> tuple[str, float]:
    """Parse ``glare`` or ``glare:3`` into ``(name, severity)``."""
    text = spec.strip()
    if ":" in text:
        name, value = text.split(":", 1)
        try:
            severity = float(value)
        except ValueError as exc:
            raise ConfigError(f"Invalid severity in {spec!r}") from exc
        return name.strip(), severity
    return text, 2.0


def problems_for_targets(targets: list[str]) -> list[str]:
    """Validate a list of problem names (used by the CLI's ``--problem``)."""
    out = []
    for spec in targets:
        name, severity = parse_problem_spec(spec)
        if name not in PROBLEM_TYPES:
            raise ConfigError(f"Unknown problem {name!r}; expected one of {PROBLEM_TYPES}")
        if not SEVERITY_RANGE[0] <= severity <= SEVERITY_RANGE[1]:
            raise ConfigError(
                f"Severity {severity} out of range for {name!r}; "
                f"expected {SEVERITY_RANGE[0]}..{SEVERITY_RANGE[1]}"
            )
        out.append(name)
    return out
