"""Layout definitions and resolution.

Coordinates are normalised to the card: ``x`` in ``[0, 1]`` across the width, ``y``
in ``[0, 1]`` down the height, origin top-left. This makes one layout valid for
any resolution, which is essential because the same card arrives as a 400 px
crop from a phone and a 2000 px scan.

The ROIs below are approximations of the real templates. They are deliberately
kept in one place, in normalised space, so they can be corrected against real
data without touching the engine (see ``docs/architecture/layouts.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

from ..config import CONFIG_DIR, load_yaml
from ..errors import ConfigError
from ..logging_utils import get_logger

_LOG = get_logger("layouts")


@dataclass(frozen=True)
class LayoutField:
    """One named region on a card, in normalised card coordinates."""

    name: str
    x0: float
    y0: float
    x1: float
    y1: float
    #: Optional per-field reading order or grouping (e.g. "title", "corner").
    group: str | None = None
    #: Expected text style, used by the text renderer.
    max_chars: int | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.x0 < self.x1 <= 1.0):
            raise ConfigError(f"Field {self.name!r} x-range [{self.x0}, {self.x1}] is invalid")
        if not (0.0 <= self.y0 < self.y1 <= 1.0):
            raise ConfigError(f"Field {self.name!r} y-range [{self.y0}, {self.y1}] is invalid")

    @property
    def roi(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    def to_pixels(self, size: tuple[int, int]) -> tuple[float, float, float, float]:
        width, height = size
        return (self.x0 * width, self.y0 * height, self.x1 * width, self.y1 * height)

    def to_dict(self) -> dict:
        out = {"roi": [round(v, 5) for v in self.roi]}
        if self.group:
            out["group"] = self.group
        if self.max_chars:
            out["max_chars"] = self.max_chars
        return out


@dataclass
class CardLayout:
    """The field map for one card type."""

    card_type: str
    fields: dict[str, LayoutField] = field(default_factory=dict)
    #: Expected card aspect ratio (height / width); 88/63 for trading cards.
    aspect: float = 88.0 / 63.0
    #: Fields that must be present on every card of this type.
    required: tuple[str, ...] = ("name",)
    source: str = "builtin"

    def field(self, name: str) -> LayoutField:
        try:
            return self.fields[name]
        except KeyError as exc:
            raise ConfigError(
                f"Layout {self.card_type} has no field {name!r}; "
                f"available: {sorted(self.fields)}"
            ) from exc

    def has(self, name: str) -> bool:
        return name in self.fields

    def rois_in_pixels(self, size: tuple[int, int],
                       names: Iterable[str] | None = None
                       ) -> dict[str, tuple[float, float, float, float]]:
        """ROIs in pixels for the given card size (all fields when ``names`` is None)."""
        selected = list(names) if names is not None else list(self.fields)
        out = {}
        for name in selected:
            if name in self.fields:
                out[name] = self.fields[name].to_pixels(size)
        return out

    def to_dict(self) -> dict:
        return {
            "card_type": self.card_type,
            "aspect": round(self.aspect, 5),
            "required": list(self.required),
            "source": self.source,
            "fields": {name: f.to_dict() for name, f in self.fields.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping, *, source: str = "config") -> "CardLayout":
        fields = {}
        for name, spec in (data.get("fields") or {}).items():
            if isinstance(spec, Mapping):
                roi = spec.get("roi", spec.get("box"))
                if roi is None or len(roi) != 4:
                    raise ConfigError(f"Field {name!r} in layout needs a 4-value 'roi'")
                fields[name] = LayoutField(
                    name, float(roi[0]), float(roi[1]), float(roi[2]), float(roi[3]),
                    group=spec.get("group"), max_chars=spec.get("max_chars"),
                )
            else:  # bare [x0, y0, x1, y1]
                fields[name] = LayoutField(name, *(float(v) for v in spec))
        if not fields:
            raise ConfigError(f"Layout {data.get('card_type')!r} defines no fields")
        return cls(
            card_type=str(data.get("card_type", "UNKNOWN")).upper(),
            fields=fields,
            aspect=float(data.get("aspect", 88.0 / 63.0)),
            required=tuple(data.get("required", ["name"])),
            source=source,
        )


def _f(name: str, x0: float, y0: float, x1: float, y1: float, **kw) -> LayoutField:
    return LayoutField(name, x0, y0, x1, y1, **kw)


#: Built-in layouts. The coordinates follow the common template of each game;
#: they are approximate and intended to be refined against real annotations.
BUILTIN_LAYOUTS: dict[str, CardLayout] = {
    "POKEMON": CardLayout(
        card_type="POKEMON",
        fields={
            "name": _f("name", 0.045, 0.045, 0.62, 0.095, group="title", max_chars=30),
            "number": _f("number", 0.72, 0.905, 0.955, 0.945, group="corner", max_chars=12),
            "set": _f("set", 0.10, 0.875, 0.45, 0.905, group="corner", max_chars=24),
        },
        required=("name", "number"),
    ),
    "MAGIC_THE_GATHERING": CardLayout(
        card_type="MAGIC_THE_GATHERING",
        fields={
            "name": _f("name", 0.07, 0.045, 0.72, 0.09, group="title", max_chars=32),
            "mana_cost": _f("mana_cost", 0.74, 0.045, 0.94, 0.09, group="corner"),
            "type_line": _f("type_line", 0.07, 0.545, 0.93, 0.58, group="type", max_chars=48),
            "number": _f("number", 0.72, 0.925, 0.94, 0.955, group="corner", max_chars=10),
            "set": _f("set", 0.42, 0.925, 0.70, 0.955, group="corner", max_chars=10),
        },
        required=("name", "number"),
    ),
    "YU_GI_OH": CardLayout(
        card_type="YU_GI_OH",
        fields={
            "name": _f("name", 0.06, 0.035, 0.72, 0.085, group="title", max_chars=32),
            "number": _f("number", 0.06, 0.925, 0.40, 0.965, group="corner", max_chars=12),
            "set": _f("set", 0.42, 0.925, 0.72, 0.965, group="corner", max_chars=10),
        },
        required=("name", "number"),
    ),
    "SPORTS": CardLayout(
        card_type="SPORTS",
        fields={
            "player": _f("player", 0.05, 0.03, 0.62, 0.09, group="title", max_chars=30),
            "team": _f("team", 0.05, 0.90, 0.55, 0.94, group="footer", max_chars=28),
            "number": _f("number", 0.72, 0.90, 0.95, 0.95, group="corner", max_chars=10),
            "year": _f("year", 0.05, 0.955, 0.30, 0.99, group="footer", max_chars=6),
        },
        required=("player", "number"),
    ),
}


@lru_cache(maxsize=1)
def load_layouts() -> dict[str, CardLayout]:
    """Load built-in layouts, overlaid with any ``configs/cards/*.yaml``."""
    layouts: dict[str, CardLayout] = dict(BUILTIN_LAYOUTS)
    cards_dir = CONFIG_DIR / "cards"
    if cards_dir.exists():
        for path in sorted(cards_dir.glob("*.yaml")) + sorted(cards_dir.glob("*.yml")):
            try:
                data = load_yaml(path)
            except ConfigError as exc:  # pragma: no cover
                _LOG.warning("Skipping layout %s: %s", path.name, exc)
                continue
            if not data or "fields" not in data:
                continue
            layout = CardLayout.from_dict(data, source=str(path))
            layouts[layout.card_type] = layout
            _LOG.debug("Loaded layout %s from %s", layout.card_type, path)
    return layouts


def resolve_layout(card_type: str, layout_name: str | None = None) -> CardLayout:
    """Resolve a layout for a card type, with aliases and a sports fallback."""
    layouts = load_layouts()
    key = (layout_name or card_type or "").upper().strip()

    if key in layouts:
        return layouts[key]

    # Card-type aliases.
    from ..datasets.schema import normalize_card_type

    try:
        canonical = normalize_card_type(key)
    except ValueError:
        canonical = None
    if canonical and canonical in layouts:
        return layouts[canonical]

    if canonical in {"BASEBALL", "FOOTBALL", "BASKETBALL", "HOCKEY"}:
        return layouts["SPORTS"]
    if canonical in {"OTHER", "UNKNOWN"} or key in {"OTHER", "UNKNOWN"}:
        # Unknown games still have a name and a number; a generic layout keeps the
        # pipeline working instead of failing closed.
        return CardLayout(
            card_type="GENERIC",
            fields={
                "name": _f("name", 0.05, 0.04, 0.70, 0.10, group="title", max_chars=32),
                "number": _f("number", 0.70, 0.90, 0.95, 0.96, group="corner", max_chars=12),
            },
            source="fallback",
        )
    raise ConfigError(
        f"No layout for card type {card_type!r}; available: {sorted(layouts)}"
    )


def available_layouts() -> list[str]:
    return sorted(load_layouts())


def validate_layouts() -> list[str]:
    """Validate every loaded layout; returns a list of problems (empty = ok)."""
    problems = []
    for name, layout in load_layouts().items():
        for required in layout.required:
            if not layout.has(required):
                problems.append(f"{name}: required field {required!r} has no ROI")
        for field_name, layout_field in layout.fields.items():
            if required_overlap(layout_field, [f for n, f in layout.fields.items()
                                               if n != field_name]):
                problems.append(f"{name}: field {field_name!r} overlaps another field")
    return problems


def required_overlap(target: LayoutField, others: list[LayoutField],
                     threshold: float = 0.5) -> bool:
    """Whether ``target`` overlaps any other field by more than ``threshold``."""
    area = (target.x1 - target.x0) * (target.y1 - target.y0)
    for other in others:
        ix = max(0.0, min(target.x1, other.x1) - max(target.x0, other.x0))
        iy = max(0.0, min(target.y1, other.y1) - max(target.y0, other.y0))
        if area > 0 and (ix * iy) / area > threshold:
            return True
    return False
