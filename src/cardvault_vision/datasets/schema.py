"""Data schema for datasets, annotations and card knowledge.

These dataclasses are the **contract** of the model factory. The synthetic scan
engine writes them, the dataset engine validates them, layouts consume them and
exported models ship their preprocessing rules in the same shape.

On-disk representation is plain JSON with an explicit ``schema_version`` so
old artifacts can be detected rather than silently mis-read.

Coordinate conventions
----------------------
* All coordinates are in **pixels of the image they annotate**, origin at the
  top-left, ``+x`` right, ``+y`` down.
* A :class:`BBox` is half-open: ``x0 <= x < x1``. Width is ``x1 - x0``.
* A card boundary is a :class:`Quad` (four corners) because a photographed card
  is a general quadrilateral: ``[top_left, top_right, bottom_right, bottom_left]``
  going clockwise. ``Quad.to_bbox()`` gives the axis-aligned envelope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Any, Iterable, Mapping

from ..version import SCHEMA_VERSION

#: Canonical card ecosystems. ``OTHER``/``UNKNOWN`` are part of the taxonomy so
#: the classifier can decline to commit rather than mislabel.
CARD_TYPES = (
    "POKEMON",
    "MAGIC_THE_GATHERING",
    "YU_GI_OH",
    # Sports cards share one physical layout, so "SPORTS" is a first-class type.
    # Collapsing it into OTHER (the earlier behaviour) threw away the player
    # field entirely and made the classifier's catch-all bucket look common.
    "SPORTS",
    "BASEBALL",
    "FOOTBALL",
    "BASKETBALL",
    "HOCKEY",
    "OTHER",
    "UNKNOWN",
)

CARD_TYPE_ALIASES = {
    "pokemon": "POKEMON",
    "pokémon": "POKEMON",
    "magic": "MAGIC_THE_GATHERING",
    "mtg": "MAGIC_THE_GATHERING",
    "magic_the_gathering": "MAGIC_THE_GATHERING",
    "magic the gathering": "MAGIC_THE_GATHERING",
    "yugioh": "YU_GI_OH",
    "yu_gi_oh": "YU_GI_OH",
    "yu-gi-oh": "YU_GI_OH",
    "yu-gi-oh!": "YU_GI_OH",
    "baseball": "BASEBALL",
    "football": "FOOTBALL",
    "basketball": "BASKETBALL",
    "hockey": "HOCKEY",
    # Generic sports spellings resolve to the shared sports layout rather than
    # the catch-all bucket.
    "sports": "SPORTS",
    "sports card": "SPORTS",
    "trading card": "OTHER",
    "other": "OTHER",
    "unknown": "UNKNOWN",
}

#: Standard field names understood by the layout resolver and both OCR heads.
FIELD_NAMES = ("name", "number", "set", "player", "team", "year", "rarity",
               "finish", "type_line", "mana_cost")

#: Fields that carry the card's title. Games name it differently - a Pokemon card
#: has a "name", a hockey card has a "player" - so "the card has a title" is a
#: property of this set, not of any single key.
TITLE_FIELDS = ("name", "player", "title")

#: Fields that carry the collector number / identifier.
NUMBER_FIELDS = ("number", "collector_number", "card_number")

#: Card finishes materially change how light behaves (Phase 2.5).
CARD_FINISHES = ("normal", "holo", "reverse_holo", "foil", "etched_foil",
                 "chrome", "refractor", "glossy", "matte")


def normalize_card_type(value: str) -> str:
    """Map free-form card-type spellings onto the canonical taxonomy."""
    key = (value or "").strip()
    if key.upper() in CARD_TYPES:
        return key.upper()
    alias = CARD_TYPE_ALIASES.get(key.lower())
    if alias:
        return alias
    raise ValueError(f"Unknown card type {value!r}; expected one of {CARD_TYPES}")


@dataclass(frozen=True)
class BBox:
    """Axis-aligned rectangle in pixels (half-open on the far edges)."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError(f"Degenerate bbox: {self!r}")

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2

    def clip(self, width: int, height: int) -> "BBox":
        """Clamp to the image extent; may collapse to a zero-area box."""
        return BBox(
            min(max(self.x0, 0.0), width),
            min(max(self.y0, 0.0), height),
            min(max(self.x1, 0.0), width),
            min(max(self.y1, 0.0), height),
        )

    def expand(self, px: float) -> "BBox":
        return BBox(self.x0 - px, self.y0 - px, self.x1 + px, self.y1 + px)

    def iou(self, other: "BBox") -> float:
        ix0, iy0 = max(self.x0, other.x0), max(self.y0, other.y0)
        ix1, iy1 = min(self.x1, other.x1), min(self.y1, other.y1)
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def intersection_area(self, other: "BBox") -> float:
        ix0, iy0 = max(self.x0, other.x0), max(self.y0, other.y0)
        ix1, iy1 = min(self.x1, other.x1), min(self.y1, other.y1)
        return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)

    def relative_to(self, parent: "BBox") -> "BBox":
        """Express this box in normalised coordinates of ``parent``."""
        if parent.width <= 0 or parent.height <= 0:
            raise ValueError("Cannot normalise against an empty parent box")
        return BBox(
            (self.x0 - parent.x0) / parent.width,
            (self.y0 - parent.y0) / parent.height,
            (self.x1 - parent.x0) / parent.width,
            (self.y1 - parent.y0) / parent.height,
        )

    def is_valid(self, min_size: float = 1.0) -> bool:
        return self.width >= min_size and self.height >= min_size

    def as_list(self) -> list[float]:
        return [round(float(self.x0), 2), round(float(self.y0), 2),
                round(float(self.x1), 2), round(float(self.y1), 2)]

    def to_dict(self) -> dict:
        return {"x0": round(float(self.x0), 2), "y0": round(float(self.y0), 2),
                "x1": round(float(self.x1), 2), "y1": round(float(self.y1), 2)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BBox":
        if isinstance(data, (list, tuple)) and len(data) == 4:
            return cls(*(float(v) for v in data))
        return cls(float(data["x0"]), float(data["y0"]), float(data["x1"]), float(data["y1"]))

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> "BBox":
        return cls(x, y, x + w, y + h)


@dataclass(frozen=True)
class Quad:
    """A four-corner card boundary: TL, TR, BR, BL (clockwise)."""

    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.points) != 4:
            raise ValueError(f"Quad needs exactly 4 points, got {len(self.points)}")

    @property
    def top_left(self) -> tuple[float, float]:
        return self.points[0]

    @property
    def top_right(self) -> tuple[float, float]:
        return self.points[1]

    @property
    def bottom_right(self) -> tuple[float, float]:
        return self.points[2]

    @property
    def bottom_left(self) -> tuple[float, float]:
        return self.points[3]

    def to_bbox(self) -> BBox:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return BBox(min(xs), min(ys), max(xs), max(ys))

    def area(self) -> float:
        """Shoelace area of the polygon."""
        total = 0.0
        pts = self.points
        for i in range(4):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % 4]
            total += x0 * y1 - x1 * y0
        return abs(total) / 2

    def is_convex(self) -> bool:
        signs = []
        pts = self.points
        for i in range(4):
            ax, ay = pts[i]
            bx, by = pts[(i + 1) % 4]
            cx, cy = pts[(i + 2) % 4]
            cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
            if abs(cross) > 1e-9:
                signs.append(cross > 0)
        return len(set(signs)) == 1

    def translate(self, dx: float, dy: float) -> "Quad":
        return Quad(tuple((x + dx, y + dy) for x, y in self.points))

    def clip_inside(self, width: int, height: int) -> bool:
        """True when every corner lies within the image."""
        return all(-0.5 <= x <= width + 0.5 and -0.5 <= y <= height + 0.5 for x, y in self.points)

    def as_list(self) -> list[list[float]]:
        return [[round(float(x), 2), round(float(y), 2)] for x, y in self.points]

    def to_dict(self) -> dict:
        return {"points": self.as_list()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | Iterable) -> "Quad":
        pts = data["points"] if isinstance(data, Mapping) else list(data)
        return cls(tuple((float(p[0]), float(p[1])) for p in pts))

    @classmethod
    def from_bbox(cls, bbox: BBox) -> "Quad":
        return cls(((bbox.x0, bbox.y0), (bbox.x1, bbox.y0),
                    (bbox.x1, bbox.y1), (bbox.x0, bbox.y1)))

    @classmethod
    def full_image(cls, width: int, height: int) -> "Quad":
        return cls(((0.0, 0.0), (float(width), 0.0), (float(width), float(height)),
                    (0.0, float(height))))


def _polygon_clip(poly: list[tuple[float, float]], bbox: BBox) -> list[tuple[float, float]]:
    """Sutherland-Hodgman clip of a convex polygon against an axis-aligned box."""
    def clip_edge(points, inside, intersect):
        out = []
        for i, current in enumerate(points):
            previous = points[i - 1]
            cur_in, prev_in = inside(current), inside(previous)
            if cur_in:
                if not prev_in:
                    out.append(intersect(previous, current))
                out.append(current)
            elif prev_in:
                out.append(intersect(previous, current))
        return out

    def lerp_x(p, q, x):
        t = (x - p[0]) / (q[0] - p[0])
        return (x, p[1] + t * (q[1] - p[1]))

    def lerp_y(p, q, y):
        t = (y - p[1]) / (q[1] - p[1])
        return (p[0] + t * (q[0] - p[0]), y)

    pts = list(poly)
    for inside, intersect in (
        (lambda p: p[0] >= bbox.x0, lambda p, q: lerp_x(p, q, bbox.x0)),
        (lambda p: p[0] <= bbox.x1, lambda p, q: lerp_x(p, q, bbox.x1)),
        (lambda p: p[1] >= bbox.y0, lambda p, q: lerp_y(p, q, bbox.y0)),
        (lambda p: p[1] <= bbox.y1, lambda p, q: lerp_y(p, q, bbox.y1)),
    ):
        if not pts:
            break
        pts = clip_edge(pts, inside, intersect)
    return pts


def quad_visible_fraction(quad: Quad, width: int, height: int) -> float:
    """Fraction of the card's own area that lies inside the frame.

    This is how the engine rejects unusable generations: a card that is 90 %
    outside the frame produces no usable training signal.
    """
    total = quad.area()
    if total <= 0:
        return 0.0
    clipped = _polygon_clip(list(quad.points), BBox(0.0, 0.0, float(width), float(height)))
    if len(clipped) < 3:
        return 0.0
    # Shoelace over the (possibly up to 8-sided) clipped polygon.
    area = 0.0
    for i in range(len(clipped)):
        x0, y0 = clipped[i]
        x1, y1 = clipped[(i + 1) % len(clipped)]
        area += x0 * y1 - x1 * y0
    visible = abs(area) / 2
    return max(0.0, min(1.0, visible / total))


@dataclass
class FieldAnnotation:
    """One recognised field: its text and where it sits.

    ``roi`` is the *pre-warp* card-space box in normalised card coordinates
    (0..1). It is layout information rather than detection evidence, and it is
    what the synthetic engine uses to place text before warping.
    """

    text: str
    bbox: BBox
    roi: BBox | None = None
    confidence: float | None = None
    language: str | None = None

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"text": self.text, "bbox": self.bbox.to_dict()}
        if self.roi is not None:
            out["roi"] = self.roi.to_dict()
        if self.confidence is not None:
            out["confidence"] = round(float(self.confidence), 4)
        if self.language:
            out["language"] = self.language
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FieldAnnotation":
        return cls(
            text=str(data.get("text", "")),
            bbox=BBox.from_dict(data["bbox"]),
            roi=BBox.from_dict(data["roi"]) if data.get("roi") else None,
            confidence=data.get("confidence"),
            language=data.get("language"),
        )


@dataclass
class CardKnowledge:
    """Facts about the card that are not read from the image.

    Used for candidate matching downstream and for stratified dataset splits;
    never used as an OCR input.
    """

    card_id: str | None = None
    card_type: str = "UNKNOWN"
    name: str | None = None
    set_name: str | None = None
    set_code: str | None = None
    number: str | None = None
    rarity: str | None = None
    year: int | None = None
    manufacturer: str | None = None
    finish: str | None = None
    language: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {k: getattr(self, k) for k in
               ("card_id", "card_type", "name", "set_name", "set_code", "number",
                "rarity", "year", "manufacturer", "finish", "language")}
        if self.extra:
            out["extra"] = self.extra
        return {k: v for k, v in out.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CardKnowledge":
        known = {f.name for f in dataclass_fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        if "card_type" in kwargs:
            kwargs["card_type"] = normalize_card_type(kwargs["card_type"])
        return cls(**kwargs)


@dataclass
class Annotation:
    """The full annotation for one image.

    Every synthetic sample carries its generation provenance in ``generation``,
    which is what makes targeted regeneration and error attribution possible
    (Phase 2.10).
    """

    image_id: str
    card_id: str
    card_type: str
    fields: dict[str, FieldAnnotation] = field(default_factory=dict)
    card_quad: Quad | None = None
    width: int = 0
    height: int = 0
    knowledge: CardKnowledge | None = None
    generation: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def field_text(self, name: str) -> str | None:
        item = self.fields.get(name)
        return item.text if item else None

    @property
    def is_synthetic(self) -> bool:
        return bool(self.generation)

    def to_dict(self) -> dict:
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "image_id": self.image_id,
            "card_id": self.card_id,
            "card_type": self.card_type,
            "width": self.width,
            "height": self.height,
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
        }
        if self.card_quad is not None:
            out["card_quad"] = self.card_quad.to_dict()
        if self.knowledge is not None:
            out["knowledge"] = self.knowledge.to_dict()
        if self.generation:
            out["generation"] = self.generation
        if self.extra:
            out["extra"] = self.extra
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Annotation":
        return cls(
            image_id=str(data["image_id"]),
            card_id=str(data["card_id"]),
            card_type=normalize_card_type(data.get("card_type", "UNKNOWN")),
            fields={k: FieldAnnotation.from_dict(v) for k, v in (data.get("fields") or {}).items()},
            card_quad=Quad.from_dict(data["card_quad"]) if data.get("card_quad") else None,
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            knowledge=CardKnowledge.from_dict(data["knowledge"]) if data.get("knowledge") else None,
            generation=dict(data.get("generation") or {}),
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            extra=dict(data.get("extra") or {}),
        )


@dataclass
class SampleRecord:
    """One image in a dataset, with its annotation and integrity hash."""

    image_id: str
    card_id: str
    image_path: str
    annotation: Annotation
    sha256: str | None = None
    width: int = 0
    height: int = 0
    split: str | None = None
    profile: str | None = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self, *, with_annotation: bool = True) -> dict:
        out: dict[str, Any] = {
            "image_id": self.image_id,
            "card_id": self.card_id,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
        }
        if self.sha256:
            out["sha256"] = self.sha256
        if self.split:
            out["split"] = self.split
        if self.profile:
            out["profile"] = self.profile
        if self.tags:
            out["tags"] = list(self.tags)
        if with_annotation:
            out["annotation"] = self.annotation.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SampleRecord":
        ann = data.get("annotation")
        annotation = Annotation.from_dict(ann) if ann else Annotation(
            image_id=data["image_id"],
            card_id=data["card_id"],
            card_type=data.get("card_type", "UNKNOWN"),
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
        )
        return cls(
            image_id=str(data["image_id"]),
            card_id=str(data["card_id"]),
            image_path=str(data["image_path"]),
            annotation=annotation,
            sha256=data.get("sha256"),
            width=int(data.get("width", annotation.width or 0)),
            height=int(data.get("height", annotation.height or 0)),
            split=data.get("split"),
            profile=data.get("profile"),
            tags=list(data.get("tags") or []),
        )


def aspect_ratio_ok(width: int, height: int, target: float, tol: float = 0.02) -> bool:
    """Whether an image's aspect ratio is close to the standard card ratio."""
    if height <= 0:
        return False
    return abs(width / height - target) <= tol * target


def round_sig(value: float, digits: int = 4) -> float:
    """Round a float to ``digits`` significant figures (stable JSON output)."""
    if value == 0 or not math.isfinite(value):
        return float(value)
    return round(value, digits - int(math.floor(math.log10(abs(value)))) - 1)
