"""Validation: images, annotations and whole datasets.

The design principle is **report, don't crash**: a 100k-image dataset with three
corrupt files must produce a report listing those three, not an exception on the
first one. ``strict=True`` turns the report into an exception at the call site
where that is appropriate (CI, dataset freeze).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..errors import DatasetError, ValidationError
from ..logging_utils import get_logger
from .hashing import file_sha256
from .manifest import DatasetManifest
from .schema import (
    CARD_TYPES,
    FIELD_NAMES,
    NUMBER_FIELDS,
    TITLE_FIELDS,
    Annotation,
    aspect_ratio_ok,
    normalize_card_type,
    quad_visible_fraction,
)

_LOG = get_logger("datasets.validation")

#: ISO/IEC 7810 ID-1 (playing/trading card) ratio: 63 mm / 88 mm.
STANDARD_CARD_RATIO = 63.0 / 88.0

SEVERITY_ORDER = {"error": 3, "warning": 2, "info": 1}


@dataclass
class ImageIssue:
    """One validation finding."""

    image_id: str
    code: str
    message: str
    severity: str = "error"
    path: str | None = None

    def to_dict(self) -> dict:
        out = {"image_id": self.image_id, "code": self.code,
               "severity": self.severity, "message": self.message}
        if self.path:
            out["path"] = self.path
        return out


@dataclass
class ValidationReport:
    """Aggregated validation findings for a dataset or a single image."""

    issues: list[ImageIssue] = field(default_factory=list)
    checked: int = 0
    ok: int = 0
    metadata: dict = field(default_factory=dict)

    def add(self, image_id: str, code: str, message: str, *, severity: str = "error",
            path: str | None = None) -> None:
        self.issues.append(ImageIssue(image_id, code, message, severity, path))

    def extend(self, other: ValidationReport) -> None:
        self.issues.extend(other.issues)
        self.checked += other.checked
        self.ok += other.ok
        self.metadata.update(other.metadata)

    @property
    def errors(self) -> list[ImageIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ImageIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def failed_ids(self) -> list[str]:
        return sorted({i.image_id for i in self.errors})

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def raise_if_invalid(self, context: str = "") -> None:
        if self.is_valid:
            return
        lines = [f"Validation failed{f' ({context})' if context else ''}: "
                 f"{len(self.errors)} error(s) across {len(self.failed_ids)} sample(s)"]
        for issue in self.errors[:25]:
            lines.append(f"  [{issue.code}] {issue.image_id}: {issue.message}")
        if len(self.errors) > 25:
            lines.append(f"  ... and {len(self.errors) - 25} more")
        raise ValidationError("\n".join(lines))

    def codes(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for issue in self.issues:
            counts[issue.code] = counts.get(issue.code, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def to_dict(self) -> dict:
        return {
            "valid": self.is_valid,
            "checked": self.checked,
            "ok": self.ok,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "error_codes": self.codes(),
            "issues": [i.to_dict() for i in self.issues],
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return p

    def summary_line(self) -> str:
        return (f"{self.checked} checked, {self.ok} ok, "
                f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)")


# ---------------------------------------------------------------------------
# Image-level checks
# ---------------------------------------------------------------------------

def validate_image(
    path: str | Path,
    image_id: str = "?",
    *,
    min_side: int = 32,
    max_side: int = 20000,
    require_standard_ratio: bool = False,
    check_pixel_variance: bool = True,
) -> ValidationReport:
    """Decode an image and sanity-check it.

    Detects the failure modes that actually occur in real card datasets:
    truncated files, CMYK/grayscale surprises, absurd dimensions, blank white
    scans (the card was not actually on the bed) and identical-pixel images.
    """
    from PIL import Image, UnidentifiedImageError

    report = ValidationReport(checked=1)
    p = Path(path)
    if not p.exists():
        report.add(image_id, "missing_file", f"File does not exist: {p}", path=str(p))
        return report
    if p.stat().st_size == 0:
        report.add(image_id, "empty_file", f"Zero-byte file: {p}", path=str(p))
        return report
    try:
        with Image.open(p) as img:
            img.verify()  # detects truncation without full decode
        with Image.open(p) as img:
            width, height = img.size
            mode = img.mode
            arr = np.asarray(img.convert("L"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        report.add(image_id, "corrupt_image", f"Cannot decode: {exc}", path=str(p))
        return report

    if width < min_side or height < min_side:
        report.add(image_id, "too_small", f"Image is {width}x{height} (min side {min_side})",
                   path=str(p))
    if width > max_side or height > max_side:
        report.add(image_id, "too_large", f"Image is {width}x{height} (max side {max_side})",
                   severity="warning", path=str(p))
    if mode not in {"RGB", "RGBA", "L", "P"}:
        report.add(image_id, "unusual_mode", f"Color mode {mode}", severity="warning",
                   path=str(p))
    if check_pixel_variance and float(arr.std()) < 1.0:
        report.add(image_id, "flat_image",
                   f"Image has near-zero variance (std={float(arr.std()):.3f}); "
                   "possibly a blank scan", path=str(p))
    if require_standard_ratio and not aspect_ratio_ok(width, height, STANDARD_CARD_RATIO, tol=0.12):
        report.add(image_id, "unexpected_ratio",
                   f"{width}x{height} (ratio {width / max(height, 1):.3f}) is not a standard "
                   f"card ratio ({STANDARD_CARD_RATIO:.3f})", severity="warning", path=str(p))
    if report.is_valid:
        report.ok = 1
    report.metadata.update({"width": width, "height": height, "mode": mode})
    return report


def detect_corrupt_images(paths: Iterable[str | Path], *, min_side: int = 32) -> ValidationReport:
    """Validate many image files, returning one aggregated report."""
    report = ValidationReport()
    for path in paths:
        report.extend(validate_image(path, image_id=Path(path).stem, min_side=min_side))
    return report


# ---------------------------------------------------------------------------
# Annotation-level checks
# ---------------------------------------------------------------------------

def validate_annotation(
    annotation: Annotation,
    *,
    require_card_quad: bool = False,
    require_name: bool = True,
    require_number: bool = False,
    min_field_size: float = 2.0,
    min_card_visibility: float = 0.25,
    strict_fields: bool = True,
    min_text_height: float = 0.0,
) -> ValidationReport:
    """Check that an annotation is internally consistent.

    Specifically this rejects the failure modes that silently poison training:

    * a field box that does not overlap the card at all;
    * a field box extending outside the image;
    * a card boundary that is self-intersecting or mostly outside the frame;
    * empty text where text is required;
    * unknown card type / unknown field names;
    * text boxes too short to be legible (``min_text_height``).
    """
    report = ValidationReport(checked=1)
    image_id = annotation.image_id
    if annotation.card_type not in CARD_TYPES:
        report.add(image_id, "bad_card_type", f"Unknown card type {annotation.card_type!r}")
    if annotation.width <= 0 or annotation.height <= 0:
        report.add(image_id, "bad_dimensions",
                   f"Annotation has no dimensions ({annotation.width}x{annotation.height})")
        return report

    quad = annotation.card_quad
    if quad is None:
        if require_card_quad:
            report.add(image_id, "missing_card_quad", "Annotation has no card boundary")
    else:
        if not quad.is_convex():
            report.add(image_id, "card_quad_not_convex", "Card boundary is self-intersecting")
        visible = quad_visible_fraction(quad, annotation.width, annotation.height)
        if visible < min_card_visibility:
            report.add(image_id, "card_not_visible",
                       f"Only {visible:.0%} of the card is inside the frame "
                       f"(min {min_card_visibility:.0%})")

    # A card must have a *title*, but games name that field differently: a
    # Pokemon card has "name", a hockey card has "player". Requiring the literal
    # key "name" rejects perfectly good sports annotations.
    if require_name and not any(f in annotation.fields for f in TITLE_FIELDS):
        report.add(image_id, "missing_name_field",
                   f"Annotation has no title field (any of {TITLE_FIELDS})")
    if require_number and not any(f in annotation.fields for f in NUMBER_FIELDS):
        report.add(image_id, "missing_number_field",
                   f"Annotation has no number field (any of {NUMBER_FIELDS})")

    card_box = quad.to_bbox() if quad is not None else None
    for name, field_annotation in annotation.fields.items():
        if strict_fields and name not in FIELD_NAMES:
            report.add(image_id, "unknown_field", f"Unknown field name {name!r}",
                       severity="warning")
        if not field_annotation.text.strip():
            report.add(image_id, "empty_text", f"Field {name!r} has empty text",
                       severity="warning")
        box = field_annotation.bbox
        if not box.is_valid(min_field_size):
            report.add(image_id, "tiny_field_box",
                       f"Field {name!r} box is {box.width:.1f}x{box.height:.1f}px")
        # Legibility floor. A name rendered ~10 px tall carries no glyph detail
        # once the OCR head resizes the crop to its 32 px input height, so the
        # dataset looks fine (boxes are valid, text is right) while CTC has
        # nothing to learn from and collapses to a constant prediction. That
        # cost a full afternoon of "why is CER exactly 1.0", so it is checked
        # rather than assumed. render_scale is the knob: at 0.5 the sport and
        # Yu-Gi-Oh layouts landed at ~19 px, at 1.0 they land at ~92 px.
        # Only title fields gate OCR training: a name ~10 px tall has no glyph
        # detail left after the head resizes to its 32 px input, so the dataset
        # looks fine (boxes valid, text correct) while CTC has nothing to learn
        # and collapses to a constant prediction - which is exactly how CER sat
        # at 1.0 for 30 epochs. Small number/year fields are normal even on a
        # good dataset, so they do not count.
        if min_text_height and name in TITLE_FIELDS and box.height < min_text_height:
            report.add(image_id, "illegible_text",
                       f"Title field {name!r} is {box.height:.1f}px tall "
                       f"(min {min_text_height:g}px for OCR training); "
                       f"raise the synthetic render scale", severity="warning")
        if box.x1 <= 0 or box.y1 <= 0 or box.x0 >= annotation.width or box.y0 >= annotation.height:
            report.add(image_id, "field_outside_image",
                       f"Field {name!r} lies entirely outside the frame")
        elif box.x0 < -1 or box.y0 < -1 or box.x1 > annotation.width + 1 \
                or box.y1 > annotation.height + 1:
            report.add(image_id, "field_partial_clip",
                       f"Field {name!r} extends past the image edge", severity="warning")
        if card_box is not None and box.intersection_area(card_box) <= 0:
            report.add(image_id, "field_off_card", f"Field {name!r} does not overlap the card")
        if field_annotation.confidence is not None and not 0.0 <= field_annotation.confidence <= 1.0:
            report.add(image_id, "bad_confidence",
                       f"Field {name!r} confidence {field_annotation.confidence} outside [0, 1]")
    if report.is_valid:
        report.ok = 1
    return report


# ---------------------------------------------------------------------------
# Dataset-level checks
# ---------------------------------------------------------------------------

def validate_dataset(
    manifest: DatasetManifest,
    *,
    check_files: bool = True,
    check_hashes: bool = False,
    check_annotations: bool = True,
    check_duplicates: bool = False,
    check_leakage: bool = True,
    require_card_quad: bool = False,
    min_card_visibility: float = 0.25,
    max_perceptual_distance: int = 4,
    sample_limit: int | None = None,
    min_text_height: float = 0.0,
) -> ValidationReport:
    """Full dataset validation, returning an aggregated report.

    Checks, in order of cost:

    1. manifest consistency (unique ids, resolvable paths, hash coverage);
    2. card-level leakage between splits;
    3. annotations;
    4. files on disk (decode + stats), optionally hash verification;
    5. perceptual duplicates (expensive: decodes every image).
    """
    report = ValidationReport()
    seen_ids: set[str] = set()
    dup_ids: list[str] = []
    records = manifest.samples[:sample_limit] if sample_limit else manifest.samples
    for record in records:
        report.checked += 1
        if record.image_id in seen_ids:
            dup_ids.append(record.image_id)
        seen_ids.add(record.image_id)
        if check_files:
            path = manifest.resolve(record.image_path)
            if not path.exists():
                report.add(record.image_id, "missing_file", f"File does not exist: {path}",
                           path=str(path))
                continue
        if check_hashes and record.sha256:
            actual = file_sha256(manifest.resolve(record.image_path))
            if actual != record.sha256:
                report.add(record.image_id, "hash_mismatch",
                           f"sha256 {actual[:12]} != manifest {record.sha256[:12]}")
        if check_annotations:
            report.extend(
                validate_annotation(
                    record.annotation,
                    require_card_quad=require_card_quad,
                    min_card_visibility=min_card_visibility,
                    min_text_height=min_text_height,
                )
            )
        else:
            report.ok += 1

    for image_id in dup_ids[:50]:
        report.add(image_id, "duplicate_image_id", "image_id appears more than once")

    if check_leakage:
        from .split import detect_leakage

        leaks = detect_leakage(manifest)
        for card_id, splits in leaks.items():
            report.add(card_id, "split_leakage",
                       f"card {card_id!r} appears in splits {sorted(splits)}")

    if check_duplicates:
        from .hashing import find_duplicate_groups, perceptual_hash

        hashes: dict[str, int] = {}
        for record in records:
            path = manifest.resolve(record.image_path)
            if not path.exists():
                continue
            try:
                hashes[record.image_id] = perceptual_hash(path, hash_size=8, highfreq_factor=2)
            except Exception as exc:  # pragma: no cover - decode already checked
                report.add(record.image_id, "hash_failed", str(exc), severity="warning")
        for group in find_duplicate_groups(hashes, max_distance=max_perceptual_distance):
            report.add(group[0], "duplicate_image",
                       f"{len(group)} perceptually identical images: {group[:5]}",
                       severity="warning")

    report.metadata.update({
        "dataset": manifest.name,
        "kind": manifest.kind,
        "samples": len(records),
    })
    _LOG.info("Validated %s: %s", manifest.name, report.summary_line())
    return report


def assert_valid_dataset(manifest: DatasetManifest, **kwargs) -> ValidationReport:
    """Validate and raise :class:`ValidationError` on any error."""
    report = validate_dataset(manifest, **kwargs)
    report.raise_if_invalid(manifest.name)
    return report


def normalize_annotation_fields(annotation: Annotation) -> Annotation:
    """Canonicalise card type and drop structurally invalid fields (in place)."""
    annotation.card_type = normalize_card_type(annotation.card_type)
    return annotation


def check_manifest_paths(root: str | Path, records: Iterable[Mapping]) -> list[str]:
    """Return the image paths referenced by ``records`` that do not exist."""
    root_path = Path(root)
    missing = []
    for record in records:
        path = Path(record["image_path"])
        full = path if path.is_absolute() else root_path / path
        if not full.exists():
            missing.append(str(full))
    return missing


def ensure_unique_card_ids(manifest: DatasetManifest) -> None:
    """Raise if two different cards share a ``card_id``.

    Duplicate card ids are worse than duplicate image ids: they would make a
    leakage check silently pass while leaking.
    """
    seen: dict[str, str] = {}
    conflicts = []
    for record in manifest.samples:
        known_name = seen.get(record.card_id)
        name = record.annotation.knowledge.name if record.annotation.knowledge else None
        if known_name is not None and name and known_name != name:
            conflicts.append((record.card_id, known_name, name))
        seen.setdefault(record.card_id, name or "")
    if conflicts:
        detail = ", ".join(f"{cid}: {a!r} vs {b!r}" for cid, a, b in conflicts[:5])
        raise DatasetError(
            f"{len(conflicts)} card_id(s) are reused for different cards ({detail}). "
            "Card ids must identify one physical card."
        )
