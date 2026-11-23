"""Automatic ground truth (Phase 2.10).

The engine's central promise: *every generated image ships with exact labels*.
Exact means derived from the same transform that produced the pixels, not
estimated from the output image.

This module owns the annotation *contract*:

* what a generation record contains (and its schema version),
* how to re-derive a sample from its own record (``regenerate_from_annotation``),
* how to audit a dataset's annotations for consistency (silent label corruption
  is the failure mode that destroys a model, so it is checked explicitly).

The annotation payload written by the pipeline is
``Annotation.to_dict()`` + ``quality`` + ``provenance``; see
:mod:`cardvault_vision.datasets.schema`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..datasets.schema import Annotation, BBox
from ..errors import ValidationError
from ..logging_utils import get_logger
from ..version import SCHEMA_VERSION

_LOG = get_logger("synthetic.annotations")


@dataclass
class AnnotationAudit:
    """Result of auditing a directory of annotation sidecars."""

    checked: int = 0
    problems: list[tuple[str, str]] = field(default_factory=list)  # (image_id, message)
    schema_versions: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems

    def add(self, image_id: str, message: str) -> None:
        self.problems.append((image_id, message))

    def raise_if_invalid(self) -> None:
        if self.ok:
            return
        lines = [f"Annotation audit failed with {len(self.problems)} problem(s):"]
        lines += [f"  {image_id}: {message}" for image_id, message in self.problems[:20]]
        raise ValidationError("\n".join(lines))

    def to_dict(self) -> dict:
        return {"checked": self.checked, "ok": self.ok,
                "schema_versions": self.schema_versions,
                "problems": [{"image_id": i, "message": m} for i, m in self.problems]}


def load_annotation(path: str | Path) -> Annotation:
    """Load a sidecar JSON into an :class:`Annotation`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return Annotation.from_dict(payload)


def audit_annotations(directory: str | Path, *,
                      image_dir: str | Path | None = None,
                      min_fields: int = 1) -> AnnotationAudit:
    """Check every sidecar in ``directory`` for internal consistency.

    Catches: unparseable JSON, unsupported schema versions, missing geometry,
    boxes outside the image, and (when ``image_dir`` is given) images that are
    missing or whose dimensions disagree with the annotation.
    """
    from PIL import Image

    audit = AnnotationAudit()
    root = Path(directory)
    if not root.exists():
        raise ValidationError(f"Annotation directory not found: {root}")
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            audit.add(path.stem, f"invalid JSON: {exc}")
            continue
        if "fields" not in payload and "samples" in payload:
            continue  # a manifest, not a sidecar
        audit.checked += 1
        schema = str(payload.get("schema_version", "0"))
        audit.schema_versions[schema] = audit.schema_versions.get(schema, 0) + 1
        if schema.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
            audit.add(path.stem, f"schema {schema} is not {SCHEMA_VERSION}")
        image_id = str(payload.get("image_id", path.stem))
        width, height = int(payload.get("width", 0)), int(payload.get("height", 0))
        if width <= 0 or height <= 0:
            audit.add(image_id, "annotation has no image dimensions")
            continue
        if "card_quad" not in payload:
            audit.add(image_id, "annotation has no card_quad")
        fields = payload.get("fields") or {}
        if len(fields) < min_fields:
            audit.add(image_id, f"annotation has {len(fields)} field(s) (min {min_fields})")
        for name, item in fields.items():
            try:
                box = BBox.from_dict(item["bbox"])
            except (KeyError, ValueError) as exc:
                audit.add(image_id, f"field {name!r} has an invalid bbox: {exc}")
                continue
            if box.x1 <= 0 or box.y1 <= 0 or box.x0 >= width or box.y0 >= height:
                audit.add(image_id, f"field {name!r} lies outside the image")
            if not item.get("text"):
                audit.add(image_id, f"field {name!r} has empty text")
        if image_dir is not None:
            candidates = list(Path(image_dir).glob(f"{image_id}.*"))
            if not candidates:
                audit.add(image_id, "no matching image file")
            else:
                try:
                    with Image.open(candidates[0]) as img:
                        if img.size != (width, height):
                            audit.add(
                                image_id,
                                f"image is {img.size[0]}x{img.size[1]} but annotation says "
                                f"{width}x{height}",
                            )
                except OSError as exc:  # pragma: no cover
                    audit.add(image_id, f"unreadable image: {exc}")
    return audit


def regenerate_from_annotation(annotation_payload: Mapping[str, Any],
                               source_card: Any,
                               ) -> Any:
    """Re-render a sample from its own recorded generation parameters.

    This is the strongest form of reproducibility the engine offers: given the
    original source card and the annotation that was shipped with an image, the
    exact same capture can be regenerated. It is what makes it possible to
    investigate a model failure years later.
    """
    from .generator import GenerationConfig, generate_sample

    generation = annotation_payload.get("generation") or {}
    if not generation:
        raise ValidationError("Annotation has no generation record; cannot regenerate")
    seed = int(generation["seed"])
    index = int(generation["index"])
    config = GenerationConfig(
        profile=generation.get("resolved_profile") or {},
        mode=str(generation.get("mode", "normal")),
        problems=list(generation.get("problems") or []),
    )
    return generate_sample(source_card, config, seed=seed, index=index,
                           image_id=str(annotation_payload.get("image_id", "")))


def summarize_annotations(directory: str | Path) -> dict:
    """Aggregate the annotations in a directory into a compact report."""
    root = Path(directory)
    total_fields = 0
    field_counts: dict[str, int] = {}
    texts: dict[str, list[str]] = {}
    problems: dict[str, int] = {}
    count = 0
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:  # pragma: no cover
            continue
        if "fields" not in payload:
            continue
        count += 1
        for name, item in (payload.get("fields") or {}).items():
            total_fields += 1
            field_counts[name] = field_counts.get(name, 0) + 1
            texts.setdefault(name, []).append(str(item.get("text", "")))
        for problem in (payload.get("generation") or {}).get("problems") or []:
            problems[problem] = problems.get(problem, 0) + 1
    return {
        "annotations": count,
        "fields": total_fields,
        "field_counts": dict(sorted(field_counts.items())),
        "problems": problems,
        "samples": {name: values[:5] for name, values in texts.items()},
    }
