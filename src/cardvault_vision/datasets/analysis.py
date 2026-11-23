"""Dataset statistics and distribution analysis.

Two questions this module answers, both of which change modelling decisions:

* **Class balance** - if 90 % of the dataset is Pokémon, the card-type
  classifier will look excellent and behave badly on hockey cards.
* **Character distribution** - if ``Q`` and ``×`` and ``é`` never appear in
  training, they will never be recognised. This is the analysis that justifies
  the universal character set in Phase 4.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .manifest import DatasetManifest

_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str, *, unicode_form: str = "NFKC") -> str:
    """Canonicalise annotation text (full-width -> ASCII, collapse whitespace)."""
    text = unicodedata.normalize(unicode_form, text or "")
    return _WHITESPACE.sub(" ", text).strip()


@dataclass
class CharacterDistribution:
    """Frequency analysis of the characters used in a dataset's text fields."""

    counts: Counter = field(default_factory=Counter)
    total_characters: int = 0
    total_strings: int = 0
    per_field: dict[str, Counter] = field(default_factory=dict)
    max_length: int = 0

    @property
    def charset(self) -> str:
        """Sorted unique characters - the basis of the OCR charset."""
        return "".join(sorted(self.counts))

    def coverage(self, charset: str) -> float:
        """Fraction of this dataset's characters representable by ``charset``."""
        if self.total_characters == 0:
            return 1.0
        covered = sum(n for ch, n in self.counts.items() if ch in charset)
        return covered / self.total_characters

    def missing(self, charset: str) -> str:
        return "".join(sorted(set(self.counts) - set(charset)))

    def length_histogram(self) -> dict[str, int]:
        """Per-field length percentiles compacted into a readable summary."""
        return {name: len(counter) for name, counter in sorted(self.per_field.items())}

    def to_dict(self) -> dict:
        return {
            "total_characters": self.total_characters,
            "total_strings": self.total_strings,
            "unique_characters": len(self.counts),
            "charset": self.charset,
            "max_length": self.max_length,
            "top_characters": self.counts.most_common(50),
            "per_field": {name: dict(counter.most_common()) for name, counter in
                          self.per_field.items()},
        }


def character_distribution(
    manifest: DatasetManifest,
    *,
    fields: tuple[str, ...] = ("name", "number"),
    normalize: bool = True,
) -> CharacterDistribution:
    """Measure character usage across one or more annotated fields."""
    dist = CharacterDistribution()
    for record in manifest.samples:
        for name in fields:
            annotation = record.annotation.fields.get(name)
            if not annotation or not annotation.text:
                continue
            text = normalize_text(annotation.text) if normalize else annotation.text
            counter = dist.per_field.setdefault(name, Counter())
            counter.update(text)
            dist.counts.update(text)
            dist.total_characters += len(text)
            dist.total_strings += 1
            dist.max_length = max(dist.max_length, len(text))
    return dist


@dataclass
class DatasetStatistics:
    """A comprehensive, JSON-serialisable description of a dataset."""

    name: str
    kind: str
    samples: int
    cards: int
    card_types: dict[str, int] = field(default_factory=dict)
    finishes: dict[str, int] = field(default_factory=dict)
    profiles: dict[str, int] = field(default_factory=dict)
    splits: dict[str, int] = field(default_factory=dict)
    resolutions: dict[str, int] = field(default_factory=dict)
    field_presence: dict[str, int] = field(default_factory=dict)
    characters: CharacterDistribution | None = None
    numeric: dict[str, float] = field(default_factory=dict)
    problems: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        out = {
            "name": self.name,
            "kind": self.kind,
            "samples": self.samples,
            "cards": self.cards,
            "card_types": self.card_types,
            "finishes": self.finishes,
            "profiles": self.profiles,
            "splits": self.splits,
            "resolutions": self.resolutions,
            "field_presence": dict(sorted(self.field_presence.items())),
            "numeric": self.numeric,
            "problems": self.problems,
            "warnings": self.warnings,
        }
        if self.characters is not None:
            out["characters"] = self.characters.to_dict()
        return out

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, default=str) + "\n", encoding="utf-8")
        return p

    def report(self) -> str:
        """Human-readable summary for the CLI."""
        lines = [
            f"Dataset      : {self.name} ({self.kind})",
            f"Samples      : {self.samples}",
            f"Cards        : {self.cards}",
            f"Card types   : {_fmt_counts(self.card_types)}",
            f"Finishes     : {_fmt_counts(self.finishes)}",
            f"Splits       : {_fmt_counts(self.splits)}",
            f"Profiles     : {_fmt_counts(self.profiles)}",
            f"Resolution   : {_fmt_counts(self.resolutions)}",
            f"Fields       : {_fmt_counts(self.field_presence)}",
        ]
        if self.characters is not None:
            lines.append(
                f"Characters   : {len(self.characters.counts)} unique, "
                f"{self.characters.total_characters} total, "
                f"max length {self.characters.max_length}"
            )
        if self.numeric:
            lines.append("Numeric      : " + ", ".join(f"{k}={v:.3f}" for k, v in self.numeric.items()))
        if self.problems:
            lines.append(f"Problems     : {_fmt_counts(self.problems)}")
        for warning in self.warnings:
            lines.append(f"WARNING      : {warning}")
        return "\n".join(lines)


def _fmt_counts(counts: dict[str, int], limit: int = 6) -> str:
    if not counts:
        return "-"
    items = sorted(counts.items(), key=lambda kv: -kv[1])[:limit]
    text = ", ".join(f"{k}={v}" for k, v in items)
    if len(counts) > limit:
        text += f", +{len(counts) - limit} more"
    return text


def _bucket_resolution(width: int, height: int) -> str:
    """Coarse resolution bucket (keeps the histogram readable)."""
    longest = max(width, height)
    for limit in (320, 640, 1024, 1600, 2048, 3200, 4800):
        if longest <= limit:
            return f"<= {limit}px"
    return "> 4800px"


def class_distribution(manifest: DatasetManifest, *, by: str = "card_type",
                       split: str | None = None) -> dict[str, int]:
    """Count samples per class (``card_type``, ``finish``, ``profile``, ``split``)."""
    counts: Counter = Counter()
    for record in manifest.samples:
        if split is not None and record.split != split:
            continue
        annotation = record.annotation
        if by == "card_type":
            counts[annotation.card_type] += 1
        elif by == "finish":
            finish = annotation.knowledge.finish if annotation.knowledge else None
            counts[finish or "UNKNOWN"] += 1
        elif by == "profile":
            counts[record.profile or "UNKNOWN"] += 1
        elif by == "split":
            counts[record.split or "unassigned"] += 1
        elif by == "card":
            counts[record.card_id] += 1
        elif by == "set":
            set_name = annotation.knowledge.set_name if annotation.knowledge else None
            counts[set_name or "UNKNOWN"] += 1
        else:
            raise ValueError(f"Unknown distribution key {by!r}")
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def dataset_statistics(
    manifest: DatasetManifest,
    *,
    include_characters: bool = True,
    text_fields: tuple[str, ...] = ("name", "number"),
    imbalance_threshold: float = 0.15,
) -> DatasetStatistics:
    """Compute the standard dataset report.

    ``imbalance_threshold`` triggers a warning when the smallest populated class
    holds less than that share of samples - a common cause of quietly
    misleading validation results.
    """
    stats = DatasetStatistics(name=manifest.name, kind=manifest.kind,
                              samples=len(manifest.samples), cards=len(manifest.card_ids))
    stats.card_types = class_distribution(manifest, by="card_type")
    stats.finishes = class_distribution(manifest, by="finish")
    stats.profiles = class_distribution(manifest, by="profile")
    stats.splits = class_distribution(manifest, by="split")

    resolutions: Counter = Counter()
    areas: list[float] = []
    card_areas: list[float] = []
    field_counts: Counter = Counter()
    problems: Counter = Counter()
    text_lengths: list[int] = []

    for record in manifest.samples:
        annotation = record.annotation
        width = record.width or annotation.width
        height = record.height or annotation.height
        if width and height:
            resolutions[_bucket_resolution(width, height)] += 1
            areas.append(float(width) * float(height))
        if annotation.card_quad is not None and width and height:
            card_areas.append(annotation.card_quad.area() / (width * height))
        elif annotation.card_quad is None:
            problems["missing_card_quad"] += 1
        for name, field_annotation in annotation.fields.items():
            field_counts[name] += 1
            text_lengths.append(len(field_annotation.text))
            if not field_annotation.bbox.is_valid():
                problems["degenerate_field_box"] += 1
        if annotation.card_type == "UNKNOWN":
            problems["unknown_card_type"] += 1
        if not record.sha256:
            problems["missing_hash"] += 1

    stats.resolutions = dict(resolutions)
    stats.field_presence = dict(field_counts)
    stats.problems = dict(problems)
    if areas:
        stats.numeric["mean_pixels"] = float(np.mean(areas))
        stats.numeric["median_pixels"] = float(np.median(areas))
    if card_areas:
        stats.numeric["mean_card_area_fraction"] = float(np.mean(card_areas))
        stats.numeric["min_card_area_fraction"] = float(np.min(card_areas))
        low_visibility = sum(1 for value in card_areas if value < 0.5)
        if low_visibility:
            stats.warnings.append(
                f"{low_visibility} sample(s) have less than half the frame covered by card"
            )
    if text_lengths:
        stats.numeric["mean_field_text_length"] = float(np.mean(text_lengths))
        stats.numeric["max_field_text_length"] = float(np.max(text_lengths))

    if include_characters:
        stats.characters = character_distribution(manifest, fields=text_fields)

    # Class-imbalance warnings.
    total = sum(stats.card_types.values())
    if total and len(stats.card_types) > 1:
        smallest = min(stats.card_types.values())
        if smallest / total < imbalance_threshold:
            minority = min(stats.card_types, key=stats.card_types.get)
            stats.warnings.append(
                f"Class imbalance: {minority} is only {smallest / total:.1%} of samples "
                f"(threshold {imbalance_threshold:.0%})"
            )
    if manifest.samples and len(manifest.card_ids) < len(manifest.samples):
        duplicates = len(manifest.samples) - len(manifest.card_ids)
        stats.warnings.append(
            f"{duplicates} samples share a card with another sample "
            "(fine for training, but check split leakage)"
        )
    return stats
