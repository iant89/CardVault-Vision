"""Hard-example mining (Phase 9).

The loop the roadmap describes:

    REAL SCAN -> MODEL -> ERROR -> CLASSIFICATION -> HARD DATASET
              -> TARGETED SYNTHETIC -> RETRAIN -> EVALUATE

This module implements the middle two steps: given per-sample predictions, decide
*which* failures matter and *why* each one failed. The "why" is the important part
- "CER is 12 %" is not actionable, but "34 % of failures are glare on holo cards"
is a direct instruction to the synthetic engine: ``--problem glare``.

Error attribution uses the generation record carried by every synthetic sample
(and by the real-scan metadata when available). When no metadata exists, we fall
back to measuring the image itself: sharpness (blur), clipped fraction (exposure),
local contrast in the failing field (glare/occlusion), and character-level
confusions (font/language issues).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .metrics import edit_distance, evaluate_ocr

#: Error categories, aligned with the synthetic engine's targeted problems so that
#: mining output maps 1:1 onto generation flags.
ERROR_TYPES = (
    "perspective",
    "blur",
    "glare",
    "low_light",
    "tiny_text",
    "occlusion",
    "compression",
    "character_confusion",
    "layout_or_language",
    "unknown",
)

#: Which image measurements classify which error type.
_DEFAULTS = {
    "min_sharpness": 60.0,      # Laplacian variance (0-255 scale)
    "max_clipped": 0.12,
    "max_crushed": 0.18,
    "min_glare": 0.02,
    "min_card_area": 0.12,
}


@dataclass
class HardExample:
    """One failing sample with its attributed cause."""

    image_id: str
    prediction: str
    reference: str
    cer: float
    confidence: float | None = None
    primary_error: str = "unknown"
    error_types: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "prediction": self.prediction,
            "reference": self.reference,
            "cer": round(self.cer, 5),
            "confidence": self.confidence,
            "primary_error": self.primary_error,
            "error_types": self.error_types,
            "evidence": self.evidence,
        }


def classify_error(record: dict, *, generation: dict | None = None,
                   thresholds: dict | None = None) -> tuple[str, list[str], dict]:
    """Attribute a failure to one or more error types.

    ``record`` is a prediction record; ``generation`` is the sample's generation
    metadata (best evidence) or ``None`` for real scans. Returns
    ``(primary, all_types, evidence)``.
    """
    limits = {**_DEFAULTS, **(thresholds or {})}
    found: list[str] = []
    evidence: dict = {}

    # 1. Explicit problems recorded at generation time are ground truth.
    recorded = list((generation or {}).get("problems") or [])
    if recorded:
        found.extend(recorded)
        evidence["recorded_problems"] = recorded

    quality = record.get("quality") or {}
    if quality:
        evidence["sharpness"] = round(float(quality.get("sharpness", 0.0)), 2)
        if float(quality.get("sharpness", 1e9)) < limits["min_sharpness"]:
            found.append("blur")
        if float(quality.get("clipped_fraction", 0.0)) > limits["max_clipped"]:
            found.append("glare")
        if float(quality.get("crushed_fraction", 0.0)) > limits["max_crushed"]:
            found.append("low_light")
        if float(quality.get("mean_luma", 0.5)) < 0.18:
            found.append("low_light")
        if float(quality.get("glare_fraction", 0.0)) > limits["min_glare"]:
            found.append("glare")
        if float(quality.get("card_area_fraction", 1.0)) < limits["min_card_area"]:
            found.append("tiny_text")
        if float(quality.get("max_field_occlusion", 0.0)) > 0.25:
            found.append("occlusion")

    camera = (generation or {}).get("camera") or {}
    if camera:
        blur_block = camera.get("blur") or {}
        evidence["defocus_sigma"] = blur_block.get("defocus_sigma")
        if float(blur_block.get("defocus_sigma", 0) or 0) > 1.2:
            found.append("blur")
        if float(blur_block.get("motion_length", 0) or 0) > 3.0:
            found.append("blur")
        if int(camera.get("jpeg_quality", 95) or 95) < 45:
            found.append("compression")

    geometry = (generation or {}).get("geometry") or {}
    pose = geometry.get("pose") or {}
    if pose:
        tilt = abs(float(pose.get("pitch", 0))) + abs(float(pose.get("yaw", 0)))
        evidence["tilt_degrees"] = round(tilt, 1)
        if tilt > 35:
            found.append("perspective")

    # 2. Character-level analysis of the actual difference.
    prediction = record.get("prediction") or ""
    reference = record.get("reference") or ""
    # Only reach for character-level analysis when the metadata found nothing:
    # a recorded cause is stronger evidence than a guess from the strings.
    if prediction != reference and not found:
        distance = edit_distance(prediction, reference)
        if distance == 1 and len(reference) > 3:
            found.append("character_confusion")
        elif abs(len(prediction) - len(reference)) > len(reference) * 0.5:
            found.append("occlusion")
        elif _mostly_disjoint(prediction, reference):
            found.append("layout_or_language")
        else:
            found.append("character_confusion")

    # De-duplicate, preserving the priority order of ERROR_TYPES.
    ordered = [name for name in ERROR_TYPES if name in set(found)]
    primary = ordered[0] if ordered else "unknown"
    return primary, ordered or ["unknown"], evidence


def _mostly_disjoint(prediction: str, reference: str) -> bool:
    """True when the strings share almost no characters (wrong field read)."""
    if not prediction or not reference:
        return bool(prediction != reference)
    shared = len(set(prediction.lower()) & set(reference.lower()))
    return shared <= max(1, len(set(reference.lower())) // 4)


def rank_hard_examples(
    records: Sequence[dict],
    *,
    generation: dict[str, dict] | None = None,
    top: int = 20,
    threshold: float = 0.0,
    thresholds: dict | None = None,
) -> dict:
    """Rank failures by severity and summarise their causes.

    ``threshold`` filters out samples whose CER is at or below it (0 keeps every
    imperfect sample).
    """
    generation = generation or {}
    examples: list[HardExample] = []
    for record in records:
        prediction = record.get("prediction") or ""
        reference = record.get("reference") or ""
        if prediction == reference and threshold <= 0:
            continue
        cer = edit_distance(prediction, reference) / max(len(reference), 1)
        if cer <= threshold:
            continue
        primary, types, evidence = classify_error(
            record, generation=generation.get(record.get("image_id", "")),
            thresholds=thresholds,
        )
        examples.append(HardExample(
            image_id=str(record.get("image_id", "")),
            prediction=prediction, reference=reference, cer=cer,
            confidence=record.get("confidence"), primary_error=primary,
            error_types=types, evidence=evidence,
        ))

    examples.sort(key=lambda item: -item.cer)
    all_predictions = [r.get("prediction") or "" for r in records]
    all_references = [r.get("reference") or "" for r in records]
    report = evaluate_ocr(all_predictions, all_references)

    by_type: Counter = Counter()
    for example in examples:
        for error_type in example.error_types:
            by_type[error_type] += 1

    # A concrete recommendation per error type: this is what feeds the retrain.
    recommendations = [
        {
            "problem": error_type,
            "failed_samples": count,
            "share_of_failures": round(count / max(len(examples), 1), 4),
            "command": (f"vision synthetic generate --problem {error_type} "
                        f"--count {max(count * 20, 200)} --mode targeted"),
        }
        for error_type, count in by_type.most_common()
        if error_type in ERROR_TYPES and error_type != "unknown"
    ]

    return {
        "summary": {
            "predictions": len(records),
            "failures": len(examples),
            "failure_rate": round(len(examples) / max(len(records), 1), 4),
            "cer": report.cer,
            "exact_match": report.exact_match,
            "by_error_type": dict(by_type.most_common()),
            "top_confusions": report.top_confusions(10),
        },
        "recommendations": recommendations,
        "examples": [example.to_dict() for example in examples[:top]],
    }


def write_hard_example_manifest(payload: dict, path: str | Path) -> Path:
    """Persist the mined examples as the input to a targeted generation run."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


@dataclass
class MiningLoop:
    """Tracks one iteration of the mine -> generate -> retrain -> evaluate loop.

    Phase 9 is a *cycle*, so the numbers that matter are the deltas between
    iterations, not any single run's CER. This records them.
    """

    history: list[dict] = field(default_factory=list)

    def record(self, iteration: int, *, cer: float, exact_match: float,
               error_types: dict[str, int], samples: int) -> None:
        self.history.append({
            "iteration": iteration,
            "cer": round(cer, 5),
            "exact_match": round(exact_match, 5),
            "samples": samples,
            "error_types": error_types,
        })

    def trend(self) -> dict:
        if not self.history:
            return {"iterations": 0}
        first, last = self.history[0], self.history[-1]
        return {
            "iterations": len(self.history),
            "cer_start": first["cer"],
            "cer_end": last["cer"],
            "cer_delta": round(last["cer"] - first["cer"], 5),
            "exact_match_delta": round(last["exact_match"] - first["exact_match"], 5),
            "converged": len(self.history) > 1 and abs(
                self.history[-1]["cer"] - self.history[-2]["cer"]) < 0.002,
        }
