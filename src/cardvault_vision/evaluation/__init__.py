"""Evaluation, metrics and hard-example mining (Phases 4, 9, 11, 18)."""

from __future__ import annotations

from .hard_examples import (
    ERROR_TYPES,
    HardExample,
    MiningLoop,
    classify_error,
    rank_hard_examples,
    write_hard_example_manifest,
)
from .metrics import (
    ClassificationReport,
    OCRReport,
    RegressionGate,
    build_confusion_matrix,
    character_accuracy,
    character_error_rate,
    edit_distance,
    evaluate_classification,
    evaluate_ocr,
    exact_match,
    word_error_rate,
)

__all__ = [
    "ClassificationReport",
    "ERROR_TYPES",
    "HardExample",
    "MiningLoop",
    "OCRReport",
    "RegressionGate",
    "build_confusion_matrix",
    "character_accuracy",
    "character_error_rate",
    "classify_error",
    "edit_distance",
    "evaluate_classification",
    "evaluate_ocr",
    "exact_match",
    "rank_hard_examples",
    "word_error_rate",
    "write_hard_example_manifest",
]
