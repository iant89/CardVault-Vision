"""Evaluation metrics (Phases 4, 11, 18).

OCR metrics are computed on *decoded strings*, never on tensors, so they are
exactly the numbers a product owner reads off the card: how often is the whole
name right (exact match), how many characters are wrong (CER), how many words.

Everything is pure Python/numpy - no torch - so metrics can be recomputed from a
saved ``predictions.jsonl`` without reinstalling a training environment. That is
what makes regression testing (Phase 18) possible on a plain CI box.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Edit distance
# ---------------------------------------------------------------------------

def edit_distance(prediction: str, reference: str) -> int:
    """Levenshtein distance (two-row dynamic programming, O(len(r)) memory)."""
    if prediction == reference:
        return 0
    if not prediction:
        return len(reference)
    if not reference:
        return len(prediction)
    previous = list(range(len(reference) + 1))
    for i, p_char in enumerate(prediction, start=1):
        current = [i]
        for j, r_char in enumerate(reference, start=1):
            cost = 0 if p_char == r_char else 1
            current.append(min(previous[j] + 1,        # deletion
                               current[j - 1] + 1,     # insertion
                               previous[j - 1] + cost))  # substitution
        previous = current
    return previous[-1]


def character_error_rate(predictions: Sequence[str], references: Sequence[str]) -> float:
    """``sum(edit_distance) / sum(len(reference))`` - the standard CER.

    Normalising by total reference length (not by sample count) is the convention
    used in OCR benchmarks, and it means one 40-character name does not count the
    same as one 3-character number.
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must be the same length")
    if not references:
        return float("nan")
    total_distance = sum(edit_distance(p, r) for p, r in zip(predictions, references, strict=True))
    total_length = sum(len(r) for r in references) or 1
    return total_distance / total_length


def word_error_rate(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Word-level error rate (whitespace tokens)."""
    if not references:
        return float("nan")
    total_distance = 0
    total_words = 0
    for prediction, reference in zip(predictions, references, strict=True):
        pred_words = prediction.split()
        ref_words = reference.split()
        total_distance += _token_distance(pred_words, ref_words)
        total_words += len(ref_words)
    return total_distance / (total_words or 1)


def _token_distance(prediction: Sequence[str], reference: Sequence[str]) -> int:
    previous = list(range(len(reference) + 1))
    for i, p_token in enumerate(prediction, start=1):
        current = [i]
        for j, r_token in enumerate(reference, start=1):
            cost = 0 if p_token == r_token else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def exact_match(predictions: Sequence[str], references: Sequence[str]) -> float:
    if not references:
        return float("nan")
    return sum(p == r for p, r in zip(predictions, references, strict=True)) / len(references)


def character_accuracy(predictions: Sequence[str], references: Sequence[str]) -> float:
    """``1 - CER``, floored at 0 (CER can exceed 1 with many insertions)."""
    cer = character_error_rate(predictions, references)
    return max(0.0, 1.0 - cer) if cer == cer else float("nan")


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@dataclass
class OCRReport:
    """The full OCR evaluation report."""

    count: int
    cer: float
    wer: float
    exact_match: float
    character_accuracy: float
    per_sample_cer: list[float] = field(default_factory=list)
    confusions: dict[str, dict[str, int]] = field(default_factory=dict)
    by_length: dict[str, dict[str, float]] = field(default_factory=dict)
    worst: list[dict] = field(default_factory=list)
    mean_confidence: float | None = None
    low_confidence_accuracy: float | None = None
    #: CER split by whether the reference string was present in the training
    #: split. This is the difference between "reads glyphs" and "recalls a
    #: vocabulary": a model that has memorised a small name list scores well on
    #: ``seen`` and collapses on ``unseen``.
    by_vocabulary: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def is_perfect(self) -> bool:
        return self.exact_match == 1.0

    def top_confusions(self, limit: int = 10) -> list[tuple[str, str, int]]:
        """Most frequent ``(reference, prediction)`` character substitutions."""
        flat = []
        for reference, predictions in self.confusions.items():
            for predicted, count in predictions.items():
                flat.append((reference, predicted, count))
        return sorted(flat, key=lambda item: -item[2])[:limit]

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "cer": self.cer,
            "wer": self.wer,
            "exact_match": self.exact_match,
            "character_accuracy": self.character_accuracy,
            "mean_confidence": self.mean_confidence,
            "low_confidence_accuracy": self.low_confidence_accuracy,
            "confusions": self.confusions,
            "by_length": self.by_length,
            "worst": self.worst,
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target

    def report(self) -> str:
        from ..console import kv_block

        lines = [
            kv_block([
                ("samples", self.count),
                ("CER", f"{self.cer:.4f}"),
                ("WER", f"{self.wer:.4f}"),
                ("exact match", f"{self.exact_match:.2%}"),
                ("char accuracy", f"{self.character_accuracy:.2%}"),
            ]),
        ]
        if self.mean_confidence is not None:
            lines.append(f"  mean confidence : {self.mean_confidence:.3f}")
        if self.low_confidence_accuracy is not None:
            lines.append(
                f"  accuracy on low-confidence half : {self.low_confidence_accuracy:.2%}"
            )
        if self.by_length:
            lines.append("\n  by reference length:")
            for bucket, values in sorted(self.by_length.items()):
                lines.append(
                    f"    {bucket:<8} n={int(values['count']):<5} "
                    f"CER={values['cer']:.4f} exact={values['exact_match']:.1%}"
                )
        if self.by_vocabulary:
            lines.append("\n  by training vocabulary:")
            for label, values in sorted(self.by_vocabulary.items()):
                lines.append(
                    f"    {label:<8} n={int(values['count']):<5} "
                    f"CER={values['cer']:.4f} exact={values['exact_match']:.1%}"
                )
        top = self.top_confusions(8)
        if top:
            lines.append("\n  top character confusions (reference -> predicted):")
            for reference, predicted, count in top:
                lines.append(f"    {reference!r} -> {predicted!r}  x{count}")
        if self.worst:
            lines.append("\n  worst samples:")
            for item in self.worst[:5]:
                lines.append(
                    f"    {item['image_id']}: CER={item['cer']:.3f} "
                    f"ref={item['reference']!r} pred={item['prediction']!r}"
                )
        return "\n".join(lines)


def _length_bucket(length: int) -> str:
    if length <= 6:
        return "1-6"
    if length <= 12:
        return "7-12"
    if length <= 20:
        return "13-20"
    return "21+"


def build_confusion_matrix(predictions: Sequence[str], references: Sequence[str], *,
                           limit: int = 60) -> dict[str, dict[str, int]]:
    """Character-level substitution counts from aligned string pairs.

    Alignment uses the Levenshtein backtrace, so insertions and deletions are
    attributed to the correct position rather than to every following character
    (which is what a naive ``zip`` would do, and it produces garbage).
    """
    matrix: dict[str, dict[str, int]] = {}
    for prediction, reference in zip(predictions, references, strict=True):
        for ref_char, pred_char in _aligned_pairs(prediction, reference):
            if ref_char == pred_char:
                continue
            key = ref_char if ref_char is not None else "<ins>"
            value = pred_char if pred_char is not None else "<del>"
            row = matrix.setdefault(key, {})
            row[value] = row.get(value, 0) + 1
    if len(matrix) > limit:
        # Keep the most active characters so the report stays readable.
        ranked = sorted(matrix.items(), key=lambda kv: -sum(kv[1].values()))[:limit]
        matrix = dict(ranked)
    return matrix


def _aligned_pairs(prediction: str, reference: str):
    """Yield ``(reference_char, predicted_char)`` pairs from an optimal alignment."""
    n, m = len(reference), len(prediction)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1] == prediction[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (
                0 if reference[i - 1] == prediction[j - 1] else 1):
            pairs.append((reference[i - 1], prediction[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            pairs.append((reference[i - 1], None))
            i -= 1
        else:
            pairs.append((None, prediction[j - 1]))
            j -= 1
    return reversed(pairs)


def evaluate_ocr(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    image_ids: Sequence[str] | None = None,
    confidences: Sequence[float] | None = None,
    worst_n: int = 10,
    known_references: Sequence[str] | None = None,
) -> OCRReport:
    """Compute the standard OCR report.

    ``known_references`` is the set of reference strings the model was trained
    on. When supplied, the report splits metrics into ``seen`` and ``unseen``.
    Without that split a CER measured on a small fixed vocabulary cannot be told
    apart from one measured on real reading.
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must be the same length")
    if not references:
        return OCRReport(count=0, cer=float("nan"), wer=float("nan"),
                         exact_match=float("nan"), character_accuracy=float("nan"))
    predictions = [p or "" for p in predictions]
    references = [r or "" for r in references]

    per_sample = [edit_distance(p, r) / max(len(r), 1)
                  for p, r in zip(predictions, references, strict=True)]

    by_vocabulary: dict[str, dict[str, float]] = {}
    if known_references is not None:
        known = set(known_references)
        for label, wanted in (("seen", True), ("unseen", False)):
            indices = [i for i, r in enumerate(references) if (r in known) is wanted]
            if not indices:
                continue
            total_length = sum(len(references[i]) for i in indices) or 1
            distance = sum(edit_distance(predictions[i], references[i]) for i in indices)
            by_vocabulary[label] = {
                "count": len(indices),
                "cer": distance / total_length,
                "exact_match": sum(predictions[i] == references[i] for i in indices)
                                / len(indices),
            }

    buckets: dict[str, dict[str, float]] = {}
    for prediction, reference in zip(predictions, references, strict=False):
        bucket = _length_bucket(len(reference))
        entry = buckets.setdefault(bucket, {"count": 0, "distance": 0, "length": 0,
                                            "exact": 0})
        entry["count"] += 1
        entry["distance"] += edit_distance(prediction, reference)
        entry["length"] += len(reference)
        entry["exact"] += int(prediction == reference)
    by_length = {
        bucket: {
            "count": values["count"],
            "cer": values["distance"] / (values["length"] or 1),
            "exact_match": values["exact"] / values["count"],
        }
        for bucket, values in buckets.items()
    }

    ids = list(image_ids) if image_ids else [f"sample_{i}" for i in range(len(references))]
    order = sorted(range(len(per_sample)), key=lambda i: -per_sample[i])[:worst_n]
    worst = [
        {
            "image_id": ids[i],
            "reference": references[i],
            "prediction": predictions[i],
            "cer": per_sample[i],
        }
        for i in order if per_sample[i] > 0
    ]

    mean_confidence = None
    low_conf_accuracy = None
    if confidences is not None and len(confidences) == len(predictions):
        mean_confidence = float(sum(confidences) / len(confidences))
        if len(confidences) >= 4:
            order_conf = sorted(range(len(confidences)), key=lambda i: confidences[i])
            half = len(order_conf) // 2
            low = order_conf[:half]
            low_conf_accuracy = sum(
                1 for i in low if predictions[i] == references[i]
            ) / max(len(low), 1)

    return OCRReport(
        count=len(references),
        cer=character_error_rate(predictions, references),
        wer=word_error_rate(predictions, references),
        exact_match=exact_match(predictions, references),
        character_accuracy=character_accuracy(predictions, references),
        per_sample_cer=per_sample,
        confusions=build_confusion_matrix(predictions, references),
        by_length=by_length,
        worst=worst,
        mean_confidence=mean_confidence,
        low_confidence_accuracy=low_conf_accuracy,
        by_vocabulary=by_vocabulary,
    )


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

@dataclass
class ClassificationReport:
    """Confusion matrix plus the metrics that actually matter for routing."""

    class_names: list[str]
    confusion: list[list[int]]
    accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    unreadable_rate: float | None = None

    @property
    def count(self) -> int:
        return sum(sum(row) for row in self.confusion)

    def to_dict(self) -> dict:
        return {
            "class_names": self.class_names,
            "confusion": self.confusion,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "per_class": self.per_class,
            "unreadable_rate": self.unreadable_rate,
            "count": self.count,
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target

    def report(self) -> str:
        rows = [[name, *(str(v) for v in row)] for name, row in
                zip(self.class_names, self.confusion, strict=True)]
        from ..console import format_table, kv_block

        lines = [
            kv_block([
                ("samples", self.count),
                ("accuracy", f"{self.accuracy:.2%}"),
                ("macro F1", f"{self.macro_f1:.4f}"),
            ]),
            "",
            format_table(["true \\ pred", *self.class_names], rows),
            "",
            format_table(["class", "precision", "recall", "f1", "support"],
                         [[name, f"{v['precision']:.3f}", f"{v['recall']:.3f}",
                           f"{v['f1']:.3f}", str(int(v['support']))]
                          for name, v in self.per_class.items()]),
        ]
        return "\n".join(lines)


def evaluate_classification(
    predictions: Iterable[int | str] | Iterable[str],
    references: Iterable[str],
    class_names: Sequence[str],
) -> ClassificationReport:
    """Build a confusion matrix and per-class precision/recall/F1."""
    index = {name: i for i, name in enumerate(class_names)}
    refs = [str(r) for r in references]
    resolved_predictions: list[str] = []
    for prediction in predictions:
        if isinstance(prediction, (int, np.integer)):
            resolved_predictions.append(
                class_names[int(prediction)] if 0 <= int(prediction) < len(class_names)
                else "<out-of-range>"
            )
        else:
            resolved_predictions.append(str(prediction))
    preds = resolved_predictions
    matrix = [[0] * len(class_names) for _ in class_names]
    total = 0
    correct = 0
    for reference, prediction in zip(refs, preds, strict=True):
        if reference not in index:
            continue
        row = index[reference]
        col = index.get(prediction, None)
        if col is not None:
            matrix[row][col] += 1
        total += 1
        if prediction == reference:
            correct += 1

    per_class: dict[str, dict[str, float]] = {}
    f1_scores = []
    for i, name in enumerate(class_names):
        true_positive = matrix[i][i]
        predicted_total = sum(matrix[r][i] for r in range(len(class_names)))
        actual_total = sum(matrix[i])
        precision = true_positive / predicted_total if predicted_total else 0.0
        recall = true_positive / actual_total if actual_total else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1,
                           "support": actual_total}
        if actual_total:
            f1_scores.append(f1)
    return ClassificationReport(
        class_names=list(class_names),
        confusion=matrix,
        accuracy=correct / total if total else float("nan"),
        macro_f1=sum(f1_scores) / len(f1_scores) if f1_scores else float("nan"),
        per_class=per_class,
    )


# ---------------------------------------------------------------------------
# Gate / regression comparison (Phase 18)
# ---------------------------------------------------------------------------

@dataclass
class RegressionGate:
    """Thresholds a candidate model must meet before it may replace production."""

    max_cer_delta: float = 0.01       # candidate CER may be at most 1 pp worse
    max_wer_delta: float = 0.02
    min_exact_match_delta: float = -0.01
    max_latency_ratio: float = 1.25    # candidate may be at most 25% slower
    max_size_ratio: float = 1.5

    def check(self, candidate: dict, baseline: dict) -> tuple[bool, list[str]]:
        """Return ``(passed, reasons)``."""
        problems = []
        cer_delta = candidate.get("cer", 1.0) - baseline.get("cer", 1.0)
        if cer_delta > self.max_cer_delta:
            problems.append(
                f"CER regressed by {cer_delta:+.4f} (limit +{self.max_cer_delta:.4f})"
            )
        wer_delta = candidate.get("wer", 1.0) - baseline.get("wer", 1.0)
        if wer_delta > self.max_wer_delta:
            problems.append(
                f"WER regressed by {wer_delta:+.4f} (limit +{self.max_wer_delta:.4f})"
            )
        em_delta = candidate.get("exact_match", 0.0) - baseline.get("exact_match", 0.0)
        if em_delta < self.min_exact_match_delta:
            problems.append(
                f"exact match dropped {em_delta:+.4f} "
                f"(limit {self.min_exact_match_delta:+.4f})"
            )
        base_latency = baseline.get("cpu_latency_ms") or baseline.get("latency_ms")
        cand_latency = candidate.get("cpu_latency_ms") or candidate.get("latency_ms")
        if base_latency and cand_latency:
            if cand_latency > base_latency * self.max_latency_ratio:
                problems.append(
                    f"latency {cand_latency:.1f}ms is "
                    f"{cand_latency / base_latency:.1f}x the baseline ({base_latency:.1f}ms)"
                )
        base_size = baseline.get("size_mb") or baseline.get("size_mb_fp32")
        cand_size = candidate.get("size_mb") or candidate.get("size_mb_fp32")
        if base_size and cand_size and cand_size > base_size * self.max_size_ratio:
            problems.append(
                f"model size {cand_size:.1f}MB is {cand_size / base_size:.1f}x the baseline"
            )
        return (not problems), problems
