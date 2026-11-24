"""Grade the CardVault Image Engine's card detection against synthetic truth.

The synthetic engine knows the exact card quad for every image it renders, which
makes it a rare thing for a detector: a benchmark with free, exact labels. This
module runs the engine's alignment stage over a dataset and reports corner error,
IoU and detection rate, so "the corners look about right" becomes a number.

It is honest about what it measures. These are *synthetic* captures - stylised
card art, a handful of backgrounds, one lighting model. A detector that scores
well here has passed a real test of geometry handling, not a test of real-world
photographs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..evaluation.alignment import (
    AlignmentCase,
    AlignmentReport,
    best_rotation,
    normalised_corner_error,
    quad_iou,
)
from ..logging_utils import get_logger
from .image_engine import AlignmentResult, align, engine_available

_LOG = get_logger("integrations.image_engine.fixtures")


@dataclass
class FixtureResult:
    """The graded report plus the raw engine results, for inspection."""

    report: AlignmentReport
    failures: list[AlignmentCase]


def grade(
    loader: Iterable,
    *,
    profile: str | None = None,
    tolerance: float = 0.02,
    limit: int | None = None,
    options: dict | None = None,
    engine: str | Path | None = None,
) -> FixtureResult:
    """Run the engine over ``loader`` and grade every sample against truth.

    ``loader`` yields :class:`cardvault_vision.datasets.loader.Sample` objects;
    those carry the pixels and the annotation together, so no second pass over
    disk is needed.
    """
    report = AlignmentReport(tolerance=tolerance)
    failures: list[AlignmentCase] = []

    for index, sample in enumerate(loader):
        if limit is not None and index >= limit:
            break
        truth_quad = sample.annotation.card_quad
        if truth_quad is None:
            _LOG.debug("Skipping %s: no card_quad to grade against", sample.image_id)
            continue
        truth = [(float(x), float(y)) for x, y in truth_quad.points]

        result: AlignmentResult = align(sample.image, profile=profile, options=options,
                                        engine=engine)
        case = AlignmentCase(
            image_id=sample.image_id,
            status=result.status,
            candidates=len(result.candidates),
            confidence=result.confidence,
            profile=sample.profile,
        )
        # Every candidate, not just the winner: the engine may have proposed the
        # right quad and ranked it below a sliver, which is a ranking bug rather
        # than a detection failure and needs to be told apart from one.
        for candidate in result.candidates:
            corners = candidate.get("corners") or []
            if len(corners) != 4:
                continue
            quad = [(float(point["x"]), float(point["y"])) for point in corners]
            iou = quad_iou(quad, truth)
            if case.oracle_iou is None or iou > case.oracle_iou:
                case.oracle_iou = iou
                case.oracle_corner_error = normalised_corner_error(quad, truth)

        if result.corners:
            case.corner_error = normalised_corner_error(result.corners, truth)
            case.iou = quad_iou(result.corners, truth)
            case.rotation_shift, _ = best_rotation(result.corners, truth)
            if case.rotation_shift != 0:
                _LOG.warning(
                    "Sample %s matched truth only after rotating %d step(s): the "
                    "engine emits TL, TR, BR, BL and CardVault Vision records the "
                    "same order, so a non-zero shift means one of them changed.",
                    sample.image_id, case.rotation_shift,
                )
        report.cases.append(case)
        if case.corner_error is None or case.corner_error > tolerance:
            failures.append(case)

    return FixtureResult(report=report, failures=failures)


def require_engine(engine: str | Path | None = None) -> None:
    """Raise a clear error when the engine is not configured."""
    if not engine_available(engine):
        raise RuntimeError(
            "The CardVault Image Engine is not available. Build it "
            "(npm run build --workspace @cardvault/image-engine) and set "
            "CVV_IMAGE_ENGINE to its dist directory."
        )


def worst(result: FixtureResult, limit: int = 10) -> Sequence[AlignmentCase]:
    """The highest corner errors first - the cases worth looking at."""
    graded = [case for case in result.report.cases if case.corner_error is not None]
    return sorted(graded, key=lambda case: -case.corner_error)[:limit]


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[position]


def distribution(result: FixtureResult) -> dict[str, float]:
    """Percentiles of the normalised corner error, for a fuller picture."""
    values = [case.corner_error for case in result.report.cases
              if case.corner_error is not None]
    return {
        "p50": _quantile(values, 0.50),
        "p75": _quantile(values, 0.75),
        "p90": _quantile(values, 0.90),
        "p99": _quantile(values, 0.99),
        "max": max(values) if values else float("nan"),
    }
