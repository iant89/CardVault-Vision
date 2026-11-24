"""Card-boundary metrics for grading an aligner against known ground truth.

The synthetic engine records the exact card quad for every image it renders, so
an aligner can be scored against truth rather than against another heuristic's
opinion. That is the only way to put a number on "the corners are close": a
detector that is consistently 40 px off in the same direction looks plausible in
a preview and is useless for cropping a name field.

Everything here is pure Python: no torch, no Node, no image decoding. The bridge
to the TypeScript engine lives in :mod:`cardvault_vision.integrations.image_engine`
and only supplies the quads.

Corner order is ``TL, TR, BR, BL`` throughout - what
:class:`cardvault_vision.datasets.schema.Quad` records, and what the image engine
emits. If a caller's quads use a different order, :func:`best_rotation` reports
it instead of silently accepting a bad match.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

Point = tuple[float, float]
Quad = Sequence[Point]


def quad_area(quad: Quad) -> float:
    """Shoelace area of a (possibly concave-ordered) quadrilateral."""
    total = 0.0
    points = list(quad)
    for index in range(len(points)):
        x0, y0 = points[index]
        x1, y1 = points[(index + 1) % len(points)]
        total += x0 * y1 - x1 * y0
    return abs(total) / 2.0


def quad_diagonal(quad: Quad) -> float:
    """Mean of the two diagonals - the natural scale for a normalised error."""
    points = list(quad)
    if len(points) != 4:
        raise ValueError(f"expected 4 corners, got {len(points)}")
    first = math.dist(points[0], points[2])
    second = math.dist(points[1], points[3])
    return (first + second) / 2.0


def corner_errors(predicted: Quad, truth: Quad) -> list[float]:
    """Per-corner Euclidean distance, in input pixels, assuming matching order."""
    if len(predicted) != 4 or len(truth) != 4:
        raise ValueError("both quads need exactly 4 corners")
    return [math.dist(p, t) for p, t in zip(predicted, truth, strict=True)]


def normalised_corner_error(predicted: Quad, truth: Quad) -> float:
    """Mean corner distance as a fraction of the truth card's diagonal.

    Scale-free, so a detector that is uniformly 2% off scores the same whether
    the card is 300 px or 3000 px across. A value of 0.05 on a card whose name
    field is a fifth of the height is not "close" - it is several characters
    worth of drift.
    """
    diagonal = quad_diagonal(truth)
    if diagonal <= 0:
        raise ValueError("truth quad has zero size")
    return sum(corner_errors(predicted, truth)) / 4.0 / diagonal


def _clip_polygon(subject: Sequence[Point], clip: Sequence[Point]) -> list[Point]:
    """Sutherland-Hodgman clipping of ``subject`` by a convex ``clip`` polygon."""
    output = list(subject)
    clip_points = list(clip)
    if len(clip_points) < 3:
        return []

    def inside(point: Point, start: Point, end: Point) -> bool:
        # Cross product sign: which side of the directed edge the point is on.
        return ((end[0] - start[0]) * (point[1] - start[1])
                - (end[1] - start[1]) * (point[0] - start[0])) >= 0

    def intersect(start: Point, end: Point, clip_start: Point, clip_end: Point) -> Point:
        x1, y1 = start
        x2, y2 = end
        x3, y3 = clip_start
        x4, y4 = clip_end
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denominator) < 1e-12:
            return end
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    for index, clip_start in enumerate(clip_points):
        clip_end = clip_points[(index + 1) % len(clip_points)]
        if not output:
            return []
        previous = output
        output = []
        for position, current in enumerate(previous):
            start = previous[position - 1]
            if inside(current, clip_start, clip_end):
                if not inside(start, clip_start, clip_end):
                    output.append(intersect(start, current, clip_start, clip_end))
                output.append(current)
            elif inside(start, clip_start, clip_end):
                output.append(intersect(start, current, clip_start, clip_end))
    return output


def quad_iou(predicted: Quad, truth: Quad) -> float:
    """Intersection over union of two convex quadrilaterals, in ``[0, 1]``.

    Corner error and IoU fail differently, which is why both are reported: a
    detector can have small corner error and still cover the wrong area if it
    drifts along an edge, and it can have a decent IoU while cutting a slice off
    one side - which is what destroys a name crop.
    """
    clipped = _clip_polygon(list(predicted), list(truth))
    intersection = quad_area(clipped) if len(clipped) >= 3 else 0.0
    union = quad_area(predicted) + quad_area(truth) - intersection
    if union <= 0:
        return 0.0
    return max(0.0, min(1.0, intersection / union))


def best_rotation(predicted: Quad, truth: Quad) -> tuple[int, float]:
    """Best cyclic shift of ``predicted`` and its corner error.

    A detector that reports ``[BL, TL, TR, BR]`` is not wrong about the card, it
    is wrong about the convention. Reporting the shift that reconciles the two
    makes that visible; silently rotating to the best match would hide a real
    contract mismatch.
    """
    best_index, best_error = 0, float("inf")
    for shift in range(4):
        rotated = [predicted[(i + shift) % 4] for i in range(4)]
        error = sum(corner_errors(rotated, truth)) / 4.0
        if error < best_error:
            best_index, best_error = shift, error
    return best_index, best_error


@dataclass
class AlignmentCase:
    """One graded sample."""

    image_id: str
    status: str
    corner_error: float | None = None       # normalised by truth diagonal
    iou: float | None = None
    rotation_shift: int = 0
    confidence: float | None = None
    candidates: int = 0
    profile: str | None = None
    #: Best IoU among *all* candidates the engine proposed, including ones it
    #: ranked below its top pick. The gap between this and :attr:`iou` is the
    #: difference between "did not find the card" and "found it and ranked it
    #: wrong" - two different bugs, with different fixes.
    oracle_iou: float | None = None
    oracle_corner_error: float | None = None

    @property
    def ranked_wrong(self) -> bool:
        """The card was among the candidates but not chosen as the top one.

        Compared on corner error rather than IoU so the flag still works when a
        caller grades only corners; either measure gives the same answer when
        both are present.
        """
        if self.oracle_corner_error is None:
            return False
        if self.corner_error is None:
            return True      # a candidate was close, the top pick was not
        return self.oracle_corner_error < self.corner_error - 1e-12


@dataclass
class AlignmentReport:
    """Aggregate of graded alignment cases."""

    cases: list[AlignmentCase] = field(default_factory=list)
    #: A sample counts as detected when the engine aligned it *and* the corners
    #: land within this normalised error.
    tolerance: float = 0.02

    @property
    def count(self) -> int:
        return len(self.cases)

    @property
    def graded(self) -> list[AlignmentCase]:
        return [case for case in self.cases if case.corner_error is not None]

    @property
    def detection_rate(self) -> float:
        if not self.cases:
            return float("nan")
        return len(self.graded) / len(self.cases)

    @property
    def median_corner_error(self) -> float:
        values = sorted(case.corner_error for case in self.graded)
        if not values:
            return float("nan")
        middle = len(values) // 2
        if len(values) % 2:
            return values[middle]
        return (values[middle - 1] + values[middle]) / 2.0

    @property
    def median_iou(self) -> float:
        values = sorted(case.iou for case in self.graded if case.iou is not None)
        if not values:
            return float("nan")
        middle = len(values) // 2
        if len(values) % 2:
            return values[middle]
        return (values[middle - 1] + values[middle]) / 2.0

    @property
    def median_oracle_corner_error(self) -> float:
        """Corner error of the *best* candidate per sample, ignoring ranking.

        The ceiling this detector could reach if it ranked candidates perfectly.
        Reading it next to :attr:`median_corner_error` separates two failures
        that look identical in an aggregate: a detector that cannot locate the
        card, and one that locates it and then picks the wrong candidate.
        """
        values = sorted(case.oracle_corner_error for case in self.cases
                        if case.oracle_corner_error is not None)
        if not values:
            return float("nan")
        middle = len(values) // 2
        if len(values) % 2:
            return values[middle]
        return (values[middle - 1] + values[middle]) / 2.0

    @property
    def within_tolerance(self) -> int:
        return sum(1 for case in self.graded if case.corner_error <= self.tolerance)

    @property
    def oracle_detection_rate(self) -> float:
        """Share of samples where *some* candidate was close to truth.

        Compare against :attr:`detection_rate`. A large gap means the detector
        proposed the card and then ranked something else first - a ranking
        problem. No gap means it never found the card at all - a detection
        problem.
        """
        if not self.cases:
            return float("nan")
        found = sum(1 for case in self.cases
                    if case.oracle_iou is not None
                    and case.oracle_corner_error is not None
                    and case.oracle_corner_error <= self.tolerance)
        return found / len(self.cases)

    @property
    def ranked_wrong(self) -> int:
        return sum(1 for case in self.cases if case.ranked_wrong)

    @property
    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for case in self.cases:
            counts[case.status] = counts.get(case.status, 0) + 1
        return counts

    def by_profile(self) -> dict[str, dict[str, float]]:
        groups: dict[str, list[AlignmentCase]] = {}
        for case in self.cases:
            groups.setdefault(case.profile or "unknown", []).append(case)
        out: dict[str, dict[str, float]] = {}
        for name, cases in groups.items():
            graded = [case for case in cases if case.corner_error is not None]
            if not graded:
                out[name] = {"count": float(len(cases)), "detection_rate": 0.0}
                continue
            errors = sorted(case.corner_error for case in graded)
            out[name] = {
                "count": float(len(cases)),
                "detection_rate": len(graded) / len(cases),
                "median_corner_error": errors[len(errors) // 2],
                "within_tolerance": float(
                    sum(1 for case in graded if case.corner_error <= self.tolerance)
                ),
            }
        return out

    def summary(self) -> dict:
        return {
            "count": self.count,
            "detected": len(self.graded),
            "detection_rate": self.detection_rate,
            "oracle_detection_rate": self.oracle_detection_rate,
            "ranked_wrong": self.ranked_wrong,
            "within_tolerance": self.within_tolerance,
            "tolerance": self.tolerance,
            "median_corner_error": self.median_corner_error,
            "median_iou": self.median_iou,
            "median_oracle_corner_error": self.median_oracle_corner_error,
            "status_counts": self.status_counts,
            "rotation_shifts": {
                str(shift): sum(1 for case in self.graded if case.rotation_shift == shift)
                for shift in sorted({case.rotation_shift for case in self.graded})
            },
            "by_profile": self.by_profile(),
        }
