"""Card detection on top of the image engine: edge presets plus candidate selection.

The engine returns a *ranked list* of candidate quads and picks the first one. On
this project's captures that pipeline fails twice over, and the two failures need
separate fixes:

1. **Detection.** The engine's default ``edgeThreshold`` of 48 is far too
   aggressive for these images. The card's outline is broken by perspective,
   shadow and its own interior artwork, so a threshold that high stops producing a
   closed boundary and the card is never proposed at all. Measured on 40 samples:
   at 48 the true card is within 0.05 of the diagonal on **0** samples; at 12-24
   with ``morphologyRadius`` 3 it is within 0.05 on **9-17**.
2. **Ranking.** The engine's confidence rewards a clean, compact mask
   (``fillScore`` 0.30, ``shapeScore`` 0.45) more than a large plausible one, so a
   3%-area sliver can outrank the real card. Re-ranking a *fixed* candidate list
   is worth roughly one extra sample at the 0.10 tolerance and a visible drop in
   median error - a real but second-order gain compared with (1).

This module attacks both, and exists to answer one question honestly: how much of
the alignment error is detection, how much is ordering, and what is left.

- :data:`EDGE_PRESETS` and :func:`detect` fix (1). Several thresholds are run and
  their candidate lists pooled, because a card that vanishes at 12 can survive at
  24 and vice versa; the union is consistently better than any single value
  (see ``docs/image-engine-integration.md`` for the table).
- :class:`SelectionPolicy` and :func:`select` fix (2), choosing from that pool.

It is a re-ranker, not a replacement detector. If the true card is not among the
candidates, no ordering can recover it - which is why
:class:`~cardvault_vision.evaluation.alignment.AlignmentReport` reports the oracle
figure alongside the top-1 figure, and why :func:`detect` keeps every raw engine
result so the two can be told apart after the fact.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

Candidate = dict
Point = tuple[float, float]

#: Edge settings worth running, in the order they are tried. Measured on 40
#: synthetic captures spanning standard, holo and foil finishes: values below 8
#: admit noise edges, values above 32 stop closing the card's boundary, and the
#: usable band is narrow. Pooling these three is what :func:`detect` does by
#: default; changing them without re-measuring will silently hurt alignment.
EDGE_PRESETS: tuple[dict, ...] = (
    {"edgeThreshold": 12, "morphologyRadius": 3},
    {"edgeThreshold": 16, "morphologyRadius": 3},
    {"edgeThreshold": 24, "morphologyRadius": 3},
)

#: The single preset to use when only one pass is affordable.
DEFAULT_ENGINE_OPTIONS: dict = EDGE_PRESETS[0]


def _corners(candidate: Candidate) -> list[Point]:
    return [(float(point["x"]), float(point["y"])) for point in candidate.get("corners") or []]


def _edge(point: Point, other: Point) -> float:
    return math.hypot(other[0] - point[0], other[1] - point[1])


def quad_area(quad: Sequence[Point]) -> float:
    """Shoelace area of a (possibly concave-ordered) quadrilateral."""
    total = 0.0
    points = list(quad)
    for index in range(len(points)):
        x0, y0 = points[index]
        x1, y1 = points[(index + 1) % len(points)]
        total += x0 * y1 - x1 * y0
    return abs(total) / 2.0


def aspect_ratio(quad: Sequence[Point]) -> float:
    """Shorter side over longer side, in ``(0, 1]``.

    The engine normalises the same way (``min/max``), which is what makes its
    profile comparison orientation-agnostic - and also why feeding it the
    reciprocal of an aspect ratio does not raise.
    """
    points = list(quad)
    if len(points) != 4:
        return 0.0
    sides = [_edge(points[i], points[(i + 1) % 4]) for i in range(4)]
    shorter = (sides[0] + sides[2]) / 2.0
    longer = (sides[1] + sides[3]) / 2.0
    shorter, longer = min(shorter, longer), max(shorter, longer)
    if longer <= 0:
        return 0.0
    return shorter / longer


def rectangularity(quad: Sequence[Point]) -> float:
    """How close the quad is to a rectangle, in ``[0, 1]``.

    Compares the quad's area to the area of its four side vectors treated as
    if they were sides of a rectangle. A perfect rectangle scores 1; a sliver or
    a heavily skewed quad scores lower.
    """
    points = list(quad)
    if len(points) != 4:
        return 0.0
    sides = [_edge(points[i], points[(i + 1) % 4]) for i in range(4)]
    width = (sides[0] + sides[2]) / 2.0
    height = (sides[1] + sides[3]) / 2.0
    ideal = width * height
    if ideal <= 0:
        return 0.0
    return min(1.0, quad_area(points) / ideal)


def border_fraction(quad: Sequence[Point], frame_width: int, frame_height: int,
                    margin: float | None = None) -> float:
    """Fraction of corners that lie on the frame's edge, in ``[0, 1]``.

    This is the term that fixes the most common wrong answer on these captures:
    the whole *frame* returned as the card. A quad that touches the image border
    is what the background-texture contour looks like, and it is large,
    rectangular and correctly proportioned - everything the other three terms
    reward. On a 40-sample set it is the top pick on 25 samples.

    A real card can graze the border (captures are cropped tight), so this is a
    penalty rather than a rejection, and the margin is a small fraction of the
    frame rather than an exact pixel test.
    """
    points = list(quad)
    if len(points) != 4:
        return 0.0
    if margin is None:
        margin = max(1.0, 0.01 * min(frame_width, frame_height))
    right, bottom = frame_width - 1 - margin, frame_height - 1 - margin
    touching = sum(1 for x, y in points
                   if x <= margin or y <= margin or x >= right or y >= bottom)
    return touching / 4.0


def convexity(quad: Sequence[Point]) -> bool:
    """True when all four corners turn the same way (no self-intersection)."""
    points = list(quad)
    if len(points) != 4:
        return False
    signs = []
    for index in range(4):
        ax, ay = points[index]
        bx, by = points[(index + 1) % 4]
        cx, cy = points[(index + 2) % 4]
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if abs(cross) > 1e-9:
            signs.append(cross > 0)
    return len(set(signs)) <= 1


@dataclass(frozen=True)
class SelectionPolicy:
    """Weights for re-ranking candidates.

    The defaults were chosen by measurement on synthetic captures, not by taste;
    see docs/image-engine-integration.md for the table that produced them.
    """

    #: Weight on the card's area as a fraction of the frame. A card occupies a
    #: substantial part of the frame; fragments do not.
    area_weight: float = 0.55
    #: Weight on aspect-ratio agreement with the requested profile.
    aspect_weight: float = 0.25
    #: Weight on geometric rectangularity.
    shape_weight: float = 0.20
    #: Weight on the *penalty* for corners sitting on the frame border. This is
    #: the largest single win after edge tuning: without it the whole frame wins
    #: on more than half the samples, because a frame-sized quad is large,
    #: rectangular and correctly proportioned. See :func:`border_fraction`.
    border_weight: float = 0.25
    #: Candidates smaller than this fraction of the frame are ignored outright.
    min_area_ratio: float = 0.08
    #: Lower bound on aspect agreement, to drop obvious non-cards early.
    min_aspect_score: float = 0.0

    def score(self, candidate: Candidate, *, frame_width: int, frame_height: int,
              target_ratio: float) -> float:
        """Score one candidate in ``[0, 1]``, or ``-1.0`` to reject it."""
        corners = _corners(candidate)
        if len(corners) != 4 or not convexity(corners):
            return -1.0
        frame_area = float(frame_width) * float(frame_height)
        area_ratio = quad_area(corners) / frame_area if frame_area > 0 else 0.0
        if area_ratio < self.min_area_ratio:
            return -1.0
        measured = aspect_ratio(corners)
        if measured <= 0:
            return -1.0
        # Relative error against the profile, clamped to [0, 1].
        aspect_score = max(0.0, 1.0 - abs(measured - target_ratio) / max(target_ratio, 1e-6))
        if aspect_score < self.min_aspect_score:
            return -1.0
        # sqrt so that a 60%-of-frame card and a 100%-of-frame card are close,
        # rather than the largest candidate winning on area alone.
        area_score = min(1.0, math.sqrt(area_ratio / 0.6))
        shape_score = rectangularity(corners)
        border_score = border_fraction(corners, frame_width, frame_height)
        return (self.area_weight * area_score
                + self.aspect_weight * aspect_score
                + self.shape_weight * shape_score
                - self.border_weight * border_score)


DEFAULT_POLICY = SelectionPolicy()

#: height/width for the engine's card profiles, keyed by the same names the
#: bridge accepts. The engine stores width/height; this table is the inverse,
#: and matches ``CardLayout.aspect`` (which is also height/width).
PROFILE_RATIOS = {
    "standard-trading-card": 88 / 63,
    "pokemon": 88 / 63,
    "magic-the-gathering": 88 / 63,
    "yu-gi-oh": 86 / 59,
    "sports": 3.5 / 2.5,
}


def target_ratio_for(profile: str | None) -> float:
    """Normalised ``min/max`` aspect ratio for a profile name."""
    ratio = PROFILE_RATIOS.get(profile or "standard-trading-card", 88 / 63)
    return min(ratio, 1.0 / ratio)


def select(
    candidates: Sequence[Candidate],
    *,
    frame_width: int,
    frame_height: int,
    profile: str | None = None,
    policy: SelectionPolicy = DEFAULT_POLICY,
) -> tuple[Candidate | None, float]:
    """Pick the best candidate under ``policy``; returns ``(candidate, score)``.

    ``None`` with a score of ``-inf`` means every candidate was rejected, which
    is a legitimate answer: a frame that contains no card-shaped quad should
    report that rather than return the least bad fragment.
    """
    target = target_ratio_for(profile)
    best: Candidate | None = None
    best_score = -math.inf
    for candidate in candidates:
        score = policy.score(candidate, frame_width=frame_width,
                             frame_height=frame_height, target_ratio=target)
        # ``-1.0`` means rejected, not "scored badly". Comparing it against
        # ``best_score`` (which starts at ``-inf``) would let the first rejected
        # candidate through and quietly return a fragment the policy refused.
        if score < 0.0:
            continue
        if score > best_score:
            best, best_score = candidate, score
    return best, best_score


@dataclass
class Detection:
    """Everything one image's detection produced, raw results included.

    The point of keeping :attr:`passes` is auditability: the aggregate numbers
    collapse to a single quad, but "the card was never proposed" and "the card
    was proposed and rejected" are different bugs and only the raw engine output
    can tell them apart after the fact.
    """

    #: Pooled candidates from every pass, each annotated with ``pass`` and
    #: ``preset`` so a result can be traced back to the settings that found it.
    candidates: list[Candidate] = field(default_factory=list)
    #: The candidate :func:`select` chose, or ``None``.
    chosen: Candidate | None = None
    #: Score of :attr:`chosen`; ``-inf`` when nothing was acceptable.
    score: float = -math.inf
    #: The engine's own result for each pass, in order.
    passes: list[object] = field(default_factory=list)
    profile: str | None = None

    @property
    def corners(self) -> list[Point] | None:
        if self.chosen is None:
            return None
        corners = _corners(self.chosen)
        return corners or None

    @property
    def status(self) -> str:
        """``ALIGNED`` when a candidate survived selection, else the first
        pass's own verdict (``CARD_NOT_FOUND`` / ``LOW_CONFIDENCE`` / ...)."""
        if self.chosen is not None:
            return "ALIGNED"
        for result in self.passes:
            status = getattr(result, "status", None)
            if status and status != "ALIGNED":
                return str(status)
        return "CARD_NOT_FOUND"

    @property
    def aligned(self) -> bool:
        return self.chosen is not None


def detect(
    image: object,
    *,
    profile: str | None = None,
    engine: str | None = None,
    node: str | None = None,
    presets: Sequence[dict] | None = None,
    options: dict | None = None,
    policy: SelectionPolicy = DEFAULT_POLICY,
    timeout: float = 300.0,
) -> Detection:
    """Run the engine over ``image`` and select a card quad from the candidates.

    ``options`` forces a single pass with exactly those engine settings - which
    is what makes an individual threshold reproducible from the command line.
    Otherwise every entry of ``presets`` (default :data:`EDGE_PRESETS`) is run and
    the candidate lists are pooled. Runs are sequential on purpose: the bridge is
    a short-lived Node process per image, and three concurrent ones buy nothing
    on a two-core machine while making a partial failure much harder to report.
    """
    import numpy as np

    from .image_engine import align

    array = np.asarray(image)
    if array.ndim < 2:
        raise ValueError(f"unsupported image shape {array.shape}")
    height, width = array.shape[0], array.shape[1]

    if options is not None:
        passes: Sequence[dict | None] = (options,)
    else:
        passes = tuple(presets) if presets is not None else EDGE_PRESETS

    detection = Detection(profile=profile)
    for index, preset in enumerate(passes):
        result = align(array, profile=profile, options=preset or None,
                       engine=engine, node=node, timeout=timeout)
        detection.passes.append(result)
        for candidate in result.candidates:
            annotated = dict(candidate)
            annotated["pass"] = index
            annotated["preset"] = dict(preset or {})
            detection.candidates.append(annotated)

    detection.chosen, detection.score = select(
        detection.candidates,
        frame_width=width,
        frame_height=height,
        profile=profile,
        policy=policy,
    )
    return detection
