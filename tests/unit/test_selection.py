"""Candidate re-ranking and edge-preset pooling.

Everything here is pure Python: no Node, no engine, no images. The shapes are
chosen so the expected values can be computed by hand, because the alternative -
comparing the re-ranker against the engine - would only prove the two agree.

The behaviour that matters in production is the *ordering*: a card-shaped quad
must beat a frame-sized one, and a fragment must lose to both. The default
weights were tuned on synthetic captures, so the tests assert the direction and
the separation rather than the exact numbers, which are free to move if the
policy is retuned.
"""

from __future__ import annotations

import math

import pytest

from cardvault_vision.integrations.selection import (
    DEFAULT_ENGINE_OPTIONS,
    DEFAULT_POLICY,
    EDGE_PRESETS,
    PROFILE_RATIOS,
    SelectionPolicy,
    aspect_ratio,
    border_fraction,
    convexity,
    detect,
    quad_area,
    rectangularity,
    select,
    target_ratio_for,
)

FRAME_W, FRAME_H = 1240, 930


def quad(x0, y0, x1, y1):
    """An axis-aligned rectangle as a ``TL, TR, BR, BL`` candidate."""
    return {"corners": [
        {"x": x0, "y": y0}, {"x": x1, "y": y0}, {"x": x1, "y": y1}, {"x": x0, "y": y1},
    ]}


def corners(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def image_area(candidate) -> float:
    return quad_area([(p["x"], p["y"]) for p in candidate["corners"]])


#: A card-shaped quad that sits inside the frame: 30% of the frame, aspect
#: 63/88 to three decimal places.
BIG_CARD = quad(150, 100, 650, 798)
#: The whole frame. Large, perfectly rectangular, correctly proportioned - and
#: wrong, which is exactly why it needs an explicit penalty.
WHOLE_FRAME = quad(0, 0, FRAME_W, FRAME_H)
#: A small clean fragment.
SLIVER = quad(20, 20, 180, 260)


class TestGeometry:
    def test_area_of_a_rectangle(self):
        assert quad_area(corners(0, 0, 10, 20)) == pytest.approx(200.0)

    def test_area_is_orientation_independent(self):
        clockwise = list(reversed(corners(0, 0, 10, 20)))
        assert quad_area(clockwise) == pytest.approx(200.0)

    def test_aspect_ratio_is_shorter_over_longer(self):
        assert aspect_ratio(corners(0, 0, 630, 880)) == pytest.approx(630 / 880)

    def test_aspect_ratio_of_a_square_is_one(self):
        assert aspect_ratio(corners(0, 0, 50, 50)) == pytest.approx(1.0)

    def test_aspect_ratio_survives_a_rotated_card(self):
        """A card turned sideways still reports the same ratio.

        Both projects normalise with ``min/max``; if this ever returns the
        reciprocal instead, the profile comparison silently inverts.
        """
        landscape = aspect_ratio(corners(0, 0, 880, 630))
        assert math.isclose(landscape, aspect_ratio(corners(0, 0, 630, 880)))

    def test_degenerate_quads_do_not_divide_by_zero(self):
        assert aspect_ratio([(0.0, 0.0)] * 4) == pytest.approx(0.0)
        assert rectangularity([(0.0, 0.0)] * 4) == pytest.approx(0.0)
        assert aspect_ratio(corners(0, 0, 10, 20)[:3]) == pytest.approx(0.0)

    def test_rectangularity_of_a_rectangle_is_one(self):
        assert rectangularity(corners(0, 0, 100, 200)) == pytest.approx(1.0)

    def test_rectangularity_penalises_shear(self):
        """It measures skew, not thinness - a straight sliver still fills its box.

        This is the arithmetic equivalent of the engine's ``fillScore``: quad
        area over the area of the rectangle its own side lengths imply. A
        sheared quad loses to a square; a thin but *straight* quad does not,
        which is why thinness has to be caught by the area term instead.
        """
        square = rectangularity(corners(0, 0, 100, 100))
        trapezoid = rectangularity([(0.0, 0.0), (100.0, 0.0), (150.0, 100.0), (50.0, 100.0)])
        assert square == pytest.approx(1.0)
        assert trapezoid < square
        assert trapezoid < 0.95

    def test_convexity_rejects_a_self_intersecting_order(self):
        bowtie = [(0.0, 0.0), (10.0, 10.0), (10.0, 0.0), (0.0, 10.0)]
        assert not convexity(bowtie)

    def test_convexity_accepts_both_windings(self):
        assert convexity(corners(0, 0, 10, 10))
        assert convexity(list(reversed(corners(0, 0, 10, 10))))

    def test_convexity_rejects_the_wrong_number_of_corners(self):
        assert not convexity(corners(0, 0, 10, 10)[:3])


class TestBorderFraction:
    def test_a_card_inside_the_frame_touches_nothing(self):
        assert border_fraction(corners(150, 100, 900, 800), FRAME_W, FRAME_H) == 0.0

    def test_the_whole_frame_touches_every_corner(self):
        assert border_fraction(corners(0, 0, FRAME_W, FRAME_H), FRAME_W, FRAME_H) == 1.0

    def test_it_counts_corners_not_sides(self):
        # Top edge on the border, the rest inside: two of four corners.
        assert border_fraction(corners(100, 0, 900, 800), FRAME_W, FRAME_H) == 0.5

    def test_the_margin_scales_with_the_frame(self):
        """A tight crop can leave the card a pixel or two off the edge.

        The margin is a percentage of the short side, so small frames get a
        proportionally small allowance rather than the same pixel budget.
        """
        near_edge = corners(5, 5, 50, 80)
        assert border_fraction(near_edge, 2000, 3000) == 0.75   # 20 px margin
        assert border_fraction(near_edge, 60, 90) == 0.0        # 1 px floor


class TestScoring:
    def test_a_card_inside_the_frame_is_accepted(self):
        score = DEFAULT_POLICY.score(BIG_CARD, frame_width=FRAME_W,
                                     frame_height=FRAME_H,
                                     target_ratio=target_ratio_for("pokemon"))
        assert 0.0 < score <= 1.0

    def test_rejection_is_a_negative_score_not_a_low_one(self):
        """``-1.0`` means \"not a candidate\"; scores themselves are >= 0.

        Conflating the two would let a rejected fragment win whenever every
        candidate was rejected.
        """
        tiny = quad(10, 10, 30, 40)
        assert DEFAULT_POLICY.score(tiny, frame_width=FRAME_W, frame_height=FRAME_H,
                                    target_ratio=target_ratio_for("pokemon")) == -1.0

    def test_wrong_aspect_ratio_scores_lower_than_a_correct_one(self):
        right = quad(150, 100, 700, 882)          # ~88/63
        wrong = quad(150, 100, 900, 300)          # a wide letterbox
        target = target_ratio_for("pokemon")
        assert (DEFAULT_POLICY.score(right, frame_width=FRAME_W, frame_height=FRAME_H,
                                     target_ratio=target)
                > DEFAULT_POLICY.score(wrong, frame_width=FRAME_W, frame_height=FRAME_H,
                                       target_ratio=target))

    def test_a_frame_sized_quad_scores_below_a_card(self):
        """The bug this policy was written for.

        Covering the whole frame maximises the area term and scores 1.0 on
        rectangularity, so without the border penalty it wins outright.
        """
        target = target_ratio_for("pokemon")
        card = DEFAULT_POLICY.score(BIG_CARD, frame_width=FRAME_W, frame_height=FRAME_H,
                                    target_ratio=target)
        frame = DEFAULT_POLICY.score(WHOLE_FRAME, frame_width=FRAME_W, frame_height=FRAME_H,
                                     target_ratio=target)
        assert frame < card

    def test_removing_the_border_term_flips_that_ordering(self):
        """Pins *why* ``border_weight`` exists, not just that it does."""
        policy = SelectionPolicy(border_weight=0.0)
        target = target_ratio_for("pokemon")
        card = policy.score(BIG_CARD, frame_width=FRAME_W, frame_height=FRAME_H,
                            target_ratio=target)
        frame = policy.score(WHOLE_FRAME, frame_width=FRAME_W, frame_height=FRAME_H,
                             target_ratio=target)
        assert frame > card

    def test_min_area_ratio_boundary(self):
        policy = SelectionPolicy(min_area_ratio=0.10)
        target = target_ratio_for("pokemon")
        # Exactly 10% of the frame: not below the bound, so accepted.
        edge = quad(0, 0, FRAME_W, FRAME_H * 0.10)
        assert policy.score(edge, frame_width=FRAME_W, frame_height=FRAME_H,
                            target_ratio=target) >= 0.0
        below = quad(0, 0, FRAME_W, FRAME_H * 0.09)
        assert policy.score(below, frame_width=FRAME_W, frame_height=FRAME_H,
                            target_ratio=target) == -1.0

    def test_a_non_convex_candidate_is_rejected(self):
        bowtie = {"corners": [{"x": 0, "y": 0}, {"x": 600, "y": 600},
                              {"x": 600, "y": 0}, {"x": 0, "y": 600}]}
        assert DEFAULT_POLICY.score(bowtie, frame_width=FRAME_W, frame_height=FRAME_H,
                                    target_ratio=target_ratio_for("pokemon")) == -1.0

    def test_a_malformed_candidate_is_rejected(self):
        assert DEFAULT_POLICY.score({"corners": []}, frame_width=FRAME_W,
                                    frame_height=FRAME_H,
                                    target_ratio=target_ratio_for("pokemon")) == -1.0
        assert DEFAULT_POLICY.score({}, frame_width=FRAME_W, frame_height=FRAME_H,
                                    target_ratio=target_ratio_for("pokemon")) == -1.0


class TestSelect:
    def test_it_prefers_the_card_over_the_frame(self):
        chosen, score = select([WHOLE_FRAME, BIG_CARD], frame_width=FRAME_W,
                               frame_height=FRAME_H, profile="pokemon")
        assert chosen is BIG_CARD
        assert score > 0.0

    def test_it_prefers_the_card_over_a_fragment(self):
        chosen, _ = select([SLIVER, BIG_CARD], frame_width=FRAME_W,
                            frame_height=FRAME_H, profile="pokemon")
        assert chosen is BIG_CARD

    def test_order_does_not_matter(self):
        forward, _ = select([SLIVER, BIG_CARD, WHOLE_FRAME], frame_width=FRAME_W,
                            frame_height=FRAME_H, profile="pokemon")
        backward, _ = select([WHOLE_FRAME, BIG_CARD, SLIVER], frame_width=FRAME_W,
                             frame_height=FRAME_H, profile="pokemon")
        assert forward is backward is BIG_CARD

    def test_no_candidates_means_no_answer(self):
        chosen, score = select([], frame_width=FRAME_W, frame_height=FRAME_H)
        assert chosen is None
        assert score == -math.inf

    def test_all_rejected_means_no_answer(self):
        """Not \"the least bad fragment\".

        ``-1.0`` is a rejection; the first rejected candidate used to be
        returned as if it had scored well.
        """
        tiny = quad(10, 10, 40, 60)
        chosen, score = select([tiny], frame_width=FRAME_W, frame_height=FRAME_H,
                               profile="pokemon")
        assert chosen is None
        assert score == -math.inf

    def test_a_degenerate_frame_does_not_raise(self):
        chosen, _ = select([BIG_CARD], frame_width=0, frame_height=0, profile="pokemon")
        assert chosen is None


class TestProfileRatios:
    def test_every_bridge_profile_has_a_ratio(self):
        from cardvault_vision.integrations.image_engine import PROFILE_NAMES

        assert set(PROFILE_NAMES) == set(PROFILE_RATIOS)

    def test_target_ratio_is_always_normalised(self):
        for profile in PROFILE_RATIOS:
            ratio = target_ratio_for(profile)
            assert 0.0 < ratio <= 1.0

    def test_trading_card_ratio_is_the_reciprocal_of_the_layout_aspect(self):
        """CardVault Vision stores height/width; the ratio here is min/max.

        Both reduce to 63/88, so a caller can move between ``CardLayout.aspect``
        and this table without a silent inversion.
        """
        from cardvault_vision.layouts.resolver import CardLayout

        layout_aspect = CardLayout(card_type="POKEMON").aspect
        assert target_ratio_for("pokemon") == pytest.approx(1.0 / layout_aspect)

    def test_unknown_profile_falls_back_to_the_standard_card(self):
        assert target_ratio_for(None) == target_ratio_for("standard-trading-card")
        assert target_ratio_for("not-a-profile") == target_ratio_for("standard-trading-card")

    def test_yu_gi_oh_is_taller_than_a_pokemon_card(self):
        """86/59 is a narrower card than 88/63, so its min/max ratio is smaller."""
        assert target_ratio_for("yu-gi-oh") < target_ratio_for("pokemon")


class TestEdgePresets:
    def test_the_default_single_pass_is_the_first_preset(self):
        assert DEFAULT_ENGINE_OPTIONS is EDGE_PRESETS[0]

    def test_presets_stay_in_the_measured_usable_band(self):
        """Outside roughly 12-24 the Sobel mask stops closing the card outline.

        Measured on 40 synthetic captures: 8 admits noise edges, 48 and above
        lose the card entirely, both far worse than 12-24 with radius 3. This
        test is a tripwire for someone \"simplifying\" the presets to one value
        outside the band, not a claim that these three are optimal.
        """
        for preset in EDGE_PRESETS:
            assert 8 <= preset["edgeThreshold"] <= 32
            assert preset["morphologyRadius"] >= 1

    def test_thresholds_are_distinct(self):
        thresholds = [preset["edgeThreshold"] for preset in EDGE_PRESETS]
        assert len(set(thresholds)) == len(thresholds)


class TestDetectPlumbing:
    """``detect`` with the bridge mocked out: the pooling/annotation contract."""

    def _patch(self, monkeypatch, results):
        from cardvault_vision.integrations import image_engine, selection

        calls = []

        def fake_align(image, *, profile=None, options=None, engine=None, **kwargs):
            calls.append({"profile": profile, "options": options})
            return results[len(calls) - 1]

        monkeypatch.setattr(image_engine, "align", fake_align)
        # ``detect`` imports align lazily from this module.
        assert selection is not None
        return calls

    def test_it_runs_every_preset_and_tags_candidates_with_their_source(self, monkeypatch):
        import numpy as np

        from cardvault_vision.integrations.image_engine import AlignmentResult

        results = [
            AlignmentResult(status="ALIGNED", corners=None, confidence=None,
                            candidates=[{**SLIVER}], total_ms=1.0, engine_path="/e"),
            AlignmentResult(status="LOW_CONFIDENCE", corners=None, confidence=None,
                            candidates=[{**BIG_CARD}], total_ms=1.0, engine_path="/e"),
        ]
        def fake_align(image, *, profile=None, options=None, **kw):
            return results[0] if options == EDGE_PRESETS[0] else results[1]

        monkeypatch.setattr("cardvault_vision.integrations.image_engine.align",
                            fake_align)
        detection = detect(np.zeros((FRAME_H, FRAME_W, 3), np.uint8), profile="pokemon",
                           presets=EDGE_PRESETS[:2], engine="/e")

        assert len(detection.passes) == 2
        assert {candidate["pass"] for candidate in detection.candidates} == {0, 1}
        assert detection.candidates[0]["preset"] == EDGE_PRESETS[0]
        assert detection.chosen is detection.candidates[1]      # the big card
        assert detection.status == "ALIGNED"
        assert detection.aligned

    def test_an_explicit_options_argument_forces_a_single_pass(self, monkeypatch):
        import numpy as np

        from cardvault_vision.integrations.image_engine import AlignmentResult

        seen = []

        def fake_align(image, *, profile=None, options=None, **kw):
            seen.append(options)
            return AlignmentResult(status="ALIGNED", corners=None, confidence=None,
                                   candidates=[{**BIG_CARD}], total_ms=1.0,
                                   engine_path="/e")

        monkeypatch.setattr("cardvault_vision.integrations.image_engine.align", fake_align)
        detect(np.zeros((FRAME_H, FRAME_W, 3), np.uint8), profile="pokemon",
               options={"edgeThreshold": 48})
        assert seen == [{"edgeThreshold": 48}]

    def test_nothing_acceptable_reports_the_engine_status(self, monkeypatch):
        import numpy as np

        from cardvault_vision.integrations.image_engine import AlignmentResult

        monkeypatch.setattr(
            "cardvault_vision.integrations.image_engine.align",
            lambda image, *, profile=None, options=None, **kw: AlignmentResult(
                status="CARD_NOT_FOUND", corners=None, confidence=None,
                candidates=[{**SLIVER}], total_ms=1.0, engine_path="/e"),
        )
        detection = detect(np.zeros((FRAME_H, FRAME_W, 3), np.uint8), profile="pokemon",
                           presets=EDGE_PRESETS[:1])
        assert detection.chosen is None
        assert detection.corners is None
        assert not detection.aligned
        assert detection.status == "CARD_NOT_FOUND"
