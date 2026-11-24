"""Corner and overlap metrics for grading a card aligner.

These are the numbers behind every alignment claim, so they are tested against
shapes whose answers can be computed by hand - not against the engine, which
would only prove the two agree with each other.
"""

from __future__ import annotations

import math

import pytest

from cardvault_vision.evaluation.alignment import (
    AlignmentCase,
    AlignmentReport,
    best_rotation,
    corner_errors,
    normalised_corner_error,
    quad_area,
    quad_diagonal,
    quad_iou,
)

UNIT_SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


class TestQuadGeometry:
    def test_area_of_a_square(self):
        assert quad_area(UNIT_SQUARE) == pytest.approx(100.0)

    def test_area_is_order_independent(self):
        """A detector may report corners clockwise or anticlockwise."""
        reversed_square = list(reversed(UNIT_SQUARE))
        assert quad_area(reversed_square) == pytest.approx(100.0)

    def test_diagonal_of_a_square(self):
        assert quad_diagonal(UNIT_SQUARE) == pytest.approx(math.sqrt(200))

    def test_degenerate_quad_rejected(self):
        with pytest.raises(ValueError):
            quad_diagonal([(0.0, 0.0), (1.0, 1.0)])

    def test_corner_errors_are_per_corner_distances(self):
        shifted = [(x + 3.0, y + 4.0) for x, y in UNIT_SQUARE]
        assert corner_errors(shifted, UNIT_SQUARE) == pytest.approx([5.0] * 4)

    def test_normalised_error_is_scale_free(self):
        """The same relative error must score the same at any card size."""
        small = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        big = [(x * 100, y * 100) for x, y in small]
        small_off = [(x + 0.1, y) for x, y in small]
        big_off = [(x + 10.0, y) for x, y in big]
        assert normalised_corner_error(small_off, small) == pytest.approx(
            normalised_corner_error(big_off, big)
        )

    def test_identical_quads_have_zero_error(self):
        assert normalised_corner_error(UNIT_SQUARE, UNIT_SQUARE) == 0.0


class TestQuadIoU:
    def test_identical_quads(self):
        assert quad_iou(UNIT_SQUARE, UNIT_SQUARE) == pytest.approx(1.0)

    def test_disjoint_quads(self):
        far = [(100.0, 100.0), (110.0, 100.0), (110.0, 110.0), (100.0, 110.0)]
        assert quad_iou(far, UNIT_SQUARE) == pytest.approx(0.0)

    def test_half_overlap_is_one_third(self):
        """A square shifted by half its width overlaps by 50 of 150 units."""
        shifted = [(5.0, 0.0), (15.0, 0.0), (15.0, 10.0), (5.0, 10.0)]
        assert quad_iou(shifted, UNIT_SQUARE) == pytest.approx(50.0 / 150.0)

    def test_contained_quad(self):
        """A smaller square inside gives IoU = area ratio."""
        inner = [(2.5, 2.5), (7.5, 2.5), (7.5, 7.5), (2.5, 7.5)]
        assert quad_iou(inner, UNIT_SQUARE) == pytest.approx(25.0 / 100.0)

    def test_iou_is_bounded(self):
        sliver = [(0.0, 0.0), (100.0, 0.0), (100.0, 0.01), (0.0, 0.01)]
        value = quad_iou(sliver, UNIT_SQUARE)
        assert 0.0 <= value <= 1.0


class TestRotationDetection:
    def test_matching_order_needs_no_shift(self):
        shift, error = best_rotation(UNIT_SQUARE, UNIT_SQUARE)
        assert shift == 0
        assert error == pytest.approx(0.0)

    def test_a_shifted_corner_order_is_reported(self):
        """TL,TR,BR,BL reported as TR,BR,BL,TL is a convention error.

        The shift is *reported* rather than silently applied: a non-zero shift
        means one side changed its contract, and hiding it would let a caller
        crop from the wrong corners forever.
        """
        rotated = [UNIT_SQUARE[1], UNIT_SQUARE[2], UNIT_SQUARE[3], UNIT_SQUARE[0]]
        shift, error = best_rotation(rotated, UNIT_SQUARE)
        assert shift == 3          # rotating back by 3 restores order
        assert error == pytest.approx(0.0)


class TestAlignmentReport:
    def _case(self, cer, iou=None, oracle=None, status="ALIGNED", profile="pokemon"):
        return AlignmentCase(image_id="x", status=status, corner_error=cer, iou=iou,
                             oracle_corner_error=oracle, profile=profile)

    def test_detection_rate_counts_only_graded_cases(self):
        report = AlignmentReport(cases=[
            self._case(0.01), self._case(0.30), self._case(None, status="CARD_NOT_FOUND"),
        ])
        assert report.detection_rate == pytest.approx(2 / 3)

    def test_within_tolerance_uses_the_configured_bound(self):
        report = AlignmentReport(cases=[self._case(0.01), self._case(0.03)],
                                 tolerance=0.02)
        assert report.within_tolerance == 1

    def test_median_corner_error_ignores_undetected(self):
        report = AlignmentReport(cases=[self._case(0.10), self._case(0.20),
                                        self._case(None, status="CARD_NOT_FOUND")])
        assert report.median_corner_error == pytest.approx(0.15)

    def test_oracle_error_exposes_ranking_failures(self):
        """Top-1 error 0.47 but the best candidate was 0.30: a ranking problem.

        This is the distinction that mattered in practice - the engine proposed
        the true card and then ranked a 3%-area sliver above it.
        """
        report = AlignmentReport(cases=[self._case(0.47, oracle=0.30)])
        assert report.median_oracle_corner_error == pytest.approx(0.30)
        assert report.median_corner_error == pytest.approx(0.47)
        assert report.cases[0].ranked_wrong

    def test_empty_report_does_not_divide_by_zero(self):
        report = AlignmentReport()
        assert math.isnan(report.detection_rate)
        assert math.isnan(report.median_corner_error)
        assert report.summary()["count"] == 0
