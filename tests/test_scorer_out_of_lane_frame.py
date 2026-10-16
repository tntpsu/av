"""Scorer out-of-lane frame contract (2026-10-15, fifth metric-frame defect).

The scorer declared "out of lane" when the car (x = 0) was not between `ground_truth/left_lane_line_x` and
`right_lane_line_x`. Unity computes those lines AT THE LOOKAHEAD (6 m ahead; the T-SAFETY-ESTOP-FRAME finding), so on a
tight curve they are displaced ~½·Ld²·κ toward the inside — 1.17 m on hairpin_15 (R15) — and a car tracking the centre
reads as outside while a car running wide on the other side reads as inside. Phase 1e arm C (at-car RMSE 0.089 m,
max 0.49 m, half-lane 1.80 m) took a −25 Safety deduction on every run; Pure Pursuit (max 1.37 m, wide) took none.

`scoring_registry.OUT_OF_LANE_GT_FRAME = "at_car"` tests the car's own road-frame cross-track against the lane half-width
derived from the line spacing (fallback 1.8 m); "lookahead" is the legacy behaviour.
"""
from __future__ import annotations

import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
from drive_summary_core import gt_out_of_lane_mask  # noqa: E402
import scoring_registry  # noqa: E402

# Recorded hairpin false-fire vector (tests/test_gt_lane_boundary_estop_frame.py): lookahead lines at +0.14 / +3.74 with the
# car 0.2 m LEFT of centre — the car at x=0 reads "left of the left line" although it is 1.6 m inside the lane.
LEFT_FALSE, RIGHT_FALSE = 0.14, 3.74


def _mask(frame, left, right, d):
    return gt_out_of_lane_mask(frame, np.array([left]), np.array([right]), None if d is None else np.array([d]), 20.0, 1.8)


class TestLookaheadDisplacedLines:
    def test_legacy_frame_flags_a_centred_car_on_a_tight_curve(self):
        mask, err, half = _mask('lookahead', LEFT_FALSE, RIGHT_FALSE, -0.2)
        assert bool(mask[0]) and err[0] == pytest.approx(0.14)                  # the defect on record

    def test_at_car_frame_keeps_the_centred_car_in_lane(self):
        mask, err, half = _mask('at_car', LEFT_FALSE, RIGHT_FALSE, -0.2)
        assert not bool(mask[0]) and err[0] == 0.0 and half == pytest.approx(1.8)

    def test_half_width_comes_from_the_line_spacing(self):
        _, _, half = _mask('at_car', -1.5, 1.5, 0.0)
        assert half == pytest.approx(1.5)


class TestGenuineDepartureStillCounts:
    @pytest.mark.parametrize("d", [+1.95, -1.95])
    def test_car_centre_past_the_edge_is_out(self, d):
        mask, err, _ = _mask('at_car', -1.8, 1.8, d)
        assert bool(mask[0]) and err[0] == pytest.approx(0.15)

    def test_pure_pursuit_wide_at_1p37_is_in_lane_in_both_frames(self):
        # PP's worst hairpin excursion sat inside the (undisplaced) lane on the straight-frame lines
        assert not bool(_mask('at_car', -1.8, 1.8, 1.37)[0][0])


class TestFallbacks:
    def test_missing_at_car_truth_falls_back_to_legacy(self):
        mask_l, _, _ = _mask('lookahead', LEFT_FALSE, RIGHT_FALSE, None)
        mask_a, _, _ = _mask('at_car', LEFT_FALSE, RIGHT_FALSE, None)
        assert bool(mask_l[0]) == bool(mask_a[0])

    def test_implausible_lines_use_the_fallback_half_width(self):
        _, _, half = _mask('at_car', 0.0, 25.0, 0.0)
        assert half == pytest.approx(1.8)

    def test_registry_default_is_at_car(self):
        assert scoring_registry.OUT_OF_LANE_GT_FRAME == "at_car"
