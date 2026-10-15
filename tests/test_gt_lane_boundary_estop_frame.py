"""GT lane-boundary emergency-stop frame contract (T-SAFETY-ESTOP-FRAME, 2026-10-15).

The safety e-stop used Unity's `groundTruthLeft/RightLaneLineX`, which are computed AT THE LOOKAHEAD (6–8 m
ahead, for perception/visualizer alignment), and tested them against an at-car-scale 0.05 m margin. On tight
curves the lane that far ahead is displaced ~½·Ld²·κ plus heading — 1.2–1.9 m at R15/R40 — so a car centred
AT THE CAR read a lane line past the margin and the e-stop fired falsely: once per run, same spot, in all twelve
LMPC-on-tight-curve runs of 2026-10-15 (hairpin at-car −0.2 m, left line +0.10 m; s_loop at-car −0.35 m,
left line +0.10 m; max at-car over any run 0.73 m, never out of lane). Pure Pursuit escaped it only by running
wide. This is the third metric-frame defect (lateral scoring frame 09-28, ACC radar range 09-23).

These vectors are the recorded values. The at_car frame must NOT fire on them; the legacy lookahead frame
must (documents the defect); a genuine departure must fire in the at_car frame.
"""
from __future__ import annotations

import pytest

from av_stack.orchestrator import gt_lane_boundary_offroad

MARGIN = 0.05
FALLBACK_HALF = 1.8

# (label, at_car_cross_track, left_line_x, right_line_x) — recorded at the e-stop frame
FALSE_FIRES = [
    ("hairpin_001217", -0.206, 0.14, 3.74),
    ("hairpin_001325", -0.18, 0.07, 3.67),
    ("hairpin_001433", -0.17, 0.09, 3.69),
    ("sloop_000055", -0.36, 0.17, 3.77),
    ("sloop_000203", -0.30, 0.11, 3.71),
    ("sloop_000419", -0.38, 0.10, 3.70),
]


class TestRecordedFalseFires:
    @pytest.mark.parametrize("label,d,left,right", FALSE_FIRES)
    def test_at_car_frame_does_not_fire_on_a_centred_car(self, label, d, left, right):
        ol, orr, det = gt_lane_boundary_offroad('at_car', d, left, right, MARGIN, FALLBACK_HALF)
        assert not ol and not orr, f"{label}: centred car ({d:+.2f} m) must not e-stop"
        assert det['half_width_m'] == pytest.approx(0.5 * (right - left), abs=1e-9)   # 3.6 m lane → 1.8

    @pytest.mark.parametrize("label,d,left,right", FALSE_FIRES)
    def test_legacy_lookahead_frame_fired_on_these_same_frames(self, label, d, left, right):
        """The defect on record: the lookahead line position crosses the 0.05 m margin while the car is centred."""
        ol, orr, _ = gt_lane_boundary_offroad('lookahead', d, left, right, MARGIN, FALLBACK_HALF)
        assert ol and not orr, f"{label}: legacy frame should have reported 'offroad left'"


class TestGenuineDepartureStillFires:
    def test_car_past_the_left_edge_fires_left(self):
        ol, orr, _ = gt_lane_boundary_offroad('at_car', -1.85, -1.8, 1.8, MARGIN, FALLBACK_HALF)
        assert ol and not orr

    def test_car_past_the_right_edge_fires_right(self):
        ol, orr, _ = gt_lane_boundary_offroad('at_car', +1.85, -1.8, 1.8, MARGIN, FALLBACK_HALF)
        assert orr and not ol

    def test_pure_pursuit_running_wide_at_1p37_is_still_in_lane(self):
        """PP's worst hairpin excursion (max at-car 1.37 m) is inside a 1.8 m half-width → no e-stop, correctly."""
        ol, orr, _ = gt_lane_boundary_offroad('at_car', +1.37, -1.8, 1.8, MARGIN, FALLBACK_HALF)
        assert not ol and not orr

    def test_margin_is_applied_inside_the_edge(self):
        half = 1.8
        just_in = half - MARGIN - 1e-3
        just_out = half - MARGIN + 1e-3
        assert gt_lane_boundary_offroad('at_car', just_in, -1.8, 1.8, MARGIN, FALLBACK_HALF)[1] is False
        assert gt_lane_boundary_offroad('at_car', just_out, -1.8, 1.8, MARGIN, FALLBACK_HALF)[1] is True


class TestHalfWidthDerivation:
    def test_uses_line_spacing_even_when_offsets_are_displaced(self):
        # Displaced lookahead lines (centre at +1.9) but spacing 3.6 → half 1.8, and the at-car test uses that.
        _, _, det = gt_lane_boundary_offroad('at_car', -0.2, 0.10, 3.70, MARGIN, FALLBACK_HALF)
        assert det['half_width_m'] == pytest.approx(1.8, abs=1e-9)

    def test_falls_back_when_lines_missing_or_insane(self):
        _, _, det = gt_lane_boundary_offroad('at_car', -0.2, None, None, MARGIN, FALLBACK_HALF)
        assert det['half_width_m'] == FALLBACK_HALF
        _, _, det = gt_lane_boundary_offroad('at_car', -0.2, 0.0, 25.0, MARGIN, FALLBACK_HALF)   # 25 m "lane"
        assert det['half_width_m'] == FALLBACK_HALF

    def test_missing_at_car_truth_never_fires(self):
        assert gt_lane_boundary_offroad('at_car', None, 0.10, 3.70, MARGIN, FALLBACK_HALF)[:2] == (False, False)
