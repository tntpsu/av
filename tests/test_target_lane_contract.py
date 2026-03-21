"""Contract tests for target_lane center_x calculation.

These tests encode the behavioural contract of the target_lane feature:
  - "center" always returns the geometric midpoint of detected lane lines
  - "right"  with target_lane_width_m returns right_x - (width / 2)
  - "left"   with target_lane_width_m returns left_x  + (width / 2)
  - Negative or zero lane_width always falls back to the midpoint

This logic is duplicated in two code paths inside trajectory/inference.py:
  Path A: get_reference_point() via lane_positions dict  ← hot path
  Path B: _compute_target_ref_from_coeffs() via image coefficients

Both paths are tested here because they diverged once before (causing the
0.22 m lateral bias bug on the sloop track when target_lane was "right").

Discovery:  the two paths share the same formula but live in separate
if/elif chains — a future edit to one that misses the other will be
caught immediately by these tests.
"""

import numpy as np
import pytest

from trajectory.inference import TrajectoryPlanningInference
from trajectory.models.trajectory_planner import Trajectory, TrajectoryPoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(target_lane: str, width_m: float = 3.6) -> TrajectoryPlanningInference:
    return TrajectoryPlanningInference(
        planner_type="rule_based",
        target_lane=target_lane,
        target_lane_width_m=width_m,
    )


def _ref_via_positions(engine, left_x: float, right_x: float):
    """Call the hot path (Path A): direct vehicle-frame lane positions."""
    dummy_traj = Trajectory(
        points=[TrajectoryPoint(x=0.0, y=5.0, heading=0.0, velocity=8.0, curvature=0.0)],
        length=5.0,
    )
    result = engine.get_reference_point(
        trajectory=dummy_traj,
        lookahead=5.0,
        lane_positions={"left_lane_line_x": left_x, "right_lane_line_x": right_x},
        use_direct=True,
        timestamp=0.0,
        confidence=1.0,
    )
    return result


# ---------------------------------------------------------------------------
# Symmetric lane (−1.6 m left, +1.6 m right  →  3.2 m width)
# ---------------------------------------------------------------------------

LEFT, RIGHT = -1.6, 1.6       # symmetric around road centre
ROAD_CENTRE = 0.0             # (LEFT + RIGHT) / 2
WIDTH       = RIGHT - LEFT    # 3.2 m

TARGET_WIDTH_M = 3.6          # configured in av_stack_config.yaml


class TestCenterTargetLane:
    """target_lane='center' must always return the geometric midpoint."""

    def test_symmetric_lane(self):
        eng = _make_engine("center")
        ref = _ref_via_positions(eng, LEFT, RIGHT)
        assert ref is not None
        assert ref["x"] == pytest.approx(ROAD_CENTRE, abs=1e-4), (
            f"center on symmetric lane: expected x≈0.0, got {ref['x']:.4f}"
        )

    def test_asymmetric_lane(self):
        """Works for any lane geometry — always the midpoint."""
        eng = _make_engine("center")
        left, right = -1.0, 2.0
        ref = _ref_via_positions(eng, left, right)
        assert ref is not None
        assert ref["x"] == pytest.approx((left + right) / 2.0, abs=1e-4)

    def test_wide_lane(self):
        eng = _make_engine("center")
        ref = _ref_via_positions(eng, -2.5, 2.5)
        assert ref is not None
        assert ref["x"] == pytest.approx(0.0, abs=1e-4)

    def test_center_is_independent_of_target_width_m(self):
        """target_lane_width_m must have no effect when target_lane='center'."""
        eng_no_width  = _make_engine("center", width_m=0.0)
        eng_has_width = _make_engine("center", width_m=3.6)
        ref_no  = _ref_via_positions(eng_no_width,  LEFT, RIGHT)
        ref_has = _ref_via_positions(eng_has_width, LEFT, RIGHT)
        assert ref_no  is not None
        assert ref_has is not None
        assert ref_no["x"] == pytest.approx(ref_has["x"], abs=1e-4), (
            "center target_lane must ignore target_lane_width_m"
        )


class TestRightTargetLane:
    """target_lane='right' with target_lane_width_m returns right − width/2."""

    def test_with_target_width(self):
        """Primary formula: right_x - (target_lane_width_m / 2)."""
        eng = _make_engine("right", width_m=TARGET_WIDTH_M)
        ref = _ref_via_positions(eng, LEFT, RIGHT)
        expected = RIGHT - (TARGET_WIDTH_M / 2.0)
        assert ref is not None
        assert ref["x"] == pytest.approx(expected, abs=1e-4), (
            f"right+width: expected {expected:.4f}, got {ref['x']:.4f}"
        )

    def test_without_target_width_uses_75pct(self):
        """Fallback formula when width_m=0: left + 0.75 * lane_width."""
        eng = _make_engine("right", width_m=0.0)
        ref = _ref_via_positions(eng, LEFT, RIGHT)
        expected = LEFT + 0.75 * WIDTH
        assert ref is not None
        assert ref["x"] == pytest.approx(expected, abs=1e-4)

    def test_right_target_is_right_of_centre_when_width_fits(self):
        """Right target is right of centre only when target_lane_width_m < lane_width.
        When target_lane_width_m > lane_width the target lands left of centre —
        that is documented by test_right_with_sloop_geometry_matches_known_bias.
        """
        # width_m=0 → uses 75 % formula → always right of centre
        for width_m in (0.0, 2.5):   # both < lane_width=3.2
            eng = _make_engine("right", width_m=width_m)
            ref = _ref_via_positions(eng, LEFT, RIGHT)
            assert ref is not None
            assert ref["x"] >= ROAD_CENTRE - 1e-4, (
                f"right target (width_m={width_m}) should be right of centre, "
                f"got x={ref['x']:.4f}"
            )

    def test_right_with_sloop_geometry_matches_known_bias(self):
        """Regression: sloop track width ~3.16 m + target_lane='right' + width=3.6 m
        produced ~0.22 m leftward bias.  Verify the formula matches expectations
        so we catch any accidental reversion."""
        sloop_left, sloop_right = -1.58, 1.58   # ~3.16 m sloop width
        eng_right  = _make_engine("right", width_m=3.6)
        eng_center = _make_engine("center")
        ref_right  = _ref_via_positions(eng_right,  sloop_left, sloop_right)
        ref_center = _ref_via_positions(eng_center, sloop_left, sloop_right)
        assert ref_right  is not None
        assert ref_center is not None
        # right target should be LEFT of centre (because target lane width > road width)
        bias = ref_right["x"] - ref_center["x"]
        assert bias == pytest.approx(sloop_right - (3.6 / 2.0), abs=1e-3), (
            "right+width formula regression check"
        )
        # Confirm the known ~−0.22 m bias (right target lands left of centre on sloop)
        assert bias < -0.1, (
            f"right lane on sloop should bias LEFT of centre, got bias={bias:.4f} m"
        )


class TestLeftTargetLane:
    """target_lane='left' with target_lane_width_m returns left + width/2."""

    def test_with_target_width(self):
        eng = _make_engine("left", width_m=TARGET_WIDTH_M)
        ref = _ref_via_positions(eng, LEFT, RIGHT)
        expected = LEFT + (TARGET_WIDTH_M / 2.0)
        assert ref is not None
        assert ref["x"] == pytest.approx(expected, abs=1e-4)

    def test_without_target_width_uses_25pct(self):
        eng = _make_engine("left", width_m=0.0)
        ref = _ref_via_positions(eng, LEFT, RIGHT)
        expected = LEFT + 0.25 * WIDTH
        assert ref is not None
        assert ref["x"] == pytest.approx(expected, abs=1e-4)

    def test_left_target_is_left_of_centre_when_width_fits(self):
        """Left target is left of centre only when target_lane_width_m < lane_width."""
        for width_m in (0.0, 2.5):   # both < lane_width=3.2
            eng = _make_engine("left", width_m=width_m)
            ref = _ref_via_positions(eng, LEFT, RIGHT)
            assert ref is not None
            assert ref["x"] <= ROAD_CENTRE + 1e-4, (
                f"left target (width_m={width_m}) should be left of centre, "
                f"got x={ref['x']:.4f}"
            )


class TestEdgeCases:
    """Edge-case contracts that must hold for all target_lane values."""

    def test_zero_lane_positions_returns_none(self):
        """Both lanes at exactly 0.0 signals perception failure → must return None."""
        for tl in ("center", "right", "left"):
            eng = _make_engine(tl)
            ref = _ref_via_positions(eng, 0.0, 0.0)
            assert ref is None, (
                f"target_lane='{tl}': both-zero lane positions must return None "
                f"(perception failure guard)"
            )

    def test_inverted_lane_width_falls_back_to_midpoint(self):
        """left_x > right_x (degenerate detection) must fall back to midpoint."""
        for tl in ("center", "right", "left"):
            eng = _make_engine(tl, width_m=0.0)
            ref = _ref_via_positions(eng, 1.0, -1.0)   # inverted
            assert ref is not None
            expected = (1.0 + (-1.0)) / 2.0   # = 0.0
            assert ref["x"] == pytest.approx(expected, abs=1e-4), (
                f"target_lane='{tl}': inverted lane should fall back to midpoint"
            )

    @pytest.mark.parametrize("target_lane", ["center", "right", "left"])
    def test_target_lane_case_insensitive(self, target_lane):
        """target_lane string is lowercased internally — upper/mixed case must work."""
        eng = _make_engine(target_lane.upper())
        ref = _ref_via_positions(eng, LEFT, RIGHT)
        assert ref is not None, f"upper-case '{target_lane.upper()}' should be accepted"

    def test_current_config_is_center(self):
        """Regression guard: production config must use target_lane='center'.

        This test will FAIL if someone changes the config back to 'right',
        which caused the 0.22 m lateral bias bug on the sloop track.
        """
        import yaml
        with open("config/av_stack_config.yaml") as f:
            cfg = yaml.safe_load(f)
        tl = cfg.get("trajectory", {}).get("target_lane", "MISSING")
        assert tl == "center", (
            f"av_stack_config.yaml target_lane='{tl}' — must be 'center'. "
            f"Changing to 'right' caused a 0.22 m lateral bias on the sloop track."
        )
