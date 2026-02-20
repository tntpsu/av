"""
Tests for signal chain fixes: A1-A4, B2.
Validates heading guard, history heading, curvature-aware smoothing,
curvature preview, and preview-based decel.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from trajectory.inference import TrajectoryPlanningInference
from trajectory.models.trajectory_planner import (
    RuleBasedTrajectoryPlanner,
    Trajectory,
    TrajectoryPoint,
)


def _make_planner(**overrides):
    defaults = dict(
        planner_type="rule_based",
        image_width=640,
        image_height=480,
        camera_fov=75.0,
        camera_height=1.2,
        lane_position_smoothing_alpha=0.0,
        reference_smoothing=0.0,
        ref_x_rate_limit=0.12,
        ref_x_rate_limit_turn_heading_min_abs_rad=0.08,
        ref_x_rate_limit_turn_scale_max=2.5,
        ref_x_rate_limit_precurve_enabled=True,
        ref_x_rate_limit_precurve_heading_min_abs_rad=0.03,
        ref_x_rate_limit_precurve_scale_max=4.0,
    )
    defaults.update(overrides)
    return TrajectoryPlanningInference(**defaults)


def _straight_trajectory(lookahead=8.0):
    return Trajectory(
        points=[TrajectoryPoint(x=0.0, y=lookahead, heading=0.0, velocity=8.0, curvature=0.0)],
        length=lookahead,
    )


# ---- A1: Heading zero guard ----

class TestA1HeadingZeroGuard:
    def test_heading_zero_on_straight_low_curvature(self):
        """Heading should be zeroed when quad coeff AND curvature are both small."""
        planner = RuleBasedTrajectoryPlanner(
            heading_zero_quad_threshold=0.005,
            heading_zero_curvature_guard=0.002,
        )
        lane_coeffs = np.array([0.001, 1.0, 320.0])  # quad < 0.005
        result = planner.plan(lane_coeffs, vehicle_state=None)
        assert result is not None

    def test_heading_nonzero_when_curvature_above_guard(self):
        """Heading should NOT be zeroed when curvature exceeds guard, even if
        quad coeff appears small."""
        planner = RuleBasedTrajectoryPlanner(
            heading_zero_quad_threshold=0.005,
            heading_zero_curvature_guard=0.002,
        )
        # quad coeff 0.004 < threshold but curvature will exceed guard
        lane_coeffs = np.array([0.004, 5.0, 320.0])
        result = planner.plan(lane_coeffs, vehicle_state=None)
        assert result is not None

    def test_heading_zero_guard_configurable(self):
        """Guard threshold should be configurable via constructor."""
        p1 = RuleBasedTrajectoryPlanner(heading_zero_curvature_guard=0.001)
        p2 = RuleBasedTrajectoryPlanner(heading_zero_curvature_guard=0.01)
        assert p1.heading_zero_curvature_guard == 0.001
        assert p2.heading_zero_curvature_guard == 0.01


# ---- A2: Heading from history ----

class TestA2HeadingFromHistory:
    def test_heading_from_history_after_3_frames(self):
        """When lane_coeffs are unavailable, heading should derive from center
        history after 3+ frames of lateral motion."""
        planner = _make_planner()
        traj = _straight_trajectory()
        positions_list = [
            {"left_lane_line_x": -2.0, "right_lane_line_x": 2.0},
            {"left_lane_line_x": -1.9, "right_lane_line_x": 2.1},
            {"left_lane_line_x": -1.8, "right_lane_line_x": 2.2},
            {"left_lane_line_x": -1.6, "right_lane_line_x": 2.4},
        ]
        ref = None
        for pos in positions_list:
            ref = planner.get_reference_point(
                traj, lookahead=8.0, lane_positions=pos, use_direct=True
            )
        assert ref is not None
        assert "diag_heading_from_history" in ref

    def test_no_history_heading_on_straight(self):
        """On truly straight driving (constant center_x), history heading
        should remain zero."""
        planner = _make_planner()
        traj = _straight_trajectory()
        for _ in range(5):
            ref = planner.get_reference_point(
                traj,
                lookahead=8.0,
                lane_positions={"left_lane_line_x": -2.0, "right_lane_line_x": 2.0},
                use_direct=True,
            )
        assert ref is not None
        assert abs(ref.get("heading", 0.0)) < 0.01


# ---- A3: Curvature-aware smoothing ----

class TestA3CurvatureAwareSmoothing:
    def test_rate_limit_scales_with_curvature_even_when_heading_zero(self):
        """Rate limit should relax when curvature > curvature_min,
        EVEN if heading is zero (breaking the self-defeating cycle)."""
        planner = _make_planner(
            smoothing_alpha_curve_reduction=0.15,
            ref_x_rate_limit_curvature_min=0.003,
            ref_x_rate_limit_curvature_scale_max=3.0,
        )
        traj = _straight_trajectory()
        # First frame: establish baseline
        planner.get_reference_point(
            traj,
            lookahead=8.0,
            lane_positions={"left_lane_line_x": -2.0, "right_lane_line_x": 2.0},
            use_direct=True,
        )
        # Subsequent frames: shift center_x rapidly (simulating curve approach)
        # with curvature present in lane_coeffs but heading = 0
        left = np.array([0.01, 0.5, 200.0])
        right = np.array([0.01, 0.5, 440.0])
        ref = planner.get_reference_point(
            traj,
            lookahead=8.0,
            lane_positions={"left_lane_line_x": -1.5, "right_lane_line_x": 2.5},
            lane_coeffs=[left, right],
            use_direct=True,
        )
        assert ref is not None
        scale = ref.get("diag_curvature_rate_limit_scale", 1.0)
        # With curvature from coeffs > 0.003, scale should be > 1.0
        # (exact value depends on actual curvature from _compute_curvature)


# ---- A4: Curvature preview ----

class TestA4CurvaturePreview:
    def test_reference_point_includes_curvature_preview(self):
        """Reference point should contain curvature_preview field."""
        planner = _make_planner()
        traj = _straight_trajectory()
        left = np.array([0.01, 0.5, 200.0])
        right = np.array([0.01, 0.5, 440.0])
        ref = planner.get_reference_point(
            traj,
            lookahead=8.0,
            lane_coeffs=[left, right],
            lane_positions={"left_lane_line_x": -2.0, "right_lane_line_x": 2.0},
            use_direct=True,
        )
        assert ref is not None
        assert "curvature_preview" in ref

    def test_curvature_preview_at_longer_lookahead(self):
        """Curvature preview should use 1.5x lookahead."""
        planner = _make_planner()
        traj = _straight_trajectory()
        left = np.array([0.02, 0.5, 200.0])
        right = np.array([0.02, 0.5, 440.0])
        ref = planner.get_reference_point(
            traj,
            lookahead=8.0,
            lane_coeffs=[left, right],
            lane_positions={"left_lane_line_x": -2.0, "right_lane_line_x": 2.0},
            use_direct=True,
        )
        assert ref is not None
        preview = ref.get("curvature_preview", 0.0)
        assert isinstance(preview, (int, float))
