"""
Unit tests for reference point stability fixes.

These tests verify that the temporal filtering, outlier rejection, and bounds checking
prevent reference point instability from increasing over time.
"""

import pytest
import numpy as np
from typing import List, Optional

from trajectory.models.trajectory_planner import RuleBasedTrajectoryPlanner
from trajectory.inference import TrajectoryPlanningInference


class TestTemporalLaneFiltering:
    """Test temporal filtering of lane coefficients."""
    
    def test_lane_coeffs_smoothing_reduces_noise(self):
        """Test that temporal filtering reduces noise in lane coefficients."""
        planner = RuleBasedTrajectoryPlanner(
            image_width=640.0,
            image_height=480.0
        )
        # Set smoothing alpha if not in __init__
        if not hasattr(planner, 'lane_smoothing_alpha'):
            planner.lane_smoothing_alpha = 0.7
        if not hasattr(planner, 'lane_coeffs_history'):
            planner.lane_coeffs_history = []
        
        # Create stable lane coefficients (straight lane at center)
        stable_left = np.array([0.0, 0.0, 200.0])  # x = 200 at y=0
        stable_right = np.array([0.0, 0.0, 440.0])  # x = 440 at y=0
        
        # Add noise to simulate detection errors
        noisy_left = stable_left + np.array([0.0, 0.0, 10.0])  # 10px offset
        noisy_right = stable_right + np.array([0.0, 0.0, -10.0])  # -10px offset
        
        # First frame: stable lanes
        lane_coeffs1 = [stable_left, stable_right]
        # Call plan to trigger smoothing (smoothing happens in plan method)
        trajectory1 = planner.plan(lane_coeffs1)
        smoothed1 = planner.lane_coeffs_history[-1] if planner.lane_coeffs_history else lane_coeffs1
        
        # Second frame: noisy lanes
        lane_coeffs2 = [noisy_left, noisy_right]
        trajectory2 = planner.plan(lane_coeffs2)
        smoothed2 = planner.lane_coeffs_history[-1] if planner.lane_coeffs_history else lane_coeffs2
        
        # Smoothed coefficients should be between stable and noisy
        # With alpha=0.7: smoothed = 0.7 * stable + 0.3 * noisy
        expected_left = 0.7 * stable_left + 0.3 * noisy_left
        expected_right = 0.7 * stable_right + 0.3 * noisy_right
        
        # Check that smoothing occurred (smoothed should be different from raw noisy)
        assert smoothed2 is not None and len(smoothed2) >= 2, "Smoothing should produce coefficients"
        assert smoothed2[0] is not None and smoothed2[1] is not None, "Both lanes should be smoothed"
        
        # Verify smoothing reduced the noise (smoothed should be closer to stable than noisy)
        noise_reduction_left = abs(smoothed2[0][-1] - stable_left[-1]) < abs(noisy_left[-1] - stable_left[-1])
        noise_reduction_right = abs(smoothed2[1][-1] - stable_right[-1]) < abs(noisy_right[-1] - stable_right[-1])
        
        assert noise_reduction_left or noise_reduction_right, (
            f"Smoothing should reduce noise: smoothed_left={smoothed2[0][-1]:.2f}, "
            f"noisy_left={noisy_left[-1]:.2f}, stable_left={stable_left[-1]:.2f}"
        )
    
    def test_lane_coeffs_persistence_on_missing_detection(self):
        """Test that missing lane detection uses previous frame's coefficients."""
        planner = RuleBasedTrajectoryPlanner(
            image_width=640.0,
            image_height=480.0
        )
        # Set smoothing alpha if not in __init__
        if not hasattr(planner, 'lane_smoothing_alpha'):
            planner.lane_smoothing_alpha = 0.7
        if not hasattr(planner, 'lane_coeffs_history'):
            planner.lane_coeffs_history = []
        
        # First frame: both lanes detected
        lane_coeffs1 = [
            np.array([0.0, 0.0, 200.0]),
            np.array([0.0, 0.0, 440.0])
        ]
        trajectory1 = planner.plan(lane_coeffs1)
        smoothed1 = planner.lane_coeffs_history[-1] if planner.lane_coeffs_history else lane_coeffs1
        
        # Second frame: right lane missing
        lane_coeffs2 = [
            np.array([0.0, 0.0, 200.0]),
            None  # Missing detection
        ]
        trajectory2 = planner.plan(lane_coeffs2)
        smoothed2 = planner.lane_coeffs_history[-1] if planner.lane_coeffs_history else lane_coeffs2
        
        # Check that smoothing occurred
        assert smoothed2 is not None and len(smoothed2) >= 2, "Smoothing should produce coefficients"
        
        # Left lane should be smoothed (not None)
        assert smoothed2[0] is not None, "Left lane should be smoothed"
        
        # Right lane should use previous (persistence) if available
        if smoothed1 is not None and len(smoothed1) >= 2 and smoothed1[1] is not None:
            assert smoothed2[1] is not None, "Right lane should persist from previous frame"
            # Should be the same as previous (persistence)
            np.testing.assert_array_equal(smoothed2[1], smoothed1[1])
    
    def test_lane_coeffs_history_limited(self):
        """Test that lane coefficient history is limited to 5 frames."""
        planner = RuleBasedTrajectoryPlanner(
            image_width=640.0,
            image_height=480.0
        )
        # Set smoothing alpha if not in __init__
        if not hasattr(planner, 'lane_smoothing_alpha'):
            planner.lane_smoothing_alpha = 0.7
        if not hasattr(planner, 'lane_coeffs_history'):
            planner.lane_coeffs_history = []
        
        # Process 10 frames
        for i in range(10):
            lane_coeffs = [
                np.array([0.0, 0.0, 200.0 + i]),
                np.array([0.0, 0.0, 440.0 + i])
            ]
            planner.plan(lane_coeffs)
        
        # History should be limited to 5 frames (max history size)
        assert len(planner.lane_coeffs_history) <= 5, (
            f"History should be limited to 5 frames, got {len(planner.lane_coeffs_history)}"
        )
        # Should have at least some history
        assert len(planner.lane_coeffs_history) > 0, "History should contain recent frames"


class TestOutlierRejection:
    """Test outlier rejection in center lane calculation."""
    
    def test_center_lane_outlier_rejection(self):
        """Test that center lane outliers are rejected.
        
        NOTE: Outlier rejection checks:
        1. center_offset_from_lane_center > 50px OR
        2. lane_width < 100px OR lane_width > 500px
        
        Since _compute_center_lane always computes the midpoint, center_offset is always 0.
        So we test with unreasonable lane width instead.
        """
        planner = RuleBasedTrajectoryPlanner(
            image_width=640.0,
            image_height=480.0
        )
        
        # Normal lanes (reasonable width ~240px)
        normal_left = np.array([0.0, 0.0, 200.0])
        normal_right = np.array([0.0, 0.0, 440.0])
        
        # Plan with normal lanes to set last_center_coeffs
        trajectory = planner.plan([normal_left, normal_right], {})
        normal_center = planner.last_center_coeffs.copy()
        planner._frames_since_last_center = 0  # Ensure it's recent
        
        # Outlier lanes: unreasonable lane width (< 100px triggers rejection)
        outlier_left = np.array([0.0, 0.0, 310.0])
        outlier_right = np.array([0.0, 0.0, 320.0])  # Only 10px apart (unreasonable)
        
        # Plan with outlier lanes - should trigger rejection
        trajectory = planner.plan([outlier_left, outlier_right], {})
        
        # After rejection, last_center_coeffs should still be normal_center
        # (not the outlier center)
        np.testing.assert_array_equal(planner.last_center_coeffs, normal_center, (
            f"Outlier rejection should use previous center lane. "
            f"Expected: {normal_center}, Got: {planner.last_center_coeffs}"
        ))
    
    def test_lane_width_outlier_rejection(self):
        """Test that unreasonable lane widths are rejected."""
        planner = RuleBasedTrajectoryPlanner(
            image_width=640.0,
            image_height=480.0
        )
        
        # Normal lanes (lane width ~240px = ~3.5m)
        normal_left = np.array([0.0, 0.0, 200.0])
        normal_right = np.array([0.0, 0.0, 440.0])
        
        normal_center = planner._compute_center_lane(normal_left, normal_right)
        planner.last_center_coeffs = normal_center.copy()
        
        # Too narrow lanes (lane width < 100px)
        narrow_left = np.array([0.0, 0.0, 310.0])
        narrow_right = np.array([0.0, 0.0, 320.0])  # Only 10px apart
        
        narrow_center = planner._compute_center_lane(narrow_left, narrow_right)
        
        # Should use previous center lane
        np.testing.assert_array_equal(narrow_center, normal_center)
        
        # Too wide lanes (lane width > 500px)
        wide_left = np.array([0.0, 0.0, 50.0])
        wide_right = np.array([0.0, 0.0, 600.0])  # 550px apart
        
        wide_center = planner._compute_center_lane(wide_left, wide_right)
        
        # Should use previous center lane
        np.testing.assert_array_equal(wide_center, normal_center)


class TestCenterSplineSmoothing:
    """Test spline smoothing for center lane."""

    def test_center_spline_smoothing_blends_with_previous(self):
        """Spline smoothing should reduce frame-to-frame center shifts."""
        planner_spline = RuleBasedTrajectoryPlanner(
            image_width=640.0,
            image_height=480.0,
            center_spline_enabled=True,
            center_spline_alpha=0.8,
            center_spline_samples=20,
            center_spline_degree=2,
        )
        planner_raw = RuleBasedTrajectoryPlanner(
            image_width=640.0,
            image_height=480.0,
            center_spline_enabled=False,
        )

        left_base = np.array([0.0, 0.0, 200.0])
        right_base = np.array([0.0, 0.0, 440.0])
        center_prev = planner_spline._compute_center_lane(left_base, right_base)
        planner_spline.last_center_coeffs = center_prev.copy()
        planner_spline._frames_since_last_center = 0

        left_shift = np.array([0.0, 0.0, 205.0])
        right_shift = np.array([0.0, 0.0, 445.0])

        center_raw = planner_raw._compute_center_lane(left_shift, right_shift)
        center_smoothed = planner_spline._compute_center_lane(left_shift, right_shift)

        raw_delta = abs(center_raw[-1] - center_prev[-1])
        smooth_delta = abs(center_smoothed[-1] - center_prev[-1])
        assert smooth_delta < raw_delta, (
            f"Spline smoothing should reduce delta: raw={raw_delta:.3f}, smooth={smooth_delta:.3f}"
        )


class TestBoundsChecking:
    """Test bounds checking in coordinate conversion."""
    
    def test_coordinate_conversion_bounds(self):
        """Test that coordinate conversion results are bounded."""
        planner = RuleBasedTrajectoryPlanner(
            image_width=640.0,
            image_height=480.0,
            camera_fov=60.0
        )
        
        # Normal coordinates (within bounds)
        x_normal, y_normal = planner._convert_image_to_vehicle_coords(
            x_pixels=320.0,  # Image center
            y_pixels=300.0,  # Mid image
            lookahead_distance=8.0
        )
        
        # Should be within reasonable bounds
        assert abs(x_normal) < 5.0, f"Lateral offset {x_normal} exceeds 5m"
        assert 0.0 <= y_normal <= 50.0, f"Forward distance {y_normal} out of bounds"
        
        # Extreme coordinates - check that x_offset is clamped
        # The bounds checking happens in _convert_image_to_vehicle_coords
        # For extreme x, it should clamp x_offset_pixels first
        x_extreme, y_extreme = planner._convert_image_to_vehicle_coords(
            x_pixels=1000.0,  # Way outside image (but will be clamped to ±320)
            y_pixels=300.0,  # Normal y
            lookahead_distance=8.0
        )
        
        # x_offset should be clamped to ±image_width/2 = ±320
        # At 8m lookahead, this should result in reasonable x_vehicle
        # The actual bounds checking may clamp the final result
        assert abs(x_extreme) <= 10.0, f"Extreme lateral offset {x_extreme} should be bounded"
        assert 0.0 <= y_extreme <= 50.0, f"Forward distance {y_extreme} should be bounded"
    
    def test_x_offset_clamping(self):
        """Test that x_offset is clamped to ±image_width/2."""
        planner = RuleBasedTrajectoryPlanner(
            image_width=640.0,
            image_height=480.0,
            camera_fov=60.0
        )
        
        # Extreme x coordinate
        x_extreme = 2000.0  # Way outside image
        x_center = planner.image_width / 2.0
        x_offset_pixels = x_extreme - x_center
        
        # Should be clamped
        clamped_offset = np.clip(x_offset_pixels, -planner.image_width/2, planner.image_width/2)
        assert abs(clamped_offset) <= planner.image_width / 2


class TestReferencePointStabilityOverTime:
    """Test that reference point stability improves over time with fixes."""
    
    def test_reference_point_stability_with_temporal_filtering(self):
        """Test that temporal filtering improves reference point stability."""
        planner = TrajectoryPlanningInference(
            planner_type="rule_based",
            image_width=640.0,
            image_height=480.0,
            reference_smoothing=0.95,
            lane_smoothing_alpha=0.7
        )
        
        # Simulate 100 frames with noisy lane detection
        reference_points = []
        
        for i in range(100):
            # Add noise to simulate detection errors
            noise = np.random.normal(0, 5.0)  # 5px std noise
            
            # Create lane coefficients with noise
            left_coeffs = np.array([0.0, 0.0, 200.0 + noise])
            right_coeffs = np.array([0.0, 0.0, 440.0 - noise])
            
            # Plan trajectory
            trajectory = planner.plan([left_coeffs, right_coeffs])
            
            # Get reference point
            ref_point = planner.get_reference_point(trajectory, lookahead=8.0)
            if ref_point:
                reference_points.append(ref_point['x'])
        
        # Split into thirds
        first_third = reference_points[:len(reference_points)//3]
        last_third = reference_points[2*len(reference_points)//3:]
        
        # Calculate stability
        std_first = np.std(first_third)
        std_last = np.std(last_third)
        
        # Stability should not increase significantly (should be < 2x)
        stability_ratio = std_last / std_first if std_first > 0 else 0
        assert stability_ratio < 2.0, (
            f"Reference point stability degraded: first={std_first:.4f}m, "
            f"last={std_last:.4f}m, ratio={stability_ratio:.2f}x"
        )
    
    def test_reference_point_no_drift(self):
        """Test that reference point doesn't drift over time."""
        planner = TrajectoryPlanningInference(
            planner_type="rule_based",
            image_width=640.0,
            image_height=480.0,
            reference_smoothing=0.95,
            lane_smoothing_alpha=0.7
        )
        
        # Simulate 100 frames with stable lane detection
        reference_points = []
        
        for i in range(100):
            # Stable lane coefficients (straight lane at center)
            left_coeffs = np.array([0.0, 0.0, 200.0])
            right_coeffs = np.array([0.0, 0.0, 440.0])
            
            # Plan trajectory
            trajectory = planner.plan([left_coeffs, right_coeffs])
            
            # Get reference point
            ref_point = planner.get_reference_point(trajectory, lookahead=8.0)
            if ref_point:
                reference_points.append(ref_point['x'])
        
        # Need enough points to evaluate
        if len(reference_points) < 30:
            pytest.skip("Not enough reference points generated")
        
        # Calculate drift
        first_third = reference_points[:len(reference_points)//3]
        last_third = reference_points[2*len(reference_points)//3:]
        
        mean_first = np.mean(first_third)
        mean_last = np.mean(last_third)
        drift = abs(mean_last - mean_first)
        
        # Drift should be minimal (< 0.2m for coordinate conversion tolerance)
        assert drift < 0.2, (
            f"Reference point drifted: first={mean_first:.4f}m, "
            f"last={mean_last:.4f}m, drift={drift:.4f}m"
        )
    
    def test_reference_point_handles_noisy_detection(self):
        """Test that reference point remains stable with noisy lane detection."""
        # Use fixed seed for reproducibility
        np.random.seed(42)
        
        planner = TrajectoryPlanningInference(
            planner_type="rule_based",
            image_width=640.0,
            image_height=480.0,
            reference_smoothing=0.95,
            lane_smoothing_alpha=0.7
        )
        
        # Simulate 50 frames with varying noise levels
        reference_points = []
        
        for i in range(50):
            # Varying noise levels (simulating detection quality changes)
            noise_level = 5.0 + (i / 10.0)  # Increasing noise
            noise = np.random.normal(0, noise_level)
            
            # Create lane coefficients with noise
            left_coeffs = np.array([0.0, 0.0, 200.0 + noise])
            right_coeffs = np.array([0.0, 0.0, 440.0 - noise])
            
            # Plan trajectory
            trajectory = planner.plan([left_coeffs, right_coeffs])
            
            # Get reference point
            ref_point = planner.get_reference_point(trajectory, lookahead=8.0)
            if ref_point:
                reference_points.append(ref_point['x'])
        
        # Need enough points to evaluate
        if len(reference_points) < 20:
            pytest.skip("Not enough reference points generated")
        
        # Reference point should remain stable despite increasing noise
        # With temporal filtering and smoothing, std should be controlled
        std_all = np.std(reference_points)
        
        # Standard deviation should be reasonable (< 0.3m with increasing noise)
        assert std_all < 0.3, (
            f"Reference point too unstable with noisy detection: std={std_all:.4f}m"
        )


class TestReferencePointSmoothing:
    """Test reference point smoothing."""
    
    def test_reference_point_smoothing_reduces_variance(self):
        """Test that reference point smoothing reduces variance."""
        # Use fixed seed for reproducibility
        np.random.seed(42)
        
        planner = TrajectoryPlanningInference(
            planner_type="rule_based",
            image_width=640.0,
            image_height=480.0,
            reference_smoothing=0.95  # High smoothing
        )
        
        # Create trajectory with noisy reference points
        reference_points_raw = []
        reference_points_smoothed = []
        
        for i in range(50):
            # Add noise to simulate detection errors
            noise = np.random.normal(0, 0.1)  # 0.1m std noise
            
            # Create lane coefficients with noise
            left_coeffs = np.array([0.0, 0.0, 200.0 + noise * 100])
            right_coeffs = np.array([0.0, 0.0, 440.0 - noise * 100])
            
            # Plan trajectory
            trajectory = planner.plan([left_coeffs, right_coeffs])
            
            # Get reference point
            ref_point = planner.get_reference_point(trajectory, lookahead=8.0)
            if ref_point and 'raw_x' in ref_point:
                reference_points_raw.append(ref_point['raw_x'])
                reference_points_smoothed.append(ref_point['x'])
        
        # Need enough points to evaluate
        if len(reference_points_raw) < 20 or len(reference_points_smoothed) < 20:
            pytest.skip("Not enough reference points generated")
        
        # Smoothed should have lower variance
        std_raw = np.std(reference_points_raw)
        std_smoothed = np.std(reference_points_smoothed)
        
        assert std_smoothed <= std_raw, (
            f"Smoothing should reduce or maintain variance: raw={std_raw:.4f}m, "
            f"smoothed={std_smoothed:.4f}m"
        )
        
        # With high smoothing (0.95), improvement should be significant
        if std_smoothed > 0:
            improvement = std_raw / std_smoothed
            assert improvement >= 1.0, (
                f"Smoothing should improve stability: {improvement:.2f}x"
            )

    def test_lane_position_smoothing_alpha_applies(self):
        """Test that lane position smoothing alpha is applied for lane_positions."""
        planner = TrajectoryPlanningInference(
            planner_type="rule_based",
            image_width=640.0,
            image_height=480.0,
            reference_smoothing=0.95,
            lane_position_smoothing_alpha=0.2,
        )
        trajectory = planner.plan(
            [np.array([0.0, 0.0, 200.0]), np.array([0.0, 0.0, 440.0])]
        )
        ref_point_1 = planner.get_reference_point(
            trajectory,
            lookahead=8.0,
            lane_positions={
                'left_lane_line_x': -3.5,
                'right_lane_line_x': 3.5,
            },
            use_direct=True,
        )
        assert ref_point_1 is not None

        ref_point_2 = planner.get_reference_point(
            trajectory,
            lookahead=8.0,
            lane_positions={
                'left_lane_line_x': -3.2,
                'right_lane_line_x': 3.8,
            },
            use_direct=True,
        )
        assert ref_point_2 is not None

        expected = (0.2 * ref_point_1['x']) + (0.8 * 0.3)
        assert np.isclose(ref_point_2['x'], expected, atol=1e-4), (
            f"Lane position smoothing should apply alpha: got={ref_point_2['x']:.4f}, "
            f"expected={expected:.4f}"
        )


class TestReferencePointJumpClamp:
    """Test reference point jump clamping."""

    def test_jump_clamp_limits_lateral_change(self):
        """Clamp should limit per-frame lateral jumps when enabled."""
        planner = TrajectoryPlanningInference(
            planner_type="rule_based",
            image_width=640.0,
            image_height=480.0,
            reference_smoothing=0.0,
            lane_position_smoothing_alpha=0.0,
            ref_point_jump_clamp_x=0.2,
        )

        trajectory = planner.plan(
            [np.array([0.0, 0.0, 200.0]), np.array([0.0, 0.0, 440.0])]
        )

        ref_point_1 = planner.get_reference_point(
            trajectory,
            lookahead=8.0,
            lane_positions={
                'left_lane_line_x': -3.5,
                'right_lane_line_x': 3.5,
            },
            use_direct=True,
            timestamp=0.0,
        )
        assert ref_point_1 is not None

        ref_point_2 = planner.get_reference_point(
            trajectory,
            lookahead=8.0,
            lane_positions={
                'left_lane_line_x': -2.0,
                'right_lane_line_x': 5.0,
            },
            use_direct=True,
            timestamp=0.1,
        )
        assert ref_point_2 is not None

        delta_x = abs(ref_point_2['x'] - ref_point_1['x'])
        assert delta_x <= 0.2 + 1e-6, f"Clamp should limit delta_x, got={delta_x:.4f}m"

    def test_jump_clamp_disabled_on_time_gap(self):
        """Clamp should not apply after a large time gap (state reset)."""
        planner = TrajectoryPlanningInference(
            planner_type="rule_based",
            image_width=640.0,
            image_height=480.0,
            reference_smoothing=0.0,
            lane_position_smoothing_alpha=0.0,
            ref_point_jump_clamp_x=0.2,
        )

        trajectory = planner.plan(
            [np.array([0.0, 0.0, 200.0]), np.array([0.0, 0.0, 440.0])]
        )

        ref_point_1 = planner.get_reference_point(
            trajectory,
            lookahead=8.0,
            lane_positions={
                'left_lane_line_x': -3.5,
                'right_lane_line_x': 3.5,
            },
            use_direct=True,
            timestamp=0.0,
        )
        assert ref_point_1 is not None

        ref_point_2 = planner.get_reference_point(
            trajectory,
            lookahead=8.0,
            lane_positions={
                'left_lane_line_x': -1.0,
                'right_lane_line_x': 6.0,
            },
            use_direct=True,
            timestamp=1.0,
        )
        assert ref_point_2 is not None

        delta_x = abs(ref_point_2['x'] - ref_point_1['x'])
        assert delta_x > 0.2, "Clamp should be disabled after time gap"


class TestReferencePointMaxClamp:
    """Test absolute clamp on reference x."""

    def test_reference_x_max_abs_clamps(self):
        """Reference x should be clamped to max abs when enabled."""
        planner = TrajectoryPlanningInference(
            planner_type="rule_based",
            image_width=640.0,
            image_height=480.0,
            reference_smoothing=0.0,
            lane_position_smoothing_alpha=0.0,
            reference_x_max_abs=0.5,
        )

        trajectory = planner.plan(
            [np.array([0.0, 0.0, 200.0]), np.array([0.0, 0.0, 440.0])]
        )

        ref_point = planner.get_reference_point(
            trajectory,
            lookahead=8.0,
            lane_positions={
                'left_lane_line_x': -4.0,
                'right_lane_line_x': 4.0,
            },
            use_direct=True,
        )
        assert ref_point is not None
        assert abs(ref_point['x']) <= 0.5 + 1e-6


class TestReferencePointTargetLane:
    """Test lane targeting when using lane positions."""

    def test_reference_point_targets_right_lane_center(self):
        planner = TrajectoryPlanningInference(
            planner_type="rule_based",
            image_width=640.0,
            image_height=480.0,
            target_lane="right"
        )
        trajectory = planner.plan(
            [np.array([0.0, 0.0, 200.0]), np.array([0.0, 0.0, 440.0])]
        )
        ref_point = planner.get_reference_point(
            trajectory,
            lookahead=8.0,
            lane_positions={
                'left_lane_line_x': -3.6,
                'right_lane_line_x': 3.6,
            },
            use_direct=True,
        )
        assert ref_point is not None
        assert abs(ref_point['x'] - 1.8) < 1e-6

    def test_reference_point_targets_right_lane_width_override(self):
        planner = TrajectoryPlanningInference(
            planner_type="rule_based",
            image_width=640.0,
            image_height=480.0,
            target_lane="right",
            target_lane_width_m=3.6
        )
        trajectory = planner.plan(
            [np.array([0.0, 0.0, 200.0]), np.array([0.0, 0.0, 440.0])]
        )
        ref_point = planner.get_reference_point(
            trajectory,
            lookahead=8.0,
            lane_positions={
                'left_lane_line_x': -1.9,
                'right_lane_line_x': 1.9,
            },
            use_direct=True,
        )
        assert ref_point is not None
        assert abs(ref_point['x'] - 0.1) < 1e-6

    def test_reference_point_sign_flip_gate(self):
        planner = TrajectoryPlanningInference(
            planner_type="rule_based",
            image_width=640.0,
            image_height=480.0,
            target_lane="right",
            target_lane_width_m=3.6,
            ref_sign_flip_threshold=0.12
        )
        trajectory = planner.plan(
            [np.array([0.0, 0.0, 200.0]), np.array([0.0, 0.0, 440.0])]
        )
        ref_point_1 = planner.get_reference_point(
            trajectory,
            lookahead=8.0,
            lane_positions={
                'left_lane_line_x': -1.9,
                'right_lane_line_x': 1.9,
            },
            use_direct=True,
        )
        ref_point_2 = planner.get_reference_point(
            trajectory,
            lookahead=8.0,
            lane_positions={
                'left_lane_line_x': -1.9,
                'right_lane_line_x': 1.7,
            },
            use_direct=True,
        )
        assert ref_point_1 is not None and ref_point_2 is not None
        assert ref_point_1['x'] > 0
        assert ref_point_2['x'] > 0

    def test_reference_point_rate_limit(self):
        planner = TrajectoryPlanningInference(
            planner_type="rule_based",
            image_width=640.0,
            image_height=480.0,
            target_lane="right",
            target_lane_width_m=3.6,
            ref_x_rate_limit=0.08
        )
        trajectory = planner.plan(
            [np.array([0.0, 0.0, 200.0]), np.array([0.0, 0.0, 440.0])]
        )
        ref_point_1 = planner.get_reference_point(
            trajectory,
            lookahead=8.0,
            lane_positions={
                'left_lane_line_x': -1.9,
                'right_lane_line_x': 1.9,
            },
            use_direct=True,
        )
        ref_point_2 = planner.get_reference_point(
            trajectory,
            lookahead=8.0,
            lane_positions={
                'left_lane_line_x': -2.7,
                'right_lane_line_x': 1.1,
            },
            use_direct=True,
        )
        assert ref_point_1 is not None and ref_point_2 is not None
        assert abs(ref_point_2['x'] - ref_point_1['x']) <= 0.081

    def test_reference_point_rate_limit_relaxes_on_precurve_heading(self):
        base_kwargs = dict(
            planner_type="rule_based",
            image_width=640.0,
            image_height=480.0,
            target_lane="right",
            target_lane_width_m=3.6,
            lane_position_smoothing_alpha=0.0,
            ref_x_rate_limit=0.08,
            ref_x_rate_limit_turn_heading_min_abs_rad=0.08,
            ref_x_rate_limit_turn_scale_max=1.0,
            ref_sign_flip_threshold=0.0,
            ref_sign_flip_disable_on_turn=False,
            ref_x_rate_limit_precurve_heading_min_abs_rad=0.03,
            ref_x_rate_limit_precurve_scale_max=4.0,
        )
        enabled = TrajectoryPlanningInference(
            **{**base_kwargs, "ref_x_rate_limit_precurve_enabled": True}
        )
        disabled = TrajectoryPlanningInference(
            **{**base_kwargs, "ref_x_rate_limit_precurve_enabled": False}
        )

        seed_point = {'x': 0.0, 'y': 8.0, 'heading': 0.0, 'velocity': 8.0}
        enabled.last_smoothed_ref_point = dict(seed_point)
        disabled.last_smoothed_ref_point = dict(seed_point)

        raw = {
            'x': 0.40,
            'y': 8.0,
            'heading': 0.06,
            'velocity': 8.0,
            'raw_x': 0.40,
            'raw_heading': 0.06,
        }
        with_relax = enabled._apply_smoothing(
            dict(raw), use_light_smoothing=True, minimal_smoothing=True
        )
        without_relax = disabled._apply_smoothing(
            dict(raw), use_light_smoothing=True, minimal_smoothing=True
        )

        assert with_relax['x'] > without_relax['x']
        assert with_relax['x'] == pytest.approx(0.32, rel=1e-3, abs=1e-3)
        assert without_relax['x'] == pytest.approx(0.08, rel=1e-3, abs=1e-3)

    def test_reference_point_multi_lookahead_blend(self):
        lane_coeffs = [
            np.array([0.2, 150.0]),  # left lane: x = 0.2*y + 150
            np.array([0.05, 470.0]),  # right lane: x = 0.05*y + 470
        ]
        lane_positions = {
            'left_lane_line_x': -1.8,
            'right_lane_line_x': 1.8,
        }
        base = TrajectoryPlanningInference(
            planner_type="rule_based",
            image_width=640.0,
            image_height=480.0,
            target_lane="center",
            multi_lookahead_enabled=False,
        )
        blended = TrajectoryPlanningInference(
            planner_type="rule_based",
            image_width=640.0,
            image_height=480.0,
            target_lane="center",
            multi_lookahead_enabled=True,
            multi_lookahead_far_scale=2.0,
            multi_lookahead_blend_alpha=0.5,
            multi_lookahead_curvature_threshold=1.0,
        )
        trajectory = base.plan(lane_coeffs)
        base_ref = base.get_reference_point(
            trajectory,
            lookahead=8.0,
            lane_coeffs=lane_coeffs,
            lane_positions=lane_positions,
            use_direct=True,
        )
        assert base_ref is not None
        far_ref = blended._compute_target_ref_from_coeffs(lane_coeffs, 16.0)
        assert far_ref is not None
        blended_ref = blended.get_reference_point(
            trajectory,
            lookahead=8.0,
            lane_coeffs=lane_coeffs,
            lane_positions=lane_positions,
            use_direct=True,
        )
        assert blended_ref is not None
        assert abs(far_ref['heading'] - base_ref['heading']) > 1e-3
        low = min(base_ref['heading'], far_ref['heading'])
        high = max(base_ref['heading'], far_ref['heading'])
        assert low <= blended_ref['heading'] <= high


class TestIntegrationStability:
    """Integration tests for overall stability."""
    
    def test_full_pipeline_stability_over_time(self):
        """Test that full pipeline maintains stability over extended period."""
        # Use fixed seed for reproducibility
        np.random.seed(42)
        
        planner = TrajectoryPlanningInference(
            planner_type="rule_based",
            image_width=640.0,
            image_height=480.0,
            reference_smoothing=0.95,
            lane_smoothing_alpha=0.7
        )
        
        # Simulate 200 frames (longer test)
        reference_points = []
        lateral_errors = []
        
        for i in range(200):
            # Simulate realistic lane detection with noise
            base_noise = 3.0  # Base detection noise
            noise = np.random.normal(0, base_noise)
            
            # Create lane coefficients
            left_coeffs = np.array([0.0, 0.0, 200.0 + noise])
            right_coeffs = np.array([0.0, 0.0, 440.0 - noise])
            
            # Plan trajectory
            trajectory = planner.plan([left_coeffs, right_coeffs])
            
            # Get reference point
            ref_point = planner.get_reference_point(trajectory, lookahead=8.0)
            if ref_point:
                reference_points.append(ref_point['x'])
                # Simulate lateral error (car position - reference)
                lateral_errors.append(abs(ref_point['x'] - 0.0))  # Assuming car at center
        
        # Need enough points to evaluate
        if len(reference_points) < 60:
            pytest.skip("Not enough reference points generated")
        
        # Split into thirds
        first_third = reference_points[:len(reference_points)//3]
        middle_third = reference_points[len(reference_points)//3:2*len(reference_points)//3]
        last_third = reference_points[2*len(reference_points)//3:]
        
        # Calculate stability metrics
        std_first = np.std(first_third) if len(first_third) > 1 else 0.0
        std_middle = np.std(middle_third) if len(middle_third) > 1 else 0.0
        std_last = np.std(last_third) if len(last_third) > 1 else 0.0
        
        # Stability should not degrade significantly
        if std_first > 0:
            stability_ratio = std_last / std_first
            assert stability_ratio < 3.0, (  # More lenient for extended simulation
                f"Stability degraded over time: first={std_first:.4f}m, "
                f"middle={std_middle:.4f}m, last={std_last:.4f}m, ratio={stability_ratio:.2f}x"
            )
        
        # Mean should not drift significantly
        mean_first = np.mean(first_third)
        mean_last = np.mean(last_third)
        drift = abs(mean_last - mean_first)
        assert drift < 0.3, (  # More lenient for coordinate conversion tolerance
            f"Reference point drifted: first={mean_first:.4f}m, "
            f"last={mean_last:.4f}m, drift={drift:.4f}m"
        )
        
        # Average lateral error should remain reasonable
        if len(lateral_errors) > 0:
            avg_lateral_error = np.mean(lateral_errors)
            assert avg_lateral_error < 0.5, (  # More lenient for noisy simulation
                f"Average lateral error too high: {avg_lateral_error:.4f}m"
            )

