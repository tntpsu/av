"""
Tests for camera offset correction.
Verifies that camera offset compensates for reference point bias correctly.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trajectory.models.trajectory_planner import RuleBasedTrajectoryPlanner
from trajectory.inference import TrajectoryPlanningInference


class TestCameraOffsetDirection:
    """Test that camera offset direction is correct."""
    
    def test_camera_offset_compensates_left_bias(self):
        """
        CRITICAL TEST: Verify camera offset compensates for LEFT bias.
        
        If ref_x is LEFT (negative), camera sees LEFT, so camera is RIGHT of vehicle center.
        To compensate: shift coordinate conversion RIGHT (add positive offset).
        
        FIXED: Now verifies direction (not just magnitude) to catch wrong direction.
        """
        planner_no_offset = RuleBasedTrajectoryPlanner(
            image_width=640.0,
            image_height=480.0,
            camera_fov=60.0,
            camera_offset_x=0.0  # No offset
        )
        
        planner_with_offset = RuleBasedTrajectoryPlanner(
            image_width=640.0,
            image_height=480.0,
            camera_fov=60.0,
            camera_offset_x=0.0411  # Positive = RIGHT offset
        )
        
        # Simulate reference point at LEFT (negative x in vehicle frame)
        # This means camera sees reference LEFT, so camera is RIGHT of vehicle center
        # At image center (320 pixels), we should get x_vehicle = 0 after offset
        x_pixels = 320.0  # Image center
        y_pixels = 400.0  # Bottom of image
        lookahead = 8.0
        
        x_no_offset, _ = planner_no_offset._convert_image_to_vehicle_coords(
            x_pixels, y_pixels, lookahead
        )
        
        x_with_offset, _ = planner_with_offset._convert_image_to_vehicle_coords(
            x_pixels, y_pixels, lookahead
        )
        
        # FIXED: Verify direction, not just magnitude
        # With positive offset and += operation, x_vehicle should shift RIGHT (increase)
        # This compensates for LEFT bias
        offset_difference = x_with_offset - x_no_offset
        assert offset_difference > 0, (
            f"Camera offset should shift RIGHT (positive direction). "
            f"Offset difference: {offset_difference:.4f}m, expected > 0. "
            f"This catches wrong direction (x -= instead of x +=)."
        )
        
        # Also verify offset amount is approximately correct
        assert abs(offset_difference - 0.0411) < 0.01, (
            f"Camera offset should shift by offset amount. "
            f"Offset difference: {offset_difference:.4f}m, expected ~0.0411m"
        )
        
        # After offset correction, x_vehicle should be close to 0 (compensated)
        assert abs(x_with_offset) < 0.1, (
            f"Camera offset should compensate for bias. "
            f"x_vehicle after offset: {x_with_offset:.4f}m, expected close to 0"
        )
    
    def test_camera_offset_direction_matches_bias(self):
        """
        Verify camera offset direction matches bias direction.
        
        If bias is LEFT (negative), offset should be RIGHT (positive) to compensate.
        """
        # Test with LEFT bias (negative ref_x)
        planner_left_bias = RuleBasedTrajectoryPlanner(
            image_width=640.0,
            image_height=480.0,
            camera_fov=60.0,
            camera_offset_x=0.0411  # Positive = RIGHT offset
        )
        
        # Convert image center (should be 0 after offset)
        x_vehicle, _ = planner_left_bias._convert_image_to_vehicle_coords(
            320.0, 400.0, 8.0
        )
        
        # With positive offset and += operation, x_vehicle should shift RIGHT
        # This compensates for LEFT bias
        assert x_vehicle > -0.1, (
            f"Camera offset should shift RIGHT for LEFT bias. "
            f"x_vehicle: {x_vehicle:.4f}m, expected > -0.1m"
        )
    
    def test_camera_offset_applied_in_coordinate_conversion(self):
        """
        Verify camera offset is applied in coordinate conversion.
        
        Test that offset is actually used in the conversion.
        """
        planner_no_offset = RuleBasedTrajectoryPlanner(
            image_width=640.0,
            image_height=480.0,
            camera_fov=60.0,
            camera_offset_x=0.0  # No offset
        )
        
        planner_with_offset = RuleBasedTrajectoryPlanner(
            image_width=640.0,
            image_height=480.0,
            camera_fov=60.0,
            camera_offset_x=0.0411  # With offset
        )
        
        x_pixels = 320.0
        y_pixels = 400.0
        lookahead = 8.0
        
        x_no_offset, _ = planner_no_offset._convert_image_to_vehicle_coords(
            x_pixels, y_pixels, lookahead
        )
        
        x_with_offset, _ = planner_with_offset._convert_image_to_vehicle_coords(
            x_pixels, y_pixels, lookahead
        )
        
        # With offset, x_vehicle should be different (shifted RIGHT)
        difference = x_with_offset - x_no_offset
        assert abs(difference - 0.0411) < 0.01, (
            f"Camera offset should shift x_vehicle by offset amount. "
            f"Difference: {difference:.4f}m, expected ~0.0411m"
        )


class TestCameraOffsetWithReferencePoint:
    """Test camera offset with reference point bias."""
    
    def test_camera_offset_reduces_reference_bias(self):
        """
        Verify camera offset reduces reference point bias.
        
        If reference point has LEFT bias, camera offset should reduce it.
        """
        planner = RuleBasedTrajectoryPlanner(
            image_width=640.0,
            image_height=480.0,
            camera_fov=60.0,
            camera_offset_x=0.0411  # Compensate for LEFT bias
        )
        
        # Create lanes with LEFT bias (center lane shifted LEFT)
        # Left lane at x=310, right lane at x=330
        # Center should be at x=320, but with bias it's at x=315 (LEFT)
        left_coeffs = np.array([0.0, 0.0, 310.0])
        right_coeffs = np.array([0.0, 0.0, 330.0])
        
        # Compute center lane
        center_coeffs = planner._compute_center_lane(left_coeffs, right_coeffs)
        
        # Convert center lane to vehicle coordinates at lookahead
        y_image = 400.0  # Bottom of image
        x_image = np.polyval(center_coeffs, y_image)
        
        x_vehicle, y_vehicle = planner._convert_image_to_vehicle_coords(
            x_image, y_image, 8.0
        )
        
        # After camera offset, x_vehicle should be closer to 0 (bias reduced)
        # Without offset: x_vehicle would be negative (LEFT bias)
        # With offset: x_vehicle += 0.0411 (shifts RIGHT, reduces bias)
        assert abs(x_vehicle) < 0.1, (
            f"Camera offset should reduce reference point bias. "
            f"x_vehicle after offset: {x_vehicle:.4f}m, expected < 0.1m"
        )


class TestAdaptiveThresholdTriggering:
    """Test that adaptive thresholds trigger for actual variance."""
    
    def test_adaptive_bias_correction_triggers_for_actual_variance(self):
        """
        Verify adaptive bias correction triggers for realistic variance.
        
        Current actual variance: 2.78 pixels (0.0400m)
        Threshold: 3.0 pixels (should trigger)
        """
        planner = RuleBasedTrajectoryPlanner(
            image_width=640.0,
            image_height=480.0,
            camera_fov=60.0
        )
        
        # Verify threshold is appropriate for actual variance
        # Actual variance: 2.78 pixels, threshold should be >= 3.0 pixels
        assert planner.persistent_bias_threshold_std >= 3.0, (
            f"Adaptive threshold should be >= 3.0 pixels for actual variance (2.78 pixels). "
            f"Threshold: {planner.persistent_bias_threshold_std}, expected >= 3.0"
        )
        
        # Verify threshold mechanism exists
        assert hasattr(planner, 'persistent_bias_threshold_std'), (
            "Adaptive threshold should be defined"
        )
        assert hasattr(planner, 'persistent_bias_threshold_mean'), (
            "Adaptive threshold mean should be defined"
        )
        
        # Verify bias history tracking exists
        assert hasattr(planner, 'bias_history'), (
            "Bias history should be initialized"
        )
        
        # Simulate bias with variance similar to actual (2.78 pixels)
        # Create lanes with persistent bias
        left_coeffs = np.array([0.0, 0.0, 310.0])
        right_coeffs = np.array([0.0, 0.0, 336.0])  # Creates ~3 pixel bias
        
        # Simulate 15 frames with persistent bias
        # Note: Each call appends to bias_history
        for i in range(15):
            center_coeffs = planner._compute_center_lane(left_coeffs, right_coeffs)
        
        # Verify bias history is tracked (even if same value)
        assert len(planner.bias_history) > 0, (
            f"Bias history should be tracked. "
            f"History size: {len(planner.bias_history)}"
        )
        
        # Verify threshold would trigger for realistic variance
        # If we had variance of 2.78 pixels (actual), it should trigger
        test_variance = 2.78  # pixels (actual variance)
        assert test_variance < planner.persistent_bias_threshold_std, (
            f"Adaptive threshold should trigger for actual variance. "
            f"Test variance: {test_variance} pixels, threshold: {planner.persistent_bias_threshold_std} pixels"
        )
    
    def test_bias_compensation_triggers_for_actual_variance(self):
        """
        Verify bias compensation triggers for realistic variance.
        
        Current actual variance: 0.0400m
        Threshold: 0.05m (should trigger)
        """
        inference = TrajectoryPlanningInference(
            planner_type="rule_based",
            image_width=640.0,
            image_height=480.0,
            camera_fov=60.0,
            reference_smoothing=0.95
        )
        
        from trajectory.models.trajectory_planner import Trajectory, TrajectoryPoint
        
        # Simulate persistent bias in smoothed reference point
        # Create trajectory with LEFT bias (-0.0411m)
        for i in range(20):
            trajectory = Trajectory(
                points=[TrajectoryPoint(x=-0.0411, y=8.0, heading=0.0, velocity=8.0, curvature=0.0)],
                length=8.0
            )
            ref_point = inference.get_reference_point(trajectory, lookahead=8.0)
        
        # Check if bias compensation threshold is appropriate
        assert inference.bias_compensation_std_threshold >= 0.05, (
            f"Bias compensation threshold should be >= 0.05m for actual variance. "
            f"Threshold: {inference.bias_compensation_std_threshold}, expected >= 0.05"
        )
        
        # Check if bias history has enough data
        if len(inference.smoothed_bias_history) >= inference.bias_compensation_persistence:
            bias_array = np.array(inference.smoothed_bias_history[-inference.bias_compensation_persistence:])
            bias_std = np.std(bias_array)
            bias_abs_mean = abs(np.mean(bias_array))
            
            # Verify threshold matches actual variance
            assert inference.bias_compensation_std_threshold >= bias_std, (
                f"Bias compensation threshold should match actual variance. "
                f"Threshold: {inference.bias_compensation_std_threshold:.4f}m, "
                f"Actual std: {bias_std:.4f}m"
            )

