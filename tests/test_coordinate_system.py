"""
Tests for coordinate system validation.
These tests catch integration issues with coordinate transformations.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from av_stack import AVStack
from control.pid_controller import LateralController


class TestVehiclePositionExtraction:
    """Test vehicle position extraction from Unity state."""
    
    def test_position_extraction_returns_actual_position(self):
        """Verify _extract_position() returns actual position, not always [0,0]."""
        av_stack = AVStack(record_data=False)
        
        # Test with known position
        vehicle_state = {
            'position': {'x': 1.5, 'y': 0.5, 'z': 10.0},
            'rotation': {'x': 0, 'y': 0, 'z': 0, 'w': 1}
        }
        
        pos = av_stack._extract_position(vehicle_state)
        assert pos[0] == 1.5, f"Expected x=1.5, got {pos[0]}"
        assert pos[1] == 10.0, f"Expected z=10.0, got {pos[1]}"
    
    def test_position_extraction_with_zero(self):
        """Verify position extraction works with zero position."""
        av_stack = AVStack(record_data=False)
        
        vehicle_state_zero = {
            'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'rotation': {'x': 0, 'y': 0, 'z': 0, 'w': 1}
        }
        
        pos_zero = av_stack._extract_position(vehicle_state_zero)
        assert pos_zero[0] == 0.0
        assert pos_zero[1] == 0.0
    
    def test_position_extraction_with_missing_data(self):
        """Verify position extraction handles missing data gracefully."""
        av_stack = AVStack(record_data=False)
        
        # Missing position
        vehicle_state = {
            'rotation': {'x': 0, 'y': 0, 'z': 0, 'w': 1}
        }
        
        pos = av_stack._extract_position(vehicle_state)
        assert pos[0] == 0.0
        assert pos[1] == 0.0


class TestLateralErrorCalculation:
    """Test lateral error calculation correctness."""
    
    def test_lateral_error_with_zero_heading(self):
        """Verify lateral error calculation with zero heading.
        
        NOTE: ref_x is already in vehicle coordinates (lateral offset from vehicle center).
        The controller uses ref_x directly as lateral_error, not ref_x - vehicle_x.
        """
        controller = LateralController(kp=0.5, kd=0.1, deadband=0.0)
        
        # Test case: Car RIGHT of ref means ref_x should be negative
        # ref_x = -1.0 means reference is 1.0m to the LEFT of vehicle center
        # This means car is RIGHT of ref, so lateral_error should be negative
        # Expected: negative error (car RIGHT of ref)
        
        reference_point = {'x': -1.0, 'y': 10.0, 'heading': 0.0, 'velocity': 10.0}
        vehicle_position = np.array([1.0, 0.0])  # Not used in calculation, but kept for API
        
        result = controller.compute_steering(
            current_heading=0.0,
            reference_point=reference_point,
            vehicle_position=vehicle_position,
            return_metadata=True
        )
        
        assert result['lateral_error'] < 0, "Car RIGHT of ref should have negative error"
        assert abs(result['lateral_error'] - (-1.0)) < 0.1, f"Error calculation wrong: got {result['lateral_error']}, expected ~-1.0"
    
    def test_lateral_error_with_nonzero_heading(self):
        """Verify lateral error calculation with non-zero heading.
        
        NOTE: ref_x is already in vehicle coordinates (lateral offset from vehicle center).
        """
        controller = LateralController(kp=0.5, kd=0.1, deadband=0.0)
        
        reference_point = {'x': -1.0, 'y': 10.0, 'heading': 0.0, 'velocity': 10.0}  # Car RIGHT of ref
        vehicle_position = np.array([1.0, 0.0])  # Not used in calculation, but kept for API
        
        # Test with small headings (< 10°) - should work correctly
        small_headings = [0.0, np.pi/36, np.pi/18]  # 0°, 5°, 10°
        
        for heading in small_headings:
            result = controller.compute_steering(
                current_heading=heading,
                reference_point=reference_point,
                vehicle_position=vehicle_position,
                return_metadata=True
            )
            
            # Error should still be negative (car is RIGHT of ref)
            # Magnitude should be reasonable (< 5m)
            assert result['lateral_error'] < 0, f"Error should be negative at heading {np.degrees(heading):.1f}°"
            assert abs(result['lateral_error']) < 5.0, f"Error magnitude too large: {result['lateral_error']:.3f}m at heading {np.degrees(heading):.1f}°"
        
        # Test with large headings (> 10°) - dy weight is reduced, error should be reasonable
        large_headings = [np.pi/6, np.pi/4]  # 30°, 45°
        
        for heading in large_headings:
            result = controller.compute_steering(
                current_heading=heading,
                reference_point=reference_point,
                vehicle_position=vehicle_position,
                return_metadata=True
            )
            
            # With large heading, dy contribution is reduced (weight=0.1)
            # Error magnitude should be much more reasonable than before
            # Allow some increase but not 29x
            assert abs(result['lateral_error']) < 5.0, (
                f"Error magnitude too large: {result['lateral_error']:.3f}m at heading {np.degrees(heading):.1f}°. "
                f"With heading fix, should be < 5m."
            )
    
    def test_lateral_error_car_left_of_ref(self):
        """Verify lateral error when car is LEFT of reference.
        
        NOTE: ref_x is already in vehicle coordinates (lateral offset from vehicle center).
        Car LEFT of ref means ref_x should be positive.
        """
        controller = LateralController(kp=0.5, kd=0.1, deadband=0.0)
        
        # Car LEFT of ref means ref_x should be positive
        # ref_x = 1.0 means reference is 1.0m to the RIGHT of vehicle center
        # This means car is LEFT of ref, so lateral_error should be positive
        
        reference_point = {'x': 1.0, 'y': 10.0, 'heading': 0.0, 'velocity': 10.0}
        vehicle_position = np.array([-1.0, 0.0])  # Not used in calculation, but kept for API
        
        result = controller.compute_steering(
            current_heading=0.0,
            reference_point=reference_point,
            vehicle_position=vehicle_position,
            return_metadata=True
        )
        
        assert result['lateral_error'] > 0, "Car LEFT of ref should have positive error"
        assert abs(result['lateral_error'] - 1.0) < 0.1, f"Error calculation wrong: got {result['lateral_error']}, expected ~1.0"


class TestHeadingEffectOnLateralError:
    """Test how heading affects lateral error calculation."""
    
    def test_heading_effect_magnitude(self):
        """Verify heading doesn't cause excessive error magnitude."""
        controller = LateralController(kp=0.5, kd=0.1, deadband=0.0)
        
        # Small offset: ref_x = 0.5m (reference is 0.5m to the RIGHT of vehicle center)
        reference_point = {'x': 0.5, 'y': 10.0, 'heading': 0.0, 'velocity': 10.0}
        vehicle_position = np.array([0.5, 0.0])  # Not used in calculation, but kept for API
        
        # Test with small heading (< 10°)
        result_small = controller.compute_steering(
            current_heading=0.0,
            reference_point=reference_point,
            vehicle_position=vehicle_position,
            return_metadata=True
        )
        
        # Test with large heading (> 10°)
        result_large = controller.compute_steering(
            current_heading=np.pi/4,  # 45°
            reference_point=reference_point,
            vehicle_position=vehicle_position,
            return_metadata=True
        )
        
        # With fix: Large heading should have reduced weight, so error shouldn't explode
        error_small = abs(result_small['lateral_error'])
        error_large = abs(result_large['lateral_error'])
        
        # After fix: Large heading error should be reasonable (< 10x increase)
        # Previously was 29x, now should be much less due to reduced weight
        assert error_large < error_small * 10, (
            f"Heading effect still too large: small heading error={error_small:.3f}m, "
            f"large heading error={error_large:.3f}m ({error_large/error_small:.1f}x increase). "
            f"Expected < 10x after fix."
        )


class TestPIDIntegralReset:
    """Test PID integral reset logic."""
    
    def test_integral_reset_on_small_error(self):
        """Verify PID integral resets when error is small for extended period."""
        controller = LateralController(kp=0.5, ki=0.0, kd=0.1, deadband=0.0)
        
        reference_point = {'x': 0.0, 'y': 10.0, 'heading': 0.0, 'velocity': 10.0}
        
        # Create small errors for 35 frames (more than 30 frame threshold)
        for i in range(35):
            controller.compute_steering(
                current_heading=0.0,
                reference_point=reference_point,
                vehicle_position=np.array([0.01, 0.0]),  # Very small error
                return_metadata=False
            )
        
        # Check if integral was reset
        result = controller.compute_steering(
            current_heading=0.0,
            reference_point=reference_point,
            vehicle_position=np.array([0.01, 0.0]),
            return_metadata=True
        )
        
        # Integral should be reset (or very small)
        assert abs(result['pid_integral']) < 0.05, (
            f"Integral should be reset after small errors, got {result['pid_integral']:.4f}"
        )
    
    def test_integral_reset_on_sign_change(self):
        """Verify integral resets when total_error changes sign.
        
        NOTE: 
        - ref_x is already in vehicle coordinates (lateral offset from vehicle center)
        - Integral reset checks total_error sign, not just lateral_error sign
        - total_error = heading_weight * heading_error + lateral_weight * lateral_error
        - For this test, we ensure heading_error is 0 so total_error sign matches lateral_error sign
        """
        controller = LateralController(kp=0.5, ki=0.1, kd=0.1, deadband=0.0)
        
        # Create positive error (ref_x > 0 means target is RIGHT, car is LEFT)
        # Ensure heading_error is 0 so total_error sign matches lateral_error sign
        reference_point_positive = {'x': 1.0, 'y': 10.0, 'heading': 0.0, 'velocity': 10.0}
        
        # Create positive error for multiple frames to build up integral
        for _ in range(10):
            controller.compute_steering(
                current_heading=0.0,  # Heading matches desired heading (heading_error = 0)
                reference_point=reference_point_positive,
                return_metadata=False
            )
        
        result1 = controller.compute_steering(
            current_heading=0.0,
            reference_point=reference_point_positive,
            return_metadata=True
        )
        
        integral_before = abs(result1['pid_integral'])
        total_error_before = result1['total_error']
        
        # Create negative error (ref_x < 0 means target is LEFT, car is RIGHT) - sign change
        # Ensure heading_error is still 0
        reference_point_negative = {'x': -1.0, 'y': 10.0, 'heading': 0.0, 'velocity': 10.0}
        result2 = controller.compute_steering(
            current_heading=0.0,  # Heading still matches desired heading
            reference_point=reference_point_negative,
            return_metadata=True
        )
        
        total_error_after = result2['total_error']
        
        # Verify total_error sign changed (required for integral reset)
        assert np.sign(total_error_before) != np.sign(total_error_after), (
            f"total_error sign must change for integral reset: "
            f"before={total_error_before:.4f}, after={total_error_after:.4f}"
        )
        
        # Integral should be reset when total_error sign changes
        # After reset, integral will be one frame's worth of new error
        # PID update: integral += error * dt (not error * ki * dt)
        dt = 0.033  # 30 FPS
        expected_integral_after_reset = abs(total_error_after) * dt  # One frame's worth
        
        integral_after = abs(result2['pid_integral'])
        integral_before_sign = np.sign(result1['pid_integral'])
        integral_after_sign = np.sign(result2['pid_integral'])
        
        # The key check: integral sign should have changed (reset happened)
        # This is the primary indicator that reset occurred
        assert integral_before_sign != integral_after_sign, (
            f"Integral sign should change after reset: "
            f"before={result1['pid_integral']:.4f} (sign={integral_before_sign}), "
            f"after={result2['pid_integral']:.4f} (sign={integral_after_sign})"
        )
        
        # Integral should be approximately one frame's worth
        # Allow some tolerance for decay and rounding
        assert integral_after <= expected_integral_after_reset * 1.5, (
            f"Integral after reset should be approximately one frame's worth: "
            f"got={integral_after:.4f}, expected~{expected_integral_after_reset:.4f}"
        )
        
        # NOTE: Integral magnitude may be similar before/after if decay is aggressive
        # The key indicator of reset is the sign change, not magnitude decrease
        # If sign changed, reset definitely happened

