"""
Tests for control stack.
"""

import pytest
import numpy as np
from control.pid_controller import PIDController, LateralController, LongitudinalController, VehicleController


def test_pid_controller():
    """Test PID controller."""
    pid = PIDController(kp=1.0, ki=0.1, kd=0.5, output_limit=(-1.0, 1.0))
    
    # Test step response
    error = 1.0
    output = pid.update(error, dt=0.1)
    
    assert output is not None
    assert -1.0 <= output <= 1.0


def test_lateral_controller():
    """Test lateral controller."""
    controller = LateralController(kp=1.0, kd=0.5)
    
    reference_point = {
        'x': 0.0,
        'y': 10.0,
        'heading': 0.0,
        'velocity': 10.0
    }
    
    steering = controller.compute_steering(
        current_heading=0.0,
        reference_point=reference_point
    )
    
    assert -1.0 <= steering <= 1.0


def test_lateral_controller_steering_direction():
    """
    CRITICAL TEST: Verify steering direction is correct.
    
    This test prevents the steering direction inversion bug.
    The car should steer TOWARD the reference, not away from it.
    
    Lesson Learned: We previously added/removed/added negation based on
    theoretical reasoning. This test ensures we catch direction issues
    empirically. See DEVELOPMENT_GUIDELINES.md for details.
    
    Note: This test uses lower kp to avoid saturation masking the direction.
    """
    # Use lower kp to avoid saturation at max_steering limit
    controller = LateralController(kp=0.5, kd=0.1, deadband=0.0, max_steering=0.5)
    
    # Test case 1: Car is RIGHT of reference
    # After coordinate system fix: lateral_error = ref_x (directly)
    # If ref_x = -1.0 (reference is 1.0m LEFT of vehicle center)
    # Then lateral_error = -1.0 (negative)
    # Car is RIGHT of ref → should steer LEFT (negative steering)
    # Standard PID: negative error → negative output → negative steering ✅
    reference_point_right = {
        'x': -1.0,  # Reference 1.0m LEFT of vehicle center (vehicle frame)
        'y': 10.0,
        'heading': 0.0,
        'velocity': 10.0
    }
    
    steering_right = controller.compute_steering(
        current_heading=0.0,
        reference_point=reference_point_right
    )
    
    # Car is RIGHT of ref (ref_x = -1.0) → lateral_error = -1.0 → should steer LEFT (negative)
    # Standard PID: negative error → negative output → negative steering ✅
    assert steering_right < 0, (
        f"Car RIGHT of ref (ref_x=-1.0) should steer LEFT (negative), got {steering_right:.3f}. "
        f"This indicates steering direction is inverted!"
    )
    
    # Test case 2: Car is LEFT of reference
    # After coordinate system fix: lateral_error = ref_x (directly)
    # If ref_x = 1.0 (reference is 1.0m RIGHT of vehicle center)
    # Then lateral_error = 1.0 (positive)
    # Car is LEFT of ref → should steer RIGHT (positive steering)
    # Standard PID: positive error → positive output → positive steering ✅
    # NOTE: Create a fresh controller to avoid rate limiting from previous test case
    controller_left = LateralController(kp=0.5, kd=0.1, deadband=0.0, max_steering=0.5)
    reference_point_left = {
        'x': 1.0,  # Reference 1.0m RIGHT of vehicle center (vehicle frame)
        'y': 10.0,
        'heading': 0.0,
        'velocity': 10.0
    }
    
    steering_left = controller_left.compute_steering(
        current_heading=0.0,
        reference_point=reference_point_left
    )
    
    # Car is LEFT of ref (ref_x = 1.0) → lateral_error = 1.0 → should steer RIGHT (positive)
    # Standard PID: positive error → positive output → positive steering ✅
    assert steering_left > 0, (
        f"Car LEFT of ref (ref_x=1.0) should steer RIGHT (positive), got {steering_left:.3f}. "
        f"This indicates steering direction is inverted!"
    )
    
    # Test case 3: Car is at reference
    # After coordinate system fix: lateral_error = ref_x (directly)
    # If ref_x = 0.0 (reference is at vehicle center)
    # Then lateral_error = 0.0 (zero)
    # Car at ref → should have minimal steering
    # NOTE: Create a fresh controller to avoid PID state from previous test cases
    controller_center = LateralController(kp=0.5, kd=0.1, deadband=0.0, max_steering=0.5)
    reference_point_center = {
        'x': 0.0,  # Reference at vehicle center (vehicle frame)
        'y': 10.0,
        'heading': 0.0,
        'velocity': 10.0
    }
    
    steering_center = controller_center.compute_steering(
        current_heading=0.0,
        reference_point=reference_point_center
    )
    
    # Car at ref (ref_x = 0.0) → lateral_error = 0.0 → should have minimal steering
    assert abs(steering_center) < 0.1, (
        f"Car at ref (ref_x=0.0) should have minimal steering, got {steering_center:.3f}"
    )
    
    print(f"✅ Steering direction test passed:")
    print(f"   Car RIGHT of ref: steering = {steering_right:.3f} (should be < 0) ✓")
    print(f"   Car LEFT of ref: steering = {steering_left:.3f} (should be > 0) ✓")
    print(f"   Car at center: steering = {steering_center:.3f} (should be ~0) ✓")


def test_longitudinal_controller():
    """Test longitudinal controller."""
    controller = LongitudinalController(target_speed=10.0)
    
    throttle, brake = controller.compute_control(current_speed=5.0)
    
    assert 0.0 <= throttle <= 1.0
    assert 0.0 <= brake <= 1.0


def test_longitudinal_rate_limit():
    """Test longitudinal throttle/brake rate limiting."""
    controller = LongitudinalController(
        target_speed=10.0,
        throttle_rate_limit=0.05,
        brake_rate_limit=0.10
    )

    throttle1, brake1 = controller.compute_control(current_speed=0.0, reference_velocity=10.0)
    throttle2, brake2 = controller.compute_control(current_speed=0.0, reference_velocity=10.0)

    assert throttle2 - throttle1 <= 0.05 + 1e-6
    assert brake2 - brake1 <= 0.10 + 1e-6


def test_longitudinal_accel_feedforward():
    """Feedforward accel should influence throttle/brake when PID gains are zero."""
    controller = LongitudinalController(
        kp=0.0,
        ki=0.0,
        kd=0.0,
        target_speed=10.0,
        max_accel=2.0,
        max_decel=2.0,
        throttle_rate_limit=1.0,
        brake_rate_limit=1.0,
        throttle_smoothing_alpha=1.0,
        speed_smoothing_alpha=0.0
    )

    throttle, brake = controller.compute_control(
        current_speed=5.0,
        reference_velocity=10.0,
        reference_accel=1.0,
        dt=0.1
    )
    assert throttle == pytest.approx(0.5, rel=1e-3)
    assert brake == pytest.approx(0.0, rel=1e-6)

    throttle2, brake2 = controller.compute_control(
        current_speed=10.0,
        reference_velocity=5.0,
        reference_accel=-1.0,
        dt=0.1
    )
    assert throttle2 == pytest.approx(0.0, rel=1e-6)
    assert brake2 == pytest.approx(0.5, rel=1e-3)


def test_longitudinal_accel_cap():
    """Acceleration cap should trigger when measured accel exceeds limits."""
    controller = LongitudinalController(
        kp=0.0,
        ki=0.0,
        kd=0.0,
        target_speed=10.0,
        max_accel=0.5,
        max_decel=0.5,
        max_jerk=1.0
    )

    controller.compute_control(
        current_speed=0.0,
        reference_velocity=10.0,
        reference_accel=1.0,
        dt=0.1
    )
    controller.compute_control(
        current_speed=1.0,
        reference_velocity=10.0,
        reference_accel=1.0,
        dt=0.1
    )

    assert controller.last_accel_capped is True


def test_vehicle_controller():
    """Test combined vehicle controller."""
    controller = VehicleController()
    
    current_state = {
        'heading': 0.0,
        'speed': 10.0,
        'position': np.array([0.0, 0.0])
    }
    
    reference_point = {
        'x': 0.0,
        'y': 10.0,
        'heading': 0.0,
        'velocity': 10.0
    }
    
    commands = controller.compute_control(current_state, reference_point)
    
    assert 'steering' in commands
    assert 'throttle' in commands
    assert 'brake' in commands
    assert -1.0 <= commands['steering'] <= 1.0
    assert 0.0 <= commands['throttle'] <= 1.0
    assert 0.0 <= commands['brake'] <= 1.0

