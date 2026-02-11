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


def test_lateral_controller_metadata_includes_steering_terms():
    """Metadata should include feedforward/feedback steering breakdown."""
    controller = LateralController(kp=1.0, kd=0.2)
    reference_point = {
        'x': 0.3,
        'y': 10.0,
        'heading': 0.0,
        'velocity': 10.0,
        'curvature': 0.01,
    }
    meta = controller.compute_steering(
        current_heading=0.0,
        reference_point=reference_point,
        return_metadata=True,
    )

    assert 'feedforward_steering' in meta
    assert 'feedback_steering' in meta
    assert 'total_error_scaled' in meta
    assert isinstance(meta['feedforward_steering'], float)
    assert isinstance(meta['feedback_steering'], float)
    assert isinstance(meta['total_error_scaled'], float)


def test_lateral_controller_sign_flip_override_on_straight():
    """Sign-flip override should activate on straight when error reverses."""
    controller = LateralController(
        kp=0.8,
        kd=0.0,
        deadband=0.0,
        max_steering=1.0,
        steering_smoothing_alpha=1.0,
    )
    ref = {'x': 1.0, 'y': 10.0, 'heading': 0.0, 'velocity': 10.0, 'curvature': 0.0}
    controller.compute_steering(current_heading=0.0, reference_point=ref)

    ref['x'] = -1.0
    meta = controller.compute_steering(current_heading=0.0, reference_point=ref, return_metadata=True)
    assert meta['straight_sign_flip_override_active'] is True


def test_lateral_controller_steering_jerk_limit():
    """Steering jerk limiter should bound change in steering rate."""
    controller = LateralController(
        kp=1.0,
        kd=0.0,
        deadband=0.0,
        max_steering=1.0,
        steering_smoothing_alpha=1.0,
        steering_jerk_limit=0.5,
    )
    reference_point = {
        'x': 0.0,
        'y': 10.0,
        'heading': 0.0,
        'velocity': 10.0
    }
    controller.compute_steering(current_heading=0.0, reference_point=reference_point)
    last_rate = getattr(controller, 'last_steering_rate', 0.0)

    reference_point['x'] = 1.0
    controller.compute_steering(current_heading=0.0, reference_point=reference_point)
    new_rate = controller.last_steering_rate

    dt = 0.033
    max_rate_delta = 0.5 * dt
    assert abs(new_rate - last_rate) <= max_rate_delta + 1e-6


def test_lateral_controller_curvature_transition_smoothing():
    """Curvature smoothing should damp abrupt curvature changes."""
    controller = LateralController(
        kp=0.0,
        kd=0.0,
        deadband=0.0,
        max_steering=1.0,
        steering_smoothing_alpha=1.0,
        curvature_smoothing_alpha=0.7,
        curvature_transition_threshold=0.0,
        curvature_transition_alpha=0.2,
    )
    reference_point = {
        'x': 0.0,
        'y': 10.0,
        'heading': 0.0,
        'velocity': 10.0,
        'curvature': 0.0,
    }
    controller.compute_steering(
        current_heading=0.0,
        reference_point=reference_point,
        return_metadata=True,
    )

    reference_point['curvature'] = 0.2
    meta = controller.compute_steering(
        current_heading=0.0,
        reference_point=reference_point,
        return_metadata=True,
    )
    assert meta['curve_feedforward_curvature_used'] < meta['path_curvature_raw']


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
        max_jerk=100.0,
        max_jerk_min=100.0,
        max_jerk_max=100.0,
        throttle_rate_limit=1.0,
        brake_rate_limit=1.0,
        throttle_smoothing_alpha=1.0,
        speed_smoothing_alpha=0.0,
        speed_for_jerk_alpha=0.0,
        speed_error_to_accel_gain=0.0,
        accel_mode_threshold=0.0,
        decel_mode_threshold=0.0,
        speed_error_accel_threshold=0.0,
        speed_error_brake_threshold=0.0,
        mode_switch_min_time=0.0,
        coast_throttle_max=1.0,
        accel_tracking_enabled=False
    )

    throttle, brake = controller.compute_control(
        current_speed=5.0,
        reference_velocity=10.0,
        reference_accel=1.0,
        dt=0.1
    )
    assert throttle == pytest.approx(0.5, rel=1e-3)
    assert brake == pytest.approx(0.0, rel=1e-6)


def test_longitudinal_accel_tracking_error_scale_reduces_throttle():
    base = LongitudinalController(
        max_accel=2.0,
        max_decel=2.0,
        speed_error_deadband=0.0,
        speed_error_gain_under=0.1,
        speed_error_gain_over=0.1,
        min_throttle_when_accel=0.0,
        min_throttle_hold=0.0,
        min_throttle_hold_speed=0.0,
        accel_tracking_enabled=True,
        accel_tracking_error_scale=1.0,
        accel_target_smoothing_alpha=0.0,
        continuous_accel_control=True,
        startup_speed_threshold=0.0,
        low_speed_accel_limit=0.0,
    )
    scaled = LongitudinalController(
        max_accel=2.0,
        max_decel=2.0,
        speed_error_deadband=0.0,
        speed_error_gain_under=0.1,
        speed_error_gain_over=0.1,
        min_throttle_when_accel=0.0,
        min_throttle_hold=0.0,
        min_throttle_hold_speed=0.0,
        accel_tracking_enabled=True,
        accel_tracking_error_scale=0.2,
        accel_target_smoothing_alpha=0.0,
        continuous_accel_control=True,
        startup_speed_threshold=0.0,
        low_speed_accel_limit=0.0,
    )

    throttle_base, _ = base.compute_control(
        current_speed=0.0,
        reference_velocity=5.0,
        dt=0.1,
        reference_accel=0.0,
    )
    throttle_scaled, _ = scaled.compute_control(
        current_speed=0.0,
        reference_velocity=5.0,
        dt=0.1,
        reference_accel=0.0,
    )

    assert throttle_scaled < throttle_base

def test_longitudinal_continuous_accel_control():
    """Continuous accel control should apply brake when overspeeding."""
    controller = LongitudinalController(
        kp=0.0,
        ki=0.0,
        kd=0.0,
        target_speed=10.0,
        max_accel=2.0,
        max_decel=2.0,
        max_jerk=100.0,
        max_jerk_min=100.0,
        max_jerk_max=100.0,
        throttle_rate_limit=1.0,
        brake_rate_limit=1.0,
        throttle_smoothing_alpha=1.0,
        speed_smoothing_alpha=0.0,
        speed_for_jerk_alpha=0.0,
        min_throttle_when_accel=0.0,
        min_throttle_hold=0.0,
        min_throttle_hold_speed=0.0,
        speed_error_to_accel_gain=1.0,
        accel_mode_threshold=0.0,
        decel_mode_threshold=0.0,
        speed_error_accel_threshold=0.0,
        speed_error_brake_threshold=0.0,
        mode_switch_min_time=0.0,
        coast_throttle_max=0.0,
        accel_tracking_enabled=False,
        continuous_accel_control=True,
        continuous_accel_deadband=0.0,
    )

    throttle, brake = controller.compute_control(
        current_speed=10.0,
        reference_velocity=5.0,
        reference_accel=0.0,
        dt=0.1,
    )

    assert throttle == pytest.approx(0.0, rel=1e-6)
    assert brake >= 0.4


def test_longitudinal_startup_ramp_caps_accel():
    """Startup ramp should cap accel during initial low-speed phase."""
    controller = LongitudinalController(
        kp=0.0,
        ki=0.0,
        kd=0.0,
        target_speed=10.0,
        max_accel=2.0,
        max_decel=2.0,
        max_jerk=100.0,
        max_jerk_min=100.0,
        max_jerk_max=100.0,
        throttle_rate_limit=1.0,
        brake_rate_limit=1.0,
        throttle_smoothing_alpha=1.0,
        speed_smoothing_alpha=0.0,
        speed_for_jerk_alpha=0.0,
        speed_error_to_accel_gain=1.0,
        accel_mode_threshold=0.0,
        decel_mode_threshold=0.0,
        speed_error_accel_threshold=0.0,
        speed_error_brake_threshold=0.0,
        mode_switch_min_time=0.0,
        coast_throttle_max=0.0,
        accel_tracking_enabled=False,
        continuous_accel_control=True,
        continuous_accel_deadband=0.0,
        startup_ramp_seconds=4.0,
        startup_accel_limit=1.0,
        startup_speed_threshold=2.0,
    )

    # First step: ramp is 0.25, accel cap is 0.25 m/s^2 => throttle <= 0.125
    throttle, brake = controller.compute_control(
        current_speed=0.5,
        reference_velocity=10.0,
        reference_accel=2.0,
        dt=1.0,
    )

    assert brake == pytest.approx(0.0, rel=1e-6)
    assert throttle <= 0.125 + 1e-3


def test_longitudinal_startup_throttle_cap_applies():
    """Startup throttle cap should limit throttle during startup window."""
    controller = LongitudinalController(
        kp=0.0,
        ki=0.0,
        kd=0.0,
        target_speed=10.0,
        max_accel=2.0,
        max_decel=2.0,
        max_jerk=100.0,
        max_jerk_min=100.0,
        max_jerk_max=100.0,
        throttle_rate_limit=1.0,
        brake_rate_limit=1.0,
        throttle_smoothing_alpha=1.0,
        speed_smoothing_alpha=0.0,
        speed_for_jerk_alpha=0.0,
        min_throttle_when_accel=0.0,
        min_throttle_hold=0.0,
        min_throttle_hold_speed=0.0,
        speed_error_to_accel_gain=1.0,
        accel_mode_threshold=0.0,
        decel_mode_threshold=0.0,
        speed_error_accel_threshold=0.0,
        speed_error_brake_threshold=0.0,
        mode_switch_min_time=0.0,
        coast_throttle_max=0.0,
        accel_tracking_enabled=False,
        continuous_accel_control=True,
        continuous_accel_deadband=0.0,
        startup_ramp_seconds=1.0,
        startup_accel_limit=10.0,
        startup_speed_threshold=2.0,
        startup_throttle_cap=0.2,
        startup_disable_accel_feedforward=True,
        low_speed_accel_limit=10.0,
        low_speed_speed_threshold=2.0,
    )

    throttle, brake = controller.compute_control(
        current_speed=0.5,
        reference_velocity=10.0,
        reference_accel=2.0,
        dt=0.1,
    )

    assert brake == pytest.approx(0.0, rel=1e-6)
    assert throttle <= 0.2 + 1e-3


def test_longitudinal_low_speed_accel_limit_applies():
    """Low-speed accel limit should cap throttle at low speeds."""
    controller = LongitudinalController(
        kp=0.0,
        ki=0.0,
        kd=0.0,
        target_speed=10.0,
        max_accel=2.0,
        max_decel=2.0,
        max_jerk=100.0,
        max_jerk_min=100.0,
        max_jerk_max=100.0,
        throttle_rate_limit=1.0,
        brake_rate_limit=1.0,
        throttle_smoothing_alpha=1.0,
        speed_smoothing_alpha=0.0,
        speed_for_jerk_alpha=0.0,
        min_throttle_when_accel=0.0,
        min_throttle_hold=0.0,
        min_throttle_hold_speed=0.0,
        speed_error_to_accel_gain=1.0,
        accel_mode_threshold=0.0,
        decel_mode_threshold=0.0,
        speed_error_accel_threshold=0.0,
        speed_error_brake_threshold=0.0,
        mode_switch_min_time=0.0,
        coast_throttle_max=0.0,
        accel_tracking_enabled=False,
        continuous_accel_control=True,
        continuous_accel_deadband=0.0,
        startup_ramp_seconds=1.0,
        startup_accel_limit=10.0,
        startup_speed_threshold=2.0,
        startup_throttle_cap=1.0,
        startup_disable_accel_feedforward=True,
        low_speed_accel_limit=0.6,
        low_speed_speed_threshold=2.0,
    )

    throttle, brake = controller.compute_control(
        current_speed=0.5,
        reference_velocity=10.0,
        reference_accel=2.0,
        dt=0.1,
    )

    assert brake == pytest.approx(0.0, rel=1e-6)
    assert throttle <= 0.3 + 1e-3


def test_longitudinal_speed_error_deadband_zeroes_small_error():
    """Deadband should zero small speed error contributions."""
    controller = LongitudinalController(
        kp=0.0,
        ki=0.0,
        kd=0.0,
        target_speed=10.0,
        max_accel=2.0,
        max_decel=2.0,
        max_jerk=100.0,
        max_jerk_min=100.0,
        max_jerk_max=100.0,
        throttle_rate_limit=1.0,
        brake_rate_limit=1.0,
        throttle_smoothing_alpha=1.0,
        speed_smoothing_alpha=0.0,
        speed_for_jerk_alpha=0.0,
        min_throttle_when_accel=0.0,
        min_throttle_hold=0.0,
        min_throttle_hold_speed=0.0,
        speed_error_to_accel_gain=0.0,
        speed_error_deadband=1.0,
        speed_error_gain_under=1.0,
        speed_error_gain_over=1.0,
        accel_mode_threshold=0.0,
        decel_mode_threshold=0.0,
        speed_error_accel_threshold=0.0,
        speed_error_brake_threshold=0.0,
        mode_switch_min_time=0.0,
        coast_throttle_max=0.0,
        accel_tracking_enabled=False,
        continuous_accel_control=True,
        continuous_accel_deadband=0.0,
    )

    throttle, brake = controller.compute_control(
        current_speed=9.6,
        reference_velocity=10.0,
        reference_accel=0.0,
        dt=0.1,
    )

    assert throttle == pytest.approx(0.0, rel=1e-6)
    assert brake == pytest.approx(0.0, rel=1e-6)


def test_longitudinal_overspeed_accel_zero_clamps_positive_accel():
    """Overspeed accel clamp should zero positive accel when above target."""
    controller = LongitudinalController(
        kp=0.0,
        ki=0.0,
        kd=0.0,
        target_speed=10.0,
        max_accel=2.0,
        max_decel=2.0,
        max_jerk=100.0,
        max_jerk_min=100.0,
        max_jerk_max=100.0,
        throttle_rate_limit=1.0,
        brake_rate_limit=1.0,
        throttle_smoothing_alpha=1.0,
        speed_smoothing_alpha=0.0,
        speed_for_jerk_alpha=0.0,
        min_throttle_when_accel=0.0,
        min_throttle_hold=0.0,
        min_throttle_hold_speed=0.0,
        speed_error_to_accel_gain=1.0,
        speed_error_deadband=0.0,
        speed_error_gain_under=1.0,
        speed_error_gain_over=1.0,
        overspeed_accel_zero_threshold=0.2,
        accel_mode_threshold=0.0,
        decel_mode_threshold=0.0,
        speed_error_accel_threshold=0.0,
        speed_error_brake_threshold=0.0,
        mode_switch_min_time=0.0,
        coast_throttle_max=0.0,
        accel_tracking_enabled=False,
        continuous_accel_control=True,
        continuous_accel_deadband=0.0,
    )

    throttle, brake = controller.compute_control(
        current_speed=10.5,
        reference_velocity=10.0,
        reference_accel=0.0,
        dt=0.1,
    )

    assert throttle == pytest.approx(0.0, rel=1e-6)
    assert brake > 0.0


def test_longitudinal_continuous_accel_respects_jerk_limit():
    controller = LongitudinalController(
        max_accel=2.0,
        max_decel=2.0,
        max_jerk=0.5,
        max_jerk_min=0.5,
        max_jerk_max=0.5,
        speed_error_deadband=0.0,
        speed_error_gain_under=0.0,
        speed_error_gain_over=0.0,
        min_throttle_when_accel=0.0,
        min_throttle_hold=0.0,
        min_throttle_hold_speed=0.0,
        accel_tracking_enabled=True,
        accel_target_smoothing_alpha=0.0,
        continuous_accel_control=True,
        startup_speed_threshold=0.0,
        low_speed_accel_limit=0.0,
    )

    throttle1, brake1 = controller.compute_control(
        current_speed=0.0,
        reference_velocity=5.0,
        dt=0.1,
        reference_accel=0.0,
    )
    accel1 = (throttle1 * controller.max_accel) - (brake1 * controller.max_decel)
    throttle2, brake2 = controller.compute_control(
        current_speed=0.0,
        reference_velocity=5.0,
        dt=0.1,
        reference_accel=2.0,
    )
    accel2 = (throttle2 * controller.max_accel) - (brake2 * controller.max_decel)

    assert accel2 - accel1 <= controller.max_jerk * 0.1 + 1e-6


def test_longitudinal_continuous_accel_pid_tracks_reference_accel():
    def _make_controller(accel_pid_kp: float) -> LongitudinalController:
        return LongitudinalController(
            max_accel=3.0,
            max_decel=3.0,
            max_jerk=0.0,
            throttle_rate_limit=0.0,
            brake_rate_limit=0.0,
            speed_error_deadband=0.0,
            speed_error_gain_under=0.0,
            speed_error_gain_over=0.0,
            min_throttle_when_accel=0.0,
            min_throttle_hold=0.0,
            min_throttle_hold_speed=0.0,
            accel_tracking_enabled=True,
            accel_pid_kp=accel_pid_kp,
            accel_pid_ki=0.0,
            accel_pid_kd=0.0,
            accel_target_smoothing_alpha=0.0,
            continuous_accel_control=True,
            startup_speed_threshold=0.0,
            low_speed_accel_limit=0.0,
        )

    controller_pid = _make_controller(accel_pid_kp=1.0)
    controller_no = _make_controller(accel_pid_kp=0.0)

    controller_pid.compute_control(
        current_speed=0.0,
        reference_velocity=0.0,
        dt=0.1,
        reference_accel=0.0,
    )
    controller_no.compute_control(
        current_speed=0.0,
        reference_velocity=0.0,
        dt=0.1,
        reference_accel=0.0,
    )

    throttle_pid, _ = controller_pid.compute_control(
        current_speed=0.0,
        reference_velocity=0.0,
        dt=0.1,
        reference_accel=1.0,
    )
    throttle_no, _ = controller_no.compute_control(
        current_speed=0.0,
        reference_velocity=0.0,
        dt=0.1,
        reference_accel=1.0,
    )

    assert throttle_pid > throttle_no


def test_longitudinal_throttle_curve_gamma_shapes_output():
    controller_linear = LongitudinalController(
        max_accel=2.0,
        max_decel=2.0,
        max_jerk=0.0,
        throttle_rate_limit=0.0,
        brake_rate_limit=0.0,
        speed_error_deadband=0.0,
        speed_error_gain_under=0.0,
        speed_error_gain_over=0.0,
        min_throttle_when_accel=0.0,
        min_throttle_hold=0.0,
        min_throttle_hold_speed=0.0,
        accel_tracking_enabled=True,
        accel_pid_kp=0.0,
        accel_pid_ki=0.0,
        accel_pid_kd=0.0,
        accel_target_smoothing_alpha=0.0,
        continuous_accel_control=True,
        startup_speed_threshold=0.0,
        low_speed_accel_limit=0.0,
        throttle_curve_gamma=1.0,
    )
    controller_gamma = LongitudinalController(
        max_accel=2.0,
        max_decel=2.0,
        max_jerk=0.0,
        throttle_rate_limit=0.0,
        brake_rate_limit=0.0,
        speed_error_deadband=0.0,
        speed_error_gain_under=0.0,
        speed_error_gain_over=0.0,
        min_throttle_when_accel=0.0,
        min_throttle_hold=0.0,
        min_throttle_hold_speed=0.0,
        accel_tracking_enabled=True,
        accel_pid_kp=0.0,
        accel_pid_ki=0.0,
        accel_pid_kd=0.0,
        accel_target_smoothing_alpha=0.0,
        continuous_accel_control=True,
        startup_speed_threshold=0.0,
        low_speed_accel_limit=0.0,
        throttle_curve_gamma=1.5,
    )

    controller_linear.compute_control(
        current_speed=0.0,
        reference_velocity=0.0,
        dt=0.1,
        reference_accel=1.0,
    )
    controller_gamma.compute_control(
        current_speed=0.0,
        reference_velocity=0.0,
        dt=0.1,
        reference_accel=1.0,
    )
    throttle_linear, _ = controller_linear.compute_control(
        current_speed=0.0,
        reference_velocity=0.0,
        dt=0.1,
        reference_accel=1.0,
    )
    throttle_gamma, _ = controller_gamma.compute_control(
        current_speed=0.0,
        reference_velocity=0.0,
        dt=0.1,
        reference_accel=1.0,
    )

    assert throttle_gamma < throttle_linear


def test_longitudinal_exclusive_throttle_brake_prefers_throttle():
    controller = LongitudinalController(
        max_accel=2.0,
        max_decel=2.0,
        max_jerk=0.0,
        throttle_rate_limit=1.0,
        brake_rate_limit=1.0,
        speed_error_deadband=0.0,
        speed_error_gain_under=0.0,
        speed_error_gain_over=0.0,
        min_throttle_when_accel=0.0,
        min_throttle_hold=0.0,
        min_throttle_hold_speed=0.0,
        accel_tracking_enabled=True,
        accel_pid_kp=0.0,
        accel_pid_ki=0.0,
        accel_pid_kd=0.0,
        accel_target_smoothing_alpha=0.0,
        continuous_accel_control=True,
        startup_speed_threshold=0.0,
        low_speed_accel_limit=0.0,
        low_speed_speed_threshold=0.0,
    )

    controller.compute_control(
        current_speed=1.0,
        reference_velocity=0.0,
        dt=0.1,
        reference_accel=-1.0,
    )
    throttle, brake = controller.compute_control(
        current_speed=1.0,
        reference_velocity=2.0,
        dt=0.1,
        reference_accel=1.0,
    )

    assert throttle > 0.0
    assert brake == pytest.approx(0.0)


def test_longitudinal_accel_cap():
    """Acceleration cap should trigger when measured accel exceeds limits."""
    controller = LongitudinalController(
        kp=0.0,
        ki=0.0,
        kd=0.0,
        target_speed=10.0,
        max_accel=0.5,
        max_decel=0.5,
        max_jerk=100.0,
        speed_for_jerk_alpha=0.0
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


def test_longitudinal_mode_dwell_prevents_flip():
    """Mode dwell should prevent rapid accel/brake switching."""
    controller = LongitudinalController(
        target_speed=10.0,
        max_speed=20.0,
        overspeed_brake_max=0.0,
        speed_error_accel_threshold=0.1,
        speed_error_brake_threshold=-0.1,
        accel_mode_threshold=0.1,
        decel_mode_threshold=0.1,
        mode_switch_min_time=0.5,
        throttle_rate_limit=1.0,
        brake_rate_limit=1.0,
        speed_smoothing_alpha=0.0,
        speed_for_jerk_alpha=0.0,
        speed_error_to_accel_gain=0.5,
        coast_throttle_max=0.2,
        coast_hold_seconds=0.5
    )

    throttle1, brake1 = controller.compute_control(
        current_speed=9.0,
        reference_velocity=10.0,
        dt=0.1
    )
    assert controller.longitudinal_mode == "accel"
    assert throttle1 >= 0.0
    assert brake1 == pytest.approx(0.0)

    throttle2, brake2 = controller.compute_control(
        current_speed=10.5,
        reference_velocity=10.0,
        dt=0.1
    )
    assert controller.longitudinal_mode == "accel"
    assert brake2 == pytest.approx(0.0)


def test_longitudinal_coast_no_brake_near_target():
    """Near-target speed should coast without braking."""
    controller = LongitudinalController(
        target_speed=10.0,
        speed_error_accel_threshold=0.2,
        speed_error_brake_threshold=-0.2,
        accel_mode_threshold=0.2,
        decel_mode_threshold=0.2,
        coast_throttle_kp=0.2,
        coast_throttle_max=0.1,
        throttle_rate_limit=1.0,
        brake_rate_limit=1.0,
        speed_smoothing_alpha=0.0,
        speed_for_jerk_alpha=0.0,
        accel_tracking_enabled=False
    )


def test_longitudinal_accel_tracking_uses_reference_accel():
    """Accel tracking should respond to reference_accel even when speed error is zero."""
    controller = LongitudinalController(
        kp=0.0,
        ki=0.0,
        kd=0.0,
        target_speed=10.0,
        max_accel=2.0,
        max_decel=2.0,
        throttle_rate_limit=1.0,
        brake_rate_limit=1.0,
        speed_smoothing_alpha=0.0,
        speed_for_jerk_alpha=0.0,
        accel_tracking_enabled=True,
        accel_pid_kp=1.0,
        accel_pid_ki=0.0,
        accel_pid_kd=0.0,
        accel_mode_threshold=0.0,
        speed_error_accel_threshold=0.0,
        mode_switch_min_time=0.0
    )

    throttle, brake = controller.compute_control(
        current_speed=10.0,
        reference_velocity=10.0,
        reference_accel=1.0,
        dt=0.1
    )
    assert throttle > 0.0
    assert brake == pytest.approx(0.0)


def test_longitudinal_straight_throttle_cap():
    """Straight throttle cap should limit throttle on low curvature."""
    controller = LongitudinalController(
        kp=0.0,
        ki=0.0,
        kd=0.0,
        target_speed=10.0,
        max_accel=2.0,
        max_decel=2.0,
        throttle_rate_limit=1.0,
        brake_rate_limit=1.0,
        speed_smoothing_alpha=0.0,
        speed_for_jerk_alpha=0.0,
        accel_tracking_enabled=True,
        accel_pid_kp=1.0,
        accel_pid_ki=0.0,
        accel_pid_kd=0.0,
        accel_mode_threshold=0.0,
        speed_error_accel_threshold=0.0,
        mode_switch_min_time=0.0,
        straight_throttle_cap=0.3,
        straight_curvature_threshold=0.003
    )

    throttle, brake = controller.compute_control(
        current_speed=5.0,
        reference_velocity=10.0,
        reference_accel=1.5,
        current_curvature=0.0,
        dt=0.1
    )
    assert brake == pytest.approx(0.0)
    assert throttle <= 0.3 + 1e-6


def test_longitudinal_coast_hold_forces_coast():
    """Coast hold should keep controller in coast even if accel is requested."""
    controller = LongitudinalController(
        target_speed=10.0,
        throttle_rate_limit=1.0,
        brake_rate_limit=1.0,
        speed_smoothing_alpha=0.0,
        speed_for_jerk_alpha=0.0,
        accel_tracking_enabled=True,
        accel_pid_kp=1.0,
        accel_pid_ki=0.0,
        accel_pid_kd=0.0,
        accel_mode_threshold=0.0,
        speed_error_accel_threshold=0.0,
        mode_switch_min_time=0.0,
        coast_hold_seconds=0.5
    )

    controller.longitudinal_mode = "coast"
    controller.coast_hold_remaining = 0.3
    throttle, brake = controller.compute_control(
        current_speed=9.0,
        reference_velocity=10.0,
        reference_accel=1.0,
        dt=0.1
    )
    assert controller.longitudinal_mode == "coast"
    assert brake == pytest.approx(0.0)


def test_longitudinal_min_throttle_hold_applies():
    """Min throttle hold should prevent drift when under target at low speed."""
    controller = LongitudinalController(
        target_speed=10.0,
        min_throttle_hold=0.2,
        min_throttle_hold_speed=4.0,
        throttle_rate_limit=1.0,
        brake_rate_limit=1.0,
        speed_smoothing_alpha=0.0,
        speed_for_jerk_alpha=0.0,
        accel_tracking_enabled=True,
        accel_pid_kp=1.0,
        accel_pid_ki=0.0,
        accel_pid_kd=0.0,
        accel_mode_threshold=0.0,
        speed_error_accel_threshold=0.0,
        mode_switch_min_time=0.0
    )

    throttle, brake = controller.compute_control(
        current_speed=1.0,
        reference_velocity=10.0,
        reference_accel=1.0,
        dt=0.1
    )
    assert brake == pytest.approx(0.0)
    assert throttle >= 0.2

    throttle, brake = controller.compute_control(
        current_speed=9.95,
        reference_velocity=10.0,
        dt=0.1
    )
    assert brake == pytest.approx(0.0)
    assert 0.0 <= throttle <= 0.3


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

