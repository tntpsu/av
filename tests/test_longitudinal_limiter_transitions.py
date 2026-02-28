from control.pid_controller import LongitudinalController


def _build_controller(**overrides) -> LongitudinalController:
    params = {
        "kp": 0.3,
        "ki": 0.0,
        "kd": 0.0,
        "target_speed": 8.0,
        "max_speed": 12.0,
        "max_accel": 0.5,
        "max_decel": 3.0,
        "max_jerk": 100.0,
        "throttle_rate_limit": 1.0,
        "brake_rate_limit": 1.0,
        "speed_smoothing_alpha": 0.0,
        "speed_for_jerk_alpha": 0.0,
        "throttle_smoothing_alpha": 0.0,
        "continuous_accel_control": False,
        "limiter_transition_enabled": True,
        "limiter_transition_hold_frames": 2,
        "limiter_transition_smoothing_alpha": 0.8,
        "limiter_transition_hysteresis": 0.0,
    }
    params.update(overrides)
    return LongitudinalController(**params)


def test_limiter_transition_hold_prevents_immediate_chatter() -> None:
    ctrl = _build_controller()

    ctrl.compute_control(current_speed=0.0, reference_velocity=6.0, dt=0.1)
    ctrl.compute_control(current_speed=2.0, reference_velocity=6.0, dt=0.1)
    assert ctrl.last_accel_capped is True
    assert ctrl._limiter_state_active is True

    ctrl.compute_control(current_speed=2.0, reference_velocity=6.0, dt=0.1)
    assert ctrl._limiter_state_active is True
    ctrl.compute_control(current_speed=2.0, reference_velocity=6.0, dt=0.1)
    assert ctrl._limiter_state_active is True
    ctrl.compute_control(current_speed=2.0, reference_velocity=6.0, dt=0.1)
    assert ctrl._limiter_state_active is False


def test_limiter_transition_smoothing_is_bounded() -> None:
    ctrl = _build_controller(limiter_transition_hold_frames=1, limiter_transition_smoothing_alpha=0.85)

    ctrl.compute_control(current_speed=0.0, reference_velocity=6.0, dt=0.1)
    prev_smoothed = ctrl.last_accel_cmd_smoothed
    ctrl.compute_control(current_speed=2.0, reference_velocity=6.0, dt=0.1)

    assert ctrl.last_limiter_transition_active is True
    lower = min(ctrl.last_accel_cmd_raw, prev_smoothed) - 1e-6
    upper = max(ctrl.last_accel_cmd_raw, prev_smoothed) + 1e-6
    assert lower <= ctrl.last_accel_cmd_smoothed <= upper
    assert ctrl.last_accel_cmd_smoothed != ctrl.last_accel_cmd_raw
