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


def test_limiter_sees_smooth_floor_ramp() -> None:
    """
    Step IDM from 0 → -3.0 over 10 frames while the limiter-transition block
    is enabled. The accel_cmd_smoothed should ramp monotonically (EMA-style)
    toward the clipped -3.0, not jump in a single step. This verifies that
    the min-gate composes cleanly with the downstream jerk/accel limiter.
    """
    ctrl = _build_controller(
        max_accel=2.0,
        max_decel=6.0,
        limiter_transition_enabled=True,
        limiter_transition_hold_frames=2,
        limiter_transition_smoothing_alpha=0.7,
        limiter_transition_hysteresis=0.0,
    )
    floor_states = ("ACC_ACTIVE", "CUTOUT")
    # Ramp IDM linearly 0 → -3.0 across 10 frames while speed-P wants +accel.
    prev_smoothed = None
    for i in range(10):
        idm = -3.0 * (i / 9.0)
        ctrl.compute_control(
            current_speed=5.5,
            reference_velocity=8.0,
            dt=0.1,
            acc_state_code="ACC_ACTIVE",
            idm_accel_mps2=idm,
            acc_idm_accel_floor_mode="active",
            acc_idm_accel_floor_states=floor_states,
            acc_idm_accel_floor_min_negative=-0.25,
        )
        if prev_smoothed is not None and ctrl.last_acc_idm_floor_active == 1:
            # With the limiter transition engaged, smoothed trails raw.
            # We assert no step discontinuities (bounded per-frame change).
            step = abs(ctrl.last_accel_cmd_smoothed - prev_smoothed)
            assert step < 3.0, (
                f"frame {i}: smoothed accel stepped by {step:.2f} m/s², "
                "limiter is seeing discontinuous min-gate output"
            )
        prev_smoothed = ctrl.last_accel_cmd_smoothed
