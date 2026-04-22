"""
Unit tests for IDM-as-accel-floor min-gate (greedy-swimming-naur.md).

Plan: when acc_state_code in {ACC_ACTIVE, CUTOUT} and idm_accel_mps2 is more
negative than acc_idm_accel_floor_min_negative, clip accel_cmd_raw down to
idm_accel_mps2 (min-gate). Modes:
  - off    : no-op, no telemetry
  - shadow : compute telemetry (last_acc_idm_floor_active, candidate) but do
             not alter accel_cmd_raw / throttle / brake
  - active : apply the clip, back-propagate to throttle/brake

The gate only reduces authority (never increases) — smell-7 safe.
"""

import math

import pytest

from control.pid_controller import LongitudinalController


FLOOR_STATES = ("ACC_ACTIVE", "CUTOUT")


def _build_controller(**overrides) -> LongitudinalController:
    params = {
        "kp": 0.3,
        "ki": 0.0,
        "kd": 0.0,
        "target_speed": 12.0,
        "max_speed": 15.0,
        "max_accel": 2.0,
        "max_decel": 6.0,
        "max_jerk": 100.0,
        "throttle_rate_limit": 5.0,
        "brake_rate_limit": 5.0,
        "speed_smoothing_alpha": 0.0,
        "speed_for_jerk_alpha": 0.0,
        "throttle_smoothing_alpha": 0.0,
        "continuous_accel_control": False,
        "limiter_transition_enabled": False,  # isolate the min-gate
    }
    params.update(overrides)
    return LongitudinalController(**params)


def _drive_positive(ctrl: LongitudinalController, **kw):
    """
    Drive controller with an underspeed condition so the speed-P loop
    produces a positive accel_cmd_raw (cruise toward target).
    Common defaults can be overridden via kwargs.
    """
    base = dict(
        current_speed=5.0,
        reference_velocity=8.0,
        dt=0.1,
        acc_state_code="ACC_ACTIVE",
    )
    base.update(kw)
    return ctrl.compute_control(**base)


def test_floor_off_mode_noop():
    """Mode=off: accel_cmd_raw unchanged; telemetry stays at defaults."""
    ctrl = _build_controller()
    _drive_positive(
        ctrl,
        idm_accel_mps2=-5.0,
        acc_idm_accel_floor_mode="off",
        acc_idm_accel_floor_states=FLOOR_STATES,
        acc_idm_accel_floor_min_negative=-0.25,
    )
    assert ctrl.last_accel_cmd_raw > 0.0  # speed-P is still producing +accel
    assert ctrl.last_acc_idm_floor_active == 0
    assert math.isnan(ctrl.last_acc_idm_floor_candidate_mps2)


def test_floor_shadow_mode_records_but_does_not_apply():
    """Mode=shadow: telemetry reports would-have-applied, but accel_cmd_raw unchanged."""
    ctrl = _build_controller()
    _drive_positive(
        ctrl,
        idm_accel_mps2=-3.0,
        acc_idm_accel_floor_mode="shadow",
        acc_idm_accel_floor_states=FLOOR_STATES,
        acc_idm_accel_floor_min_negative=-0.25,
    )
    # The candidate is recorded; the active flag fires; but accel_cmd_raw is
    # still the speed-P-loop's positive value, NOT -3.0.
    assert ctrl.last_acc_idm_floor_active == 1
    assert ctrl.last_acc_idm_floor_candidate_mps2 == pytest.approx(-3.0)
    assert ctrl.last_accel_cmd_raw > 0.0, "shadow must not mutate accel_cmd_raw"


def test_floor_active_clips_to_idm():
    """Mode=active with idm<<0 while speed-P wants +accel: clip down to idm."""
    ctrl = _build_controller()
    throttle, brake = _drive_positive(
        ctrl,
        idm_accel_mps2=-3.0,
        acc_idm_accel_floor_mode="active",
        acc_idm_accel_floor_states=FLOOR_STATES,
        acc_idm_accel_floor_min_negative=-0.25,
    )
    assert ctrl.last_acc_idm_floor_active == 1
    assert ctrl.last_accel_cmd_raw == pytest.approx(-3.0)
    # Back-propagation: throttle=0, brake solves accel = -brake * max_decel
    # With max_decel=6.0 and accel=-3.0, brake should be 0.5 (pre-limiter).
    assert throttle == pytest.approx(0.0)
    assert brake == pytest.approx(-(-3.0) / 6.0, rel=1e-3)


def test_floor_respects_state_list():
    """If state is not in floor_states (e.g. EMERGENCY_BRAKE), gate is inactive."""
    ctrl = _build_controller()
    _drive_positive(
        ctrl,
        idm_accel_mps2=-5.0,
        acc_state_code="EMERGENCY_BRAKE",  # B1 owns this path
        acc_idm_accel_floor_mode="active",
        acc_idm_accel_floor_states=FLOOR_STATES,
        acc_idm_accel_floor_min_negative=-0.25,
    )
    assert ctrl.last_acc_idm_floor_active == 0
    assert ctrl.last_accel_cmd_raw > 0.0  # unchanged — outside gate states


def test_floor_respects_dead_band():
    """idm is negative but above the dead-band: gate stays inactive."""
    ctrl = _build_controller()
    _drive_positive(
        ctrl,
        idm_accel_mps2=-0.1,                     # closer to 0 than min_negative
        acc_idm_accel_floor_mode="active",
        acc_idm_accel_floor_states=FLOOR_STATES,
        acc_idm_accel_floor_min_negative=-0.25,
    )
    assert ctrl.last_acc_idm_floor_active == 0
    assert ctrl.last_accel_cmd_raw > 0.0


def test_floor_no_accel_above_speed_p():
    """Cruise state: idm>0, speed_p>0; min-gate is a no-op (never ADDS authority)."""
    ctrl = _build_controller()
    _drive_positive(
        ctrl,
        idm_accel_mps2=+1.5,                     # cruise, IDM concurs
        acc_idm_accel_floor_mode="active",
        acc_idm_accel_floor_states=FLOOR_STATES,
        acc_idm_accel_floor_min_negative=-0.25,
    )
    assert ctrl.last_acc_idm_floor_active == 0
    assert ctrl.last_accel_cmd_raw > 0.0, "positive IDM must not replace positive speed-P"


def test_floor_g2_onset_1_replay():
    """
    Synthesize the f363-370 IDM sequence from Phase 0.5 evidence.
    In active mode, accel_cmd_raw must be <= idm at every frame where idm is below
    dead-band.
    """
    ctrl = _build_controller()
    idm_sequence = [-2.08, -2.55, -3.00, -3.43, -4.18, -4.82, -5.20, -5.67]

    for i, idm in enumerate(idm_sequence):
        ctrl.compute_control(
            current_speed=5.5 + 0.07 * i,         # speed rising 5.5 → ~6.0
            reference_velocity=8.0,
            dt=0.1,
            acc_state_code="ACC_ACTIVE",
            idm_accel_mps2=idm,
            acc_idm_accel_floor_mode="active",
            acc_idm_accel_floor_states=FLOOR_STATES,
            acc_idm_accel_floor_min_negative=-0.25,
        )
        assert ctrl.last_acc_idm_floor_active == 1, (
            f"frame {i} (idm={idm}): gate should be active"
        )
        # accel_cmd_raw should equal idm at each frame (idm<speed_p and idm<dead-band)
        assert ctrl.last_accel_cmd_raw == pytest.approx(idm, abs=1e-6), (
            f"frame {i}: expected accel={idm}, got {ctrl.last_accel_cmd_raw}"
        )


def test_floor_compatible_with_emergency_relax():
    """
    Both floor (ACC_ACTIVE/CUTOUT) and emergency_relax_enabled
    (EMERGENCY_BRAKE/TTC_ESTOP/COLLAPSED_GAP_STOP) refer to disjoint state
    sets; enabling both must not double-apply or cross-contaminate.
    In ACC_ACTIVE with emergency_relax_enabled=True, the emergency predicate
    is False so only the floor path fires.
    """
    ctrl = _build_controller()
    _drive_positive(
        ctrl,
        idm_accel_mps2=-3.0,
        acc_state_code="ACC_ACTIVE",  # in floor states, NOT in emergency states
        emergency_relax_enabled=True,
        emergency_decel_floor=-12.0,
        emergency_jerk_max=10.0,
        acc_idm_accel_floor_mode="active",
        acc_idm_accel_floor_states=FLOOR_STATES,
        acc_idm_accel_floor_min_negative=-0.25,
    )
    # Floor fires because state=ACC_ACTIVE is in floor_states
    assert ctrl.last_acc_idm_floor_active == 1
    assert ctrl.last_accel_cmd_raw == pytest.approx(-3.0)


def test_floor_cutout_state_also_gated():
    """Both ACC_ACTIVE and CUTOUT must trigger the gate (covers onset-2)."""
    ctrl = _build_controller()
    _drive_positive(
        ctrl,
        current_speed=2.0,
        reference_velocity=8.0,
        idm_accel_mps2=-2.35,                    # onset-2 terminal IDM value
        acc_state_code="CUTOUT",
        acc_idm_accel_floor_mode="active",
        acc_idm_accel_floor_states=FLOOR_STATES,
        acc_idm_accel_floor_min_negative=-0.25,
    )
    assert ctrl.last_acc_idm_floor_active == 1
    assert ctrl.last_accel_cmd_raw == pytest.approx(-2.35)
