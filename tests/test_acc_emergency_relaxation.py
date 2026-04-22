"""
Unit tests for ACC emergency-state relaxation in LongitudinalController.

Validates acc-idm-accel-plumbing.md Commit C: when the ACC state machine
is in EMERGENCY_BRAKE / TTC_ESTOP / COLLAPSED_GAP_STOP AND relaxation is
enabled, the controller's effective_max_decel and dynamic_max_jerk widen
so the IDM-commanded -12 m/s² can actually reach the brake output
instead of being clipped to the nominal -3 m/s² floor.

These tests drive LongitudinalController.compute_control directly — no
Unity, no HDF5.
"""

from __future__ import annotations

import pytest

from control.pid_controller import LongitudinalController


def _make_controller() -> LongitudinalController:
    """Nominal controller with continuous_accel_control=true (matches live config)."""
    return LongitudinalController(
        target_speed=8.0,
        max_speed=12.0,
        max_accel=2.5,
        max_decel=3.0,
        max_jerk=6.0,
        max_jerk_min=1.5,
        max_jerk_max=6.0,
        accel_feedforward_gain=1.0,
        decel_feedforward_gain=1.0,
        continuous_accel_control=True,
        startup_ramp_seconds=0.0,
        startup_accel_limit=100.0,  # disable startup cap for test determinism
        startup_speed_threshold=0.0,
    )


def _step(
    ctrl: LongitudinalController,
    *,
    current_speed: float,
    reference_velocity: float,
    reference_accel: float,
    dt: float,
    acc_state_code: str = "",
    emergency_relax_enabled: bool = False,
    emergency_decel_floor: float = -3.0,
    emergency_jerk_max: float = 4.0,
) -> tuple[float, float]:
    return ctrl.compute_control(
        current_speed=current_speed,
        reference_velocity=reference_velocity,
        dt=dt,
        reference_accel=reference_accel,
        acc_state_code=acc_state_code,
        emergency_relax_enabled=emergency_relax_enabled,
        emergency_decel_floor=emergency_decel_floor,
        emergency_jerk_max=emergency_jerk_max,
    )


def _ramp_to_steady(ctrl: LongitudinalController, speed: float, dt: float) -> None:
    """Feed a nominal follow-mode state so the accel limiter is ready."""
    for _ in range(20):
        ctrl.compute_control(
            current_speed=speed,
            reference_velocity=speed,
            dt=dt,
            reference_accel=0.0,
        )


class TestEmergencyDecelWidening:
    def test_nominal_state_clips_decel_at_3mps2(self):
        """Baseline: without emergency label, IDM -12 m/s² gets clamped to ~-3."""
        ctrl = _make_controller()
        _ramp_to_steady(ctrl, 8.0, 0.033)
        # Many frames so jerk limiter has time to ramp down to -3
        brake_vals = []
        for _ in range(200):
            _, brake = _step(
                ctrl,
                current_speed=8.0,
                reference_velocity=8.0,
                reference_accel=-12.0,
                dt=0.033,
                acc_state_code="ACC_ACTIVE",
                emergency_relax_enabled=False,  # relax off
                emergency_decel_floor=-12.0,
                emergency_jerk_max=10.0,
            )
            brake_vals.append(brake)
        # With max_decel=3.0 and no relaxation, brake saturates around 1.0 (3/3=1.0)
        # but accel_cmd itself is clipped at -3.0 — so brake should NOT report >1 frames
        # meaningfully beyond that. The last_accel_cmd internal state should be -3.0, not -12.
        assert abs(ctrl.last_accel_cmd) <= 3.0 + 1e-3, (
            f"without relaxation, accel should clip to -3; got {ctrl.last_accel_cmd}"
        )

    def test_emergency_brake_widens_decel_to_12mps2(self):
        """EMERGENCY_BRAKE + relax=True → accel can reach -12 m/s²."""
        ctrl = _make_controller()
        _ramp_to_steady(ctrl, 8.0, 0.033)
        for _ in range(200):
            _step(
                ctrl,
                current_speed=8.0,
                reference_velocity=8.0,
                reference_accel=-12.0,
                dt=0.033,
                acc_state_code="EMERGENCY_BRAKE",
                emergency_relax_enabled=True,
                emergency_decel_floor=-12.0,
                emergency_jerk_max=10.0,
            )
        # With widening, accel_cmd should reach -12 (or very close) after ramp.
        assert ctrl.last_accel_cmd <= -10.0, (
            f"with relaxation, accel should approach -12; got {ctrl.last_accel_cmd}"
        )

    @pytest.mark.parametrize("state", ["TTC_ESTOP", "COLLAPSED_GAP_STOP"])
    def test_other_emergency_states_also_widen(self, state: str):
        """TTC_ESTOP and COLLAPSED_GAP_STOP also authorize widening."""
        ctrl = _make_controller()
        _ramp_to_steady(ctrl, 8.0, 0.033)
        for _ in range(200):
            _step(
                ctrl,
                current_speed=8.0,
                reference_velocity=8.0,
                reference_accel=-12.0,
                dt=0.033,
                acc_state_code=state,
                emergency_relax_enabled=True,
                emergency_decel_floor=-12.0,
                emergency_jerk_max=10.0,
            )
        assert ctrl.last_accel_cmd <= -10.0, (
            f"{state} with relaxation should reach -12; got {ctrl.last_accel_cmd}"
        )

    def test_non_exempt_state_does_not_widen(self):
        """CUTOUT is not exempt — decel should still clip at -3 even with relax=True."""
        ctrl = _make_controller()
        _ramp_to_steady(ctrl, 8.0, 0.033)
        for _ in range(200):
            _step(
                ctrl,
                current_speed=8.0,
                reference_velocity=8.0,
                reference_accel=-12.0,
                dt=0.033,
                acc_state_code="CUTOUT",
                emergency_relax_enabled=True,
                emergency_decel_floor=-12.0,
                emergency_jerk_max=10.0,
            )
        assert abs(ctrl.last_accel_cmd) <= 3.0 + 1e-3, (
            f"CUTOUT must NOT widen; got {ctrl.last_accel_cmd}"
        )

    def test_emergency_relax_disabled_blocks_widening(self):
        """Master switch off → even EMERGENCY_BRAKE does not widen."""
        ctrl = _make_controller()
        _ramp_to_steady(ctrl, 8.0, 0.033)
        for _ in range(200):
            _step(
                ctrl,
                current_speed=8.0,
                reference_velocity=8.0,
                reference_accel=-12.0,
                dt=0.033,
                acc_state_code="EMERGENCY_BRAKE",
                emergency_relax_enabled=False,  # master off
                emergency_decel_floor=-12.0,
                emergency_jerk_max=10.0,
            )
        assert abs(ctrl.last_accel_cmd) <= 3.0 + 1e-3, (
            f"relax=False must block widening; got {ctrl.last_accel_cmd}"
        )


class TestEmergencyJerkCooldownBypass:
    """
    Regression coverage for Commit C.1: during sustained emergency demand the
    jerk limiter fires every frame, refreshing jerk_cooldown_remaining. Without
    the emergency bypass, the per-frame `accel_cmd *= jerk_cooldown_scale` (0.4)
    pins accel_cmd near the nominal floor and defeats the Commit C widening.

    The default _make_controller() has jerk_cooldown_frames=0 so the base
    widening tests don't exercise this path — live config uses 8.
    """

    def _make_controller_with_cooldown(self) -> LongitudinalController:
        return LongitudinalController(
            target_speed=8.0,
            max_speed=12.0,
            max_accel=2.5,
            max_decel=3.0,
            max_jerk=6.0,
            max_jerk_min=1.5,
            max_jerk_max=6.0,
            accel_feedforward_gain=1.0,
            decel_feedforward_gain=1.0,
            continuous_accel_control=True,
            startup_ramp_seconds=0.0,
            startup_accel_limit=100.0,
            startup_speed_threshold=0.0,
            jerk_cooldown_frames=8,          # live-config value
            jerk_cooldown_scale=0.4,         # live-config value
        )

    def test_emergency_bypasses_jerk_cooldown(self):
        """With cooldown active (live config), emergency state must still reach -12."""
        ctrl = self._make_controller_with_cooldown()
        _ramp_to_steady(ctrl, 8.0, 0.033)
        # Sustained emergency demand. With the bypass, ~16 frames at dt=0.033 and
        # emergency_jerk_max=10 can cover the ~12 m/s² span; 200 frames is ample.
        for _ in range(200):
            _step(
                ctrl,
                current_speed=8.0,
                reference_velocity=8.0,
                reference_accel=-12.0,
                dt=0.033,
                acc_state_code="EMERGENCY_BRAKE",
                emergency_relax_enabled=True,
                emergency_decel_floor=-12.0,
                emergency_jerk_max=10.0,
            )
        assert ctrl.last_accel_cmd <= -10.0, (
            f"emergency state must bypass cooldown pin; got {ctrl.last_accel_cmd}"
        )

    def test_nominal_state_still_applies_cooldown(self):
        """Outside emergency, cooldown behaves exactly as before (0.4x multiplier)."""
        ctrl = self._make_controller_with_cooldown()
        _ramp_to_steady(ctrl, 8.0, 0.033)
        # Nominal state (no emergency label) — cooldown must still pin decel at
        # ~ -3 * 0.4 = -1.2 or oscillate near that; crucially it must NOT reach -10.
        for _ in range(200):
            _step(
                ctrl,
                current_speed=8.0,
                reference_velocity=8.0,
                reference_accel=-12.0,
                dt=0.033,
                acc_state_code="ACC_ACTIVE",
                emergency_relax_enabled=False,
                emergency_decel_floor=-3.0,
                emergency_jerk_max=4.0,
            )
        assert ctrl.last_accel_cmd > -10.0, (
            f"non-emergency cooldown must still throttle authority; got {ctrl.last_accel_cmd}"
        )


class TestEmergencyJerkWidening:
    def test_emergency_brake_widens_jerk_ceiling(self):
        """Per-frame accel step should be bounded by emergency_jerk_max*dt, not nominal."""
        ctrl_relax = _make_controller()
        ctrl_nominal = _make_controller()
        _ramp_to_steady(ctrl_relax, 8.0, 0.033)
        _ramp_to_steady(ctrl_nominal, 8.0, 0.033)
        # One step with large decel demand from steady state — compare how much
        # accel_cmd moves in ONE step between relax=True and relax=False.
        _step(
            ctrl_relax, current_speed=8.0, reference_velocity=8.0,
            reference_accel=-12.0, dt=0.033,
            acc_state_code="EMERGENCY_BRAKE",
            emergency_relax_enabled=True,
            emergency_decel_floor=-12.0, emergency_jerk_max=10.0,
        )
        _step(
            ctrl_nominal, current_speed=8.0, reference_velocity=8.0,
            reference_accel=-12.0, dt=0.033,
            acc_state_code="EMERGENCY_BRAKE",
            emergency_relax_enabled=False,
            emergency_decel_floor=-3.0, emergency_jerk_max=4.0,
        )
        # Relaxed controller should have moved further toward -12 in one step.
        assert ctrl_relax.last_accel_cmd < ctrl_nominal.last_accel_cmd, (
            f"relax should decel faster: relax={ctrl_relax.last_accel_cmd}, "
            f"nominal={ctrl_nominal.last_accel_cmd}"
        )
