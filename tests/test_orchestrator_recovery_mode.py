"""Integration tests for orchestrator recovery-mode post-limiter deletion.

The steering multiplier (1.2x / 1.5x applied AFTER the PP rate/jerk limiters)
was removed from `av_stack/orchestrator.py::_pf_apply_safety`. Its safety
intent is now served upstream by the proportional recovery term in
LateralController (tested in `test_pp_recovery_term.py`).

These tests verify three invariants:
  1. Under PP with high lateral error, the orchestrator no longer overwrites
     the steering command with a scaled version of itself.
  2. Throttle reductions (0.5x at hard-limit, 0.8x at recovery threshold) are
     retained — throttle has its own smoothing and "slow down when off-lane"
     is a valid safety behavior independent of the deleted steering clamp.
  3. The emergency-stop branch remains unchanged: a catastrophic e_lat still
     produces steering=0, brake=1.0.

Design reference: project_recovery_mode_post_limiter.md
"""

from types import SimpleNamespace

import pytest

from av_stack.orchestrator import AVStack
from control.regime_selector import ControlRegime


def _fake_self() -> SimpleNamespace:
    """Minimal `self` stand-in exposing every attribute `_pf_apply_safety` reads."""
    return SimpleNamespace(
        safety_config={
            "max_speed": 10.0,
            "max_lateral_error": 2.0,
            "recovery_lateral_error": 1.0,
            "emergency_stop_error": 3.0,
            "lane_width": 7.0,
            "car_width": 1.85,
            "allowed_outside_lane": 1.0,
            "emergency_stop_release_speed": 0.2,
            "emergency_stop_use_gt_lane_boundaries": False,
            "emergency_stop_gt_lane_boundary_margin": 0.05,
        },
        last_gt_lane_width=None,
        controller=SimpleNamespace(
            lateral_controller=SimpleNamespace(reset=lambda: None),
        ),
        frame_count=100,
        emergency_stop_logged=False,
        emergency_stop_type=None,
        emergency_stop_latched=False,
        emergency_stop_latched_since_wall_time=None,
        perception_frozen_frames=0,
        last_timestamp=1.0,
        last_timestamp_before_freeze=None,
        post_jump_cooldown_frames=0,
        last_teleport_distance=0.0,
        last_teleport_dt=None,
        running=True,
        _gt_boundary_corrupt_streak=0,
    )


def _args(
    *,
    steering: float,
    throttle: float,
    at_car_m: float,
    regime: ControlRegime = ControlRegime.PURE_PURSUIT,
) -> tuple:
    """Build the tuple (cmd, traj, gated, gov, vehicle_state, fv, ts)."""
    cmd = {
        "steering": steering,
        "throttle": throttle,
        "brake": 0.0,
        "lateral_error": at_car_m,
        "regime": int(regime),
    }
    traj = {
        "reference_point": {
            "x": at_car_m,
            "y": 8.0,
            "heading": 0.0,
            "gt_cross_track_road_frame_at_car_m": at_car_m,
        }
    }
    gated = {"current_speed": 8.0, "using_stale_data": False}
    gov = {}
    vehicle_state = {}
    fv = {"teleport_guard_active": False}
    ts = 1.0
    return cmd, traj, gated, gov, vehicle_state, fv, ts


class TestOrchestratorRecoveryModeDeletion:
    def test_pp_high_lateral_error_does_not_scale_steering(self) -> None:
        """Regression: orchestrator must no longer apply a 1.2x multiplier at
        lateral_error in the recovery band. The PP recovery term handles this
        upstream of the limiters."""
        fake = _fake_self()
        input_steering = 0.31
        cmd, traj, gated, gov, vs, fv, ts = _args(
            steering=input_steering, throttle=0.4, at_car_m=1.5
        )
        out = AVStack._pf_apply_safety(fake, cmd, traj, gated, gov, vs, fv, ts)
        # Recovery band (1.0 < e_lat < 2.0): steering untouched.
        assert out["steering"] == pytest.approx(input_steering, abs=1e-9)
        # 1.2x scaling would have produced 0.372; 1.5x would have produced 0.465.
        assert out["steering"] < 0.35, (
            f"steering={out['steering']} suggests post-limiter scaling is still active"
        )

    def test_pp_hard_limit_band_does_not_scale_steering(self) -> None:
        """Regression: orchestrator must no longer apply a 1.5x multiplier
        when e_lat > max_lateral_error."""
        fake = _fake_self()
        input_steering = 0.40
        cmd, traj, gated, gov, vs, fv, ts = _args(
            steering=input_steering, throttle=0.4, at_car_m=2.5
        )
        out = AVStack._pf_apply_safety(fake, cmd, traj, gated, gov, vs, fv, ts)
        assert out["steering"] == pytest.approx(input_steering, abs=1e-9)
        # 1.5x scaling would have produced 0.60.
        assert out["steering"] < 0.45, (
            f"steering={out['steering']} suggests hard-limit multiplier is still active"
        )

    def test_pp_recovery_band_still_reduces_throttle(self) -> None:
        """Confirm the retained behavior: throttle is scaled 0.8x in the
        recovery band and 0.5x in the hard-limit band."""
        # Recovery band.
        fake = _fake_self()
        cmd, traj, gated, gov, vs, fv, ts = _args(
            steering=0.2, throttle=0.4, at_car_m=1.5
        )
        out = AVStack._pf_apply_safety(fake, cmd, traj, gated, gov, vs, fv, ts)
        assert out["throttle"] == pytest.approx(0.4 * 0.8, rel=1e-6)

        # Hard-limit band.
        fake2 = _fake_self()
        cmd2, traj2, gated2, gov2, vs2, fv2, ts2 = _args(
            steering=0.2, throttle=0.4, at_car_m=2.5
        )
        out2 = AVStack._pf_apply_safety(fake2, cmd2, traj2, gated2, gov2, vs2, fv2, ts2)
        assert out2["throttle"] == pytest.approx(0.4 * 0.5, rel=1e-6)

    def test_emergency_stop_branch_unchanged(self) -> None:
        """Catastrophic e_lat (>emergency_stop_error=3.0m) must still produce
        a full stop regardless of the multiplier deletion."""
        fake = _fake_self()
        cmd, traj, gated, gov, vs, fv, ts = _args(
            steering=0.9, throttle=0.6, at_car_m=4.0
        )
        out = AVStack._pf_apply_safety(fake, cmd, traj, gated, gov, vs, fv, ts)
        assert out["steering"] == pytest.approx(0.0, abs=1e-9)
        assert out["throttle"] == pytest.approx(0.0, abs=1e-9)
        assert out["brake"] == pytest.approx(1.0, abs=1e-9)
        assert out.get("emergency_stop") is True
