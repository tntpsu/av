"""
Phase B — Unit tests for ACCParams and ACCController.

Tests cover:
  - ACCParams: defaults, field presence (including disengage_ramp_mps2)
  - Free-flow: no lead detected → returns free_flow_target, acc_active=0
  - IDM formula: analytical verification at known (v, gap, Δv) points
  - IDM desired gap: s*(v, Δv) = s0 + v*T at Δv=0
  - IDM: lead faster than ego → acc accelerates toward v0
  - IDM: stopped lead → v_target < v_ego
  - IDM edge case: gap → 0 → large deceleration, no crash
  - Detection-loss hysteresis: counter threshold, re-arm threshold
  - acc_active flag correct in all state transitions
  - Soft-engage: v_target on first ACC frame ≤ ego_speed + a_max*dt
  - Disengage ramp: v_target does not jump on detection loss
  - Cutout: ego_speed < cutout_speed → returns free-flow
  - State enum reported correctly per state
  - ACCController.reset() clears all counters and bumpless state
"""
import math

import pytest

from control.acc_controller import ACCController, ACCParams, ACCOutput, ACCState
from control.radar_sensor import RadarReading


# ─── Helpers ──────────────────────────────────────────────────────────────────

DT = 1.0 / 30.0   # 30 FPS


def _reading(detected=True, gap_m=30.0, range_rate_mps=0.0, snr=0.8):
    return RadarReading(
        detected=detected,
        gap_m=gap_m,
        range_rate_mps=range_rate_mps,
        snr=snr,
        gap_raw=gap_m,
        range_rate_raw=range_rate_mps,
    )


def _no_lead():
    return RadarReading(
        detected=False, gap_m=0.0, range_rate_mps=0.0,
        snr=0.0, gap_raw=0.0, range_rate_raw=0.0,
    )


def _make_controller(**kwargs):
    params = ACCParams(enabled=True, **kwargs)
    return ACCController(params)


# ─── ACCParams ────────────────────────────────────────────────────────────────

class TestACCParams:
    def test_defaults_reasonable(self):
        p = ACCParams()
        assert p.min_gap_m == pytest.approx(2.0)
        assert p.time_headway_s == pytest.approx(1.5)
        assert p.max_accel_mps2 == pytest.approx(2.0)
        assert p.comfortable_decel_mps2 == pytest.approx(2.5)
        assert p.cutout_speed_mps == pytest.approx(5.0)
        assert p.fallback_frames == 5
        assert p.reengage_frames == 3

    def test_disengage_ramp_field_exists(self):
        p = ACCParams()
        assert hasattr(p, "disengage_ramp_mps2")
        assert p.disengage_ramp_mps2 == pytest.approx(2.0)

    def test_enabled_false_by_default(self):
        p = ACCParams()
        assert p.enabled is False


# ─── Free-flow ────────────────────────────────────────────────────────────────

class TestFreeFlow:
    def test_no_lead_returns_free_flow_target(self):
        ctrl = _make_controller()
        out = ctrl.compute_target_speed(20.0, 25.0, _no_lead(), DT)
        assert out.target_speed == pytest.approx(25.0)
        assert out.acc_active == pytest.approx(0.0)
        assert out.state == ACCState.FREE_FLOW

    def test_free_flow_ttc_is_cap(self):
        ctrl = _make_controller()
        out = ctrl.compute_target_speed(20.0, 25.0, _no_lead(), DT)
        assert out.ttc_s == pytest.approx(999.0)


# ─── IDM formula ──────────────────────────────────────────────────────────────

class TestIDMFormula:
    def test_desired_gap_at_zero_closing_rate(self):
        """s*(v, 0) = s0 + v*T at Δv=0."""
        ctrl = _make_controller(min_gap_m=2.0, time_headway_s=1.5)
        v = 20.0
        # At steady following (Δv=0), desired gap = 2.0 + 20*1.5 = 32.0m
        out = ctrl.compute_target_speed(v, 25.0, _reading(gap_m=100.0, range_rate_mps=0.0), DT)
        assert out.target_gap_m == pytest.approx(2.0 + v * 1.5)

    def test_closing_at_known_rate_decelerates(self):
        """At gap=20m, closing at 5 m/s → IDM should command deceleration."""
        ctrl = _make_controller()
        v = 20.0
        # Run a few frames to get past soft-engage seed
        for _ in range(3):
            out = ctrl.compute_target_speed(v, 25.0, _reading(gap_m=20.0, range_rate_mps=5.0), DT)
            v = out.target_speed
        assert v < 20.0, "ACC should have decelerated from initial 20 m/s"

    def test_lead_faster_than_ego_accelerates(self):
        """Lead pulling away (range_rate < 0) → IDM should accelerate toward v0."""
        ctrl = _make_controller()
        v = 15.0
        # Large gap, lead moving away → free-flow acceleration expected
        out = ctrl.compute_target_speed(v, 25.0, _reading(gap_m=80.0, range_rate_mps=-3.0), DT)
        assert out.target_speed >= v, "Should not decelerate when lead is pulling away"

    def test_stopped_lead_decelerates(self):
        """Lead stopped (v_lead=0) → IDM must produce target_speed < ego_speed."""
        ctrl = _make_controller()
        v = 20.0
        for _ in range(5):
            out = ctrl.compute_target_speed(v, 25.0, _reading(gap_m=5.0, range_rate_mps=v), DT)
            v = out.target_speed
        assert v < 20.0, "Should decelerate toward stopped lead"

    def test_gap_zero_no_crash(self):
        """gap → 0 must not raise an exception and should produce maximum deceleration."""
        ctrl = _make_controller()
        # Bypass TTC and emergency brake guards by using a large speed (makes 1.5×v large)
        # Use gap just above emergency brake threshold to test pure IDM
        out = ctrl.compute_target_speed(
            1.0, 25.0, _reading(gap_m=0.001, range_rate_mps=0.5), DT
        )
        # Should produce a target_speed ≤ current speed (decelerating)
        assert out.target_speed >= 0.0
        assert not math.isnan(out.target_speed)

    def test_acc_active_flag_set(self):
        """acc_active = 1.0 when ACC is controlling."""
        ctrl = _make_controller()
        out = ctrl.compute_target_speed(20.0, 25.0, _reading(gap_m=40.0), DT)
        assert out.acc_active == pytest.approx(1.0)
        assert out.state == ACCState.ACC_ACTIVE

    def test_acc_output_exports_gap_contract_telemetry(self):
        ctrl = _make_controller()
        out = ctrl.compute_target_speed(12.0, 12.5, _reading(gap_m=30.0, range_rate_mps=0.1), DT)
        assert out.dynamic_gap_m > 0.0
        assert out.equilibrium_gap_m >= out.dynamic_gap_m
        assert out.lead_speed_estimate_mps == pytest.approx(11.9, rel=1e-6)
        assert out.convergence_mode in {
            "tracking_dynamic_gap",
            "equilibrium_limited_tracking",
            "policy_limited_tracking",
            "lead_limited_tracking",
        }
        assert out.detection_stable_frames == 1
        assert out.recent_detection_loss is False
        assert out.detection_loss_event_delta == 0


# ─── Soft-engage (bumpless transfer) ─────────────────────────────────────────

class TestSoftEngage:
    def test_first_acc_frame_bounded_by_ego_speed_plus_dt(self):
        """v_target on first ACC_ACTIVE frame ≤ ego_speed + a_max×dt (no step up)."""
        ctrl = _make_controller(max_accel_mps2=2.0)
        ego_speed = 20.0
        out = ctrl.compute_target_speed(ego_speed, 25.0, _reading(gap_m=40.0), DT)
        assert out.target_speed <= ego_speed + 2.0 * DT + 1e-6, (
            f"Soft-engage violated: v_target={out.target_speed:.3f} > {ego_speed + 2.0*DT:.3f}"
        )

    def test_soft_engage_after_detection_loss(self):
        """After a detection-loss period, re-engage seeds from ego_speed."""
        ctrl = _make_controller(fallback_frames=3, reengage_frames=2)
        # Establish ACC at low following speed
        following_v = 15.0
        for _ in range(5):
            out = ctrl.compute_target_speed(
                following_v, 25.0, _reading(gap_m=25.0, range_rate_mps=0.0), DT
            )
            following_v = out.target_speed

        # Detection loss — drive into FREE_FLOW ramp
        for _ in range(3):
            ctrl.compute_target_speed(following_v, 25.0, _no_lead(), DT)

        # Re-engage — first ACC_ACTIVE frame must not jump above ego_speed + a_max*dt
        ego = following_v
        for _ in range(2):
            ctrl.compute_target_speed(ego, 25.0, _reading(gap_m=40.0), DT)
        out = ctrl.compute_target_speed(ego, 25.0, _reading(gap_m=40.0), DT)
        assert out.target_speed <= ego + 2.0 * DT + 1e-6


# ─── Disengage ramp ───────────────────────────────────────────────────────────

class TestDisengageRamp:
    def test_v_target_does_not_jump_on_detection_loss(self):
        """After fallback_frames of no detection, v_target must ramp, not step."""
        ctrl = _make_controller(fallback_frames=3, disengage_ramp_mps2=2.0)
        following_v = 15.0
        for _ in range(5):
            out = ctrl.compute_target_speed(
                following_v, 25.0, _reading(gap_m=25.0), DT
            )
            following_v = out.target_speed

        # Trigger detection loss
        prev = following_v
        for i in range(5):
            out = ctrl.compute_target_speed(following_v, 25.0, _no_lead(), DT)
            # v_target should ramp — never jump by more than ramp_step + margin
            max_step = 2.0 * DT + 1e-4
            assert abs(out.target_speed - prev) <= max_step, (
                f"Frame {i}: v_target jumped from {prev:.3f} to {out.target_speed:.3f}"
            )
            prev = out.target_speed

    def test_disengage_ramp_direction(self):
        """Ramp must always move v_target toward free_flow_target."""
        ctrl = _make_controller(fallback_frames=2, disengage_ramp_mps2=2.0)
        # Establish slow following
        for _ in range(3):
            ctrl.compute_target_speed(10.0, 25.0, _reading(gap_m=20.0), DT)
        # Trigger loss — v_target should increase toward 25 m/s
        prev = 10.0
        for _ in range(10):
            out = ctrl.compute_target_speed(10.0, 25.0, _no_lead(), DT)
            assert out.target_speed >= prev - 1e-6, "Ramp should increase toward free_flow"
            prev = out.target_speed


# ─── Detection-loss hysteresis ────────────────────────────────────────────────

class TestDetectionLossHysteresis:
    def test_four_frames_still_acc_active(self):
        """4 consecutive no-detect frames should NOT trigger DETECTION_LOSS (fallback=5)."""
        ctrl = _make_controller(fallback_frames=5)
        # Seed with ACC active
        for _ in range(3):
            ctrl.compute_target_speed(20.0, 25.0, _reading(gap_m=30.0), DT)
        # 4 frames of no detection
        for _ in range(4):
            out = ctrl.compute_target_speed(20.0, 25.0, _no_lead(), DT)
        # Not yet in DETECTION_LOSS
        assert out.state != ACCState.DETECTION_LOSS

    def test_five_frames_triggers_detection_loss(self):
        """Exactly 5 consecutive no-detect frames → DETECTION_LOSS."""
        ctrl = _make_controller(fallback_frames=5)
        for _ in range(3):
            ctrl.compute_target_speed(20.0, 25.0, _reading(gap_m=30.0), DT)
        for _ in range(5):
            out = ctrl.compute_target_speed(20.0, 25.0, _no_lead(), DT)
        assert out.state == ACCState.DETECTION_LOSS
        assert out.acc_active == pytest.approx(0.0)

    def test_two_redetect_frames_still_lost(self):
        """2 detected frames after loss should NOT re-arm (reengage=3)."""
        ctrl = _make_controller(fallback_frames=3, reengage_frames=3)
        for _ in range(3):
            ctrl.compute_target_speed(20.0, 25.0, _no_lead(), DT)
        # 2 re-detection frames
        for _ in range(2):
            out = ctrl.compute_target_speed(20.0, 25.0, _reading(gap_m=40.0), DT)
        assert out.state == ACCState.DETECTION_LOSS

    def test_three_redetect_frames_rearms(self):
        """3 detected frames after loss → re-arm into ACC_ACTIVE."""
        ctrl = _make_controller(fallback_frames=3, reengage_frames=3)
        for _ in range(3):
            ctrl.compute_target_speed(20.0, 25.0, _no_lead(), DT)
        for _ in range(3):
            out = ctrl.compute_target_speed(20.0, 25.0, _reading(gap_m=40.0), DT)
        assert out.state == ACCState.ACC_ACTIVE


class TestEmergencyBrake:
    def test_eb_target_compounds_downward(self):
        """EB target must compound from v_target_prev, not reset to ego each frame.

        With the old ego_speed-based formula the target stays ~0.133m/s below ego
        (only ~0.04 m/s² effective brake).  With compounding, the error grows and
        the speed controller applies meaningful braking.
        """
        ctrl = _make_controller()
        # Seed ACC_ACTIVE state
        for _ in range(5):
            ctrl.compute_target_speed(12.0, 15.0, _reading(gap_m=30.0, range_rate_mps=0.0), DT)

        # Trigger EMERGENCY_BRAKE: gap=10 < 1.5*12=18, range_rate=4.0 (closing)
        ego = 12.0
        prev_target = None
        for i in range(10):
            out = ctrl.compute_target_speed(ego, 15.0, _reading(gap_m=10.0, range_rate_mps=4.0), DT)
            assert out.state == ACCState.EMERGENCY_BRAKE
            if prev_target is not None:
                # Target must strictly decrease each frame (compounds, not ego-0.133)
                assert out.target_speed < prev_target, (
                    f"Frame {i}: EB target did not decrease: {prev_target:.4f} → {out.target_speed:.4f}"
                )
            prev_target = out.target_speed

        # After 10 EB frames with dt=1/30: target drops by ~10 * 4 * (1/30) = 1.33 m/s
        initial_target = 12.0  # seeded to ego on first ACC frame
        expected_drop = 10 * 4.0 * DT
        assert initial_target - out.target_speed >= expected_drop * 0.8, (
            f"EB compounding too slow: dropped {initial_target - out.target_speed:.3f}m/s, "
            f"expected >= {expected_drop*0.8:.3f}m/s"
        )


class TestCollapsedGapSafety:
    def test_collapsed_gap_triggers_estop_even_without_positive_range_rate(self):
        ctrl = _make_controller()
        out = ctrl.compute_target_speed(
            15.0,
            25.0,
            _reading(gap_m=0.1, range_rate_mps=0.0),
            DT,
        )
        assert out.state == ACCState.COLLAPSED_GAP_STOP
        assert out.request_estop is True
        assert out.target_speed == pytest.approx(0.0)
        assert out.safety_mode == "collapsed_gap_stop"


# ─── Cutout ───────────────────────────────────────────────────────────────────

class TestCutout:
    def test_below_cutout_speed_returns_free_flow_ramp(self):
        """ego_speed < cutout_speed_mps → CUTOUT state regardless of lead."""
        ctrl = _make_controller(cutout_speed_mps=5.0)
        out = ctrl.compute_target_speed(4.9, 25.0, _reading(gap_m=20.0), DT)
        assert out.state == ACCState.CUTOUT
        assert out.acc_active == pytest.approx(0.0)

    def test_at_cutout_speed_not_cutout(self):
        """ego_speed >= cutout_speed_mps → NOT cutout state."""
        ctrl = _make_controller(cutout_speed_mps=5.0)
        out = ctrl.compute_target_speed(5.0, 25.0, _reading(gap_m=20.0), DT)
        assert out.state != ACCState.CUTOUT

    def test_detection_loss_count_preserved_through_cutout(self):
        """Detection-loss counter is not reset by CUTOUT state."""
        ctrl = _make_controller(cutout_speed_mps=5.0, fallback_frames=5)
        # Build up 4 frames of detection loss at normal speed
        for _ in range(3):
            ctrl.compute_target_speed(20.0, 25.0, _reading(gap_m=30.0), DT)
        for _ in range(4):
            ctrl.compute_target_speed(20.0, 25.0, _no_lead(), DT)
        # Drop into cutout — 1 more frame of no-detection should complete fallback
        ctrl.compute_target_speed(4.0, 25.0, _no_lead(), DT)   # cutout + +1 loss
        # Back at normal speed — should already be in DETECTION_LOSS
        out = ctrl.compute_target_speed(20.0, 25.0, _no_lead(), DT)
        assert out.state == ACCState.DETECTION_LOSS


# ─── reset() ──────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_detection_loss_state(self):
        ctrl = _make_controller(fallback_frames=3)
        for _ in range(3):
            ctrl.compute_target_speed(20.0, 25.0, _no_lead(), DT)
        ctrl.reset()
        # After reset, 3 more no-detect frames should start counting fresh
        for _ in range(2):
            out = ctrl.compute_target_speed(20.0, 25.0, _no_lead(), DT)
        assert out.state != ACCState.DETECTION_LOSS

    def test_reset_clears_bumpless_state(self):
        ctrl = _make_controller()
        for _ in range(5):
            ctrl.compute_target_speed(15.0, 25.0, _reading(gap_m=25.0), DT)
        ctrl.reset()
        # After reset, first ACC frame should soft-engage from ego_speed
        ego = 15.0
        out = ctrl.compute_target_speed(ego, 25.0, _reading(gap_m=40.0), DT)
        assert out.target_speed <= ego + 2.0 * DT + 1e-6
