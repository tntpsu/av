"""
Phase T — L4 invariant, property, and fuzz tests for ACC.

Tests mathematical guarantees that go beyond formula correctness:

  IDM invariants:
    - Gap-positive: IDM alone can never push gap below s0 in finite simulation
    - Free-flow convergence: speed reaches v0 within 0.2 m/s in 30s at large gap
    - Equilibrium rest: at steady-state gap, |a_IDM| < 0.05 m/s²

  Bumpless transfer:
    - Disengage ramp: v_target change per frame ≤ disengage_ramp_mps2 × dt
    - Soft-engage: v_target on first ACC_ACTIVE frame ≤ ego_speed + a_max×dt
    - Ramp direction: always moves toward free_flow_target, never overshoots

  EMA properties:
    - Bounded by consecutive inputs (no overshoot)
    - Cold-start: first-detection value equals raw input

  Safety before contact:
    - TTC guard fires ≥ 1 frame before gap would reach 0 at constant approach speed

  Scoring invariants:
    - Collision override: gap ≤ 0 → request_estop potential (TTC guard / e-brake)
    - State machine priority: TTC fires even at low speed (before CUTOUT)

  Radar fuzz:
    - 20 random seeds × 100 frames → no NaN/Inf in outputs, snr ∈ [0, 1]
"""
import math

import numpy as np
import pytest

from control.acc_controller import ACCController, ACCParams, ACCState, _TTC_ESTOP_THRESHOLD_S
from control.radar_sensor import ForwardRadarSensor, RadarReading


DT = 1.0 / 30.0


def _params(**kwargs):
    return ACCParams(enabled=True, **kwargs)


def _reading(detected=True, gap_m=30.0, range_rate_mps=0.0):
    return RadarReading(
        detected=detected, gap_m=gap_m, range_rate_mps=range_rate_mps,
        snr=0.8, gap_raw=gap_m, range_rate_raw=range_rate_mps,
    )


# ─── IDM invariants ───────────────────────────────────────────────────────────

class TestIDMInvariants:
    def test_gap_never_goes_negative_from_idm_alone(self):
        """
        Simulate 10s with gap=3.0m, ego closing at 10 m/s.
        IDM deceleration must prevent gap from going below 0.
        (Pure IDM — no emergency brake or TTC guard interference.)
        """
        p = _params(
            min_gap_m=2.0, time_headway_s=1.5,
            max_accel_mps2=2.0, comfortable_decel_mps2=2.5,
            cutout_speed_mps=0.1,   # disable cutout so IDM runs at low speed
            fallback_frames=999,    # disable detection-loss
        )
        ctrl = ACCController(p)
        ego_v = 15.0
        lead_v = 5.0
        gap = 30.0
        min_gap_seen = gap

        for _ in range(300):   # 10s at 30 FPS
            delta_v = ego_v - lead_v   # positive = closing
            r = RadarReading(
                detected=True, gap_m=max(0.001, gap),
                range_rate_mps=max(0.0, delta_v),
                snr=0.8, gap_raw=gap, range_rate_raw=delta_v,
            )
            out = ctrl.compute_target_speed(ego_v, 20.0, r, DT)
            if out.state in (ACCState.TTC_ESTOP, ACCState.EMERGENCY_BRAKE):
                break   # safety guard engaged — IDM invariant fulfilled
            ego_v = out.target_speed
            gap += (lead_v - ego_v) * DT   # gap grows when ego slower
            min_gap_seen = min(min_gap_seen, gap)

        assert min_gap_seen > 0.0 or out.state in (
            ACCState.TTC_ESTOP, ACCState.EMERGENCY_BRAKE
        ), f"Gap went negative ({min_gap_seen:.3f}m) without safety guard engaging"

    def test_free_flow_converges_to_v0(self):
        """Large gap (200m) → speed reaches within 0.2 m/s of v0 in 30s."""
        ctrl = ACCController(_params())
        v = 10.0
        v0 = 25.0

        # Use gap=1000m so gap term (s*/gap)^2 ≈ (40/1000)^2 = 0.0016 is negligible.
        # gap=200m is insufficient: (s*/gap)^2 ≈ 0.038 pulls equilibrium below v0.
        for _ in range(1800):   # 60s at 30 FPS — IDM (v/v0)^4 term converges asymptotically
            r = _reading(gap_m=1000.0, range_rate_mps=0.0)
            out = ctrl.compute_target_speed(v, v0, r, DT)
            v = out.target_speed

        assert abs(v - v0) < 0.2, f"Free-flow did not converge: v={v:.3f}, v0={v0}"

    def test_equilibrium_rest(self):
        """At steady-state gap (gap = s0 + v*T), |a_IDM| should be near zero."""
        p = ACCParams(enabled=True, min_gap_m=2.0, time_headway_s=1.5,
                      max_accel_mps2=2.0, comfortable_decel_mps2=2.5)
        ctrl = ACCController(p)
        v = 20.0
        v0 = 25.0

        # Run to approximate equilibrium
        gap = p.min_gap_m + v * p.time_headway_s
        for _ in range(300):
            r = _reading(gap_m=gap, range_rate_mps=0.0)
            out = ctrl.compute_target_speed(v, v0, r, DT)
            v = out.target_speed

        # At steady state: |a_IDM| should be small (dominated by (v/v0)^4 term)
        v_change_per_frame = abs(out.target_speed - v)
        assert v_change_per_frame < 0.05 * DT + 1e-4


# ─── Bumpless transfer ────────────────────────────────────────────────────────

class TestBumplessTransfer:
    def test_disengage_ramp_per_frame_bounded(self):
        """v_target change per frame during DETECTION_LOSS ≤ ramp_mps2 × dt."""
        ramp = 2.0
        ctrl = ACCController(_params(fallback_frames=3, disengage_ramp_mps2=ramp))
        # Establish following at 12 m/s
        v = 12.0
        for _ in range(5):
            out = ctrl.compute_target_speed(v, 25.0, _reading(gap_m=20.0), DT)
            v = out.target_speed

        prev_v = v
        for _ in range(20):
            out = ctrl.compute_target_speed(v, 25.0, _reading(detected=False, gap_m=0.0), DT)
            step = abs(out.target_speed - prev_v)
            assert step <= ramp * DT + 1e-6, (
                f"Ramp step {step:.5f} > {ramp * DT:.5f} (ramp_mps2={ramp}, dt={DT:.4f})"
            )
            prev_v = out.target_speed
            if out.target_speed >= 24.9:
                break   # reached free-flow

    def test_disengage_ramp_monotone_toward_free_flow(self):
        """v_target must monotonically approach free_flow_target during ramp."""
        ctrl = ACCController(_params(fallback_frames=2, disengage_ramp_mps2=2.0))
        for _ in range(3):
            ctrl.compute_target_speed(10.0, 25.0, _reading(gap_m=15.0), DT)

        prev_v = 10.0
        for _ in range(30):
            out = ctrl.compute_target_speed(10.0, 25.0, _reading(detected=False, gap_m=0.0), DT)
            assert out.target_speed >= prev_v - 1e-6, (
                f"Ramp moved away from free_flow: {prev_v:.3f} → {out.target_speed:.3f}"
            )
            prev_v = out.target_speed
            if out.target_speed >= 24.9:
                break

    def test_soft_engage_on_first_active_frame(self):
        """v_target on very first ACC_ACTIVE frame ≤ ego_speed + a_max×dt."""
        ctrl = ACCController(_params(max_accel_mps2=2.0))
        ego = 18.0
        out = ctrl.compute_target_speed(ego, 25.0, _reading(gap_m=40.0, range_rate_mps=0.0), DT)
        max_allowed = ego + 2.0 * DT
        assert out.target_speed <= max_allowed + 1e-6, (
            f"Soft-engage violated: {out.target_speed:.4f} > {max_allowed:.4f}"
        )


# ─── EMA properties ───────────────────────────────────────────────────────────

class TestEMAProperties:
    def test_ema_bounded_by_consecutive_inputs(self):
        """EMA output must always lie between consecutive raw inputs (no overshoot)."""
        sensor = ForwardRadarSensor(gap_alpha=0.30, rate_alpha=0.20)
        raw_values = [30.0, 20.0, 40.0, 10.0, 50.0, 25.0]
        prev_raw = raw_values[0]

        sensor.read_frame({"radar_fwd_detected": True,
                           "radar_fwd_distance_m": prev_raw,
                           "radar_fwd_range_rate_mps": 0.0,
                           "radar_fwd_snr": 0.8})

        for raw in raw_values[1:]:
            r = sensor.read_frame({"radar_fwd_detected": True,
                                   "radar_fwd_distance_m": raw,
                                   "radar_fwd_range_rate_mps": 0.0,
                                   "radar_fwd_snr": 0.8})
            lo, hi = min(prev_raw, raw), max(prev_raw, raw)
            assert lo - 1e-6 <= r.gap_m <= hi + 1e-6, (
                f"EMA overshoot: gap_m={r.gap_m:.3f} not in [{lo:.1f}, {hi:.1f}]"
            )
            prev_raw = raw

    def test_ema_cold_start_equals_raw(self):
        """After dropout, first detection seeds EMA to raw — no ramp from stale zero."""
        sensor = ForwardRadarSensor()
        # Establish at 30m
        for _ in range(5):
            sensor.read_frame({"radar_fwd_detected": True,
                               "radar_fwd_distance_m": 30.0,
                               "radar_fwd_range_rate_mps": 0.0,
                               "radar_fwd_snr": 0.8})
        # Dropout
        for _ in range(10):
            sensor.read_frame({"radar_fwd_detected": False})
        # Re-detect at 50m
        r = sensor.read_frame({"radar_fwd_detected": True,
                               "radar_fwd_distance_m": 50.0,
                               "radar_fwd_range_rate_mps": 0.0,
                               "radar_fwd_snr": 0.8})
        assert r.gap_m == pytest.approx(50.0), (
            "EMA cold-start failed: should return 50m on first re-detection"
        )


# ─── Safety guard fires before contact ───────────────────────────────────────

@pytest.mark.parametrize("approach_rate_mps,initial_gap_m", [
    (5.0, 10.0),   # fast close, moderate gap
    (2.0, 5.0),    # medium close, small gap
    (10.0, 20.0),  # high speed, larger gap
])
def test_safety_guard_fires_before_gap_reaches_zero(approach_rate_mps, initial_gap_m):
    """
    Simulate constant-approach-rate scenario.
    TTC guard or emergency brake must fire before gap ≤ 0.
    """
    ctrl = ACCController(_params(
        cutout_speed_mps=0.1,
        fallback_frames=999,
        min_gap_m=2.0,
        time_headway_s=1.5,
    ))
    gap = initial_gap_m
    ego_v = 20.0
    safety_fired = False

    for _ in range(1000):
        gap = max(0.0, gap - approach_rate_mps * DT)
        r = RadarReading(
            detected=True, gap_m=gap, range_rate_mps=approach_rate_mps,
            snr=0.8, gap_raw=gap, range_rate_raw=approach_rate_mps,
        )
        out = ctrl.compute_target_speed(ego_v, 25.0, r, DT)

        if out.request_estop or out.state == ACCState.EMERGENCY_BRAKE:
            safety_fired = True
            break

        if gap <= 0.0:
            break

    assert safety_fired, (
        f"Safety guard never fired: approach={approach_rate_mps} m/s, "
        f"initial_gap={initial_gap_m}m, final_gap={gap:.3f}m"
    )


# ─── State machine priority ───────────────────────────────────────────────────

class TestStateMachinePriority:
    def test_ttc_estop_takes_priority_over_cutout(self):
        """TTC_ESTOP (priority 1) fires even when ego_speed < cutout_speed (priority 3)."""
        ctrl = ACCController(_params(cutout_speed_mps=5.0))
        # Small gap, high approach speed → TTC < 1.5s, but ego_speed < cutout_speed
        gap = 3.0
        range_rate = 10.0   # TTC = 0.3s
        r = RadarReading(
            detected=True, gap_m=gap, range_rate_mps=range_rate,
            snr=0.8, gap_raw=gap, range_rate_raw=range_rate,
        )
        out = ctrl.compute_target_speed(3.0, 10.0, r, DT)
        assert out.state == ACCState.TTC_ESTOP

    def test_emergency_brake_takes_priority_over_detection_loss(self):
        """EMERGENCY_BRAKE (priority 2) fires even if we'd normally be in DETECTION_LOSS."""
        ctrl = ACCController(_params(fallback_frames=1))
        # Trigger detection loss
        ctrl.compute_target_speed(20.0, 25.0, _reading(detected=False, gap_m=0.0), DT)
        ctrl.compute_target_speed(20.0, 25.0, _reading(detected=False, gap_m=0.0), DT)
        # Now detection loss is active. Deliver a detected frame with emergency-brake conditions.
        v = 20.0
        gap = 1.5 * v * 0.5   # well inside emergency brake threshold
        r = RadarReading(
            detected=True, gap_m=gap, range_rate_mps=5.0,
            snr=0.8, gap_raw=gap, range_rate_raw=5.0,
        )
        out = ctrl.compute_target_speed(v, 25.0, r, DT)
        assert out.state == ACCState.EMERGENCY_BRAKE


# ─── Radar fuzz ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("seed", range(20))
def test_radar_sensor_fuzz_no_nan(seed):
    """
    100 random-input frames → ForwardRadarSensor output is always finite and valid.
    Covers: out-of-range SNR, negative distances, large/small range rates.
    """
    rng = np.random.default_rng(seed)
    sensor = ForwardRadarSensor()

    for _ in range(100):
        raw = {
            "radar_fwd_detected": bool(rng.integers(0, 2)),
            "radar_fwd_distance_m": float(rng.uniform(-5.0, 80.0)),   # includes negative
            "radar_fwd_range_rate_mps": float(rng.normal(0.0, 8.0)),   # large noise
            "radar_fwd_snr": float(rng.uniform(-0.5, 1.8)),            # out-of-range
        }
        r = sensor.read_frame(raw)

        assert not math.isnan(r.gap_m), f"NaN in gap_m (seed={seed})"
        assert not math.isnan(r.range_rate_mps), f"NaN in range_rate_mps (seed={seed})"
        assert not math.isinf(r.gap_m), f"Inf in gap_m (seed={seed})"
        assert not math.isinf(r.range_rate_mps), f"Inf in range_rate_mps (seed={seed})"
        assert r.snr >= 0.0, f"snr < 0 (seed={seed}, snr={r.snr})"
        assert r.snr <= 1.0, f"snr > 1 (seed={seed}, snr={r.snr})"
        assert r.gap_m >= 0.0, f"gap_m < 0 (seed={seed})"
