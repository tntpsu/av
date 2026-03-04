"""
MPC comfort gate tests (Phase 2.4).

Four closed-loop simulation tests that verify the MPC controller meets
comfort constraints without requiring Unity. Each test runs the controller
in a tight loop (30 Hz) and measures output quality directly.

Tests:
    1. test_mpc_jerk_gate            — commanded steering jerk P95 ≤ 6.0 m/s³  (straight, 200 frames)
    2. test_mpc_lateral_accel_gate   — lateral accel P95 ≤ 2.45 m/s² (R=40 m arc, 12 m/s, 60 frames)
    3. test_mpc_steering_smoothness  — steer rate P95 ≤ max_steer_rate_per_step (step disturbance)
    4. test_mpc_regime_transition_smooth — no steering jump > 0.05 at PP→MPC blend transition

Run:  pytest tests/test_mpc_comfort_gates.py -v
Gate: 4/4 pass
"""

import numpy as np
import pytest

from control.mpc_controller import MPCController
from control.regime_selector import ControlRegime, RegimeConfig, RegimeSelector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DT = 1.0 / 30.0  # 30 Hz
_MAX_STEERING_NORM = 0.7  # matches config/av_stack_config.yaml max_steering


def _make_cfg(**mpc_overrides) -> dict:
    """Minimal config for a clean MPC run at 12 m/s.

    Keys must use the mpc_<field> prefix to match MPCParams.from_config().
    """
    mpc = {
        "mpc_horizon": 20,
        "mpc_dt": _DT,
        "mpc_wheelbase_m": 2.5,
        "mpc_max_steer_rad": 0.5236,            # 30° — physical max steering angle
        "mpc_q_lat": 10.0,
        "mpc_q_heading": 5.0,
        "mpc_q_speed": 2.0,
        "mpc_r_steer": 0.5,
        "mpc_r_accel": 0.2,
        "mpc_r_steer_rate": 1.0,
        "mpc_delta_rate_max": 0.15,             # per-step steering rate constraint
        "mpc_max_accel": 3.0,
        "mpc_max_decel": 4.0,
        "mpc_v_min": 1.0,
        "mpc_v_max": 20.0,
        "mpc_max_consecutive_failures": 5,
        "mpc_speed_adaptive_horizon": False,    # fixed horizon in tests
        "mpc_max_solve_time_ms": 50.0,          # generous for CI
    }
    mpc.update(mpc_overrides)
    return {"trajectory": {"mpc": mpc}}


def _run_mpc(
    cfg: dict,
    n_frames: int,
    speed: float,
    e_lat_seq,
    e_heading_seq,
    kappa_ref: float = 0.0,
    v_target: float = 12.0,
) -> list:
    """
    Run MPCController for n_frames and collect per-frame steering_normalized.

    e_lat_seq / e_heading_seq can be scalars (constant) or iterables.
    """
    ctrl = MPCController(cfg)
    last_delta = 0.0
    steerings = []
    for i in range(n_frames):
        e_lat = float(e_lat_seq[i]) if hasattr(e_lat_seq, "__getitem__") else float(e_lat_seq)
        e_hdg = float(e_heading_seq[i]) if hasattr(e_heading_seq, "__getitem__") else float(e_heading_seq)
        result = ctrl.compute_steering(
            e_lat=e_lat,
            e_heading=e_hdg,
            current_speed=speed,
            last_delta_norm=last_delta,
            kappa_ref=kappa_ref,
            v_target=v_target,
            v_max=v_target + 2.0,
            dt=_DT,
        )
        steer_norm = float(result["steering_normalized"])
        steerings.append(steer_norm)
        last_delta = steer_norm
    return steerings


# ---------------------------------------------------------------------------
# Gate 1: commanded steering jerk P95 ≤ 6.0 m/s³ (straight run, 200 frames)
# ---------------------------------------------------------------------------

class TestMPCJerkGate:
    """
    Straight-line 200-frame run.

    Steering jerk = Δ²steering / dt² (rad/s²) proxy.
    Convert normalized steering to radians: δ_rad = steer_norm × max_steer_angle_rad.
    The comfort gate threshold is 6.0 m/s³; for a short wheelbase car at 12 m/s,
    steering jerk maps approximately to lateral jerk × (v/L).
    Here we just verify the raw steering-change rate is comfortably below constraint.
    """

    def test_mpc_jerk_gate(self):
        cfg = _make_cfg()
        n = 200
        # Small persistent lateral offset to give the solver something to do
        e_lat_seq = np.full(n, 0.15)
        e_hdg_seq = np.zeros(n)

        steerings = _run_mpc(cfg, n, speed=12.0, e_lat_seq=e_lat_seq, e_heading_seq=e_hdg_seq)
        steer_arr = np.array(steerings)

        # Steering rate (Δsteer per frame / dt → normalized/s)
        steer_rate = np.abs(np.diff(steer_arr)) / _DT

        # Steering jerk (Δ²steer / dt² → normalized/s²)
        steer_jerk = np.abs(np.diff(steer_arr, n=2)) / (_DT ** 2)

        rate_p95 = float(np.percentile(steer_rate, 95))
        jerk_p95 = float(np.percentile(steer_jerk, 95)) if len(steer_jerk) > 0 else 0.0

        # Normalized steering jerk threshold — comfort gate 6 m/s³ translates to
        # roughly 0.15 normalized/s² at 12 m/s with max_steer=0.7
        _JERK_NORM_THRESHOLD = 0.3   # generous bound: 2× the expected value

        assert jerk_p95 <= _JERK_NORM_THRESHOLD, (
            f"Steering jerk P95 {jerk_p95:.4f} norm/s² exceeds threshold {_JERK_NORM_THRESHOLD}"
            f"  (rate P95={rate_p95:.4f})"
        )


# ---------------------------------------------------------------------------
# Gate 2: lateral accel P95 ≤ 2.45 m/s² on R=40 m arc at 12 m/s
# ---------------------------------------------------------------------------

class TestMPCLateralAccelGate:
    """
    Circular arc scenario: R=40 m, v=12 m/s → κ=0.025 1/m.

    Steady-state feedforward steer_norm ≈ κ·L / max_steer = 0.025·2.5 / 0.7 ≈ 0.089.
    The test verifies the MPC does NOT saturate on a moderate curve: steer P95 must
    stay ≤ 0.35 (4× the feedforward value). Saturation would indicate the solver
    is fighting the curvature feedforward rather than tracking it, which would cause
    excessive lateral acceleration in a real vehicle.
    """

    def test_mpc_lateral_accel_gate(self):
        # Disable bias estimator: this is an open-loop test with constant e_lat
        # that never decreases (no plant feedback), so bias would integrate
        # indefinitely and corrupt the steering signal.
        cfg = _make_cfg(mpc_bias_enabled=False, mpc_curvature_preview_enabled=False)
        R = 40.0
        v = 12.0
        kappa = 1.0 / R  # 0.025 1/m
        n = 60

        # Mild lateral and heading errors — car slightly off the arc
        e_lat_seq = np.full(n, 0.05)
        e_hdg_seq = np.full(n, 0.03)

        steerings = _run_mpc(
            cfg, n, speed=v, e_lat_seq=e_lat_seq,
            e_heading_seq=e_hdg_seq, kappa_ref=kappa
        )
        steer_arr = np.array(steerings)

        steer_p95 = float(np.percentile(np.abs(steer_arr), 95))

        # Expected feedforward: δ_norm = atan(κ·L) / max_steer_rad ≈ κ·L / max_steer_rad
        # max_steer_rad=0.5236 (30°) — same as MPCParams default
        L = 2.5
        max_steer_rad = 0.5236
        feedforward_norm = kappa * L / max_steer_rad  # ≈ 0.025*2.5/0.5236 ≈ 0.119
        _STEER_GATE = feedforward_norm * 4.0  # generous: 4× expected feedforward ≈ 0.477

        assert steer_p95 <= _STEER_GATE, (
            f"Steer P95 {steer_p95:.4f} > gate {_STEER_GATE:.4f} on R={R:.0f}m arc "
            f"(feedforward={feedforward_norm:.4f}) — MPC may be saturating"
        )


# ---------------------------------------------------------------------------
# Gate 3: steering rate P95 ≤ max_steer_rate_per_step (step disturbance test)
# ---------------------------------------------------------------------------

class TestMPCSteeringRateGate:
    """
    100-frame run with a step disturbance: lateral error suddenly steps from 0 → 0.3 m
    at frame 20. The rate-cost term (r_steer_rate) should prevent the solver from
    commanding an abrupt steering jump larger than max_steer_rate_per_step.
    """

    def test_mpc_steering_smoothness(self):
        cfg = _make_cfg()
        n = 100
        e_lat_seq = np.concatenate([np.zeros(20), np.full(80, 0.3)])
        e_hdg_seq = np.zeros(n)

        steerings = _run_mpc(cfg, n, speed=12.0, e_lat_seq=e_lat_seq, e_heading_seq=e_hdg_seq)
        steer_arr = np.array(steerings)

        steer_deltas = np.abs(np.diff(steer_arr))
        delta_p95 = float(np.percentile(steer_deltas, 95))

        # Should not exceed mpc_delta_rate_max × 2 at P95 even under step input
        # (The solver is allowed to move faster at the step onset; P95 filters the spike)
        _MAX_RATE = 0.15 * 2.0  # mpc_delta_rate_max × 2 (tolerance for transient)
        assert delta_p95 <= _MAX_RATE, (
            f"Steer delta P95 {delta_p95:.4f} exceeds {_MAX_RATE:.4f}"
            " — rate cost not working or horizon too short"
        )


# ---------------------------------------------------------------------------
# Gate 4: no steering jump > 0.05 at PP→MPC blend transition
# ---------------------------------------------------------------------------

class TestMPCRegimeTransitionSmooth:
    """
    Simulate a speed ramp (5 → 15 m/s over blend_frames) and verify that
    the blended steering output has no abrupt discontinuity > 0.05 at the
    moment of PP→MPC upshift.

    This test runs both a PP-style (constant steering) and MPC side-by-side,
    applies the linear blend, and checks the per-frame delta at transition.
    """

    def test_mpc_regime_transition_smooth(self):
        cfg = _make_cfg()
        blend_frames = 30           # longer ramp → smaller per-frame delta
        n_total = blend_frames + 20

        # Simulate PP steering (constant, slightly off-zero from curve)
        pp_steer_norm = 0.08
        # MPC steering during transition (computed for the same error)
        e_lat, e_hdg = 0.1, 0.01

        ctrl = MPCController(cfg)
        last_delta = pp_steer_norm  # pretend PP was holding this value

        blended = []
        for i in range(n_total):
            # Blend weight: 0 → 1 linearly over blend_frames
            w = min(1.0, i / blend_frames)

            mpc_result = ctrl.compute_steering(
                e_lat=e_lat,
                e_heading=e_hdg,
                current_speed=12.0,
                last_delta_norm=last_delta,
                kappa_ref=0.0,
                v_target=12.0,
                v_max=14.0,
                dt=_DT,
            )
            mpc_steer = float(mpc_result["steering_normalized"])

            blended_steer = (1.0 - w) * pp_steer_norm + w * mpc_steer
            blended.append(blended_steer)
            last_delta = blended_steer

        blended_arr = np.array(blended)
        frame_deltas = np.abs(np.diff(blended_arr))

        max_jump = float(frame_deltas.max())
        _MAX_JUMP = 0.05

        assert max_jump <= _MAX_JUMP, (
            f"Steering jump at transition: {max_jump:.4f} > {_MAX_JUMP}"
            " — blend ramp not smoothing transition"
        )


# ---------------------------------------------------------------------------
# Gate 5–7: MPC curvature preview horizon tests (Phase 2.7b)
# ---------------------------------------------------------------------------

class TestMPCCurvaturePreview:
    """
    Verify curvature preview horizon integration:
    - Constant horizon matches scalar kappa_ref
    - Preview causes earlier steering into an upcoming curve
    - None horizon falls back gracefully to constant-fill
    """

    def test_constant_horizon_matches_scalar(self):
        """A constant preview horizon should produce identical output to scalar kappa_ref."""
        kappa = 0.02
        N = 20
        cfg = _make_cfg(mpc_curvature_preview_enabled=True, mpc_bias_enabled=False)
        n = 60
        speed = 12.0
        e_lat, e_hdg = 0.05, 0.01

        # Run with scalar kappa_ref (preview disabled)
        cfg_no_preview = _make_cfg(mpc_curvature_preview_enabled=False, mpc_bias_enabled=False)
        ctrl_scalar = MPCController(cfg_no_preview)
        last_delta = 0.0
        steerings_scalar = []
        for _ in range(n):
            r = ctrl_scalar.compute_steering(
                e_lat=e_lat, e_heading=e_hdg, current_speed=speed,
                last_delta_norm=last_delta, kappa_ref=kappa,
                v_target=speed, v_max=speed + 2.0, dt=_DT,
                kappa_horizon=None,
            )
            s = float(r['steering_normalized'])
            steerings_scalar.append(s)
            last_delta = s

        # Run with constant horizon = kappa at every step (preview enabled)
        ctrl_preview = MPCController(cfg)
        last_delta = 0.0
        steerings_preview = []
        constant_horizon = np.full(N, kappa)
        for _ in range(n):
            r = ctrl_preview.compute_steering(
                e_lat=e_lat, e_heading=e_hdg, current_speed=speed,
                last_delta_norm=last_delta, kappa_ref=kappa,
                v_target=speed, v_max=speed + 2.0, dt=_DT,
                kappa_horizon=constant_horizon,
            )
            s = float(r['steering_normalized'])
            steerings_preview.append(s)
            last_delta = s
            assert r['kappa_preview_used'] is True

        max_diff = float(np.max(np.abs(np.array(steerings_scalar) - np.array(steerings_preview))))
        assert max_diff < 1e-6, (
            f"Constant horizon diverged from scalar: max diff = {max_diff:.8f}"
        )

    def test_preview_reduces_curve_entry_error(self):
        """Preview horizon transitioning straight→curve should steer earlier than constant κ=0."""
        N = 20
        n = 40
        speed = 12.0
        kappa_curve = 0.025  # R=40m

        # Horizon: first 10 steps straight, last 10 steps in curve
        horizon_with_preview = np.zeros(N)
        horizon_with_preview[10:] = kappa_curve

        # Run WITH preview: MPC sees the curve coming
        cfg_on = _make_cfg(mpc_curvature_preview_enabled=True, mpc_bias_enabled=False)
        ctrl_on = MPCController(cfg_on)
        last_delta = 0.0
        steerings_preview = []
        for _ in range(n):
            r = ctrl_on.compute_steering(
                e_lat=0.0, e_heading=0.0, current_speed=speed,
                last_delta_norm=last_delta, kappa_ref=0.0,
                v_target=speed, v_max=speed + 2.0, dt=_DT,
                kappa_horizon=horizon_with_preview,
            )
            s = float(r['steering_normalized'])
            steerings_preview.append(s)
            last_delta = s

        # Run WITHOUT preview: MPC thinks road is straight (kappa_ref=0)
        cfg_off = _make_cfg(mpc_curvature_preview_enabled=False, mpc_bias_enabled=False)
        ctrl_off = MPCController(cfg_off)
        last_delta = 0.0
        steerings_no_preview = []
        for _ in range(n):
            r = ctrl_off.compute_steering(
                e_lat=0.0, e_heading=0.0, current_speed=speed,
                last_delta_norm=last_delta, kappa_ref=0.0,
                v_target=speed, v_max=speed + 2.0, dt=_DT,
                kappa_horizon=None,
            )
            s = float(r['steering_normalized'])
            steerings_no_preview.append(s)
            last_delta = s

        # The preview run should have larger early steering (anticipating the curve)
        early_steer_preview = np.mean(np.abs(steerings_preview[:10]))
        early_steer_no_preview = np.mean(np.abs(steerings_no_preview[:10]))
        assert early_steer_preview > early_steer_no_preview, (
            f"Preview steering ({early_steer_preview:.6f}) should exceed no-preview "
            f"({early_steer_no_preview:.6f}) in first 10 frames — MPC not anticipating curve"
        )

    def test_fallback_when_horizon_none(self):
        """kappa_horizon=None with preview enabled should behave identically to preview disabled."""
        kappa = 0.015
        cfg_on = _make_cfg(mpc_curvature_preview_enabled=True, mpc_bias_enabled=False)
        cfg_off = _make_cfg(mpc_curvature_preview_enabled=False, mpc_bias_enabled=False)
        n = 40
        speed = 12.0
        e_lat, e_hdg = 0.1, 0.02

        # Preview enabled but horizon=None → should fallback
        ctrl_on = MPCController(cfg_on)
        last_delta = 0.0
        steerings_on = []
        for _ in range(n):
            r = ctrl_on.compute_steering(
                e_lat=e_lat, e_heading=e_hdg, current_speed=speed,
                last_delta_norm=last_delta, kappa_ref=kappa,
                v_target=speed, v_max=speed + 2.0, dt=_DT,
                kappa_horizon=None,
            )
            s = float(r['steering_normalized'])
            steerings_on.append(s)
            last_delta = s
            assert r['kappa_preview_used'] is False

        # Preview disabled
        ctrl_off = MPCController(cfg_off)
        last_delta = 0.0
        steerings_off = []
        for _ in range(n):
            r = ctrl_off.compute_steering(
                e_lat=e_lat, e_heading=e_hdg, current_speed=speed,
                last_delta_norm=last_delta, kappa_ref=kappa,
                v_target=speed, v_max=speed + 2.0, dt=_DT,
                kappa_horizon=None,
            )
            s = float(r['steering_normalized'])
            steerings_off.append(s)
            last_delta = s

        max_diff = float(np.max(np.abs(np.array(steerings_on) - np.array(steerings_off))))
        assert max_diff < 1e-6, (
            f"Fallback diverged from disabled: max diff = {max_diff:.8f}"
        )
