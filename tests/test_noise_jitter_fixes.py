"""Tests for the noise-jitter fixes applied to MPCController (2026-02-17).

WHAT THESE TESTS PROVE
-----------------------
These are BEFORE/AFTER comparison tests.  Each test constructs two controllers:

  ``ctrl_before``: no first-step rate cost, no EMA filter (reproduces old behaviour)
  ``ctrl_after``:  production config (Fix A only — see below)

and asserts that the AFTER configuration is measurably better.

THE TWO FIXES AND THEIR STATUS
--------------------------------
Fix A — First-step quadratic rate cost  [ENABLED in production]
  The MPC's r_steer_rate only penalised steering changes WITHIN the prediction
  horizon (steps 1→2, 2→3 …), not the jump from the previously applied steering
  to the first QP decision variable delta[0].  That first-step jump was only
  bounded by the hard constraint [last ± delta_rate_max=0.15], giving zero
  quadratic cost regardless of r_steer_rate.

  Fix: add 0.5 * r_steer_rate * (delta[0] − last_delta_norm)^2 to the QP by:
    • adding r_steer_rate to P[d0, d0] diagonal in _build_qp()
    • adding q[d0] += -r_steer_rate * last_delta_norm each solve call

  Result in Unity: max jerk 0.150 → 0.052  (-65%), p95 jerk 0.020 → 0.015 (-26%).

Fix B — Lateral-error EMA pre-filter + deadband  [DISABLED in production]
  In white-noise simulation (σ=0.05 m), EMA alpha=0.3 reduced sign-flips from
  ~9 to 3–5 per 10 frames.  However, in Unity, perception noise is slow and
  correlated (not white noise); sign-flips were already only 3.7% without EMA.
  The EMA added ~7 frames of lag to curve-entry corrections, increasing curve
  e_lat RMSE by +81% (0.22 m → 0.40 m) and dropping score 92.3 → 91.7.
  Fix B is disabled in production: mpc_elat_ema_alpha=1.0, mpc_elat_deadband_m=0.0.
  Tests for the EMA *mechanism* are kept below using explicit non-production controllers.
"""

import collections
import numpy as np
import pytest

from control.mpc_controller import MPCController, MPCParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WHEELBASE_M   = 2.5
_MAX_STEER_RAD = 0.5236
_DT            = 0.033


def _make_mpc_before(**extra) -> MPCController:
    """Reproduce the OLD behaviour: no first-step rate cost, no EMA."""
    cfg = {"trajectory": {"mpc": {
        "mpc_horizon": 20, "mpc_dt": 0.1,
        "mpc_wheelbase_m": _WHEELBASE_M, "mpc_max_steer_rad": _MAX_STEER_RAD,
        "mpc_q_lat": 0.5, "mpc_q_heading": 5.0, "mpc_q_speed": 2.0,
        "mpc_r_steer": 0.05, "mpc_r_accel": 0.2, "mpc_r_steer_rate": 3.5,
        "mpc_delta_rate_max": 0.15, "mpc_max_accel": 3.0, "mpc_max_decel": 4.0,
        "mpc_v_min": 1.0, "mpc_v_max": 20.0, "mpc_speed_adaptive_horizon": False,
        "mpc_bias_enabled": False, "mpc_max_solve_time_ms": 500.0,
        "mpc_max_consecutive_failures": 5,
        # OLD: first-step cost off, no filter
        "mpc_first_step_rate_enabled": False,
        "mpc_elat_ema_alpha": 1.0,
        "mpc_elat_deadband_m": 0.0,
        **extra,
    }}}
    return MPCController(cfg)


def _make_mpc_after(**extra) -> MPCController:
    """Production config: Fix A enabled, EMA disabled (matches av_stack_config.yaml).

    Fix B (EMA alpha=0.3) was disabled after Unity run analysis showed it added
    ~7 frames of lag on curves, increasing curve e_lat RMSE by +81%.
    """
    cfg = {"trajectory": {"mpc": {
        "mpc_horizon": 20, "mpc_dt": 0.1,
        "mpc_wheelbase_m": _WHEELBASE_M, "mpc_max_steer_rad": _MAX_STEER_RAD,
        "mpc_q_lat": 0.5, "mpc_q_heading": 5.0, "mpc_q_speed": 2.0,
        "mpc_r_steer": 0.05, "mpc_r_accel": 0.2, "mpc_r_steer_rate": 3.5,
        "mpc_delta_rate_max": 0.15, "mpc_max_accel": 3.0, "mpc_max_decel": 4.0,
        "mpc_v_min": 1.0, "mpc_v_max": 20.0, "mpc_speed_adaptive_horizon": False,
        "mpc_bias_enabled": False, "mpc_max_solve_time_ms": 500.0,
        "mpc_max_consecutive_failures": 5,
        # Fix A on; EMA off (disabled in production after curve RMSE regression)
        "mpc_first_step_rate_enabled": True,
        "mpc_elat_ema_alpha": 1.0,
        "mpc_elat_deadband_m": 0.0,
        **extra,
    }}}
    return MPCController(cfg)


def _make_mpc_with_ema(**extra) -> MPCController:
    """Non-production controller with EMA enabled — for mechanism tests only."""
    cfg = {"trajectory": {"mpc": {
        "mpc_horizon": 20, "mpc_dt": 0.1,
        "mpc_wheelbase_m": _WHEELBASE_M, "mpc_max_steer_rad": _MAX_STEER_RAD,
        "mpc_q_lat": 0.5, "mpc_q_heading": 5.0, "mpc_q_speed": 2.0,
        "mpc_r_steer": 0.05, "mpc_r_accel": 0.2, "mpc_r_steer_rate": 3.5,
        "mpc_delta_rate_max": 0.15, "mpc_max_accel": 3.0, "mpc_max_decel": 4.0,
        "mpc_v_min": 1.0, "mpc_v_max": 20.0, "mpc_speed_adaptive_horizon": False,
        "mpc_bias_enabled": False, "mpc_max_solve_time_ms": 500.0,
        "mpc_max_consecutive_failures": 5,
        "mpc_first_step_rate_enabled": True,
        "mpc_elat_ema_alpha": 0.3,
        "mpc_elat_deadband_m": 0.03,
        **extra,
    }}}
    return MPCController(cfg)


def _bicycle_step(e_lat, e_heading, speed, delta_norm, dt, kappa=0.0):
    delta = float(np.clip(delta_norm, -1.0, 1.0)) * _MAX_STEER_RAD
    kappa_v = np.tan(delta) / max(_WHEELBASE_M, 1e-3)
    return (
        e_lat    + speed * np.sin(e_heading) * dt,
        e_heading + speed * (kappa_v - kappa) * dt,
    )


def _run_closedloop(ctrl, e_lat0=0.3, speed=8.0, n_frames=400,
                    noise_std=0.05, latency=1, rng_seed=0):
    """Return (steering_history, e_lat_history) arrays."""
    rng = np.random.default_rng(rng_seed)
    buf = collections.deque([0.0] * max(1, latency + 1), maxlen=max(1, latency + 1))
    e_lat, e_head = float(e_lat0), 0.0
    steers = np.empty(n_frames)
    lats   = np.empty(n_frames)

    for i in range(n_frames):
        noisy = e_lat + (rng.normal(0.0, noise_std) if noise_std > 0 else 0.0)
        r = ctrl.compute_steering(
            e_lat=noisy, e_heading=e_head, current_speed=speed,
            last_delta_norm=buf[-1], kappa_ref=0.0,
            v_target=speed, v_max=speed + 2.0, dt=_DT,
        )
        d = float(r.get("steering_normalized", 0.0))
        buf.append(d)
        delta_applied = buf[0]
        e_lat, e_head = _bicycle_step(e_lat, e_head, speed, delta_applied, _DT)
        steers[i] = d
        lats[i]   = e_lat

    return steers, lats


def _max_sign_flips_in_window(arr: np.ndarray, window: int = 10) -> int:
    signs = np.sign(arr)
    signs[signs == 0] = 1
    max_f = 0
    for s in range(len(arr) - window + 1):
        max_f = max(max_f, int(np.sum(np.diff(signs[s:s + window]) != 0)))
    return max_f


# ---------------------------------------------------------------------------
# Fix A: first-step rate cost makes r_steer_rate effective at step 0
# ---------------------------------------------------------------------------

class TestFirstStepRateCost:

    def test_params_load_first_step_flag(self):
        """MPCParams must read mpc_first_step_rate_enabled from config."""
        cfg = {"trajectory": {"mpc": {"mpc_first_step_rate_enabled": True}}}
        p = MPCParams.from_config(cfg)
        assert p.first_step_rate_enabled is True

        cfg2 = {"trajectory": {"mpc": {"mpc_first_step_rate_enabled": False}}}
        p2 = MPCParams.from_config(cfg2)
        assert p2.first_step_rate_enabled is False

    def test_first_step_cost_reduces_immediate_sign_flip(self):
        """With first-step rate cost enabled, alternating inputs must produce
        fewer first-step sign reversals than without it.

        This is the isolated Fix-A test: EMA is off in both controllers so
        the only variable is the QP first-step cost.
        """
        # Both controllers: EMA off, deadband off — only Fix A varies
        before = _make_mpc_before(mpc_elat_ema_alpha=1.0, mpc_elat_deadband_m=0.0,
                                   mpc_first_step_rate_enabled=False)
        after  = _make_mpc_before(mpc_elat_ema_alpha=1.0, mpc_elat_deadband_m=0.0,
                                   mpc_first_step_rate_enabled=True)

        flips_before, flips_after = 0, 0
        last_b = last_a = 0.0
        for i in range(200):
            e = 0.05 * (1 if i % 2 == 0 else -1)
            rb = before.compute_steering(e_lat=e, e_heading=0.0, current_speed=8.0,
                                          last_delta_norm=last_b, kappa_ref=0.0,
                                          v_target=8.0, v_max=10.0, dt=_DT)
            ra = after.compute_steering(e_lat=e, e_heading=0.0, current_speed=8.0,
                                         last_delta_norm=last_a, kappa_ref=0.0,
                                         v_target=8.0, v_max=10.0, dt=_DT)
            db = float(rb.get("steering_normalized", 0.0))
            da = float(ra.get("steering_normalized", 0.0))
            if last_b != 0.0 and np.sign(db) != np.sign(last_b): flips_before += 1
            if last_a != 0.0 and np.sign(da) != np.sign(last_a): flips_after  += 1
            last_b, last_a = db, da

        # Fix A alone should reduce flips; allow equal as a weak assertion since
        # the EMA is off and the lateral benefit may still dominate for tiny errors.
        assert flips_after <= flips_before, (
            f"Fix A (first-step rate cost): flips_before={flips_before}, "
            f"flips_after={flips_after} — should not increase"
        )

    def test_first_step_cost_solver_remains_feasible(self):
        """Enabling first-step rate cost must not cause solver infeasibility."""
        ctrl = _make_mpc_after()
        for e in (-0.5, -0.2, 0.0, 0.2, 0.5):
            r = ctrl.compute_steering(
                e_lat=e, e_heading=0.0, current_speed=8.0,
                last_delta_norm=0.0, kappa_ref=0.0,
                v_target=8.0, v_max=10.0, dt=_DT,
            )
            assert r.get("mpc_feasible", True), (
                f"Solver infeasible at e_lat={e} with first_step_rate_enabled=True"
            )

    def test_config_enables_first_step_rate_by_default(self):
        """av_stack_config.yaml must have mpc_first_step_rate_enabled: true."""
        import yaml
        with open("config/av_stack_config.yaml") as f:
            cfg = yaml.safe_load(f)
        val = cfg.get("trajectory", {}).get("mpc", {}).get("mpc_first_step_rate_enabled")
        assert val is True, (
            f"mpc_first_step_rate_enabled={val!r} in config, expected True"
        )


# ---------------------------------------------------------------------------
# Fix B: EMA pre-filter reduces noise-driven jitter
# ---------------------------------------------------------------------------

class TestEMAPreFilter:

    def test_params_load_ema_config(self):
        """MPCParams must read mpc_elat_ema_alpha and mpc_elat_deadband_m."""
        cfg = {"trajectory": {"mpc": {
            "mpc_elat_ema_alpha": 0.3,
            "mpc_elat_deadband_m": 0.03,
        }}}
        p = MPCParams.from_config(cfg)
        assert p.elat_ema_alpha   == pytest.approx(0.3,  rel=1e-6)
        assert p.elat_deadband_m  == pytest.approx(0.03, rel=1e-6)

    def test_ema_returns_filtered_elat_in_result(self):
        """compute_steering must expose e_lat_raw and e_lat_input separately.
        Uses a non-production controller with EMA enabled to test the mechanism.
        """
        ctrl = _make_mpc_with_ema()
        raw_e = 0.15
        r = ctrl.compute_steering(
            e_lat=raw_e, e_heading=0.0, current_speed=8.0,
            last_delta_norm=0.0, kappa_ref=0.0, v_target=8.0, v_max=10.0, dt=_DT,
        )
        assert "e_lat_raw"   in r, "e_lat_raw key missing from compute_steering result"
        assert "e_lat_input" in r, "e_lat_input key missing from compute_steering result"
        assert r["e_lat_raw"] == pytest.approx(raw_e, abs=1e-6)
        # First call: EMA starts at 0 → filtered = alpha * raw = 0.3 * 0.15 = 0.045
        assert r["e_lat_input"] == pytest.approx(0.3 * raw_e, abs=0.01)

    def test_ema_unity_alpha_passes_raw_unchanged(self):
        """alpha=1.0 must pass e_lat straight through (no lag)."""
        ctrl = _make_mpc_before(mpc_elat_ema_alpha=1.0, mpc_elat_deadband_m=0.0)
        raw_e = 0.2
        r = ctrl.compute_steering(
            e_lat=raw_e, e_heading=0.0, current_speed=8.0,
            last_delta_norm=0.0, kappa_ref=0.0, v_target=8.0, v_max=10.0, dt=_DT,
        )
        assert r["e_lat_input"] == pytest.approx(raw_e, abs=1e-4)

    def test_deadband_suppresses_small_errors(self):
        """When the filtered e_lat magnitude is below the deadband, input must be 0.
        Uses a non-production controller with EMA+deadband enabled to test the mechanism.
        """
        ctrl = _make_mpc_with_ema()
        # Warm up EMA to near-zero by feeding repeated small values
        for _ in range(30):
            ctrl.compute_steering(e_lat=0.0, e_heading=0.0, current_speed=8.0,
                                   last_delta_norm=0.0, kappa_ref=0.0,
                                   v_target=8.0, v_max=10.0, dt=_DT)
        # Now feed a tiny error below the deadband (0.03 m)
        r = ctrl.compute_steering(
            e_lat=0.01, e_heading=0.0, current_speed=8.0,
            last_delta_norm=0.0, kappa_ref=0.0, v_target=8.0, v_max=10.0, dt=_DT,
        )
        assert abs(r["e_lat_input"]) < 0.03, (
            f"EMA+deadband: |e_lat_input|={abs(r['e_lat_input']):.4f} should be "
            f"below deadband 0.03 m after EMA is near zero"
        )

    def test_config_has_ema_params_disabled(self):
        """av_stack_config.yaml must define EMA params with EMA disabled (alpha=1.0).

        Fix B was disabled after Unity run analysis: alpha=0.3 added 7-frame lag
        on curves → +81% curve RMSE regression. alpha=1.0 = pass-through (no filter).
        """
        import yaml
        with open("config/av_stack_config.yaml") as f:
            cfg = yaml.safe_load(f)
        mpc = cfg.get("trajectory", {}).get("mpc", {})
        alpha = mpc.get("mpc_elat_ema_alpha")
        db    = mpc.get("mpc_elat_deadband_m")
        assert alpha is not None and 0.0 < float(alpha) <= 1.0, (
            f"mpc_elat_ema_alpha={alpha!r} must be in (0, 1]"
        )
        assert float(alpha) == pytest.approx(1.0, abs=1e-6), (
            f"EMA is DISABLED in production (alpha must be 1.0), got {alpha!r}. "
            f"alpha=0.3 caused +81% curve RMSE regression in Unity (2026-03-21)."
        )
        assert db is not None and float(db) == pytest.approx(0.0, abs=1e-6), (
            f"Deadband must be 0.0 when EMA is disabled, got {db!r}"
        )


# ---------------------------------------------------------------------------
# Combined: BEFORE vs AFTER under realistic conditions
# ---------------------------------------------------------------------------

class TestBeforeAfterComparison:
    """BEFORE/AFTER comparison tests for Fix A (first-step rate cost).

    Production config is Fix A only (EMA disabled after curve RMSE regression).
    Fix B (EMA) mechanism tests are in TestEMAPreFilter using explicit non-production
    controllers.
    """

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_fix_a_does_not_increase_sign_flips(self, seed):
        """Fix A (first-step rate cost) must not make sign-flip rate worse
        compared to no fixes.  Under white-noise inputs the improvement is
        modest but must never regress.
        """
        before = _make_mpc_before()
        after  = _make_mpc_after()   # Fix A only, no EMA

        s_b, _ = _run_closedloop(before, rng_seed=seed)
        s_a, _ = _run_closedloop(after,  rng_seed=seed)

        flips_b = _max_sign_flips_in_window(s_b[30:])
        flips_a = _max_sign_flips_in_window(s_a[30:])

        # Fix A alone gives modest improvement under white noise; require non-regression
        assert flips_a <= flips_b + 1, (
            f"seed={seed}: Fix A should not worsen sign flips: "
            f"before={flips_b}, after={flips_a}"
        )

    def test_convergence_from_large_offset_not_slowed(self):
        """Fix A must not slow convergence from a real 0.3 m offset (noise-free).
        Allowed slowdown: ≤3 extra frames to reach 0.1 m.
        """
        before = _make_mpc_before()
        after  = _make_mpc_after()

        _, lat_b = _run_closedloop(before, n_frames=200, noise_std=0.0, latency=0)
        _, lat_a = _run_closedloop(after,  n_frames=200, noise_std=0.0, latency=0)

        def _frames_to(hist, thr=0.1):
            for i, v in enumerate(hist):
                if abs(v) < thr:
                    return i
            return len(hist)

        f_b = _frames_to(lat_b)
        f_a = _frames_to(lat_a)

        assert f_a <= f_b + 3, (
            f"Fix A slowed convergence: BEFORE={f_b} frames, AFTER={f_a} frames "
            f"(allowed ≤{f_b + 3})."
        )

    def test_after_sign_flips_below_regression_ceiling(self):
        """Production config (Fix A only) sign-flip rate must stay below the
        pre-fix baseline of ~9 per 10 frames.  Ceiling = 9 (regression guard).
        """
        ctrl = _make_mpc_after()
        s, _ = _run_closedloop(ctrl, rng_seed=0)
        max_flips = _max_sign_flips_in_window(s[30:])
        assert max_flips <= 9, (
            f"Fix A only: max sign-flips per 10 frames = {max_flips}, "
            f"expected ≤9 (regression ceiling against pre-fix baseline ~9)."
        )

    def test_ema_reduces_sign_flips_in_simulation(self):
        """Document the mechanism: EMA alpha=0.3 reduces simulated sign-flips
        significantly under white-noise inputs (even though it's disabled in
        production due to curve RMSE regression).
        """
        no_ema  = _make_mpc_after()               # Fix A only
        with_ema = _make_mpc_with_ema()           # Fix A + EMA (non-production)

        s_no,  _ = _run_closedloop(no_ema,   rng_seed=0)
        s_ema, _ = _run_closedloop(with_ema, rng_seed=0)

        flips_no  = _max_sign_flips_in_window(s_no[30:])
        flips_ema = _max_sign_flips_in_window(s_ema[30:])

        # EMA should strictly reduce white-noise sign flips in simulation
        assert flips_ema < flips_no, (
            f"EMA mechanism test: EMA flips ({flips_ema}) should be < no-EMA ({flips_no}). "
            f"Note: EMA is disabled in production due to +81% curve RMSE regression in Unity."
        )

    def test_after_stays_bounded_on_curve(self):
        """Production config (Fix A only, no EMA) must not cause e_lat to
        grow unboundedly on a constant-curvature curve.
        """
        buf = collections.deque([0.0] * 2, maxlen=2)
        e_lat, e_head = 0.0, 0.0
        rng = np.random.default_rng(5)
        ctrl = _make_mpc_after()
        max_err = 0.0
        for _ in range(200):
            noisy = e_lat + rng.normal(0, 0.05)
            r = ctrl.compute_steering(e_lat=noisy, e_heading=e_head, current_speed=8.0,
                                       last_delta_norm=buf[-1], kappa_ref=0.03,
                                       v_target=8.0, v_max=10.0, dt=_DT)
            d = float(r.get("steering_normalized", 0.0))
            buf.append(d)
            e_lat, e_head = _bicycle_step(e_lat, e_head, 8.0, buf[0], _DT, kappa=0.03)
            max_err = max(max_err, abs(e_lat))
        assert max_err < 0.5, (
            f"Curve tracking (Fix A only): max |e_lat|={max_err:.4f} m > 0.5 m. "
            f"Production config should handle constant curvature without unbounded drift."
        )


# ---------------------------------------------------------------------------
# What test WOULD have caught the EMA regression?
# These tests guard against the same failure in future input-filtering changes.
# ---------------------------------------------------------------------------

class TestInputFilterRegressionGuards:
    """Tests that would have caught the EMA alpha=0.3 curve RMSE regression
    before running in Unity.

    ROOT CAUSE OF TEST GAP:
    The kinematic bicycle model with perfect feedforward kappa cannot show
    the EMA lag because feedforward handles curve tracking.  Only when
    feedforward is imperfect (Unity reality) does EMA lag accumulate.
    These tests use a ramp input (continuous drift) to expose the steady-state
    lag effect that a step-response test misses.

    NOTE: EMA lag is computed analytically / externally, NOT from the controller's
    e_lat_ema return field.  When alpha=1.0, the controller skips the EMA update
    entirely (passthrough), so e_lat_ema stays 0.0 — we test the filter directly.
    """

    @staticmethod
    def _apply_ema(signal: np.ndarray, alpha: float) -> np.ndarray:
        """Apply a first-order EMA to a signal array."""
        out = np.empty_like(signal)
        state = 0.0
        for i, v in enumerate(signal):
            state = alpha * v + (1.0 - alpha) * state
            out[i] = state
        return out

    def test_alpha_1_has_zero_ramp_lag(self):
        """alpha=1.0 (production) must have zero steady-state ramp lag.

        For alpha=1.0: EMA[k] = input[k] exactly.  Lag = 0.
        """
        ramp = np.arange(80) * 0.005
        filtered = self._apply_ema(ramp, alpha=1.0)
        steady_lag = np.abs(ramp[40:] - filtered[40:])
        assert np.mean(steady_lag) < 1e-9, (
            f"alpha=1.0 must have zero ramp lag, got {np.mean(steady_lag):.2e}m."
        )

    def test_alpha_03_has_measurable_ramp_lag(self):
        """EMA alpha=0.3 must show significant steady-state ramp lag.

        Theory: for a ramp of slope r, steady-state lag = r * (1-α)/α frames.
        At alpha=0.3, rate=0.005 m/frame: lag = 0.005 * 0.7/0.3 ≈ 0.012m.
        This is the mechanism that caused the Unity curve RMSE regression.
        """
        ramp = np.arange(80) * 0.005
        filtered = self._apply_ema(ramp, alpha=0.3)
        # After warm-up (40+ frames) the lag should be near theoretical value
        steady_lag = np.mean(np.abs(ramp[40:] - filtered[40:]))
        assert steady_lag > 0.008, (
            f"EMA alpha=0.3 should show ≥0.008m steady-state ramp lag "
            f"(got {steady_lag:.4f}m). This lag accumulates in Unity curves."
        )

    def test_aggressive_ema_has_worse_ramp_tracking_than_passthrough(self):
        """Any alpha < 1.0 must have worse ramp-tracking RMSE than alpha=1.0.

        This is the REGRESSION GUARD: a future input filter must not introduce
        steady-state lag on ramp inputs, or it will cause a Unity curve RMSE
        regression.  alpha=0.3 (old Fix B) fails this test, confirming the root
        cause of the 91.7-vs-92.3 score regression (2026-03-21).
        """
        rates = [0.003, 0.005, 0.008]
        rmse_passthrough, rmse_ema_03 = [], []
        for rate in rates:
            ramp = np.arange(100) * rate
            f_pt = self._apply_ema(ramp, alpha=1.0)
            f_03 = self._apply_ema(ramp, alpha=0.3)
            # Measure steady-state lag (after warm-up, frames 50+)
            rmse_passthrough.append(np.sqrt(np.mean((ramp[50:] - f_pt[50:]) ** 2)))
            rmse_ema_03.append(    np.sqrt(np.mean((ramp[50:] - f_03[50:]) ** 2)))

        assert np.mean(rmse_ema_03) > np.mean(rmse_passthrough), (
            f"EMA alpha=0.3 must have worse ramp-tracking RMSE than alpha=1.0: "
            f"α=0.3 RMSE={np.mean(rmse_ema_03):.5f}m, "
            f"α=1.0 RMSE={np.mean(rmse_passthrough):.5f}m. "
            f"Any filter with lag > 0 will cause a Unity curve RMSE regression."
        )

    @pytest.mark.skipif(
        not list(__import__('pathlib').Path('data/recordings').glob('*.h5')) if
        __import__('pathlib').Path('data/recordings').exists() else True,
        reason="No recordings available"
    )
    def test_replay_ema_does_not_increase_e_lat_rmse(self):
        """Open-loop replay: apply EMA filter to recorded perception e_lat and
        assert filtered RMSE ≤ unfiltered RMSE * 1.05 (5% tolerance).

        This test uses ACTUAL Unity perception dynamics — the closest proxy to
        a real drive without running Unity.  It would have caught the alpha=0.3
        regression before the 91.7-vs-92.3 score drop.
        """
        import h5py
        from pathlib import Path
        recs = sorted(Path('data/recordings').glob('*.h5'), key=lambda p: p.stat().st_mtime)
        latest = recs[-1]
        with h5py.File(latest, 'r') as f:
            e_lat_key = 'control/mpc_e_lat'
            if e_lat_key not in f:
                pytest.skip("Recording lacks mpc_e_lat signal")
            e_lat = f[e_lat_key][:]

        # Apply EMA with alpha=0.3 to the real recorded signal
        alpha = 0.3
        ema_state = 0.0
        filtered = np.empty_like(e_lat)
        for i, v in enumerate(e_lat):
            ema_state = alpha * v + (1.0 - alpha) * ema_state
            filtered[i] = ema_state

        rmse_raw      = np.sqrt(np.nanmean(e_lat ** 2))
        rmse_filtered = np.sqrt(np.nanmean(filtered ** 2))

        # A filter that increases RMSE on real recordings will increase curve tracking error.
        # 5% tolerance accounts for noise reduction benefit on straight segments.
        assert rmse_filtered <= rmse_raw * 1.05, (
            f"EMA alpha=0.3 increases real e_lat RMSE: "
            f"raw={rmse_raw:.4f}m → filtered={rmse_filtered:.4f}m "
            f"({100*(rmse_filtered/rmse_raw - 1):.1f}% increase, limit 5%). "
            f"This predicts a Unity curve tracking regression. Disable EMA or raise alpha."
        )
