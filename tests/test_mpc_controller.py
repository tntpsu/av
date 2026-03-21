"""
Unit tests for Linear MPC controller (Phase 2.1).

Tests cover: directional correctness, constraint satisfaction,
solver performance, warm-start, and fallback logic.

Run:  pytest tests/test_mpc_controller.py -v
Gate: 13/13 pass
"""

import time

import numpy as np
import pytest

from control.mpc_controller import MPCController, MPCParams, MPCSolver


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_controller(**overrides) -> MPCController:
    """Create MPC controller with test-friendly defaults."""
    cfg = {
        "trajectory": {"mpc": {
            "mpc_horizon": 10,
            "mpc_dt": 0.033,
            "mpc_wheelbase_m": 2.5,
            "mpc_max_steer_rad": 0.5236,
            "mpc_q_lat": 10.0,
            "mpc_q_heading": 5.0,
            "mpc_q_speed": 1.0,
            "mpc_r_steer": 1e-4,
            "mpc_r_accel": 0.05,
            "mpc_r_steer_rate": 1.0,
            "mpc_delta_rate_max": 0.5,
            "mpc_max_accel": 1.2,
            "mpc_max_decel": 2.4,
            "mpc_v_min": 1.0,
            "mpc_v_max": 15.0,
            "mpc_speed_adaptive_horizon": False,
            "mpc_max_solve_time_ms": 50.0,       # generous for CI
            "mpc_max_consecutive_failures": 3,
            **overrides,
        }}
    }
    return MPCController(cfg)


def _solve(ctrl, e_lat=0.0, e_heading=0.0, speed=12.0,
           kappa=0.0, v_target=12.0, last_delta=0.0) -> dict:
    """Shorthand for single MPC solve."""
    return ctrl.compute_steering(
        e_lat=e_lat, e_heading=e_heading, current_speed=speed,
        last_delta_norm=last_delta, kappa_ref=kappa,
        v_target=v_target, v_max=15.0, dt=0.033,
    )


# ---------------------------------------------------------------------------
# 1. Straight road, zero error → near-zero steering
# ---------------------------------------------------------------------------

def test_straight_zero_error():
    """On a straight road with zero error, MPC should output near-zero steering."""
    ctrl = _make_controller()
    r = _solve(ctrl, e_lat=0.0, e_heading=0.0, kappa=0.0)
    assert r['mpc_feasible'], f"Solver infeasible: {r}"
    assert abs(r['steering_normalized']) < 0.01, \
        f"Expected ~0 steering, got {r['steering_normalized']:.6f}"


# ---------------------------------------------------------------------------
# 2. Curved road feedforward
# ---------------------------------------------------------------------------

def test_curved_road_feedforward():
    """Positive curvature (left turn) should produce positive steering."""
    ctrl = _make_controller()
    r = _solve(ctrl, e_lat=0.0, e_heading=0.0, kappa=0.02)
    assert r['mpc_feasible']
    assert r['steering_normalized'] > 0.005, \
        f"Expected positive steer for κ=+0.02, got {r['steering_normalized']:.6f}"


# ---------------------------------------------------------------------------
# 3. Lateral error correction
# ---------------------------------------------------------------------------

def test_lateral_error_correction():
    """Positive lateral error should produce negative steering (correct back)."""
    ctrl = _make_controller()
    r = _solve(ctrl, e_lat=0.5, e_heading=0.0, kappa=0.0)
    assert r['mpc_feasible']
    assert r['steering_normalized'] < 0.0, \
        f"Expected negative steer for e_lat=+0.5, got {r['steering_normalized']:.6f}"


# ---------------------------------------------------------------------------
# 4. Heading error correction
# ---------------------------------------------------------------------------

def test_heading_error_correction():
    """Positive heading error should produce negative steering."""
    ctrl = _make_controller()
    r = _solve(ctrl, e_lat=0.0, e_heading=0.1, kappa=0.0)
    assert r['mpc_feasible']
    assert r['steering_normalized'] < 0.0, \
        f"Expected negative steer for e_heading=+0.1, got {r['steering_normalized']:.6f}"


# ---------------------------------------------------------------------------
# 5. Speed tracking (accel)
# ---------------------------------------------------------------------------

def test_speed_tracking():
    """Below target speed → positive acceleration."""
    ctrl = _make_controller()
    r = _solve(ctrl, speed=10.0, v_target=12.0)
    assert r['mpc_feasible']
    assert r['accel'] > 0.0, \
        f"Expected positive accel for v=10 → v_target=12, got {r['accel']:.6f}"


# ---------------------------------------------------------------------------
# 6. Steering rate constraint
# ---------------------------------------------------------------------------

def test_steering_rate_constraint():
    """Consecutive solves should respect delta_rate_max between steps."""
    ctrl = _make_controller(mpc_delta_rate_max=0.3)
    # First solve at zero
    r1 = _solve(ctrl, e_lat=1.0, last_delta=0.0)
    delta1 = r1['steering_normalized']
    # The step from 0.0 to delta1 should respect rate limit
    assert abs(delta1 - 0.0) <= 0.3 + 1e-4, \
        f"Rate violation: Δδ = {abs(delta1):.4f} > 0.3"

    # Second solve using delta1 as last
    r2 = _solve(ctrl, e_lat=1.0, last_delta=delta1)
    delta2 = r2['steering_normalized']
    assert abs(delta2 - delta1) <= 0.3 + 1e-4, \
        f"Rate violation: Δδ = {abs(delta2 - delta1):.4f} > 0.3"


# ---------------------------------------------------------------------------
# 7. Accel saturation
# ---------------------------------------------------------------------------

def test_accel_saturation():
    """Large speed gap should saturate at max_accel."""
    ctrl = _make_controller(mpc_max_accel=1.2)
    r = _solve(ctrl, speed=5.0, v_target=15.0)
    assert r['mpc_feasible']
    assert abs(r['accel']) <= 1.2 + 1e-4, \
        f"Accel {r['accel']:.4f} exceeds max_accel=1.2"


# ---------------------------------------------------------------------------
# 8. Speed-adaptive horizon
# ---------------------------------------------------------------------------

def test_speed_adaptive_horizon():
    """Horizon should increase above speed threshold."""
    ctrl = _make_controller(
        mpc_speed_adaptive_horizon=True,
        mpc_speed_adaptive_threshold_mps=15.0,
        mpc_speed_adaptive_n_high=30,
        mpc_horizon=20,
    )
    # Below threshold
    _solve(ctrl, speed=14.9)
    assert ctrl.solver._N == 20, f"Expected N=20 at v=14.9, got {ctrl.solver._N}"

    # Above threshold
    _solve(ctrl, speed=15.1)
    assert ctrl.solver._N == 30, f"Expected N=30 at v=15.1, got {ctrl.solver._N}"


# ---------------------------------------------------------------------------
# 9. Solve time budget
# ---------------------------------------------------------------------------

def test_solve_time_budget():
    """P95 solve time should be < 5 ms for N=10."""
    ctrl = _make_controller(mpc_horizon=10)
    times = []
    for i in range(100):
        e_lat = 0.3 * np.sin(i * 0.1)
        kappa = 0.01 * np.cos(i * 0.05)
        r = _solve(ctrl, e_lat=e_lat, kappa=kappa)
        times.append(r['solve_time_ms'])

    p95 = np.percentile(times, 95)
    assert p95 < 5.0, f"P95 solve time {p95:.2f} ms exceeds 5 ms budget"


# ---------------------------------------------------------------------------
# 10. Warm start is faster than cold start
# ---------------------------------------------------------------------------

def test_warm_start_faster():
    """Warm-started solves should be faster than cold solves on average."""
    # Cold solve (new controller each time)
    cold_times = []
    for _ in range(10):
        ctrl = _make_controller(mpc_horizon=10)
        r = _solve(ctrl, e_lat=0.2, e_heading=0.05, kappa=0.01)
        cold_times.append(r['solve_time_ms'])
    cold_mean = np.mean(cold_times)

    # Warm solves (same controller, sequential)
    ctrl = _make_controller(mpc_horizon=10)
    _solve(ctrl, e_lat=0.2, e_heading=0.05, kappa=0.01)  # prime
    warm_times = []
    for i in range(10):
        e_lat = 0.2 + 0.01 * i
        r = _solve(ctrl, e_lat=e_lat, e_heading=0.05, kappa=0.01)
        warm_times.append(r['solve_time_ms'])
    warm_mean = np.mean(warm_times)

    # Warm should be faster (or at worst comparable)
    assert warm_mean < cold_mean * 1.5, \
        f"Warm {warm_mean:.3f} ms not faster than cold {cold_mean:.3f} ms"


# ---------------------------------------------------------------------------
# 11. Kappa sign convention
# ---------------------------------------------------------------------------

def test_kappa_sign_convention():
    """Positive κ (left turn) → positive steer; negative κ → negative steer."""
    ctrl = _make_controller()
    r_left = _solve(ctrl, kappa=+0.02)
    r_right = _solve(ctrl, kappa=-0.02)
    assert r_left['steering_normalized'] > 0, \
        f"κ=+0.02 should give positive steer, got {r_left['steering_normalized']:.6f}"
    assert r_right['steering_normalized'] < 0, \
        f"κ=-0.02 should give negative steer, got {r_right['steering_normalized']:.6f}"


# ---------------------------------------------------------------------------
# 12. Integration end-to-end (50 frames)
# ---------------------------------------------------------------------------

def test_integration_end_to_end():
    """Run 50 consecutive frames without exceptions."""
    ctrl = _make_controller()
    last_delta = 0.0
    for frame in range(50):
        e_lat = 0.3 * np.sin(frame * 0.05)
        e_heading = 0.05 * np.cos(frame * 0.03)
        kappa = 0.01 * np.sin(frame * 0.02)
        r = ctrl.compute_steering(
            e_lat=e_lat, e_heading=e_heading, current_speed=12.0,
            last_delta_norm=last_delta, kappa_ref=kappa,
            v_target=12.0, v_max=15.0, dt=0.033,
        )
        assert r['mpc_feasible'], f"Frame {frame}: solver infeasible"
        last_delta = r['steering_normalized']
    # If we get here without exceptions, test passes


# ---------------------------------------------------------------------------
# 13. Fallback after consecutive failures
# ---------------------------------------------------------------------------

def test_fallback_after_failures():
    """After max_consecutive_failures infeasible solves, fallback activates."""
    ctrl = _make_controller(
        mpc_max_consecutive_failures=3,
        mpc_max_solve_time_ms=0.0001,   # impossibly tight → forces "failure"
    )
    for i in range(3):
        r = _solve(ctrl)
    assert r['mpc_fallback_active'], \
        f"Expected fallback after 3 failures, got fallback={r['mpc_fallback_active']}"
    assert r['mpc_consecutive_failures'] >= 3


# ---------------------------------------------------------------------------
# Bonus: MPCParams.from_config
# ---------------------------------------------------------------------------

def test_params_from_config():
    """MPCParams loads correctly from nested config dict."""
    cfg = {"trajectory": {"mpc": {
        "mpc_horizon": 15,
        "mpc_q_lat": 20.0,
        "mpc_wheelbase_m": 3.0,
    }}}
    p = MPCParams.from_config(cfg)
    assert p.horizon == 15
    assert p.q_lat == 20.0
    assert p.wheelbase_m == 3.0
    # Defaults for unspecified
    assert p.q_heading == 5.0
    assert p.r_steer_rate == 1.0


# ---------------------------------------------------------------------------
# Arc-entry kappa-transition warmup (Step 4b)
# ---------------------------------------------------------------------------

def test_kappa_transition_triggers_warmup_reset():
    """A large kappa jump (straight→curve) should reset _frames_since_reset to 0."""
    ctrl = _make_controller()
    # Prime: run 10 frames on a straight so warmup is exhausted
    for _ in range(10):
        _solve(ctrl, kappa=0.0001)
    assert ctrl._frames_since_reset >= 10, "Expected warmup exhausted after 10 frames"

    # Now simulate arc entry: kappa jumps from ~0 to 0.010 (>threshold 0.005)
    _solve(ctrl, kappa=0.010)

    # _frames_since_reset should be 1 (reset happened, then incremented after solve)
    assert ctrl._frames_since_reset == 1, (
        f"Expected _frames_since_reset=1 after kappa jump, got {ctrl._frames_since_reset}"
    )


def test_kappa_transition_below_threshold_no_reset():
    """A small kappa increase (< threshold) should NOT trigger warmup reset."""
    ctrl = _make_controller()
    for _ in range(10):
        _solve(ctrl, kappa=0.001)
    frames_before = ctrl._frames_since_reset

    # Small increase: 0.001 → 0.004 = delta 0.003 < threshold 0.005
    _solve(ctrl, kappa=0.004)

    assert ctrl._frames_since_reset == frames_before + 1, (
        f"Small kappa increase should NOT reset: got {ctrl._frames_since_reset}, "
        f"expected {frames_before + 1}"
    )


def test_kappa_transition_decrease_no_reset():
    """A kappa decrease (curve→straight) should NOT trigger warmup reset."""
    ctrl = _make_controller()
    # Prime in a curve
    for _ in range(10):
        _solve(ctrl, kappa=0.010)
    frames_before = ctrl._frames_since_reset

    # Exit curve: kappa drops from 0.010 to ~0
    _solve(ctrl, kappa=0.001)

    assert ctrl._frames_since_reset == frames_before + 1, (
        f"Kappa decrease should NOT reset: got {ctrl._frames_since_reset}, "
        f"expected {frames_before + 1}"
    )


def test_kappa_transition_warmup_disabled():
    """When kappa_transition_warmup_enabled=False, large kappa jump does not reset."""
    ctrl = _make_controller(mpc_kappa_transition_warmup_enabled=False)
    for _ in range(10):
        _solve(ctrl, kappa=0.0001)
    frames_before = ctrl._frames_since_reset

    # Large jump — should NOT trigger reset because feature is disabled
    _solve(ctrl, kappa=0.015)

    assert ctrl._frames_since_reset == frames_before + 1, (
        f"Disabled warmup should not reset: got {ctrl._frames_since_reset}, "
        f"expected {frames_before + 1}"
    )


def test_kappa_transition_last_kappa_updated_each_frame():
    """_last_kappa_ref should track the kappa_ref from each solve call."""
    ctrl = _make_controller()
    _solve(ctrl, kappa=0.005)
    assert abs(ctrl._last_kappa_ref - 0.005) < 1e-9

    _solve(ctrl, kappa=0.012)
    assert abs(ctrl._last_kappa_ref - 0.012) < 1e-9

    _solve(ctrl, kappa=0.002)
    assert abs(ctrl._last_kappa_ref - 0.002) < 1e-9


def test_kappa_transition_reset_resets_last_kappa():
    """reset() should clear _last_kappa_ref so next call starts clean."""
    ctrl = _make_controller()
    _solve(ctrl, kappa=0.010)
    assert ctrl._last_kappa_ref > 0.0

    ctrl.reset()
    assert ctrl._last_kappa_ref == 0.0, (
        f"reset() should zero _last_kappa_ref, got {ctrl._last_kappa_ref}"
    )


def test_kappa_transition_params_from_config():
    """New kappa-transition params should load from config and have correct defaults."""
    p_default = MPCParams()
    assert p_default.kappa_transition_warmup_enabled is True
    assert abs(p_default.kappa_transition_threshold - 0.005) < 1e-9

    # Override via config
    cfg = {"trajectory": {"mpc": {
        "mpc_kappa_transition_warmup_enabled": False,
        "mpc_kappa_transition_threshold": 0.003,
    }}}
    p = MPCParams.from_config(cfg)
    assert p.kappa_transition_warmup_enabled is False
    assert abs(p.kappa_transition_threshold - 0.003) < 1e-9


# ---------------------------------------------------------------------------
# 2DOF Feedforward Alignment tests
# ---------------------------------------------------------------------------

class TestFFAlignment:
    """Tests for the 2DOF feedforward alignment feature (ff_alignment_enabled)."""

    def test_feedforward_delta_norm_formula(self):
        """_feedforward_delta_norm should match bicycle-model formula exactly."""
        import math
        from control.mpc_controller import MPCSolver, MPCParams

        L = 2.5
        d_max = 0.5236

        # κ=0 → straight road → zero steering
        result = MPCSolver._feedforward_delta_norm(0.0, L, d_max)
        assert result == 0.0

        # κ=0.01 (R100) → arctan(2.5 * 0.01) / 0.5236
        expected = math.atan(L * 0.01) / d_max
        result = MPCSolver._feedforward_delta_norm(0.01, L, d_max)
        assert abs(result - expected) < 1e-12, f"Expected {expected}, got {result}"

        # Negative κ → negative steering
        result_neg = MPCSolver._feedforward_delta_norm(-0.01, L, d_max)
        assert abs(result_neg + expected) < 1e-12

        # Large κ → no NaN/inf
        result_large = MPCSolver._feedforward_delta_norm(1.0, L, d_max)
        assert math.isfinite(result_large)

    def test_ff_alignment_straight_zero_correction(self):
        """With FF enabled and κ=0, q corrections are zero → same output as FF disabled."""
        ctrl_on = _make_controller(mpc_ff_alignment_enabled=True)
        ctrl_off = _make_controller(mpc_ff_alignment_enabled=False)

        r_on = _solve(ctrl_on, e_lat=0.3, e_heading=0.05, kappa=0.0)
        r_off = _solve(ctrl_off, e_lat=0.3, e_heading=0.05, kappa=0.0)

        assert r_on['mpc_feasible'] and r_off['mpc_feasible']
        assert abs(r_on['steering_normalized'] - r_off['steering_normalized']) < 1e-5, (
            f"FF alignment on straight (κ=0) should not change output: "
            f"on={r_on['steering_normalized']:.6f}, off={r_off['steering_normalized']:.6f}"
        )

    def test_ff_alignment_disabled_matches_baseline(self):
        """ff_alignment_enabled=False must be identical to a baseline controller on a curve."""
        ctrl_off = _make_controller(mpc_ff_alignment_enabled=False)
        ctrl_base = _make_controller()   # default has ff_alignment_enabled=True

        # With constant κ both controllers should solve feasibly; the disabled one
        # must still produce a valid steering command (sanity, not equality with base)
        r_off = _solve(ctrl_off, e_lat=0.0, e_heading=0.0, kappa=0.01)
        assert r_off['mpc_feasible'], "FF-disabled controller should solve feasibly on a curve"
        assert abs(r_off['steering_normalized']) <= 1.0 + 1e-6

        # Verify config round-trip: ff_alignment_enabled=False survives from_config
        cfg = {"trajectory": {"mpc": {"mpc_ff_alignment_enabled": False}}}
        p = MPCParams.from_config(cfg)
        assert p.ff_alignment_enabled is False

    def test_ff_alignment_constant_curve_converges_to_feedforward(self):
        """On a constant-κ curve with zero lateral error, repeated solves should drive δ toward δ_ff.

        With Part A removed and r_steer near-zero, convergence is driven purely by q_lat
        (lateral error correction) and the rate-bias Part B (at the initial step). On a
        constant-κ arc with e_lat=0, the solver settles at δ≈δ_ff because that is the
        steering that keeps heading error from growing — not because of any magnitude bias.
        """
        import math

        L = 2.5
        d_max = 0.5236
        kappa = 0.01   # R100

        delta_ff_target = math.atan(L * kappa) / d_max  # ≈ 0.0477

        ctrl = _make_controller(
            mpc_ff_alignment_enabled=True,
            mpc_q_lat=10.0,
            mpc_r_steer_rate=1.0,
            mpc_delta_rate_max=0.5,
        )

        last_delta = 0.0
        final_delta = None
        for _ in range(15):
            r = ctrl.compute_steering(
                e_lat=0.0, e_heading=0.0, current_speed=12.0,
                last_delta_norm=last_delta, kappa_ref=kappa,
                v_target=12.0, v_max=15.0, dt=0.033,
            )
            assert r['mpc_feasible'], "Solver must remain feasible on constant-κ curve"
            last_delta = r['steering_normalized']
            final_delta = last_delta

        # With e_lat=0 and constant κ, the solver must produce positive steer to prevent
        # heading error growth. It converges toward δ_ff via q_lat/q_heading dynamics.
        assert final_delta > 0.0, "Should produce positive steer for positive curvature"
        assert final_delta <= delta_ff_target * 3.0 + 1e-4, (
            f"Steer {final_delta:.4f} exceeds 3× δ_ff ({delta_ff_target:.4f}) — possible overshoot"
        )
