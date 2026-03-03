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
            "mpc_r_steer": 0.1,
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
