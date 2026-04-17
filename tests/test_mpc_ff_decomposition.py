"""Tests for 2DOF feedforward/feedback decomposition in MPC.

The decomposition reformulates the QP so the decision variable is
ε = δ − δ_ff (correction from feedforward) instead of δ (total steering).
Curvature cancels in dynamics — MPC sees a near-straight-road problem.
"""

import math

import numpy as np
import pytest

from control.mpc_controller import MPCParams, MPCSolver


def _make_params(**overrides) -> MPCParams:
    defaults = dict(
        ff_decomposition_enabled=True,
        ff_alignment_enabled=True,  # should be auto-skipped when decomp is on
        horizon=20,
        dt=0.1,
        q_lat=2.0,
        q_heading=5.0,
        q_speed=1.0,
        r_steer=1e-4,
        r_accel=0.05,
        r_steer_rate=2.0,
        delta_rate_max=0.5,
        first_step_rate_enabled=True,
        wheelbase_m=2.5,
    )
    defaults.update(overrides)
    return MPCParams(**defaults)


def _solve(solver, **kwargs):
    defaults = dict(
        e_lat=0.0, e_heading=0.0, v=10.0,
        last_delta_norm=0.0,
        kappa_ref_horizon=np.zeros(solver._N),
        v_target=12.0, v_max=15.0, dt=0.1,
    )
    defaults.update(kwargs)
    return solver.solve(**defaults)


# ─────────────────── Construction ───────────────────


class TestFFDecompConstruction:
    def test_nx_remains_3(self):
        """Decomposition doesn't add states — it only transforms the control variable."""
        p = _make_params()
        s = MPCSolver(p)
        assert s._nx == 3

    def test_last_delta_ff_initialized_zero(self):
        p = _make_params()
        s = MPCSolver(p)
        assert s._last_delta_ff == 0.0

    def test_disabled_by_default(self):
        """Default MPCParams has decomposition disabled."""
        p = MPCParams()
        assert p.ff_decomposition_enabled is False


# ─────────────────── Feasibility ───────────────────


class TestFFDecompFeasibility:
    def test_straight_road_feasible(self):
        s = MPCSolver(_make_params())
        r = _solve(s)
        assert r['feasible']

    def test_curved_road_feasible(self):
        s = MPCSolver(_make_params())
        kappa = np.full(s._N, 0.02)  # R=50m
        r = _solve(s, kappa_ref_horizon=kappa)
        assert r['feasible']

    def test_tight_curve_feasible(self):
        s = MPCSolver(_make_params())
        kappa = np.full(s._N, 0.067)  # R=15m
        r = _solve(s, kappa_ref_horizon=kappa, v=5.0)
        assert r['feasible']

    def test_multiple_frames_feasible(self):
        """Multi-frame stability: solve 10 consecutive frames."""
        s = MPCSolver(_make_params())
        kappa = np.full(s._N, 0.02)
        last_delta = 0.0
        for _ in range(10):
            r = _solve(s, kappa_ref_horizon=kappa, last_delta_norm=last_delta)
            assert r['feasible']
            last_delta = r['steering_normalized']


# ─────────────────── Straight road invariance ───────────────────


class TestFFDecompStraightRoad:
    """On straight road (κ=0), δ_ff=0, so ε=δ — results should match non-decomp."""

    def test_straight_matches_baseline(self):
        p_on = _make_params(ff_decomposition_enabled=True)
        p_off = _make_params(ff_decomposition_enabled=False)
        s_on = MPCSolver(p_on)
        s_off = MPCSolver(p_off)

        r_on = _solve(s_on, e_lat=0.1)
        r_off = _solve(s_off, e_lat=0.1)

        assert abs(r_on['steering_normalized'] - r_off['steering_normalized']) < 0.01

    def test_straight_delta_ff_is_zero(self):
        s = MPCSolver(_make_params())
        r = _solve(s)
        assert r.get('ff_decomp_delta_ff', 0.0) == pytest.approx(0.0, abs=1e-6)


# ─────────────────── Curve tracking ───────────────────


class TestFFDecompCurveTracking:
    def test_curve_produces_nonzero_delta_ff(self):
        s = MPCSolver(_make_params())
        kappa = np.full(s._N, 0.02)  # R=50m
        r = _solve(s, kappa_ref_horizon=kappa)
        assert abs(r['ff_decomp_delta_ff']) > 0.001

    def test_curve_delta_ff_matches_kinematic(self):
        """δ_ff should be arctan(L·κ)/max_steer_rad."""
        p = _make_params()
        s = MPCSolver(p)
        kappa_val = 0.02
        kappa = np.full(s._N, kappa_val)
        r = _solve(s, kappa_ref_horizon=kappa, v=10.0)

        max_steer = s._max_steer_at_speed(10.0)
        expected_ff = math.atan(p.wheelbase_m * kappa_val) / max_steer
        assert r['ff_decomp_delta_ff'] == pytest.approx(expected_ff, abs=1e-6)

    def test_curve_total_steering_includes_ff(self):
        """Total δ should be approximately δ_ff + small correction for zero e_lat."""
        s = MPCSolver(_make_params())
        kappa = np.full(s._N, 0.02)
        r = _solve(s, kappa_ref_horizon=kappa, e_lat=0.0, e_heading=0.0)
        total = r['steering_normalized']
        delta_ff = r['ff_decomp_delta_ff']
        eps = r['ff_decomp_epsilon']
        # total = ε + δ_ff (before clip)
        assert total == pytest.approx(eps + delta_ff, abs=0.02)

    def test_epsilon_small_on_perfect_track(self):
        """With zero error on a curve, ε should be near zero — feedforward handles it."""
        s = MPCSolver(_make_params())
        kappa = np.full(s._N, 0.02)
        r = _solve(s, kappa_ref_horizon=kappa, e_lat=0.0, e_heading=0.0)
        assert abs(r['ff_decomp_epsilon']) < 0.15

    def test_decomp_steers_more_aggressively_into_curve(self):
        """With decomposition, curve-entry steering should be >= non-decomp at same q_lat."""
        p_on = _make_params(ff_decomposition_enabled=True, q_lat=2.0)
        p_off = _make_params(ff_decomposition_enabled=False, q_lat=2.0)
        s_on = MPCSolver(p_on)
        s_off = MPCSolver(p_off)

        kappa = np.full(s_on._N, 0.02)
        r_on = _solve(s_on, kappa_ref_horizon=kappa, e_lat=0.0)
        r_off = _solve(s_off, kappa_ref_horizon=kappa, e_lat=0.0)

        # Decomp should provide at least as much steering into the curve
        assert abs(r_on['steering_normalized']) >= abs(r_off['steering_normalized']) * 0.9


# ─────────────────── State persistence ───────────────────


class TestFFDecompStatePersistence:
    def test_last_delta_ff_updated(self):
        s = MPCSolver(_make_params())
        kappa = np.full(s._N, 0.02)
        _solve(s, kappa_ref_horizon=kappa)
        assert s._last_delta_ff != 0.0

    def test_last_delta_ff_reset_on_straight(self):
        s = MPCSolver(_make_params())
        # First: curve
        kappa = np.full(s._N, 0.02)
        _solve(s, kappa_ref_horizon=kappa)
        assert s._last_delta_ff != 0.0
        # Then: straight
        _solve(s, kappa_ref_horizon=np.zeros(s._N))
        assert s._last_delta_ff == pytest.approx(0.0, abs=1e-6)

    def test_infeasible_preserves_last_delta_ff(self):
        """On infeasible solve, _last_delta_ff should NOT be updated."""
        s = MPCSolver(_make_params(delta_rate_max=0.001))
        kappa = np.full(s._N, 0.02)
        old_ff = s._last_delta_ff
        # Force a large jump that should be infeasible with tiny rate limit
        r = _solve(s, kappa_ref_horizon=kappa, last_delta_norm=-0.9)
        if not r['feasible']:
            assert s._last_delta_ff == old_ff


# ─────────────────── Diagnostics ───────────────────


class TestFFDecompDiagnostics:
    def test_result_contains_diagnostics(self):
        s = MPCSolver(_make_params())
        kappa = np.full(s._N, 0.02)
        r = _solve(s, kappa_ref_horizon=kappa)
        assert 'ff_decomp_delta_ff' in r
        assert 'ff_decomp_epsilon' in r

    def test_no_diagnostics_when_disabled(self):
        s = MPCSolver(_make_params(ff_decomposition_enabled=False))
        r = _solve(s, e_lat=0.1)
        assert 'ff_decomp_delta_ff' not in r
        assert 'ff_decomp_epsilon' not in r


# ─────────────────── Interaction with other features ───────────────────


class TestFFDecompInteractions:
    def test_ff_alignment_skipped_when_decomp_active(self):
        """ff_alignment should be skipped when decomposition is on."""
        p1 = _make_params(ff_decomposition_enabled=True, ff_alignment_enabled=True)
        p2 = _make_params(ff_decomposition_enabled=True, ff_alignment_enabled=False)
        s1 = MPCSolver(p1)
        s2 = MPCSolver(p2)

        kappa = np.full(s1._N, 0.02)
        r1 = _solve(s1, kappa_ref_horizon=kappa, e_lat=0.1)
        r2 = _solve(s2, kappa_ref_horizon=kappa, e_lat=0.1)
        # Both should produce identical results since ff_alignment is skipped
        assert r1['steering_normalized'] == pytest.approx(r2['steering_normalized'], abs=1e-6)

    def test_incompatible_with_dynamic_model(self):
        """Dynamic model should override decomposition (decomp only for kinematic)."""
        p = _make_params(ff_decomposition_enabled=True, dynamic_model_enabled=True)
        s = MPCSolver(p)
        kappa = np.full(s._N, 0.02)
        r = _solve(s, kappa_ref_horizon=kappa, e_lat=0.1,
                   v_y_init=0.0, r_init=0.0)
        # Should NOT have decomp diagnostics (dynamic overrides)
        assert 'ff_decomp_delta_ff' not in r

    def test_incompatible_with_actuator_model(self):
        """Actuator model changes nx; decomp should still work or be skipped gracefully."""
        p = _make_params(ff_decomposition_enabled=True,
                         actuator_model_enabled=True,
                         actuator_tau_s=0.70)
        s = MPCSolver(p)
        kappa = np.full(s._N, 0.02)
        r = _solve(s, kappa_ref_horizon=kappa, delta_actual_norm=0.0)
        assert r['feasible']
