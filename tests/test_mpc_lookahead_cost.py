"""
Tests for lookahead-augmented MPC cost.

Validates that the QP cost function correctly penalises the lookahead
cross-track error  e_lat + Ld·e_heading  via cross-terms in the P matrix,
while keeping the dynamics model in the at-car Frenet frame where the
bicycle model is accurate.

Design properties tested:
  1. Ld=0 → identical to baseline (no cross-terms)
  2. P matrix has correct cross-term structure
  3. Heading error produces stronger steering when Ld > 0
  4. Cost function is equivalent to penalising lookahead cross-track
  5. Delay compensation is compatible with lookahead cost
  6. Closed-loop stability with lookahead cost
  7. Disabled flag → no effect regardless of Ld
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "control"))

from mpc_controller import MPCParams, MPCSolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lc_params(**overrides) -> MPCParams:
    """MPCParams with lookahead cost ready."""
    base = dict(
        dynamic_model_enabled=True,
        lookahead_cost_enabled=True,
        delay_compensation_enabled=False,
        horizon=20,
        dt=0.033,
        wheelbase_m=2.5,
        max_steer_rad=0.5236,
        q_lat=10.0,
        q_heading=5.0,
        q_speed=1.0,
        q_vy=0.1,
        q_yawrate=0.5,
        r_steer=1e-4,
        r_accel=0.05,
        r_steer_rate=1.0,
        vehicle_mass_kg=1500.0,
        vehicle_iz_kgm2=2428.0,
        vehicle_lf_m=1.125,
        vehicle_lr_m=1.125,
        tire_cf_nominal=36662.0,
        tire_cr_nominal=71799.0,
        r_steer_rate_scheduling_enabled=False,
        curvature_preview_enabled=False,
        leff_estimation_enabled=False,
        speed_adaptive_horizon=False,
        bias_enabled=False,
        ff_alignment_enabled=True,
        first_step_rate_enabled=True,
    )
    base.update(overrides)
    return MPCParams(**base)


def _solve(params, Ld=0.0, **kw):
    """Convenience solve wrapper with defaults."""
    s = MPCSolver(params)
    defaults = dict(
        e_lat=0.05,
        e_heading=0.02,
        v=12.0,
        last_delta_norm=0.0,
        kappa_ref_horizon=np.zeros(params.horizon),
        v_target=12.0,
        v_max=15.0,
        dt=params.dt,
        tire_cf=36662.0,
        tire_cr=71799.0,
        v_y_init=0.0,
        r_init=0.0,
        lookahead_distance=Ld,
    )
    defaults.update(kw)
    return s.solve(**defaults)


# ---------------------------------------------------------------------------
# Test: Ld=0 matches baseline
# ---------------------------------------------------------------------------

class TestLdZeroMatchesBaseline:
    """With Ld=0, lookahead cost produces identical output to disabled."""

    def test_ld_zero_same_as_disabled(self):
        p_on = _lc_params(lookahead_cost_enabled=True)
        p_off = _lc_params(lookahead_cost_enabled=False)
        r_on = _solve(p_on, Ld=0.0)
        r_off = _solve(p_off, Ld=0.0)
        assert r_on['feasible'] and r_off['feasible']
        assert r_on['steering_normalized'] == pytest.approx(
            r_off['steering_normalized'], abs=1e-8
        )

    def test_disabled_ignores_ld(self):
        """When disabled, Ld>0 has no effect."""
        p_off = _lc_params(lookahead_cost_enabled=False)
        r1 = _solve(p_off, Ld=0.0)
        r2 = _solve(p_off, Ld=5.0)
        r3 = _solve(p_off, Ld=10.0)
        assert r1['feasible'] and r2['feasible'] and r3['feasible']
        assert r1['steering_normalized'] == pytest.approx(r2['steering_normalized'], abs=1e-10)
        assert r1['steering_normalized'] == pytest.approx(r3['steering_normalized'], abs=1e-10)


# ---------------------------------------------------------------------------
# Test: P matrix cross-term structure
# ---------------------------------------------------------------------------

class TestPMatrixStructure:
    """Verify the P matrix has cross-terms when Ld > 0."""

    def test_cross_term_present(self):
        """P should have off-diagonal entries at (e_lat, e_heading) per step."""
        p = _lc_params(lookahead_cost_enabled=True)
        s = MPCSolver(p)
        # Set Ld and rebuild
        s._current_Ld = 5.0
        s._build_qp()
        P = s._P.toarray()
        nx, nu = s._nx, s._nu
        N = s._N
        # Check first step cross-term
        base = 0
        cross_val = P[base + 0, base + 1]  # upper triangle
        expected = p.q_lat * 5.0 * p.q_lat_preview_step0_scale
        assert cross_val == pytest.approx(expected, rel=1e-6), (
            f"Cross-term at step 0: {cross_val}, expected {expected}"
        )
        # Check last step (full ramp weight = 1.0)
        base_last = (N - 1) * (nx + nu)
        cross_last = P[base_last + 0, base_last + 1]
        expected_last = p.q_lat * 5.0  # _w_k = 1.0 at last step
        assert cross_last == pytest.approx(expected_last, rel=1e-6)
        # Terminal
        term_base = N * (nx + nu)
        cross_term = P[term_base + 0, term_base + 1]
        expected_term = p.q_lat * p.q_lat_terminal_scale * 5.0
        assert cross_term == pytest.approx(expected_term, rel=1e-6)

    def test_heading_diagonal_augmented(self):
        """P[e_heading, e_heading] should include q_lat * Ld²."""
        p = _lc_params(lookahead_cost_enabled=True, q_lat=10.0, q_heading=5.0)
        s = MPCSolver(p)
        s._current_Ld = 4.0
        s._build_qp()
        P = s._P.toarray()
        nx, nu = s._nx, s._nu
        N = s._N
        # Last step (full weight)
        base = (N - 1) * (nx + nu)
        heading_diag = P[base + 1, base + 1]
        # q_heading + q_lat * Ld² * _w_k (at last step, _w_k=1.0)
        expected = 5.0 + 10.0 * 16.0  # 165.0
        assert heading_diag == pytest.approx(expected, rel=1e-6)

    def test_no_cross_term_when_ld_zero(self):
        """No cross-terms when Ld=0 even with lookahead_cost_enabled=True."""
        p = _lc_params(lookahead_cost_enabled=True)
        s = MPCSolver(p)
        s._current_Ld = 0.0
        s._build_qp()
        P = s._P.toarray()
        nx, nu = s._nx, s._nu
        base = 0
        cross_val = P[base + 0, base + 1]
        assert cross_val == 0.0


# ---------------------------------------------------------------------------
# Test: heading error produces stronger correction with Ld > 0
# ---------------------------------------------------------------------------

class TestHeadingCoupling:
    """Heading error should produce more aggressive steering when Ld > 0,
    because the cost penalises heading projected forward to the lookahead."""

    def test_heading_drives_stronger_steer(self):
        """Pure heading error (e_lat=0): Ld>0 should steer more than Ld=0."""
        p = _lc_params(q_lat=5.0, q_heading=2.0)
        # Pure heading error, no lateral offset
        r_ld0 = _solve(p, Ld=0.0, e_lat=0.0, e_heading=0.03)
        r_ld5 = _solve(p, Ld=5.0, e_lat=0.0, e_heading=0.03)
        assert r_ld0['feasible'] and r_ld5['feasible']
        # Heading error means car will drift — with Ld>0, the cost sees
        # e_lat_la = 0 + 5.0*0.03 = 0.15m lookahead error. Should steer harder.
        assert abs(r_ld5['steering_normalized']) > abs(r_ld0['steering_normalized']) * 1.3, (
            f"Ld=5: {r_ld5['steering_normalized']:.6f}, "
            f"Ld=0: {r_ld0['steering_normalized']:.6f}"
        )

    def test_same_sign_correction(self):
        """e_lat and e_heading in same direction: Ld>0 reinforces, doesn't flip."""
        p = _lc_params(q_lat=5.0, q_heading=2.0)
        # Car right (e_lat>0) and pointed right (e_heading>0): should steer left (negative)
        r_ld0 = _solve(p, Ld=0.0, e_lat=0.03, e_heading=0.02)
        r_ld5 = _solve(p, Ld=5.0, e_lat=0.03, e_heading=0.02)
        assert r_ld0['feasible'] and r_ld5['feasible']
        # Both should steer negative (left). Ld>0 should be MORE negative.
        assert r_ld0['steering_normalized'] < 0
        assert r_ld5['steering_normalized'] < 0
        assert r_ld5['steering_normalized'] < r_ld0['steering_normalized']


# ---------------------------------------------------------------------------
# Test: equivalence to lookahead cross-track penalty
# ---------------------------------------------------------------------------

class TestLookaheadEquivalence:
    """The augmented cost at step 0 is q_lat*(e_lat + Ld*e_heading)².
    Verify this by comparing steering at two (e_lat, e_heading) pairs that
    produce the same lookahead cross-track."""

    def test_same_lookahead_same_steer(self):
        """Two states with identical e_lat + Ld*e_heading should produce
        similar initial-step cost, hence similar first steering command."""
        Ld = 5.0
        # State A: e_lat=0.05, e_heading=0.01 → la = 0.05 + 5*0.01 = 0.10
        # State B: e_lat=0.10, e_heading=0.00 → la = 0.10 + 5*0.00 = 0.10
        # Same lookahead error → similar cost at step 0
        p = _lc_params(q_lat=5.0, q_heading=0.1)  # low q_heading so cross-term dominates
        r_a = _solve(p, Ld=Ld, e_lat=0.05, e_heading=0.01)
        r_b = _solve(p, Ld=Ld, e_lat=0.10, e_heading=0.00)
        assert r_a['feasible'] and r_b['feasible']
        # Not exactly equal (dynamics differ because e_heading affects trajectory),
        # but the first-step steering should be in the same ballpark.
        diff = abs(r_a['steering_normalized'] - r_b['steering_normalized'])
        avg = (abs(r_a['steering_normalized']) + abs(r_b['steering_normalized'])) / 2
        assert diff < avg * 0.5, (
            f"Expected similar steering for same lookahead error: "
            f"A={r_a['steering_normalized']:.6f}, B={r_b['steering_normalized']:.6f}"
        )


# ---------------------------------------------------------------------------
# Test: delay compensation compatibility
# ---------------------------------------------------------------------------

class TestDelayCompCompatibility:
    """Delay comp + lookahead cost should work together."""

    def test_delay_comp_with_lookahead_cost(self):
        p = _lc_params(
            lookahead_cost_enabled=True,
            delay_compensation_enabled=True,
            delay_compensation_frames=2,
        )
        r = _solve(p, Ld=5.0, committed_steering_norm=[0.0, 0.0])
        assert r['feasible']
        assert r['delay_comp_active'] is True
        assert r['lookahead_cost_active'] is True

    def test_delay_comp_shifts_state_with_la_cost(self):
        """Forward sim should shift e_lat with at-car dynamics (accurate)."""
        p_on = _lc_params(
            lookahead_cost_enabled=True,
            delay_compensation_enabled=True,
            delay_compensation_frames=2,
            q_lat=2.0,
        )
        p_off = _lc_params(
            lookahead_cost_enabled=True,
            delay_compensation_enabled=False,
            q_lat=2.0,
        )
        r_on = _solve(p_on, Ld=5.0, committed_steering_norm=[0.0, 0.0],
                       e_lat=0.02, e_heading=0.01)
        r_off = _solve(p_off, Ld=5.0, e_lat=0.02, e_heading=0.01)
        assert r_on['feasible'] and r_off['feasible']
        # Steering should differ because initial state is forward-simulated
        assert r_on['steering_normalized'] != pytest.approx(
            r_off['steering_normalized'], abs=1e-4
        )


# ---------------------------------------------------------------------------
# Test: closed-loop stability
# ---------------------------------------------------------------------------

class TestClosedLoopStability:
    """100-frame closed-loop with lookahead cost at q_lat=10 → converges."""

    def test_convergence(self):
        p = _lc_params(q_lat=10.0, q_heading=5.0)
        solver = MPCSolver(p)
        e_lat, e_heading = 0.15, 0.0
        v_y, r = 0.0, 0.0
        v = 12.0
        last_delta = 0.0
        kappa = np.zeros(p.horizon)
        dt = p.dt
        Ld = 5.3  # typical lookahead at 12 m/s

        e_lats = [e_lat]
        steerings = []
        for _ in range(100):
            result = solver.solve(
                e_lat=e_lat, e_heading=e_heading, v=v,
                last_delta_norm=last_delta, kappa_ref_horizon=kappa,
                v_target=12.0, v_max=15.0, dt=dt,
                tire_cf=36662.0, tire_cr=71799.0,
                v_y_init=v_y, r_init=r,
                lookahead_distance=Ld,
            )
            if not result['feasible']:
                continue
            delta = result['steering_normalized']
            steerings.append(delta)
            last_delta = delta
            # Plant sim (kinematic)
            _ms = solver._max_steer_at_speed(v)
            steer_rad = delta * _ms
            e_heading += (v / p.wheelbase_m) * steer_rad * dt
            e_lat += v * e_heading * dt
            e_lats.append(e_lat)

        # Should converge
        assert abs(e_lats[-1]) < abs(e_lats[0]) * 0.3, (
            f"e_lat did not converge: {e_lats[0]:.4f} → {e_lats[-1]:.4f}"
        )

    def test_no_oscillation(self):
        p = _lc_params(q_lat=10.0, q_heading=5.0)
        solver = MPCSolver(p)
        e_lat, e_heading = 0.15, 0.0
        v = 12.0
        last_delta = 0.0
        kappa = np.zeros(p.horizon)
        dt = p.dt
        Ld = 5.3

        steerings = []
        for _ in range(100):
            result = solver.solve(
                e_lat=e_lat, e_heading=e_heading, v=v,
                last_delta_norm=last_delta, kappa_ref_horizon=kappa,
                v_target=12.0, v_max=15.0, dt=dt,
                tire_cf=36662.0, tire_cr=71799.0,
                v_y_init=0.0, r_init=0.0,
                lookahead_distance=Ld,
            )
            if not result['feasible']:
                continue
            delta = result['steering_normalized']
            steerings.append(delta)
            last_delta = delta
            _ms = solver._max_steer_at_speed(v)
            steer_rad = delta * _ms
            e_heading += (v / p.wheelbase_m) * steer_rad * dt
            e_lat += v * e_heading * dt

        sign_changes = sum(
            1 for i in range(1, len(steerings))
            if steerings[i] * steerings[i-1] < 0
        )
        osc_ratio = sign_changes / max(len(steerings) - 1, 1)
        assert osc_ratio < 0.15, (
            f"Oscillation: {sign_changes}/{len(steerings)-1} sign changes ({osc_ratio:.0%})"
        )


# ---------------------------------------------------------------------------
# Test: diagnostics
# ---------------------------------------------------------------------------

class TestDiagnostics:
    """Result dict includes lookahead cost diagnostics."""

    def test_active_when_enabled_with_ld(self):
        p = _lc_params(lookahead_cost_enabled=True)
        r = _solve(p, Ld=5.0)
        assert r['lookahead_cost_active'] is True
        assert r['lookahead_cost_Ld'] == pytest.approx(5.0, abs=0.5)

    def test_inactive_when_ld_zero(self):
        p = _lc_params(lookahead_cost_enabled=True)
        r = _solve(p, Ld=0.0)
        assert r['lookahead_cost_active'] is False

    def test_inactive_when_disabled(self):
        p = _lc_params(lookahead_cost_enabled=False)
        r = _solve(p, Ld=5.0)
        assert r['lookahead_cost_active'] is False


# ---------------------------------------------------------------------------
# Test: Ld rebuild hysteresis
# ---------------------------------------------------------------------------

class TestLdRebuild:
    """P matrix rebuilds only when Ld changes by >0.5m."""

    def test_small_ld_change_no_rebuild(self):
        """Ld change of 0.3m should NOT trigger rebuild."""
        p = _lc_params(lookahead_cost_enabled=True)
        s = MPCSolver(p)
        r1 = s.solve(
            e_lat=0.05, e_heading=0.02, v=12.0, last_delta_norm=0.0,
            kappa_ref_horizon=np.zeros(p.horizon), v_target=12.0, v_max=15.0,
            dt=p.dt, tire_cf=36662.0, tire_cr=71799.0,
            v_y_init=0.0, r_init=0.0, lookahead_distance=5.0,
        )
        r2 = s.solve(
            e_lat=0.05, e_heading=0.02, v=12.0, last_delta_norm=0.0,
            kappa_ref_horizon=np.zeros(p.horizon), v_target=12.0, v_max=15.0,
            dt=p.dt, tire_cf=36662.0, tire_cr=71799.0,
            v_y_init=0.0, r_init=0.0, lookahead_distance=5.3,
        )
        assert r1['feasible'] and r2['feasible']
        # Same P matrix → same result (identical inputs)
        assert r1['steering_normalized'] == pytest.approx(
            r2['steering_normalized'], abs=1e-10
        )

    def test_large_ld_change_triggers_rebuild(self):
        """Ld change of 1.0m should trigger rebuild → different output."""
        # Use small errors to avoid steering saturation
        p = _lc_params(lookahead_cost_enabled=True, q_lat=2.0)
        s = MPCSolver(p)
        r1 = s.solve(
            e_lat=0.01, e_heading=0.005, v=12.0, last_delta_norm=0.0,
            kappa_ref_horizon=np.zeros(p.horizon), v_target=12.0, v_max=15.0,
            dt=p.dt, tire_cf=36662.0, tire_cr=71799.0,
            v_y_init=0.0, r_init=0.0, lookahead_distance=5.0,
        )
        r2 = s.solve(
            e_lat=0.01, e_heading=0.005, v=12.0, last_delta_norm=0.0,
            kappa_ref_horizon=np.zeros(p.horizon), v_target=12.0, v_max=15.0,
            dt=p.dt, tire_cf=36662.0, tire_cr=71799.0,
            v_y_init=0.0, r_init=0.0, lookahead_distance=6.5,
        )
        assert r1['feasible'] and r2['feasible']
        # Ld changed by 1.5 > 0.5 threshold → different P → different result
        assert r1['steering_normalized'] != pytest.approx(
            r2['steering_normalized'], abs=1e-6
        ), "Expected different steering after Ld rebuild"


# ---------------------------------------------------------------------------
# Test: kinematic model supported
# ---------------------------------------------------------------------------

class TestKinematicModel:
    """Lookahead cost works with the kinematic (3-state) model too."""

    def test_kinematic_with_lookahead_cost(self):
        p = _lc_params(dynamic_model_enabled=False)
        r = _solve(p, Ld=5.0)
        assert r['feasible']
        assert r['lookahead_cost_active'] is True

    def test_kinematic_heading_coupling(self):
        """Same heading coupling test with kinematic model."""
        p = _lc_params(dynamic_model_enabled=False, q_lat=5.0, q_heading=2.0)
        r_ld0 = _solve(p, Ld=0.0, e_lat=0.0, e_heading=0.03)
        r_ld5 = _solve(p, Ld=5.0, e_lat=0.0, e_heading=0.03)
        assert r_ld0['feasible'] and r_ld5['feasible']
        assert abs(r_ld5['steering_normalized']) > abs(r_ld0['steering_normalized']) * 1.2
