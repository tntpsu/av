"""
Tests for MPC delay compensation (forward-simulate committed pipeline steps).

Validates that the optimizer's initial state is shifted forward by d committed
frames so its first free decision aligns with reality, preventing the
overcorrection → delayed response → oscillation cycle.
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

def _dc_params(**overrides) -> MPCParams:
    """MPCParams with delay compensation enabled."""
    base = dict(
        dynamic_model_enabled=True,
        delay_compensation_enabled=True,
        delay_compensation_frames=2,
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


def _solve(params, committed=None, **kw):
    """Convenience solve wrapper with defaults."""
    s = MPCSolver(params)
    defaults = dict(
        e_lat=0.1,
        e_heading=0.05,
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
    )
    defaults.update(kw)
    return s.solve(**defaults, committed_steering_norm=committed)


# ---------------------------------------------------------------------------
# Test: disabled matches baseline
# ---------------------------------------------------------------------------

class TestDisabledMatchesBaseline:
    """delay_comp OFF → identical output regardless of committed_steering_norm."""

    def test_disabled_ignores_committed(self):
        p_off = _dc_params(delay_compensation_enabled=False)
        r1 = _solve(p_off, committed=None)
        r2 = _solve(p_off, committed=[0.0, 0.0])
        r3 = _solve(p_off, committed=[0.5, 0.3])
        assert r1['feasible'] and r2['feasible'] and r3['feasible']
        assert r1['steering_normalized'] == pytest.approx(r2['steering_normalized'], abs=1e-10)
        assert r1['steering_normalized'] == pytest.approx(r3['steering_normalized'], abs=1e-10)

    def test_disabled_no_delay_comp_flag(self):
        p_off = _dc_params(delay_compensation_enabled=False)
        r = _solve(p_off, committed=[0.0, 0.0])
        assert r['delay_comp_active'] is False
        assert 'delay_comp_e_lat_predicted' not in r


# ---------------------------------------------------------------------------
# Test: forward sim shifts initial state
# ---------------------------------------------------------------------------

class TestForwardSimShiftsState:
    """e_lat=0.1, e_heading=0.05, committed=[0,0] → QP starts from different state."""

    def test_state_shifted_vs_no_comp(self):
        """With committed=[0,0], the forward sim propagates e_lat via
        v*e_heading*dt for 2 steps. The QP should see a different state."""
        # Use small errors to avoid steering saturation (q_lat=2 keeps it linear)
        p_on = _dc_params(delay_compensation_enabled=True, q_lat=2.0)
        p_off = _dc_params(delay_compensation_enabled=False, q_lat=2.0)
        r_on = _solve(p_on, committed=[0.0, 0.0], e_lat=0.02, e_heading=0.01)
        r_off = _solve(p_off, committed=None, e_lat=0.02, e_heading=0.01)
        assert r_on['feasible'] and r_off['feasible']
        assert r_on['delay_comp_active'] is True
        # Predicted e_lat should differ from raw (0.02)
        pred_e_lat = r_on['delay_comp_e_lat_predicted']
        # At v=12, e_heading=0.01, dt=0.033: e_lat grows by ~12*0.01*0.033≈0.004 per step
        assert pred_e_lat > 0.02, f"Forward sim should shift e_lat forward, got {pred_e_lat}"
        # Steering should differ because initial state differs
        assert r_on['steering_normalized'] != pytest.approx(
            r_off['steering_normalized'], abs=1e-4
        ), "Delay comp should change QP output"

    def test_diagnostics_populated(self):
        p = _dc_params()
        r = _solve(p, committed=[0.0, 0.0])
        assert r['delay_comp_active'] is True
        assert 'delay_comp_e_lat_predicted' in r
        assert 'delay_comp_e_heading_predicted' in r
        assert 'delay_comp_last_committed_norm' in r


# ---------------------------------------------------------------------------
# Test: forward sim accuracy — manual vs solver
# ---------------------------------------------------------------------------

class TestForwardSimAccuracy:
    """Manual A@x+B@u+c computation vs solver's internal forward simulation."""

    def test_manual_vs_solver_dynamic(self):
        """Compute one-step forward sim manually and verify it matches
        the solver's delay_comp_e_lat_predicted for d=1."""
        p = _dc_params(delay_compensation_frames=1)
        e_lat, e_heading, v_y, r, v = 0.1, 0.05, 0.0, 0.0, 12.0
        u_steer = 0.2
        dt = p.dt
        C_f, C_r = 36662.0, 71799.0
        m = p.vehicle_mass_kg
        Iz = p.vehicle_iz_kgm2
        l_f, l_r = p.vehicle_lf_m, p.vehicle_lr_m
        _ms = MPCSolver(p)._max_steer_at_speed(v)

        # Manual one-step (kappa=0)
        e_lat_next = e_lat + v * e_heading * dt + v_y * dt
        e_heading_next = e_heading + r * dt  # kappa=0
        v_y_next = (v_y * (1.0 - (C_f + C_r) / (m * v) * dt)
                    + r * (-v - (C_f * l_f - C_r * l_r) / (m * v)) * dt
                    + u_steer * C_f * _ms / m * dt)

        result = _solve(p, committed=[u_steer],
                        e_lat=e_lat, e_heading=e_heading,
                        v_y_init=v_y, r_init=r, v=v)
        assert result['delay_comp_active'] is True
        assert result['delay_comp_e_lat_predicted'] == pytest.approx(e_lat_next, abs=1e-8)
        assert result['delay_comp_e_heading_predicted'] == pytest.approx(e_heading_next, abs=1e-8)


# ---------------------------------------------------------------------------
# Test: rate constraint uses last committed
# ---------------------------------------------------------------------------

class TestRateConstraintUsesLastCommitted:
    """committed=[0.0, 0.1] → first free cmd rate-bounded from 0.1, not last_delta_norm."""

    def test_rate_ref_from_last_committed(self):
        p = _dc_params()
        # With delay comp: last committed is 0.3, so rate is bounded from 0.3
        # Without: rate is bounded from last_delta_norm=0.0
        r_on = _solve(p, committed=[0.0, 0.3], last_delta_norm=0.0)
        r_off_params = _dc_params(delay_compensation_enabled=False)
        r_off = _solve(r_off_params, committed=None, last_delta_norm=0.0)
        assert r_on['feasible'] and r_off['feasible']
        assert r_on['delay_comp_last_committed_norm'] == pytest.approx(0.3, abs=1e-10)
        # The steering outputs should differ because the rate reference differs
        # (and the initial state differs due to forward sim)
        assert abs(r_on['steering_normalized'] - r_off['steering_normalized']) > 1e-4


# ---------------------------------------------------------------------------
# Test: kappa shift
# ---------------------------------------------------------------------------

class TestKappaShift:
    """Distinctive kappa pattern → QP sees shifted curvature."""

    def test_kappa_shifted_by_d(self):
        """Create a kappa with step at index 5. With d=2, the QP should
        see the step at index 3 (shifted left by 2)."""
        # Use small errors and q_lat=2 to keep steering in the linear region
        p = _dc_params(q_lat=2.0)
        N = p.horizon
        kappa = np.zeros(N)
        kappa[5:] = 0.05  # Step change at index 5

        r_on = _solve(p, committed=[0.0, 0.0], kappa_ref_horizon=kappa,
                       e_lat=0.02, e_heading=0.01)
        p_off = _dc_params(delay_compensation_enabled=False, q_lat=2.0)
        r_off = _solve(p_off, committed=None, kappa_ref_horizon=kappa,
                        e_lat=0.02, e_heading=0.01)
        assert r_on['feasible'] and r_off['feasible']
        # Both produce feasible solutions; steering differs because
        # (a) initial state is forward-simulated and (b) kappa is shifted.
        assert r_on['steering_normalized'] != pytest.approx(
            r_off['steering_normalized'], abs=1e-4
        )


# ---------------------------------------------------------------------------
# Test: kinematic model supported
# ---------------------------------------------------------------------------

class TestKinematicModelSupported:
    """Same delay comp tests with dynamic_model_enabled=False."""

    def test_kinematic_forward_sim(self):
        p = _dc_params(dynamic_model_enabled=False)
        r = _solve(p, committed=[0.0, 0.0],
                    tire_cf=36662.0, tire_cr=71799.0)
        assert r['feasible']
        assert r['delay_comp_active'] is True
        pred_e_lat = r['delay_comp_e_lat_predicted']
        # Kinematic: e_lat_next = e_lat + v * e_heading * dt
        # After 2 steps: ~0.1 + 2*(12*0.05*0.033) ≈ 0.140
        assert pred_e_lat > 0.1

    def test_kinematic_disabled_matches(self):
        p_off = _dc_params(dynamic_model_enabled=False, delay_compensation_enabled=False)
        r1 = _solve(p_off, committed=None)
        r2 = _solve(p_off, committed=[0.1, 0.2])
        assert r1['steering_normalized'] == pytest.approx(r2['steering_normalized'], abs=1e-10)


# ---------------------------------------------------------------------------
# Test: closed-loop stability at higher q_lat
# ---------------------------------------------------------------------------

class TestHigherQLat:
    """100-frame closed-loop at q_lat=4.0 with delay comp → no oscillation."""

    def test_no_oscillation_at_q4(self):
        p = _dc_params(q_lat=4.0)
        solver = MPCSolver(p)
        e_lat, e_heading = 0.15, 0.0
        v_y, r = 0.0, 0.0
        v = 12.0
        last_delta = 0.0
        kappa = np.zeros(p.horizon)
        dt = p.dt

        steerings = []
        for _ in range(100):
            # Ring buffer: last 2 commands
            committed = [steerings[-2] if len(steerings) >= 2 else 0.0,
                         steerings[-1] if len(steerings) >= 1 else 0.0]
            result = solver.solve(
                e_lat=e_lat, e_heading=e_heading, v=v,
                last_delta_norm=last_delta, kappa_ref_horizon=kappa,
                v_target=12.0, v_max=15.0, dt=dt,
                tire_cf=36662.0, tire_cr=71799.0,
                v_y_init=v_y, r_init=r,
                committed_steering_norm=committed,
            )
            if not result['feasible']:
                continue
            delta = result['steering_normalized']
            steerings.append(delta)
            last_delta = delta

            # Simple kinematic plant sim for closed-loop
            _ms = solver._max_steer_at_speed(v)
            steer_rad = delta * _ms
            e_heading += (v / p.wheelbase_m) * steer_rad * dt
            e_lat += v * e_heading * dt

        # Count sign changes in steering — oscillation shows as frequent reversals
        sign_changes = sum(
            1 for i in range(1, len(steerings))
            if steerings[i] * steerings[i-1] < 0
        )
        osc_ratio = sign_changes / max(len(steerings) - 1, 1)
        assert osc_ratio < 0.15, (
            f"Oscillation detected: {sign_changes}/{len(steerings)-1} sign changes "
            f"({osc_ratio:.0%})"
        )

    def test_e_lat_converges(self):
        """e_lat should converge toward 0 within 100 frames at q_lat=4."""
        p = _dc_params(q_lat=4.0)
        solver = MPCSolver(p)
        e_lat, e_heading = 0.15, 0.0
        v_y, r = 0.0, 0.0
        v = 12.0
        last_delta = 0.0
        kappa = np.zeros(p.horizon)
        dt = p.dt

        steerings = []
        e_lats = [e_lat]
        for _ in range(100):
            committed = [steerings[-2] if len(steerings) >= 2 else 0.0,
                         steerings[-1] if len(steerings) >= 1 else 0.0]
            result = solver.solve(
                e_lat=e_lat, e_heading=e_heading, v=v,
                last_delta_norm=last_delta, kappa_ref_horizon=kappa,
                v_target=12.0, v_max=15.0, dt=dt,
                tire_cf=36662.0, tire_cr=71799.0,
                v_y_init=v_y, r_init=r,
                committed_steering_norm=committed,
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
            e_lats.append(e_lat)

        # Final e_lat should be much smaller than initial
        assert abs(e_lats[-1]) < abs(e_lats[0]) * 0.3, (
            f"e_lat did not converge: {e_lats[0]:.4f} → {e_lats[-1]:.4f}"
        )


# ---------------------------------------------------------------------------
# Test: Smith predictor bypassed
# ---------------------------------------------------------------------------

class TestSmithPredictorBypassed:
    """When delay_comp ON, raw e_lat should be passed to QP (not Smith-predicted).
    This is tested at the pid_controller level — here we verify at the solver
    level that committed_steering_norm actually causes forward simulation."""

    def test_committed_none_no_forward_sim(self):
        """Even with delay_comp enabled, if committed is None, no forward sim."""
        p = _dc_params(delay_compensation_enabled=True)
        r = _solve(p, committed=None)
        assert r['delay_comp_active'] is False

    def test_committed_empty_no_forward_sim(self):
        """Empty list → no forward sim."""
        p = _dc_params(delay_compensation_enabled=True)
        r = _solve(p, committed=[])
        assert r['delay_comp_active'] is False

    def test_low_speed_no_forward_sim(self):
        """At v < 3.0 m/s, delay comp is skipped even with committed commands.
        The forward sim uses max(v, 3.0) internally, so at real speeds below 3.0
        the model-plant mismatch causes infeasibility."""
        p = _dc_params(delay_compensation_enabled=True)
        r = _solve(p, committed=[0.0, 0.0], v=1.5)
        assert r['delay_comp_active'] is False
