"""Tests for actuator-augmented MPC (4-state kinematic model).

Validates that the MPC correctly models steering actuator delay and
produces earlier/stronger commands to compensate for the lag.
"""

import math

import numpy as np
import pytest

from control.mpc_controller import MPCParams, MPCSolver


def _make_params(**overrides) -> MPCParams:
    defaults = dict(
        actuator_model_enabled=True,
        actuator_tau_s=0.70,
        actuator_rate_limit_deg_per_s=90.0,
        q_actuator=0.0,
        horizon=20,
        dt=0.1,
        q_lat=10.0,
        q_heading=5.0,
        q_speed=1.0,
        r_steer=1e-4,
        r_accel=0.05,
        r_steer_rate=1.0,
        delta_rate_max=0.5,
        first_step_rate_enabled=True,
        ff_alignment_enabled=False,
    )
    defaults.update(overrides)
    return MPCParams(**defaults)


def _solve(solver, **kwargs):
    defaults = dict(
        e_lat=0.0, e_heading=0.0, v=10.0,
        last_delta_norm=0.0,
        kappa_ref_horizon=np.zeros(solver._N),
        v_target=12.0, v_max=15.0, dt=0.1,
        delta_actual_norm=0.0,
    )
    defaults.update(kwargs)
    return solver.solve(**defaults)


# ─────────────────── Construction ───────────────────


class TestActuatorConstruction:
    def test_nx_is_4(self):
        p = _make_params()
        s = MPCSolver(p)
        assert s._nx == 4

    def test_nz_correct(self):
        p = _make_params(horizon=20)
        s = MPCSolver(p)
        # nz = (nx + nu) * N + nx = (4+2)*20 + 4 = 124
        assert s._nz == 124

    def test_disabled_nx_is_3(self):
        p = _make_params(actuator_model_enabled=False)
        s = MPCSolver(p)
        assert s._nx == 3

    def test_dynamic_overrides_actuator(self):
        """Dynamic model takes precedence — actuator flag is ignored."""
        p = _make_params(actuator_model_enabled=True, dynamic_model_enabled=True)
        s = MPCSolver(p)
        assert s._nx == 5  # dynamic model, not actuator


# ─────────────────── Feasibility ───────────────────


class TestActuatorFeasibility:
    def test_straight_road_feasible(self):
        s = MPCSolver(_make_params())
        r = _solve(s)
        assert r['feasible']

    def test_curve_feasible(self):
        s = MPCSolver(_make_params())
        kappa = np.full(20, 0.01)  # R100 curve
        r = _solve(s, kappa_ref_horizon=kappa)
        assert r['feasible']

    def test_large_error_feasible(self):
        s = MPCSolver(_make_params())
        r = _solve(s, e_lat=1.0, e_heading=0.1)
        assert r['feasible']

    def test_solve_time_under_budget(self):
        s = MPCSolver(_make_params())
        # Cold first solve includes OSQP setup; warm solves are faster
        _solve(s)  # warm up
        r = _solve(s)
        assert r['solve_time_ms'] < 8.0


# ─────────────────── Actuator Dynamics ───────────────────


class TestActuatorDynamics:
    def test_delta_actual_init_from_feedback(self):
        """δ_actual initial condition should match Unity feedback."""
        s = MPCSolver(_make_params())
        r = _solve(s, delta_actual_norm=0.05)
        assert abs(r['actuator_delta_actual_init'] - 0.05) < 1e-6

    def test_delta_actual_init_fallback_to_last_cmd(self):
        """When no feedback, δ_actual should use last_delta_norm."""
        s = MPCSolver(_make_params())
        r = _solve(s, delta_actual_norm=None, last_delta_norm=0.08)
        assert abs(r['actuator_delta_actual_init'] - 0.08) < 1e-6

    def test_actuator_alpha(self):
        """Actuator alpha should be dt/τ."""
        p = _make_params(actuator_tau_s=0.70)
        s = MPCSolver(p)
        r = _solve(s)
        expected_alpha = 0.1 / 0.70  # dt/τ
        assert abs(r['actuator_alpha'] - expected_alpha) < 1e-6

    def test_delta_actual_converges_to_command(self):
        """Over the horizon, δ_actual should approach δ_cmd."""
        s = MPCSolver(_make_params(horizon=40, actuator_tau_s=0.50))
        r = _solve(s, e_lat=0.3)
        pred = r['predicted_trajectory']
        # δ_actual at terminal should be closer to the steady-state
        # than the initial value
        assert abs(pred[-1, 3]) > abs(pred[0, 3])

    def test_higher_tau_slower_response(self):
        """Higher τ should produce larger actuator alpha (faster per-step response)."""
        p_fast = _make_params(actuator_tau_s=0.3)
        p_slow = _make_params(actuator_tau_s=1.0)
        s_fast = MPCSolver(p_fast)
        s_slow = MPCSolver(p_slow)

        r_fast = _solve(s_fast, e_lat=0.3)
        r_slow = _solve(s_slow, e_lat=0.3)

        # Faster actuator (smaller τ) → larger α → more response per step
        assert r_fast['actuator_alpha'] > r_slow['actuator_alpha']
        # α_fast = 0.1/0.3 ≈ 0.333, α_slow = 0.1/1.0 = 0.10
        assert abs(r_fast['actuator_alpha'] - 1.0/3.0) < 0.01
        assert abs(r_slow['actuator_alpha'] - 0.10) < 0.01


# ─────────────────── Compensation Behavior ───────────────────


class TestActuatorCompensation:
    def test_commands_stronger_than_baseline(self):
        """With actuator lag, MPC should command MORE aggressively than without."""
        s_act = MPCSolver(_make_params(actuator_model_enabled=True))
        s_base = MPCSolver(_make_params(actuator_model_enabled=False))

        r_act = _solve(s_act, e_lat=0.3, e_heading=0.02)
        r_base = _solve(s_base, e_lat=0.3, e_heading=0.02)

        # Actuator-aware MPC should command equal or stronger (in magnitude)
        # because it knows the actuator will attenuate the command
        assert abs(r_act['steering_normalized']) >= abs(r_base['steering_normalized']) - 0.01

    def test_lag_awareness_with_curve(self):
        """On curve entry, actuator MPC should pre-steer."""
        s = MPCSolver(_make_params())
        # Curve starts at step 5
        kappa = np.zeros(20)
        kappa[5:] = 0.01  # R100 curve approaching

        r = _solve(s, e_lat=0.0, e_heading=0.0)
        r_curve = _solve(s, e_lat=0.0, e_heading=0.0, kappa_ref_horizon=kappa)

        # With upcoming curve, MPC should command non-zero steering even
        # though current error is zero (pre-steering for the delay)
        assert abs(r_curve['steering_normalized']) > abs(r['steering_normalized'])

    def test_actuator_mismatch_correction(self):
        """When actual steering lags command, MPC should compensate."""
        s = MPCSolver(_make_params())

        # Case 1: actuator in sync with command
        r_sync = _solve(s, e_lat=0.3, last_delta_norm=0.05, delta_actual_norm=0.05)

        # Case 2: actuator lagging (actual < command)
        r_lag = _solve(s, e_lat=0.3, last_delta_norm=0.05, delta_actual_norm=0.01)

        # With more lag, MPC should command stronger to compensate
        assert abs(r_lag['steering_normalized']) >= abs(r_sync['steering_normalized']) - 0.01


# ─────────────────── No Regression When Disabled ───────────────────


class TestNoRegressionDisabled:
    def test_disabled_same_as_kinematic(self):
        """With actuator disabled, results should match 3-state kinematic."""
        p_kin = MPCParams(
            actuator_model_enabled=False, horizon=20, dt=0.1,
            q_lat=10.0, q_heading=5.0, q_speed=1.0,
            r_steer=1e-4, r_accel=0.05, r_steer_rate=1.0,
            delta_rate_max=0.5, first_step_rate_enabled=True,
            ff_alignment_enabled=False,
        )
        s_kin = MPCSolver(p_kin)
        r_kin = s_kin.solve(
            e_lat=0.3, e_heading=0.02, v=10.0,
            last_delta_norm=0.0,
            kappa_ref_horizon=np.zeros(20),
            v_target=12.0, v_max=15.0, dt=0.1,
        )
        assert r_kin['feasible']
        assert s_kin._nx == 3
        assert 'actuator_delta_actual_init' not in r_kin

    def test_delta_actual_norm_ignored_when_disabled(self):
        """Passing delta_actual_norm when disabled should be harmless."""
        p = MPCParams(actuator_model_enabled=False, horizon=20, dt=0.1)
        s = MPCSolver(p)
        r = s.solve(
            e_lat=0.3, e_heading=0.02, v=10.0,
            last_delta_norm=0.0,
            kappa_ref_horizon=np.zeros(20),
            v_target=12.0, v_max=15.0, dt=0.1,
            delta_actual_norm=0.05,  # should be ignored
        )
        assert r['feasible']


# ─────────────────── Config Loading ───────────────────


class TestConfigLoading:
    def test_from_config_loads_actuator_params(self):
        cfg = {
            'trajectory': {
                'mpc': {
                    'mpc_actuator_model_enabled': True,
                    'mpc_actuator_tau_s': 0.50,
                    'mpc_actuator_rate_limit_deg_per_s': 60.0,
                    'mpc_q_actuator': 0.1,
                }
            }
        }
        p = MPCParams.from_config(cfg)
        assert p.actuator_model_enabled is True
        assert p.actuator_tau_s == 0.5
        assert p.actuator_rate_limit_deg_per_s == 60.0
        assert p.q_actuator == 0.1

    def test_from_config_defaults_disabled(self):
        cfg = {'trajectory': {'mpc': {}}}
        p = MPCParams.from_config(cfg)
        assert p.actuator_model_enabled is False


# ─────────────────── Gain-consistent actuator state (2026-10-11) ───────────────────

class TestActuatorGainRatio:
    """The real actuator settles at gain·cmd·max_steer (identified 0.65–0.85). With the legacy 1.0 the
    lag model doubles the curve under-steer offset; with the gain in the state and in the feedforward
    alignment the QP commands δ_ff/g in a steady curve."""

    def test_default_is_legacy_and_qp_identical(self):
        p1 = _make_params(actuator_tau_s=0.25)
        p2 = _make_params(actuator_tau_s=0.25, actuator_gain_ratio=1.0)
        r1 = _solve(MPCSolver(p1), e_lat=0.3, kappa_ref_horizon=np.full(20, 0.005))
        r2 = _solve(MPCSolver(p2), e_lat=0.3, kappa_ref_horizon=np.full(20, 0.005))
        assert r1["steering_normalized"] == pytest.approx(r2["steering_normalized"], abs=1e-9)

    def test_actuator_state_settles_at_gain_times_command(self):
        """Predicted δ_actual at the end of the horizon tracks g·δ_cmd, not δ_cmd."""
        for g in (1.0, 0.78):
            p = _make_params(actuator_tau_s=0.25, actuator_gain_ratio=g, q_lat=0.0, q_heading=0.0, r_steer=1e-4)
            s = MPCSolver(p)
            r = _solve(s, kappa_ref_horizon=np.full(20, 0.004), delta_actual_norm=0.0)
            traj = np.asarray(r["predicted_trajectory"])
            d_act_end = traj[-1, 3]
            # the steering sequence settles; compare the last state with g times the last command
            u_last = float(r.get("steering_sequence", [r["steering_normalized"]])[-1]) if r.get("steering_sequence") is not None else r["steering_normalized"]
            assert abs(d_act_end) > 1e-4
            # ratio of settled state to the command is g within the lag horizon tolerance
            assert d_act_end / u_last == pytest.approx(g, rel=0.25)

    def test_gain_consistent_model_commands_more_steer_in_a_curve(self):
        """Same curve, same state: with g=0.78 the QP asks for ≈1/0.78 of the g=1.0 command."""
        base = MPCSolver(_make_params(actuator_tau_s=0.25, actuator_gain_ratio=1.0))
        gc = MPCSolver(_make_params(actuator_tau_s=0.25, actuator_gain_ratio=0.78))
        kw = dict(e_lat=0.0, e_heading=0.0, v=11.6, kappa_ref_horizon=np.full(20, 0.002), delta_actual_norm=None)
        u1 = _solve(base, **kw)["steering_normalized"]; u2 = _solve(gc, **kw)["steering_normalized"]
        assert abs(u1) > 1e-3
        assert u2 / u1 == pytest.approx(1 / 0.78, rel=0.2)

    def test_from_config_loads_gain_ratio(self):
        p = MPCParams.from_config({"trajectory": {"mpc": {"mpc_actuator_gain_ratio": 0.78}}})
        assert p.actuator_gain_ratio == 0.78

    def test_gain_applies_in_the_feedforward_decomposition_path_too(self):
        """Production runs ff_decomposition (base config true): the curve command must still be δ_ff/g."""
        kw = dict(e_lat=0.0, e_heading=0.0, v=11.6, kappa_ref_horizon=np.full(20, 0.002), delta_actual_norm=None)
        base = MPCSolver(_make_params(actuator_tau_s=0.25, actuator_gain_ratio=1.0, ff_decomposition_enabled=True, ff_alignment_enabled=True))
        gc = MPCSolver(_make_params(actuator_tau_s=0.25, actuator_gain_ratio=0.78, ff_decomposition_enabled=True, ff_alignment_enabled=True))
        u1 = _solve(base, **kw)["steering_normalized"]; u2 = _solve(gc, **kw)["steering_normalized"]
        assert abs(u1) > 1e-3
        assert u2 / u1 == pytest.approx(1 / 0.78, rel=0.2)

    def test_actuator_q_lat_only_applies_with_the_model(self):
        p_off = _make_params(actuator_model_enabled=False, q_lat=1.0, actuator_q_lat=4.0)
        p_on = _make_params(actuator_model_enabled=True, q_lat=1.0, actuator_q_lat=4.0)
        assert MPCSolver(p_off).p.q_lat == 1.0
        assert MPCSolver(p_on).p.q_lat == 4.0
        assert MPCSolver(_make_params(actuator_model_enabled=True, q_lat=1.0)).p.q_lat == 1.0

    def test_actuator_q_lat_restores_correction_strength(self):
        """At a lateral error the augmented model with its own weight corrects at least as hard as the
        kinematic production model, where with the kinematic weight it corrected ~half as hard."""
        kin = MPCSolver(_make_params(actuator_model_enabled=False, q_lat=1.0, r_steer=0.05, r_steer_rate=2.0, ff_decomposition_enabled=True))
        weak = MPCSolver(_make_params(actuator_model_enabled=True, actuator_tau_s=0.25, q_lat=1.0, r_steer=0.05, r_steer_rate=2.0, ff_decomposition_enabled=True))
        strong = MPCSolver(_make_params(actuator_model_enabled=True, actuator_tau_s=0.25, q_lat=1.0, actuator_q_lat=4.0, r_steer=0.05, r_steer_rate=2.0, ff_decomposition_enabled=True))
        # recorded arc state, highway_65 R500 at 11.6 m/s (recording_20261011_130336 frame 1211):
        # e_lat +0.28 m, e_heading -0.008 rad, last command -0.022, actual angle +0.55 deg (= -0.034 normalised)
        kw = dict(e_lat=0.28, e_heading=-0.008, v=11.6, last_delta_norm=-0.022, kappa_ref_horizon=np.full(20, 0.002), delta_actual_norm=-0.034)
        u_k = _solve(kin, **kw)["steering_normalized"]; u_w = _solve(weak, **kw)["steering_normalized"]; u_s = _solve(strong, **kw)["steering_normalized"]
        assert u_k < -0.03                                   # kinematic corrects the +0.28 m error
        assert abs(u_w) < 0.5 * abs(u_k)                     # augmented model with the kinematic weight: under half the correction
        assert np.sign(u_s) == np.sign(u_k) and abs(u_s) >= 0.8 * abs(u_k)   # with its own weight: restored
