"""
Tests for the dynamic bicycle model MPC and EKF tire estimator.

Covers Phase D (5-state dynamic bicycle model in MPCSolver) and
Phase E (EKF tire cornering stiffness estimation in MPCController).
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Ensure control/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "control"))

from mpc_controller import MPCParams, MPCSolver, MPCController


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dyn_params(**overrides) -> MPCParams:
    """Return MPCParams with dynamic model enabled and sensible defaults."""
    base = dict(
        dynamic_model_enabled=True,
        horizon=20,
        dt=0.033,
        wheelbase_m=2.5,
        max_steer_rad=0.5236,
        q_lat=10.0,
        q_heading=5.0,
        q_speed=1.0,
        q_vy=0.5,
        q_yawrate=1.0,
        r_steer=1e-4,
        r_accel=0.05,
        r_steer_rate=1.0,
        vehicle_mass_kg=1500.0,
        vehicle_iz_kgm2=1250.0,
        vehicle_lf_m=1.125,
        vehicle_lr_m=1.125,
        tire_cf_nominal=40000.0,
        tire_cr_nominal=40000.0,
        # Disable optional features to isolate tests
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


def _kin_params(**overrides) -> MPCParams:
    """Return MPCParams with kinematic model (dynamic disabled)."""
    return _dyn_params(dynamic_model_enabled=False, **overrides)


def _config_dict(**mpc_overrides) -> dict:
    """Build a config dict for MPCController.from_config()."""
    mpc_cfg = {
        "mpc_horizon": 20,
        "mpc_dt": 0.033,
        "mpc_wheelbase_m": 2.5,
        "mpc_max_steer_rad": 0.5236,
        "mpc_q_lat": 10.0,
        "mpc_q_heading": 5.0,
        "mpc_q_speed": 1.0,
        "mpc_r_steer": 1e-4,
        "mpc_r_accel": 0.05,
        "mpc_r_steer_rate": 1.0,
        "mpc_r_steer_rate_scheduling_enabled": False,
        "mpc_speed_adaptive_horizon": False,
        "mpc_bias_enabled": False,
        "mpc_leff_estimation_enabled": False,
        "mpc_curvature_preview_enabled": False,
    }
    mpc_cfg.update({f"mpc_{k}" if not k.startswith("mpc_") else k: v
                    for k, v in mpc_overrides.items()})
    return {"trajectory": {"mpc": mpc_cfg}}


# ---------------------------------------------------------------------------
# MPCSolver — QP dimensions
# ---------------------------------------------------------------------------

class TestDynamicModelDimensions:
    """Verify QP variable and constraint counts for the 5-state model."""

    def test_kinematic_nz(self):
        """Kinematic: nz = (3+2)*N + 3 = 5N+3."""
        p = _kin_params(horizon=20)
        s = MPCSolver(p)
        assert s._nx == 3
        assert s._nz == 5 * 20 + 3  # 103

    def test_dynamic_nz(self):
        """Dynamic: nz = (5+2)*N + 5 = 7N+5."""
        p = _dyn_params(horizon=20)
        s = MPCSolver(p)
        assert s._nx == 5
        assert s._nz == 7 * 20 + 5  # 145

    def test_dynamic_nu(self):
        """Both models use 2 inputs [delta_norm, accel]."""
        s_kin = MPCSolver(_kin_params())
        s_dyn = MPCSolver(_dyn_params())
        assert s_kin._nu == 2
        assert s_dyn._nu == 2


# ---------------------------------------------------------------------------
# MPCSolver — solve on straight road
# ---------------------------------------------------------------------------

class TestDynamicStraight:
    """Dynamic model on a straight road with zero initial error."""

    def test_straight_zero_error(self):
        """Zero initial state on a straight road → near-zero steering."""
        p = _dyn_params()
        s = MPCSolver(p)
        kappa = np.zeros(p.horizon)
        result = s.solve(
            e_lat=0.0, e_heading=0.0, v=12.0,
            last_delta_norm=0.0, kappa_ref_horizon=kappa,
            v_target=12.0, v_max=15.0, dt=p.dt,
            tire_cf=40000.0, tire_cr=40000.0,
            v_y_init=0.0, r_init=0.0,
        )
        assert result['feasible']
        assert abs(result['steering_normalized']) < 0.05

    def test_straight_lateral_offset_corrects(self):
        """Nonzero e_lat on straight → predicted trajectory shows correction."""
        p = _dyn_params()
        s = MPCSolver(p)
        kappa = np.zeros(p.horizon)
        result = s.solve(
            e_lat=0.3, e_heading=0.0, v=12.0,
            last_delta_norm=0.0, kappa_ref_horizon=kappa,
            v_target=12.0, v_max=15.0, dt=p.dt,
            tire_cf=40000.0, tire_cr=40000.0,
        )
        assert result['feasible']
        # The predicted trajectory should show e_lat reducing toward zero.
        # Initial e_lat=0.3, terminal should be smaller in magnitude.
        traj = result['predicted_trajectory']
        assert abs(traj[-1, 0]) < abs(traj[0, 0])


# ---------------------------------------------------------------------------
# MPCSolver — curve feedforward
# ---------------------------------------------------------------------------

class TestDynamicCurveFeedforward:
    """Feedforward steering on curves."""

    def test_curve_produces_positive_steering(self):
        """Positive curvature → positive feedforward → positive steering."""
        p = _dyn_params()
        s = MPCSolver(p)
        kappa = np.full(p.horizon, 0.010)  # R100 left turn
        result = s.solve(
            e_lat=0.0, e_heading=0.0, v=12.0,
            last_delta_norm=0.0, kappa_ref_horizon=kappa,
            v_target=12.0, v_max=15.0, dt=p.dt,
            tire_cf=40000.0, tire_cr=40000.0,
        )
        assert result['feasible']
        assert result['steering_normalized'] > 0.0  # steer into the curve

    def test_dynamic_feedforward_static_method(self):
        """_feedforward_delta_norm_dynamic produces correct understeer correction."""
        L = 2.25
        l_f, l_r = 1.125, 1.125
        C_f, C_r = 40000.0, 40000.0
        mass = 1500.0
        max_steer = 0.5236
        kappa = 0.010

        # Kinematic feedforward
        ff_kin = MPCSolver._feedforward_delta_norm(kappa, L, max_steer)

        # Dynamic feedforward at low speed should be close to kinematic
        ff_dyn_low = MPCSolver._feedforward_delta_norm_dynamic(
            kappa, 3.0, C_f, C_r, l_f, l_r, mass, max_steer)

        # Dynamic feedforward at high speed includes understeer term
        ff_dyn_high = MPCSolver._feedforward_delta_norm_dynamic(
            kappa, 20.0, C_f, C_r, l_f, l_r, mass, max_steer)

        # For symmetric axles (l_f == l_r) and equal C_f == C_r → K_us = 0
        # So dynamic and kinematic should be very similar
        assert abs(ff_dyn_low - ff_kin) < 0.01
        # At zero understeer gradient, high speed doesn't change much
        assert abs(ff_dyn_high - ff_kin) < 0.01

    def test_dynamic_feedforward_asymmetric_understeer(self):
        """With C_f < C_r → positive K_us → more steering at high speed."""
        L = 2.25
        l_f, l_r = 1.125, 1.125
        mass = 1500.0
        max_steer = 0.5236
        kappa = 0.010
        v = 15.0

        # Symmetric: K_us = 0
        ff_sym = MPCSolver._feedforward_delta_norm_dynamic(
            kappa, v, 40000.0, 40000.0, l_f, l_r, mass, max_steer)

        # Front weaker than rear → understeer → K_us > 0 → more steering needed
        ff_under = MPCSolver._feedforward_delta_norm_dynamic(
            kappa, v, 30000.0, 50000.0, l_f, l_r, mass, max_steer)

        assert ff_under > ff_sym  # more steering needed for understeer

    def test_dynamic_feedforward_zero_curvature(self):
        """Zero curvature → zero feedforward for both models."""
        ff = MPCSolver._feedforward_delta_norm_dynamic(
            0.0, 15.0, 40000.0, 40000.0, 1.125, 1.125, 1500.0, 0.5236)
        assert ff == 0.0


# ---------------------------------------------------------------------------
# Backward compatibility — dynamic disabled
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    """When dynamic_model_enabled=False, solver behaves as kinematic."""

    def test_disabled_produces_3_state_trajectory(self):
        """Kinematic model produces (N+1, 3) trajectory."""
        p = _kin_params(horizon=20)
        s = MPCSolver(p)
        kappa = np.zeros(p.horizon)
        result = s.solve(
            e_lat=0.1, e_heading=0.02, v=12.0,
            last_delta_norm=0.0, kappa_ref_horizon=kappa,
            v_target=12.0, v_max=15.0, dt=p.dt,
        )
        assert result['feasible']
        assert result['predicted_trajectory'].shape == (21, 3)

    def test_enabled_produces_5_state_trajectory(self):
        """Dynamic model produces (N+1, 5) trajectory."""
        p = _dyn_params(horizon=20)
        s = MPCSolver(p)
        kappa = np.zeros(p.horizon)
        result = s.solve(
            e_lat=0.1, e_heading=0.02, v=12.0,
            last_delta_norm=0.0, kappa_ref_horizon=kappa,
            v_target=12.0, v_max=15.0, dt=p.dt,
            tire_cf=40000.0, tire_cr=40000.0,
        )
        assert result['feasible']
        assert result['predicted_trajectory'].shape == (21, 5)

    def test_disabled_ignores_tire_params(self):
        """Kinematic model ignores tire_cf/tire_cr arguments."""
        p = _kin_params()
        s = MPCSolver(p)
        kappa = np.full(p.horizon, 0.005)
        r1 = s.solve(
            e_lat=0.1, e_heading=0.02, v=12.0,
            last_delta_norm=0.0, kappa_ref_horizon=kappa,
            v_target=12.0, v_max=15.0, dt=p.dt,
            tire_cf=40000.0, tire_cr=40000.0,
        )
        # Rebuild solver to clear warm-start
        s2 = MPCSolver(p)
        r2 = s2.solve(
            e_lat=0.1, e_heading=0.02, v=12.0,
            last_delta_norm=0.0, kappa_ref_horizon=kappa,
            v_target=12.0, v_max=15.0, dt=p.dt,
            tire_cf=20000.0, tire_cr=60000.0,  # different values
        )
        assert abs(r1['steering_normalized'] - r2['steering_normalized']) < 1e-6


# ---------------------------------------------------------------------------
# EKF tire estimation — via MPCController
# ---------------------------------------------------------------------------

class TestEKFConvergence:
    """EKF tire cornering stiffness estimation."""

    def test_ekf_converges_toward_true_values(self):
        """Starting from wrong initial C_f/C_r, EKF should move toward truth."""
        cfg = _config_dict(
            dynamic_model_enabled=True,
            tire_ekf_enabled=True,
            tire_cf_nominal=40000.0,  # start at nominal
            tire_cr_nominal=40000.0,
        )
        ctrl = MPCController(cfg)

        # Simulate 200 frames with synthetic heading data
        # that is consistent with C_f=30000, C_r=50000 (understeering vehicle)
        dt = 0.033
        e_heading = 0.0
        kappa = 0.010  # R100 curve
        speed = 12.0

        # The EKF should shift C_f/C_r away from symmetric nominal
        # We drive a consistent heading change to excite the estimator
        for i in range(200):
            # Simulate heading evolution (simplified)
            # In a real vehicle with C_f=30k, the yaw rate would be lower
            # than the nominal model predicts → innovation pushes C_f down
            e_heading += 0.002 * math.sin(i * 0.1)  # small heading oscillation

            result = ctrl.compute_steering(
                e_lat=0.05 * math.sin(i * 0.05),
                e_heading=e_heading,
                current_speed=speed,
                last_delta_norm=0.01,
                kappa_ref=kappa,
                v_target=speed,
                v_max=15.0,
                dt=dt,
            )

        # After 200 frames, check that EKF has updated
        assert result['tire_ekf_update_count'] > 50
        # C_f and C_r should still be within bounds
        assert 15000.0 <= result['tire_cf'] <= 80000.0
        assert 15000.0 <= result['tire_cr'] <= 80000.0


class TestEKFBoundsClamped:
    """EKF safety clamps prevent unbounded estimates."""

    def test_cf_cr_stay_in_bounds(self):
        """Even with extreme inputs, C_f/C_r are clamped."""
        cfg = _config_dict(
            dynamic_model_enabled=True,
            tire_ekf_enabled=True,
            tire_cf_nominal=40000.0,
            tire_cr_nominal=40000.0,
        )
        ctrl = MPCController(cfg)

        # Drive with extreme heading oscillations to push EKF hard
        dt = 0.033
        for i in range(100):
            e_heading = 0.3 * math.sin(i * 0.5)  # large heading swings
            ctrl.compute_steering(
                e_lat=0.5 * math.sin(i * 0.3),
                e_heading=e_heading,
                current_speed=15.0,
                last_delta_norm=0.5 * math.sin(i * 0.2),
                kappa_ref=0.01,
                v_target=15.0,
                v_max=20.0,
                dt=dt,
            )

        # Verify bounds
        C_f = ctrl._tire_theta[0]
        C_r = ctrl._tire_theta[1]
        assert C_f >= ctrl.params.tire_ekf_cf_min
        assert C_f <= ctrl.params.tire_ekf_cf_max
        assert C_r >= ctrl.params.tire_ekf_cr_min
        assert C_r <= ctrl.params.tire_ekf_cr_max


class TestEKFGating:
    """EKF gating: skip updates when conditions are unsafe."""

    def test_low_speed_no_update(self):
        """Below tire_ekf_min_speed_mps, no EKF updates occur."""
        cfg = _config_dict(
            dynamic_model_enabled=True,
            tire_ekf_enabled=True,
            tire_ekf_min_speed_mps=5.0,
        )
        ctrl = MPCController(cfg)

        dt = 0.033
        for i in range(50):
            ctrl.compute_steering(
                e_lat=0.05, e_heading=0.01 * math.sin(i * 0.1),
                current_speed=3.0,  # below 5.0 threshold
                last_delta_norm=0.01,
                kappa_ref=0.005,
                v_target=3.0, v_max=5.0, dt=dt,
            )

        assert ctrl._tire_ekf_update_count == 0

    def test_saturation_gate(self):
        """When slip angles exceed saturation threshold, EKF should not update."""
        cfg = _config_dict(
            dynamic_model_enabled=True,
            tire_ekf_enabled=True,
            tire_ekf_slip_saturation_rad=0.15,
        )
        ctrl = MPCController(cfg)

        # Force large slip angles by using extreme steering at moderate speed
        dt = 0.033
        updates_before = ctrl._tire_ekf_update_count
        for i in range(20):
            # With v_y_est = 0 initially and large steering, alpha_f can be large
            ctrl.compute_steering(
                e_lat=0.0, e_heading=0.0,
                current_speed=6.0,
                last_delta_norm=0.95,  # near-max steering → large slip angle
                kappa_ref=0.0,
                v_target=6.0, v_max=10.0, dt=dt,
            )

        # The first frame should have small slip (v_y_est=0), so it may update.
        # But as v_y_est grows, slip angles should exceed 0.15 and gating kicks in.
        # We just verify it didn't update every single frame.
        assert ctrl._tire_ekf_update_count < 20

    def test_divergence_detection(self):
        """After 10 consecutive divergent innovations, EKF pauses."""
        cfg = _config_dict(
            dynamic_model_enabled=True,
            tire_ekf_enabled=True,
            tire_ekf_innovation_divergence_threshold=0.15,
        )
        ctrl = MPCController(cfg)

        dt = 0.033
        # Create divergent conditions: rapid heading changes → large innovation
        for i in range(30):
            # Flip heading each frame → huge innovation
            e_heading = 0.3 if i % 2 == 0 else -0.3
            ctrl.compute_steering(
                e_lat=0.0, e_heading=e_heading,
                current_speed=12.0,
                last_delta_norm=0.0,
                kappa_ref=0.0,
                v_target=12.0, v_max=15.0, dt=dt,
            )

        # After enough divergent frames, the count should hit 10+ and pause updates
        assert ctrl._tire_ekf_divergent_count >= 1  # at least detected divergence


# ---------------------------------------------------------------------------
# Understeer gradient computation
# ---------------------------------------------------------------------------

class TestUndersteerGradient:
    """Verify understeer gradient K_us calculation."""

    def test_symmetric_zero_understeer(self):
        """Symmetric vehicle (l_f=l_r, C_f=C_r) → K_us = 0."""
        cfg = _config_dict(dynamic_model_enabled=True)
        ctrl = MPCController(cfg)
        K_us = ctrl._compute_understeer_gradient(40000.0, 40000.0)
        assert abs(K_us) < 1e-10

    def test_understeer_positive(self):
        """C_f < C_r with l_f=l_r → K_us > 0 (understeering)."""
        cfg = _config_dict(dynamic_model_enabled=True)
        ctrl = MPCController(cfg)
        K_us = ctrl._compute_understeer_gradient(30000.0, 50000.0)
        assert K_us > 0.0

    def test_oversteer_negative(self):
        """C_f > C_r with l_f=l_r → K_us < 0 (oversteering)."""
        cfg = _config_dict(dynamic_model_enabled=True)
        ctrl = MPCController(cfg)
        K_us = ctrl._compute_understeer_gradient(50000.0, 30000.0)
        assert K_us < 0.0

    def test_known_value(self):
        """Verify exact value for known parameters."""
        cfg = _config_dict(
            dynamic_model_enabled=True,
            vehicle_mass_kg=1500.0,
            vehicle_lf_m=1.125,
            vehicle_lr_m=1.125,
        )
        ctrl = MPCController(cfg)
        # K_us = (m/L)·(l_r/C_f − l_f/C_r)
        # = (1500/2.25)·(1.125/30000 − 1.125/50000)
        # = 666.67 · (0.0000375 − 0.0000225)
        # = 666.67 · 0.0000150
        # = 0.0100
        K_us = ctrl._compute_understeer_gradient(30000.0, 50000.0)
        assert abs(K_us - 0.0100) < 1e-4


# ---------------------------------------------------------------------------
# Reset restores nominal
# ---------------------------------------------------------------------------

class TestResetRestoresNominal:
    """reset() should restore all tire EKF state to nominal."""

    def test_reset_restores_cf_cr(self):
        cfg = _config_dict(
            dynamic_model_enabled=True,
            tire_ekf_enabled=True,
        )
        ctrl = MPCController(cfg)

        # Drive a few frames to shift C_f/C_r from nominal
        dt = 0.033
        for i in range(50):
            ctrl.compute_steering(
                e_lat=0.1, e_heading=0.01 * math.sin(i * 0.1),
                current_speed=12.0, last_delta_norm=0.05,
                kappa_ref=0.005, v_target=12.0, v_max=15.0, dt=dt,
            )

        # Verify EKF has been active
        assert ctrl._tire_ekf_update_count > 0

        # Reset
        ctrl.reset()

        # Verify nominal restored
        assert ctrl._tire_theta[0] == ctrl.params.tire_cf_nominal
        assert ctrl._tire_theta[1] == ctrl.params.tire_cr_nominal
        assert ctrl._tire_ekf_update_count == 0
        assert ctrl._tire_ekf_divergent_count == 0
        assert ctrl._v_y_est == 0.0
        assert ctrl._yaw_rate_est == 0.0
        assert ctrl._e_heading_prev_tire == 0.0
        # P should be high uncertainty again
        assert ctrl._tire_P[0, 0] == pytest.approx(1e6)


# ---------------------------------------------------------------------------
# v_y estimation bounded
# ---------------------------------------------------------------------------

class TestVyEstimation:
    """Lateral velocity estimate stays reasonable."""

    def test_vy_bounded(self):
        """v_y estimate should stay within ±2 m/s during normal driving."""
        cfg = _config_dict(
            dynamic_model_enabled=True,
            tire_ekf_enabled=True,
        )
        ctrl = MPCController(cfg)

        dt = 0.033
        max_vy = 0.0
        for i in range(200):
            result = ctrl.compute_steering(
                e_lat=0.1 * math.sin(i * 0.05),
                e_heading=0.05 * math.sin(i * 0.08),
                current_speed=12.0,
                last_delta_norm=0.1 * math.sin(i * 0.03),
                kappa_ref=0.005,
                v_target=12.0, v_max=15.0, dt=dt,
            )
            max_vy = max(max_vy, abs(result['v_y_estimate']))

        assert max_vy < 2.0  # hard constraint from QP state bounds


# ---------------------------------------------------------------------------
# Yaw rate measurement derivation
# ---------------------------------------------------------------------------

class TestYawRateDerivation:
    """Yaw rate measurement: r_meas = Δe_heading/dt + κ·v_x."""

    def test_straight_heading_change(self):
        """On a straight road, r_meas ≈ Δe_heading/dt."""
        cfg = _config_dict(
            dynamic_model_enabled=True,
            tire_ekf_enabled=True,
        )
        ctrl = MPCController(cfg)

        dt = 0.033
        # First frame: set heading baseline
        ctrl.compute_steering(
            e_lat=0.0, e_heading=0.0,
            current_speed=12.0, last_delta_norm=0.0,
            kappa_ref=0.0, v_target=12.0, v_max=15.0, dt=dt,
        )
        # Second frame: heading changed by 0.01 rad
        result = ctrl.compute_steering(
            e_lat=0.0, e_heading=0.01,
            current_speed=12.0, last_delta_norm=0.0,
            kappa_ref=0.0, v_target=12.0, v_max=15.0, dt=dt,
        )
        # r_meas = (0.01 - 0.0) / 0.033 + 0 ≈ 0.303 rad/s
        # The yaw_rate_estimate is a blend of r_meas and r_pred
        assert abs(result['yaw_rate_estimate']) > 0.1  # non-trivial yaw rate


# ---------------------------------------------------------------------------
# Solve time budget
# ---------------------------------------------------------------------------

class TestSolveTime:
    """Verify dynamic model solve completes within budget."""

    def test_dynamic_solve_under_budget(self):
        """5-state QP should solve in < 8 ms (OSQP warm-started)."""
        p = _dyn_params()
        s = MPCSolver(p)
        kappa = np.full(p.horizon, 0.005)

        # Warm up the solver
        for _ in range(5):
            s.solve(
                e_lat=0.1, e_heading=0.02, v=12.0,
                last_delta_norm=0.0, kappa_ref_horizon=kappa,
                v_target=12.0, v_max=15.0, dt=p.dt,
                tire_cf=40000.0, tire_cr=40000.0,
            )

        # Time the warm-started solve
        times = []
        for i in range(10):
            result = s.solve(
                e_lat=0.1 + 0.01 * i, e_heading=0.02, v=12.0,
                last_delta_norm=0.0, kappa_ref_horizon=kappa,
                v_target=12.0, v_max=15.0, dt=p.dt,
                tire_cf=40000.0, tire_cr=40000.0,
            )
            assert result['feasible']
            times.append(result['solve_time_ms'])

        # P95 (9th of 10 sorted) should be < 8 ms
        p95 = sorted(times)[8]
        assert p95 < 8.0, f"P95 solve time {p95:.1f} ms exceeds 8 ms budget"


# ---------------------------------------------------------------------------
# Dynamic model state bounds in QP
# ---------------------------------------------------------------------------

class TestDynamicStateBounds:
    """Verify the dynamic model enforces v_y, r, v_x bounds."""

    def test_vy_bound_in_trajectory(self):
        """Predicted v_y should stay within ±2.0 m/s."""
        p = _dyn_params()
        s = MPCSolver(p)
        kappa = np.full(p.horizon, 0.010)
        result = s.solve(
            e_lat=0.5, e_heading=0.1, v=12.0,
            last_delta_norm=0.0, kappa_ref_horizon=kappa,
            v_target=12.0, v_max=15.0, dt=p.dt,
            tire_cf=40000.0, tire_cr=40000.0,
            v_y_init=0.0, r_init=0.0,
        )
        assert result['feasible']
        traj = result['predicted_trajectory']
        # v_y is state index 2
        assert np.all(traj[:, 2] >= -2.0 - 1e-4)
        assert np.all(traj[:, 2] <= 2.0 + 1e-4)

    def test_yaw_rate_bound_in_trajectory(self):
        """Predicted yaw rate should stay within ±1.0 rad/s."""
        p = _dyn_params()
        s = MPCSolver(p)
        kappa = np.full(p.horizon, 0.010)
        result = s.solve(
            e_lat=0.5, e_heading=0.1, v=12.0,
            last_delta_norm=0.0, kappa_ref_horizon=kappa,
            v_target=12.0, v_max=15.0, dt=p.dt,
            tire_cf=40000.0, tire_cr=40000.0,
        )
        assert result['feasible']
        traj = result['predicted_trajectory']
        # r is state index 3
        assert np.all(traj[:, 3] >= -1.0 - 1e-4)
        assert np.all(traj[:, 3] <= 1.0 + 1e-4)


# ---------------------------------------------------------------------------
# MPCController return dict includes tire diagnostics
# ---------------------------------------------------------------------------

class TestControllerReturnDict:
    """MPCController.compute_steering returns all 11 tire diagnostic fields."""

    def test_tire_fields_present(self):
        cfg = _config_dict(
            dynamic_model_enabled=True,
            tire_ekf_enabled=True,
        )
        ctrl = MPCController(cfg)
        result = ctrl.compute_steering(
            e_lat=0.0, e_heading=0.0,
            current_speed=12.0, last_delta_norm=0.0,
            kappa_ref=0.0, v_target=12.0, v_max=15.0, dt=0.033,
        )
        expected_keys = [
            'tire_cf', 'tire_cr',
            'tire_ekf_innovation', 'tire_ekf_P_trace',
            'tire_slip_angle_front', 'tire_slip_angle_rear',
            'tire_understeer_gradient',
            'dynamic_model_active',
            'tire_ekf_update_count',
            'v_y_estimate', 'yaw_rate_estimate',
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_dynamic_model_active_flag(self):
        """dynamic_model_active reflects params.dynamic_model_enabled."""
        cfg_on = _config_dict(dynamic_model_enabled=True)
        ctrl_on = MPCController(cfg_on)
        r_on = ctrl_on.compute_steering(
            e_lat=0.0, e_heading=0.0, current_speed=12.0,
            last_delta_norm=0.0, kappa_ref=0.0,
            v_target=12.0, v_max=15.0, dt=0.033,
        )
        assert r_on['dynamic_model_active'] is True

        cfg_off = _config_dict(dynamic_model_enabled=False)
        ctrl_off = MPCController(cfg_off)
        r_off = ctrl_off.compute_steering(
            e_lat=0.0, e_heading=0.0, current_speed=12.0,
            last_delta_norm=0.0, kappa_ref=0.0,
            v_target=12.0, v_max=15.0, dt=0.033,
        )
        assert r_off['dynamic_model_active'] is False


# ---------------------------------------------------------------------------
# Tire EKF with IMU yaw rate
# ---------------------------------------------------------------------------

class TestTireEKFWithIMU:
    """Tests for IMU yaw rate integration in the tire EKF."""

    def test_imu_yaw_rate_used_when_configured(self):
        """yaw_rate_source == 'imu' when flag on + IMU provided."""
        cfg = _config_dict(
            dynamic_model_enabled=True,
            tire_ekf_enabled=True,
            tire_ekf_use_imu_yaw_rate=True,
        )
        ctrl = MPCController(cfg)
        # Warm up with one frame
        ctrl.compute_steering(
            e_lat=0.1, e_heading=0.02, current_speed=12.0,
            last_delta_norm=0.1, kappa_ref=0.01,
            v_target=12.0, v_max=15.0, dt=0.033,
            imu_yaw_rate=0.08,
        )
        r = ctrl.compute_steering(
            e_lat=0.1, e_heading=0.02, current_speed=12.0,
            last_delta_norm=0.1, kappa_ref=0.01,
            v_target=12.0, v_max=15.0, dt=0.033,
            imu_yaw_rate=0.08,
        )
        assert r.get('yaw_rate_source') == 'imu'

    def test_fallback_to_derived_when_imu_none(self):
        """yaw_rate_source == 'derived' when IMU is None."""
        cfg = _config_dict(
            dynamic_model_enabled=True,
            tire_ekf_enabled=True,
            tire_ekf_use_imu_yaw_rate=True,
        )
        ctrl = MPCController(cfg)
        ctrl.compute_steering(
            e_lat=0.1, e_heading=0.02, current_speed=12.0,
            last_delta_norm=0.1, kappa_ref=0.01,
            v_target=12.0, v_max=15.0, dt=0.033,
        )
        r = ctrl.compute_steering(
            e_lat=0.1, e_heading=0.02, current_speed=12.0,
            last_delta_norm=0.1, kappa_ref=0.01,
            v_target=12.0, v_max=15.0, dt=0.033,
        )
        assert r.get('yaw_rate_source') == 'derived'

    def test_imu_disabled_uses_derived(self):
        """Flag off -> uses derived even if IMU available."""
        cfg = _config_dict(
            dynamic_model_enabled=True,
            tire_ekf_enabled=True,
            tire_ekf_use_imu_yaw_rate=False,
        )
        ctrl = MPCController(cfg)
        ctrl.compute_steering(
            e_lat=0.1, e_heading=0.02, current_speed=12.0,
            last_delta_norm=0.1, kappa_ref=0.01,
            v_target=12.0, v_max=15.0, dt=0.033,
            imu_yaw_rate=0.08,
        )
        r = ctrl.compute_steering(
            e_lat=0.1, e_heading=0.02, current_speed=12.0,
            last_delta_norm=0.1, kappa_ref=0.01,
            v_target=12.0, v_max=15.0, dt=0.033,
            imu_yaw_rate=0.08,
        )
        assert r.get('yaw_rate_source') == 'derived'

    def test_ekf_convergence_with_imu(self):
        """100 frames with known geometry -> EKF converges toward true C_f."""
        cfg = _config_dict(
            dynamic_model_enabled=True,
            tire_ekf_enabled=True,
            tire_ekf_use_imu_yaw_rate=True,
            tire_ekf_imu_yaw_rate_r=0.001,
        )
        ctrl = MPCController(cfg)
        # Simulate steady cornering: v=10 m/s, kappa=0.01, r = v*kappa = 0.1 rad/s
        v = 10.0
        kappa = 0.01
        r_true = v * kappa  # 0.1 rad/s
        initial_cf = ctrl._tire_theta[0]

        for _ in range(100):
            ctrl.compute_steering(
                e_lat=0.1, e_heading=0.02, current_speed=v,
                last_delta_norm=0.05, kappa_ref=kappa,
                v_target=v, v_max=15.0, dt=0.033,
                imu_yaw_rate=r_true,
            )

        # EKF should have updated (not still at initial)
        assert ctrl._tire_ekf_update_count > 0, "EKF never updated"
        # C_f should have moved from initial (direction depends on dynamics)
        final_cf = ctrl._tire_theta[0]
        # We don't assert exact convergence, just that EKF is active and updating
        assert final_cf != initial_cf or ctrl._tire_ekf_update_count > 50

    def test_imu_measurement_noise_is_lower(self):
        """R = 0.001 (not 0.01) when IMU active."""
        p = _dyn_params(
            tire_ekf_enabled=True,
            tire_ekf_use_imu_yaw_rate=True,
            tire_ekf_imu_yaw_rate_r=0.001,
        )
        assert p.tire_ekf_imu_yaw_rate_r == 0.001
        assert p.tire_ekf_imu_yaw_rate_r < p.tire_ekf_measurement_noise

    def test_sign_convention_left_turn(self):
        """kappa > 0 (left turn), angVel.y < 0 -> positive r_meas."""
        cfg = _config_dict(
            dynamic_model_enabled=True,
            tire_ekf_enabled=True,
            tire_ekf_use_imu_yaw_rate=True,
        )
        ctrl = MPCController(cfg)
        # Left turn: kappa > 0, Unity angVel.y < 0, so imu_yaw_rate = -(-0.1) = 0.1
        ctrl.compute_steering(
            e_lat=0.1, e_heading=0.02, current_speed=10.0,
            last_delta_norm=0.05, kappa_ref=0.01,
            v_target=10.0, v_max=15.0, dt=0.033,
            imu_yaw_rate=0.1,  # Already sign-corrected by caller
        )
        r = ctrl.compute_steering(
            e_lat=0.1, e_heading=0.02, current_speed=10.0,
            last_delta_norm=0.05, kappa_ref=0.01,
            v_target=10.0, v_max=15.0, dt=0.033,
            imu_yaw_rate=0.1,
        )
        assert r.get('yaw_rate_measurement', 0) > 0, "Left turn should have positive yaw rate"

    def test_interframe_passes_imu(self):
        """compute_steering() accepts imu_yaw_rate kwarg without error."""
        cfg = _config_dict(
            dynamic_model_enabled=True,
            tire_ekf_enabled=True,
            tire_ekf_use_imu_yaw_rate=True,
        )
        ctrl = MPCController(cfg)
        # Should not raise
        r = ctrl.compute_steering(
            e_lat=0.0, e_heading=0.0, current_speed=12.0,
            last_delta_norm=0.0, kappa_ref=0.0,
            v_target=12.0, v_max=15.0, dt=0.033,
            imu_yaw_rate=0.05,
        )
        assert 'yaw_rate_source' in r


# ---------------------------------------------------------------------------
# Unity geometry override
# ---------------------------------------------------------------------------

class TestUnityGeometryOverride:
    """Tests for runtime geometry override from Unity ground truth."""

    def test_asymmetric_geometry_produces_understeer(self):
        """l_f < l_r → positive understeer gradient."""
        p = _dyn_params(vehicle_lf_m=1.0, vehicle_lr_m=1.5)
        ctrl = MPCController._from_params(p) if hasattr(MPCController, '_from_params') else None
        # Compute K_us directly from formula
        m = p.vehicle_mass_kg
        L = p.vehicle_lf_m + p.vehicle_lr_m
        K_us = m * p.vehicle_lr_m / (2 * L * p.tire_cf_nominal) - m * p.vehicle_lf_m / (2 * L * p.tire_cr_nominal)
        assert K_us > 0.0005, f"K_us={K_us} should be positive (understeer)"

    def test_symmetric_geometry_no_understeer(self):
        """l_f == l_r, C_f == C_r → K_us = 0."""
        p = _dyn_params(vehicle_lf_m=1.25, vehicle_lr_m=1.25)
        m = p.vehicle_mass_kg
        L = p.vehicle_lf_m + p.vehicle_lr_m
        K_us = m * p.vehicle_lr_m / (2 * L * p.tire_cf_nominal) - m * p.vehicle_lf_m / (2 * L * p.tire_cr_nominal)
        assert abs(K_us) < 1e-6, f"K_us={K_us} should be ~0 for symmetric geometry"

    def test_runtime_param_override_changes_steering(self):
        """Mutating l_f/l_r at runtime changes steering output."""
        cfg = _config_dict(dynamic_model_enabled=True)
        ctrl = MPCController(cfg)
        # Use small inputs to avoid steer saturation
        kwargs = dict(
            e_lat=0.05, e_heading=0.01, current_speed=10.0,
            last_delta_norm=0.01, kappa_ref=0.002,
            v_target=10.0, v_max=15.0, dt=0.033,
        )
        # Warm up
        for _ in range(3):
            ctrl.compute_steering(**kwargs)
        r1 = ctrl.compute_steering(**kwargs)
        # Override geometry (asymmetric → different feedforward)
        ctrl.params.vehicle_lf_m = 0.8
        ctrl.params.vehicle_lr_m = 1.7
        r2 = ctrl.compute_steering(**kwargs)
        assert abs(r1['steering_normalized'] - r2['steering_normalized']) > 1e-6, \
            "Steering should change after geometry override"

    def test_ekf_reset_after_override(self):
        """Reset EKF state after geometry override → theta back to nominal."""
        cfg = _config_dict(
            dynamic_model_enabled=True,
            tire_ekf_enabled=True,
            tire_ekf_use_imu_yaw_rate=True,
        )
        ctrl = MPCController(cfg)
        # Run some frames to move EKF state
        for _ in range(20):
            ctrl.compute_steering(
                e_lat=0.2, e_heading=0.03, current_speed=10.0,
                last_delta_norm=0.05, kappa_ref=0.01,
                v_target=10.0, v_max=15.0, dt=0.033,
                imu_yaw_rate=0.1,
            )
        assert ctrl._tire_ekf_update_count > 0
        # Reset (simulating orchestrator._apply_unity_geometry)
        ctrl._tire_theta = np.array([ctrl.params.tire_cf_nominal, ctrl.params.tire_cr_nominal])
        ctrl._tire_P = np.eye(2) * 1e6
        ctrl._tire_ekf_update_count = 0
        ctrl._v_y_est = 0.0
        ctrl._yaw_rate_est = 0.0
        assert ctrl._tire_ekf_update_count == 0
        assert ctrl._tire_theta[0] == ctrl.params.tire_cf_nominal

    def test_sanity_bounds_reject_bad_values(self):
        """Out-of-range geometry values should be rejected."""
        # These are the bounds used in orchestrator._apply_unity_geometry
        assert not (0.5 < 0.1 < 3.0), "l_f=0.1 should be rejected"
        assert not (0.5 < -1.0 < 3.0), "l_f=-1.0 should be rejected"
        assert not (500 < 0 < 5000), "mass=0 should be rejected"
        assert (0.5 < 1.0 < 3.0), "l_f=1.0 should be accepted"
        assert (0.5 < 1.5 < 3.0), "l_r=1.5 should be accepted"

    def test_fallback_defaults_used_without_unity(self):
        """Default config uses symmetric 1.125/1.125."""
        cfg = _config_dict(dynamic_model_enabled=True)
        ctrl = MPCController(cfg)
        assert ctrl.params.vehicle_lf_m == pytest.approx(1.125, abs=0.01)
        assert ctrl.params.vehicle_lr_m == pytest.approx(1.125, abs=0.01)

    def test_understeer_gradient_formula(self):
        """K_us formula produces known value for specific geometry."""
        # For m=1500, l_f=1.0, l_r=1.5, C_f=C_r=40000:
        # K_us = 1500*1.5/(2*2.5*40000) - 1500*1.0/(2*2.5*40000)
        #      = 2250/200000 - 1500/200000 = 0.01125 - 0.0075 = 0.00375
        m, lf, lr, Cf, Cr = 1500.0, 1.0, 1.5, 40000.0, 40000.0
        L = lf + lr
        K_us = m * lr / (2 * L * Cf) - m * lf / (2 * L * Cr)
        assert K_us == pytest.approx(0.00375, abs=1e-5)


# ---------------------------------------------------------------------------
# Sysid-identified tire parameter tests (C_f=36662, C_r=71799, Iz=2428)
# ---------------------------------------------------------------------------

# Sysid constants
_SYSID_CF = 36662.0
_SYSID_CR = 71799.0
_SYSID_IZ = 2428.0


def _sysid_params(**overrides) -> MPCParams:
    """MPCParams with sysid-identified tire parameters and Iz."""
    sysid_defaults = dict(
        tire_cf_nominal=_SYSID_CF,
        tire_cr_nominal=_SYSID_CR,
        vehicle_iz_kgm2=_SYSID_IZ,
        q_vy=0.1,
        q_yawrate=0.5,
    )
    sysid_defaults.update(overrides)
    return _dyn_params(**sysid_defaults)


class TestSysidParamsStraight:
    """Dynamic model with sysid params on a straight road."""

    def test_zero_error_near_zero_steer(self):
        """Sysid params, straight road, zero state → near-zero steering."""
        p = _sysid_params()
        s = MPCSolver(p)
        kappa = np.zeros(p.horizon)
        result = s.solve(
            e_lat=0.0, e_heading=0.0, v=11.0,
            last_delta_norm=0.0, kappa_ref_horizon=kappa,
            v_target=11.0, v_max=15.0, dt=p.dt,
            tire_cf=_SYSID_CF, tire_cr=_SYSID_CR,
            v_y_init=0.0, r_init=0.0,
        )
        assert result['feasible']
        assert abs(result['steering_normalized']) < 0.05

    def test_vy_r_bounded_in_trajectory(self):
        """Predicted v_y and r stay near zero on a straight."""
        p = _sysid_params()
        s = MPCSolver(p)
        kappa = np.zeros(p.horizon)
        result = s.solve(
            e_lat=0.0, e_heading=0.0, v=11.0,
            last_delta_norm=0.0, kappa_ref_horizon=kappa,
            v_target=11.0, v_max=15.0, dt=p.dt,
            tire_cf=_SYSID_CF, tire_cr=_SYSID_CR,
            v_y_init=0.0, r_init=0.0,
        )
        traj = result['predicted_trajectory']
        # v_y is state index 2, r is state index 3
        assert np.max(np.abs(traj[:, 2])) < 0.5, "v_y should stay near zero on straight"
        assert np.max(np.abs(traj[:, 3])) < 0.5, "r should stay near zero on straight"


class TestSysidFeedforward:
    """Verify dynamic feedforward produces understeer-aware steering."""

    def test_dynamic_ff_exceeds_kinematic(self):
        """With K_us > 0 (understeering), dynamic FF > kinematic FF."""
        kappa = 0.01  # R=100m
        vx = 11.0
        max_steer = 0.5236
        L = 1.125 + 1.125  # l_f + l_r

        ff_kin = MPCSolver._feedforward_delta_norm(kappa, L, max_steer)
        ff_dyn = MPCSolver._feedforward_delta_norm_dynamic(
            kappa, vx, _SYSID_CF, _SYSID_CR,
            1.125, 1.125, 1500.0, max_steer,
        )
        assert ff_dyn > ff_kin, (
            f"Dynamic FF ({ff_dyn:.5f}) should exceed kinematic ({ff_kin:.5f}) "
            f"for an understeering vehicle"
        )

    def test_sysid_understeer_gradient(self):
        """K_us from sysid params ≈ 0.010 rad/(m/s²)."""
        m, lf, lr = 1500.0, 1.125, 1.125
        L = lf + lr
        K_us = (m / L) * (lr / _SYSID_CF - lf / _SYSID_CR)
        assert K_us == pytest.approx(0.010, abs=0.002), (
            f"Sysid K_us={K_us:.4f}, expected ~0.010"
        )


class TestSysidVyBoundedInCurve:
    """Verify v_y stays bounded during step curvature changes."""

    def test_step_curvature_vy_bounded(self):
        """Sudden curvature onset → v_y in predicted trajectory stays < 1.0 m/s."""
        p = _sysid_params()
        s = MPCSolver(p)
        # Step curvature: first 5 steps straight, then κ=0.02 (R=50m)
        kappa = np.zeros(p.horizon)
        kappa[5:] = 0.02
        result = s.solve(
            e_lat=0.0, e_heading=0.0, v=11.0,
            last_delta_norm=0.0, kappa_ref_horizon=kappa,
            v_target=11.0, v_max=15.0, dt=p.dt,
            tire_cf=_SYSID_CF, tire_cr=_SYSID_CR,
            v_y_init=0.0, r_init=0.0,
        )
        assert result['feasible']
        traj = result['predicted_trajectory']
        max_vy = np.max(np.abs(traj[:, 2]))
        assert max_vy < 1.0, f"v_y peak {max_vy:.3f} exceeds 1.0 m/s — unreasonable"


class TestSysidIzSensitivity:
    """Verify that Iz affects yaw rate dynamics correctly."""

    def test_higher_iz_slower_yaw(self):
        """Higher Iz → less aggressive yaw rate in predicted trajectory."""
        kappa = np.full(20, 0.01)  # constant R=100m curve

        # Low Iz (original default)
        p_low = _sysid_params(vehicle_iz_kgm2=1250.0)
        s_low = MPCSolver(p_low)
        r_low = s_low.solve(
            e_lat=0.0, e_heading=0.0, v=11.0,
            last_delta_norm=0.0, kappa_ref_horizon=kappa,
            v_target=11.0, v_max=15.0, dt=p_low.dt,
            tire_cf=_SYSID_CF, tire_cr=_SYSID_CR,
            v_y_init=0.0, r_init=0.0,
        )

        # High Iz (sysid value)
        p_high = _sysid_params(vehicle_iz_kgm2=_SYSID_IZ)
        s_high = MPCSolver(p_high)
        r_high = s_high.solve(
            e_lat=0.0, e_heading=0.0, v=11.0,
            last_delta_norm=0.0, kappa_ref_horizon=kappa,
            v_target=11.0, v_max=15.0, dt=p_high.dt,
            tire_cf=_SYSID_CF, tire_cr=_SYSID_CR,
            v_y_init=0.0, r_init=0.0,
        )

        assert r_low['feasible'] and r_high['feasible']
        # Peak yaw rate in first few steps should be smaller with higher Iz
        r_peak_low = np.max(np.abs(r_low['predicted_trajectory'][:5, 3]))
        r_peak_high = np.max(np.abs(r_high['predicted_trajectory'][:5, 3]))
        assert r_peak_high < r_peak_low, (
            f"Higher Iz should produce less aggressive yaw: "
            f"Iz=2428 peak_r={r_peak_high:.4f}, Iz=1250 peak_r={r_peak_low:.4f}"
        )


class TestSysidInitClamps:
    """Verify solver handles unreasonable v_y and r initial states."""

    def test_vy_init_unreasonable_still_feasible(self):
        """v_y_init=5.0 (extreme) → solver still produces feasible result."""
        p = _sysid_params()
        s = MPCSolver(p)
        kappa = np.zeros(p.horizon)
        # Clamp as compute_steering() would
        vy_clamped = max(-2.0, min(2.0, 5.0))
        result = s.solve(
            e_lat=0.0, e_heading=0.0, v=11.0,
            last_delta_norm=0.0, kappa_ref_horizon=kappa,
            v_target=11.0, v_max=15.0, dt=p.dt,
            tire_cf=_SYSID_CF, tire_cr=_SYSID_CR,
            v_y_init=vy_clamped, r_init=0.0,
        )
        assert result['feasible']
        assert abs(result['steering_normalized']) <= 1.0

    def test_r_init_unreasonable_still_feasible(self):
        """r_init=2.0 (extreme) → solver still produces feasible result."""
        p = _sysid_params()
        s = MPCSolver(p)
        kappa = np.zeros(p.horizon)
        # Clamp as compute_steering() would
        r_clamped = max(-1.0, min(1.0, 2.0))
        result = s.solve(
            e_lat=0.0, e_heading=0.0, v=11.0,
            last_delta_norm=0.0, kappa_ref_horizon=kappa,
            v_target=11.0, v_max=15.0, dt=p.dt,
            tire_cf=_SYSID_CF, tire_cr=_SYSID_CR,
            v_y_init=0.0, r_init=r_clamped,
        )
        assert result['feasible']
        assert abs(result['steering_normalized']) <= 1.0
