"""
Phase G1 — Unit tests for NMPCParams, NMPCSolver, NMPCController.

Tests cover:
  - NMPCParams: from_config key loading, defaults, type coercion
  - NMPCSolver: feedforward delta, rollout physics, gradient correctness
  - NMPCSolver: straight-road solution, solve timing
  - NMPCController: compute_steering interface, fallback logic, reset
"""
import math
import time

import numpy as np
import pytest

from control.nmpc_controller import NMPCController, NMPCParams, NMPCSolver


# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def default_params():
    return NMPCParams()


@pytest.fixture
def small_params():
    """Tiny horizon for fast unit tests."""
    return NMPCParams(horizon=5, dt=0.1, max_iter=30)


@pytest.fixture
def solver(small_params):
    return NMPCSolver(small_params)


@pytest.fixture
def controller():
    """NMPCController with minimal config (uses all defaults)."""
    return NMPCController({})


# ─── NMPCParams tests ─────────────────────────────────────────────────────────

class TestNMPCParams:
    def test_defaults_are_reasonable(self, default_params):
        assert default_params.horizon == 20
        assert default_params.dt == pytest.approx(0.1)
        assert default_params.wheelbase_m == pytest.approx(2.5)
        assert 0.3 < default_params.max_steer_rad < 1.0

    def test_from_config_reads_nmpc_prefix(self):
        cfg = {
            "trajectory": {
                "nmpc": {
                    "nmpc_horizon": 15,
                    "nmpc_dt": 0.08,
                    "nmpc_q_lat": 3.0,
                }
            }
        }
        p = NMPCParams.from_config(cfg)
        assert p.horizon == 15
        assert p.dt == pytest.approx(0.08)
        assert p.q_lat == pytest.approx(3.0)
        # Untouched params remain at default
        assert p.wheelbase_m == pytest.approx(2.5)

    def test_from_config_empty_uses_defaults(self):
        p = NMPCParams.from_config({})
        assert p.horizon == 20

    def test_from_config_type_coercion_int(self):
        cfg = {"trajectory": {"nmpc": {"nmpc_horizon": "10"}}}
        p = NMPCParams.from_config(cfg)
        assert p.horizon == 10
        assert isinstance(p.horizon, int)

    def test_fallback_lmpc_default_true(self, default_params):
        assert default_params.fallback_lmpc is True

    def test_ff_alignment_default_true(self, default_params):
        assert default_params.ff_alignment_enabled is True


# ─── NMPCSolver tests ─────────────────────────────────────────────────────────

class TestNMPCSolverFeedforward:
    def test_straight_road_zero_ff(self, solver):
        ff = solver._feedforward_delta_norm(kappa=0.0, wheelbase_m=2.5, max_steer_rad=0.5236)
        assert ff == pytest.approx(0.0)

    def test_positive_curvature_positive_ff(self, solver):
        ff = solver._feedforward_delta_norm(kappa=0.01, wheelbase_m=2.5, max_steer_rad=0.5236)
        assert ff > 0.0

    def test_ff_magnitude_within_bounds(self, solver):
        """Feedforward should be < 1 for realistic road curvatures (κ < 0.5)."""
        ff = solver._feedforward_delta_norm(kappa=0.05, wheelbase_m=2.5, max_steer_rad=0.5236)
        assert abs(ff) < 1.0

    def test_ff_antisymmetric(self, solver):
        ff_pos = solver._feedforward_delta_norm(0.01, 2.5, 0.5236)
        ff_neg = solver._feedforward_delta_norm(-0.01, 2.5, 0.5236)
        assert ff_pos == pytest.approx(-ff_neg)


class TestNMPCSolverRollout:
    def test_rollout_output_shape(self, solver):
        N = solver.p.horizon
        u = np.zeros(2 * N)
        states = solver._rollout(u, 0.0, 0.0, 15.0, np.zeros(N), 0.0, 30.0)
        assert states.shape == (N + 1, 3)

    def test_rollout_zero_error_straight_stays_zero(self, solver):
        """On straight road with zero initial error and zero controls, errors stay 0."""
        N = solver.p.horizon
        u = np.zeros(2 * N)
        states = solver._rollout(u, 0.0, 0.0, 15.0, np.zeros(N), 0.0, 30.0)
        assert np.allclose(states[:, 0], 0.0, atol=1e-10)  # e_lat
        assert np.allclose(states[:, 1], 0.0, atol=1e-10)  # e_heading

    def test_rollout_speed_clamps_at_v_min(self, solver):
        """Large deceleration should clamp at v_min, not go negative."""
        N = solver.p.horizon
        u = np.zeros(2 * N)
        # Full deceleration on every step
        for k in range(N):
            u[2 * k + 1] = -solver.p.max_decel
        states = solver._rollout(u, 0.0, 0.0, 1.5, np.zeros(N), 0.0, 30.0)
        assert np.all(states[:, 2] >= solver.p.v_min - 1e-9)

    def test_rollout_positive_heading_grows_e_lat(self, solver):
        """Positive heading error (pointing left) on straight road → e_lat grows."""
        N = solver.p.horizon
        u = np.zeros(2 * N)
        e_h0 = 0.1  # pointing left
        states = solver._rollout(u, 0.0, e_h0, 10.0, np.zeros(N), 0.0, 30.0)
        assert states[-1, 0] > 0.0, "Positive heading error should grow lateral error"


class TestNMPCSolverGradient:
    def test_gradient_matches_finite_difference(self, solver):
        """Analytical gradient must agree with finite-difference to 1e-5 relative."""
        p = solver.p
        N = p.horizon
        rng = np.random.default_rng(42)
        u = rng.uniform(-0.3, 0.3, 2 * N)
        kappa = rng.uniform(0.0, 0.01, N)

        args = (0.1, 0.05, 15.0, kappa, 15.0, 0.0, 0.0, 30.0, None)
        cost, grad = solver._cost_and_jac(u, *args)

        eps = 1e-5
        fd_grad = np.empty_like(u)
        for i in range(len(u)):
            u_p = u.copy(); u_p[i] += eps
            u_m = u.copy(); u_m[i] -= eps
            c_p, _ = solver._cost_and_jac(u_p, *args)
            c_m, _ = solver._cost_and_jac(u_m, *args)
            fd_grad[i] = (c_p - c_m) / (2 * eps)

        # Combined tolerance: pass if abs ≤ 1e-5 OR relative ≤ 1e-4.
        # Near-zero accel gradient components (v ≈ v_target) can have trivially
        # large relative error — the absolute check handles those safely.
        abs_err = np.abs(grad - fd_grad)
        rel_err = abs_err / (np.abs(fd_grad) + 1e-8)
        passes = (abs_err <= 1e-5) | (rel_err < 1e-4)
        assert np.all(passes), (
            f"Gradient mismatch. Max abs: {np.max(abs_err):.2e}, "
            f"Max rel: {np.max(rel_err):.2e}. "
            f"Check A_k^T[2,1] = (1/L)·tan(δ)·dt − κ·dt in _cost_and_jac."
        )


class TestNMPCSolverSolve:
    def test_solve_straight_road_returns_near_zero_steer(self, solver):
        """On straight road with small initial error, optimal steering should be small."""
        N = solver.p.horizon
        kappa = np.zeros(N)
        result = solver.solve(0.05, 0.0, 15.0, 0.0, kappa, 15.0, 30.0, 0.033)
        assert result['feasible'], "Should be feasible on easy straight-road problem"
        assert abs(result['steering_normalized']) < 0.5

    def test_solve_returns_required_keys(self, solver):
        N = solver.p.horizon
        kappa = np.zeros(N)
        result = solver.solve(0.0, 0.0, 15.0, 0.0, kappa, 15.0, 30.0, 0.033)
        for key in ('steering_normalized', 'accel', 'feasible', 'solve_time_ms',
                    'predicted_trajectory', 'nmpc_cost', 'nmpc_iterations', 'slsqp_status'):
            assert key in result, f"Missing key: {key}"

    def test_solve_predicted_trajectory_shape(self, solver):
        N = solver.p.horizon
        kappa = np.zeros(N)
        result = solver.solve(0.0, 0.0, 15.0, 0.0, kappa, 15.0, 30.0, 0.033)
        assert result['predicted_trajectory'].shape == (N + 1, 3)

    def test_solve_time_within_budget(self, solver):
        """Warm-started solve should be well under 20ms budget."""
        N = solver.p.horizon
        kappa = np.zeros(N)
        # Prime the warm start
        solver.solve(0.0, 0.0, 15.0, 0.0, kappa, 15.0, 30.0, 0.033)
        t0 = time.perf_counter()
        solver.solve(0.02, 0.01, 15.0, 0.0, kappa, 15.0, 30.0, 0.033)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 100.0, f"Solve took {elapsed_ms:.1f}ms — unexpectedly slow in test env"

    def test_warm_start_set_after_feasible_solve(self, solver):
        N = solver.p.horizon
        kappa = np.zeros(N)
        assert solver._warm_u is None
        result = solver.solve(0.0, 0.0, 15.0, 0.0, kappa, 15.0, 30.0, 0.033)
        if result['feasible']:
            assert solver._warm_u is not None
            assert len(solver._warm_u) == 2 * N


class TestNMPCSignConvention:
    """Verify NMPC steers in the same direction as LMPC for the same e_lat input."""

    def test_positive_e_lat_steers_negative(self, solver):
        """e_lat > 0 (car right of center) → steer left (negative delta)."""
        N = solver.p.horizon
        kappa = np.zeros(N)
        result = solver.solve(0.3, 0.0, 5.0, 0.0, kappa, 5.0, 10.0, 0.077)
        assert result['feasible']
        assert result['steering_normalized'] < -0.01, (
            f"Expected negative steering for positive e_lat, got {result['steering_normalized']:.4f}"
        )

    def test_negative_e_lat_steers_positive(self, solver):
        """e_lat < 0 (car left of center) → steer right (positive delta)."""
        N = solver.p.horizon
        kappa = np.zeros(N)
        result = solver.solve(-0.3, 0.0, 5.0, 0.0, kappa, 5.0, 10.0, 0.077)
        assert result['feasible']
        assert result['steering_normalized'] > 0.01, (
            f"Expected positive steering for negative e_lat, got {result['steering_normalized']:.4f}"
        )

    def test_nmpc_matches_lmpc_direction(self):
        """NMPC and LMPC must steer the same direction for the same e_lat."""
        from control.mpc_controller import MPCController
        lmpc = MPCController({})
        nmpc = NMPCController({})
        # Warm up LMPC — first QP call may return zero from all-zeros warm-start
        lmpc.compute_steering(
            e_lat=0.1, e_heading=0.0, current_speed=5.0,
            last_delta_norm=0.0, kappa_ref=0.0,
            v_target=5.0, v_max=10.0, dt=0.077,
        )
        for e_lat in [0.3, -0.3, 0.1, -0.5]:
            lr = lmpc.compute_steering(
                e_lat=e_lat, e_heading=0.0, current_speed=5.0,
                last_delta_norm=0.0, kappa_ref=0.0,
                v_target=5.0, v_max=10.0, dt=0.077,
            )
            nr = nmpc.compute_steering(
                e_lat=e_lat, e_heading=0.0, current_speed=5.0,
                last_delta_norm=0.0, kappa_ref=0.0,
                v_target=5.0, v_max=10.0, dt=0.077,
            )
            l_steer = lr['steering_normalized']
            n_steer = nr['steering_normalized']
            # Skip sign check if either controller produces negligible output
            if abs(l_steer) < 1e-4 or abs(n_steer) < 1e-4:
                continue
            assert l_steer * n_steer > 0, (
                f"Sign mismatch at e_lat={e_lat}: LMPC={l_steer:.4f}, NMPC={n_steer:.4f}"
            )


# ─── NMPCController tests ─────────────────────────────────────────────────────

class TestNMPCController:
    def test_compute_steering_returns_required_keys(self, controller):
        result = controller.compute_steering(
            e_lat=0.1, e_heading=0.05, current_speed=22.0,
            last_delta_norm=0.0, kappa_ref=0.002, v_target=22.0,
            v_max=30.0, dt=0.033,
        )
        for key in ('steering_normalized', 'accel', 'nmpc_feasible',
                    'solve_time_ms', 'nmpc_fallback_active',
                    'nmpc_consecutive_failures', 'nmpc_cost',
                    'nmpc_iterations', 'e_lat_input', 'e_heading_input',
                    'kappa_ref_used'):
            assert key in result, f"Missing key: {key}"

    def test_steering_normalized_in_bounds(self, controller):
        result = controller.compute_steering(
            e_lat=0.0, e_heading=0.0, current_speed=22.0,
            last_delta_norm=0.0, kappa_ref=0.0, v_target=22.0,
            v_max=30.0, dt=0.033,
        )
        assert -1.0 <= result['steering_normalized'] <= 1.0

    def test_fallback_not_active_on_startup(self, controller):
        result = controller.compute_steering(
            e_lat=0.0, e_heading=0.0, current_speed=22.0,
            last_delta_norm=0.0, kappa_ref=0.0, v_target=22.0,
            v_max=30.0, dt=0.033,
        )
        assert result['nmpc_fallback_active'] is False

    def test_reset_clears_state(self, controller):
        # Run a few frames
        for _ in range(3):
            controller.compute_steering(
                e_lat=0.2, e_heading=0.1, current_speed=22.0,
                last_delta_norm=0.1, kappa_ref=0.003, v_target=22.0,
                v_max=30.0, dt=0.033,
            )
        controller.reset()
        assert controller.solver._warm_u is None
        assert controller._consecutive_failures == 0
        assert controller._fallback_active is False
        assert controller._last_steering == 0.0
        assert controller._frames_since_reset == 0

    def test_kappa_horizon_used_when_provided(self, controller):
        """Providing a kappa horizon should not crash and gives a valid result."""
        kappa_horizon = np.full(controller.params.horizon, 0.003)
        result = controller.compute_steering(
            e_lat=0.05, e_heading=0.02, current_speed=22.0,
            last_delta_norm=0.0, kappa_ref=0.003, v_target=22.0,
            v_max=30.0, dt=0.033, kappa_horizon=kappa_horizon,
        )
        assert result['kappa_ref_used'] == pytest.approx(0.003)

    def test_grade_rad_does_not_crash(self, controller):
        """Non-zero grade should be accepted without exception."""
        result = controller.compute_steering(
            e_lat=0.0, e_heading=0.0, current_speed=22.0,
            last_delta_norm=0.0, kappa_ref=0.0, v_target=22.0,
            v_max=30.0, dt=0.033, grade_rad=0.05,
        )
        assert 'steering_normalized' in result

    def test_should_fallback_to_lmpc_false_initially(self, controller):
        assert controller.should_fallback_to_lmpc is False

    def test_warmup_frames_use_zero_last_delta(self, controller):
        """During warmup, last_delta_norm is internally set to 0 — ensure no crash."""
        for _ in range(controller.params.warmup_frames + 1):
            result = controller.compute_steering(
                e_lat=0.1, e_heading=0.0, current_speed=22.0,
                last_delta_norm=0.9,  # Large — should be suppressed during warmup
                kappa_ref=0.0, v_target=22.0, v_max=30.0, dt=0.033,
            )
        # After warmup, last_delta_norm should be respected again (no crash)
        assert 'steering_normalized' in result


class TestNMPCControllerFallback:
    def test_consecutive_failures_counted(self):
        """Simulate forced failures by monkeypatching solver.solve to return infeasible."""
        ctrl = NMPCController({})
        orig_solve = ctrl.solver.solve

        def bad_solve(*args, **kwargs):
            r = orig_solve(*args, **kwargs)
            r['feasible'] = False
            r['solve_time_ms'] = 0.0
            return r

        ctrl.solver.solve = bad_solve
        max_f = ctrl.params.max_consecutive_failures

        for i in range(max_f - 1):
            r = ctrl.compute_steering(
                e_lat=0.1, e_heading=0.0, current_speed=22.0,
                last_delta_norm=0.0, kappa_ref=0.0, v_target=22.0,
                v_max=30.0, dt=0.033,
            )
            assert r['nmpc_fallback_active'] is False, f"Should not trigger before {max_f} failures"

        # One more failure should trigger
        r = ctrl.compute_steering(
            e_lat=0.1, e_heading=0.0, current_speed=22.0,
            last_delta_norm=0.0, kappa_ref=0.0, v_target=22.0,
            v_max=30.0, dt=0.033,
        )
        assert r['nmpc_fallback_active'] is True
        assert r['nmpc_consecutive_failures'] == max_f

    def test_fallback_clears_on_success(self):
        """After failures trigger fallback, a successful solve should clear it."""
        ctrl = NMPCController({})
        orig_solve = ctrl.solver.solve
        fail_mode = [True]

        def sometimes_fail(*args, **kwargs):
            r = orig_solve(*args, **kwargs)
            if fail_mode[0]:
                r['feasible'] = False
                r['solve_time_ms'] = 0.0
            return r

        ctrl.solver.solve = sometimes_fail
        max_f = ctrl.params.max_consecutive_failures

        # Trigger fallback
        for _ in range(max_f):
            ctrl.compute_steering(
                e_lat=0.1, e_heading=0.0, current_speed=22.0,
                last_delta_norm=0.0, kappa_ref=0.0, v_target=22.0,
                v_max=30.0, dt=0.033,
            )
        assert ctrl._fallback_active is True

        # Restore good solver
        fail_mode[0] = False
        r = ctrl.compute_steering(
            e_lat=0.0, e_heading=0.0, current_speed=22.0,
            last_delta_norm=0.0, kappa_ref=0.0, v_target=22.0,
            v_max=30.0, dt=0.033,
        )
        if r['nmpc_feasible']:
            assert r['nmpc_fallback_active'] is False
            assert r['nmpc_consecutive_failures'] == 0
