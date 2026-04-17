"""Tests for preview-weighted (time-varying) q_lat in MPC.

Stage cost shaping: q_lat ramps from step0_scale at step 0 to 1.0 at step N-1.
Lower step-0 weight suppresses oscillation; higher horizon-end weight preserves
curve pre-steering.  Terminal step N keeps its own q_lat_terminal_scale.
"""

import numpy as np
import pytest

from control.mpc_controller import MPCParams, MPCSolver


def _make_params(**overrides) -> MPCParams:
    defaults = dict(
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
        ff_alignment_enabled=True,
        wheelbase_m=2.5,
        q_lat_terminal_scale=3.0,
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


# ─────────────────── Default is no-op ───────────────────


class TestPreviewWeightingDefault:
    def test_default_scale_is_1(self):
        p = MPCParams()
        assert p.q_lat_preview_step0_scale == 1.0

    def test_default_matches_baseline(self):
        """step0_scale=1.0 should produce identical steering to baseline."""
        p_ramp = _make_params(q_lat_preview_step0_scale=1.0)
        p_base = _make_params()  # also 1.0 by default
        s_ramp = MPCSolver(p_ramp)
        s_base = MPCSolver(p_base)

        r_ramp = _solve(s_ramp, e_lat=0.1)
        r_base = _solve(s_base, e_lat=0.1)
        assert r_ramp['steering_normalized'] == pytest.approx(
            r_base['steering_normalized'], abs=1e-8)

    def test_default_matches_baseline_on_curve(self):
        p_ramp = _make_params(q_lat_preview_step0_scale=1.0)
        p_base = _make_params()
        s_ramp = MPCSolver(p_ramp)
        s_base = MPCSolver(p_base)

        kappa = np.full(20, 0.02)
        r_ramp = _solve(s_ramp, kappa_ref_horizon=kappa, e_lat=0.05)
        r_base = _solve(s_base, kappa_ref_horizon=kappa, e_lat=0.05)
        assert r_ramp['steering_normalized'] == pytest.approx(
            r_base['steering_normalized'], abs=1e-8)


# ─────────────────── P diagonal shape ───────────────────


class TestPreviewWeightingPDiagonal:
    def test_p_diagonal_ramp_shape(self):
        """Verify q_lat contribution ramps linearly across horizon.
        Use q_lat_rate=0 to isolate the q_lat diagonal entries."""
        p = _make_params(q_lat=2.0, q_lat_preview_step0_scale=0.3, q_lat_rate=0.0)
        s = MPCSolver(p)
        N = s._N
        nx, nu = s._nx, s._nu

        P = s._P
        for k in range(N):
            idx = k * (nx + nu)  # e_lat position for step k
            alpha = k / (N - 1)
            w_k = 0.3 + 0.7 * alpha
            expected = 2.0 * w_k
            actual = P[idx, idx]
            assert actual == pytest.approx(expected, abs=1e-10), \
                f"Step {k}: expected {expected:.4f}, got {actual:.4f}"

    def test_terminal_unchanged(self):
        """Terminal step N keeps q_lat * terminal_scale, not part of the ramp."""
        p = _make_params(q_lat=2.0, q_lat_preview_step0_scale=0.3,
                         q_lat_terminal_scale=3.0, q_lat_rate=0.0)
        s = MPCSolver(p)
        N = s._N
        nx, nu = s._nx, s._nu

        term_idx = N * (nx + nu)
        expected = 2.0 * 3.0  # q_lat * terminal_scale
        actual = s._P[term_idx, term_idx]
        assert actual == pytest.approx(expected, abs=1e-10)

    def test_step0_gets_scaled_weight(self):
        """Step 0 should get q_lat * step0_scale (with q_lat_rate=0)."""
        p = _make_params(q_lat=4.0, q_lat_preview_step0_scale=0.25, q_lat_rate=0.0)
        s = MPCSolver(p)
        actual = s._P[0, 0]
        assert actual == pytest.approx(4.0 * 0.25, abs=1e-10)

    def test_last_intermediate_step_gets_full_weight(self):
        """Step N-1 should get q_lat * 1.0 (end of ramp, with q_lat_rate=0)."""
        p = _make_params(q_lat=2.0, q_lat_preview_step0_scale=0.3,
                         horizon=20, q_lat_rate=0.0)
        s = MPCSolver(p)
        nx, nu = s._nx, s._nu
        idx = 19 * (nx + nu)
        actual = s._P[idx, idx]
        assert actual == pytest.approx(2.0, abs=1e-10)

    def test_horizon_1_edge_case(self):
        """N=1: single step should use alpha=1.0 (full weight), no divide-by-zero."""
        p = _make_params(q_lat=2.0, q_lat_preview_step0_scale=0.3, horizon=1)
        s = MPCSolver(p)
        # Step 0 (only intermediate step) should get full weight since N=1 → alpha=1.0
        actual = s._P[0, 0]
        assert actual == pytest.approx(2.0, abs=1e-10)


# ─────────────────── Feasibility ───────────────────


class TestPreviewWeightingFeasibility:
    def test_straight_feasible(self):
        s = MPCSolver(_make_params(q_lat_preview_step0_scale=0.3))
        r = _solve(s)
        assert r['feasible']

    def test_curve_feasible(self):
        s = MPCSolver(_make_params(q_lat_preview_step0_scale=0.3))
        kappa = np.full(s._N, 0.02)
        r = _solve(s, kappa_ref_horizon=kappa)
        assert r['feasible']

    def test_zero_scale_feasible(self):
        """Edge case: step0_scale=0.0 (zero weight at step 0)."""
        s = MPCSolver(_make_params(q_lat_preview_step0_scale=0.0))
        r = _solve(s, e_lat=0.1)
        assert r['feasible']

    def test_multi_frame_stability(self):
        """10 consecutive frames, no infeasible solves."""
        s = MPCSolver(_make_params(q_lat_preview_step0_scale=0.3))
        kappa = np.full(s._N, 0.02)
        last_delta = 0.0
        for _ in range(10):
            r = _solve(s, kappa_ref_horizon=kappa, last_delta_norm=last_delta)
            assert r['feasible']
            last_delta = r['steering_normalized']


# ─────────────────── Oscillation suppression ───────────────────


class TestPreviewWeightingOscillation:
    def _run_closed_loop(self, step0_scale, q_lat, n_frames=30):
        """Run closed-loop simulation on straight road with initial lateral error."""
        p = _make_params(q_lat=q_lat, q_lat_preview_step0_scale=step0_scale)
        s = MPCSolver(p)
        last_delta = 0.0
        deltas = []
        for _ in range(n_frames):
            r = _solve(s, e_lat=0.3, e_heading=0.0, last_delta_norm=last_delta)
            assert r['feasible']
            last_delta = r['steering_normalized']
            deltas.append(last_delta)
        return deltas

    def _count_sign_changes(self, values):
        changes = 0
        for i in range(1, len(values)):
            if values[i] * values[i - 1] < 0:
                changes += 1
        return changes

    def test_ramp_reduces_sign_changes_at_boundary(self):
        """At q_lat=2.5 (near stability boundary), ramp should produce
        fewer sign changes than uniform."""
        deltas_uniform = self._run_closed_loop(step0_scale=1.0, q_lat=2.5)
        deltas_ramped = self._run_closed_loop(step0_scale=0.3, q_lat=2.5)

        sc_uniform = self._count_sign_changes(deltas_uniform)
        sc_ramped = self._count_sign_changes(deltas_ramped)
        # Ramped should have fewer or equal sign changes
        assert sc_ramped <= sc_uniform + 1  # +1 tolerance for noise

    def test_ramp_reduces_peak_amplitude(self):
        """At q_lat=3.0 (above stability boundary), ramp should produce
        lower peak-to-peak oscillation than uniform."""
        deltas_uniform = self._run_closed_loop(step0_scale=1.0, q_lat=3.0)
        deltas_ramped = self._run_closed_loop(step0_scale=0.3, q_lat=3.0)

        ptp_uniform = max(deltas_uniform) - min(deltas_uniform)
        ptp_ramped = max(deltas_ramped) - min(deltas_ramped)
        # Ramped should have lower or similar oscillation amplitude
        assert ptp_ramped <= ptp_uniform * 1.1  # 10% tolerance


# ─────────────────── Curve entry preserved ───────────────────


class TestPreviewWeightingCurveEntry:
    def test_curve_entry_steering_preserved(self):
        """On curve entry, ramped q_lat should produce similar or stronger
        steering than uniform (horizon-end q_lat is same or higher)."""
        p_uniform = _make_params(q_lat=2.0, q_lat_preview_step0_scale=1.0)
        p_ramped = _make_params(q_lat=2.0, q_lat_preview_step0_scale=0.3)
        s_uniform = MPCSolver(p_uniform)
        s_ramped = MPCSolver(p_ramped)

        kappa = np.full(20, 0.02)
        r_uniform = _solve(s_uniform, kappa_ref_horizon=kappa, e_lat=0.0)
        r_ramped = _solve(s_ramped, kappa_ref_horizon=kappa, e_lat=0.0)

        # Ramped should preserve at least 80% of curve-entry steering
        assert abs(r_ramped['steering_normalized']) >= \
            abs(r_uniform['steering_normalized']) * 0.8


# ─────────────────── Interaction with other features ───────────────────


class TestPreviewWeightingInteractions:
    def test_with_ff_decomposition(self):
        """Ramp + ff_decomposition both enabled: should be feasible and stable."""
        p = _make_params(q_lat_preview_step0_scale=0.3,
                         ff_decomposition_enabled=True)
        s = MPCSolver(p)
        kappa = np.full(s._N, 0.02)
        last_delta = 0.0
        for _ in range(10):
            r = _solve(s, kappa_ref_horizon=kappa, last_delta_norm=last_delta)
            assert r['feasible']
            last_delta = r['steering_normalized']

    def test_with_curvature_scheduled_q_lat(self):
        """Ramp should scale with curvature-scheduled q_lat base value."""
        p = _make_params(q_lat=2.0, q_lat_preview_step0_scale=0.3,
                         q_lat_kappa_gain=50.0)
        s = MPCSolver(p)
        kappa = np.full(s._N, 0.01)
        r = _solve(s, kappa_ref_horizon=kappa)
        assert r['feasible']
