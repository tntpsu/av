"""Tests for three score-improvement fixes applied in av_stack_config.yaml:

Fix A – mpc_r_steer_rate: 2.0 → 3.5
    Higher steering-rate cost in the MPC objective directly damps the ~0.5 Hz
    oscillation whose period matches the 2 s MPC horizon.  The solver penalises
    rapid steering reversals more, smoothing out underdamped weave.

Fix B – reference_lookahead_tight_scale: 0.35 → 0.50
    On tight-curvature road the planned ref_y now contracts to 50 % of base
    (was 35 %).  At 9 m base lookahead this raises the pre-floor ld from 3.15 m
    to 4.50 m, cutting the PP floor rescue gap roughly in half.

Fix C – traj_heading_zero_gate_curvature_preview_off_threshold: 0.002 → 0.0015
    The heading-zero gate releases 25 % earlier on curve approach, preventing
    heading-error accumulation on long straights that feeds the oscillation.
"""

import pytest
import numpy as np

from control.mpc_controller import MPCController, MPCParams
from trajectory.utils import _compute_tight_curve_scale, compute_reference_lookahead
from trajectory.inference import TrajectoryPlanningInference


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mpc(**overrides) -> MPCController:
    cfg = {"trajectory": {"mpc": {
        "mpc_horizon": 10,
        "mpc_dt": 0.1,
        "mpc_wheelbase_m": 2.5,
        "mpc_max_steer_rad": 0.5236,
        "mpc_q_lat": 0.5,
        "mpc_q_heading": 5.0,
        "mpc_q_speed": 2.0,
        "mpc_r_steer": 0.05,
        "mpc_r_accel": 0.2,
        "mpc_r_steer_rate": 2.0,
        "mpc_delta_rate_max": 0.15,
        "mpc_max_accel": 3.0,
        "mpc_max_decel": 4.0,
        "mpc_v_min": 1.0,
        "mpc_v_max": 15.0,
        "mpc_speed_adaptive_horizon": False,
        "mpc_max_solve_time_ms": 200.0,
        "mpc_max_consecutive_failures": 3,
        **overrides,
    }}}
    return MPCController(cfg)


def _solve(ctrl, e_lat=0.0, e_heading=0.0, speed=8.0, kappa=0.0,
           v_target=8.0, last_delta=0.0) -> dict:
    return ctrl.compute_steering(
        e_lat=e_lat, e_heading=e_heading,
        current_speed=speed, last_delta_norm=last_delta,
        kappa_ref=kappa, v_target=v_target,
        v_max=v_target + 2.0, dt=0.033,
    )


def _base_traj_config(extra: dict | None = None) -> dict:
    cfg = {
        "dynamic_reference_lookahead": True,
        "reference_lookahead": 9.0,
        "curve_scheduler_mode": "phase_active",
        "reference_lookahead_speed_table": [
            {"speed_mps": 0.0, "lookahead_m": 7.0},
            {"speed_mps": 8.94, "lookahead_m": 11.0},
            {"speed_mps": 13.41, "lookahead_m": 15.0},
        ],
        "reference_lookahead_speed_table_entry": [
            {"speed_mps": 0.0, "lookahead_m": 7.0},
            {"speed_mps": 8.94, "lookahead_m": 9.0},
            {"speed_mps": 13.41, "lookahead_m": 10.0},
        ],
        "reference_lookahead_min": 2.5,
        "reference_lookahead_tight_curvature_threshold": 0.01,
        "reference_lookahead_tight_blend_band": 0.006,
    }
    if extra:
        cfg.update(extra)
    return cfg


# ===========================================================================
# Fix A: MPC steering-rate cost damps oscillation
# ===========================================================================

class TestMPCSteeringRateDamping:
    """Higher mpc_r_steer_rate should produce smaller steering changes
    across consecutive corrections, damping steering oscillation."""

    def test_higher_rate_cost_reduces_steering_reversal(self):
        """With higher r_steer_rate, the solver limits steering reversals."""
        low_rate  = _make_mpc(mpc_r_steer_rate=2.0)
        high_rate = _make_mpc(mpc_r_steer_rate=3.5)

        # Simulate alternating lateral errors (oscillation scenario)
        low_deltas, high_deltas = [], []
        prev_low = prev_high = 0.0
        errors = [0.3, -0.2, 0.3, -0.2, 0.3]

        for e in errors:
            r_low  = _solve(low_rate,  e_lat=e, last_delta=prev_low)
            r_high = _solve(high_rate, e_lat=e, last_delta=prev_high)
            if r_low.get("feasible") and r_high.get("feasible"):
                delta_low  = r_low.get("steering_norm",  r_low.get("delta_norm",  0.0))
                delta_high = r_high.get("steering_norm", r_high.get("delta_norm", 0.0))
                low_deltas.append(abs(delta_low  - prev_low))
                high_deltas.append(abs(delta_high - prev_high))
                prev_low  = delta_low
                prev_high = delta_high

        if low_deltas and high_deltas:
            avg_reversal_low  = np.mean(low_deltas)
            avg_reversal_high = np.mean(high_deltas)
            assert avg_reversal_high <= avg_reversal_low + 1e-4, (
                f"Higher r_steer_rate should not increase average reversal size: "
                f"low={avg_reversal_low:.4f}, high={avg_reversal_high:.4f}"
            )

    def test_mpc_params_loads_steer_rate_from_config(self):
        """MPCParams must pick up mpc_r_steer_rate from nested config."""
        cfg = {"trajectory": {"mpc": {"mpc_r_steer_rate": 3.5}}}
        params = MPCParams.from_config(cfg)
        assert params.r_steer_rate == pytest.approx(3.5, rel=1e-6), (
            "MPCParams.r_steer_rate should equal the configured value 3.5"
        )

    def test_mpc_params_default_steer_rate_unchanged(self):
        """Ensure the MPCParams default is still usable (not broken by config)."""
        params = MPCParams()
        assert params.r_steer_rate > 0.0, "r_steer_rate must be positive"

    def test_higher_rate_cost_produces_feasible_solve(self):
        """r_steer_rate=3.5 must not cause solver infeasibility."""
        ctrl = _make_mpc(mpc_r_steer_rate=3.5)
        result = _solve(ctrl, e_lat=0.3, e_heading=0.05, speed=10.0)
        assert result.get("feasible", True), (
            "Solver should remain feasible with mpc_r_steer_rate=3.5"
        )


# ===========================================================================
# Fix B: tight_scale 0.35 → 0.50 reduces floor rescue gap
# ===========================================================================

class TestTightCurveScaleReduction:
    """With tight_scale=0.50, the planned ref_y contracts to 50% (was 35%)
    on tight curves, leaving a smaller gap for the PP floor to rescue."""

    def _scale_at_curvature(self, tight_scale: float, curvature: float) -> float:
        config = {
            "reference_lookahead_tight_curvature_threshold": 0.01,
            "reference_lookahead_tight_scale": tight_scale,
            "reference_lookahead_tight_blend_band": 0.006,
        }
        return _compute_tight_curve_scale(curvature, config)

    def test_old_scale_gives_35_pct_on_tight_curve(self):
        """Original scale=0.35 should fully apply on κ >> threshold."""
        scale = self._scale_at_curvature(tight_scale=0.35, curvature=0.02)
        assert scale == pytest.approx(0.35, rel=1e-3), (
            f"Expected scale ≈ 0.35, got {scale:.4f}"
        )

    def test_new_scale_gives_50_pct_on_tight_curve(self):
        """New scale=0.50 should fully apply on κ >> threshold."""
        scale = self._scale_at_curvature(tight_scale=0.50, curvature=0.02)
        assert scale == pytest.approx(0.50, rel=1e-3), (
            f"Expected scale ≈ 0.50, got {scale:.4f}"
        )

    def test_new_scale_produces_larger_pre_floor_ld(self):
        """At the same base lookahead, tight_scale=0.50 yields a higher pre-floor ld
        than 0.35, directly reducing the PP floor rescue magnitude."""
        base = 9.0
        ld_old = base * self._scale_at_curvature(0.35, 0.02)
        ld_new = base * self._scale_at_curvature(0.50, 0.02)
        assert ld_new > ld_old, (
            f"New scale should give larger pre-floor ld: {ld_new:.2f} > {ld_old:.2f}"
        )
        # Quantify improvement: gap to a 5 m floor
        floor_m = 5.0
        rescue_old = max(0.0, floor_m - ld_old)
        rescue_new = max(0.0, floor_m - ld_new)
        assert rescue_new < rescue_old, (
            f"Floor rescue should decrease: {rescue_new:.2f} < {rescue_old:.2f}"
        )

    def test_scale_is_1_on_straight(self):
        """On a straight (κ < threshold) neither scale should shrink the lookahead."""
        for scale in (0.35, 0.50):
            s = self._scale_at_curvature(scale, curvature=0.001)
            assert s == pytest.approx(1.0, rel=1e-3), (
                f"tight_scale={scale}: should be 1.0 on straight, got {s:.4f}"
            )

    def test_scale_blends_smoothly_near_threshold(self):
        """With blend_band=0.006 the scale transitions smoothly around threshold."""
        old_mid = self._scale_at_curvature(0.35, 0.01)  # at threshold midpoint
        new_mid = self._scale_at_curvature(0.50, 0.01)

        # At threshold both should be between straight (1.0) and full scale
        assert 0.35 < old_mid < 1.0, f"Old scale mid={old_mid:.4f} should be blended"
        assert 0.50 < new_mid < 1.0, f"New scale mid={new_mid:.4f} should be blended"
        assert new_mid > old_mid, (
            "New scale midpoint should be larger than old (less contraction)"
        )


# ===========================================================================
# Fix C: heading gate threshold 0.002 → 0.0015
# ===========================================================================

class TestHeadingGateLowerThreshold:
    """With threshold=0.0015 the gate releases when preview curvature exceeds
    0.0015 rad/m, which is 25 % earlier than the previous 0.002 threshold."""

    def _make_engine(self, threshold: float) -> TrajectoryPlanningInference:
        return TrajectoryPlanningInference(
            planner_type="rule_based",
            target_lane="center",
            target_lane_width_m=3.6,
            traj_heading_zero_gate_curvature_preview_off_threshold=threshold,
        )

    def test_old_threshold_does_not_release_at_0018(self):
        """curvature_preview=0.0018 should NOT release gate with old threshold=0.002."""
        eng = self._make_engine(threshold=0.002)
        eng.heading_zero_gate_active = True
        eng._map_preview_curvature_abs = 0.0018

        coeffs = [np.array([0.001, 0.0, 1.8]), np.array([0.001, 0.0, -1.8])]
        result = eng._update_heading_zero_gate(coeffs, raw_heading_rad=0.02)
        assert result is True, (
            "With threshold=0.002, preview=0.0018 should NOT release the gate"
        )

    def test_new_threshold_releases_at_0018(self):
        """curvature_preview=0.0018 SHOULD release gate with new threshold=0.0015."""
        eng = self._make_engine(threshold=0.0015)
        eng.heading_zero_gate_active = True
        eng._map_preview_curvature_abs = 0.0018

        coeffs = [np.array([0.001, 0.0, 1.8]), np.array([0.001, 0.0, -1.8])]
        result = eng._update_heading_zero_gate(coeffs, raw_heading_rad=0.02)
        assert result is False, (
            "With threshold=0.0015, preview=0.0018 SHOULD release the gate earlier"
        )

    def test_both_thresholds_release_above_0003(self):
        """Both old and new thresholds must release when preview curvature is high."""
        for threshold in (0.002, 0.0015):
            eng = self._make_engine(threshold=threshold)
            eng.heading_zero_gate_active = True
            eng._map_preview_curvature_abs = 0.003

            coeffs = [np.array([0.001, 0.0, 1.8]), np.array([0.001, 0.0, -1.8])]
            result = eng._update_heading_zero_gate(coeffs, raw_heading_rad=0.02)
            assert result is False, (
                f"threshold={threshold}: gate must release when preview=0.003"
            )

    def test_both_thresholds_latch_on_straight(self):
        """Lowering the OFF threshold must not prevent the gate from turning ON."""
        for threshold in (0.002, 0.0015):
            eng = self._make_engine(threshold=threshold)
            eng.heading_zero_gate_active = False
            eng._map_preview_curvature_abs = 0.0

            coeffs = [np.array([0.001, 0.0, 1.8]), np.array([0.001, 0.0, -1.8])]
            result = eng._update_heading_zero_gate(coeffs, raw_heading_rad=0.01)
            assert result is True, (
                f"threshold={threshold}: gate must still latch ON on straights"
            )

    def test_earlier_release_quantified(self):
        """New threshold should release at a lower curvature preview than old."""
        def min_release_curvature(threshold: float) -> float:
            """Binary-search the minimum preview curvature that releases the gate."""
            lo, hi = 0.0, 0.01
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                eng = self._make_engine(threshold=threshold)
                eng.heading_zero_gate_active = True
                eng._map_preview_curvature_abs = mid
                coeffs = [np.array([0.001, 0.0, 1.8]), np.array([0.001, 0.0, -1.8])]
                released = not eng._update_heading_zero_gate(coeffs, raw_heading_rad=0.02)
                if released:
                    hi = mid
                else:
                    lo = mid
            return hi

        release_old = min_release_curvature(0.002)
        release_new = min_release_curvature(0.0015)
        assert release_new < release_old, (
            f"New threshold must release earlier: {release_new:.5f} < {release_old:.5f}"
        )
        assert release_new == pytest.approx(0.0015, rel=0.01), (
            f"New threshold should release right at 0.0015, got {release_new:.5f}"
        )
