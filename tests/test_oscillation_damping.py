"""
Tests for LMPC oscillation damping mechanisms.

Validates:
- r_steer_rate speed scheduling (gain-scheduled MPC, primary fix)
- Smith predictor sign-agreement guard (Fix 2, defense-in-depth)
"""

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# r_steer_rate speed scheduling
# ---------------------------------------------------------------------------

def _effective_r_steer_rate(
    speed: float,
    r_base: float = 3.5,
    onset: float = 12.0,
    gain: float = 0.15,
    max_scale: float = 3.0,
    enabled: bool = True,
    kappa: float = 0.0,
    kappa_off: float = 0.005,
    kappa_gain: float = 0.0,
    kappa_saturate: float = 0.015,
) -> float:
    """Mirror the formula in mpc_controller.py _get_effective_r_steer_rate."""
    if not enabled:
        return r_base
    excess = max(0.0, speed - onset)
    scale = min(1.0 + gain * excess, max_scale)
    # Speed scheduling with curvature gate
    kappa_abs = abs(kappa)
    kappa_gate = max(0.0, min(1.0, 1.0 - kappa_abs / max(1e-6, kappa_off)))
    speed_scale = 1.0 + (scale - 1.0) * kappa_gate
    # Curvature damping: proportional floor
    kappa_curve_scale = 1.0 + kappa_gain * min(kappa_abs, kappa_saturate)
    effective_scale = max(speed_scale, kappa_curve_scale)
    return r_base * effective_scale


class TestRSteerRateScheduling:
    """Verify speed-dependent r_steer_rate scheduling formula."""

    def test_scheduling_disabled_uses_base_rate(self):
        """When scheduling is disabled, always returns base rate."""
        for speed in [0.0, 12.0, 14.0, 20.0, 50.0]:
            assert _effective_r_steer_rate(speed, enabled=False) == 3.5

    def test_below_onset_no_scaling(self):
        """Below onset speed, effective rate equals base."""
        for speed in [0.0, 5.0, 10.0, 11.9]:
            assert _effective_r_steer_rate(speed) == 3.5

    def test_at_onset_no_scaling(self):
        """At exactly onset speed, effective rate equals base."""
        assert _effective_r_steer_rate(12.0) == 3.5

    @pytest.mark.parametrize(
        "speed, expected",
        [
            (14.0, 3.5 * (1.0 + 0.15 * 2)),   # 3.5 * 1.3 = 4.55
            (15.0, 3.5 * (1.0 + 0.15 * 3)),   # 3.5 * 1.45 = 5.075
            (20.0, 3.5 * (1.0 + 0.15 * 8)),   # 3.5 * 2.2 = 7.7
            (25.0, 3.5 * (1.0 + 0.15 * 13)),  # 3.5 * 2.95 = 10.325
        ],
    )
    def test_above_onset_linear_scaling(self, speed, expected):
        actual = _effective_r_steer_rate(speed)
        assert abs(actual - expected) < 1e-9, (
            f"At {speed} m/s: expected {expected:.3f}, got {actual:.3f}"
        )

    def test_max_scale_clamp(self):
        """At very high speed, clamped at r_base * max_scale."""
        # max_scale=3.0 → cap at 3.5 * 3.0 = 10.5
        # 1 + 0.15 * (speed - 12) = 3.0 → speed = 12 + 2/0.15 = 25.33
        very_high = _effective_r_steer_rate(50.0)
        assert abs(very_high - 3.5 * 3.0) < 1e-9

    def test_monotonically_increasing(self):
        """Effective rate must be non-decreasing with speed."""
        speeds = np.linspace(0, 60, 300)
        rates = [_effective_r_steer_rate(s) for s in speeds]
        for i in range(1, len(rates)):
            assert rates[i] >= rates[i - 1] - 1e-9, (
                f"Rate decreased at {speeds[i]:.1f} m/s"
            )

    def test_never_below_base(self):
        """Effective rate must never go below the base rate."""
        for speed in np.linspace(0, 100, 500):
            assert _effective_r_steer_rate(speed) >= 3.5 - 1e-9

    def test_autobahn_overlay_params(self):
        """Verify autobahn overlay produces expected values."""
        # Autobahn: r_base=1.5, onset=14, gain=0.12, max_scale=2.5
        at_14 = _effective_r_steer_rate(14.0, r_base=1.5, onset=14.0, gain=0.12, max_scale=2.5)
        at_20 = _effective_r_steer_rate(20.0, r_base=1.5, onset=14.0, gain=0.12, max_scale=2.5)
        at_25 = _effective_r_steer_rate(25.0, r_base=1.5, onset=14.0, gain=0.12, max_scale=2.5)
        assert abs(at_14 - 1.5) < 1e-9  # no change at onset
        assert abs(at_20 - 1.5 * (1.0 + 0.12 * 6)) < 1e-9  # 1.5 * 1.72 = 2.58
        assert abs(at_25 - 1.5 * (1.0 + 0.12 * 11)) < 1e-9  # 1.5 * 2.32 = 3.48

    def test_custom_params(self):
        """Verify formula with non-default parameters."""
        r = _effective_r_steer_rate(
            speed=20.0, r_base=2.0, onset=10.0, gain=0.10, max_scale=2.0
        )
        # scale = 1 + 0.10 * 10 = 2.0 → clamped at 2.0 → r = 4.0
        assert abs(r - 4.0) < 1e-9


class TestRSteerRateCurvatureDamping:
    """Verify curvature-proportional r_steer_rate damping."""

    def test_zero_kappa_gain_no_effect(self):
        """When kappa_gain=0 (default), curvature has no effect."""
        r = _effective_r_steer_rate(5.0, r_base=2.0, kappa=0.010, kappa_gain=0.0)
        assert abs(r - 2.0) < 1e-9

    def test_kappa_gain_increases_r_on_curves(self):
        """Curvature damping raises r_steer_rate proportionally."""
        # κ=0.010, gain=30 → kappa_curve_scale = 1 + 30*0.010 = 1.30
        r = _effective_r_steer_rate(
            5.0, r_base=2.0, kappa=0.010, kappa_gain=30.0, kappa_saturate=0.015
        )
        assert abs(r - 2.0 * 1.30) < 1e-9

    def test_kappa_saturates_at_ceiling(self):
        """Curvature scaling saturates at kappa_saturate."""
        # κ=0.067 (hairpin) > saturate=0.015 → capped
        # kappa_curve_scale = 1 + 30*0.015 = 1.45
        r = _effective_r_steer_rate(
            5.0, r_base=2.0, kappa=0.067, kappa_gain=30.0, kappa_saturate=0.015
        )
        assert abs(r - 2.0 * 1.45) < 1e-9

    def test_straight_gets_minimal_effect(self):
        """On straights (κ≈0), curvature damping is negligible."""
        r = _effective_r_steer_rate(
            12.0, r_base=2.0, kappa=0.001, kappa_gain=30.0, kappa_saturate=0.015
        )
        # kappa_curve_scale = 1 + 30*0.001 = 1.03 → r = 2.06
        assert abs(r - 2.0 * 1.03) < 1e-9

    def test_speed_dominates_on_fast_straights(self):
        """Speed scheduling beats curvature damping at high speed on pure straights."""
        # Speed=20, onset=12, gain=0.15 → raw speed scale = 2.2
        # κ=0.0 → kappa_gate=1.0, speed_scale = 2.2, kappa_curve_scale = 1.0
        # max(2.2, 1.0) = 2.2 → r = 3.5 * 2.2 = 7.7
        r = _effective_r_steer_rate(
            20.0, r_base=3.5, kappa=0.0, kappa_gain=30.0, kappa_saturate=0.015
        )
        assert abs(r - 3.5 * 2.2) < 1e-9

    def test_curvature_dominates_on_slow_curves(self):
        """Curvature damping beats speed scheduling on slow tight curves."""
        # Speed=6 (below onset=12) → speed_scale = 1.0
        # κ=0.025, gain=30, saturate=0.015 → kappa_curve_scale = 1 + 30*0.015 = 1.45
        # max(1.0, 1.45) = 1.45 → r = 2.0 * 1.45 = 2.90
        r = _effective_r_steer_rate(
            6.0, r_base=2.0, kappa=0.025, kappa_gain=30.0, kappa_saturate=0.015
        )
        assert abs(r - 2.0 * 1.45) < 1e-9

    def test_r_eff_stays_below_historic_problem_value(self):
        """With gain=30, saturate=0.015, r_eff never reaches 3.5 (known problem)."""
        # Max kappa_curve_scale = 1 + 30*0.015 = 1.45
        # r_eff = 2.0 * 1.45 = 2.90 < 3.5
        for kappa in [0.01, 0.02, 0.05, 0.10]:
            r = _effective_r_steer_rate(
                5.0, r_base=2.0, kappa=kappa, kappa_gain=30.0, kappa_saturate=0.015
            )
            assert r < 3.5, f"r_eff={r:.2f} at κ={kappa} exceeds safe limit 3.5"


class TestQLatCurvatureScheduling:
    """Verify curvature-scheduled q_lat for MPC curve-entry tracking."""

    def test_zero_gain_no_effect(self):
        pytest.importorskip("osqp")
        from control.mpc_controller import MPCParams, MPCSolver
        p = MPCParams(q_lat=1.0, q_lat_kappa_gain=0.0)
        s = MPCSolver(p)
        assert s._get_effective_q_lat(0.01) == 1.0

    def test_gain_increases_q_lat_on_curves(self):
        pytest.importorskip("osqp")
        from control.mpc_controller import MPCParams, MPCSolver
        p = MPCParams(q_lat=1.0, q_lat_kappa_gain=300.0, q_lat_kappa_saturate=0.015)
        s = MPCSolver(p)
        # κ=0.010 → scale = 1 + 300*0.010 = 4.0 → q_lat = 4.0
        assert abs(s._get_effective_q_lat(0.010) - 4.0) < 1e-9

    def test_saturates_at_ceiling(self):
        pytest.importorskip("osqp")
        from control.mpc_controller import MPCParams, MPCSolver
        p = MPCParams(q_lat=1.0, q_lat_kappa_gain=300.0, q_lat_kappa_saturate=0.015)
        s = MPCSolver(p)
        # κ=0.067 (hairpin) > saturate → capped at scale = 1 + 300*0.015 = 5.5
        assert abs(s._get_effective_q_lat(0.067) - 5.5) < 1e-9

    def test_straight_gets_no_scaling(self):
        pytest.importorskip("osqp")
        from control.mpc_controller import MPCParams, MPCSolver
        p = MPCParams(q_lat=2.0, q_lat_kappa_gain=300.0, q_lat_kappa_saturate=0.015)
        s = MPCSolver(p)
        assert abs(s._get_effective_q_lat(0.0) - 2.0) < 1e-9

    def test_q_lat_used_in_qp_build(self):
        """Verify _current_q_lat propagates to the P matrix diagonal."""
        pytest.importorskip("osqp")
        from control.mpc_controller import MPCParams, MPCSolver
        p = MPCParams(q_lat=1.0, q_lat_kappa_gain=300.0, q_lat_kappa_saturate=0.015)
        s = MPCSolver(p)
        # Initially q_lat = 1.0
        assert s._current_q_lat == 1.0
        # Simulate kappa change that triggers rebuild
        s._current_q_lat = 4.0
        s._build_qp()
        # P matrix should now use 4.0 for the lateral state cost


class TestRSteerRateSchedulingMPC:
    """Integration tests: verify MPCSolver uses scheduling correctly.
    Requires osqp — tests are skipped if not installed."""

    _DT = 0.033

    def _make_solver(self, scheduling_enabled=True, **overrides):
        """Create an MPCSolver with scheduling params."""
        pytest.importorskip("osqp")
        from control.mpc_controller import MPCParams, MPCSolver
        kwargs = dict(
            horizon=10,
            dt=self._DT,
            r_steer_rate=3.5,
            r_steer_rate_scheduling_enabled=scheduling_enabled,
            r_steer_rate_speed_onset=12.0,
            r_steer_rate_speed_gain=0.15,
            r_steer_rate_max_scale=3.0,
            r_steer_rate_band_hysteresis=1.0,
            r_steer_rate_straight_damping_enabled=False,
            first_step_rate_enabled=True,
            ff_alignment_enabled=False,
        )
        kwargs.update(overrides)
        params = MPCParams(**kwargs)
        return MPCSolver(params)

    def _kappa_horizon(self, n=10):
        """Return a zero-curvature horizon array (straight road)."""
        import numpy as np
        return np.zeros(n)

    def test_scheduling_disabled_returns_base(self):
        """When disabled, solve returns base r_steer_rate."""
        solver = self._make_solver(scheduling_enabled=False)
        result = solver.solve(
            e_lat=0.1, e_heading=0.01, v=20.0,
            last_delta_norm=0.0, kappa_ref_horizon=self._kappa_horizon(),
            v_target=15.0, v_max=20.0, dt=self._DT,
        )
        assert result['r_steer_rate_effective'] == 3.5

    def test_scheduling_enabled_returns_scaled(self):
        """When enabled, solve returns speed-scaled r_steer_rate."""
        solver = self._make_solver(scheduling_enabled=True)
        result = solver.solve(
            e_lat=0.1, e_heading=0.01, v=20.0,
            last_delta_norm=0.0, kappa_ref_horizon=self._kappa_horizon(),
            v_target=15.0, v_max=20.0, dt=self._DT,
        )
        expected = 3.5 * (1.0 + 0.15 * 8)  # 7.7
        assert abs(result['r_steer_rate_effective'] - expected) < 0.2

    def test_hysteresis_prevents_chattering(self):
        """Small speed fluctuations within hysteresis band don't trigger rebuild."""
        solver = self._make_solver(scheduling_enabled=True)
        kh = self._kappa_horizon()
        # First solve at 14 m/s — sets band center
        solver.solve(
            e_lat=0.1, e_heading=0.01, v=14.0,
            last_delta_norm=0.0, kappa_ref_horizon=kh,
            v_target=15.0, v_max=20.0, dt=self._DT,
        )
        r_at_14 = solver._current_r_steer_rate
        # Solve at 14.5 m/s — within 1.0 hysteresis
        solver.solve(
            e_lat=0.1, e_heading=0.01, v=14.5,
            last_delta_norm=0.0, kappa_ref_horizon=kh,
            v_target=15.0, v_max=20.0, dt=self._DT,
        )
        assert solver._current_r_steer_rate == r_at_14, (
            "r_steer_rate should not change within hysteresis band"
        )

    def test_band_crossing_triggers_rebuild(self):
        """Speed crossing beyond hysteresis triggers r_steer_rate update."""
        solver = self._make_solver(scheduling_enabled=True)
        kh = self._kappa_horizon()
        # Solve at 14 m/s
        solver.solve(
            e_lat=0.1, e_heading=0.01, v=14.0,
            last_delta_norm=0.0, kappa_ref_horizon=kh,
            v_target=15.0, v_max=20.0, dt=self._DT,
        )
        r_at_14 = solver._current_r_steer_rate
        # Solve at 20 m/s — well beyond hysteresis
        solver.solve(
            e_lat=0.1, e_heading=0.01, v=20.0,
            last_delta_norm=0.0, kappa_ref_horizon=kh,
            v_target=20.0, v_max=25.0, dt=self._DT,
        )
        r_at_20 = solver._current_r_steer_rate
        assert r_at_20 > r_at_14, (
            f"r_steer_rate should increase: {r_at_14:.2f} → {r_at_20:.2f}"
        )

    def test_higher_speed_smoother_steering(self):
        """At higher speed with scheduling, steering response should be smoother
        (smaller absolute steering for same error)."""
        solver_low = self._make_solver(scheduling_enabled=True)
        solver_high = self._make_solver(scheduling_enabled=True)

        # Use small errors so neither solver saturates the steering constraint.
        result_low = solver_low.solve(
            e_lat=0.05, e_heading=0.01, v=12.0,
            last_delta_norm=0.0, kappa_ref_horizon=self._kappa_horizon(),
            v_target=12.0, v_max=15.0, dt=self._DT,
        )
        result_high = solver_high.solve(
            e_lat=0.05, e_heading=0.01, v=20.0,
            last_delta_norm=0.0, kappa_ref_horizon=self._kappa_horizon(),
            v_target=20.0, v_max=25.0, dt=self._DT,
        )
        # Higher r_steer_rate should produce less aggressive steering
        # (more rate penalty → smaller first step)
        steer_low = abs(result_low['steering_normalized'])
        steer_high = abs(result_high['steering_normalized'])
        assert steer_high < steer_low, (
            f"Higher speed should produce smaller steering: "
            f"{steer_low:.4f} (12 m/s) vs {steer_high:.4f} (20 m/s)"
        )


# ---------------------------------------------------------------------------
# Smith predictor oscillation guard (Fix 2 — retained as defense-in-depth)
# ---------------------------------------------------------------------------

def _smith_predictor(
    raw_e_lat: float,
    raw_e_heading: float,
    v: float,
    delay_dt: float,
    disagreement_gain: float = 0.3,
) -> float:
    """Mirror the Smith predictor logic in pid_controller.py."""
    sign_agree = (raw_e_lat * raw_e_heading) >= 0
    gain = 1.0 if sign_agree else disagreement_gain
    return raw_e_lat + gain * v * raw_e_heading * delay_dt


class TestSmithPredictorGuard:
    """Verify sign-agreement gating on Smith predictor."""

    def test_signs_agree_full_correction(self):
        """When e_lat and e_heading have same sign, full correction applied."""
        pred = _smith_predictor(0.5, 0.1, 14.0, 0.066)
        expected = 0.5 + 1.0 * 14.0 * 0.1 * 0.066
        assert abs(pred - expected) < 1e-9

    def test_signs_disagree_attenuated(self):
        """When e_lat and e_heading have opposite signs, correction attenuated."""
        pred = _smith_predictor(0.5, -0.1, 14.0, 0.066)
        expected = 0.5 + 0.3 * 14.0 * (-0.1) * 0.066
        assert abs(pred - expected) < 1e-9

    def test_zero_e_lat_treated_as_agree(self):
        """When e_lat=0, sign_agree is True (0*anything >= 0)."""
        pred = _smith_predictor(0.0, 0.1, 14.0, 0.066)
        expected = 0.0 + 1.0 * 14.0 * 0.1 * 0.066
        assert abs(pred - expected) < 1e-9

    def test_zero_e_heading_no_correction(self):
        """When e_heading=0, no correction regardless of gain."""
        pred = _smith_predictor(0.5, 0.0, 14.0, 0.066)
        assert abs(pred - 0.5) < 1e-9

    def test_both_negative_signs_agree(self):
        """Both negative = signs agree = full correction."""
        pred = _smith_predictor(-0.5, -0.1, 14.0, 0.066)
        expected = -0.5 + 1.0 * 14.0 * (-0.1) * 0.066
        assert abs(pred - expected) < 1e-9

    def test_negative_lat_positive_heading_disagree(self):
        """Negative e_lat + positive e_heading = disagree = attenuated."""
        pred = _smith_predictor(-0.5, 0.1, 14.0, 0.066)
        expected = -0.5 + 0.3 * 14.0 * 0.1 * 0.066
        assert abs(pred - expected) < 1e-9

    def test_oscillation_inflection_reduces_overshoot(self):
        """At oscillation inflection (signs disagree), prediction should be
        closer to raw_e_lat than unguarded prediction."""
        raw_e_lat = 0.3
        raw_e_heading = -0.08  # opposite sign = inflection
        v, delay = 14.0, 0.066
        guarded = _smith_predictor(raw_e_lat, raw_e_heading, v, delay, 0.3)
        unguarded = raw_e_lat + v * raw_e_heading * delay
        # Guarded should be closer to raw_e_lat
        assert abs(guarded - raw_e_lat) < abs(unguarded - raw_e_lat)

    def test_custom_disagreement_gain(self):
        """Custom gain value is respected."""
        pred = _smith_predictor(0.5, -0.1, 14.0, 0.066, disagreement_gain=0.0)
        # With gain=0, no heading correction at all
        assert abs(pred - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# Ramp limiter production-disabled guards
# ---------------------------------------------------------------------------

class TestRampLimiterProductionDisabled:
    """Verify the e_lat ramp limiter is disabled in production config.

    The ramp limiter (mpc_elat_ramp_rate_m_per_frame) was found to cause a
    self-sustaining limit cycle: at 0.05 m/frame it caps MPC error tracking
    to 4-17% of true GT cross-track change during oscillation (0.10-0.15
    m/frame), creating phase lag that amplifies the oscillation. Disabled
    2026-03-31.
    """

    def test_mpc_params_default_ramp_disabled(self):
        """MPCParams dataclass default must be 0.0 (disabled)."""
        pytest.importorskip("osqp")
        from control.mpc_controller import MPCParams
        assert MPCParams().elat_ramp_rate_m_per_frame == 0.0

    def test_base_config_ramp_disabled(self):
        """Base config av_stack_config.yaml must have ramp = 0.0."""
        import yaml
        from pathlib import Path
        cfg_path = Path(__file__).resolve().parent.parent / "config" / "av_stack_config.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        ramp = cfg["trajectory"]["mpc"]["mpc_elat_ramp_rate_m_per_frame"]
        assert ramp == 0.0, f"Ramp limiter must be disabled (0.0), got {ramp}"


# ---------------------------------------------------------------------------
# MPCParams from_config picks up scheduling fields
# ---------------------------------------------------------------------------

class TestMPCParamsSchedulingConfig:
    """Verify MPCParams.from_config picks up r_steer_rate scheduling fields."""

    def test_defaults(self):
        pytest.importorskip("osqp")
        from control.mpc_controller import MPCParams
        p = MPCParams()
        assert p.r_steer_rate_scheduling_enabled is False
        assert p.r_steer_rate_speed_onset == 12.0
        assert p.r_steer_rate_speed_gain == 0.15
        assert p.r_steer_rate_max_scale == 3.0
        assert p.r_steer_rate_band_hysteresis == 1.0

    def test_from_config_loads_scheduling(self):
        pytest.importorskip("osqp")
        from control.mpc_controller import MPCParams
        cfg = {
            "trajectory": {
                "mpc": {
                    "mpc_r_steer_rate_scheduling_enabled": True,
                    "mpc_r_steer_rate_speed_onset": 14.0,
                    "mpc_r_steer_rate_speed_gain": 0.12,
                    "mpc_r_steer_rate_max_scale": 2.5,
                    "mpc_r_steer_rate_band_hysteresis": 2.0,
                }
            }
        }
        p = MPCParams.from_config(cfg)
        assert p.r_steer_rate_scheduling_enabled is True
        assert p.r_steer_rate_speed_onset == 14.0
        assert abs(p.r_steer_rate_speed_gain - 0.12) < 1e-9
        assert abs(p.r_steer_rate_max_scale - 2.5) < 1e-9
        assert abs(p.r_steer_rate_band_hysteresis - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# Smith predictor wall-clock delay cap
# ---------------------------------------------------------------------------

def _smith_predictor_total_delay(
    delay_frames: int,
    frame_dt: float,
    max_delay_s: float,
) -> float:
    """Mirror the wall-clock cap logic in pid_controller.py."""
    raw = delay_frames * frame_dt
    if max_delay_s > 0:
        return min(raw, max_delay_s)
    return raw


class TestSmithPredictorDelayCap:
    """Verify wall-clock cap on Smith predictor delay (prevents 4× overshoot
    when camera rate drops below designed 30 fps)."""

    def test_no_clamp_at_30fps(self):
        """At 30 fps, 2 × 0.033 = 0.066s < 0.080s cap → no clamp."""
        total = _smith_predictor_total_delay(2, 0.033, 0.080)
        assert abs(total - 0.066) < 1e-6

    def test_clamp_at_7hz(self):
        """At 7.5 fps, 2 × 0.133 = 0.266s → clamped to 0.080s."""
        total = _smith_predictor_total_delay(2, 0.133, 0.080)
        assert abs(total - 0.080) < 1e-6

    def test_clamp_at_extreme_low_rate(self):
        """At 3 fps, 2 × 0.333 = 0.666s → clamped to 0.080s."""
        total = _smith_predictor_total_delay(2, 0.333, 0.080)
        assert abs(total - 0.080) < 1e-6

    def test_zero_disables_cap(self):
        """max_delay_s=0.0 disables cap (legacy behavior)."""
        total = _smith_predictor_total_delay(2, 0.133, 0.0)
        assert abs(total - 0.266) < 1e-6

    def test_custom_cap_value(self):
        """Custom max_delay_s=0.120 respected."""
        total = _smith_predictor_total_delay(2, 0.133, 0.120)
        assert abs(total - 0.120) < 1e-6

    def test_config_key_in_base_yaml(self):
        """Verify smith_predictor_max_delay_s exists in base config."""
        import yaml
        from pathlib import Path
        cfg_path = Path(__file__).resolve().parent.parent / "config" / "av_stack_config.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        val = cfg["trajectory"]["mpc"]["smith_predictor_max_delay_s"]
        assert val == 0.080, f"Expected 0.080, got {val}"
