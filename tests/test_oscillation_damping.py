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
) -> float:
    """Mirror the formula in mpc_controller.py _get_effective_r_steer_rate."""
    if not enabled:
        return r_base
    excess = max(0.0, speed - onset)
    scale = min(1.0 + gain * excess, max_scale)
    return r_base * scale


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


class TestRSteerRateSchedulingMPC:
    """Integration tests: verify MPCSolver uses scheduling correctly.
    Requires osqp — tests are skipped if not installed."""

    def _make_solver(self, scheduling_enabled=True, **overrides):
        """Create an MPCSolver with scheduling params."""
        pytest.importorskip("osqp")
        from control.mpc_controller import MPCParams, MPCSolver
        kwargs = dict(
            horizon=10,
            dt=0.033,
            r_steer_rate=3.5,
            r_steer_rate_scheduling_enabled=scheduling_enabled,
            r_steer_rate_speed_onset=12.0,
            r_steer_rate_speed_gain=0.15,
            r_steer_rate_max_scale=3.0,
            r_steer_rate_band_hysteresis=1.0,
            first_step_rate_enabled=True,
            ff_alignment_enabled=False,
        )
        kwargs.update(overrides)
        params = MPCParams(**kwargs)
        return MPCSolver(params)

    def test_scheduling_disabled_returns_base(self):
        """When disabled, solve returns base r_steer_rate."""
        solver = self._make_solver(scheduling_enabled=False)
        result = solver.solve(
            e_lat=0.1, e_heading=0.01, v=20.0,
            last_delta_norm=0.0, v_target=15.0, v_max=20.0
        )
        assert result['r_steer_rate_effective'] == 3.5

    def test_scheduling_enabled_returns_scaled(self):
        """When enabled, solve returns speed-scaled r_steer_rate."""
        solver = self._make_solver(scheduling_enabled=True)
        result = solver.solve(
            e_lat=0.1, e_heading=0.01, v=20.0,
            last_delta_norm=0.0, v_target=15.0, v_max=20.0
        )
        expected = 3.5 * (1.0 + 0.15 * 8)  # 7.7
        assert abs(result['r_steer_rate_effective'] - expected) < 0.2

    def test_hysteresis_prevents_chattering(self):
        """Small speed fluctuations within hysteresis band don't trigger rebuild."""
        solver = self._make_solver(scheduling_enabled=True)
        # First solve at 14 m/s — sets band center
        solver.solve(
            e_lat=0.1, e_heading=0.01, v=14.0,
            last_delta_norm=0.0, v_target=15.0, v_max=20.0
        )
        r_at_14 = solver._current_r_steer_rate
        # Solve at 14.5 m/s — within 1.0 hysteresis
        solver.solve(
            e_lat=0.1, e_heading=0.01, v=14.5,
            last_delta_norm=0.0, v_target=15.0, v_max=20.0
        )
        assert solver._current_r_steer_rate == r_at_14, (
            "r_steer_rate should not change within hysteresis band"
        )

    def test_band_crossing_triggers_rebuild(self):
        """Speed crossing beyond hysteresis triggers r_steer_rate update."""
        solver = self._make_solver(scheduling_enabled=True)
        # Solve at 14 m/s
        solver.solve(
            e_lat=0.1, e_heading=0.01, v=14.0,
            last_delta_norm=0.0, v_target=15.0, v_max=20.0
        )
        r_at_14 = solver._current_r_steer_rate
        # Solve at 20 m/s — well beyond hysteresis
        solver.solve(
            e_lat=0.1, e_heading=0.01, v=20.0,
            last_delta_norm=0.0, v_target=20.0, v_max=25.0
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

        result_low = solver_low.solve(
            e_lat=0.3, e_heading=0.05, v=12.0,
            last_delta_norm=0.0, v_target=12.0, v_max=15.0
        )
        result_high = solver_high.solve(
            e_lat=0.3, e_heading=0.05, v=20.0,
            last_delta_norm=0.0, v_target=20.0, v_max=25.0
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
# e_lat speed attenuation (complements r_steer_rate scheduling)
# ---------------------------------------------------------------------------

def _elat_attenuation(
    e_lat: float,
    speed: float,
    kappa: float = 0.0,
    onset: float = 12.0,
    rate: float = 0.10,
    min_factor: float = 0.5,
    kappa_off: float = 0.005,
) -> float:
    """Mirror the e_lat attenuation formula in pid_controller.py."""
    if speed <= onset:
        return e_lat
    kappa_gate = max(0.0, min(1.0, 1.0 - abs(kappa) / max(1e-6, kappa_off)))
    raw_factor = max(min_factor, 1.0 - rate * (speed - onset))
    factor = 1.0 - (1.0 - raw_factor) * kappa_gate
    return e_lat * factor


class TestElatSpeedAttenuation:
    """Verify e_lat speed attenuation formula."""

    def test_below_onset_no_attenuation(self):
        for speed in [0.0, 5.0, 10.0, 12.0]:
            assert _elat_attenuation(1.0, speed) == 1.0

    def test_above_onset_attenuates(self):
        result = _elat_attenuation(1.0, 14.0)
        # rate=0.10, excess=2 → factor = 0.80
        assert abs(result - 0.80) < 1e-9

    def test_high_speed_clamps_at_min(self):
        result = _elat_attenuation(1.0, 30.0)
        # rate=0.10, excess=18 → raw = 1-1.8 = -0.8, clamped to 0.5
        assert abs(result - 0.5) < 1e-9

    def test_on_curve_no_attenuation(self):
        """At κ >= kappa_off, attenuation is fully disabled."""
        result = _elat_attenuation(1.0, 20.0, kappa=0.005)
        assert abs(result - 1.0) < 1e-9

    def test_partial_curvature_gating(self):
        """At κ = kappa_off/2, half the attenuation is applied."""
        result = _elat_attenuation(1.0, 14.0, kappa=0.0025)
        # raw_factor = 0.80, kappa_gate = 1 - 0.0025/0.005 = 0.5
        # factor = 1 - (1-0.80)*0.5 = 1 - 0.10 = 0.90
        assert abs(result - 0.90) < 1e-9

    def test_preserves_sign(self):
        """Attenuation preserves e_lat sign."""
        pos = _elat_attenuation(0.5, 14.0)
        neg = _elat_attenuation(-0.5, 14.0)
        assert pos > 0
        assert neg < 0
        assert abs(pos + neg) < 1e-9

    def test_monotonically_decreasing_with_speed(self):
        speeds = np.linspace(0, 40, 200)
        results = [abs(_elat_attenuation(1.0, s)) for s in speeds]
        for i in range(1, len(results)):
            assert results[i] <= results[i - 1] + 1e-9

    def test_never_amplifies(self):
        """Attenuation factor is always <= 1.0."""
        for speed in np.linspace(0, 50, 300):
            for kappa in [0.0, 0.001, 0.003, 0.005, 0.010]:
                assert abs(_elat_attenuation(1.0, speed, kappa)) <= 1.0 + 1e-9


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
