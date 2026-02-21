"""
Unit tests for control/speed_governor.py.

Tests the SpeedGovernor in isolation (no av_stack.py dependency):
- Comfort governor physics
- Preview deceleration
- Horizon guardrail
- Speed planner integration
- Edge cases
- Active limiter attribution
"""

import math

from control.speed_governor import (
    SpeedGovernor,
    SpeedGovernorConfig,
    SpeedGovernorOutput,
    build_speed_governor,
    G,
)
from trajectory.speed_planner import SpeedPlannerConfig


def _make_governor(
    max_lat_accel_g: float = 0.25,
    planner_enabled: bool = True,
    horizon_enabled: bool = True,
    preview_enabled: bool = True,
    curvature_calibration_scale: float = 1.0,
    curvature_history_frames: int = 5,
    **overrides,
) -> SpeedGovernor:
    """Helper to build a SpeedGovernor with common defaults."""
    gov_cfg = SpeedGovernorConfig(
        comfort_governor_max_lat_accel_g=max_lat_accel_g,
        comfort_governor_min_speed=3.0,
        curvature_calibration_scale=curvature_calibration_scale,
        curvature_history_frames=curvature_history_frames,
        curve_preview_enabled=preview_enabled,
        curve_preview_lookahead_scale=1.6,
        curve_preview_max_decel_mps2=1.8,
        curve_preview_min_curvature=0.002,
        horizon_guardrail_enabled=horizon_enabled,
        horizon_guardrail_time_headway_s=overrides.get("time_headway", 0.8),
        horizon_guardrail_margin_m=1.0,
        horizon_guardrail_min_speed_mps=3.0,
        horizon_guardrail_gain=1.0,
        dynamic_effective_horizon_enabled=horizon_enabled,
        speed_planner_enabled=planner_enabled,
        speed_planner_speed_limit_bias=0.0,
    )
    planner_cfg = SpeedPlannerConfig(
        max_accel=2.0, max_decel=2.5, max_jerk=1.2, default_dt=1.0 / 30.0,
    )
    return SpeedGovernor(gov_cfg, planner_cfg)


# --- Comfort Governor Physics ---

class TestComfortGovernor:
    def test_straight_road_no_limit(self):
        """Zero curvature should not limit speed."""
        gov = _make_governor(planner_enabled=False, horizon_enabled=False)
        out = gov.compute_target_speed(
            track_speed_limit=12.0, curvature=0.0, preview_curvature=None,
            perception_horizon_m=20.0, current_speed=5.0, timestamp=0.0,
        )
        assert out.target_speed == 12.0
        assert out.comfort_speed == float("inf")
        assert out.active_limiter == "none"

    def test_comfort_speed_matches_physics(self):
        """v_comfort = sqrt(a_lat_max / |curvature|) should match."""
        max_g = 0.25
        curvature = 0.01  # 100m radius
        expected = math.sqrt(max_g * G / curvature)

        gov = _make_governor(max_lat_accel_g=max_g, planner_enabled=False,
                             horizon_enabled=False, preview_enabled=False)
        out = gov.compute_target_speed(
            track_speed_limit=20.0, curvature=curvature, preview_curvature=None,
            perception_horizon_m=20.0, current_speed=5.0, timestamp=0.0,
        )
        assert abs(out.comfort_speed - expected) < 0.01
        assert abs(out.target_speed - expected) < 0.01
        assert out.active_limiter == "comfort"

    def test_comfort_respects_min_speed(self):
        """Very tight curvature should floor at min_speed."""
        gov = _make_governor(planner_enabled=False, horizon_enabled=False,
                             preview_enabled=False)
        out = gov.compute_target_speed(
            track_speed_limit=12.0, curvature=1.0, preview_curvature=None,
            perception_horizon_m=20.0, current_speed=5.0, timestamp=0.0,
        )
        assert out.comfort_speed >= 3.0

    def test_higher_lat_accel_allows_higher_speed(self):
        """Raising max_lat_accel_g should increase comfort speed."""
        curvature = 0.01
        gov_low = _make_governor(max_lat_accel_g=0.15, planner_enabled=False,
                                  horizon_enabled=False, preview_enabled=False)
        gov_high = _make_governor(max_lat_accel_g=0.35, planner_enabled=False,
                                   horizon_enabled=False, preview_enabled=False)
        out_low = gov_low.compute_target_speed(
            20.0, curvature, None, 20.0, 5.0, 0.0)
        out_high = gov_high.compute_target_speed(
            20.0, curvature, None, 20.0, 5.0, 0.0)
        assert out_high.comfort_speed > out_low.comfort_speed

    def test_track_limit_caps_comfort(self):
        """If track limit is below comfort speed, target should be track limit."""
        gov = _make_governor(planner_enabled=False, horizon_enabled=False,
                             preview_enabled=False)
        out = gov.compute_target_speed(
            track_speed_limit=5.0, curvature=0.0, preview_curvature=None,
            perception_horizon_m=20.0, current_speed=3.0, timestamp=0.0,
        )
        assert out.target_speed == 5.0
        assert out.active_limiter == "none"


# --- Preview Deceleration ---

class TestPreviewDecel:
    def test_preview_reduces_speed_before_curve(self):
        """Preview curvature tighter than current should reduce speed."""
        gov = _make_governor(planner_enabled=False, horizon_enabled=False)
        out = gov.compute_target_speed(
            track_speed_limit=12.0, curvature=0.0,
            preview_curvature=0.05,  # tight curve ahead (20m radius)
            perception_horizon_m=20.0, current_speed=10.0, timestamp=0.0,
        )
        assert out.preview_speed is not None
        assert out.target_speed < 12.0
        assert out.active_limiter in ("comfort", "preview")

    def test_no_preview_on_straight(self):
        """No preview decel when preview curvature is below threshold."""
        gov = _make_governor(planner_enabled=False, horizon_enabled=False)
        out = gov.compute_target_speed(
            track_speed_limit=12.0, curvature=0.0,
            preview_curvature=0.001,  # below min threshold
            perception_horizon_m=20.0, current_speed=5.0, timestamp=0.0,
        )
        assert out.preview_speed is None
        assert out.target_speed == 12.0

    def test_no_preview_when_disabled(self):
        """Preview should not fire when disabled."""
        gov = _make_governor(planner_enabled=False, horizon_enabled=False,
                             preview_enabled=False)
        out = gov.compute_target_speed(
            track_speed_limit=12.0, curvature=0.0,
            preview_curvature=0.03,
            perception_horizon_m=20.0, current_speed=10.0, timestamp=0.0,
        )
        assert out.preview_speed is None


# --- Horizon Guardrail ---

class TestHorizonGuardrail:
    def test_long_horizon_no_limit(self):
        """With 20m perception horizon and 0.8s headway, speed up to
        (20-1)/0.8 = 23.75 m/s allowed, so 12 m/s target should pass."""
        gov = _make_governor(planner_enabled=False, preview_enabled=False)
        out = gov.compute_target_speed(
            track_speed_limit=12.0, curvature=0.0, preview_curvature=None,
            perception_horizon_m=20.0, current_speed=5.0, timestamp=0.0,
        )
        assert out.target_speed == 12.0
        assert not out.horizon_guardrail_active

    def test_short_horizon_caps_speed(self):
        """With only 5m perception horizon: allowed = (5-1)/0.8 = 5.0 m/s."""
        gov = _make_governor(planner_enabled=False, preview_enabled=False)
        out = gov.compute_target_speed(
            track_speed_limit=12.0, curvature=0.0, preview_curvature=None,
            perception_horizon_m=5.0, current_speed=3.0, timestamp=0.0,
        )
        assert out.target_speed <= 5.5
        assert out.horizon_guardrail_active
        assert out.active_limiter == "horizon"

    def test_guardrail_disabled(self):
        """When guardrail is disabled, no horizon speed limit."""
        gov = _make_governor(planner_enabled=False, horizon_enabled=False,
                             preview_enabled=False)
        out = gov.compute_target_speed(
            track_speed_limit=12.0, curvature=0.0, preview_curvature=None,
            perception_horizon_m=3.0, current_speed=5.0, timestamp=0.0,
        )
        assert out.target_speed == 12.0
        assert not out.horizon_guardrail_active


# --- Speed Planner Integration ---

class TestPlannerIntegration:
    def test_planner_jerk_limits_ramp(self):
        """With planner enabled, speed should ramp gradually, not jump."""
        gov = _make_governor(horizon_enabled=False, preview_enabled=False)
        out = gov.compute_target_speed(
            track_speed_limit=12.0, curvature=0.0, preview_curvature=None,
            perception_horizon_m=20.0, current_speed=0.0, timestamp=0.0,
        )
        assert out.planned_speed is not None
        assert out.planned_speed < 12.0
        assert out.planned_speed > 0.0
        assert out.planner_active

    def test_planner_eventually_reaches_target(self):
        """Over many steps, planner should converge to target."""
        gov = _make_governor(horizon_enabled=False, preview_enabled=False)
        dt = 1.0 / 30.0
        speed = 0.0
        for i in range(300):
            out = gov.compute_target_speed(
                track_speed_limit=10.0, curvature=0.0, preview_curvature=None,
                perception_horizon_m=20.0, current_speed=speed, timestamp=i * dt,
            )
            speed = out.target_speed
        assert speed > 9.0, f"Should converge to ~10, got {speed}"

    def test_planner_disabled(self):
        """Without planner, target speed is computed but not jerk-limited."""
        gov = _make_governor(planner_enabled=False, horizon_enabled=False,
                             preview_enabled=False)
        out = gov.compute_target_speed(
            track_speed_limit=12.0, curvature=0.0, preview_curvature=None,
            perception_horizon_m=20.0, current_speed=0.0, timestamp=0.0,
        )
        assert out.planned_speed is None
        assert out.target_speed == 12.0


# --- Active Limiter Attribution ---

class TestActiveLimiter:
    def test_limiter_none_on_straight(self):
        """No limiter should be active on straight road at track limit."""
        gov = _make_governor(planner_enabled=False, horizon_enabled=False,
                             preview_enabled=False)
        out = gov.compute_target_speed(
            12.0, 0.0, None, 20.0, 5.0, 0.0)
        assert out.active_limiter == "none"

    def test_limiter_comfort_on_curve(self):
        """Comfort governor should be the limiter on a curve."""
        gov = _make_governor(planner_enabled=False, horizon_enabled=False,
                             preview_enabled=False)
        # curvature=0.03: v_comfort = sqrt(0.25*9.81/0.03) = 9.04 < 12
        out = gov.compute_target_speed(
            12.0, 0.03, None, 20.0, 5.0, 0.0)
        assert out.active_limiter == "comfort"
        assert out.target_speed < 12.0

    def test_limiter_horizon_on_short_perception(self):
        """Horizon should be the limiter when perception is short."""
        gov = _make_governor(planner_enabled=False, preview_enabled=False)
        out = gov.compute_target_speed(
            12.0, 0.0, None, 4.0, 3.0, 0.0)
        assert out.active_limiter == "horizon"


# --- Builder Function ---

class TestBuildSpeedGovernor:
    def test_build_from_config_dicts(self):
        """build_speed_governor should produce a working governor."""
        trajectory_cfg = {
            "speed_governor": {
                "comfort_governor_max_lat_accel_g": 0.3,
            },
            "lookahead_distance": 20.0,
            "min_speed_floor": 3.0,
        }
        speed_planner_cfg = {
            "enabled": True,
            "max_accel": 2.0,
            "max_decel": 2.5,
            "max_jerk": 1.2,
        }
        gov = build_speed_governor(trajectory_cfg, speed_planner_cfg)
        assert gov.config.comfort_governor_max_lat_accel_g == 0.3
        assert gov.speed_planner is not None

        out = gov.compute_target_speed(
            track_speed_limit=12.0, curvature=0.0, preview_curvature=None,
            perception_horizon_m=20.0, current_speed=0.0, timestamp=0.0,
        )
        assert out.target_speed > 0.0

    def test_build_with_calibration_scale(self):
        """build_speed_governor should read curvature_calibration_scale."""
        trajectory_cfg = {
            "speed_governor": {
                "comfort_governor_max_lat_accel_g": 0.20,
                "curvature_calibration_scale": 2.5,
                "curvature_history_frames": 5,
            },
        }
        speed_planner_cfg = {"enabled": False}
        gov = build_speed_governor(trajectory_cfg, speed_planner_cfg)
        assert gov.config.curvature_calibration_scale == 2.5
        assert gov.config.curvature_history_frames == 5

    def test_build_fallback_to_trajectory_cfg(self):
        """When speed_governor section is absent, should fall back to
        trajectory config values."""
        trajectory_cfg = {
            "curve_speed_preview_enabled": False,
            "speed_horizon_guardrail_enabled": False,
            "min_speed_floor": 2.0,
        }
        speed_planner_cfg = {"enabled": False}
        gov = build_speed_governor(trajectory_cfg, speed_planner_cfg)
        assert not gov.config.curve_preview_enabled
        assert not gov.config.horizon_guardrail_enabled
        assert gov.speed_planner is None


# --- Curvature Calibration Scale ---

class TestCurvatureCalibrationScale:
    def test_scaled_curvature_gives_lower_speed(self):
        """With calibration scale > 1, comfort speed should be lower than unscaled."""
        curvature = 0.01
        gov_unscaled = _make_governor(
            max_lat_accel_g=0.20, curvature_calibration_scale=1.0,
            planner_enabled=False, horizon_enabled=False, preview_enabled=False,
        )
        gov_scaled = _make_governor(
            max_lat_accel_g=0.20, curvature_calibration_scale=2.5,
            planner_enabled=False, horizon_enabled=False, preview_enabled=False,
        )
        out_unscaled = gov_unscaled.compute_target_speed(
            20.0, curvature, None, 20.0, 5.0, 0.0)
        out_scaled = gov_scaled.compute_target_speed(
            20.0, curvature, None, 20.0, 5.0, 0.0)
        assert out_scaled.comfort_speed < out_unscaled.comfort_speed
        ratio = out_unscaled.comfort_speed / out_scaled.comfort_speed
        expected_ratio = math.sqrt(2.5)
        assert abs(ratio - expected_ratio) < 0.01

    def test_calibration_scale_applies_to_preview(self):
        """Preview governor should also use calibration scale."""
        preview_k = 0.01
        gov_unscaled = _make_governor(
            max_lat_accel_g=0.20, curvature_calibration_scale=1.0,
            planner_enabled=False, horizon_enabled=False,
        )
        gov_scaled = _make_governor(
            max_lat_accel_g=0.20, curvature_calibration_scale=2.5,
            planner_enabled=False, horizon_enabled=False,
        )
        out_unscaled = gov_unscaled.compute_target_speed(
            20.0, 0.0, preview_k, 20.0, 10.0, 0.0)
        out_scaled = gov_scaled.compute_target_speed(
            20.0, 0.0, preview_k, 20.0, 10.0, 0.0)
        assert out_scaled.target_speed < out_unscaled.target_speed

    def test_scale_1_matches_unscaled_physics(self):
        """With scale=1.0, formula should match original physics exactly."""
        curvature = 0.015
        max_g = 0.20
        expected = math.sqrt(max_g * G / curvature)
        gov = _make_governor(
            max_lat_accel_g=max_g, curvature_calibration_scale=1.0,
            planner_enabled=False, horizon_enabled=False, preview_enabled=False,
        )
        out = gov.compute_target_speed(20.0, curvature, None, 20.0, 5.0, 0.0)
        assert abs(out.comfort_speed - expected) < 0.01


# --- Curvature History Max ---

class TestCurvatureHistoryMax:
    def test_zero_curvature_after_curve_holds_speed(self):
        """After seeing curvature, a transition to zero should still limit speed
        via history max for N frames."""
        gov = _make_governor(
            max_lat_accel_g=0.20, curvature_calibration_scale=2.5,
            curvature_history_frames=5,
            planner_enabled=False, horizon_enabled=False, preview_enabled=False,
        )
        out_curve = gov.compute_target_speed(
            20.0, 0.010, None, 20.0, 5.0, 0.0)
        comfort_during_curve = out_curve.comfort_speed

        out_zero = gov.compute_target_speed(
            20.0, 0.0, None, 20.0, 5.0, 0.1)
        assert out_zero.comfort_speed == comfort_during_curve, (
            "History max should hold the curvature from previous frame"
        )

    def test_history_expires_after_n_frames(self):
        """After N+1 frames of zero curvature, history should clear."""
        gov = _make_governor(
            max_lat_accel_g=0.20, curvature_calibration_scale=2.5,
            curvature_history_frames=3,
            planner_enabled=False, horizon_enabled=False, preview_enabled=False,
        )
        gov.compute_target_speed(20.0, 0.010, None, 20.0, 5.0, 0.0)
        for i in range(4):
            out = gov.compute_target_speed(
                20.0, 0.0, None, 20.0, 5.0, (i + 1) * 0.033)
        assert out.comfort_speed == float("inf"), (
            "After 4 zero-curvature frames with history=3, curve should expire"
        )

    def test_reset_clears_history(self):
        """reset() should clear curvature history."""
        gov = _make_governor(
            max_lat_accel_g=0.20, curvature_calibration_scale=2.5,
            curvature_history_frames=5,
            planner_enabled=False, horizon_enabled=False, preview_enabled=False,
        )
        gov.compute_target_speed(20.0, 0.010, None, 20.0, 5.0, 0.0)
        gov.reset()
        out = gov.compute_target_speed(20.0, 0.0, None, 20.0, 5.0, 0.1)
        assert out.comfort_speed == float("inf")


# --- Different Radii Produce Different Speeds ---

class TestDifferentRadiiSpeeds:
    def test_tighter_curve_slower_speed(self):
        """Tighter curvature (smaller radius) should produce lower comfort speed."""
        gov = _make_governor(
            max_lat_accel_g=0.20, curvature_calibration_scale=2.5,
            planner_enabled=False, horizon_enabled=False, preview_enabled=False,
        )
        radii = [80, 60, 40, 25]  # meters
        speeds = []
        for r in radii:
            gov.reset()
            k = 1.0 / r
            out = gov.compute_target_speed(20.0, k, None, 20.0, 5.0, 0.0)
            speeds.append(out.comfort_speed)

        for i in range(len(speeds) - 1):
            assert speeds[i] > speeds[i + 1], (
                f"Radius {radii[i]}m ({speeds[i]:.1f} m/s) should give higher "
                f"speed than {radii[i+1]}m ({speeds[i+1]:.1f} m/s)"
            )

    def test_expected_speed_at_sloop_curvature(self):
        """At typical S-loop measured curvature (0.010), speed should be ~8.9 m/s."""
        gov = _make_governor(
            max_lat_accel_g=0.20, curvature_calibration_scale=2.5,
            planner_enabled=False, horizon_enabled=False, preview_enabled=False,
        )
        out = gov.compute_target_speed(20.0, 0.010, None, 20.0, 5.0, 0.0)
        expected = math.sqrt(0.20 * G / (0.010 * 2.5))
        assert abs(out.comfort_speed - expected) < 0.1
        assert 8.0 < out.comfort_speed < 10.0, f"Expected ~8.9, got {out.comfort_speed}"
