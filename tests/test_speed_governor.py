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
    **overrides,
) -> SpeedGovernor:
    """Helper to build a SpeedGovernor with common defaults."""
    gov_cfg = SpeedGovernorConfig(
        comfort_governor_max_lat_accel_g=max_lat_accel_g,
        comfort_governor_min_speed=3.0,
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
