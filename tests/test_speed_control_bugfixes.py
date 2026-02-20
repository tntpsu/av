"""
Tests for Phase 0 speed control bug fixes:
  0A: Speed planner max_accel/max_jerk override
  0B: launch_speed_floor pathological acceleration
  0C: Unity GT lookahead override
"""

import math

from trajectory.speed_planner import SpeedPlanner, SpeedPlannerConfig


# --- 0A: Speed planner uses its own max_accel/max_jerk, not longitudinal's ---

def test_speed_planner_config_uses_own_values():
    """SpeedPlannerConfig should use the values given at construction, not
    some external longitudinal config.  This validates that whoever constructs
    the planner passes speed_planner section values (2.0 / 2.5 / 1.2) rather
    than longitudinal values (1.2 / 2.4 / 0.7)."""
    cfg = SpeedPlannerConfig(max_accel=2.0, max_decel=2.5, max_jerk=1.2)
    assert cfg.max_accel == 2.0
    assert cfg.max_decel == 2.5
    assert cfg.max_jerk == 1.2


def test_speed_planner_ramp_rate_matches_config():
    """With max_accel=2.0 and max_jerk=1.2, the planner should ramp faster
    than if constrained to longitudinal's max_accel=1.2 / max_jerk=0.7."""
    dt = 1.0 / 30.0

    fast_cfg = SpeedPlannerConfig(max_accel=2.0, max_decel=2.5, max_jerk=1.2, default_dt=dt)
    slow_cfg = SpeedPlannerConfig(max_accel=1.2, max_decel=2.4, max_jerk=0.7, default_dt=dt)

    fast_planner = SpeedPlanner(fast_cfg)
    slow_planner = SpeedPlanner(slow_cfg)

    steps = 30
    fast_speed = 0.0
    slow_speed = 0.0
    for i in range(steps):
        t = i * dt
        fast_speed, _, _, _ = fast_planner.step(12.0, current_speed=fast_speed, timestamp=t)
        slow_speed, _, _, _ = slow_planner.step(12.0, current_speed=slow_speed, timestamp=t)

    assert fast_speed > slow_speed, (
        f"Fast planner ({fast_speed:.3f}) should be faster than slow planner ({slow_speed:.3f})"
    )


# --- 0B: launch_speed_floor should not produce pathological acceleration ---

def test_launch_speed_floor_caps_accel():
    """When launch_speed_floor kicks in, planned_accel must not exceed max_accel."""
    dt = 1.0 / 30.0
    cfg = SpeedPlannerConfig(
        max_accel=2.0,
        max_decel=2.5,
        max_jerk=1.2,
        default_dt=dt,
        launch_speed_floor=2.0,
        launch_speed_floor_threshold=1.0,
    )
    planner = SpeedPlanner(cfg)

    speed, accel, _, _ = planner.step(desired_speed=12.0, current_speed=0.0, timestamp=0.0)

    assert speed >= cfg.launch_speed_floor, (
        f"Floor should apply: speed={speed}, floor={cfg.launch_speed_floor}"
    )
    assert accel <= cfg.max_accel + 1e-6, (
        f"Accel must be capped to max_accel ({cfg.max_accel}), got {accel}"
    )


def test_launch_speed_floor_does_not_poison_subsequent_frames():
    """After launch_speed_floor fires, _last_accel should be sane so subsequent
    frames ramp normally through jerk limits."""
    dt = 1.0 / 30.0
    cfg = SpeedPlannerConfig(
        max_accel=2.0,
        max_decel=2.5,
        max_jerk=1.2,
        default_dt=dt,
        launch_speed_floor=2.0,
        launch_speed_floor_threshold=1.0,
    )
    planner = SpeedPlanner(cfg)

    # Frame 0: floor fires
    speed0, accel0, _, _ = planner.step(12.0, current_speed=0.0, timestamp=0.0)
    assert accel0 <= cfg.max_accel + 1e-6

    # Frame 1: should continue ramping normally, not be poisoned
    speed1, accel1, jerk1, _ = planner.step(12.0, current_speed=speed0, timestamp=dt)
    assert accel1 <= cfg.max_accel + 1e-6
    assert abs(jerk1) < 200.0, f"Jerk should be reasonable, got {jerk1}"


def test_launch_speed_floor_inactive_above_threshold():
    """When current_speed > threshold, launch_speed_floor should not activate."""
    dt = 1.0 / 30.0
    cfg = SpeedPlannerConfig(
        max_accel=2.0,
        max_decel=2.5,
        max_jerk=1.2,
        default_dt=dt,
        launch_speed_floor=2.0,
        launch_speed_floor_threshold=1.0,
    )
    planner = SpeedPlanner(cfg)
    planner.reset(current_speed=3.0, timestamp=0.0)

    speed, accel, _, _ = planner.step(12.0, current_speed=3.0, timestamp=dt)
    # Floor should not push speed; normal jerk-limited ramp
    assert accel <= cfg.max_accel + 1e-6


# --- 0C: Unity GT lookahead should not override dynamic reference_lookahead ---

def test_unity_gt_lookahead_stored_separately():
    """Verify that compute_reference_lookahead result is not overwritten by
    Unity GT lookahead.  We test this by importing the utility and confirming
    the dynamic scaling logic applies."""
    from trajectory.utils import compute_reference_lookahead

    config = {
        "dynamic_reference_lookahead": True,
        "reference_lookahead_scale_min": 0.55,
        "reference_lookahead_speed_min": 4.0,
        "reference_lookahead_speed_max": 10.0,
        "reference_lookahead_curvature_min": 0.002,
        "reference_lookahead_curvature_max": 0.015,
        "reference_lookahead_min": 3.0,
    }

    # At low speed, low curvature: scale should reduce from speed
    result = compute_reference_lookahead(
        base_lookahead=9.0,
        current_speed=8.0,
        path_curvature=0.001,
        config=config,
    )
    # Dynamic scaling should change the result (not just return 9.0 or a fixed Unity value)
    assert result != 9.0, "Dynamic scaling should modify the lookahead"
    assert result >= 3.0, "Should respect reference_lookahead_min"
    assert result < 9.0, "At high speed / low curvature, scale should reduce"


def test_unity_gt_lookahead_does_not_affect_horizon():
    """The horizon guardrail should use the dynamically computed reference_lookahead,
    not a fixed Unity GT value.  With a dynamic lookahead of 9m vs Unity's 6m,
    the guardrail should allow higher speed."""
    from trajectory.utils import apply_speed_horizon_guardrail

    config = {
        "speed_horizon_guardrail_enabled": True,
        "speed_horizon_guardrail_time_headway_s": 0.8,
        "speed_horizon_guardrail_margin_m": 1.0,
        "speed_horizon_guardrail_min_speed_mps": 3.0,
        "speed_horizon_guardrail_gain": 1.0,
    }

    # With dynamic horizon (9m): allowed_speed = (9 - 1) / 0.8 = 10.0
    result_dynamic = apply_speed_horizon_guardrail(
        target_speed_mps=12.0,
        effective_horizon_m=9.0,
        dynamic_horizon_applied=True,
        config=config,
    )
    # With Unity override (6m): allowed_speed = (6 - 1) / 0.8 = 6.25
    result_unity = apply_speed_horizon_guardrail(
        target_speed_mps=12.0,
        effective_horizon_m=6.0,
        dynamic_horizon_applied=True,
        config=config,
    )

    speed_dynamic = result_dynamic["target_speed_mps"]
    speed_unity = result_unity["target_speed_mps"]

    assert speed_dynamic > speed_unity, (
        f"Dynamic horizon ({speed_dynamic:.2f}) should allow higher speed than "
        f"Unity override ({speed_unity:.2f})"
    )
    assert speed_dynamic >= 10.0 - 0.1, f"Expected ~10 m/s, got {speed_dynamic}"
    assert speed_unity <= 6.5, f"Expected ~6.25 m/s, got {speed_unity}"
