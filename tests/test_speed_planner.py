"""
Unit tests for jerk-limited speed planner.
"""

from trajectory.speed_planner import SpeedPlanner, SpeedPlannerConfig


def test_speed_planner_limits_jerk():
    cfg = SpeedPlannerConfig(max_accel=2.0, max_decel=2.0, max_jerk=1.0, default_dt=0.1)
    planner = SpeedPlanner(cfg)

    speed, accel, jerk, _ = planner.step(desired_speed=10.0, current_speed=0.0, timestamp=0.0)

    assert accel <= cfg.max_jerk * cfg.default_dt + 1e-6
    assert jerk <= cfg.max_jerk + 1e-6
    assert speed > 0.0


def test_speed_planner_limits_decel():
    cfg = SpeedPlannerConfig(max_accel=2.0, max_decel=1.5, max_jerk=5.0, default_dt=0.1)
    planner = SpeedPlanner(cfg)
    planner.reset(current_speed=5.0, timestamp=0.0)

    speed, accel, _, _ = planner.step(desired_speed=0.0, current_speed=5.0, timestamp=0.1)

    assert accel >= -cfg.max_decel - 1e-6
    assert speed <= 5.0


def test_speed_planner_speed_error_gain_applies_bias():
    cfg = SpeedPlannerConfig(
        max_accel=2.0,
        max_decel=2.0,
        max_jerk=5.0,
        default_dt=0.1,
        speed_error_gain=0.5,
        speed_error_max_delta=2.0,
        speed_error_deadband=0.0,
    )
    planner = SpeedPlanner(cfg)
    planner.reset(current_speed=5.0, timestamp=0.0)

    speed, accel, _, _ = planner.step(desired_speed=10.0, current_speed=5.0, timestamp=0.1)
    assert speed > 5.0
    assert accel > 0.0


def test_speed_planner_speed_error_deadband():
    cfg = SpeedPlannerConfig(
        max_accel=2.0,
        max_decel=2.0,
        max_jerk=5.0,
        default_dt=0.1,
        speed_error_gain=0.5,
        speed_error_max_delta=2.0,
        speed_error_deadband=0.5,
    )
    planner = SpeedPlanner(cfg)
    planner.reset(current_speed=10.0, timestamp=0.0)

    speed, _, _, _ = planner.step(desired_speed=10.2, current_speed=10.0, timestamp=0.1)
    assert speed <= 10.2


def test_speed_planner_sync_under_target():
    cfg = SpeedPlannerConfig(
        max_accel=2.0,
        max_decel=2.0,
        max_jerk=5.0,
        default_dt=0.1,
        sync_under_target=True,
        sync_under_target_error=0.2,
    )
    planner = SpeedPlanner(cfg)
    planner.reset(current_speed=10.0, timestamp=0.0)
    # Simulate desync: planner thinks speed is higher than actual.
    planner._last_speed = 12.0

    speed, accel, _, _ = planner.step(desired_speed=10.0, current_speed=5.0, timestamp=0.1)
    assert accel >= -1e-6


def test_speed_planner_speed_dependent_accel_limit():
    cfg = SpeedPlannerConfig(
        max_accel=3.0,
        max_decel=3.0,
        max_jerk=100.0,
        default_dt=0.1,
        use_speed_dependent_limits=True,
        accel_speed_min=0.0,
        accel_speed_max=10.0,
        max_accel_at_speed_min=2.0,
        max_accel_at_speed_max=1.0,
        decel_speed_min=0.0,
        decel_speed_max=10.0,
        max_decel_at_speed_min=3.0,
        max_decel_at_speed_max=3.0,
    )
    planner = SpeedPlanner(cfg)
    planner.reset(current_speed=10.0, timestamp=0.0)

    _, accel, _, _ = planner.step(desired_speed=20.0, current_speed=10.0, timestamp=0.1)
    assert accel <= 1.0 + 1e-6


def test_speed_planner_desired_speed_slew():
    cfg = SpeedPlannerConfig(
        max_accel=2.0,
        max_decel=2.0,
        max_jerk=100.0,
        default_dt=0.1,
        enforce_desired_speed_slew=True,
    )
    planner = SpeedPlanner(cfg)
    planner.reset(current_speed=0.0, timestamp=0.0)

    speed, accel, _, _ = planner.step(desired_speed=10.0, current_speed=0.0, timestamp=0.1)
    assert speed <= 0.25  # accel ~2.0, dt=0.1 => v <= 0.2-ish
    assert accel <= cfg.max_accel + 1e-6
