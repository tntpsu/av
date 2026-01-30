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
