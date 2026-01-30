"""
Jerk-limited speed planner for AV stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SpeedPlannerConfig:
    """Configuration for jerk-limited speed planning."""

    max_accel: float = 2.0  # m/s^2
    max_decel: float = 3.0  # m/s^2
    max_jerk: float = 2.0  # m/s^3
    min_speed: float = 0.0  # m/s
    reset_gap_seconds: float = 0.5
    sync_speed_threshold: float = 3.0
    default_dt: float = 1.0 / 30.0
    launch_speed_floor: float = 0.0
    launch_speed_floor_threshold: float = 0.0


class SpeedPlanner:
    """Simple jerk-limited speed planner."""

    def __init__(self, config: SpeedPlannerConfig) -> None:
        self.config = config
        self._last_speed: Optional[float] = None
        self._last_accel: float = 0.0
        self._last_time: Optional[float] = None

    def reset(self, current_speed: Optional[float] = None, timestamp: Optional[float] = None) -> None:
        """Reset planner state."""
        self._last_speed = current_speed
        self._last_accel = 0.0
        self._last_time = timestamp

    def step(
        self,
        desired_speed: float,
        current_speed: Optional[float],
        timestamp: Optional[float],
    ) -> Tuple[float, float, float, bool]:
        """Advance planner one step.

        Returns: (planned_speed, planned_accel, planned_jerk, planner_active)
        """
        desired_speed = float(desired_speed)
        if desired_speed < self.config.min_speed:
            desired_speed = self.config.min_speed

        if self._last_speed is None:
            self._last_speed = float(current_speed) if current_speed is not None else desired_speed
            self._last_time = timestamp

        dt = self.config.default_dt
        if timestamp is not None and self._last_time is not None:
            dt = float(timestamp) - float(self._last_time)
        if dt <= 0.0:
            dt = self.config.default_dt

        if dt > self.config.reset_gap_seconds:
            self.reset(current_speed=float(current_speed) if current_speed is not None else self._last_speed,
                       timestamp=timestamp)
            dt = self.config.default_dt

        if current_speed is not None and self._last_speed is not None:
            if abs(float(current_speed) - self._last_speed) > self.config.sync_speed_threshold:
                self._last_speed = float(current_speed)
                self._last_accel = 0.0

        speed_error = desired_speed - self._last_speed
        target_accel = speed_error / max(dt, 1e-3)
        target_accel = max(-self.config.max_decel, min(self.config.max_accel, target_accel))

        max_accel_delta = self.config.max_jerk * dt
        accel_delta = target_accel - self._last_accel
        accel_delta = max(-max_accel_delta, min(max_accel_delta, accel_delta))
        planned_accel = self._last_accel + accel_delta

        planned_speed = self._last_speed + planned_accel * dt
        if desired_speed >= self._last_speed and planned_speed > desired_speed:
            planned_speed = desired_speed
            planned_accel = 0.0
        elif desired_speed <= self._last_speed and planned_speed < desired_speed:
            planned_speed = desired_speed
            planned_accel = 0.0

        if (
            current_speed is not None
            and current_speed < self.config.launch_speed_floor_threshold
            and desired_speed > self.config.launch_speed_floor
        ):
            planned_speed = max(planned_speed, self.config.launch_speed_floor)
            planned_accel = (planned_speed - self._last_speed) / max(dt, 1e-3)

        if planned_speed < self.config.min_speed:
            planned_speed = self.config.min_speed
            planned_accel = 0.0

        planned_jerk = (planned_accel - self._last_accel) / max(dt, 1e-3)
        planner_active = abs(desired_speed - planned_speed) > 1e-3 or abs(planned_accel) > 1e-3

        self._last_speed = planned_speed
        self._last_accel = planned_accel
        self._last_time = timestamp

        return planned_speed, planned_accel, planned_jerk, planner_active
