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
    max_jerk_min: float = 0.0
    max_jerk_max: float = 0.0
    jerk_error_min: float = 0.0
    jerk_error_max: float = 0.0
    min_speed: float = 0.0  # m/s
    reset_gap_seconds: float = 0.5
    sync_speed_threshold: float = 3.0
    default_dt: float = 1.0 / 30.0
    launch_speed_floor: float = 0.0
    launch_speed_floor_threshold: float = 0.0
    speed_error_gain: float = 0.0
    speed_error_max_delta: float = 0.0
    speed_error_deadband: float = 0.0
    sync_under_target: bool = True
    sync_under_target_error: float = 0.2
    use_speed_dependent_limits: bool = False
    accel_speed_min: float = 0.0
    accel_speed_max: float = 0.0
    max_accel_at_speed_min: float = 0.0
    max_accel_at_speed_max: float = 0.0
    decel_speed_min: float = 0.0
    decel_speed_max: float = 0.0
    max_decel_at_speed_min: float = 0.0
    max_decel_at_speed_max: float = 0.0
    enforce_desired_speed_slew: bool = False


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
        if current_speed is not None and self.config.speed_error_gain > 0.0:
            speed_error = desired_speed - float(current_speed)
            if abs(speed_error) < self.config.speed_error_deadband:
                speed_error = 0.0
            speed_adjust = self.config.speed_error_gain * speed_error
            if self.config.speed_error_max_delta > 0.0:
                max_delta = self.config.speed_error_max_delta
                speed_adjust = max(-max_delta, min(max_delta, speed_adjust))
            desired_speed += speed_adjust
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

        if (
            self.config.sync_under_target
            and current_speed is not None
            and desired_speed > float(current_speed) + self.config.sync_under_target_error
            and self._last_speed is not None
            and self._last_speed > float(current_speed)
        ):
            self._last_speed = float(current_speed)
            if self._last_accel < 0.0:
                self._last_accel = 0.0

        max_accel = self.config.max_accel
        max_decel = self.config.max_decel
        if self.config.use_speed_dependent_limits:
            ref_speed = (
                float(current_speed)
                if current_speed is not None
                else float(self._last_speed)
                if self._last_speed is not None
                else float(desired_speed)
            )
            max_accel = self._interpolate_limit(
                ref_speed,
                self.config.accel_speed_min,
                self.config.accel_speed_max,
                self.config.max_accel_at_speed_min,
                self.config.max_accel_at_speed_max,
                max_accel,
            )
            max_decel = self._interpolate_limit(
                ref_speed,
                self.config.decel_speed_min,
                self.config.decel_speed_max,
                self.config.max_decel_at_speed_min,
                self.config.max_decel_at_speed_max,
                max_decel,
            )
        if self.config.enforce_desired_speed_slew and self._last_speed is not None:
            max_slew = max_accel * max(dt, 1e-3)
            if desired_speed > self._last_speed + max_slew:
                desired_speed = self._last_speed + max_slew

        speed_error = desired_speed - self._last_speed
        target_accel = speed_error / max(dt, 1e-3)
        target_accel = max(-max_decel, min(max_accel, target_accel))

        max_jerk = self.config.max_jerk
        if (
            self.config.max_jerk_max > self.config.max_jerk_min
            and self.config.jerk_error_max > self.config.jerk_error_min
        ):
            ref_speed = float(current_speed) if current_speed is not None else float(self._last_speed)
            speed_error_mag = abs(desired_speed - ref_speed)
            jerk_ratio = min(
                1.0,
                max(
                    0.0,
                    (speed_error_mag - self.config.jerk_error_min)
                    / (self.config.jerk_error_max - self.config.jerk_error_min),
                ),
            )
            max_jerk = self.config.max_jerk_min + jerk_ratio * (
                self.config.max_jerk_max - self.config.max_jerk_min
            )
        max_accel_delta = max_jerk * dt
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
            planned_accel = min(planned_accel, max_accel)

        if planned_speed < self.config.min_speed:
            planned_speed = self.config.min_speed
            planned_accel = 0.0

        planned_jerk = (planned_accel - self._last_accel) / max(dt, 1e-3)
        planner_active = abs(desired_speed - planned_speed) > 1e-3 or abs(planned_accel) > 1e-3

        self._last_speed = planned_speed
        self._last_accel = planned_accel
        self._last_time = timestamp

        return planned_speed, planned_accel, planned_jerk, planner_active

    @staticmethod
    def _interpolate_limit(
        speed: float,
        speed_min: float,
        speed_max: float,
        limit_min: float,
        limit_max: float,
        fallback: float,
    ) -> float:
        if speed_max <= speed_min or limit_min <= 0.0 or limit_max <= 0.0:
            return float(fallback)
        if speed <= speed_min:
            return float(limit_min)
        if speed >= speed_max:
            return float(limit_max)
        ratio = (speed - speed_min) / (speed_max - speed_min)
        return float(limit_min + ratio * (limit_max - limit_min))
