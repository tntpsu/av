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
    cap_tracking_enabled: bool = False
    cap_tracking_error_on_mps: float = 0.35
    cap_tracking_error_off_mps: float = 0.12
    cap_tracking_hold_frames: int = 4
    cap_tracking_decel_gain: float = 2.0
    cap_tracking_jerk_gain: float = 1.2
    cap_tracking_max_decel_mps2: float = 3.4
    cap_tracking_max_jerk_mps3: float = 2.8
    cap_tracking_hard_ceiling_epsilon_mps: float = 0.05


class SpeedPlanner:
    """Simple jerk-limited speed planner."""

    def __init__(self, config: SpeedPlannerConfig) -> None:
        self.config = config
        self._last_speed: Optional[float] = None
        self._last_accel: float = 0.0
        self._last_time: Optional[float] = None
        self._cap_tracking_mode: str = "inactive"
        self._cap_tracking_hold_remaining: int = 0
        self._cap_tracking_recovery_frames: int = 0
        self._last_cap_active: bool = False
        self.last_cap_tracking_active: bool = False
        self.last_cap_tracking_error_mps: float = 0.0
        self.last_cap_tracking_mode: str = "inactive"
        self.last_cap_tracking_recovery_frames: int = 0

    def reset(self, current_speed: Optional[float] = None, timestamp: Optional[float] = None) -> None:
        """Reset planner state."""
        self._last_speed = current_speed
        self._last_accel = 0.0
        self._last_time = timestamp
        self._cap_tracking_mode = "inactive"
        self._cap_tracking_hold_remaining = 0
        self._cap_tracking_recovery_frames = 0
        self._last_cap_active = False
        self.last_cap_tracking_active = False
        self.last_cap_tracking_error_mps = 0.0
        self.last_cap_tracking_mode = "inactive"
        self.last_cap_tracking_recovery_frames = 0

    def step(
        self,
        desired_speed: float,
        current_speed: Optional[float],
        timestamp: Optional[float],
        hard_upper_speed: Optional[float] = None,
        cap_active: bool = False,
        cap_reason: Optional[str] = None,
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

        _ = cap_reason

        cap_limit: Optional[float] = None
        cap_limit_valid = (
            hard_upper_speed is not None
            and float(hard_upper_speed) >= 0.0
        )
        if cap_limit_valid:
            cap_limit = float(max(self.config.min_speed, float(hard_upper_speed)))
            if cap_active:
                desired_speed = min(desired_speed, cap_limit)

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

        prev_cap_mode = self._cap_tracking_mode
        prev_cap_active = self._last_cap_active
        cap_error_pre = 0.0
        cap_tracking_enabled_now = bool(
            self.config.cap_tracking_enabled and cap_active and cap_limit is not None
        )
        if cap_tracking_enabled_now and self._last_speed is not None and cap_limit is not None:
            cap_error_pre = max(0.0, float(self._last_speed) - float(cap_limit))
            on = max(0.0, float(self.config.cap_tracking_error_on_mps))
            off = max(0.0, float(self.config.cap_tracking_error_off_mps))
            if self._cap_tracking_mode == "inactive":
                if cap_error_pre >= on:
                    self._cap_tracking_mode = "catch_up"
                    self._cap_tracking_recovery_frames = 0
            elif self._cap_tracking_mode == "catch_up":
                if cap_error_pre <= off:
                    self._cap_tracking_mode = "hold"
                    self._cap_tracking_hold_remaining = max(0, int(self.config.cap_tracking_hold_frames))
                else:
                    self._cap_tracking_recovery_frames += 1
            elif self._cap_tracking_mode == "hold":
                if cap_error_pre >= on:
                    self._cap_tracking_mode = "catch_up"
                    self._cap_tracking_recovery_frames = 0
                elif self._cap_tracking_hold_remaining > 0:
                    self._cap_tracking_hold_remaining -= 1
                else:
                    self._cap_tracking_mode = "inactive"
                    self._cap_tracking_recovery_frames = 0
        else:
            self._cap_tracking_mode = "inactive"
            self._cap_tracking_hold_remaining = 0
            self._cap_tracking_recovery_frames = 0

        # Accel memory reset on cap-tracking exit: prevents negative accel
        # accumulated during catch_up from leaking into the recovery phase.
        # Only fires when transitioning to inactive AND we need to accelerate.
        if prev_cap_mode in {"catch_up", "hold"} and self._cap_tracking_mode == "inactive":
            if self._last_speed is not None and desired_speed > self._last_speed:
                self._last_accel = max(0.0, self._last_accel)

        # Sub-threshold cap exit clamp: brief cap events (vehicle error below the
        # cap_tracking activation threshold) never enter catch_up/hold but the
        # governor's pre-planner min(target, cap_speed) still contaminates _last_accel
        # with a large negative value.  When cap_active flips True→False and we need
        # to accelerate, clamp the accel memory so the planner doesn't keep
        # decelerating from momentum built during the cap period.
        if prev_cap_active and not cap_active:
            if self._last_speed is not None and desired_speed > self._last_speed:
                self._last_accel = max(0.0, self._last_accel)

        self._last_cap_active = cap_active

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
        if self._cap_tracking_mode == "catch_up":
            max_decel = max(max_decel, float(self.config.cap_tracking_max_decel_mps2))

        if self.config.enforce_desired_speed_slew and self._last_speed is not None:
            max_slew = max_accel * max(dt, 1e-3)
            if desired_speed > self._last_speed + max_slew:
                desired_speed = self._last_speed + max_slew

        speed_error = desired_speed - self._last_speed
        target_accel = speed_error / max(dt, 1e-3)
        target_accel = max(-max_decel, min(max_accel, target_accel))

        if cap_tracking_enabled_now and cap_limit is not None:
            cap_error_for_accel = max(0.0, float(self._last_speed) - float(cap_limit))
            if self._cap_tracking_mode == "catch_up":
                desired_decel = min(
                    float(self.config.cap_tracking_max_decel_mps2),
                    float(self.config.cap_tracking_decel_gain) * cap_error_for_accel,
                )
                target_accel = min(target_accel, -desired_decel)
            if cap_error_for_accel > float(self.config.cap_tracking_error_off_mps):
                target_accel = min(target_accel, 0.0)

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
        if cap_tracking_enabled_now and cap_limit is not None and self._cap_tracking_mode == "catch_up":
            cap_error_for_jerk = max(0.0, float(self._last_speed) - float(cap_limit))
            cap_jerk = min(
                float(self.config.cap_tracking_max_jerk_mps3),
                float(max_jerk) + float(self.config.cap_tracking_jerk_gain) * cap_error_for_jerk,
            )
            max_jerk = max(float(max_jerk), float(cap_jerk))
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

        if cap_tracking_enabled_now and cap_limit is not None:
            cap_ceiling = float(cap_limit) + max(0.0, float(self.config.cap_tracking_hard_ceiling_epsilon_mps))
            if planned_speed > cap_ceiling:
                planned_speed = cap_ceiling
                # The ceiling is an external position clamp, not a commanded decel.
                # Recomputing planned_accel as (ceiling - _last_speed)/dt poisons _last_accel
                # with a large negative value (can be -7+ m/s²) that takes 100+ frames to
                # recover via jerk limiting, causing a speed dip to near-zero.
                # Instead: preserve the pre-clip accel direction clamped to ≤ 0 so the
                # planner starts the next frame from rest (0) rather than a large decel.
                planned_accel = min(planned_accel, 0.0)
            cap_error_post = max(0.0, planned_speed - float(cap_limit))
            if (
                cap_error_post > float(self.config.cap_tracking_error_off_mps)
                and planned_accel > 0.0
            ):
                planned_speed = min(planned_speed, float(self._last_speed))
                planned_accel = 0.0

        planned_jerk = (planned_accel - self._last_accel) / max(dt, 1e-3)
        planner_active = abs(desired_speed - planned_speed) > 1e-3 or abs(planned_accel) > 1e-3

        self.last_cap_tracking_active = bool(
            cap_tracking_enabled_now and self._cap_tracking_mode in {"catch_up", "hold"}
        )
        if cap_limit is not None:
            self.last_cap_tracking_error_mps = float(max(0.0, planned_speed - cap_limit))
        else:
            self.last_cap_tracking_error_mps = 0.0
        self.last_cap_tracking_mode = str(self._cap_tracking_mode)
        self.last_cap_tracking_recovery_frames = (
            int(self._cap_tracking_recovery_frames) if self.last_cap_tracking_active else 0
        )

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
