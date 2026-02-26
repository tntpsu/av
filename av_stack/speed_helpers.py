import math
import numpy as np
from typing import Optional, Tuple


def _slew_limit_value(previous: float, target: float, max_rate: float, dt: float) -> float:
    """Slew-limit a value by max_rate per second."""
    if max_rate <= 0.0 or dt <= 0.0:
        return float(target)
    max_delta = max_rate * dt
    return float(previous + np.clip(target - previous, -max_delta, max_delta))


def _apply_speed_limit_preview(
    base_speed: float,
    preview_limit: float,
    preview_distance: float,
    max_decel: float,
) -> float:
    """Return the preview-based cap (independent of current speed)."""
    if preview_limit <= 0.0 or preview_distance <= 0.0 or max_decel <= 0.0:
        return float(base_speed)
    max_allowed = math.sqrt(max((preview_limit ** 2) + (2.0 * max_decel * preview_distance), 0.0))
    return float(min(base_speed, max_allowed))


def _apply_curve_speed_preview(
    base_speed: float,
    curvature: float,
    max_lateral_accel: float,
    min_curve_speed: float,
    preview_distance: float,
    max_decel: float,
) -> Tuple[float, Optional[float]]:
    """Preview-based cap using curvature lookahead (reduces speed before the curve)."""
    if (
        not isinstance(curvature, (int, float))
        or abs(curvature) <= 1e-6
        or max_lateral_accel <= 0.0
        or preview_distance <= 0.0
        or max_decel <= 0.0
    ):
        return float(base_speed), None
    curve_speed = (float(max_lateral_accel) / abs(curvature)) ** 0.5
    if min_curve_speed > 0.0:
        curve_speed = max(curve_speed, float(min_curve_speed))
    capped = _apply_speed_limit_preview(
        float(base_speed),
        float(curve_speed),
        float(preview_distance),
        float(max_decel),
    )
    return float(capped), float(curve_speed)


def _preview_min_distance_allows_release(
    preview_distance: float | None,
    preview_min_distance: float | None,
    margin: float,
) -> bool:
    """Return True if the preview min-distance suggests we can release the clamp."""
    if not isinstance(preview_distance, (int, float)) or preview_distance <= 0.0:
        return True
    if not isinstance(preview_min_distance, (int, float)) or preview_min_distance <= 0.0:
        return True
    return float(preview_min_distance) <= float(margin)


def _apply_target_speed_slew(previous: float, target: float, rate_up: float,
                             rate_down: float, dt: float) -> float:
    """Slew-limit target speed with asymmetric up/down rates."""
    if dt <= 0.0:
        return float(target)
    adjusted = float(target)
    if rate_up > 0.0 and adjusted > previous:
        adjusted = min(adjusted, previous + (rate_up * dt))
    if rate_down > 0.0 and adjusted < previous:
        adjusted = max(adjusted, previous - (rate_down * dt))
    return float(adjusted)


def _apply_restart_ramp(desired_speed: float, current_speed: float,
                        ramp_start_time: Optional[float], timestamp: float,
                        ramp_seconds: float, stop_threshold: float) -> Tuple[float, Optional[float], bool]:
    """Ramp target speed up after coming to a stop."""
    if ramp_seconds <= 0.0:
        return float(desired_speed), None, False
    if current_speed <= stop_threshold and desired_speed > 0.0:
        if ramp_start_time is None:
            ramp_start_time = float(timestamp)
        elapsed = max(0.0, float(timestamp) - float(ramp_start_time))
        ramp_limit = float(desired_speed) * min(1.0, elapsed / ramp_seconds)
        return float(min(desired_speed, ramp_limit)), ramp_start_time, True
    return float(desired_speed), None, False


def _apply_steering_speed_guard(
    desired_speed: float,
    last_steering: Optional[float],
    threshold: float,
    scale: float,
    min_speed: float,
) -> Tuple[float, bool]:
    """Reduce target speed when steering is saturated to prevent overshoot."""
    if last_steering is None or threshold <= 0.0 or scale <= 0.0:
        return float(desired_speed), False
    if abs(float(last_steering)) < threshold:
        return float(desired_speed), False
    capped = max(float(min_speed), float(desired_speed) * float(scale))
    if capped >= desired_speed:
        return float(desired_speed), False
    return float(capped), True


def _is_teleport_jump(
    previous_position: Optional[np.ndarray],
    current_position: Optional[np.ndarray],
    distance_threshold: float,
) -> Tuple[bool, float]:
    """Detect sudden position jumps that indicate Unity teleport/discontinuity."""
    if previous_position is None or current_position is None:
        return False, 0.0
    distance = float(np.linalg.norm(current_position - previous_position))
    return distance > distance_threshold, distance
