import math
import numpy as np
from typing import Optional, Tuple, Dict, Any

from sync_contract import (
    TELEPORT_GUARD_REASON_CONTINUITY_SUPPRESSED,
    TELEPORT_GUARD_REASON_EXPECTED_MOTION_CONSISTENT,
    TELEPORT_GUARD_REASON_HARD_OVERRIDE,
    TELEPORT_GUARD_REASON_NONE,
    TELEPORT_GUARD_REASON_TRUE_DISCONTINUITY,
)


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


def _compute_motion_continuity_metrics(
    previous_position: Optional[np.ndarray],
    current_position: Optional[np.ndarray],
    speed_mps: float,
    dt_s: float,
) -> Tuple[float, float]:
    """Return observed motion distance and observed/expected ratio for continuity debugging."""
    observed_distance = 0.0
    if previous_position is not None and current_position is not None:
        observed_distance = float(np.linalg.norm(current_position - previous_position))
    expected_distance = 0.0
    if isinstance(speed_mps, (int, float)) and math.isfinite(float(speed_mps)) and dt_s > 0.0:
        expected_distance = max(0.0, float(speed_mps) * float(dt_s))
    ratio = float("nan")
    if expected_distance > 1e-6:
        ratio = observed_distance / expected_distance
    return expected_distance, ratio


def _classify_teleport_discontinuity(
    previous_position: Optional[np.ndarray],
    current_position: Optional[np.ndarray],
    *,
    base_distance_threshold_m: float,
    speed_mps: float,
    control_dt_s: float,
    unity_dt_s: float,
    skipped_unity_frames: int,
    fallback_active: bool,
    payload_selected_fresh: Optional[bool],
    payload_selected_age_ms: Optional[float],
    expected_motion_ratio_threshold: float,
    expected_motion_margin_m: float,
    hard_override_distance_m: float,
    hard_override_ratio_threshold: float,
    continuity_skip_frames_threshold: int,
    continuity_warn_age_ms: float,
) -> Dict[str, Any]:
    """Classify large pose deltas as either true discontinuities or transport-backed continuity gaps."""
    observed_distance = 0.0
    if previous_position is not None and current_position is not None:
        observed_distance = float(np.linalg.norm(current_position - previous_position))

    effective_dt_s = 0.0
    for candidate in (unity_dt_s, control_dt_s):
        if isinstance(candidate, (int, float)) and math.isfinite(float(candidate)) and float(candidate) > 0.0:
            effective_dt_s = max(effective_dt_s, float(candidate))

    expected_motion_m = 0.0
    if isinstance(speed_mps, (int, float)) and math.isfinite(float(speed_mps)) and effective_dt_s > 0.0:
        expected_motion_m = max(0.0, float(speed_mps) * effective_dt_s)

    motion_ratio = float("nan")
    if expected_motion_m > 1e-6:
        motion_ratio = observed_distance / expected_motion_m

    dynamic_threshold_m = float(base_distance_threshold_m)
    if expected_motion_m > 0.0:
        dynamic_threshold_m = max(
            float(base_distance_threshold_m),
            expected_motion_m + max(0.0, float(expected_motion_margin_m)),
            expected_motion_m * max(1.0, float(expected_motion_ratio_threshold)),
        )

    hard_override_threshold_m_effective = max(
        float(base_distance_threshold_m),
        float(hard_override_distance_m),
    )
    if expected_motion_m > 0.0:
        hard_override_threshold_m_effective = max(
            hard_override_threshold_m_effective,
            expected_motion_m * max(1.0, float(hard_override_ratio_threshold)),
        )

    continuity_suspect = False
    if int(skipped_unity_frames or 0) >= int(max(0, continuity_skip_frames_threshold)):
        continuity_suspect = True
    if bool(fallback_active):
        continuity_suspect = True
    if payload_selected_fresh is False:
        continuity_suspect = True
    if (
        isinstance(payload_selected_age_ms, (int, float))
        and math.isfinite(float(payload_selected_age_ms))
        and float(payload_selected_age_ms) > max(0.0, float(continuity_warn_age_ms))
    ):
        continuity_suspect = True

    base_threshold_exceeded = observed_distance > float(base_distance_threshold_m)
    dynamic_threshold_exceeded = observed_distance > dynamic_threshold_m
    hard_override = observed_distance > hard_override_threshold_m_effective

    teleport_detected = False
    guard_suppressed = False
    reason_code = TELEPORT_GUARD_REASON_NONE
    if not base_threshold_exceeded:
        reason_code = TELEPORT_GUARD_REASON_NONE
    elif hard_override:
        teleport_detected = True
        reason_code = TELEPORT_GUARD_REASON_HARD_OVERRIDE
    elif dynamic_threshold_exceeded and not continuity_suspect:
        teleport_detected = True
        reason_code = TELEPORT_GUARD_REASON_TRUE_DISCONTINUITY
    elif continuity_suspect:
        guard_suppressed = True
        reason_code = TELEPORT_GUARD_REASON_CONTINUITY_SUPPRESSED
    else:
        reason_code = TELEPORT_GUARD_REASON_EXPECTED_MOTION_CONSISTENT

    return {
        "teleport_detected": bool(teleport_detected),
        "guard_suppressed": bool(guard_suppressed),
        "reason_code": str(reason_code),
        "observed_distance_m": float(observed_distance),
        "expected_motion_m": float(expected_motion_m),
        "motion_ratio": float(motion_ratio),
        "base_threshold_exceeded": bool(base_threshold_exceeded),
        "dynamic_threshold_m": float(dynamic_threshold_m),
        "hard_override_threshold_m": float(hard_override_threshold_m_effective),
        "continuity_suspect": bool(continuity_suspect),
        "effective_dt_s": float(effective_dt_s),
        "unity_dt_s": float(unity_dt_s) if isinstance(unity_dt_s, (int, float)) and math.isfinite(float(unity_dt_s)) else float("nan"),
    }
