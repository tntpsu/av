from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence, Tuple


def compute_reference_lookahead(
    base_lookahead: float,
    current_speed: float,
    path_curvature: float,
    config: dict,
) -> float:
    """
    Compute a dynamic reference lookahead based on speed and curvature.

    Shorter lookahead increases responsiveness on gentle curves at higher speeds,
    while preserving the base lookahead on sharper curves and low speeds.
    """
    if not config.get("dynamic_reference_lookahead", False):
        return base_lookahead

    scale_min = float(config.get("reference_lookahead_scale_min", 0.7))
    scale_min = max(0.1, min(1.0, scale_min))
    scale = 1.0

    speed_min = float(config.get("reference_lookahead_speed_min", 4.0))
    speed_max = float(config.get("reference_lookahead_speed_max", 10.0))
    if speed_max > speed_min:
        if current_speed <= speed_min:
            speed_scale = 1.0
        elif current_speed >= speed_max:
            speed_scale = scale_min
        else:
            ratio = (current_speed - speed_min) / (speed_max - speed_min)
            speed_scale = 1.0 - ratio * (1.0 - scale_min)
        scale = min(scale, speed_scale)

    curv_min = float(config.get("reference_lookahead_curvature_min", 0.002))
    curv_max = float(config.get("reference_lookahead_curvature_max", 0.015))
    abs_curv = abs(path_curvature)
    if curv_max > curv_min:
        if abs_curv <= curv_min:
            curv_scale = scale_min
        elif abs_curv >= curv_max:
            curv_scale = 1.0
        else:
            ratio = (abs_curv - curv_min) / (curv_max - curv_min)
            curv_scale = scale_min + ratio * (1.0 - scale_min)
        scale = min(scale, curv_scale)

    tight_curv_threshold = float(config.get("reference_lookahead_tight_curvature_threshold", 0.0))
    tight_scale = float(config.get("reference_lookahead_tight_scale", 1.0))
    if tight_curv_threshold > 0.0 and abs_curv >= tight_curv_threshold:
        tight_scale = max(0.1, min(1.0, tight_scale))
        scale = min(scale, tight_scale)

    min_lookahead = float(config.get("reference_lookahead_min", 4.0))
    min_lookahead = max(0.1, min_lookahead)
    return max(min_lookahead, base_lookahead * scale)


def curvature_smoothing_alpha(distance_m: float, window_m: float) -> float:
    """Compute EMA alpha for a distance-based smoothing window."""
    if window_m <= 0.0:
        return 1.0
    distance_m = max(0.0, float(distance_m))
    return float(1.0 - math.exp(-distance_m / window_m))


def smooth_curvature_distance(
    curvature: Iterable[float],
    speed: Iterable[float],
    timestamps: Iterable[float],
    window_m: float,
    min_speed: float = 0.0,
) -> list[float]:
    """Smooth curvature using a distance-based EMA over a spatial window."""
    curvatures = list(curvature)
    speeds = list(speed)
    times = list(timestamps)
    if not curvatures or len(curvatures) != len(speeds) or len(curvatures) != len(times):
        return curvatures
    if window_m <= 0.0:
        return curvatures
    min_speed = max(0.0, float(min_speed))
    smoothed = [float(curvatures[0])]
    for i in range(1, len(curvatures)):
        dt = float(times[i]) - float(times[i - 1])
        if dt <= 0.0:
            dt = 1e-3
        distance = max(float(speeds[i]), min_speed) * dt
        alpha = curvature_smoothing_alpha(distance, window_m)
        smoothed_value = alpha * float(curvatures[i]) + (1.0 - alpha) * smoothed[-1]
        smoothed.append(smoothed_value)
    return smoothed


def select_curvature_bin_limits(
    abs_curvature: float,
    bins: Sequence[Mapping[str, float]] | None,
    default_max_lateral_accel: float,
    default_min_curve_speed: float,
) -> Tuple[float, float]:
    """Select max lateral accel and min curve speed based on curvature bins."""
    max_lateral_accel = float(default_max_lateral_accel)
    min_curve_speed = float(default_min_curve_speed)
    if not bins:
        return max_lateral_accel, min_curve_speed
    if not isinstance(abs_curvature, (int, float)):
        return max_lateral_accel, min_curve_speed

    curvature_val = float(abs_curvature)
    for bin_cfg in bins:
        if not isinstance(bin_cfg, Mapping):
            continue
        min_curv = float(bin_cfg.get("min_curvature", 0.0))
        max_curv = float(bin_cfg.get("max_curvature", float("inf")))
        if curvature_val < min_curv or curvature_val >= max_curv:
            continue
        if "max_lateral_accel" in bin_cfg:
            max_lateral_accel = float(bin_cfg["max_lateral_accel"])
        if "min_curve_speed" in bin_cfg:
            min_curve_speed = float(bin_cfg["min_curve_speed"])
        break
    return max_lateral_accel, min_curve_speed


def compute_dynamic_effective_horizon(
    base_horizon_m: float,
    current_speed_mps: float,
    path_curvature: float,
    confidence: float | None,
    config: Mapping[str, object],
) -> dict[str, float]:
    """
    Compute diagnostics for a dynamic effective horizon policy.

    Phase 1 is diagnostics-only: this function reports what the policy would do,
    but does not change downstream behavior unless a future phase explicitly applies it.
    """
    base = max(0.1, float(base_horizon_m))
    min_m = max(0.1, float(config.get("dynamic_effective_horizon_min_m", 6.0)))
    max_m = max(min_m, float(config.get("dynamic_effective_horizon_max_m", base)))

    speed_min = float(config.get("reference_lookahead_speed_min", 4.0))
    speed_max = float(config.get("reference_lookahead_speed_max", 10.0))
    curv_min = float(config.get("reference_lookahead_curvature_min", 0.002))
    curv_max = float(config.get("reference_lookahead_curvature_max", 0.015))

    speed_gain = max(0.0, min(1.0, float(config.get("dynamic_effective_horizon_speed_scale", 0.0))))
    curvature_gain = max(
        0.0,
        min(1.0, float(config.get("dynamic_effective_horizon_curvature_scale", 0.0))),
    )
    confidence_floor = max(
        0.1,
        min(1.0, float(config.get("dynamic_effective_horizon_confidence_floor", 0.6))),
    )
    confidence_fallback = max(
        0.1,
        min(1.0, float(config.get("dynamic_effective_horizon_confidence_fallback", 1.0))),
    )

    speed_norm = 0.0
    if speed_max > speed_min:
        speed_norm = (float(current_speed_mps) - speed_min) / (speed_max - speed_min)
        speed_norm = max(0.0, min(1.0, speed_norm))
    speed_scale = 1.0 - (speed_gain * speed_norm)
    speed_scale = max(0.1, min(1.0, speed_scale))

    abs_curv = abs(float(path_curvature))
    curvature_norm = 0.0
    if curv_max > curv_min:
        curvature_norm = (abs_curv - curv_min) / (curv_max - curv_min)
        curvature_norm = max(0.0, min(1.0, curvature_norm))
    curvature_scale = 1.0 - (curvature_gain * curvature_norm)
    curvature_scale = max(0.1, min(1.0, curvature_scale))

    conf_used = (
        confidence_fallback
        if confidence is None or not math.isfinite(float(confidence))
        else max(0.0, min(1.0, float(confidence)))
    )
    confidence_scale = max(confidence_floor, conf_used)
    confidence_scale = max(0.1, min(1.0, confidence_scale))

    final_scale = min(speed_scale, curvature_scale, confidence_scale)
    unclamped_horizon = base * final_scale
    effective_horizon = max(min_m, min(max_m, unclamped_horizon))

    # limiter_code: 0=none, 1=speed, 2=curvature, 3=confidence
    limiter_code = 0.0
    limiter_scale = final_scale
    if limiter_scale < 0.999:
        candidates = [
            (speed_scale, 1.0),
            (curvature_scale, 2.0),
            (confidence_scale, 3.0),
        ]
        limiter_code = min(candidates, key=lambda item: item[0])[1]

    # Phase 1 keeps this diagnostics-only.
    applied = 0.0
    if bool(config.get("dynamic_effective_horizon_enabled", False)):
        applied = 0.0

    return {
        "diag_dynamic_effective_horizon_m": float(effective_horizon),
        "diag_dynamic_effective_horizon_base_m": float(base),
        "diag_dynamic_effective_horizon_min_m": float(min_m),
        "diag_dynamic_effective_horizon_max_m": float(max_m),
        "diag_dynamic_effective_horizon_speed_scale": float(speed_scale),
        "diag_dynamic_effective_horizon_curvature_scale": float(curvature_scale),
        "diag_dynamic_effective_horizon_confidence_scale": float(confidence_scale),
        "diag_dynamic_effective_horizon_final_scale": float(final_scale),
        "diag_dynamic_effective_horizon_speed_mps": float(current_speed_mps),
        "diag_dynamic_effective_horizon_curvature_abs": float(abs_curv),
        "diag_dynamic_effective_horizon_confidence_used": float(conf_used),
        "diag_dynamic_effective_horizon_limiter_code": float(limiter_code),
        "diag_dynamic_effective_horizon_applied": float(applied),
    }
