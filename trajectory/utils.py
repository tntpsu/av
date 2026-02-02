from __future__ import annotations

import math
from typing import Iterable


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
