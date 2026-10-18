"""Steer-command interface: physical wheel angle (rad) → the plant's normalized command (T-STANLEY-FF-SPEED-MAP, 2026-10-17).

Unity applies a normalized command as `desiredSteerAngle = steerInput × Lerp(30°, 16°, v / 12 m/s)`
(CarController.cs, maxSteerAngleLowSpeed / maxSteerAngleHighSpeed / maxSteerAngleFullSpeed), then a 0.25 s actuator filter
and a 90°/s rate limit. A controller that normalizes its radians by a fixed 30° therefore delivers only
r(v) = 1 − 0.467·min(v/12, 1) of what it asked for (0.875 at 3.2 m/s, 0.79 at 5.3, 0.55 at 11.6). Verified on the
2026-10-17 recordings: steering_angle_actual / steering_input = 26.1° at 3 m/s (formula 26.5), 23.8 at 5.5 (23.6),
20.6 at 7 (21.8), 18.3 at 9 (19.5).

Preferred source is Unity's own per-frame value (`maxSteerAtSpeedDeg` in the vehicle state); `unity_max_steer_rad` is the
documented mirror used as a fallback and pinned by a recordings contract test.
"""
from __future__ import annotations

import math

UNITY_MAX_STEER_LOW_SPEED_DEG = 30.0
UNITY_MAX_STEER_HIGH_SPEED_DEG = 16.0
UNITY_MAX_STEER_FULL_SPEED_MPS = 12.0
MIN_MAX_STEER_RAD = 1e-3


def unity_max_steer_deg(speed_mps: float) -> float:
    """Mirror of CarController.cs: Lerp(30°, 16°, InverseLerp(0, 12 m/s, v))."""
    try:
        v = float(speed_mps)
    except (TypeError, ValueError):
        v = 0.0
    if not math.isfinite(v):
        v = 0.0
    t = min(max(v / UNITY_MAX_STEER_FULL_SPEED_MPS, 0.0), 1.0)
    return UNITY_MAX_STEER_LOW_SPEED_DEG + (UNITY_MAX_STEER_HIGH_SPEED_DEG - UNITY_MAX_STEER_LOW_SPEED_DEG) * t


def unity_max_steer_rad(speed_mps: float) -> float:
    return math.radians(unity_max_steer_deg(speed_mps))


def resolve_max_steer_rad(speed_mps: float, max_steer_deg_from_plant=None) -> float:
    """Plant-reported max steer (deg) when finite and plausible (5°–45°), else the mirror."""
    try:
        d = float(max_steer_deg_from_plant)
        if math.isfinite(d) and 5.0 <= d <= 45.0:
            return math.radians(d)
    except (TypeError, ValueError):
        pass
    return unity_max_steer_rad(speed_mps)


def steer_rad_to_normalized(delta_rad: float, max_steer_rad: float) -> float:
    """Physical wheel angle → normalized command in [-1, 1] for a plant whose full-scale is max_steer_rad."""
    m = max(float(max_steer_rad), MIN_MAX_STEER_RAD)
    x = float(delta_rad) / m
    return -1.0 if x < -1.0 else 1.0 if x > 1.0 else x
