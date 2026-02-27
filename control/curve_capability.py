"""
Shared curve capability utilities used by longitudinal and lateral modules.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional


G = 9.81


@dataclass
class TurnFeasibility:
    active: bool
    infeasible: bool
    curvature_abs: float
    speed_mps: float
    required_lat_accel_g: float
    comfort_limit_g: float
    peak_limit_g: float
    selected_limit_g: float
    effective_limit_g: float
    margin_g: float
    speed_limit_mps: float
    speed_delta_mps: float


def compute_turn_feasibility(
    *,
    speed_mps: float,
    curvature_abs: float,
    comfort_limit_g: float,
    peak_limit_g: float,
    use_peak_bound: bool = True,
    guardband_g: float = 0.015,
    curvature_min: float = 0.002,
    active_when: bool = True,
) -> TurnFeasibility:
    """Compute feasibility metrics for a given speed/curvature pair."""
    v = max(0.0, float(speed_mps))
    k = max(0.0, float(curvature_abs))
    comfort = max(0.0, float(comfort_limit_g))
    peak = max(0.0, float(peak_limit_g))
    selected = peak if use_peak_bound else comfort
    required = (v * v * k) / G if k > 0.0 else 0.0
    active = bool(active_when and k >= max(1e-9, float(curvature_min)))

    effective_limit = 0.0
    margin = 0.0
    speed_limit = 0.0
    delta = 0.0
    infeasible = False

    if active:
        effective_limit = max(0.0, selected - max(0.0, float(guardband_g)))
        margin = float(effective_limit - required)
        if k > 1e-9 and effective_limit > 0.0:
            speed_limit = float(math.sqrt((effective_limit * G) / k))
        delta = float(max(0.0, v - speed_limit))
        infeasible = bool(margin < 0.0)

    return TurnFeasibility(
        active=active,
        infeasible=infeasible,
        curvature_abs=float(k),
        speed_mps=float(v),
        required_lat_accel_g=float(required),
        comfort_limit_g=float(comfort),
        peak_limit_g=float(peak),
        selected_limit_g=float(selected),
        effective_limit_g=float(effective_limit),
        margin_g=float(margin),
        speed_limit_mps=float(speed_limit),
        speed_delta_mps=float(delta),
    )


def curve_intent_state_weight(state: Optional[str]) -> float:
    """Map curve intent scheduler state to a normalized urgency weight."""
    s = str(state or "").strip().upper()
    if s == "COMMIT":
        return 1.0
    if s == "ENTRY":
        return 0.75
    if s == "REARM":
        return 0.25
    return 0.0

