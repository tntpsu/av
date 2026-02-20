"""
Speed governor: computes target speed from track limits, comfort physics,
and safety constraints.

Architecture:
  1. Comfort governor: v_comfort = sqrt(max_lat_accel_g * 9.81 / |curvature|)
  2. Curve preview: anticipatory deceleration before curves
  3. Perception horizon guardrail: don't outrun vision
  4. Speed planner: jerk-limited smooth transitions

This replaces the 12-layer serial suppression cascade that was previously
inline in av_stack.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from trajectory.speed_planner import SpeedPlanner, SpeedPlannerConfig
from trajectory.utils import apply_speed_horizon_guardrail, compute_dynamic_effective_horizon


G = 9.81  # m/s^2


@dataclass
class SpeedGovernorConfig:
    """Configuration for the speed governor."""

    # Comfort governor
    comfort_governor_max_lat_accel_g: float = 0.25
    comfort_governor_min_speed: float = 3.0

    # Curve preview (anticipatory deceleration)
    curve_preview_enabled: bool = True
    curve_preview_lookahead_scale: float = 1.6
    curve_preview_max_decel_mps2: float = 1.8
    curve_preview_min_curvature: float = 0.002

    # Horizon guardrail
    horizon_guardrail_enabled: bool = True
    horizon_guardrail_time_headway_s: float = 0.8
    horizon_guardrail_margin_m: float = 1.0
    horizon_guardrail_min_speed_mps: float = 3.0
    horizon_guardrail_gain: float = 1.0

    # Dynamic effective horizon parameters
    dynamic_effective_horizon_enabled: bool = True
    dynamic_effective_horizon_min_m: float = 3.0
    dynamic_effective_horizon_max_m: float = 20.0
    dynamic_effective_horizon_speed_scale: float = 0.35
    dynamic_effective_horizon_curvature_scale: float = 0.15
    dynamic_effective_horizon_confidence_floor: float = 0.6
    dynamic_effective_horizon_confidence_fallback: float = 1.0
    reference_lookahead_speed_min: float = 4.0
    reference_lookahead_speed_max: float = 10.0
    reference_lookahead_curvature_min: float = 0.002
    reference_lookahead_curvature_max: float = 0.015

    # Speed planner (jerk-limited)
    speed_planner_enabled: bool = True
    speed_planner_speed_limit_bias: float = -0.2


@dataclass
class SpeedGovernorOutput:
    """Output from the speed governor."""

    target_speed: float
    comfort_speed: float
    preview_speed: Optional[float]
    horizon_speed: Optional[float]
    planned_speed: Optional[float]
    planned_accel: Optional[float]
    planned_jerk: Optional[float]
    planner_active: bool
    active_limiter: str  # "none", "comfort", "preview", "horizon", "planner"

    # Diagnostics for HDF5 / PhilViz
    horizon_guardrail_active: bool = False
    horizon_guardrail_margin_m: float = 0.0
    effective_horizon_m: float = 0.0
    horizon_diag: dict = field(default_factory=dict)


class SpeedGovernor:
    """Computes target speed from track limits, comfort physics, and safety constraints.

    Replaces the 12-layer serial speed suppression cascade with 4 clean concerns:
    1. Comfort governor (physics-based lat accel limit)
    2. Curve preview (anticipatory deceleration)
    3. Perception horizon guardrail (don't outrun vision)
    4. Speed planner (jerk-limited transitions)
    """

    def __init__(self, config: SpeedGovernorConfig, planner_config: SpeedPlannerConfig) -> None:
        self.config = config
        self.speed_planner: Optional[SpeedPlanner] = None
        if config.speed_planner_enabled:
            self.speed_planner = SpeedPlanner(planner_config)

    def reset(self) -> None:
        """Reset internal state (speed planner)."""
        if self.speed_planner is not None:
            self.speed_planner.reset()

    def compute_target_speed(
        self,
        track_speed_limit: float,
        curvature: float,
        preview_curvature: Optional[float],
        perception_horizon_m: float,
        current_speed: float,
        timestamp: float,
        confidence: Optional[float] = None,
    ) -> SpeedGovernorOutput:
        """Compute the target speed for the longitudinal controller.

        Args:
            track_speed_limit: Safety speed limit from track config.
            curvature: Smoothed path curvature at the car position.
            preview_curvature: Curvature at lookahead distance (for preview decel).
            perception_horizon_m: How far perception can reliably see (meters).
            current_speed: Actual vehicle speed (m/s).
            timestamp: Current time (seconds).
            confidence: Perception confidence (0-1), or None.

        Returns:
            SpeedGovernorOutput with target speed and diagnostics.
        """
        target = max(0.0, float(track_speed_limit))

        # --- 1. Comfort governor ---
        comfort_speed = self._compute_comfort_speed(curvature)
        target = min(target, comfort_speed)

        # --- 2. Curve preview ---
        preview_speed = self._compute_preview_speed(target, preview_curvature)
        if preview_speed is not None:
            target = min(target, preview_speed)

        # --- 3. Perception horizon guardrail ---
        horizon_speed, horizon_active, horizon_margin, eff_horizon, horizon_diag = (
            self._apply_horizon_guardrail(
                target, perception_horizon_m, current_speed, curvature, confidence
            )
        )
        if horizon_speed is not None:
            target = min(target, horizon_speed)

        # Determine active limiter before planner smoothing
        active_limiter = "none"
        if target < track_speed_limit - 0.01:
            if comfort_speed <= target + 0.01:
                active_limiter = "comfort"
            elif preview_speed is not None and preview_speed <= target + 0.01:
                active_limiter = "preview"
            elif horizon_speed is not None and horizon_speed <= target + 0.01:
                active_limiter = "horizon"

        # --- 4. Speed planner (jerk-limited) ---
        planned_speed = None
        planned_accel = None
        planned_jerk = None
        planner_active = False
        if self.speed_planner is not None:
            planner_target = target + self.config.speed_planner_speed_limit_bias
            planner_target = max(0.0, planner_target)
            ps, pa, pj, pa_flag = self.speed_planner.step(
                planner_target, current_speed=current_speed, timestamp=timestamp
            )
            planned_speed = ps
            planned_accel = pa
            planned_jerk = pj
            planner_active = pa_flag
            if ps < target - 0.01:
                active_limiter = "planner"
            target = ps

        return SpeedGovernorOutput(
            target_speed=target,
            comfort_speed=comfort_speed,
            preview_speed=preview_speed,
            horizon_speed=horizon_speed,
            planned_speed=planned_speed,
            planned_accel=planned_accel,
            planned_jerk=planned_jerk,
            planner_active=planner_active,
            active_limiter=active_limiter,
            horizon_guardrail_active=horizon_active,
            horizon_guardrail_margin_m=horizon_margin,
            effective_horizon_m=eff_horizon,
            horizon_diag=horizon_diag,
        )

    def _compute_comfort_speed(self, curvature: float) -> float:
        """Physics-based comfortable speed: v = sqrt(a_lat_max / |curvature|)."""
        abs_curv = abs(float(curvature)) if isinstance(curvature, (int, float)) else 0.0
        if abs_curv < 1e-6:
            return float("inf")
        a_lat_max = self.config.comfort_governor_max_lat_accel_g * G
        v_comfort = math.sqrt(a_lat_max / abs_curv)
        return max(v_comfort, self.config.comfort_governor_min_speed)

    def _compute_preview_speed(
        self, current_target: float, preview_curvature: Optional[float]
    ) -> Optional[float]:
        """Anticipatory deceleration based on preview curvature ahead."""
        if not self.config.curve_preview_enabled:
            return None
        if preview_curvature is None:
            return None
        abs_curv = abs(float(preview_curvature))
        if abs_curv < self.config.curve_preview_min_curvature:
            return None

        a_lat_max = self.config.comfort_governor_max_lat_accel_g * G
        curve_speed = math.sqrt(a_lat_max / abs_curv)
        curve_speed = max(curve_speed, self.config.comfort_governor_min_speed)

        if curve_speed >= current_target:
            return None

        # v_entry^2 = v_curve^2 + 2 * decel * distance
        # For now use a fixed preview distance based on current speed
        preview_distance = max(1.0, current_target * 0.8)
        max_decel = self.config.curve_preview_max_decel_mps2
        v_entry_sq = curve_speed ** 2 + 2.0 * max_decel * preview_distance
        v_entry = math.sqrt(max(0.0, v_entry_sq))
        return min(current_target, v_entry)

    def _apply_horizon_guardrail(
        self,
        target_speed: float,
        perception_horizon_m: float,
        current_speed: float,
        curvature: float,
        confidence: Optional[float],
    ) -> tuple[Optional[float], bool, float, float, dict]:
        """Apply perception horizon guardrail: don't outrun your vision."""
        if not self.config.horizon_guardrail_enabled:
            return None, False, 0.0, perception_horizon_m, {}

        horizon_config = {
            "dynamic_effective_horizon_enabled": self.config.dynamic_effective_horizon_enabled,
            "dynamic_effective_horizon_min_m": self.config.dynamic_effective_horizon_min_m,
            "dynamic_effective_horizon_max_m": self.config.dynamic_effective_horizon_max_m,
            "dynamic_effective_horizon_speed_scale": self.config.dynamic_effective_horizon_speed_scale,
            "dynamic_effective_horizon_curvature_scale": self.config.dynamic_effective_horizon_curvature_scale,
            "dynamic_effective_horizon_confidence_floor": self.config.dynamic_effective_horizon_confidence_floor,
            "dynamic_effective_horizon_confidence_fallback": self.config.dynamic_effective_horizon_confidence_fallback,
            "reference_lookahead_speed_min": self.config.reference_lookahead_speed_min,
            "reference_lookahead_speed_max": self.config.reference_lookahead_speed_max,
            "reference_lookahead_curvature_min": self.config.reference_lookahead_curvature_min,
            "reference_lookahead_curvature_max": self.config.reference_lookahead_curvature_max,
            "speed_horizon_guardrail_enabled": True,
            "speed_horizon_guardrail_time_headway_s": self.config.horizon_guardrail_time_headway_s,
            "speed_horizon_guardrail_margin_m": self.config.horizon_guardrail_margin_m,
            "speed_horizon_guardrail_min_speed_mps": self.config.horizon_guardrail_min_speed_mps,
            "speed_horizon_guardrail_gain": self.config.horizon_guardrail_gain,
        }

        horizon_diag = compute_dynamic_effective_horizon(
            base_horizon_m=float(perception_horizon_m),
            current_speed_mps=float(current_speed),
            path_curvature=float(curvature),
            confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
            config=horizon_config,
        )

        eff_horizon = float(horizon_diag.get("diag_dynamic_effective_horizon_m", perception_horizon_m))
        applied = float(horizon_diag.get("diag_dynamic_effective_horizon_applied", 0.0)) > 0.5

        guardrail_result = apply_speed_horizon_guardrail(
            target_speed_mps=float(target_speed),
            effective_horizon_m=eff_horizon,
            dynamic_horizon_applied=applied,
            config=horizon_config,
        )

        horizon_speed = float(guardrail_result.get("target_speed_mps", target_speed))
        guardrail_active = float(guardrail_result.get("diag_speed_horizon_guardrail_active", 0.0)) > 0.5
        margin = float(guardrail_result.get("diag_speed_horizon_guardrail_margin_m", 0.0))

        full_diag = {**horizon_diag, **guardrail_result}

        if not guardrail_active:
            return None, False, margin, eff_horizon, full_diag

        return horizon_speed, True, margin, eff_horizon, full_diag


def build_speed_governor(trajectory_cfg: dict, speed_planner_cfg: dict) -> SpeedGovernor:
    """Build a SpeedGovernor from the config dictionaries."""
    gov_cfg_section = trajectory_cfg.get("speed_governor", {})

    gov_config = SpeedGovernorConfig(
        comfort_governor_max_lat_accel_g=float(
            gov_cfg_section.get("comfort_governor_max_lat_accel_g", 0.25)
        ),
        comfort_governor_min_speed=float(
            gov_cfg_section.get("comfort_governor_min_speed",
                                trajectory_cfg.get("min_speed_floor", 3.0))
        ),
        curve_preview_enabled=bool(
            gov_cfg_section.get("curve_preview_enabled",
                                trajectory_cfg.get("curve_speed_preview_enabled", True))
        ),
        curve_preview_lookahead_scale=float(
            gov_cfg_section.get("curve_preview_lookahead_scale",
                                trajectory_cfg.get("curve_speed_preview_lookahead_scale", 1.6))
        ),
        curve_preview_max_decel_mps2=float(
            gov_cfg_section.get("curve_preview_max_decel_mps2",
                                trajectory_cfg.get("curve_speed_preview_decel", 1.8))
        ),
        curve_preview_min_curvature=float(
            gov_cfg_section.get("curve_preview_min_curvature",
                                trajectory_cfg.get("curve_speed_preview_min_curvature", 0.002))
        ),
        horizon_guardrail_enabled=bool(
            gov_cfg_section.get("horizon_guardrail_enabled",
                                trajectory_cfg.get("speed_horizon_guardrail_enabled", True))
        ),
        horizon_guardrail_time_headway_s=float(
            gov_cfg_section.get("horizon_guardrail_time_headway_s",
                                trajectory_cfg.get("speed_horizon_guardrail_time_headway_s", 0.8))
        ),
        horizon_guardrail_margin_m=float(
            gov_cfg_section.get("horizon_guardrail_margin_m",
                                trajectory_cfg.get("speed_horizon_guardrail_margin_m", 1.0))
        ),
        horizon_guardrail_min_speed_mps=float(
            gov_cfg_section.get("horizon_guardrail_min_speed_mps",
                                trajectory_cfg.get("speed_horizon_guardrail_min_speed_mps", 3.0))
        ),
        horizon_guardrail_gain=float(
            gov_cfg_section.get("horizon_guardrail_gain",
                                trajectory_cfg.get("speed_horizon_guardrail_gain", 1.0))
        ),
        dynamic_effective_horizon_enabled=bool(
            gov_cfg_section.get("dynamic_effective_horizon_enabled",
                                trajectory_cfg.get("dynamic_effective_horizon_enabled", True))
        ),
        dynamic_effective_horizon_min_m=float(
            gov_cfg_section.get("dynamic_effective_horizon_min_m",
                                trajectory_cfg.get("dynamic_effective_horizon_min_m", 3.0))
        ),
        dynamic_effective_horizon_max_m=float(
            gov_cfg_section.get("dynamic_effective_horizon_max_m",
                                trajectory_cfg.get("dynamic_effective_horizon_max_m", 20.0))
        ),
        dynamic_effective_horizon_speed_scale=float(
            gov_cfg_section.get("dynamic_effective_horizon_speed_scale",
                                trajectory_cfg.get("dynamic_effective_horizon_speed_scale", 0.35))
        ),
        dynamic_effective_horizon_curvature_scale=float(
            gov_cfg_section.get("dynamic_effective_horizon_curvature_scale",
                                trajectory_cfg.get("dynamic_effective_horizon_curvature_scale", 0.15))
        ),
        dynamic_effective_horizon_confidence_floor=float(
            gov_cfg_section.get("dynamic_effective_horizon_confidence_floor",
                                trajectory_cfg.get("dynamic_effective_horizon_confidence_floor", 0.6))
        ),
        dynamic_effective_horizon_confidence_fallback=float(
            gov_cfg_section.get("dynamic_effective_horizon_confidence_fallback",
                                trajectory_cfg.get("dynamic_effective_horizon_confidence_fallback", 1.0))
        ),
        reference_lookahead_speed_min=float(
            trajectory_cfg.get("reference_lookahead_speed_min", 4.0)
        ),
        reference_lookahead_speed_max=float(
            trajectory_cfg.get("reference_lookahead_speed_max", 10.0)
        ),
        reference_lookahead_curvature_min=float(
            trajectory_cfg.get("reference_lookahead_curvature_min", 0.002)
        ),
        reference_lookahead_curvature_max=float(
            trajectory_cfg.get("reference_lookahead_curvature_max", 0.015)
        ),
        speed_planner_enabled=bool(speed_planner_cfg.get("enabled", False)),
        speed_planner_speed_limit_bias=float(
            speed_planner_cfg.get("speed_limit_bias", 0.0)
        ),
    )

    planner_config = SpeedPlannerConfig(
        max_accel=float(speed_planner_cfg.get("max_accel", 2.0)),
        max_decel=float(speed_planner_cfg.get("max_decel", 2.5)),
        max_jerk=float(speed_planner_cfg.get("max_jerk", 1.2)),
        max_jerk_min=float(speed_planner_cfg.get("max_jerk_min", 0.0)),
        max_jerk_max=float(speed_planner_cfg.get("max_jerk_max", 0.0)),
        jerk_error_min=float(speed_planner_cfg.get("jerk_error_min", 0.0)),
        jerk_error_max=float(speed_planner_cfg.get("jerk_error_max", 0.0)),
        min_speed=float(speed_planner_cfg.get("min_speed", 0.0)),
        launch_speed_floor=float(speed_planner_cfg.get("launch_speed_floor", 0.0)),
        launch_speed_floor_threshold=float(speed_planner_cfg.get("launch_speed_floor_threshold", 0.0)),
        reset_gap_seconds=float(speed_planner_cfg.get("reset_gap_seconds", 0.5)),
        sync_speed_threshold=float(speed_planner_cfg.get("sync_speed_threshold", 3.0)),
        default_dt=float(speed_planner_cfg.get("default_dt", 1.0 / 30.0)),
        speed_error_gain=float(speed_planner_cfg.get("speed_error_gain", 0.0)),
        speed_error_max_delta=float(speed_planner_cfg.get("speed_error_max_delta", 0.0)),
        speed_error_deadband=float(speed_planner_cfg.get("speed_error_deadband", 0.0)),
        sync_under_target=bool(speed_planner_cfg.get("sync_under_target", True)),
        sync_under_target_error=float(speed_planner_cfg.get("sync_under_target_error", 0.2)),
        use_speed_dependent_limits=bool(speed_planner_cfg.get("use_speed_dependent_limits", False)),
        accel_speed_min=float(speed_planner_cfg.get("accel_speed_min", 0.0)),
        accel_speed_max=float(speed_planner_cfg.get("accel_speed_max", 0.0)),
        max_accel_at_speed_min=float(speed_planner_cfg.get("max_accel_at_speed_min", 0.0)),
        max_accel_at_speed_max=float(speed_planner_cfg.get("max_accel_at_speed_max", 0.0)),
        decel_speed_min=float(speed_planner_cfg.get("decel_speed_min", 0.0)),
        decel_speed_max=float(speed_planner_cfg.get("decel_speed_max", 0.0)),
        max_decel_at_speed_min=float(speed_planner_cfg.get("max_decel_at_speed_min", 0.0)),
        max_decel_at_speed_max=float(speed_planner_cfg.get("max_decel_at_speed_max", 0.0)),
        enforce_desired_speed_slew=bool(speed_planner_cfg.get("enforce_desired_speed_slew", False)),
    )

    return SpeedGovernor(gov_config, planner_config)
