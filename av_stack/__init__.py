"""AV stack package. Re-exports all public symbols for backwards compatibility."""

from av_stack.lane_gating import (
    clamp_lane_center_and_width,
    clamp_lane_line_deltas,
    apply_lane_ema_gating,
    apply_lane_alpha_beta_gating,
    finalize_reject_reason,
    is_lane_low_visibility,
    is_lane_low_visibility_at_lookahead,
    estimate_single_lane_pair,
    blend_lane_pair_with_previous,
)
from av_stack.config import (
    ControlConfig,
    TrajectoryConfig,
    SafetyConfig,
    load_config,
)
from av_stack.speed_helpers import (
    _slew_limit_value,
    _apply_speed_limit_preview,
    _apply_curve_speed_preview,
    _preview_min_distance_allows_release,
    _apply_target_speed_slew,
    _apply_restart_ramp,
    _apply_steering_speed_guard,
    _is_teleport_jump,
)
from av_stack.orchestrator import AVStack, main

__all__ = [
    "clamp_lane_center_and_width", "clamp_lane_line_deltas",
    "apply_lane_ema_gating", "apply_lane_alpha_beta_gating",
    "finalize_reject_reason", "is_lane_low_visibility",
    "is_lane_low_visibility_at_lookahead", "estimate_single_lane_pair",
    "blend_lane_pair_with_previous",
    "ControlConfig", "TrajectoryConfig", "SafetyConfig", "load_config",
    "_slew_limit_value", "_apply_speed_limit_preview",
    "_apply_curve_speed_preview", "_preview_min_distance_allows_release",
    "_apply_target_speed_slew", "_apply_restart_ramp",
    "_apply_steering_speed_guard", "_is_teleport_jump",
    "AVStack", "main",
]
