"""
Main AV stack integration script.
Connects all components: perception, trajectory planning, and control.
"""

import time
import math
import numpy as np
from typing import Optional, Tuple
import sys
import os
import json
import hashlib
import subprocess
from pathlib import Path
import logging
import yaml
from dataclasses import dataclass
import cv2

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from bridge.client import UnityBridgeClient
from perception.inference import LaneDetectionInference
from trajectory.inference import TrajectoryPlanningInference
from trajectory.speed_planner import SpeedPlanner, SpeedPlannerConfig
from trajectory.utils import (
    compute_reference_lookahead,
    compute_dynamic_effective_horizon,
    apply_speed_horizon_guardrail,
    curvature_smoothing_alpha,
    select_curvature_bin_limits,
)
from control.pid_controller import VehicleController
from control.speed_governor import build_speed_governor, SpeedGovernorOutput
from data.recorder import DataRecorder
from data.formats.data_format import CameraFrame, VehicleState, ControlCommand, PerceptionOutput, TrajectoryOutput, RecordingFrame

# Configure logging
# Ensure tmp/logs directory exists
log_dir = Path(__file__).parent / 'tmp' / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / 'av_stack.log'

logging.basicConfig(
    level=logging.ERROR,  # Changed from INFO to ERROR to reduce expensive print operations
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(log_file))
    ]
)
logger = logging.getLogger(__name__)


def clamp_lane_center_and_width(
    current_center: float,
    current_width: float,
    previous_center: float,
    previous_width: float,
    max_center_delta: float,
    max_width_delta: float,
) -> Tuple[float, float, bool]:
    """Clamp sudden lane center/width changes to avoid reference point jumps."""
    clamped_center = current_center
    clamped_width = current_width

    if max_center_delta > 0.0:
        center_delta = current_center - previous_center
        if abs(center_delta) > max_center_delta:
            clamped_center = previous_center + np.sign(center_delta) * max_center_delta

    if max_width_delta > 0.0:
        width_delta = current_width - previous_width
        if abs(width_delta) > max_width_delta:
            clamped_width = previous_width + np.sign(width_delta) * max_width_delta

    was_clamped = (clamped_center != current_center) or (clamped_width != current_width)
    return clamped_center, clamped_width, was_clamped


def clamp_lane_line_deltas(
    current_left: float,
    current_right: float,
    previous_left: float,
    previous_right: float,
    max_delta: float,
) -> Tuple[float, float, bool]:
    """Clamp sudden per-lane x changes to reduce perception jitter."""
    if max_delta <= 0.0:
        return current_left, current_right, False
    clamped_left = previous_left + np.clip(current_left - previous_left, -max_delta, max_delta)
    clamped_right = previous_right + np.clip(current_right - previous_right, -max_delta, max_delta)
    was_clamped = (clamped_left != current_left) or (clamped_right != current_right)
    return clamped_left, clamped_right, was_clamped


def apply_lane_ema_gating(
    lane_center: float,
    lane_width: float,
    lane_center_ema: float | None,
    lane_width_ema: float | None,
    center_gate_m: float,
    width_gate_m: float,
    center_alpha: float,
    width_alpha: float,
) -> Tuple[float, float, float, float, list, bool]:
    """Apply EMA smoothing with measurement gating for lane center/width."""
    gate_events: list = []
    gated = False
    center_alpha = min(max(center_alpha, 0.0), 1.0)
    width_alpha = min(max(width_alpha, 0.0), 1.0)

    if lane_center_ema is None or lane_width_ema is None:
        return lane_center, lane_width, lane_center, lane_width, gate_events, gated

    center_delta = abs(lane_center - lane_center_ema)
    width_delta = abs(lane_width - lane_width_ema)
    if center_gate_m > 0.0 and center_delta > center_gate_m:
        gate_events.append("lane_center_gate_reject")
        gated = True
    if width_gate_m > 0.0 and width_delta > width_gate_m:
        gate_events.append("lane_width_gate_reject")
        gated = True
    if gated:
        return lane_center_ema, lane_width_ema, lane_center_ema, lane_width_ema, gate_events, gated

    lane_center_ema = (center_alpha * lane_center) + ((1.0 - center_alpha) * lane_center_ema)
    lane_width_ema = (width_alpha * lane_width) + ((1.0 - width_alpha) * lane_width_ema)
    return lane_center_ema, lane_width_ema, lane_center_ema, lane_width_ema, gate_events, gated


def apply_lane_alpha_beta_gating(
    lane_center: float,
    lane_width: float,
    state_center: float | None,
    state_center_vel: float | None,
    state_width: float | None,
    state_width_vel: float | None,
    center_gate_m: float,
    width_gate_m: float,
    center_alpha: float,
    center_beta: float,
    width_alpha: float,
    width_beta: float,
    dt: float,
) -> Tuple[float, float, float, float, float, float, list, bool]:
    """Apply alpha-beta filter with measurement gating for lane center/width."""
    gate_events: list = []
    gated = False
    center_alpha = min(max(center_alpha, 0.0), 1.0)
    width_alpha = min(max(width_alpha, 0.0), 1.0)
    center_beta = max(center_beta, 0.0)
    width_beta = max(width_beta, 0.0)
    dt = max(dt, 1e-3)

    if state_center is None or state_center_vel is None:
        state_center = lane_center
        state_center_vel = 0.0
    if state_width is None or state_width_vel is None:
        state_width = lane_width
        state_width_vel = 0.0

    # Predict
    pred_center = state_center + state_center_vel * dt
    pred_width = state_width + state_width_vel * dt

    center_residual = lane_center - pred_center
    width_residual = lane_width - pred_width

    if center_gate_m > 0.0 and abs(center_residual) > center_gate_m:
        gate_events.append("lane_center_gate_reject")
        gated = True
    if width_gate_m > 0.0 and abs(width_residual) > width_gate_m:
        gate_events.append("lane_width_gate_reject")
        gated = True

    if gated:
        return (
            pred_center,
            pred_width,
            pred_center,
            state_center_vel,
            pred_width,
            state_width_vel,
            gate_events,
            gated,
        )

    # Update
    updated_center = pred_center + center_alpha * center_residual
    updated_center_vel = state_center_vel + (center_beta * center_residual / dt)
    updated_width = pred_width + width_alpha * width_residual
    updated_width_vel = state_width_vel + (width_beta * width_residual / dt)

    return (
        updated_center,
        updated_width,
        updated_center,
        updated_center_vel,
        updated_width,
        updated_width_vel,
        gate_events,
        gated,
    )


def finalize_reject_reason(
    reject_reason: Optional[str],
    stale_data_reason: Optional[str],
    clamp_events: Optional[list],
) -> Optional[str]:
    """Pick a human-readable rejection reason for diagnostics."""
    if reject_reason:
        return reject_reason
    if stale_data_reason:
        return stale_data_reason
    if clamp_events:
        return clamp_events[0]
    return None


def is_lane_low_visibility(
    points: Optional[list],
    image_width: float,
    side: str,
    min_points: int = 6,
    edge_margin: int = 12,
) -> bool:
    """Return True when lane points are too few or hugging the image edge."""
    if not points or len(points) < min_points:
        return True
    xs = [p[0] for p in points if isinstance(p, (list, tuple)) and len(p) >= 2]
    if not xs:
        return True
    if side == "left":
        return min(xs) < edge_margin
    return max(xs) > (image_width - edge_margin)


def is_lane_low_visibility_at_lookahead(
    points: Optional[list],
    image_width: float,
    side: str,
    y_lookahead: float,
    min_points_in_band: int = 4,
    y_band_half_height: float = 18.0,
    edge_margin: int = 12,
) -> bool:
    """
    Return True when lane support is weak specifically around the lookahead row.

    This catches dashed-line gaps at lookahead that global fit-point counts can miss.
    """
    if points is None:
        return True
    valid = [p for p in points if isinstance(p, (list, tuple)) and len(p) >= 2]
    if not valid:
        return True
    band_points = [p for p in valid if abs(float(p[1]) - float(y_lookahead)) <= y_band_half_height]
    if len(band_points) < min_points_in_band:
        return True
    xs = [float(p[0]) for p in band_points]
    if side == "left":
        return min(xs) < edge_margin
    return max(xs) > (image_width - edge_margin)


def estimate_single_lane_pair(
    single_x_vehicle: float,
    is_left_lane: bool,
    last_width: Optional[float],
    default_width: float,
    width_min: float,
    width_max: float,
) -> Tuple[float, float]:
    """Estimate missing lane boundary using last known width when available."""
    width = default_width
    if last_width is not None and width_min <= last_width <= width_max:
        width = last_width
    if is_left_lane:
        left_lane_line_x = float(single_x_vehicle)
        right_lane_line_x = float(single_x_vehicle + width)
    else:
        left_lane_line_x = float(single_x_vehicle - width)
        right_lane_line_x = float(single_x_vehicle)
    return left_lane_line_x, right_lane_line_x


def blend_lane_pair_with_previous(
    estimated_left_lane_x: float,
    estimated_right_lane_x: float,
    previous_left_lane_x: float,
    previous_right_lane_x: float,
    blend_alpha: float,
    center_shift_cap_m: float,
) -> Tuple[float, float]:
    """
    Blend fallback-estimated lanes with previous lanes and cap center shift.

    This keeps fallback bounded when dashed-line gaps are transient so
    synthetic lane reconstruction does not cause large centerline jumps.
    """
    alpha = float(np.clip(blend_alpha, 0.0, 1.0))
    left_lane_x = (1.0 - alpha) * float(previous_left_lane_x) + alpha * float(
        estimated_left_lane_x
    )
    right_lane_x = (1.0 - alpha) * float(previous_right_lane_x) + alpha * float(
        estimated_right_lane_x
    )
    prev_center = 0.5 * (float(previous_left_lane_x) + float(previous_right_lane_x))
    center = 0.5 * (left_lane_x + right_lane_x)
    center_shift = center - prev_center
    cap = max(0.0, float(center_shift_cap_m))
    if cap > 0.0 and abs(center_shift) > cap:
        correction = np.sign(center_shift) * (abs(center_shift) - cap)
        left_lane_x -= correction
        right_lane_x -= correction
    return float(left_lane_x), float(right_lane_x)


@dataclass
class ControlConfig:
    """Control configuration parameters."""
    lateral_kp: float = 0.3
    lateral_ki: float = 0.0
    lateral_kd: float = 0.1
    lateral_max_steering: float = 0.5
    lateral_deadband: float = 0.02
    lateral_heading_weight: float = 0.5
    lateral_lateral_weight: float = 0.5
    lateral_error_clip: float = np.pi / 4
    lateral_integral_limit: float = 0.3
    
    longitudinal_kp: float = 0.3
    longitudinal_ki: float = 0.05
    longitudinal_kd: float = 0.02
    longitudinal_target_speed: float = 8.0
    longitudinal_max_speed: float = 10.0
    longitudinal_speed_smoothing: float = 0.7
    longitudinal_speed_deadband: float = 0.1
    longitudinal_throttle_limit_threshold: float = 0.8
    longitudinal_throttle_reduction_factor: float = 0.3
    longitudinal_brake_aggression: float = 3.0


@dataclass
class TrajectoryConfig:
    """Trajectory planning configuration parameters."""
    lookahead_distance: float = 20.0
    point_spacing: float = 1.0
    target_speed: float = 8.0
    reference_lookahead: float = 8.0
    image_width: float = 640.0
    image_height: float = 480.0
    camera_fov: float = 75.0
    camera_height: float = 1.2
    bias_correction_threshold: float = 10.0


@dataclass
class SafetyConfig:
    """Safety configuration parameters."""
    max_speed: float = 10.0
    emergency_brake_threshold: float = 2.0
    speed_prevention_threshold: float = 0.85
    speed_prevention_brake_threshold: float = 0.9
    speed_prevention_brake_amount: float = 0.2
    lane_width: float = 7.0  # Lane width in meters
    car_width: float = 1.85  # Car width in meters
    allowed_outside_lane: float = 1.0  # Allowed distance outside lane before emergency stop (meters)


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file or use defaults."""
    if config_path is None:
        config_path = Path(__file__).parent / "config" / "av_stack_config.yaml"
    else:
        config_path = Path(config_path)
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    else:
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return {}


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


class AVStack:
    """Main AV stack integrating all components."""
    
    def __init__(self, bridge_url: str = "http://localhost:8000",
                 model_path: Optional[str] = None,
                 record_data: bool = True,  # Enable by default for comprehensive logging
                 recording_dir: str = "data/recordings",
                 config_path: Optional[str] = None,
                 use_segmentation: bool = False,
                 segmentation_model_path: Optional[str] = None):
        """
        Initialize AV stack.
        
        Args:
            bridge_url: URL of Unity bridge server
            model_path: Path to trained perception model
            record_data: Whether to record data (default: True for comprehensive logging)
            recording_dir: Directory for recordings
        """
        # Load configuration
        config = load_config(config_path)
        control_cfg = config.get('control', {})
        trajectory_cfg = config.get('trajectory', {})
        safety_cfg = config.get('safety', {})
        perception_cfg = config.get("perception", {}) if isinstance(config, dict) else {}
        
        # Bridge client
        self.bridge = UnityBridgeClient(bridge_url)
        self.last_processed_unity_frame_count = None
        
        # Perception
        self.perception = LaneDetectionInference(
            model_path=model_path,
            segmentation_model_path=segmentation_model_path,
            segmentation_mode=use_segmentation,
            segmentation_fit_min_row_ratio=float(
                perception_cfg.get("segmentation_fit_min_row_ratio", 0.45)
            ),
            segmentation_fit_max_row_ratio=float(
                perception_cfg.get("segmentation_fit_max_row_ratio", 0.75)
            ),
            segmentation_ransac_enabled=bool(
                perception_cfg.get("segmentation_ransac_enabled", False)
            ),
            segmentation_ransac_residual_px=float(
                perception_cfg.get("segmentation_ransac_residual_px", 6.0)
            ),
            segmentation_ransac_min_inliers=int(
                perception_cfg.get("segmentation_ransac_min_inliers", 10)
            ),
            segmentation_ransac_max_trials=int(
                perception_cfg.get("segmentation_ransac_max_trials", 40)
            ),
            model_fallback_confidence_hard_threshold=float(
                perception_cfg.get("model_fallback_confidence_hard_threshold", 0.1)
            ),
            model_fallback_zero_lane_confidence_threshold=float(
                perception_cfg.get("model_fallback_zero_lane_confidence_threshold", 0.6)
            ),
        )
        
        # Trajectory planning - Load from config
        lateral_cfg = control_cfg.get('lateral', {})
        longitudinal_cfg = control_cfg.get('longitudinal', {})
        
        self.trajectory_planner = TrajectoryPlanningInference(
            planner_type="rule_based",
            lookahead_distance=trajectory_cfg.get('lookahead_distance', 20.0),
            target_speed=trajectory_cfg.get('target_speed', 8.0),
            image_width=trajectory_cfg.get('image_width', 640.0),
            image_height=trajectory_cfg.get('image_height', 480.0),
            reference_smoothing=trajectory_cfg.get('reference_smoothing', 0.95),  # Increased to 0.95
            lane_smoothing_alpha=trajectory_cfg.get('lane_smoothing_alpha', 0.7),  # NEW: Lane coefficient smoothing
            lane_position_smoothing_alpha=trajectory_cfg.get('lane_position_smoothing_alpha', 0.1),
            ref_point_jump_clamp_x=trajectory_cfg.get('ref_point_jump_clamp_x', 0.0),
            ref_point_jump_clamp_heading=trajectory_cfg.get('ref_point_jump_clamp_heading', 0.0),
            reference_x_max_abs=trajectory_cfg.get('reference_x_max_abs', 0.0),
            camera_fov=trajectory_cfg.get('camera_fov', 75.0),
            camera_height=trajectory_cfg.get('camera_height', 1.2),
            bias_correction_threshold=trajectory_cfg.get('bias_correction_threshold', 10.0),
            camera_offset_x=trajectory_cfg.get('camera_offset_x', 0.0),  # NEW: Camera offset correction
            distance_scaling_factor=trajectory_cfg.get('distance_scaling_factor', 0.875),  # NEW: Distance scaling (7/8 = 0.875)
            target_lane=trajectory_cfg.get('target_lane', 'center'),
            target_lane_width_m=trajectory_cfg.get('target_lane_width_m', 0.0),
            ref_sign_flip_threshold=trajectory_cfg.get('ref_sign_flip_threshold', 0.0),
            ref_x_rate_limit=trajectory_cfg.get('ref_x_rate_limit', 0.0),
            ref_x_rate_limit_turn_heading_min_abs_rad=trajectory_cfg.get(
                'ref_x_rate_limit_turn_heading_min_abs_rad', 0.08
            ),
            ref_x_rate_limit_turn_scale_max=trajectory_cfg.get(
                'ref_x_rate_limit_turn_scale_max', 2.5
            ),
            ref_sign_flip_disable_on_turn=bool(
                trajectory_cfg.get('ref_sign_flip_disable_on_turn', True)
            ),
            ref_sign_flip_disable_heading_min_abs_rad=float(
                trajectory_cfg.get('ref_sign_flip_disable_heading_min_abs_rad', 0.03)
            ),
            ref_x_rate_limit_precurve_enabled=bool(
                trajectory_cfg.get('ref_x_rate_limit_precurve_enabled', True)
            ),
            ref_x_rate_limit_precurve_heading_min_abs_rad=float(
                trajectory_cfg.get('ref_x_rate_limit_precurve_heading_min_abs_rad', 0.03)
            ),
            ref_x_rate_limit_precurve_scale_max=float(
                trajectory_cfg.get('ref_x_rate_limit_precurve_scale_max', 4.0)
            ),
            multi_lookahead_enabled=trajectory_cfg.get('multi_lookahead_enabled', False),
            multi_lookahead_far_scale=trajectory_cfg.get('multi_lookahead_far_scale', 1.5),
            multi_lookahead_blend_alpha=trajectory_cfg.get('multi_lookahead_blend_alpha', 0.35),
            multi_lookahead_curvature_threshold=trajectory_cfg.get(
                'multi_lookahead_curvature_threshold', 0.01
            ),
            multi_lookahead_curve_heading_smoothing_alpha=trajectory_cfg.get(
                'multi_lookahead_curve_heading_smoothing_alpha',
                trajectory_cfg.get('reference_smoothing', 0.95),
            ),
            multi_lookahead_curve_heading_min_abs_rad=trajectory_cfg.get(
                'multi_lookahead_curve_heading_min_abs_rad', 0.05
            ),
            traj_heading_zero_gate_center_a_on_abs_max=trajectory_cfg.get(
                'traj_heading_zero_gate_center_a_on_abs_max', 0.004
            ),
            traj_heading_zero_gate_center_a_off_abs_max=trajectory_cfg.get(
                'traj_heading_zero_gate_center_a_off_abs_max', 0.008
            ),
            traj_heading_zero_gate_heading_on_abs_rad=trajectory_cfg.get(
                'traj_heading_zero_gate_heading_on_abs_rad', 0.035
            ),
            traj_heading_zero_gate_heading_off_abs_rad=trajectory_cfg.get(
                'traj_heading_zero_gate_heading_off_abs_rad', 0.061
            ),
            center_spline_enabled=trajectory_cfg.get('center_spline_enabled', False),
            center_spline_degree=trajectory_cfg.get('center_spline_degree', 2),
            center_spline_samples=trajectory_cfg.get('center_spline_samples', 20),
            center_spline_alpha=trajectory_cfg.get('center_spline_alpha', 0.7),
            x_clip_enabled=trajectory_cfg.get('x_clip_enabled', True),
            x_clip_limit_m=trajectory_cfg.get('x_clip_limit_m', 10.0),
            trajectory_eval_y_min_ratio=trajectory_cfg.get('trajectory_eval_y_min_ratio', 0.0),
            centerline_midpoint_mode=trajectory_cfg.get('centerline_midpoint_mode', 'pointwise'),
            projection_model=trajectory_cfg.get('projection_model', 'legacy'),
            calibrated_fx_px=trajectory_cfg.get('calibrated_fx_px', 0.0),
            calibrated_fy_px=trajectory_cfg.get('calibrated_fy_px', 0.0),
            calibrated_cx_px=trajectory_cfg.get('calibrated_cx_px', 0.0),
            calibrated_cy_px=trajectory_cfg.get('calibrated_cy_px', 0.0),
            dynamic_effective_horizon_farfield_scale_min=trajectory_cfg.get(
                'dynamic_effective_horizon_farfield_scale_min', 0.85
            ),
            far_band_contribution_cap_enabled=trajectory_cfg.get(
                'far_band_contribution_cap_enabled', True
            ),
            far_band_contribution_cap_start_m=trajectory_cfg.get(
                'far_band_contribution_cap_start_m', 12.0
            ),
            far_band_contribution_cap_gain=trajectory_cfg.get(
                'far_band_contribution_cap_gain', 0.35
            ),
            heading_zero_quad_threshold=trajectory_cfg.get(
                'heading_zero_quad_threshold', 0.005
            ),
            heading_zero_curvature_guard=trajectory_cfg.get(
                'heading_zero_curvature_guard', 0.002
            ),
            smoothing_alpha_curve_reduction=trajectory_cfg.get(
                'smoothing_alpha_curve_reduction', 0.15
            ),
            ref_x_rate_limit_curvature_min=trajectory_cfg.get(
                'ref_x_rate_limit_curvature_min', 0.003
            ),
            ref_x_rate_limit_curvature_scale_max=trajectory_cfg.get(
                'ref_x_rate_limit_curvature_scale_max', 3.0
            ),
        )
        
        # Control - Load from config
        lateral_kp = lateral_cfg.get('kp', 0.3)
        lateral_ki = lateral_cfg.get('ki', 0.0)
        lateral_kd = lateral_cfg.get('kd', 0.1)
        speed_planner_cfg = trajectory_cfg.get('speed_planner', {})
        longitudinal_max_accel = float(
            longitudinal_cfg.get('max_accel', speed_planner_cfg.get('max_accel', 2.5))
        )
        longitudinal_max_decel = float(
            longitudinal_cfg.get('max_decel', speed_planner_cfg.get('max_decel', 3.0))
        )
        longitudinal_max_jerk = float(
            longitudinal_cfg.get('max_jerk', speed_planner_cfg.get('max_jerk', 2.0))
        )
        
        # Log controller parameters for debugging
        logger.info(f"[CONTROLLER] Lateral control parameters: kp={lateral_kp}, ki={lateral_ki}, kd={lateral_kd}")
        
        self.controller = VehicleController(
            lateral_kp=lateral_kp,
            lateral_ki=lateral_ki,
            lateral_kd=lateral_kd,
            longitudinal_kp=longitudinal_cfg.get('kp', 0.3),
            longitudinal_ki=longitudinal_cfg.get('ki', 0.05),
            longitudinal_kd=longitudinal_cfg.get('kd', 0.02),
            lookahead_distance=trajectory_cfg.get('reference_lookahead', 8.0),
            target_speed=longitudinal_cfg.get('target_speed', 8.0),
            max_speed=safety_cfg.get('max_speed', 10.0),
            throttle_rate_limit=longitudinal_cfg.get('throttle_rate_limit', 0.08),
            brake_rate_limit=longitudinal_cfg.get('brake_rate_limit', 0.15),
            throttle_smoothing_alpha=longitudinal_cfg.get('throttle_smoothing_alpha', 0.6),
            speed_smoothing_alpha=longitudinal_cfg.get('speed_smoothing', 0.6),
            min_throttle_when_accel=longitudinal_cfg.get('min_throttle_when_accel', 0.0),
            longitudinal_max_accel=longitudinal_max_accel,
            longitudinal_max_decel=longitudinal_max_decel,
            longitudinal_max_jerk=longitudinal_max_jerk,
            longitudinal_accel_feedforward_gain=longitudinal_cfg.get('accel_feedforward_gain', 1.0),
            longitudinal_decel_feedforward_gain=longitudinal_cfg.get('decel_feedforward_gain', 1.0),
            longitudinal_speed_for_jerk_alpha=longitudinal_cfg.get('speed_for_jerk_alpha', 0.7),
            longitudinal_jerk_error_min=longitudinal_cfg.get('jerk_error_min', 0.5),
            longitudinal_jerk_error_max=longitudinal_cfg.get('jerk_error_max', 3.0),
            longitudinal_max_jerk_min=longitudinal_cfg.get('max_jerk_min', longitudinal_max_jerk),
            longitudinal_max_jerk_max=longitudinal_cfg.get('max_jerk_max', max(longitudinal_max_jerk, 6.0)),
            longitudinal_jerk_cooldown_frames=int(longitudinal_cfg.get('jerk_cooldown_frames', 0)),
            longitudinal_jerk_cooldown_scale=float(longitudinal_cfg.get('jerk_cooldown_scale', 0.6)),
            longitudinal_min_throttle_hold=float(longitudinal_cfg.get('min_throttle_hold', 0.12)),
            longitudinal_min_throttle_hold_speed=float(
                longitudinal_cfg.get('min_throttle_hold_speed', 4.0)
            ),
            longitudinal_speed_error_to_accel_gain=float(longitudinal_cfg.get('speed_error_to_accel_gain', 0.5)),
            longitudinal_speed_error_deadband=float(longitudinal_cfg.get('speed_error_deadband', 0.0)),
            longitudinal_speed_error_gain_under=longitudinal_cfg.get('speed_error_gain_under'),
            longitudinal_speed_error_gain_over=longitudinal_cfg.get('speed_error_gain_over'),
            longitudinal_overspeed_accel_zero_threshold=float(
                longitudinal_cfg.get('overspeed_accel_zero_threshold', 0.0)
            ),
            longitudinal_overspeed_brake_threshold=float(
                longitudinal_cfg.get('overspeed_brake_threshold', 0.3)
            ),
            longitudinal_overspeed_brake_min=float(
                longitudinal_cfg.get('overspeed_brake_min', 0.02)
            ),
            longitudinal_overspeed_brake_max=float(
                longitudinal_cfg.get('overspeed_brake_max', 0.2)
            ),
            longitudinal_overspeed_brake_gain=float(
                longitudinal_cfg.get('overspeed_brake_gain', 0.2)
            ),
            longitudinal_accel_mode_threshold=float(longitudinal_cfg.get('accel_mode_threshold', 0.2)),
            longitudinal_decel_mode_threshold=float(longitudinal_cfg.get('decel_mode_threshold', 0.2)),
            longitudinal_speed_error_accel_threshold=float(longitudinal_cfg.get('speed_error_accel_threshold', 0.2)),
            longitudinal_speed_error_brake_threshold=float(longitudinal_cfg.get('speed_error_brake_threshold', -0.2)),
            longitudinal_mode_switch_min_time=float(longitudinal_cfg.get('mode_switch_min_time', 0.4)),
            longitudinal_coast_hold_seconds=float(longitudinal_cfg.get('coast_hold_seconds', 0.5)),
            longitudinal_coast_throttle_kp=float(longitudinal_cfg.get('coast_throttle_kp', 0.1)),
            longitudinal_coast_throttle_max=float(longitudinal_cfg.get('coast_throttle_max', 0.12)),
            longitudinal_straight_throttle_cap=float(longitudinal_cfg.get('straight_throttle_cap', 0.45)),
            longitudinal_straight_curvature_threshold=float(
                longitudinal_cfg.get('straight_curvature_threshold', 0.003)
            ),
            longitudinal_accel_tracking_enabled=bool(longitudinal_cfg.get('accel_tracking_enabled', True)),
            longitudinal_accel_tracking_error_scale=float(
                longitudinal_cfg.get('accel_tracking_error_scale', 1.0)
            ),
            longitudinal_accel_pid_kp=float(longitudinal_cfg.get('accel_pid_kp', 0.4)),
            longitudinal_accel_pid_ki=float(longitudinal_cfg.get('accel_pid_ki', 0.05)),
            longitudinal_accel_pid_kd=float(longitudinal_cfg.get('accel_pid_kd', 0.02)),
            longitudinal_throttle_curve_gamma=float(
                longitudinal_cfg.get('throttle_curve_gamma', 1.0)
            ),
            longitudinal_throttle_curve_min=float(
                longitudinal_cfg.get('throttle_curve_min', 0.0)
            ),
            longitudinal_speed_drag_gain=float(
                longitudinal_cfg.get('speed_drag_gain', 0.0)
            ),
            longitudinal_accel_target_smoothing_alpha=float(
                longitudinal_cfg.get('accel_target_smoothing_alpha', 0.6)
            ),
            longitudinal_continuous_accel_control=bool(
                longitudinal_cfg.get('continuous_accel_control', False)
            ),
            longitudinal_continuous_accel_deadband=float(
                longitudinal_cfg.get('continuous_accel_deadband', 0.05)
            ),
            longitudinal_startup_ramp_seconds=float(
                longitudinal_cfg.get('startup_ramp_seconds', 2.5)
            ),
            longitudinal_startup_accel_limit=float(
                longitudinal_cfg.get('startup_accel_limit', 1.2)
            ),
            longitudinal_startup_speed_threshold=float(
                longitudinal_cfg.get('startup_speed_threshold', 2.0)
            ),
            longitudinal_startup_throttle_cap=float(
                longitudinal_cfg.get('startup_throttle_cap', 0.0)
            ),
            longitudinal_startup_disable_accel_feedforward=bool(
                longitudinal_cfg.get('startup_disable_accel_feedforward', True)
            ),
            longitudinal_low_speed_accel_limit=float(
                longitudinal_cfg.get('low_speed_accel_limit', 0.0)
            ),
            longitudinal_low_speed_speed_threshold=float(
                longitudinal_cfg.get('low_speed_speed_threshold', 0.0)
            ),
            max_steering=lateral_cfg.get('max_steering', 0.5),
            lateral_deadband=lateral_cfg.get('deadband', 0.02),
            lateral_heading_weight=lateral_cfg.get('heading_weight', 0.5),
            lateral_lateral_weight=lateral_cfg.get('lateral_weight', 0.5),
            lateral_error_clip=lateral_cfg.get('error_clip', np.pi / 4),
            lateral_integral_limit=lateral_cfg.get('integral_limit', 0.10),  # FIXED: Read from config (default 0.10)
            steering_smoothing_alpha=lateral_cfg.get('steering_smoothing_alpha', 0.7),
            curve_feedforward_gain=lateral_cfg.get('curve_feedforward_gain', 1.0),
            curve_feedforward_threshold=lateral_cfg.get('curve_feedforward_threshold', 0.02),
            curve_feedforward_gain_min=lateral_cfg.get('curve_feedforward_gain_min', 1.0),
            curve_feedforward_gain_max=lateral_cfg.get('curve_feedforward_gain_max', 1.0),
            curve_feedforward_curvature_min=lateral_cfg.get('curve_feedforward_curvature_min', 0.005),
            curve_feedforward_curvature_max=lateral_cfg.get('curve_feedforward_curvature_max', 0.03),
            curve_feedforward_curvature_clamp=lateral_cfg.get('curve_feedforward_curvature_clamp', 0.03),
            curve_feedforward_bins=lateral_cfg.get('curve_feedforward_bins', []),
            curvature_scale_factor=lateral_cfg.get('curvature_scale_factor', 1.0),
            curvature_scale_threshold=lateral_cfg.get('curvature_scale_threshold', 0.0005),
            curvature_smoothing_alpha=lateral_cfg.get('curvature_smoothing_alpha', 0.7),
            curvature_transition_threshold=lateral_cfg.get('curvature_transition_threshold', 0.01),
            curvature_transition_alpha=lateral_cfg.get('curvature_transition_alpha', 0.3),
            straight_curvature_threshold=lateral_cfg.get('straight_curvature_threshold', 0.01),
            curve_upcoming_enter_threshold=float(
                lateral_cfg.get('curve_upcoming_enter_threshold', 0.012)
            ),
            curve_upcoming_exit_threshold=float(
                lateral_cfg.get('curve_upcoming_exit_threshold', 0.009)
            ),
            curve_upcoming_on_frames=int(lateral_cfg.get('curve_upcoming_on_frames', 2)),
            curve_upcoming_off_frames=int(lateral_cfg.get('curve_upcoming_off_frames', 2)),
            curve_phase_use_distance_track=bool(
                lateral_cfg.get('curve_phase_use_distance_track', False)
            ),
            curve_phase_track_name=lateral_cfg.get('curve_phase_track_name'),
            curve_at_car_distance_min_m=float(
                lateral_cfg.get('curve_at_car_distance_min_m', 0.0)
            ),
            road_curve_enter_threshold=lateral_cfg.get('road_curve_enter_threshold'),
            road_curve_exit_threshold=lateral_cfg.get('road_curve_exit_threshold'),
            road_straight_hold_invalid_frames=int(
                lateral_cfg.get('road_straight_hold_invalid_frames', 6)
            ),
            steering_rate_curvature_min=lateral_cfg.get('steering_rate_curvature_min', 0.005),
            steering_rate_curvature_max=lateral_cfg.get('steering_rate_curvature_max', 0.03),
            steering_rate_scale_min=lateral_cfg.get('steering_rate_scale_min', 0.5),
            curve_rate_floor_moderate_error=lateral_cfg.get('curve_rate_floor_moderate_error', 0.20),
            curve_rate_floor_large_error=lateral_cfg.get('curve_rate_floor_large_error', 0.24),
            straight_sign_flip_error_threshold=lateral_cfg.get('straight_sign_flip_error_threshold', 0.02),
            straight_sign_flip_rate=lateral_cfg.get('straight_sign_flip_rate', 0.2),
            straight_sign_flip_frames=lateral_cfg.get('straight_sign_flip_frames', 6),
            steering_jerk_limit=lateral_cfg.get('steering_jerk_limit', 0.0),
            steering_jerk_curve_scale_max=lateral_cfg.get('steering_jerk_curve_scale_max', 1.0),
            steering_jerk_curve_min=lateral_cfg.get('steering_jerk_curve_min', 0.003),
            steering_jerk_curve_max=lateral_cfg.get('steering_jerk_curve_max', 0.015),
            curve_entry_assist_enabled=bool(lateral_cfg.get('curve_entry_assist_enabled', False)),
            curve_entry_assist_error_min=float(lateral_cfg.get('curve_entry_assist_error_min', 0.30)),
            curve_entry_assist_heading_error_min=float(
                lateral_cfg.get('curve_entry_assist_heading_error_min', 0.10)
            ),
            curve_entry_assist_curvature_min=float(
                lateral_cfg.get('curve_entry_assist_curvature_min', 0.010)
            ),
            curve_entry_assist_rate_boost=float(lateral_cfg.get('curve_entry_assist_rate_boost', 1.15)),
            curve_entry_assist_jerk_boost=float(lateral_cfg.get('curve_entry_assist_jerk_boost', 1.10)),
            curve_entry_assist_max_frames=int(lateral_cfg.get('curve_entry_assist_max_frames', 18)),
            curve_entry_assist_watchdog_rate_delta_max=float(
                lateral_cfg.get('curve_entry_assist_watchdog_rate_delta_max', 0.22)
            ),
            curve_entry_assist_rearm_frames=int(lateral_cfg.get('curve_entry_assist_rearm_frames', 20)),
            curve_entry_schedule_enabled=bool(
                lateral_cfg.get('curve_entry_schedule_enabled', False)
            ),
            curve_entry_schedule_frames=int(
                lateral_cfg.get('curve_entry_schedule_frames', 18)
            ),
            curve_entry_schedule_min_rate=float(
                lateral_cfg.get('curve_entry_schedule_min_rate', 0.22)
            ),
            curve_entry_schedule_min_jerk=float(
                lateral_cfg.get('curve_entry_schedule_min_jerk', 0.14)
            ),
            curve_entry_schedule_min_hold_frames=int(
                lateral_cfg.get('curve_entry_schedule_min_hold_frames', 12)
            ),
            curve_entry_schedule_min_curve_progress_ratio=float(
                lateral_cfg.get('curve_entry_schedule_min_curve_progress_ratio', 0.20)
            ),
            curve_entry_schedule_fallback_only_when_dynamic=bool(
                lateral_cfg.get('curve_entry_schedule_fallback_only_when_dynamic', False)
            ),
            curve_entry_schedule_fallback_deficit_frames=int(
                lateral_cfg.get('curve_entry_schedule_fallback_deficit_frames', 6)
            ),
            curve_entry_schedule_fallback_rate_deficit_min=float(
                lateral_cfg.get('curve_entry_schedule_fallback_rate_deficit_min', 0.03)
            ),
            curve_entry_schedule_fallback_rearm_cooldown_frames=int(
                lateral_cfg.get('curve_entry_schedule_fallback_rearm_cooldown_frames', 18)
            ),
            curve_entry_schedule_handoff_transfer_ratio=float(
                lateral_cfg.get('curve_entry_schedule_handoff_transfer_ratio', 0.65)
            ),
            curve_entry_schedule_handoff_error_fall=float(
                lateral_cfg.get('curve_entry_schedule_handoff_error_fall', 0.03)
            ),
            dynamic_curve_authority_enabled=bool(
                lateral_cfg.get('dynamic_curve_authority_enabled', True)
            ),
            dynamic_curve_rate_deficit_deadband=float(
                lateral_cfg.get('dynamic_curve_rate_deficit_deadband', 0.01)
            ),
            dynamic_curve_rate_boost_gain=float(
                lateral_cfg.get('dynamic_curve_rate_boost_gain', 1.0)
            ),
            dynamic_curve_rate_boost_max=float(
                lateral_cfg.get('dynamic_curve_rate_boost_max', 0.30)
            ),
            dynamic_curve_jerk_boost_gain=float(
                lateral_cfg.get('dynamic_curve_jerk_boost_gain', 4.0)
            ),
            dynamic_curve_jerk_boost_max_factor=float(
                lateral_cfg.get('dynamic_curve_jerk_boost_max_factor', 3.5)
            ),
            dynamic_curve_hard_clip_boost_gain=float(
                lateral_cfg.get('dynamic_curve_hard_clip_boost_gain', 1.0)
            ),
            dynamic_curve_hard_clip_boost_max=float(
                lateral_cfg.get('dynamic_curve_hard_clip_boost_max', 0.12)
            ),
            dynamic_curve_entry_governor_enabled=bool(
                lateral_cfg.get('dynamic_curve_entry_governor_enabled', True)
            ),
            dynamic_curve_entry_governor_gain=float(
                lateral_cfg.get('dynamic_curve_entry_governor_gain', 1.2)
            ),
            dynamic_curve_entry_governor_max_scale=float(
                lateral_cfg.get('dynamic_curve_entry_governor_max_scale', 1.8)
            ),
            dynamic_curve_entry_governor_stale_floor_scale=float(
                lateral_cfg.get('dynamic_curve_entry_governor_stale_floor_scale', 1.15)
            ),
            dynamic_curve_entry_governor_exclusive_mode=bool(
                lateral_cfg.get('dynamic_curve_entry_governor_exclusive_mode', True)
            ),
            dynamic_curve_entry_governor_anticipatory_enabled=bool(
                lateral_cfg.get('dynamic_curve_entry_governor_anticipatory_enabled', True)
            ),
            dynamic_curve_entry_governor_upcoming_phase_weight=float(
                lateral_cfg.get('dynamic_curve_entry_governor_upcoming_phase_weight', 0.55)
            ),
            dynamic_curve_authority_precurve_enabled=bool(
                lateral_cfg.get('dynamic_curve_authority_precurve_enabled', True)
            ),
            dynamic_curve_authority_precurve_scale=float(
                lateral_cfg.get('dynamic_curve_authority_precurve_scale', 0.8)
            ),
            dynamic_curve_single_owner_mode=bool(
                lateral_cfg.get('dynamic_curve_single_owner_mode', True)
            ),
            dynamic_curve_single_owner_min_rate=float(
                lateral_cfg.get('dynamic_curve_single_owner_min_rate', 0.22)
            ),
            dynamic_curve_single_owner_min_jerk=float(
                lateral_cfg.get('dynamic_curve_single_owner_min_jerk', 0.6)
            ),
            dynamic_curve_comfort_lat_accel_comfort_max_g=float(
                lateral_cfg.get('dynamic_curve_comfort_lat_accel_comfort_max_g', 0.18)
            ),
            dynamic_curve_comfort_lat_accel_peak_max_g=float(
                lateral_cfg.get('dynamic_curve_comfort_lat_accel_peak_max_g', 0.25)
            ),
            dynamic_curve_comfort_lat_jerk_comfort_max_gps=float(
                lateral_cfg.get('dynamic_curve_comfort_lat_jerk_comfort_max_gps', 0.30)
            ),
            dynamic_curve_lat_jerk_smoothing_alpha=float(
                lateral_cfg.get('dynamic_curve_lat_jerk_smoothing_alpha', 0.25)
            ),
            dynamic_curve_lat_jerk_soft_start_ratio=float(
                lateral_cfg.get('dynamic_curve_lat_jerk_soft_start_ratio', 0.60)
            ),
            dynamic_curve_lat_jerk_soft_floor_scale=float(
                lateral_cfg.get('dynamic_curve_lat_jerk_soft_floor_scale', 0.35)
            ),
            dynamic_curve_speed_low_mps=float(
                lateral_cfg.get('dynamic_curve_speed_low_mps', 4.0)
            ),
            dynamic_curve_speed_high_mps=float(
                lateral_cfg.get('dynamic_curve_speed_high_mps', 10.0)
            ),
            dynamic_curve_speed_boost_max_scale=float(
                lateral_cfg.get('dynamic_curve_speed_boost_max_scale', 1.4)
            ),
            turn_feasibility_governor_enabled=bool(
                lateral_cfg.get('turn_feasibility_governor_enabled', True)
            ),
            turn_feasibility_curvature_min=float(
                lateral_cfg.get('turn_feasibility_curvature_min', 0.002)
            ),
            turn_feasibility_guardband_g=float(
                lateral_cfg.get('turn_feasibility_guardband_g', 0.015)
            ),
            turn_feasibility_use_peak_bound=bool(
                lateral_cfg.get('turn_feasibility_use_peak_bound', True)
            ),
            curve_unwind_policy_enabled=bool(
                lateral_cfg.get('curve_unwind_policy_enabled', False)
            ),
            curve_unwind_frames=int(
                lateral_cfg.get('curve_unwind_frames', 12)
            ),
            curve_unwind_rate_scale_start=float(
                lateral_cfg.get('curve_unwind_rate_scale_start', 1.0)
            ),
            curve_unwind_rate_scale_end=float(
                lateral_cfg.get('curve_unwind_rate_scale_end', 0.8)
            ),
            curve_unwind_jerk_scale_start=float(
                lateral_cfg.get('curve_unwind_jerk_scale_start', 1.0)
            ),
            curve_unwind_jerk_scale_end=float(
                lateral_cfg.get('curve_unwind_jerk_scale_end', 0.7)
            ),
            curve_unwind_integral_decay=float(
                lateral_cfg.get('curve_unwind_integral_decay', 0.85)
            ),
            curve_commit_mode_enabled=bool(
                lateral_cfg.get('curve_commit_mode_enabled', False)
            ),
            curve_commit_mode_max_frames=int(
                lateral_cfg.get('curve_commit_mode_max_frames', 20)
            ),
            curve_commit_mode_min_rate=float(
                lateral_cfg.get('curve_commit_mode_min_rate', 0.22)
            ),
            curve_commit_mode_min_jerk=float(
                lateral_cfg.get('curve_commit_mode_min_jerk', 0.14)
            ),
            curve_commit_mode_transfer_ratio_target=float(
                lateral_cfg.get('curve_commit_mode_transfer_ratio_target', 0.72)
            ),
            curve_commit_mode_error_fall=float(
                lateral_cfg.get('curve_commit_mode_error_fall', 0.03)
            ),
            curve_commit_mode_exit_consecutive_frames=int(
                lateral_cfg.get('curve_commit_mode_exit_consecutive_frames', 4)
            ),
            curve_commit_mode_retrigger_on_dynamic_deficit=bool(
                lateral_cfg.get('curve_commit_mode_retrigger_on_dynamic_deficit', True)
            ),
            curve_commit_mode_dynamic_deficit_frames=int(
                lateral_cfg.get('curve_commit_mode_dynamic_deficit_frames', 8)
            ),
            curve_commit_mode_dynamic_deficit_min=float(
                lateral_cfg.get('curve_commit_mode_dynamic_deficit_min', 0.03)
            ),
            curve_commit_mode_retrigger_cooldown_frames=int(
                lateral_cfg.get('curve_commit_mode_retrigger_cooldown_frames', 12)
            ),
            speed_gain_min_speed=lateral_cfg.get('speed_gain_min_speed', 4.0),
            speed_gain_max_speed=lateral_cfg.get('speed_gain_max_speed', 10.0),
            speed_gain_min=lateral_cfg.get('speed_gain_min', 1.0),
            speed_gain_max=lateral_cfg.get('speed_gain_max', 1.2),
            speed_gain_curvature_min=lateral_cfg.get('speed_gain_curvature_min', 0.002),
            speed_gain_curvature_max=lateral_cfg.get('speed_gain_curvature_max', 0.015),
            control_mode=lateral_cfg.get('control_mode', 'pid'),
            stanley_k=lateral_cfg.get('stanley_k', 1.0),
            stanley_soft_speed=lateral_cfg.get('stanley_soft_speed', 2.0),
            stanley_heading_weight=lateral_cfg.get('stanley_heading_weight', 1.0),
            pp_feedback_gain=lateral_cfg.get('pp_feedback_gain', 0.15),
            pp_min_lookahead=lateral_cfg.get('pp_min_lookahead', 0.5),
            pp_ref_jump_clamp=lateral_cfg.get('pp_ref_jump_clamp', 0.5),
            pp_stale_decay=lateral_cfg.get('pp_stale_decay', 0.98),
            pp_max_steering_rate=lateral_cfg.get('pp_max_steering_rate', 0.4),
            feedback_gain_min=lateral_cfg.get('feedback_gain_min', 1.0),
            feedback_gain_max=lateral_cfg.get('feedback_gain_max', 1.2),
            feedback_gain_curvature_min=lateral_cfg.get('feedback_gain_curvature_min', 0.002),
            feedback_gain_curvature_max=lateral_cfg.get('feedback_gain_curvature_max', 0.015),
            curvature_stale_hold_seconds=lateral_cfg.get('curvature_stale_hold_seconds', 0.30),
            curvature_stale_hold_min_abs=lateral_cfg.get('curvature_stale_hold_min_abs', 0.0005),
            base_error_smoothing_alpha=lateral_cfg.get('base_error_smoothing_alpha', 0.7),
            heading_error_smoothing_alpha=lateral_cfg.get('heading_error_smoothing_alpha', 0.45),
            straight_window_frames=int(lateral_cfg.get('straight_window_frames', 60)),
            straight_oscillation_high=lateral_cfg.get('straight_oscillation_high', 0.20),
            straight_oscillation_low=lateral_cfg.get('straight_oscillation_low', 0.05),
        )
        
        # Store config for use in _process_frame
        self.config = config
        self.control_config = control_cfg
        self.trajectory_config = trajectory_cfg
        self.trajectory_source = str(trajectory_cfg.get('trajectory_source', 'planner')).lower()
        self.safety_config = safety_cfg
        self.curve_mode_speed_cap_enabled = bool(
            lateral_cfg.get('curve_mode_speed_cap_enabled', False)
        )
        self.curve_mode_speed_cap_mps = float(
            lateral_cfg.get('curve_mode_speed_cap_mps', 7.0)
        )
        self.curve_mode_speed_cap_min_ratio = float(
            lateral_cfg.get('curve_mode_speed_cap_min_ratio', 0.55)
        )
        self.record_segmentation_mask = bool(
            perception_cfg.get("record_segmentation_mask", False)
        )

        # Speed planner (jerk-limited) configuration
        self.speed_planner_enabled = bool(speed_planner_cfg.get('enabled', False))
        self.speed_planner = None
        self.speed_planner_speed_limit_bias = float(speed_planner_cfg.get('speed_limit_bias', 0.0))
        if self.speed_planner_enabled:
            default_dt = float(speed_planner_cfg.get('default_dt', 1.0 / 30.0))
            planner_config = SpeedPlannerConfig(
                max_accel=float(speed_planner_cfg.get('max_accel', 2.0)),
                max_decel=float(speed_planner_cfg.get('max_decel', 2.5)),
                max_jerk=float(speed_planner_cfg.get('max_jerk', 1.2)),
                max_jerk_min=float(speed_planner_cfg.get('max_jerk_min', 0.0)),
                max_jerk_max=float(speed_planner_cfg.get('max_jerk_max', 0.0)),
                jerk_error_min=float(speed_planner_cfg.get('jerk_error_min', 0.0)),
                jerk_error_max=float(speed_planner_cfg.get('jerk_error_max', 0.0)),
                min_speed=float(speed_planner_cfg.get('min_speed', 0.0)),
                launch_speed_floor=float(speed_planner_cfg.get('launch_speed_floor', 0.0)),
                launch_speed_floor_threshold=float(speed_planner_cfg.get('launch_speed_floor_threshold', 0.0)),
                reset_gap_seconds=float(speed_planner_cfg.get('reset_gap_seconds', 0.5)),
                sync_speed_threshold=float(speed_planner_cfg.get('sync_speed_threshold', 3.0)),
                default_dt=default_dt,
                speed_error_gain=float(speed_planner_cfg.get('speed_error_gain', 0.0)),
                speed_error_max_delta=float(speed_planner_cfg.get('speed_error_max_delta', 0.0)),
                speed_error_deadband=float(speed_planner_cfg.get('speed_error_deadband', 0.0)),
                sync_under_target=bool(speed_planner_cfg.get('sync_under_target', True)),
                sync_under_target_error=float(speed_planner_cfg.get('sync_under_target_error', 0.2)),
                use_speed_dependent_limits=bool(
                    speed_planner_cfg.get('use_speed_dependent_limits', False)
                ),
                accel_speed_min=float(speed_planner_cfg.get('accel_speed_min', 0.0)),
                accel_speed_max=float(speed_planner_cfg.get('accel_speed_max', 0.0)),
                max_accel_at_speed_min=float(
                    speed_planner_cfg.get('max_accel_at_speed_min', 0.0)
                ),
                max_accel_at_speed_max=float(
                    speed_planner_cfg.get('max_accel_at_speed_max', 0.0)
                ),
                decel_speed_min=float(speed_planner_cfg.get('decel_speed_min', 0.0)),
                decel_speed_max=float(speed_planner_cfg.get('decel_speed_max', 0.0)),
                max_decel_at_speed_min=float(
                    speed_planner_cfg.get('max_decel_at_speed_min', 0.0)
                ),
                max_decel_at_speed_max=float(
                    speed_planner_cfg.get('max_decel_at_speed_max', 0.0)
                ),
                enforce_desired_speed_slew=bool(
                    speed_planner_cfg.get('enforce_desired_speed_slew', False)
                ),
            )
            self.speed_planner = SpeedPlanner(planner_config)

        # Speed governor: clean replacement for 12-layer speed suppression cascade
        self.speed_governor = build_speed_governor(trajectory_cfg, speed_planner_cfg)
        self.use_speed_governor = bool(trajectory_cfg.get('speed_governor', {}).get('enabled', True))
        self.consecutive_stale_frames = 0
        self.stale_speed_hold_threshold = int(
            trajectory_cfg.get('speed_governor', {}).get('stale_speed_hold_frames', 3)
        )

        # Data recording (enabled by default for comprehensive logging)
        self.recorder = None
        if record_data:
            self.recorder = DataRecorder(recording_dir)
            logger.info(f"Data recording enabled: {self.recorder.output_file}")
            self.recorder.metadata["segmentation_enabled"] = bool(use_segmentation)
            if segmentation_model_path:
                self.recorder.metadata["segmentation_checkpoint"] = str(
                    segmentation_model_path
                )
            self.recorder.metadata["record_segmentation_mask"] = bool(
                self.record_segmentation_mask
            )
            # Attach provenance metadata so recording selection/compare can be release-aware.
            git_sha_full = "unknown"
            git_sha_short = "unknown"
            try:
                repo_root = Path(__file__).resolve().parent
                git_sha_full = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(repo_root),
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip() or "unknown"
                git_sha_short = git_sha_full[:8] if git_sha_full != "unknown" else "unknown"
            except Exception:
                pass

            try:
                cfg_fingerprint = hashlib.sha256(
                    json.dumps(self.config or {}, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest()
            except Exception:
                cfg_fingerprint = "unknown"

            replay_type = os.getenv("AV_REPLAY_TYPE", "live")
            candidate_label = os.getenv("AV_CANDIDATE_LABEL", "candidate")
            policy_profile = os.getenv("AV_POLICY_PROFILE", "unknown")
            software_version = os.getenv("AV_SOFTWARE_VERSION", "unknown")
            build_label = os.getenv("AV_BUILD_LABEL", "unknown")
            track_id = os.getenv("AV_TRACK_ID", "unknown")
            duration_target_s = float(os.getenv("AV_DURATION_TARGET_S", "0") or 0.0)
            start_t = float(os.getenv("AV_START_T", "0") or 0.0)
            run_command = os.getenv("AV_RUN_COMMAND", "")

            self.recorder.metadata["recording_provenance"] = {
                "software_version": software_version,
                "build_label": build_label,
                "git_sha_full": git_sha_full,
                "git_sha_short": git_sha_short,
                "config_fingerprint_sha256": cfg_fingerprint,
                "replay_type": replay_type,
                "policy_profile": policy_profile,
                "track_id": track_id,
                "duration_target_s": duration_target_s,
                "start_t": start_t,
                "run_command": run_command,
                "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "analyze_to_failure_default": True,
                "notes": "",
                "candidate_label": candidate_label,
            }
        else:
            logger.info("Data recording disabled")
        
        # State
        self.running = False
        self.frame_count = 0
        self.target_fps = 30.0  # Match Unity's frame rate
        self.frame_interval = 1.0 / self.target_fps

        # Initialize previous lane history for stale-data fallback paths
        self.previous_left_lane_x = None
        self.previous_right_lane_x = None
        self.low_visibility_fallback_streak = 0
        self.lane_center_ema = None
        self.lane_width_ema = None
        self.last_gt_lane_width = None
        self.lane_center_ab = None
        self.lane_center_ab_vel = None
        self.lane_width_ab = None
        self.lane_width_ab_vel = None
        self.last_lane_filter_time = None
        
        # Track if emergency stop has been logged (to prevent repeated messages)
        self.emergency_stop_logged = False
        self.emergency_stop_type = None  # Track which type of emergency stop (for logging)
        self.emergency_stop_latched = False
        self.emergency_stop_latched_since_wall_time: Optional[float] = None

        # Teleport/jump guard state
        self.last_vehicle_position = None
        self.last_vehicle_timestamp = None
        self.teleport_guard_frames_remaining = 0
        self.last_teleport_distance = None
        self.last_teleport_dt = None
        self.post_jump_cooldown_frames = 0
        
        # NEW: Perception health monitoring
        self.consecutive_bad_detection_frames = 0  # Track consecutive frames with <2 lanes
        self.perception_health_history = []  # Recent detection quality (True/False for good/bad)
        self.perception_health_history_size = 60  # Track last 60 frames (~2 seconds at 30 FPS)
        self.perception_health_score = 1.0  # Current health score (1.0 = perfect, 0.0 = failed)
        self.perception_health_status = "healthy"  # "healthy", "degraded", "poor", "critical"
        self.cv_instability_breach_frames = 0
        self.original_target_speed = trajectory_cfg.get('target_speed', 8.0)  # Store original target speed for recovery
        # Speed limit smoothing state
        self.last_speed_limit: Optional[float] = None
        self.last_speed_limit_time: Optional[float] = None
        # Speed limit preview hold state (prevents surge/slow oscillations before curves)
        self.preview_hold_target: Optional[float] = None
        self.preview_hold_active = False
        self.preview_hold_curvature_threshold = float(
            trajectory_cfg.get('speed_limit_preview_hold_curvature_threshold', 0.01)
        )
        self.preview_hold_release_delta = float(
            trajectory_cfg.get('speed_limit_preview_hold_release_delta', 0.5)
        )
        self.preview_hold_min_distance_margin = float(
            trajectory_cfg.get('speed_limit_preview_hold_min_distance_margin', 0.5)
        )
        self.preview_hold_release_frames = int(
            trajectory_cfg.get('speed_limit_preview_hold_release_frames', 0)
        )
        self.preview_hold_release_count = 0
        self.preview_clamp_active = False
        self.preview_clamp_target: Optional[float] = None
        # Target speed blending based on distance traveled (reduces surge/slow on short straights)
        self.target_speed_blend_window_m = float(
            trajectory_cfg.get('target_speed_blend_window_m', 0.0)
        )
        self.target_speed_blend_reset_seconds = float(
            trajectory_cfg.get('target_speed_blend_reset_seconds', 0.5)
        )
        self.target_speed_blend_prev: Optional[float] = None
        self.target_speed_blend_last_time: Optional[float] = None
        # Straight-segment target speed smoothing state (reduce oscillations on straights)
        self.straight_target_speed_smoothed: Optional[float] = None
        self.straight_speed_smoothing_alpha = float(
            trajectory_cfg.get('straight_speed_smoothing_alpha', 0.0)
        )
        self.straight_speed_smoothing_curvature_threshold = float(
            trajectory_cfg.get('straight_speed_smoothing_curvature_threshold', 0.01)
        )
        self.straight_speed_slew_rate_up = float(
            trajectory_cfg.get('straight_speed_slew_rate_up', 0.0)
        )
        self.straight_speed_last_time: Optional[float] = None
        self.straight_target_hold_seconds = float(
            trajectory_cfg.get('straight_target_hold_seconds', 0.0)
        )
        self.straight_target_hold_limit_delta = float(
            trajectory_cfg.get('straight_target_hold_limit_delta', 0.2)
        )
        self.straight_target_hold_target: Optional[float] = None
        self.straight_target_hold_start_time: Optional[float] = None
        self.straight_target_hold_last_limit: Optional[float] = None
        self.straight_target_hold_last_preview: Optional[float] = None
        # Steering saturation speed guard (prevents overshoot on straights leading into curves)
        self.steering_speed_guard_enabled = bool(
            trajectory_cfg.get('steering_speed_guard_enabled', False)
        )
        self.steering_speed_guard_threshold = float(
            trajectory_cfg.get('steering_speed_guard_threshold', 0.0)
        )
        self.steering_speed_guard_scale = float(
            trajectory_cfg.get('steering_speed_guard_scale', 1.0)
        )
        self.steering_speed_guard_min_speed = float(
            trajectory_cfg.get('steering_speed_guard_min_speed', 0.0)
        )
        self.last_steering_command: Optional[float] = None
        # Target speed smoothing state
        self.last_target_speed: Optional[float] = None
        self.last_target_speed_time: Optional[float] = None
        self.restart_ramp_start_time: Optional[float] = None
        # Launch throttle ramp state
        self.launch_throttle_ramp_start_time: Optional[float] = None
        self.launch_throttle_ramp_armed: bool = True
        self.launch_stop_candidate_start_time: Optional[float] = None

        # Curvature smoothing state (distance-based window)
        self.curvature_smoothing_enabled = bool(
            trajectory_cfg.get('curvature_smoothing_enabled', False)
        )
        self.curvature_smoothing_window_m = float(
            trajectory_cfg.get('curvature_smoothing_window_m', 12.0)
        )
        # Preview stability filter (avoid oscillations from transient preview limits)
        self.preview_limit_stability_frames = int(
            trajectory_cfg.get('speed_limit_preview_stability_frames', 0)
        )
        self.preview_limit_change_delta = float(
            trajectory_cfg.get('speed_limit_preview_change_delta', 0.2)
        )
        self.preview_limit_last: Optional[float] = None
        self.preview_limit_streak = 0
        self.curvature_smoothing_min_speed = float(
            trajectory_cfg.get('curvature_smoothing_min_speed', 2.0)
        )
        # Curve speed preview (lookahead ramp before curve entry)
        self.curve_speed_preview_enabled = bool(
            trajectory_cfg.get('curve_speed_preview_enabled', False)
        )
        self.curve_speed_preview_lookahead_scale = float(
            trajectory_cfg.get('curve_speed_preview_lookahead_scale', 1.5)
        )
        self.curve_speed_preview_distance = float(
            trajectory_cfg.get('curve_speed_preview_distance', 12.0)
        )
        self.curve_speed_preview_decel = float(
            trajectory_cfg.get('curve_speed_preview_decel', 2.5)
        )
        self.curve_speed_preview_min_curvature = float(
            trajectory_cfg.get('curve_speed_preview_min_curvature', 0.002)
        )
        self.smoothed_path_curvature: Optional[float] = None
        self.last_curvature_time: Optional[float] = None
    
    def run(self, max_frames: Optional[int] = None, duration: Optional[float] = None):
        """
        Run AV stack main loop.
        
        Args:
            max_frames: Maximum number of frames to process (None for infinite)
            duration: Maximum duration in seconds (None for infinite)
        """
        logger.info("Starting AV Stack...")
        
        # Check bridge connection with retry
        if not self._wait_for_bridge(max_retries=10, initial_delay=1.0):
            logger.error("Bridge server is not available after retries!")
            logger.error("Please start the bridge server: python -m bridge.server")
            return
        
        logger.info("Bridge server connected")
        
        # Wait for first frame from Unity (with exponential backoff)
        logger.info("Waiting for Unity to send first frame...")
        if not self._wait_for_first_frame(max_retries=30, initial_delay=0.5):
            logger.warning("No frames received from Unity. Starting anyway...")
        else:
            logger.info("First frame received! Starting main loop...")
        
        self.running = True
        self.frame_count = 0
        camera_wait_start: Optional[float] = None
        last_camera_wall_time: Optional[float] = None
        last_camera_timestamp: Optional[float] = None
        last_loop_time: Optional[float] = None
        last_frame_time = time.time()
        start_time = time.time()
        
        try:
            while self.running:
                loop_start = time.time()
                # Check frame limit
                if max_frames and self.frame_count >= max_frames:
                    logger.info(f"Reached frame limit: {max_frames}")
                    break

                # Optional early-stop: end run a short time after emergency stop latches.
                end_after_emergency_s = float(
                    self.safety_config.get('emergency_stop_end_run_after_seconds', 0.0)
                )
                if (
                    end_after_emergency_s > 0.0
                    and self.emergency_stop_latched
                    and self.emergency_stop_latched_since_wall_time is not None
                ):
                    elapsed_since_latch = time.time() - self.emergency_stop_latched_since_wall_time
                    if elapsed_since_latch >= end_after_emergency_s:
                        logger.info(
                            f"Ending run {end_after_emergency_s:.1f}s after emergency stop latch "
                            f"(elapsed={elapsed_since_latch:.2f}s)"
                        )
                        break
                
                # Check duration limit
                if duration is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= duration:
                        logger.info(f"Reached duration limit: {duration}s")
                        break
                
                # Rate limiting: maintain target FPS
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < self.frame_interval:
                    time.sleep(self.frame_interval - elapsed)
                
                last_frame_time = time.time()
                
                # Get latest camera frame
                camera_fetch_start = time.time()
                frame_data = self.bridge.get_latest_camera_frame_with_metadata()
                camera_fetch_duration = time.time() - camera_fetch_start
                if camera_fetch_duration > 0.1:
                    logger.warning(
                        "[CAMERA_FETCH_SLOW] duration=%.3fs frame=%s",
                        camera_fetch_duration,
                        self.frame_count,
                    )
                if frame_data is None:
                    if camera_wait_start is None:
                        camera_wait_start = time.time()
                    # No frame available - don't spam, just wait
                    time.sleep(self.frame_interval)
                    continue
                if camera_wait_start is not None:
                    wait_duration = time.time() - camera_wait_start
                    if wait_duration > 0.2:
                        logger.warning(
                            "[CAMERA_WAIT_GAP] duration=%.3fs frame=%s",
                            wait_duration,
                            self.frame_count,
                        )
                    camera_wait_start = None
                
                image, timestamp, camera_frame_id, camera_frame_meta = frame_data
                now = time.time()
                if last_camera_wall_time is not None and last_camera_timestamp is not None:
                    wall_gap = now - last_camera_wall_time
                    ts_gap = timestamp - last_camera_timestamp
                    if ts_gap > 0.2:
                        logger.warning(
                            "[CAMERA_TIMESTAMP_GAP] ts_gap=%.3fs wall_gap=%.3fs frame=%s",
                            ts_gap,
                            wall_gap,
                            self.frame_count,
                        )
                    drift = ts_gap - wall_gap
                    if abs(drift) > 0.2:
                        logger.warning(
                            "[CAMERA_TIME_DRIFT] ts_gap=%.3fs wall_gap=%.3fs drift=%.3fs frame=%s",
                            ts_gap,
                            wall_gap,
                            drift,
                            self.frame_count,
                        )
                last_camera_wall_time = now
                last_camera_timestamp = timestamp

                if camera_frame_id is not None:
                    if getattr(self, "last_camera_frame_id", None) == camera_frame_id:
                        if self.frame_count % 30 == 0:
                            logger.debug(
                                f"[Frame {self.frame_count}] Skipping duplicate camera frame "
                                f"(camera_frame_id={camera_frame_id})."
                            )
                        continue
                    self.last_camera_frame_id = camera_frame_id
                
                # Get vehicle state
                vehicle_state_dict = self.bridge.get_latest_vehicle_state()
                if vehicle_state_dict is None:
                    time.sleep(self.frame_interval)
                    continue
                
                # Process frame
                self._process_frame(
                    image,
                    timestamp,
                    vehicle_state_dict,
                    camera_frame_id=camera_frame_id,
                    camera_frame_meta=camera_frame_meta,
                )
                
                self.frame_count += 1

                if last_loop_time is not None:
                    loop_duration = time.time() - loop_start
                    if loop_duration > 0.2:
                        logger.warning(
                            "[LOOP_SLOW] duration=%.3fs frame=%s",
                            loop_duration,
                            self.frame_count,
                        )
                last_loop_time = loop_start
        
        except KeyboardInterrupt:
            logger.info("\nStopping AV Stack...")
        finally:
            self.stop()
    
    def _wait_for_bridge(self, max_retries: int = 10, initial_delay: float = 1.0) -> bool:
        """Wait for bridge server to be available with exponential backoff."""
        delay = initial_delay
        for attempt in range(max_retries):
            if self.bridge.health_check():
                return True
            logger.info(f"Waiting for bridge server... (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
            delay = min(delay * 1.5, 5.0)  # Exponential backoff, max 5s
        return False
    
    def _wait_for_first_frame(self, max_retries: int = 30, initial_delay: float = 0.5) -> bool:
        """Wait for first frame from Unity with exponential backoff."""
        delay = initial_delay
        for attempt in range(max_retries):
            frame_data = self.bridge.get_latest_camera_frame()
            if frame_data is not None:
                logger.info(f"First frame received after {attempt + 1} attempts")
                return True
            if attempt % 5 == 0:  # Log every 5 attempts
                logger.info(f"Waiting for Unity frames... (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
            delay = min(delay * 1.2, 2.0)  # Exponential backoff, max 2s
        return False
    
    def _process_frame(
        self,
        image: np.ndarray,
        timestamp: float,
        vehicle_state_dict: dict,
        camera_frame_id: int | None = None,
        camera_frame_meta: dict | None = None,
        topdown_frame_data: tuple[np.ndarray, float, int | None] | None = None,
    ):
        """
        Process a single frame through the AV stack.
        
        Args:
            image: Camera image
            timestamp: Frame timestamp
            vehicle_state_dict: Vehicle state dictionary
        """
        process_start = time.time()
        unity_frame_count = vehicle_state_dict.get(
            "unityFrameCount", vehicle_state_dict.get("unity_frame_count")
        )
        if unity_frame_count is not None:
            if self.last_processed_unity_frame_count == unity_frame_count:
                if self.frame_count % 30 == 0:
                    logger.debug(
                        f"[Frame {self.frame_count}] Skipping duplicate Unity frame "
                        f"(unity_frame_count={unity_frame_count})."
                    )
                return
            self.last_processed_unity_frame_count = unity_frame_count
        # Track timestamp for perception frozen detection
        # CRITICAL: Save previous timestamp BEFORE updating (we need it for comparisons)
        prev_timestamp = self.last_timestamp if hasattr(self, 'last_timestamp') else None
        # Update last_timestamp at the END of processing (after all timestamp comparisons)

        # Compute control dt from timestamps (fallback to target FPS)
        raw_dt = None
        if prev_timestamp is not None:
            raw_dt = float(timestamp) - float(prev_timestamp)
        default_control_dt = 1.0 / float(getattr(self, "target_fps", 30.0))
        if raw_dt is not None and 0.0 < raw_dt < 0.5:
            control_dt = raw_dt
        else:
            control_dt = default_control_dt
        if self.frame_count % 120 == 0:
            logger.debug(
                f"[Frame {self.frame_count}] Longitudinal dt={control_dt:.4f}s "
                f"(raw_dt={raw_dt if raw_dt is not None else 'N/A'})"
            )

        # Teleport/jump guard: detect sudden vehicle position discontinuities
        current_position = self._extract_position(vehicle_state_dict)
        teleport_distance_threshold = self.safety_config.get('teleport_distance_threshold', 2.0)
        teleport_detected, teleport_distance = _is_teleport_jump(
            self.last_vehicle_position,
            current_position,
            teleport_distance_threshold,
        )
        if teleport_detected:
            dt = None
            if self.last_vehicle_timestamp is not None:
                dt = timestamp - self.last_vehicle_timestamp
            self.teleport_guard_frames_remaining = int(
                self.safety_config.get('teleport_guard_frames', 1)
            )
            self.last_teleport_distance = teleport_distance
            self.last_teleport_dt = dt
            self.controller.reset()
            if hasattr(self.trajectory_planner, 'last_smoothed_ref_point'):
                self.trajectory_planner.last_smoothed_ref_point = None
            if hasattr(self.trajectory_planner, 'smoothed_bias_history'):
                self.trajectory_planner.smoothed_bias_history = []
            if hasattr(self.trajectory_planner, 'last_timestamp'):
                self.trajectory_planner.last_timestamp = None
            self.post_jump_cooldown_frames = int(
                self.safety_config.get('post_jump_cooldown_frames', 30)
            )
            logger.warning(
                f"[Frame {self.frame_count}] [TELEPORT GUARD] Position jump detected "
                f"({teleport_distance:.2f}m, dt={dt if dt is not None else 'N/A'}s). "
                "Resetting controllers and skipping emergency stop for this frame."
            )
        teleport_guard_active = self.teleport_guard_frames_remaining > 0
        
        # 1. Perception: Detect lanes
        # detect() now returns: (lane_coeffs, confidence, detection_method, num_lanes_detected)
        perception_start = time.time()
        perception_result = self.perception.detect(image)
        perception_duration = time.time() - perception_start
        if len(perception_result) == 4:
            lane_coeffs, confidence, detection_method, num_lanes_detected = perception_result
        else:
            # Fallback for old format (2 values)
            lane_coeffs, confidence = perception_result
            num_lanes_detected = sum(1 for c in lane_coeffs if c is not None)
            # Determine detection method (check if CV fallback was used)
            detection_method = "ml"  # Default
            if hasattr(self.perception, 'last_detection_method'):
                detection_method = self.perception.last_detection_method
            elif not hasattr(self.perception, 'model_trained') or not self.perception.model_trained:
                detection_method = "cv"  # Likely using CV fallback
        
        # DEBUG: Save visualization frames periodically to diagnose road edge detection
        frame_count = getattr(self, '_frame_count', 0)
        self._frame_count = frame_count + 1
        enable_debug_vis = (frame_count % 30 == 0)  # Every 30 frames (1 second at 30fps)
        
        # Check if we should create debug visualization
        # Try both direct cv_detector access and through inference
        should_debug = enable_debug_vis and detection_method == "cv"
        cv_detector = None
        if hasattr(self.perception, 'cv_detector'):
            cv_detector = self.perception.cv_detector
        elif hasattr(self.perception, 'detector') and hasattr(self.perception.detector, 'cv_detector'):
            cv_detector = self.perception.detector.cv_detector
        
        # DEBUG: Log why debug visualization might not run
        if enable_debug_vis:
            logger.info(f"[DEBUG VIS] Frame {frame_count}: enable_debug_vis={enable_debug_vis}, "
                       f"detection_method={detection_method}, has_cv_detector={cv_detector is not None}, "
                       f"should_debug={should_debug}")
        
        # Store fit_points from debug_info (if available)
        fit_points_left = None
        fit_points_right = None
        
        if should_debug and cv_detector is not None:
            try:
                logger.info(f"[DEBUG VIS] Creating debug visualization for frame {frame_count}")
                # Re-detect with debug info
                # detect() returns (lane_coeffs, debug_info) when return_debug=True
                result = cv_detector.detect(image, return_debug=True)
                logger.info(f"[DEBUG VIS] detect() returned type: {type(result)}, length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
                if isinstance(result, tuple) and len(result) == 2:
                    cv_lanes, debug_info = result
                    logger.info(f"[DEBUG VIS] Unpacked: cv_lanes type={type(cv_lanes)}, debug_info type={type(debug_info)}")
                    # Extract fit_points from debug_info
                    if isinstance(debug_info, dict) and 'fit_points' in debug_info:
                        fit_points = debug_info['fit_points']
                        if isinstance(fit_points, dict):
                            fit_points_left = fit_points.get('left')
                            fit_points_right = fit_points.get('right')
                else:
                    # Fallback if return format is different
                    logger.warning(f"[DEBUG VIS] Unexpected return format: {type(result)}")
                    cv_lanes = result if isinstance(result, list) else [None, None]
                    debug_info = {}
            except Exception as e:
                logger.error(f"[DEBUG VIS] Error getting debug info: {e}")
                debug_info = {}
                fit_points_left = None
                fit_points_right = None
        else:
            # Even if not creating debug vis, try to get fit_points for recording
            # Only do this if CV detector is available and we're using CV method
            if detection_method == "cv" and cv_detector is not None:
                try:
                    result = cv_detector.detect(image, return_debug=True)
                    if isinstance(result, tuple) and len(result) == 2:
                        _, debug_info = result
                        if isinstance(debug_info, dict) and 'fit_points' in debug_info:
                            fit_points = debug_info['fit_points']
                            if isinstance(fit_points, dict):
                                fit_points_left = fit_points.get('left')
                                fit_points_right = fit_points.get('right')
                except Exception as e:
                    logger.debug(f"Failed to get fit_points: {e}")
                
                # Create visualization
                try:
                    vis_image = cv_detector.visualize_detection(
                        image, cv_lanes, debug_info=debug_info
                    )
                    
                    # Save to debug directory
                    debug_dir = Path('tmp/debug_visualizations')
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    save_path = debug_dir / f'frame_{frame_count:06d}.png'
                    
                    # DEBUG: Also save intermediate images to diagnose detection
                    logger.info(f"[DEBUG VIS] debug_info keys: {list(debug_info.keys()) if debug_info else 'empty'}")
                    if debug_info:
                        # Save masked edges (what HoughLinesP sees)
                        logger.info(f"[DEBUG VIS] Checking masked_edges: {'masked_edges' in debug_info}, "
                                   f"value is None: {debug_info.get('masked_edges') is None if 'masked_edges' in debug_info else 'key missing'}")
                        if 'masked_edges' in debug_info and debug_info['masked_edges'] is not None:
                            edges_path = debug_dir / f'frame_{frame_count:06d}_edges.png'
                            cv2.imwrite(str(edges_path), debug_info['masked_edges'])
                        
                        # Save yellow mask (yellow lane detection - primary for yellow lanes)
                        if 'yellow_mask' in debug_info and debug_info['yellow_mask'] is not None:
                            yellow_path = debug_dir / f'frame_{frame_count:06d}_yellow_mask.png'
                            cv2.imwrite(str(yellow_path), debug_info['yellow_mask'])
                        
                        # Save combined (edges + color)
                        if 'combined' in debug_info and debug_info['combined'] is not None:
                            combined_path = debug_dir / f'frame_{frame_count:06d}_combined.png'
                            cv2.imwrite(str(combined_path), debug_info['combined'])
                        
                        # Log detection stats
                        num_lines = debug_info.get('num_lines_detected', 0)
                        left_count = debug_info.get('left_lines_count', 0)
                        right_count = debug_info.get('right_lines_count', 0)
                        skipped_short = debug_info.get('skipped_short', 0)
                        skipped_center = debug_info.get('skipped_center', 0)
                        validation_failures = debug_info.get('validation_failures', {})
                        
                        logger.info(f"[DEBUG VIS] Frame {frame_count}: {num_lines} lines detected by HoughLinesP")
                        logger.info(f"[DEBUG VIS] Frame {frame_count}: Left={left_count}, Right={right_count}, "
                                   f"Skipped: short={skipped_short}, center={skipped_center}")
                        if validation_failures:
                            logger.info(f"[DEBUG VIS] Frame {frame_count}: Validation failures: {validation_failures}")
                    
                    cv2.imwrite(str(save_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                    
                    # Also create histogram of line x-positions to check for bimodal peaks
                    if debug_info and 'all_lines' in debug_info and debug_info['all_lines'] is not None:
                        line_centers = []
                        for line in debug_info['all_lines']:
                            x1, y1, x2, y2 = line[0]
                            line_center_x = (x1 + x2) / 2
                            line_centers.append(line_center_x)
                        
                        if line_centers:
                            import matplotlib
                            matplotlib.use('Agg')  # Non-interactive backend
                            import matplotlib.pyplot as plt
                            
                            plt.figure(figsize=(10, 6))
                            plt.hist(line_centers, bins=50, edgecolor='black')
                            plt.axvline(image.shape[1] // 2, color='r', linestyle='--', label='Image Center')
                            plt.xlabel('Line Center X Position (pixels)')
                            plt.ylabel('Frequency')
                            plt.title(f'Detected Line Positions (Frame {frame_count})')
                            plt.legend()
                            plt.grid(True, alpha=0.3)
                            
                            hist_path = debug_dir / f'line_histogram_{frame_count:06d}.png'
                            plt.savefig(hist_path, dpi=100, bbox_inches='tight')
                            plt.close()
                            
                            logger.info(f"Saved debug visualization: {save_path}, histogram: {hist_path}")
                except Exception as e:
                    logger.warning(f"Failed to create debug visualization: {e}")

        segmentation_mask_png = None
        if detection_method == "segmentation":
            seg_debug = getattr(self.perception, "last_segmentation_debug", None)
            if isinstance(seg_debug, dict):
                if fit_points_left is None:
                    fit_points_left = seg_debug.get("fit_points_left")
                if fit_points_right is None:
                    fit_points_right = seg_debug.get("fit_points_right")
                if self.record_segmentation_mask:
                    mask = seg_debug.get("mask")
                    if mask is not None:
                        try:
                            mask_uint8 = np.asarray(mask, dtype=np.uint8)
                            success, encoded = cv2.imencode(".png", mask_uint8)
                            if success:
                                segmentation_mask_png = encoded.tobytes()
                                if not getattr(self, "_segmentation_mask_logged", False):
                                    logger.info(
                                        "[SEGMENTATION MASK] Encoded mask bytes=%d shape=%s dtype=%s",
                                        len(segmentation_mask_png),
                                        getattr(mask_uint8, "shape", None),
                                        getattr(mask_uint8, "dtype", None),
                                    )
                                    self._segmentation_mask_logged = True
                        except Exception as e:
                            logger.debug(f"Failed to encode segmentation mask: {e}")
                    if not getattr(self, "_segmentation_mask_logged", False):
                        logger.info(
                            "[SEGMENTATION MASK] record=%s has_mask=%s shape=%s dtype=%s",
                            self.record_segmentation_mask,
                            mask is not None,
                            getattr(mask, "shape", None),
                            getattr(mask, "dtype", None),
                        )
                        self._segmentation_mask_logged = True
        
        # Calculate lane line positions at lookahead distance (for trajectory centering verification)
        left_lane_line_x = None
        right_lane_line_x = None
        
        # Initialize diagnostic tracking variables (available for all code paths)
        using_stale_data = False
        stale_data_reason = None
        left_jump_magnitude = None
        right_jump_magnitude = None
        jump_threshold_used = None
        # NEW: Diagnostic fields for perception instability (initialized to None, set only when instability detected)
        actual_detected_left_lane_x = None
        actual_detected_right_lane_x = None
        reject_reason = None
        clamp_events = []
        instability_width_change = None
        instability_center_shift = None
        
        current_speed = vehicle_state_dict.get('speed', 0.0)
        reference_lookahead = self.trajectory_config.get('reference_lookahead', 8.0)
        reference_lookahead = compute_reference_lookahead(
            base_lookahead=float(reference_lookahead),
            current_speed=float(current_speed),
            path_curvature=float(
                vehicle_state_dict.get('groundTruthPathCurvature', 0.0)
                or vehicle_state_dict.get('ground_truth_path_curvature', 0.0)
            ),
            config=self.trajectory_config,
        )
        
        # Extract camera_8m_screen_y early so we can use it for lane evaluation
        camera_8m_screen_y = vehicle_state_dict.get('camera8mScreenY')
        if camera_8m_screen_y is None:
            camera_8m_screen_y = vehicle_state_dict.get('camera_8m_screen_y', -1.0)
        camera_lookahead_screen_y = vehicle_state_dict.get('cameraLookaheadScreenY')
        if camera_lookahead_screen_y is None:
            camera_lookahead_screen_y = vehicle_state_dict.get('camera_lookahead_screen_y', -1.0)
        ground_truth_lookahead_distance = vehicle_state_dict.get('groundTruthLookaheadDistance')
        if ground_truth_lookahead_distance is None:
            ground_truth_lookahead_distance = vehicle_state_dict.get('ground_truth_lookahead_distance', 8.0)

        speed_limit = vehicle_state_dict.get('speedLimit')
        if speed_limit is None:
            speed_limit = vehicle_state_dict.get('speed_limit', 0.0)
        speed_limit_preview = vehicle_state_dict.get('speedLimitPreview')
        if speed_limit_preview is None:
            speed_limit_preview = vehicle_state_dict.get('speed_limit_preview', 0.0)
        speed_limit_preview_distance = vehicle_state_dict.get('speedLimitPreviewDistance')
        if speed_limit_preview_distance is None:
            speed_limit_preview_distance = vehicle_state_dict.get(
                'speed_limit_preview_distance',
                self.trajectory_config.get('speed_limit_preview_distance', 0.0),
            )
        speed_limit_preview_min_distance = vehicle_state_dict.get(
            'speedLimitPreviewMinDistance',
            vehicle_state_dict.get('speed_limit_preview_min_distance', 0.0),
        )
        speed_limit_preview_mid = vehicle_state_dict.get(
            'speedLimitPreviewMid',
            vehicle_state_dict.get('speed_limit_preview_mid', 0.0),
        )
        speed_limit_preview_mid_distance = vehicle_state_dict.get(
            'speedLimitPreviewMidDistance',
            vehicle_state_dict.get('speed_limit_preview_mid_distance', 0.0),
        )
        speed_limit_preview_mid_min_distance = vehicle_state_dict.get(
            'speedLimitPreviewMidMinDistance',
            vehicle_state_dict.get('speed_limit_preview_mid_min_distance', 0.0),
        )
        speed_limit_preview_long = vehicle_state_dict.get(
            'speedLimitPreviewLong',
            vehicle_state_dict.get('speed_limit_preview_long', 0.0),
        )
        speed_limit_preview_long_distance = vehicle_state_dict.get(
            'speedLimitPreviewLongDistance',
            vehicle_state_dict.get('speed_limit_preview_long_distance', 0.0),
        )
        speed_limit_preview_long_min_distance = vehicle_state_dict.get(
            'speedLimitPreviewLongMinDistance',
            vehicle_state_dict.get('speed_limit_preview_long_min_distance', 0.0),
        )
        camera_lookahead_screen_y = vehicle_state_dict.get('cameraLookaheadScreenY')
        if camera_lookahead_screen_y is None:
            camera_lookahead_screen_y = vehicle_state_dict.get('camera_lookahead_screen_y', -1.0)
        ground_truth_lookahead_distance = vehicle_state_dict.get('groundTruthLookaheadDistance')
        if ground_truth_lookahead_distance is None:
            ground_truth_lookahead_distance = vehicle_state_dict.get('ground_truth_lookahead_distance', 8.0)
        camera_lookahead_screen_y = vehicle_state_dict.get('cameraLookaheadScreenY')
        if camera_lookahead_screen_y is None:
            camera_lookahead_screen_y = vehicle_state_dict.get('camera_lookahead_screen_y', -1.0)
        ground_truth_lookahead_distance = vehicle_state_dict.get('groundTruthLookaheadDistance')
        if ground_truth_lookahead_distance is None:
            ground_truth_lookahead_distance = vehicle_state_dict.get('ground_truth_lookahead_distance', 8.0)
        camera_lookahead_screen_y = vehicle_state_dict.get('cameraLookaheadScreenY')
        if camera_lookahead_screen_y is None:
            camera_lookahead_screen_y = vehicle_state_dict.get('camera_lookahead_screen_y', -1.0)
        camera_lookahead_screen_y = vehicle_state_dict.get('cameraLookaheadScreenY')
        if camera_lookahead_screen_y is None:
            camera_lookahead_screen_y = vehicle_state_dict.get('camera_lookahead_screen_y', -1.0)
        ground_truth_lookahead_distance = vehicle_state_dict.get('groundTruthLookaheadDistance')
        if ground_truth_lookahead_distance is None:
            ground_truth_lookahead_distance = vehicle_state_dict.get('ground_truth_lookahead_distance', 8.0)
        camera_lookahead_screen_y = vehicle_state_dict.get('cameraLookaheadScreenY')
        if camera_lookahead_screen_y is None:
            camera_lookahead_screen_y = vehicle_state_dict.get('camera_lookahead_screen_y', -1.0)
        lookahead_distance_from_unity = vehicle_state_dict.get('groundTruthLookaheadDistance')
        if lookahead_distance_from_unity is None:
            lookahead_distance_from_unity = vehicle_state_dict.get('ground_truth_lookahead_distance')
        gt_lookahead_distance = None
        if isinstance(lookahead_distance_from_unity, (int, float)) and lookahead_distance_from_unity > 0:
            gt_lookahead_distance = float(lookahead_distance_from_unity)

        dynamic_horizon_diag = compute_dynamic_effective_horizon(
            base_horizon_m=float(reference_lookahead),
            current_speed_mps=float(current_speed),
            path_curvature=float(
                vehicle_state_dict.get('groundTruthPathCurvature', 0.0)
                or vehicle_state_dict.get('ground_truth_path_curvature', 0.0)
            ),
            confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
            config=self.trajectory_config,
        )
        
        # DEBUG: Log what we extracted (first few frames only)
        if self.frame_count < 3:
            logger.info(f"[COORD DEBUG] Frame {self.frame_count}: Extracted camera_8m_screen_y={camera_8m_screen_y} "
                       f"(from dict keys: {list(vehicle_state_dict.keys())[:10]}...)")
        
        # DEBUG: Log lane_coeffs format for debugging
        if self.frame_count == 0:
            logger.debug(f"[Frame {self.frame_count}] [DEBUG] lane_coeffs type={type(lane_coeffs)}, len={len(lane_coeffs) if hasattr(lane_coeffs, '__len__') else 'N/A'}")
            if hasattr(lane_coeffs, '__len__') and len(lane_coeffs) > 0:
                logger.debug(f"[Frame {self.frame_count}] [DEBUG] lane_coeffs[0] type={type(lane_coeffs[0])}, value={lane_coeffs[0] if lane_coeffs[0] is not None else 'None'}")
            if hasattr(lane_coeffs, '__len__') and len(lane_coeffs) > 1:
                logger.debug(f"[Frame {self.frame_count}] [DEBUG] lane_coeffs[1] type={type(lane_coeffs[1])}, value={lane_coeffs[1] if lane_coeffs[1] is not None else 'None'}")
            logger.debug(f"[Frame {self.frame_count}] [DEBUG] Condition check: len >= 2: {len(lane_coeffs) >= 2 if hasattr(lane_coeffs, '__len__') else False}, [0] not None: {lane_coeffs[0] is not None if hasattr(lane_coeffs, '__len__') and len(lane_coeffs) > 0 else False}, [1] not None: {lane_coeffs[1] is not None if hasattr(lane_coeffs, '__len__') and len(lane_coeffs) > 1 else False}")
        
        if len(lane_coeffs) >= 2 and lane_coeffs[0] is not None and lane_coeffs[1] is not None:
            try:
                # Get trajectory planner's conversion parameters
                planner = self.trajectory_planner.planner if hasattr(self.trajectory_planner, 'planner') else self.trajectory_planner
                image_height = planner.image_height if hasattr(planner, 'image_height') else 480.0
                image_width = planner.image_width if hasattr(planner, 'image_width') else 640.0
                pixel_to_meter_y = planner.pixel_to_meter_y if hasattr(planner, 'pixel_to_meter_y') else 0.02
                
                # Ensure pixel_to_meter_y is a numeric value (not a Mock in tests)
                if not isinstance(pixel_to_meter_y, (int, float)):
                    pixel_to_meter_y = 0.02
                
                # CRITICAL FIX: Use actual 8m position from Unity FIRST (most accurate)
                # Unity's WorldToScreenPoint tells us exactly where 8m appears in the image
                # This is more accurate than estimating from camera model
                # Use the already-extracted camera_8m_screen_y (extracted earlier at line 432)
                # DEBUG: Log what we're checking
                logger.debug(f"[COORD DEBUG] Checking camera_8m_screen_y: value={camera_8m_screen_y}, type={type(camera_8m_screen_y)}, >0={camera_8m_screen_y > 0 if camera_8m_screen_y is not None else 'None'}")
                coord_conversion_distance = gt_lookahead_distance if gt_lookahead_distance is not None else reference_lookahead
                if camera_lookahead_screen_y is not None and camera_lookahead_screen_y > 0:
                    y_image_at_lookahead = camera_lookahead_screen_y
                    logger.info(
                        f"[COORD DEBUG] ✅ Using Unity's camera_lookahead_screen_y={camera_lookahead_screen_y:.1f}px "
                        f"(lookahead={coord_conversion_distance:.2f}m) for lane evaluation"
                    )
                elif camera_8m_screen_y is not None and camera_8m_screen_y > 0 and abs(coord_conversion_distance - 8.0) < 0.5:
                    # Use Unity's actual 8m position (most accurate for 8m lookahead)
                    y_image_at_lookahead = camera_8m_screen_y
                    logger.info(f"[COORD DEBUG] ✅ Using Unity's camera_8m_screen_y={camera_8m_screen_y:.1f}px for lane evaluation")
                else:
                    # Fallback: Estimate y position from reference_lookahead distance
                    # Use inverse of the coordinate conversion fallback logic
                    # At bottom (y=height): distance ≈ 1.5m
                    # At top (y=0): distance ≈ far
                    # For 8m: y_normalized = 1.5 / 8.0 = 0.1875
                    # y_from_bottom = 0.1875 * image_height = 90 pixels
                    # y_pixels = image_height - 90 = 390 pixels
                    base_distance = 1.5  # meters at bottom
                    if coord_conversion_distance < base_distance:
                        y_image_at_lookahead = image_height
                    else:
                        y_normalized = base_distance / coord_conversion_distance
                        y_from_bottom = y_normalized * image_height
                        y_image_at_lookahead = image_height - y_from_bottom
                    logger.warning(f"[COORD DEBUG] camera_8m_screen_y not available ({camera_8m_screen_y}), using fallback y={y_image_at_lookahead:.1f}px")
                
                y_image_at_lookahead = max(0, min(image_height - 1, int(y_image_at_lookahead)))
                
                # Get lane x positions in image coordinates
                # CRITICAL FIX: Lane detection may return lanes in wrong order
                # Always check and swap if needed to ensure left < right
                lane0_coeffs = lane_coeffs[0]
                lane1_coeffs = lane_coeffs[1]
                
                # CRITICAL: Use the ACTUAL distance from Unity, not a fixed value
                # Unity's camera_8m_screen_y tells us where 8m appears in the image
                # If we're evaluating at that y position, the distance IS 8m
                # Don't use a fixed 7m - use the actual distance Unity reports
                if camera_lookahead_screen_y > 0:
                    conversion_distance = coord_conversion_distance
                    logger.info(f"[COORD DEBUG] Using Unity lookahead distance: {coord_conversion_distance:.2f}m")
                elif camera_8m_screen_y > 0 and abs(coord_conversion_distance - 8.0) < 0.5 and abs(y_image_at_lookahead - camera_8m_screen_y) < 5.0:
                    conversion_distance = 8.0
                    logger.info(f"[COORD DEBUG] Using Unity's actual distance: 8.0m (at y={camera_8m_screen_y:.1f}px)")
                else:
                    conversion_distance = coord_conversion_distance
                    logger.warning(f"[COORD DEBUG] Using coord_conversion_distance={coord_conversion_distance}m (Unity distance not available)")
                
                logger.info(f"[COORD DEBUG] Evaluating at y={y_image_at_lookahead}px, "
                           f"using fixed distance={conversion_distance:.2f}m for coordinate conversion")
                
                # Determine which is left and which is right (smaller x = left in image coords)
                # Check at bottom of image (closest to vehicle) for most reliable ordering
                x0_at_bottom = np.polyval(lane0_coeffs, image_height)
                x1_at_bottom = np.polyval(lane1_coeffs, image_height)
                
                # Also check at lookahead distance for consistency
                x0_at_lookahead = np.polyval(lane0_coeffs, y_image_at_lookahead)
                x1_at_lookahead = np.polyval(lane1_coeffs, y_image_at_lookahead)
                
                # Use both checks to be more robust
                # If both agree, use that ordering. If they disagree, use bottom (closer to vehicle)
                if x0_at_bottom < x1_at_bottom and x0_at_lookahead < x1_at_lookahead:
                    # Consistent: lane0 is left, lane1 is right
                    actual_left_coeffs = lane0_coeffs
                    actual_right_coeffs = lane1_coeffs
                elif x0_at_bottom > x1_at_bottom and x0_at_lookahead > x1_at_lookahead:
                    # Consistent: lane0 is right, lane1 is left (swap needed)
                    actual_left_coeffs = lane1_coeffs
                    actual_right_coeffs = lane0_coeffs
                else:
                    # Inconsistent: use bottom position (closer to vehicle, more reliable)
                    if x0_at_bottom < x1_at_bottom:
                        actual_left_coeffs = lane0_coeffs
                        actual_right_coeffs = lane1_coeffs
                    else:
                        actual_left_coeffs = lane1_coeffs
                        actual_right_coeffs = lane0_coeffs
                
                # Get x positions at lookahead in image coordinates
                left_x_image = np.polyval(actual_left_coeffs, y_image_at_lookahead)
                right_x_image = np.polyval(actual_right_coeffs, y_image_at_lookahead)
                
                # CRITICAL FIX: Verify ordering is correct
                # Left lane should have smaller x (left side of image), right should have larger x
                if left_x_image > right_x_image:
                    logger.warning(f"[COORD DEBUG] Lane ordering reversed at y={y_image_at_lookahead}px! "
                                 f"left_x={left_x_image:.1f}px > right_x={right_x_image:.1f}px. Swapping.")
                    # Swap the positions
                    left_x_image, right_x_image = right_x_image, left_x_image
                    # Also swap the coefficients for consistency
                    actual_left_coeffs, actual_right_coeffs = actual_right_coeffs, actual_left_coeffs
                
                # FIXED: Don't clamp to image bounds - allow lanes outside image to be converted
                # This allows us to see true detected positions even if they're beyond image edges
                # The coordinate conversion will handle out-of-bounds values correctly
                # Only log a warning if lanes are way outside reasonable bounds
                if left_x_image < -100 or left_x_image > image_width + 100:
                    logger.warning(f"[COORD DEBUG] Left lane x={left_x_image:.1f}px is way outside image bounds (0-{image_width})")
                if right_x_image < -100 or right_x_image > image_width + 100:
                    logger.warning(f"[COORD DEBUG] Right lane x={right_x_image:.1f}px is way outside image bounds (0-{image_width})")
                
                # Final check: Ensure left < right after clamping
                if left_x_image >= right_x_image:
                    logger.warning(f"[COORD DEBUG] After clamping, lanes still reversed! "
                                 f"left_x={left_x_image:.1f}px, right_x={right_x_image:.1f}px. "
                                 f"Using center of image as fallback.")
                    # Fallback: Use center of image ± half lane width
                    image_center = image_width / 2.0
                    lane_half_width_px = 100  # Approximate half lane width in pixels at lookahead
                    left_x_image = image_center - lane_half_width_px
                    right_x_image = image_center + lane_half_width_px
                
                # Convert from image coordinates to vehicle coordinates
                # CRITICAL FIX: Use calculated distance for conversion (corresponds to y position)
                # This ensures pixel-to-meter conversion matches the actual distance
                logger.info(f"[COORD DEBUG] Input: left_x_image={left_x_image:.1f}, right_x_image={right_x_image:.1f}, "
                           f"y_image_at_lookahead={y_image_at_lookahead:.1f}, conversion_distance={conversion_distance:.2f}m")
                # Use Unity's reported horizontal FOV if available (most accurate)
                camera_horizontal_fov = vehicle_state_dict.get('cameraHorizontalFOV')
                if camera_horizontal_fov is None:
                    camera_horizontal_fov = vehicle_state_dict.get('camera_horizontal_fov', None)
                
                # DEBUG: Log before coordinate conversion (frame 0 only)
                if self.frame_count == 0:
                    logger.debug(f"[Frame {self.frame_count}] [DEBUG] About to call _convert_image_to_vehicle_coords")
                    logger.debug(f"[Frame {self.frame_count}] [DEBUG] left_x_image={left_x_image:.1f}, right_x_image={right_x_image:.1f}, y={y_image_at_lookahead:.1f}, dist={conversion_distance:.2f}m")
                
                left_x_vehicle, _ = planner._convert_image_to_vehicle_coords(
                    left_x_image, y_image_at_lookahead, lookahead_distance=conversion_distance,
                    horizontal_fov_override=camera_horizontal_fov if camera_horizontal_fov and camera_horizontal_fov > 0 else None
                )
                right_x_vehicle, _ = planner._convert_image_to_vehicle_coords(
                    right_x_image, y_image_at_lookahead, lookahead_distance=conversion_distance,
                    horizontal_fov_override=camera_horizontal_fov if camera_horizontal_fov and camera_horizontal_fov > 0 else None
                )
                calculated_lane_width = right_x_vehicle - left_x_vehicle
                
                # DEBUG: Log after coordinate conversion (frame 0 only)
                if self.frame_count == 0:
                    logger.debug(f"[Frame {self.frame_count}] [DEBUG] Coordinate conversion SUCCESS!")
                    logger.debug(f"[Frame {self.frame_count}] [DEBUG] left_x_vehicle={left_x_vehicle:.3f}m, right_x_vehicle={right_x_vehicle:.3f}m, width={calculated_lane_width:.3f}m")
                
                logger.info(f"[COORD DEBUG] Result: left_x_vehicle={left_x_vehicle:.3f}m, "
                           f"right_x_vehicle={right_x_vehicle:.3f}m, lane_width={calculated_lane_width:.3f}m")
                
                # FINAL SAFETY CHECK: Ensure left < right after coordinate conversion
                # Coordinate conversion should preserve order, but check just in case
                if left_x_vehicle > right_x_vehicle:
                    logger.warning(f"[COORD DEBUG] Order reversed after conversion! Swapping: "
                                 f"left={left_x_vehicle:.3f}m, right={right_x_vehicle:.3f}m")
                    left_x_vehicle, right_x_vehicle = right_x_vehicle, left_x_vehicle
                    calculated_lane_width = right_x_vehicle - left_x_vehicle

                perception_cfg = self.config.get("perception", {}) if isinstance(self.config, dict) else {}
                low_visibility_fallback_enabled = bool(
                    perception_cfg.get("low_visibility_fallback_enabled", True)
                )
                low_visibility_fallback_blend_alpha = float(
                    perception_cfg.get("low_visibility_fallback_blend_alpha", 0.35)
                )
                low_visibility_fallback_max_consecutive_frames = int(
                    perception_cfg.get("low_visibility_fallback_max_consecutive_frames", 8)
                )
                low_visibility_fallback_center_shift_cap_m = float(
                    perception_cfg.get("low_visibility_fallback_center_shift_cap_m", 0.25)
                )
                low_visibility_fallback_used = False
                # Lookahead-local visibility gate: when dashed centerline gaps land near
                # lookahead row, one lane can be weak while global fit still exists.
                if (
                    low_visibility_fallback_enabled
                    and self.previous_left_lane_x is not None
                    and self.previous_right_lane_x is not None
                    and not using_stale_data
                ):
                    left_low_lookahead = is_lane_low_visibility_at_lookahead(
                        fit_points_left, image_width, "left", y_image_at_lookahead
                    )
                    right_low_lookahead = is_lane_low_visibility_at_lookahead(
                        fit_points_right, image_width, "right", y_image_at_lookahead
                    )
                    if right_low_lookahead and not left_low_lookahead:
                        clamp_events.append("right_lane_lookahead_low_visibility")
                        actual_detected_left_lane_x = left_x_vehicle
                        actual_detected_right_lane_x = right_x_vehicle
                        if self.low_visibility_fallback_streak < low_visibility_fallback_max_consecutive_frames:
                            last_width = float(self.previous_right_lane_x - self.previous_left_lane_x)
                            est_left, est_right = estimate_single_lane_pair(
                                single_x_vehicle=float(left_x_vehicle),
                                is_left_lane=True,
                                last_width=last_width,
                                default_width=7.0,
                                width_min=1.0,
                                width_max=10.0,
                            )
                            left_x_vehicle, right_x_vehicle = blend_lane_pair_with_previous(
                                est_left,
                                est_right,
                                float(self.previous_left_lane_x),
                                float(self.previous_right_lane_x),
                                low_visibility_fallback_blend_alpha,
                                low_visibility_fallback_center_shift_cap_m,
                            )
                            calculated_lane_width = right_x_vehicle - left_x_vehicle
                            using_stale_data = True
                            stale_data_reason = "right_lane_low_visibility"
                            reject_reason = reject_reason or stale_data_reason
                            low_visibility_fallback_used = True
                            logger.warning(
                                f"[Frame {self.frame_count}] [PERCEPTION VALIDATION] "
                                "Right lane low visibility near lookahead; using bounded blended fallback."
                            )
                        else:
                            clamp_events.append("low_visibility_fallback_ttl_exceeded")
                            logger.warning(
                                f"[Frame {self.frame_count}] [PERCEPTION VALIDATION] "
                                "Right lane low visibility fallback skipped (TTL exceeded)."
                            )
                    elif left_low_lookahead and not right_low_lookahead:
                        clamp_events.append("left_lane_lookahead_low_visibility")
                        actual_detected_left_lane_x = left_x_vehicle
                        actual_detected_right_lane_x = right_x_vehicle
                        if self.low_visibility_fallback_streak < low_visibility_fallback_max_consecutive_frames:
                            last_width = float(self.previous_right_lane_x - self.previous_left_lane_x)
                            est_left, est_right = estimate_single_lane_pair(
                                single_x_vehicle=float(right_x_vehicle),
                                is_left_lane=False,
                                last_width=last_width,
                                default_width=7.0,
                                width_min=1.0,
                                width_max=10.0,
                            )
                            left_x_vehicle, right_x_vehicle = blend_lane_pair_with_previous(
                                est_left,
                                est_right,
                                float(self.previous_left_lane_x),
                                float(self.previous_right_lane_x),
                                low_visibility_fallback_blend_alpha,
                                low_visibility_fallback_center_shift_cap_m,
                            )
                            calculated_lane_width = right_x_vehicle - left_x_vehicle
                            using_stale_data = True
                            stale_data_reason = "left_lane_low_visibility"
                            reject_reason = reject_reason or stale_data_reason
                            low_visibility_fallback_used = True
                            logger.warning(
                                f"[Frame {self.frame_count}] [PERCEPTION VALIDATION] "
                                "Left lane low visibility near lookahead; using bounded blended fallback."
                            )
                        else:
                            clamp_events.append("low_visibility_fallback_ttl_exceeded")
                            logger.warning(
                                f"[Frame {self.frame_count}] [PERCEPTION VALIDATION] "
                                "Left lane low visibility fallback skipped (TTL exceeded)."
                            )
                if low_visibility_fallback_used:
                    self.low_visibility_fallback_streak += 1
                else:
                    self.low_visibility_fallback_streak = 0
                
                # CRITICAL FIX: Validate lane side consistency (left should be negative, right positive)
                perception_cfg = self.config.get("perception", {}) if isinstance(self.config, dict) else {}
                enforce_lane_sign = bool(perception_cfg.get("lane_side_sign_enforce", False))
                sign_min_abs = float(perception_cfg.get("lane_side_sign_min_abs_m", 0.2))
                if enforce_lane_sign and \
                   self.previous_left_lane_x is not None and self.previous_right_lane_x is not None:
                    skip_sign_enforce = False
                    road_offset = vehicle_state_dict.get('roadFrameLaneCenterOffset')
                    if road_offset is not None:
                        lane_width_est = float(self.safety_config.get('lane_width', 7.2))
                        lane_edge_threshold = max(0.1, (lane_width_est / 2.0) - 0.1)
                        if abs(float(road_offset)) > lane_edge_threshold:
                            skip_sign_enforce = True
                            clamp_events.append("lane_side_sign_skip_out_of_bounds")
                    if not skip_sign_enforce and (left_x_vehicle > -sign_min_abs or right_x_vehicle < sign_min_abs):
                        clamp_events.append("lane_side_sign_violation")
                        actual_detected_left_lane_x = left_x_vehicle
                        actual_detected_right_lane_x = right_x_vehicle
                        logger.warning(
                            f"[Frame {self.frame_count}] [PERCEPTION VALIDATION] Lane side sign violation "
                            f"(left={left_x_vehicle:.3f}m right={right_x_vehicle:.3f}m). Using previous."
                        )
                        left_x_vehicle = float(self.previous_left_lane_x)
                        right_x_vehicle = float(self.previous_right_lane_x)
                        calculated_lane_width = right_x_vehicle - left_x_vehicle
                        using_stale_data = True
                        stale_data_reason = "lane_side_sign_violation"
                        reject_reason = reject_reason or stale_data_reason

                # CRITICAL FIX: Validate lane width in vehicle coordinates (meters)
                # This catches perception failures that weren't caught in pixel space
                # Example: Frame 100 had width=1.712m (should be ~7.0m) - this would catch it!
                perception_cfg = self.config.get("perception", {}) if isinstance(self.config, dict) else {}
                min_lane_width = float(perception_cfg.get("lane_width_min_m", 2.0))
                max_lane_width = float(perception_cfg.get("lane_width_max_m", 10.0))
                
                if calculated_lane_width < min_lane_width or calculated_lane_width > max_lane_width:
                    logger.error(
                        f"[Frame {self.frame_count}] [PERCEPTION VALIDATION] ❌ INVALID LANE WIDTH: "
                        f"{calculated_lane_width:.3f}m (expected {min_lane_width}-{max_lane_width}m)"
                    )
                    logger.error(
                        f"[Frame {self.frame_count}]   Left lane: {left_x_vehicle:.3f}m, "
                        f"Right lane: {right_x_vehicle:.3f}m"
                    )
                    
                    filled_from_visibility = False
                    if (
                        low_visibility_fallback_enabled
                        and self.previous_left_lane_x is not None
                        and self.previous_right_lane_x is not None
                    ):
                        left_low = is_lane_low_visibility(
                            fit_points_left, image_width, "left"
                        ) or is_lane_low_visibility_at_lookahead(
                            fit_points_left, image_width, "left", y_image_at_lookahead
                        )
                        right_low = is_lane_low_visibility(
                            fit_points_right, image_width, "right"
                        ) or is_lane_low_visibility_at_lookahead(
                            fit_points_right, image_width, "right", y_image_at_lookahead
                        )
                        last_width = float(self.previous_right_lane_x - self.previous_left_lane_x)

                        if right_low and not left_low:
                            if self.low_visibility_fallback_streak < low_visibility_fallback_max_consecutive_frames:
                                est_left, est_right = estimate_single_lane_pair(
                                    single_x_vehicle=float(left_x_vehicle),
                                    is_left_lane=True,
                                    last_width=last_width,
                                    default_width=7.0,
                                    width_min=min_lane_width,
                                    width_max=max_lane_width,
                                )
                                left_lane_line_x, right_lane_line_x = blend_lane_pair_with_previous(
                                    est_left,
                                    est_right,
                                    float(self.previous_left_lane_x),
                                    float(self.previous_right_lane_x),
                                    low_visibility_fallback_blend_alpha,
                                    low_visibility_fallback_center_shift_cap_m,
                                )
                                calculated_lane_width = right_lane_line_x - left_lane_line_x
                                logger.warning(
                                    f"[Frame {self.frame_count}] [PERCEPTION VALIDATION] "
                                    "Right lane low visibility; using bounded blended fallback to fill right."
                                )
                                using_stale_data = True
                                stale_data_reason = "right_lane_low_visibility"
                                reject_reason = reject_reason or stale_data_reason
                                low_visibility_fallback_used = True
                                filled_from_visibility = True
                            else:
                                clamp_events.append("low_visibility_fallback_ttl_exceeded")
                                logger.warning(
                                    f"[Frame {self.frame_count}] [PERCEPTION VALIDATION] "
                                    "Right lane low visibility fallback skipped (TTL exceeded)."
                                )
                        elif left_low and not right_low:
                            if self.low_visibility_fallback_streak < low_visibility_fallback_max_consecutive_frames:
                                est_left, est_right = estimate_single_lane_pair(
                                    single_x_vehicle=float(right_x_vehicle),
                                    is_left_lane=False,
                                    last_width=last_width,
                                    default_width=7.0,
                                    width_min=min_lane_width,
                                    width_max=max_lane_width,
                                )
                                left_lane_line_x, right_lane_line_x = blend_lane_pair_with_previous(
                                    est_left,
                                    est_right,
                                    float(self.previous_left_lane_x),
                                    float(self.previous_right_lane_x),
                                    low_visibility_fallback_blend_alpha,
                                    low_visibility_fallback_center_shift_cap_m,
                                )
                                calculated_lane_width = right_lane_line_x - left_lane_line_x
                                logger.warning(
                                    f"[Frame {self.frame_count}] [PERCEPTION VALIDATION] "
                                    "Left lane low visibility; using bounded blended fallback to fill left."
                                )
                                using_stale_data = True
                                stale_data_reason = "left_lane_low_visibility"
                                reject_reason = reject_reason or stale_data_reason
                                low_visibility_fallback_used = True
                                filled_from_visibility = True
                            else:
                                clamp_events.append("low_visibility_fallback_ttl_exceeded")
                                logger.warning(
                                    f"[Frame {self.frame_count}] [PERCEPTION VALIDATION] "
                                    "Left lane low visibility fallback skipped (TTL exceeded)."
                                )

                    is_first_frame = (self.frame_count == 0)
                    if is_first_frame and not filled_from_visibility:
                        catastrophic_min = 0.1
                        catastrophic_max = 20.0
                        if calculated_lane_width < catastrophic_min or calculated_lane_width > catastrophic_max:
                            logger.error(
                                f"[Frame {self.frame_count}]   FIRST FRAME: Catastrophically wrong width "
                                f"({calculated_lane_width:.3f}m) - rejecting!"
                            )
                            left_lane_line_x = None
                            right_lane_line_x = None
                            clamp_events.append("invalid_width_reject")
                            clamp_events.append("invalid_width_reject")
                        else:
                            if calculated_lane_width < 2.0 or calculated_lane_width > 10.0:
                                logger.warning(
                                    f"[Frame {self.frame_count}] ⚠️  FIRST FRAME: Invalid width "
                                    f"({calculated_lane_width:.3f}m) but not catastrophic. Using detection anyway."
                                )
                            else:
                                logger.info(
                                    f"[Frame {self.frame_count}] ✓ FIRST FRAME: Valid width "
                                    f"({calculated_lane_width:.3f}m)"
                                )
                            left_lane_line_x = float(left_x_vehicle)
                            right_lane_line_x = float(right_x_vehicle)
                            if not using_stale_data:
                                using_stale_data = False
                                stale_data_reason = None
                    elif not filled_from_visibility:
                        logger.error(
                            f"[Frame {self.frame_count}]   This is a perception failure - rejecting detection!"
                        )
                        left_lane_line_x = None
                        right_lane_line_x = None
                        if self.previous_left_lane_x is not None and self.previous_right_lane_x is not None:
                            using_stale_data = True
                            stale_data_reason = "invalid_width"
                            reject_reason = reject_reason or stale_data_reason
                            left_lane_line_x = float(self.previous_left_lane_x)
                            right_lane_line_x = float(self.previous_right_lane_x)
                            logger.warning(
                                f"[Frame {self.frame_count}] Using STALE perception data "
                                f"(reason: {stale_data_reason}) - invalid lane width detected"
                            )
                else:
                    # NEW: Validate polynomial coefficients to catch extreme curves
                    # Even if width at lookahead is valid, polynomial can produce extreme values elsewhere
                    # This catches cases like frame 398 where width=7.4m is valid but polynomial goes way outside image
                    image_height = planner.image_height if hasattr(planner, 'image_height') else 480.0
                    image_width = planner.image_width if hasattr(planner, 'image_width') else 640.0
                    
                    # Evaluate polynomial at the lookahead distance where we actually use it
                    # CRITICAL: Don't validate at top of image (y=0) - extrapolation there can be extreme on curves
                    # We only use the polynomial at the lookahead distance (y_image_at_lookahead), so validate there
                    # Also check bottom of image (where lanes are closest to vehicle) for consistency
                    is_first_frame = (self.frame_count == 0)
                    
                    # CRITICAL FIX: Only validate where we actually use the polynomial
                    # The lookahead distance is where we evaluate for coordinate conversion
                    # Top of image (y=0) is extrapolation and can be extreme on curves - that's OK
                    # Bottom of image (y=image_height-1) may have missing dashed lines, causing incorrect extrapolation
                    # We only use the polynomial at the lookahead distance, so only validate there
                    y_check_positions = [y_image_at_lookahead]  # Only check where we actually use it
                    
                    # Use reasonable thresholds - allow some extrapolation for curves
                    max_reasonable_x = image_width * 2.5  # Allow up to 2.5x image width (for curves)
                    min_reasonable_x = -image_width * 1.5  # Allow some negative (for curves)
                    
                    # On first frame, be even more lenient since we have no previous data
                    if is_first_frame:
                        max_reasonable_x = image_width * 3.0  # More lenient for first frame
                        min_reasonable_x = -image_width * 2.0
                    
                    extreme_coeffs_detected = False
                    for lane_idx, lane_coeffs_item in enumerate(lane_coeffs):
                        if lane_coeffs_item is not None:
                            # Convert to numpy array and ensure it's 1D
                            lane_coeffs_array = np.asarray(lane_coeffs_item)
                            if lane_coeffs_array.ndim == 1 and len(lane_coeffs_array) >= 3:
                                # Evaluate polynomial: x = a*y^2 + b*y + c
                                for y_check in y_check_positions:
                                    x_eval = lane_coeffs_array[0] * y_check * y_check + lane_coeffs_array[1] * y_check + lane_coeffs_array[2]
                                    if x_eval < min_reasonable_x or x_eval > max_reasonable_x:
                                        extreme_coeffs_detected = True
                                        logger.error(f"[Frame {self.frame_count}] [PERCEPTION VALIDATION] ❌ EXTREME POLYNOMIAL COEFFICIENTS! "
                                                   f"Lane {lane_idx} at y={y_check}px: x={x_eval:.1f}px (expected {min_reasonable_x:.0f} to {max_reasonable_x:.0f}px)")
                                        logger.error(f"[Frame {self.frame_count}]   Coefficients: a={lane_coeffs_array[0]:.6f}, b={lane_coeffs_array[1]:.6f}, c={lane_coeffs_array[2]:.6f}")
                                        if is_first_frame:
                                            logger.warning(f"[Frame {self.frame_count}]   NOTE: First frame - validation relaxed, but this is still extreme!")
                                        break
                                if extreme_coeffs_detected:
                                    break
                    
                    if extreme_coeffs_detected:
                        # Reject this detection - polynomial produces extreme values
                        logger.error(f"[Frame {self.frame_count}]   This is a perception failure - rejecting detection!")
                        
                        # CRITICAL: On first frame, we have no previous data to fall back to
                        # If validation fails on first frame, we need to either:
                        # 1. Accept the detection anyway (risky but better than no data)
                        # 2. Use a default/straight trajectory
                        # For now, we'll log a warning but still try to use the detection if width is reasonable
                        if is_first_frame:
                            logger.warning(f"[Frame {self.frame_count}] ⚠️  FIRST FRAME: Extreme coefficients detected but no previous data available!")
                            logger.warning(f"[Frame {self.frame_count}]   Will attempt to use detection anyway - width check will catch truly bad detections")
                            # Don't set to None - let it continue to see if width check passes
                            # The width check below will catch truly bad detections
                            extreme_coeffs_detected = False  # Allow it through, but width check will catch it
                        else:
                            # Subsequent frames: Reject and use stale data
                            left_lane_line_x = None
                            right_lane_line_x = None
                            clamp_events.append("extreme_coeffs_reject")
                            # Track that we're using stale data due to extreme coefficients
                            if self.previous_left_lane_x is not None and self.previous_right_lane_x is not None:
                                using_stale_data = True
                                stale_data_reason = "extreme_coefficients"
                                left_lane_line_x = float(self.previous_left_lane_x)
                                right_lane_line_x = float(self.previous_right_lane_x)
                                logger.warning(f"[Frame {self.frame_count}] Using STALE perception data (reason: {stale_data_reason}) - extreme polynomial coefficients detected")
                    else:
                        # CRITICAL FIX: Check for sudden jumps in lane positions (temporal validation)
                        # If lane position jumps >2m from previous frame, it's likely a false detection
                        if not hasattr(self, 'previous_left_lane_x'):
                            self.previous_left_lane_x = None
                            self.previous_right_lane_x = None
                        
                        # CRITICAL FIX: On first frame (no previous data), accept current detection immediately
                        # Don't skip the entire block - we need to assign values!
                        if self.previous_left_lane_x is None or self.previous_right_lane_x is None:
                            # First frame or no previous data - accept current detection
                            if self.frame_count == 0:
                                logger.debug(f"[Frame {self.frame_count}] [DEBUG] First frame - assigning values: left={left_x_vehicle:.3f}m, right={right_x_vehicle:.3f}m")
                            left_lane_line_x = float(left_x_vehicle)
                            right_lane_line_x = float(right_x_vehicle)
                            if self.frame_count == 0:
                                logger.debug(f"[Frame {self.frame_count}] [DEBUG] After assignment: left_lane_line_x={left_lane_line_x}, right_lane_line_x={right_lane_line_x}")
                            
                            # Update previous values for next frame
                            self.previous_left_lane_x = left_x_vehicle
                            self.previous_right_lane_x = right_x_vehicle
                        elif self.previous_left_lane_x is not None and self.previous_right_lane_x is not None:
                            left_jump = abs(left_x_vehicle - self.previous_left_lane_x)
                            right_jump = abs(right_x_vehicle - self.previous_right_lane_x)
                            # RECOMMENDED: Increased from 2.0m to 3.5m to handle curves better
                            # On curves, lane positions can change by 2-3m between frames when car is turning
                            # 2.0m was too strict and caused false rejections on curves (e.g., frame 201: 3.094m jump)
                            # NEW: Adaptive threshold based on path curvature for even better curve handling
                            path_curvature = vehicle_state_dict.get('groundTruthPathCurvature') or vehicle_state_dict.get('ground_truth_path_curvature', 0.0)
                            abs_curvature = abs(path_curvature)
                            if abs_curvature > 0.1:  # On a curve
                                max_jump_threshold = 4.5  # meters - more lenient for curves (increased from 3.5m)
                            else:  # On straight road
                                max_jump_threshold = 3.5  # meters - standard threshold for straights
                            
                            # CRITICAL FIX: Check for timestamp gap (Unity pause) OR frozen timestamp before applying jump detection
                            # After a Unity pause, large jumps are expected and should be accepted
                            # Also detect frozen timestamps (dt = 0.0) which indicates Unity stopped updating
                            has_time_gap = False
                            has_frozen_timestamp = False
                            if prev_timestamp is None:
                                dt = 0.0  # First frame - no previous timestamp
                            else:
                                dt = timestamp - prev_timestamp
                                if dt > 0.5:  # Large time gap = Unity paused
                                    has_time_gap = True
                                    logger.info(f"[PERCEPTION VALIDATION] Unity pause detected (time gap: {dt:.3f}s) - relaxing jump detection")
                                    # After Unity pause, accept larger jumps (6.0m instead of 3.5m)
                                    max_jump_threshold = 6.0
                                elif abs(dt) < 0.001:  # Timestamp frozen (dt ≈ 0) = Unity stopped updating
                                    has_frozen_timestamp = True
                                    logger.warning(f"[PERCEPTION VALIDATION] ⚠️  FROZEN TIMESTAMP detected (dt={dt:.6f}s) - Unity stopped updating timestamps!")
                                    logger.warning("  This means Unity paused/froze but Python kept processing the same frame")
                                    logger.warning("  Perception will return identical results until Unity resumes")
                                    # Treat frozen timestamp like a time gap - accept any changes when Unity resumes
                                    # But for now, we're still processing the same frame, so values will be identical
                                    # Don't update last_timestamp - keep it as the last valid timestamp
                                    # This will help detect when Unity resumes (timestamp will jump)
                            
                            # CRITICAL: Detect if perception values are frozen (same as previous frame)
                            # This could indicate:
                            # 1. Unity paused (no new frames) - check timestamp gap
                            # 2. Perception died (no new detections, but Unity still sending frames)
                            # 3. Bridge disconnected (no frames at all)
                            # CRITICAL FIX: Only mark as frozen if timestamp is actually frozen OR values unchanged for 3+ frames
                            # This prevents false positives from legitimate repeated detections (1-2 frames)
                            values_match = (abs(left_x_vehicle - self.previous_left_lane_x) < 0.001 and 
                                          abs(right_x_vehicle - self.previous_right_lane_x) < 0.001)
                            
                            # Check timestamp to distinguish actual freeze from legitimate repeated detections
                            if prev_timestamp is None:
                                dt = 0.0  # First frame - no previous timestamp
                            else:
                                dt = timestamp - prev_timestamp
                            timestamp_frozen = abs(dt) < 0.001  # Timestamp not updating = Unity frozen
                            
                            # Track consecutive frames with matching values
                            if not hasattr(self, 'perception_frozen_frames'):
                                self.perception_frozen_frames = 0
                            
                            if values_match:
                                self.perception_frozen_frames += 1
                            else:
                                self.perception_frozen_frames = 0  # Reset counter if values change
                            
                            # Only mark as frozen if timestamp is frozen OR values unchanged for 3+ frames
                            # This prevents false positives from legitimate repeated detections (1-2 frames)
                            perception_frozen = timestamp_frozen or (values_match and self.perception_frozen_frames >= 3)
                            
                            if perception_frozen:
                                # Perception values are frozen - diagnose the cause
                                # Note: perception_frozen_frames counter already updated above
                                if not hasattr(self, 'last_timestamp_before_freeze'):
                                    self.last_timestamp_before_freeze = None
                                
                                # Track that we're using stale data due to frozen perception
                                if not using_stale_data:  # Only set if not already set by jump detection
                                    using_stale_data = True
                                    stale_data_reason = "frozen"
                                    if timestamp_frozen:
                                        logger.warning(f"[Frame {self.frame_count}] Using STALE perception data (reason: {stale_data_reason}) - timestamp frozen (dt={dt:.6f}s)")
                                    else:
                                        logger.warning(f"[Frame {self.frame_count}] Using STALE perception data (reason: {stale_data_reason}) - perception values unchanged for {self.perception_frozen_frames} frames")
                                
                                # Check timestamp to distinguish Unity pause vs perception failure
                                if not hasattr(self, 'last_timestamp_before_freeze'):
                                    self.last_timestamp_before_freeze = None
                                
                                if self.perception_frozen_frames == 1 or self.last_timestamp_before_freeze is None:
                                    # First frozen frame - record previous timestamp
                                    self.last_timestamp_before_freeze = prev_timestamp if prev_timestamp is not None else timestamp
                                
                                # Diagnose cause
                                if timestamp_frozen:  # Frozen timestamp (dt ≈ 0) = Unity stopped updating timestamps
                                    if self.perception_frozen_frames == 1:
                                        logger.warning(f"[PERCEPTION VALIDATION] ⚠️  FROZEN TIMESTAMP detected (dt={dt:.6f}s) - Unity stopped updating!")
                                        logger.warning("  Unity paused/froze but Python kept processing the same frame")
                                        logger.warning("  Perception will return identical results until Unity resumes")
                                    # Don't update last_timestamp - keep it as the last valid timestamp
                                    # This helps detect when Unity resumes (timestamp will jump)
                                elif dt > 0.5:  # Large time gap = Unity paused
                                    if self.perception_frozen_frames == 1:
                                        logger.warning(f"[PERCEPTION VALIDATION] ⚠️  Unity paused (time gap: {dt:.3f}s) - perception values frozen")
                                    # Don't update last_timestamp here - update at end of function
                                elif dt < 0.1:  # Normal time gap but values frozen = perception died
                                    if self.perception_frozen_frames > 3:
                                        logger.error(f"[PERCEPTION VALIDATION] ❌ PERCEPTION FAILED! "
                                                   f"Values unchanged for {self.perception_frozen_frames} frames "
                                                   f"(Unity still sending frames, dt={dt:.3f}s)")
                                        logger.error(f"  Left: {left_x_vehicle:.3f}m, Right: {right_x_vehicle:.3f}m")
                                        logger.error("  This indicates perception processing has stopped - emergency stop needed")
                                    # Don't update last_timestamp here - update at end of function
                                else:
                                    # Medium time gap - could be either
                                    if self.perception_frozen_frames > 3:
                                        logger.warning(f"[PERCEPTION VALIDATION] ⚠️  Perception values frozen for {self.perception_frozen_frames} frames "
                                                     f"(time gap: {dt:.3f}s)")
                                    # Don't update last_timestamp here - update at end of function
                            else:
                                # Perception is updating - check if we just recovered from frozen timestamp
                                if hasattr(self, 'perception_frozen_frames') and self.perception_frozen_frames > 0:
                                    # Check if timestamp jumped (Unity resumed)
                                    if prev_timestamp is not None:
                                        recovery_dt = timestamp - prev_timestamp
                                        if abs(recovery_dt) > 0.001:  # Timestamp is updating again
                                            logger.info(f"[PERCEPTION VALIDATION] ✓ Unity resumed! Timestamp updated (dt={recovery_dt:.3f}s) after {self.perception_frozen_frames} frozen frames")
                                
                                # Reset frozen counter
                                if hasattr(self, 'perception_frozen_frames'):
                                    if self.perception_frozen_frames > 0:
                                        logger.info(f"[PERCEPTION VALIDATION] ✓ Perception recovered after {self.perception_frozen_frames} frozen frames")
                                    self.perception_frozen_frames = 0
                                # Don't update last_timestamp here - update at end of function
                            
                            if left_jump > max_jump_threshold or right_jump > max_jump_threshold:
                                if has_time_gap:
                                    # After Unity pause, accept the jump (it's expected)
                                    logger.info("[PERCEPTION VALIDATION] Large jump after Unity pause - accepting detection")
                                    logger.info(f"  Left lane jump: {left_jump:.3f}m (threshold: {max_jump_threshold}m)")
                                    logger.info(f"  Right lane jump: {right_jump:.3f}m (threshold: {max_jump_threshold}m)")
                                    logger.info(f"  Previous: left={self.previous_left_lane_x:.3f}m, right={self.previous_right_lane_x:.3f}m")
                                    logger.info(f"  Current: left={left_x_vehicle:.3f}m, right={right_x_vehicle:.3f}m")
                                    # Accept the new detection after Unity pause
                                    left_lane_line_x = float(left_x_vehicle)
                                    right_lane_line_x = float(right_x_vehicle)
                                    # Update previous values
                                    self.previous_left_lane_x = left_x_vehicle
                                    self.previous_right_lane_x = right_x_vehicle
                                else:
                                    if detection_method == "segmentation":
                                        # Segmentation outputs can change more between frames without being wrong.
                                        # Accept the new detection to avoid freezing on stale data.
                                        left_lane_line_x = float(left_x_vehicle)
                                        right_lane_line_x = float(right_x_vehicle)
                                        self.previous_left_lane_x = left_x_vehicle
                                        self.previous_right_lane_x = right_x_vehicle
                                        logger.info(
                                            f"[Frame {self.frame_count}] [PERCEPTION VALIDATION] Segmentation jump accepted "
                                            f"(left={left_jump:.3f}m, right={right_jump:.3f}m, threshold={max_jump_threshold:.3f}m)"
                                        )
                                        # Skip stale handling for segmentation
                                        if not using_stale_data:
                                            using_stale_data = False
                                            stale_data_reason = None
                                    else:
                                        # Normal jump detection (no time gap) - reject if too large
                                        logger.error(f"[PERCEPTION VALIDATION] ❌ SUDDEN LANE JUMP DETECTED! (Frame {self.frame_count})")
                                        logger.error(f"  Left lane jump: {left_jump:.3f}m (threshold: {max_jump_threshold}m)")
                                        logger.error(f"  Right lane jump: {right_jump:.3f}m (threshold: {max_jump_threshold}m)")
                                        logger.error(f"  Previous: left={self.previous_left_lane_x:.3f}m, right={self.previous_right_lane_x:.3f}m")
                                        logger.error(f"  Current: left={left_x_vehicle:.3f}m, right={right_x_vehicle:.3f}m")
                                        logger.error("  Rejecting detection - using previous frame's values")
                                        
                                        # Use previous frame's values instead of current (failed) detection
                                        left_lane_line_x = float(self.previous_left_lane_x)
                                        right_lane_line_x = float(self.previous_right_lane_x)
                                        # Track that we're using stale data due to jump detection
                                        using_stale_data = True
                                        stale_data_reason = "jump_detection"
                                        reject_reason = reject_reason or stale_data_reason
                                        clamp_events.append("jump_detection_reject")
                                        left_jump_magnitude = left_jump
                                        right_jump_magnitude = right_jump
                                        jump_threshold_used = max_jump_threshold
                                        logger.warning(f"[Frame {self.frame_count}] Using STALE perception data (reason: {stale_data_reason}) - jump detected: left={left_jump:.3f}m, right={right_jump:.3f}m (threshold={jump_threshold_used:.3f}m)")
                            else:
                                # NEW: Detect lane width/center changes that indicate perception instability
                                # Even if jump detection didn't trigger, small changes can cause control errors
                                # If lane width or center changes significantly, use stale data to prevent control oscillations
                                lane_center_current = (left_x_vehicle + right_x_vehicle) / 2.0
                                lane_width_current = calculated_lane_width
                                
                                if self.previous_left_lane_x is not None and self.previous_right_lane_x is not None:
                                    lane_center_previous = (self.previous_left_lane_x + self.previous_right_lane_x) / 2.0
                                    lane_width_previous = self.previous_right_lane_x - self.previous_left_lane_x
                                    
                                    perception_cfg = self.config.get("perception", {}) if isinstance(self.config, dict) else {}
                                    max_lane_delta = float(perception_cfg.get("lane_line_change_clamp_m", 0.0))
                                    if max_lane_delta > 0.0:
                                        clamped_left, clamped_right, was_lane_clamped = clamp_lane_line_deltas(
                                            left_x_vehicle,
                                            right_x_vehicle,
                                            self.previous_left_lane_x,
                                            self.previous_right_lane_x,
                                            max_lane_delta,
                                        )
                                        if was_lane_clamped:
                                            clamp_events.append("lane_line_delta_clamped")
                                            left_x_vehicle = clamped_left
                                            right_x_vehicle = clamped_right
                                            lane_center_current = (left_x_vehicle + right_x_vehicle) / 2.0
                                            lane_width_current = right_x_vehicle - left_x_vehicle
                                            calculated_lane_width = lane_width_current
                                            logger.warning(
                                                f"[Frame {self.frame_count}] [PERCEPTION VALIDATION] Lane line clamp "
                                                f"(left_delta={left_x_vehicle - self.previous_left_lane_x:.3f}m, "
                                                f"right_delta={right_x_vehicle - self.previous_right_lane_x:.3f}m)"
                                            )

                                    max_center_delta = float(perception_cfg.get("lane_center_change_clamp_m", 0.4))
                                    max_width_delta = float(perception_cfg.get("lane_width_change_clamp_m", 0.5))
                                    clamped_center, clamped_width, was_clamped = clamp_lane_center_and_width(
                                        lane_center_current,
                                        lane_width_current,
                                        lane_center_previous,
                                        lane_width_previous,
                                        max_center_delta,
                                        max_width_delta,
                                    )
                                    if was_clamped:
                                        clamp_events.append("lane_center_width_clamped")
                                        left_x_vehicle = clamped_center - (clamped_width / 2.0)
                                        right_x_vehicle = clamped_center + (clamped_width / 2.0)
                                        lane_center_current = clamped_center
                                        lane_width_current = clamped_width
                                        calculated_lane_width = lane_width_current
                                        logger.warning(
                                            f"[Frame {self.frame_count}] [PERCEPTION VALIDATION] Lane shift clamped "
                                            f"(center_delta={lane_center_current - lane_center_previous:.3f}m, "
                                            f"width_delta={lane_width_current - lane_width_previous:.3f}m)"
                                        )

                                    # NEW: Temporal filtering + measurement gating for lane center/width
                                    center_gate_m = float(perception_cfg.get("lane_center_gate_m", 0.0))
                                    width_gate_m = float(perception_cfg.get("lane_width_gate_m", 0.0))
                                    center_alpha = float(perception_cfg.get("lane_center_ema_alpha", 0.0))
                                    width_alpha = float(perception_cfg.get("lane_width_ema_alpha", 0.0))
                                    center_beta = float(perception_cfg.get("lane_center_ab_beta", 0.0))
                                    width_beta = float(perception_cfg.get("lane_width_ab_beta", 0.0))
                                    now_time = float(timestamp) if timestamp is not None else None
                                    if self.last_lane_filter_time is None:
                                        dt_lane = 1.0 / 30.0
                                    elif now_time is not None:
                                        dt_lane = max(1e-3, now_time - self.last_lane_filter_time)
                                    else:
                                        dt_lane = 1.0 / 30.0

                                    if center_beta > 0.0 or width_beta > 0.0:
                                        (
                                            lane_center_current,
                                            lane_width_current,
                                            self.lane_center_ab,
                                            self.lane_center_ab_vel,
                                            self.lane_width_ab,
                                            self.lane_width_ab_vel,
                                            gate_events,
                                            gated,
                                        ) = apply_lane_alpha_beta_gating(
                                            lane_center_current,
                                            lane_width_current,
                                            self.lane_center_ab,
                                            self.lane_center_ab_vel,
                                            self.lane_width_ab,
                                            self.lane_width_ab_vel,
                                            center_gate_m,
                                            width_gate_m,
                                            center_alpha,
                                            center_beta,
                                            width_alpha,
                                            width_beta,
                                            dt_lane,
                                        )
                                    elif center_alpha > 0.0 or width_alpha > 0.0:
                                        (
                                            lane_center_current,
                                            lane_width_current,
                                            self.lane_center_ema,
                                            self.lane_width_ema,
                                            gate_events,
                                            gated,
                                        ) = apply_lane_ema_gating(
                                            lane_center_current,
                                            lane_width_current,
                                            self.lane_center_ema,
                                            self.lane_width_ema,
                                            center_gate_m,
                                            width_gate_m,
                                            center_alpha,
                                            width_alpha,
                                        )
                                        if gate_events:
                                            clamp_events.extend(gate_events)
                                        if gated:
                                            using_stale_data = True
                                        stale_data_reason = stale_data_reason or "lane_center_width_gate"
                                        reject_reason = reject_reason or stale_data_reason
                                        left_x_vehicle = lane_center_current - (lane_width_current / 2.0)
                                        right_x_vehicle = lane_center_current + (lane_width_current / 2.0)
                                        calculated_lane_width = lane_width_current
                                    if now_time is not None:
                                        self.last_lane_filter_time = now_time
                                    
                                    # NEW: Adaptive thresholds based on path curvature
                                    
                                    # On curves (|curvature| > 0.1 1/m), lane positions change naturally
                                    # Use more lenient thresholds to avoid rejecting legitimate curve-induced changes
                                    if abs_curvature > 0.1:  # On a curve
                                        width_change_threshold = 1.5  # meters - much more lenient for curves (increased from 1.3m to 1.5m)
                                        center_shift_threshold = 1.2  # meters - more lenient for curves (increased from 0.8m to 1.2m)
                                        threshold_mode = "curve"
                                    else:  # On straight road
                                        width_change_threshold = 0.3  # meters - strict for straights (catches noise)
                                        center_shift_threshold = 0.2  # meters - strict for straights (catches noise)
                                        threshold_mode = "straight"
                                    
                                    # Detect significant lane width change (threshold depends on curvature)
                                    width_change = abs(lane_width_current - lane_width_previous)
                                    
                                    # Detect significant lane center shift (threshold depends on curvature)
                                    center_shift = abs(lane_center_current - lane_center_previous)
                                    
                                    # CRITICAL FIX: Skip instability check on first frame with previous data (frame 1)
                                    # On frame 1, this is the first comparison, and small changes are expected
                                    # due to normal perception variation. Only check instability from frame 2 onwards.
                                    skip_instability_check = (self.frame_count == 1)  # Skip on frame 1 only
                                    instability_breach = (
                                        width_change > width_change_threshold
                                        or center_shift > center_shift_threshold
                                    )
                                    if not skip_instability_check and detection_method == "cv" and instability_breach:
                                        self.cv_instability_breach_frames += 1
                                    elif detection_method == "cv":
                                        self.cv_instability_breach_frames = 0
                                    
                                    if not skip_instability_check and instability_breach:
                                        if detection_method == "segmentation":
                                            # Segmentation can have larger natural shifts without being wrong.
                                            # Accept the new detection to avoid freezing.
                                            left_lane_line_x = float(left_x_vehicle)
                                            right_lane_line_x = float(right_x_vehicle)
                                            self.previous_left_lane_x = left_x_vehicle
                                            self.previous_right_lane_x = right_x_vehicle
                                            logger.info(
                                                f"[Frame {self.frame_count}] [PERCEPTION VALIDATION] Segmentation instability accepted "
                                                f"(width_change={width_change:.3f}m, center_shift={center_shift:.3f}m)"
                                            )
                                            if not using_stale_data:
                                                using_stale_data = False
                                                stale_data_reason = None
                                        elif detection_method == "cv" and self.cv_instability_breach_frames < 2:
                                            # For CV mode, require persistence before freezing to avoid latching on tiny one-frame overshoots.
                                            left_lane_line_x = float(left_x_vehicle)
                                            right_lane_line_x = float(right_x_vehicle)
                                            self.previous_left_lane_x = left_x_vehicle
                                            self.previous_right_lane_x = right_x_vehicle
                                            using_stale_data = False
                                            stale_data_reason = None
                                            logger.info(
                                                f"[Frame {self.frame_count}] [PERCEPTION VALIDATION] CV instability pre-filter accepted "
                                                f"(count={self.cv_instability_breach_frames}, width_change={width_change:.3f}m, center_shift={center_shift:.3f}m)"
                                            )
                                        else:
                                            # Perception instability detected - use stale data to prevent control oscillations
                                            logger.warning(f"[Frame {self.frame_count}] [PERCEPTION VALIDATION] ⚠️  PERCEPTION INSTABILITY DETECTED!")
                                            logger.warning(f"  Width change: {width_change:.3f}m (threshold: {width_change_threshold}m [{threshold_mode}])")
                                            logger.warning(f"  Center shift: {center_shift:.3f}m (threshold: {center_shift_threshold}m [{threshold_mode}])")
                                            logger.warning(f"  Path curvature: {path_curvature:.4f} 1/m (|curvature|={abs_curvature:.4f})")
                                            logger.warning("  Using stale data to prevent control oscillations")
                                            
                                            using_stale_data = True
                                            stale_data_reason = "perception_instability"
                                            reject_reason = reject_reason or stale_data_reason
                                            # Store actual detected values for debugging (even though we reject them)
                                            actual_detected_left_lane_x = float(left_x_vehicle)
                                            actual_detected_right_lane_x = float(right_x_vehicle)
                                            instability_width_change = width_change
                                            instability_center_shift = center_shift
                                            # Use stale values for control
                                            left_lane_line_x = float(self.previous_left_lane_x)
                                            right_lane_line_x = float(self.previous_right_lane_x)
                                            # CRITICAL FIX: Don't update previous values when instability detected
                                            # This allows recovery: next frame will compare against stale values
                                        # If next frame's detection is within thresholds vs stale values, recovery happens
                                    else:
                                        # Valid detection - use it (either first valid detection or recovery from instability)
                                        if using_stale_data and stale_data_reason == "perception_instability":
                                            logger.info(f"[Frame {self.frame_count}] [PERCEPTION VALIDATION] ✅ RECOVERED from perception instability!")
                                            logger.info(f"  Width change: {width_change:.3f}m (threshold: {width_change_threshold}m [{threshold_mode}])")
                                            logger.info(f"  Center shift: {center_shift:.3f}m (threshold: {center_shift_threshold}m [{threshold_mode}])")
                                            logger.info("  Accepting new detection and updating previous values")
                                        
                                        left_lane_line_x = float(left_x_vehicle)
                                        right_lane_line_x = float(right_x_vehicle)
                                        
                                        # Update previous values for next frame (enables recovery)
                                        self.previous_left_lane_x = left_x_vehicle
                                        self.previous_right_lane_x = right_x_vehicle
                                        if detection_method == "cv":
                                            self.cv_instability_breach_frames = 0
                                else:
                                    # First frame or no previous data - accept current detection
                                    if self.frame_count == 0:
                                        logger.debug(f"[Frame {self.frame_count}] [DEBUG] First frame - assigning values: left={left_x_vehicle:.3f}m, right={right_x_vehicle:.3f}m")
                                    left_lane_line_x = float(left_x_vehicle)
                                    right_lane_line_x = float(right_x_vehicle)
                                    if self.frame_count == 0:
                                        logger.debug(f"[Frame {self.frame_count}] [DEBUG] After assignment: left_lane_line_x={left_lane_line_x}, right_lane_line_x={right_lane_line_x}")
                                    
                                    # Update previous values for next frame
                                    self.previous_left_lane_x = left_x_vehicle
                                    self.previous_right_lane_x = right_x_vehicle
            except Exception as e:  # CHANGED: Catch ALL exceptions, not just AttributeError/TypeError/ValueError
                # If planner is a Mock or conversion fails, skip lane data calculation
                # This is acceptable in unit tests
                # CRITICAL: Log the exception so we can diagnose why coordinate conversion failed
                import traceback
                logger.error(f"[Frame {self.frame_count}] [COORD CONVERSION] Exception during coordinate conversion: {type(e).__name__}: {e}")
                logger.error(f"[Frame {self.frame_count}]   Traceback:\n{traceback.format_exc()}")
                logger.error(f"[Frame {self.frame_count}]   This means left_lane_line_x and right_lane_line_x will remain None (converted to 0.0 in recorder)")
                logger.error(f"[Frame {self.frame_count}]   If overlays look good, this is a bug - coordinate conversion should work!")
                
                # DEBUG: Log what we have
                logger.error(f"[Frame {self.frame_count}]   DEBUG: lane_coeffs type={type(lane_coeffs)}, len={len(lane_coeffs) if hasattr(lane_coeffs, '__len__') else 'N/A'}")
                if len(lane_coeffs) >= 2:
                    logger.error(f"[Frame {self.frame_count}]   DEBUG: lane_coeffs[0] type={type(lane_coeffs[0])}, shape={lane_coeffs[0].shape if hasattr(lane_coeffs[0], 'shape') else 'N/A'}")
                    logger.error(f"[Frame {self.frame_count}]   DEBUG: lane_coeffs[1] type={type(lane_coeffs[1])}, shape={lane_coeffs[1].shape if hasattr(lane_coeffs[1], 'shape') else 'N/A'}")
                logger.error(f"[Frame {self.frame_count}]   DEBUG: planner type={type(planner) if 'planner' in locals() else 'not defined'}")
                if 'planner' in locals():
                    logger.error(f"[Frame {self.frame_count}]   DEBUG: planner has image_height={hasattr(planner, 'image_height')}, image_width={hasattr(planner, 'image_width')}")
                    logger.error(f"[Frame {self.frame_count}]   DEBUG: planner has _convert_image_to_vehicle_coords={hasattr(planner, '_convert_image_to_vehicle_coords')}")
                
                # On first frame, try to use a fallback if we have valid coefficients
                if self.frame_count == 0:
                    logger.warning(f"[Frame {self.frame_count}] ⚠️  FIRST FRAME: Coordinate conversion failed but overlays look good!")
                    logger.warning(f"[Frame {self.frame_count}]   This suggests a bug in coordinate conversion, not perception")
                    # Don't set to None - leave as None so the check at line 1013 can catch it
                    # But log a warning so we know what happened
                pass
        elif num_lanes_detected == 1:
            # FIX: Handle single-lane-line detection case
            # When only 1 lane line is detected, estimate the other lane line based on standard lane width
            # CRITICAL: Check if we have valid coefficients (not None)
            valid_coeffs = [c for c in lane_coeffs if c is not None]
            if len(valid_coeffs) >= 1:
                try:
                    perception_cfg = self.config.get("perception", {}) if isinstance(self.config, dict) else {}
                    width_min = float(perception_cfg.get("lane_width_min_m", 1.0))
                    width_max = float(perception_cfg.get("lane_width_max_m", 10.0))
                    last_width = None
                    if self.previous_left_lane_x is not None and self.previous_right_lane_x is not None:
                        last_width = float(self.previous_right_lane_x - self.previous_left_lane_x)

                    planner = self.trajectory_planner.planner if hasattr(self.trajectory_planner, 'planner') else self.trajectory_planner
                    image_height = planner.image_height if hasattr(planner, 'image_height') else 480.0
                    image_width = planner.image_width if hasattr(planner, 'image_width') else 640.0
                    reference_lookahead = planner.lookahead_distance if hasattr(planner, 'lookahead_distance') else 8.0
                    
                    # Evaluate single lane line at lookahead
                    y_image_at_lookahead = 350  # Within detected range
                    single_lane_coeffs = valid_coeffs[0]  # Use first valid coefficient
                    single_x_image = np.polyval(single_lane_coeffs, y_image_at_lookahead)
                    
                    # Convert to vehicle coordinates
                    conversion_distance = reference_lookahead
                    # Use Unity's reported horizontal FOV if available
                    camera_horizontal_fov = vehicle_state_dict.get('cameraHorizontalFOV')
                    if camera_horizontal_fov is None:
                        camera_horizontal_fov = vehicle_state_dict.get('camera_horizontal_fov', None)
                    
                    single_x_vehicle, _ = planner._convert_image_to_vehicle_coords(
                        single_x_image, y_image_at_lookahead, lookahead_distance=conversion_distance,
                        horizontal_fov_override=camera_horizontal_fov if camera_horizontal_fov and camera_horizontal_fov > 0 else None
                    )
                    
                    # Estimate other lane based on last known width when available
                    standard_lane_width = 7.0  # meters
                    
                    # Determine if detected lane is left or right based on position relative to image center
                    image_center_x = image_width / 2.0
                    is_left_lane = single_x_image < image_center_x
                    left_lane_line_x, right_lane_line_x = estimate_single_lane_pair(
                        single_x_vehicle=single_x_vehicle,
                        is_left_lane=is_left_lane,
                        last_width=last_width,
                        default_width=standard_lane_width,
                        width_min=width_min,
                        width_max=width_max,
                    )
                    
                    logger.warning(f"[SINGLE LANE] Only 1 lane line detected. Detected at x={single_x_vehicle:.3f}m. "
                                 f"Estimated: left={left_lane_line_x:.3f}m, right={right_lane_line_x:.3f}m "
                                 f"(last_width={last_width}, bounds=({width_min}-{width_max}))")
                except (AttributeError, TypeError, ValueError) as e:
                    logger.warning(f"[SINGLE LANE] Failed to process single lane line: {e}")
                    # Leave as 0.0 (default)
                    pass
            else:
                # num_lanes_detected = 1 but no valid coefficients - detection failed
                logger.warning(f"[SINGLE LANE] num_lanes_detected=1 but no valid coefficients! "
                             f"lane_coeffs={lane_coeffs}. Detection may have failed.")
                # Leave as 0.0 (default)
        
        # CRITICAL FIX: If both lanes are 0.0 (perception failed), set to None
        # This ensures reference_point will be None, preventing movement
        if left_lane_line_x is not None and right_lane_line_x is not None:
            if abs(left_lane_line_x) < 0.001 and abs(right_lane_line_x) < 0.001:
                logger.warning("[PERCEPTION] Perception failed (both lanes = 0.0) - setting to None to prevent movement")
                left_lane_line_x = None
                right_lane_line_x = None
        
        # Record perception output with diagnostic fields
        # NEW: Perception health monitoring
        # Track detection quality (good = 2+ lanes detected)
        is_good_detection = num_lanes_detected >= 2
        timestamp_frozen = False
        
        # NEW: Check events that affect health score.
        bad_events = []
        managed_events = []
        
        # Check 1: Insufficient lanes
        if not is_good_detection:
            bad_events.append("insufficient_lanes")
        
        # Check 2: Extreme polynomial coefficients (if we have coefficients)
        # CRITICAL: Only check where we actually use the polynomial (lookahead distance)
        # Don't check top of image (y=0) - extrapolation there can be extreme on curves
        if lane_coeffs is not None and len(lane_coeffs) >= 2:
            # Get image dimensions (try to get from planner, fallback to defaults)
            try:
                planner = self.trajectory_planner.planner if hasattr(self.trajectory_planner, 'planner') else self.trajectory_planner
                image_height = planner.image_height if hasattr(planner, 'image_height') else 480.0
                image_width = planner.image_width if hasattr(planner, 'image_width') else 640.0
            except:
                image_height = 480.0
                image_width = 640.0
            
            # Get lookahead y position - use camera_8m_screen_y if available (extracted earlier in _process_frame)
            # Otherwise use middle of image as fallback
            if camera_8m_screen_y is not None and camera_8m_screen_y > 0:
                health_y_image_at_lookahead = int(camera_8m_screen_y)
            else:
                # Fallback: Use middle of image (typical lookahead position)
                health_y_image_at_lookahead = int(image_height * 0.7)  # Bottom 30% of image
            
            # Only check at lookahead distance and bottom (where we actually use the polynomial)
            # Only check at lookahead distance (where we actually use the polynomial)
            # Don't check bottom - missing dashed lines can cause incorrect extrapolation there
            health_check_y_positions = [health_y_image_at_lookahead]
            
            # Use same thresholds as validation
            health_max_reasonable_x = image_width * 2.5
            health_min_reasonable_x = -image_width * 1.5
            
            for lane_idx, lane_coeffs_item in enumerate(lane_coeffs):
                # Check if lane_coeffs_item is a valid array (not None, not a scalar)
                if lane_coeffs_item is not None:
                    # Convert to numpy array if needed and check shape
                    lane_coeffs_array = np.asarray(lane_coeffs_item)
                    if lane_coeffs_array.ndim == 1 and len(lane_coeffs_array) >= 3:
                        for y_check in health_check_y_positions:
                            x_eval = lane_coeffs_array[0] * y_check * y_check + lane_coeffs_array[1] * y_check + lane_coeffs_array[2]
                            if x_eval < health_min_reasonable_x or x_eval > health_max_reasonable_x:
                                bad_events.append("extreme_coefficients")
                                break
                        if "extreme_coefficients" in bad_events:
                            break
        
        # Check 3: Using stale data.
        # Treat dashed-line visibility fallback as managed behavior (informational),
        # not a hard health failure, unless additional bad events are present.
        if using_stale_data:
            managed_visibility_fallback = False
            if stale_data_reason in {"left_lane_low_visibility", "right_lane_low_visibility"}:
                managed_visibility_fallback = True
            if not managed_visibility_fallback and clamp_events:
                managed_visibility_fallback = any(
                    ev in {"left_lane_lookahead_low_visibility", "right_lane_lookahead_low_visibility"}
                    for ev in clamp_events
                )
            if not managed_visibility_fallback:
                bad_events.append("stale_data")
            else:
                managed_events.append("managed_visibility_fallback")
        
        # Track most recent bad events for UI visibility when health lags.
        if bad_events:
            self.last_perception_bad_events = list(bad_events)

        # Track health: good detection AND no hard bad events = truly good
        is_truly_good = is_good_detection and len(bad_events) == 0
        
        # DEBUG: Log health monitoring on early frames to diagnose issues
        if self.frame_count <= 5:
            logger.debug(f"[Frame {self.frame_count}] [HEALTH DEBUG] is_good_detection={is_good_detection}, num_lanes={num_lanes_detected}, bad_events={bad_events}, is_truly_good={is_truly_good}")
        
        self.perception_health_history.append(is_truly_good)
        if len(self.perception_health_history) > self.perception_health_history_size:
            self.perception_health_history.pop(0)
        
        # Update consecutive bad detection frames
        if is_truly_good:
            self.consecutive_bad_detection_frames = 0
        else:
            self.consecutive_bad_detection_frames += 1
            # DEBUG: Log why frame is marked as bad
            if self.frame_count <= 5:
                logger.debug(f"[Frame {self.frame_count}] [HEALTH DEBUG] Frame marked as BAD: consecutive_bad={self.consecutive_bad_detection_frames}, is_good_detection={is_good_detection}, bad_events={bad_events}")
        
        # Calculate health score (fraction of recent frames with truly good detection)
        # NEW: Penalize frames with bad events even if detection rate is good
        if len(self.perception_health_history) > 0:
            good_detection_count = sum(self.perception_health_history)
            base_score = good_detection_count / len(self.perception_health_history)
            
            # Apply immediate penalties for current-frame events.
            # Hard failures penalize strongly; managed visibility fallback penalizes lightly.
            hard_penalty = min(0.3, len(bad_events) * 0.1)
            managed_penalty = min(0.06, len(managed_events) * 0.02)
            penalty = hard_penalty + managed_penalty
            self.perception_health_score = max(0.0, base_score - penalty)
        else:
            self.perception_health_score = 1.0 if len(bad_events) == 0 else 0.7
        
        # Determine health status
        if self.perception_health_score >= 0.9:
            self.perception_health_status = "healthy"
        elif self.perception_health_score >= 0.7:
            self.perception_health_status = "degraded"
        elif self.perception_health_score >= 0.5:
            self.perception_health_status = "poor"
        else:
            self.perception_health_status = "critical"

        recent_bad_events = None
        if not bad_events and self.perception_health_status != "healthy":
            recent_bad_events = getattr(self, "last_perception_bad_events", None)
        
        # Log health degradation warnings
        if self.consecutive_bad_detection_frames >= 10 and self.consecutive_bad_detection_frames % 10 == 0:
            logger.warning(f"[Frame {self.frame_count}] ⚠️  Perception health degraded: {self.consecutive_bad_detection_frames} consecutive bad frames, "
                         f"health score: {self.perception_health_score:.2f} ({self.perception_health_status})")
        
        # CRITICAL: Update last_timestamp at the END of processing (after all timestamp comparisons)
        self.last_timestamp = timestamp

        # Fallback: if lane positions are missing but we have previous values, use stale data
        # This prevents left/right/width from collapsing to 0.0 when validation rejects positions.
        if (left_lane_line_x is None or right_lane_line_x is None) and \
           hasattr(self, 'previous_left_lane_x') and hasattr(self, 'previous_right_lane_x') and \
           self.previous_left_lane_x is not None and self.previous_right_lane_x is not None:
            using_stale_data = True
            stale_data_reason = stale_data_reason or "missing_lane_positions"
            reject_reason = reject_reason or stale_data_reason
            clamp_events.append("missing_lane_positions")
            left_lane_line_x = float(self.previous_left_lane_x)
            right_lane_line_x = float(self.previous_right_lane_x)
            logger.warning(
                f"[Frame {self.frame_count}] Using STALE perception data (reason: {stale_data_reason}) - "
                "lane positions missing after validation"
            )
        
        perception_output = PerceptionOutput(
            timestamp=timestamp,
            lane_line_coefficients=np.array([c for c in lane_coeffs if c is not None]) if any(c is not None for c in lane_coeffs) else None,
            confidence=confidence,
            detection_method=detection_method,
            num_lanes_detected=num_lanes_detected,
            left_lane_line_x=left_lane_line_x,
            right_lane_line_x=right_lane_line_x,
            # Diagnostic fields for tracking stale data usage
            using_stale_data=using_stale_data,
            stale_data_reason=stale_data_reason,
            left_jump_magnitude=left_jump_magnitude,
            right_jump_magnitude=right_jump_magnitude,
            jump_threshold=jump_threshold_used,
            # NEW: Diagnostic fields for perception instability
            actual_detected_left_lane_x=actual_detected_left_lane_x,
            actual_detected_right_lane_x=actual_detected_right_lane_x,
            instability_width_change=instability_width_change,
            instability_center_shift=instability_center_shift,
            # NEW: Perception health monitoring fields
            consecutive_bad_detection_frames=self.consecutive_bad_detection_frames,
            perception_health_score=self.perception_health_score,
            perception_health_status=self.perception_health_status,
            perception_bad_events=bad_events if bad_events else None,
            perception_bad_events_recent=recent_bad_events,
            perception_timestamp_frozen=timestamp_frozen,
            perception_clamp_events=clamp_events if clamp_events else None,
            reject_reason=finalize_reject_reason(reject_reason, stale_data_reason, clamp_events),
            # NEW: Points used for polynomial fitting (for debug visualization)
            fit_points_left=fit_points_left,
            fit_points_right=fit_points_right,
            segmentation_mask_png=segmentation_mask_png
        )
        
        # --- Input preparation (stays in av_stack) ---
        path_curvature_raw = vehicle_state_dict.get('groundTruthPathCurvature')
        if path_curvature_raw is None:
            path_curvature_raw = vehicle_state_dict.get('ground_truth_path_curvature', 0.0)
        path_curvature_smoothed = self._smooth_path_curvature(
            float(path_curvature_raw) if isinstance(path_curvature_raw, (int, float)) else 0.0,
            current_speed,
            timestamp,
        )
        path_curvature = (
            path_curvature_smoothed
            if self.curvature_smoothing_enabled
            else path_curvature_raw
        )
        max_lateral_accel = self.trajectory_config.get('max_lateral_accel', 2.5)
        min_curve_speed = self.trajectory_config.get('min_curve_speed', 0.0)
        curvature_bins = self.trajectory_config.get('curvature_bins', None)
        if isinstance(path_curvature, (int, float)):
            max_lateral_accel, min_curve_speed = select_curvature_bin_limits(
                abs(path_curvature),
                curvature_bins,
                max_lateral_accel,
                min_curve_speed,
            )

        # Get preview curvature for anticipatory deceleration
        preview_curvature = None
        if lane_coeffs is not None:
            preview_lookahead = reference_lookahead * self.speed_governor.config.curve_preview_lookahead_scale
            preview_curvature = self.trajectory_planner.get_curvature_at_lookahead(
                lane_coeffs, preview_lookahead
            )

        # Use perception horizon (lookahead_distance from config) for guardrail,
        # not the control reference_lookahead
        perception_horizon_m = float(self.trajectory_config.get('lookahead_distance', 20.0))

        # Stale perception speed hold: don't accelerate when blind
        if using_stale_data:
            self.consecutive_stale_frames += 1
        else:
            self.consecutive_stale_frames = 0
        stale_speed_hold_active = (
            self.consecutive_stale_frames >= self.stale_speed_hold_threshold
        )
        effective_track_limit = float(self.original_target_speed)
        if stale_speed_hold_active and current_speed > 0:
            effective_track_limit = min(effective_track_limit, float(current_speed))

        # --- Speed Governor: single call replaces 12 serial suppression layers ---
        gov_output: SpeedGovernorOutput = self.speed_governor.compute_target_speed(
            track_speed_limit=effective_track_limit,
            curvature=float(path_curvature) if isinstance(path_curvature, (int, float)) else 0.0,
            preview_curvature=float(preview_curvature) if preview_curvature is not None else None,
            perception_horizon_m=perception_horizon_m,
            current_speed=float(current_speed),
            timestamp=float(timestamp) if timestamp is not None else 0.0,
            confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
        )

        adjusted_target_speed = gov_output.target_speed
        target_speed_final = adjusted_target_speed
        base_speed = float(self.original_target_speed)
        target_speed_planned = gov_output.planned_speed
        planned_accel = gov_output.planned_accel
        target_speed_ramp_active = False
        target_speed_slew_active = False

        # Telemetry compatibility: legacy variables for HDF5 recording
        curve_speed_limit = gov_output.comfort_speed if gov_output.comfort_speed < 1e6 else None
        curve_preview_speed_limit = gov_output.preview_speed
        steering_speed_guard_active = False
        curve_mode_speed_cap_active = False
        curve_mode_speed_cap_clamped = False
        curve_mode_speed_cap_value = None

        # Horizon guardrail diagnostics from governor
        speed_horizon_guardrail_diag = {}
        for k, v in gov_output.horizon_diag.items():
            if k.startswith('diag_speed_horizon_guardrail_'):
                speed_horizon_guardrail_diag[k] = float(v) if isinstance(v, (int, float)) else float('nan')
        if not speed_horizon_guardrail_diag:
            speed_horizon_guardrail_diag = {
                'diag_speed_horizon_guardrail_active': 1.0 if gov_output.horizon_guardrail_active else 0.0,
                'diag_speed_horizon_guardrail_margin_m': gov_output.horizon_guardrail_margin_m,
                'diag_speed_horizon_guardrail_horizon_m': gov_output.effective_horizon_m,
                'diag_speed_horizon_guardrail_time_headway_s': self.speed_governor.config.horizon_guardrail_time_headway_s,
                'diag_speed_horizon_guardrail_margin_buffer_m': self.speed_governor.config.horizon_guardrail_margin_m,
                'diag_speed_horizon_guardrail_allowed_speed_mps': gov_output.horizon_speed if gov_output.horizon_speed is not None else float('nan'),
                'diag_speed_horizon_guardrail_target_speed_before_mps': float(self.original_target_speed),
                'diag_speed_horizon_guardrail_target_speed_after_mps': adjusted_target_speed,
            }

        if adjusted_target_speed < base_speed and self.frame_count % 30 == 0:
            curve_speed = None
            if isinstance(path_curvature, (int, float)) and abs(path_curvature) > 1e-6 and max_lateral_accel > 0:
                curve_speed = (max_lateral_accel / abs(path_curvature)) ** 0.5
                if min_curve_speed > 0:
                    curve_speed = max(curve_speed, min_curve_speed)
            logger.info(
                f"[Frame {self.frame_count}] Speed limit applied: "
                f"base={base_speed:.2f} m/s, "
                f"map_limit={float(speed_limit) if speed_limit else 0.0:.2f} m/s, "
                f"curve_limit={curve_speed:.2f} m/s"
                if curve_speed is not None
                else f"[Frame {self.frame_count}] Speed limit applied: "
                     f"base={base_speed:.2f} m/s, "
                     f"map_limit={float(speed_limit) if speed_limit else 0.0:.2f} m/s"
            )
        
        # Update trajectory planner's target speed (if it supports dynamic updates)
        if hasattr(self.trajectory_planner, 'planner') and hasattr(self.trajectory_planner.planner, 'target_speed'):
            self.trajectory_planner.planner.target_speed = adjusted_target_speed
        elif hasattr(self.trajectory_planner, 'target_speed'):
            self.trajectory_planner.target_speed = adjusted_target_speed
        
        # Log speed adjustment if health is degraded
        if self.perception_health_status != "healthy" and self.frame_count % 30 == 0:  # Log every second
            logger.warning(f"[Frame {self.frame_count}] ⚠️  Perception health-based speed reduction: "
                         f"{self.perception_health_status} -> target speed: {adjusted_target_speed:.2f} m/s "
                         f"(original: {self.original_target_speed:.2f} m/s)")

        # Phase 2 integration: pass effective horizon policy into trajectory generation.
        gov_horizon_m = gov_output.effective_horizon_m if gov_output.effective_horizon_m > 0 else float(
            dynamic_horizon_diag.get("diag_dynamic_effective_horizon_m", reference_lookahead)
        )
        gov_horizon_applied = 1.0 if gov_output.horizon_guardrail_active else float(
            dynamic_horizon_diag.get("diag_dynamic_effective_horizon_applied", 0.0)
        )
        vehicle_state_dict["dynamicEffectiveHorizonMeters"] = gov_horizon_m
        vehicle_state_dict["dynamic_effective_horizon_m"] = gov_horizon_m
        vehicle_state_dict["dynamicEffectiveHorizonApplied"] = gov_horizon_applied
        vehicle_state_dict["dynamic_effective_horizon_applied"] = gov_horizon_applied
        
        # 2. Trajectory Planning: Plan path
        trajectory = self.trajectory_planner.plan(lane_coeffs, vehicle_state_dict)
        oracle_points_xy = self._extract_oracle_points_xy(vehicle_state_dict)
        trajectory_source_active = self.trajectory_source if self.trajectory_source in ("planner", "oracle") else "planner"
        
        # Get current speed for safety checks (check BEFORE computing control)
        max_speed = self.safety_config.get('max_speed', 10.0)
        emergency_threshold = self.safety_config.get('emergency_brake_threshold', 2.0)
        prevention_threshold = self.safety_config.get('speed_prevention_threshold', 0.85)
        prevention_brake_threshold = self.safety_config.get('speed_prevention_brake_threshold', 0.9)
        prevention_brake_amount = self.safety_config.get('speed_prevention_brake_amount', 0.2)
        
        # NEW: Safety bounds for lateral error (hard limits)
        # Critical: 89.4% steering direction errors and 8.95m max error indicate need for safety bounds
        max_lateral_error = self.safety_config.get('max_lateral_error', 2.0)  # Hard limit: 2.0m
        recovery_lateral_error = self.safety_config.get('recovery_lateral_error', 1.0)  # Recovery trigger: 1.0m
        emergency_stop_error = self.safety_config.get('emergency_stop_error', 3.0)  # Emergency stop: 3.0m
        
        # Initialize reference_point for logging
        reference_point = None
        
        # Speed prevention: Check speed BEFORE computing control to prevent acceleration
        # Emergency brake if speed is dangerously high
        # FIXED: Still compute steering during braking for safety (steer while braking)
        # CRITICAL FIX: Make emergency brake more gradual to prevent stop-start cycle
        emergency_brake = False
        brake_override = 0.0
        if current_speed > max_speed * emergency_threshold:
            # Emergency brake: Speed is dangerously high (> 1.5x max = 15 m/s)
            # Use proportional braking instead of full brake to prevent sudden stops
            speed_excess = current_speed - (max_speed * emergency_threshold)
            # Proportional brake: 0.6 base + up to 0.4 more for extreme overspeed
            brake_override = min(1.0, 0.6 + (speed_excess / 5.0))
            emergency_brake = True
            logger.warning(f"Emergency brake! Speed: {current_speed:.2f} m/s, Brake: {brake_override:.2f}")
        elif current_speed > max_speed:
            # Over speed limit - proportional braking (not full brake)
            speed_excess = current_speed - max_speed
            # Gradual braking: 0.4 base + proportional to excess (max 0.8 total)
            brake_override = min(0.8, 0.4 + (speed_excess / 2.0))
            logger.warning(f"Speed limit exceeded! Speed: {current_speed:.2f} m/s, Brake: {brake_override:.2f}")
        
        # Always compute steering (even during braking) for safety
        # Get reference point - use configurable lookahead
        # NEW: Use direct midpoint computation (simpler, more accurate)
        reference_lookahead = float(reference_lookahead)
        dynamic_horizon_applied = float(
            dynamic_horizon_diag.get("diag_dynamic_effective_horizon_applied", 0.0)
        )
        if dynamic_horizon_applied > 0.5:
            reference_lookahead = float(
                dynamic_horizon_diag.get("diag_dynamic_effective_horizon_m", reference_lookahead)
            )
        # Pass timestamp for jump detection (handles Unity pauses)
        reference_point = self.trajectory_planner.get_reference_point(
            trajectory, 
            lookahead=reference_lookahead,
            lane_coeffs=lane_coeffs,  # Pass lane coefficients for direct computation
            lane_positions={'left_lane_line_x': left_lane_line_x, 'right_lane_line_x': right_lane_line_x},  # Preferred: use vehicle coords
            use_direct=True,  # Use direct midpoint computation
            timestamp=timestamp,  # Pass timestamp for time gap detection
            confidence=confidence,
            dynamic_horizon_diag=dynamic_horizon_diag,
        )

        if trajectory_source_active == 'oracle':
            oracle_ref = self._build_oracle_reference_point(
                oracle_points_xy,
                reference_lookahead,
                adjusted_target_speed,
            )
            if oracle_ref is not None:
                reference_point = oracle_ref
            elif self.frame_count % 60 == 0:
                logger.warning(
                    "[Frame %s] trajectory_source=oracle requested but oracle points unavailable; falling back to planner reference.",
                    self.frame_count,
                )

        # Optional: Override reference x using vehicle-frame lookahead offset from Unity.
        if reference_point is not None and trajectory_source_active != 'oracle':
            use_vehicle_frame_lookahead = self.trajectory_config.get(
                'use_vehicle_frame_lookahead_ref', True
            )
            lookahead_offset = vehicle_state_dict.get('vehicleFrameLookaheadOffset', None)
            if use_vehicle_frame_lookahead and lookahead_offset is not None:
                try:
                    lookahead_offset = float(lookahead_offset)
                    left_scale = float(
                        self.trajectory_config.get('vehicle_frame_lookahead_scale_left', 1.0)
                    )
                    right_scale = float(
                        self.trajectory_config.get('vehicle_frame_lookahead_scale_right', 1.0)
                    )
                    base_scale = float(
                        self.trajectory_config.get('vehicle_frame_lookahead_scale', 1.0)
                    )
                    curvature_sign = 0.0
                    if reference_point is not None:
                        curvature_sign = float(reference_point.get('curvature', 0.0) or 0.0)
                    if curvature_sign == 0.0:
                        curvature_sign = float(
                            vehicle_state_dict.get('groundTruthPathCurvature', 0.0) or 0.0
                        )
                    if curvature_sign > 0.0:
                        lookahead_scale = left_scale
                    elif curvature_sign < 0.0:
                        lookahead_scale = right_scale
                    else:
                        lookahead_scale = base_scale
                    original_x = reference_point.get('x', 0.0)
                    reference_point['vehicle_frame_lookahead_original_x'] = original_x
                    reference_point['vehicle_frame_lookahead_offset'] = lookahead_offset
                    reference_point['vehicle_frame_lookahead_scale'] = lookahead_scale
                    reference_point['x'] = lookahead_offset * lookahead_scale
                    reference_point['method'] = 'vehicle_frame_lookahead'
                except (TypeError, ValueError):
                    pass
        
        if reference_point is None:
            # No valid trajectory - stop vehicle safely
            # FIXED: Still compute steering from last known reference or use zero
            if current_speed > 1.0:
                # Apply brake to slow down (proportional to speed)
                brake_override = max(brake_override, min(1.0, 0.3 + (current_speed / 10.0)))
            
            # Use zero steering when no trajectory (safety fallback)
            control_command = {
                'steering': 0.0, 
                'throttle': 0.0, 
                'brake': brake_override,
                'lateral_error': 0.0,
                'heading_error': 0.0,
                'total_error': 0.0,
                'pid_integral': 0.0,
                'pid_derivative': 0.0
            }
        else:
                # Post-jump cooldown: reduce aggressiveness for a short window after teleport
                if self.post_jump_cooldown_frames > 0:
                    cooldown_scale = self.safety_config.get('post_jump_speed_scale', 0.5)
                    reference_point['velocity'] = reference_point.get('velocity', 8.0) * cooldown_scale

                # Limit reference velocity to prevent requesting high speeds
                if reference_point.get('velocity', 8.0) > max_speed:
                    reference_point['velocity'] = max_speed * 0.9  # 90% of max for safety
                gov_min_speed = self.speed_governor.config.comfort_governor_min_speed
                if gov_min_speed > 0:
                    reference_point['min_speed_floor'] = float(gov_min_speed)

                # B2: Use curvature_preview from A4 to further cap speed.
                # The existing curve_speed_preview uses lane_coeffs at 1.6x;
                # curvature_preview from the reference point provides a
                # complementary signal at 1.5x that survives smoothing.
                curv_preview = float(reference_point.get('curvature_preview', 0.0) or 0.0)
                speed_cap_curvature_target = 0.0
                if (
                    self.curve_mode_speed_cap_enabled
                    and abs(curv_preview) > 0.003
                ):
                    a_lat_max = 0.25 * 9.80665  # 0.25g comfort limit
                    v_max_feasible = float(np.sqrt(a_lat_max / abs(curv_preview)))
                    v_max_feasible = max(v_max_feasible, 3.0)
                    speed_cap_curvature_target = v_max_feasible
                    current_vel = reference_point.get('velocity', 8.0)
                    if current_vel > v_max_feasible:
                        reference_point['velocity'] = v_max_feasible
                reference_point['speed_cap_curvature_target_mps'] = speed_cap_curvature_target
                
                # 3. Control: Compute control commands
                current_state = {
                    'heading': self._extract_heading(vehicle_state_dict),
                    'speed': current_speed,
                    'position': self._extract_position(vehicle_state_dict),
                    'road_center_reference_t': vehicle_state_dict.get(
                        'roadCenterReferenceT',
                        vehicle_state_dict.get('road_center_reference_t')
                    ),
                }
                
                # Get control command with metadata for recording
                control_command = self.controller.compute_control(
                    current_state,
                    reference_point,
                    return_metadata=True,
                    dt=control_dt,
                    reference_accel=planned_accel,
                    using_stale_perception=using_stale_data
                )
                control_command['target_speed_raw'] = base_speed
                control_command['target_speed_post_limits'] = adjusted_target_speed
                control_command['target_speed_planned'] = target_speed_planned
                control_command['target_speed_final'] = target_speed_final
                control_command['target_speed_slew_active'] = target_speed_slew_active
                control_command['target_speed_ramp_active'] = target_speed_ramp_active
                control_command['curve_mode_speed_cap_active'] = curve_mode_speed_cap_active
                control_command['curve_mode_speed_cap_clamped'] = curve_mode_speed_cap_clamped
                control_command['curve_mode_speed_cap_value'] = curve_mode_speed_cap_value
                control_command['speed_governor_active_limiter'] = gov_output.active_limiter
                control_command['speed_governor_comfort_speed'] = gov_output.comfort_speed if gov_output.comfort_speed < 1e6 else -1.0
                control_command['speed_governor_preview_speed'] = gov_output.preview_speed if gov_output.preview_speed is not None else -1.0
                control_command['speed_governor_horizon_speed'] = gov_output.horizon_speed if gov_output.horizon_speed is not None else -1.0
                
                # If emergency braking was triggered, override throttle/brake but keep steering
                if brake_override > 0:
                    control_command['throttle'] = 0.0
                    control_command['brake'] = brake_override
                    # Keep steering for safety (steer while braking)
                
                # Post-jump cooldown: clamp steering/throttle to avoid oscillation
                if self.post_jump_cooldown_frames > 0:
                    steering_scale = self.safety_config.get('post_jump_steering_scale', 0.6)
                    control_command['steering'] = np.clip(
                        control_command.get('steering', 0.0) * steering_scale,
                        -self.controller.lateral_controller.max_steering,
                        self.controller.lateral_controller.max_steering,
                    )
                    control_command['throttle'] = control_command.get('throttle', 0.0) * cooldown_scale
                
                # Speed prevention: Limit throttle when approaching speed limit
                # FIXED: Start prevention earlier (at 70% instead of 85%)
                longitudinal_cfg = self.control_config.get('longitudinal', {})
                throttle_limit_threshold = longitudinal_cfg.get('throttle_limit_threshold', 0.70)  # FIXED: was 0.8
                throttle_reduction_factor = longitudinal_cfg.get('throttle_reduction_factor', 0.2)  # FIXED: was 0.3
                
                # Start prevention earlier - at 70% of max speed (7.0 m/s for 10 m/s max)
                if current_speed > max_speed * prevention_threshold:
                    # Aggressively reduce throttle to prevent exceeding limit
                    # FIXED: More aggressive reduction
                    throttle_reduction = 1.0 - ((current_speed - max_speed * prevention_threshold) / (max_speed * (1.0 - prevention_threshold)))
                    throttle_reduction = max(0.0, min(1.0, throttle_reduction))
                    control_command['throttle'] = control_command.get('throttle', 0.0) * throttle_reduction * throttle_reduction_factor
                    
                    # If still accelerating, apply light brake (FIXED: start at 80% instead of 90%)
                    if current_speed > max_speed * prevention_brake_threshold:
                        control_command['throttle'] = 0.0
                        # FIXED: More aggressive brake amount (was 0.2, now 0.4)
                        control_command['brake'] = max(control_command.get('brake', 0.0), prevention_brake_amount)
                
                # Launch throttle ramp: cap throttle during stop -> start transitions
                launch_cap_active = False
                launch_cap = 0.0
                if not self.speed_planner_enabled:
                    launch_stop_threshold = longitudinal_cfg.get('launch_throttle_stop_threshold', 0.5)
                    launch_ramp_seconds = longitudinal_cfg.get('launch_throttle_ramp_seconds', 0.0)
                    launch_cap_min = longitudinal_cfg.get('launch_throttle_cap_min', 0.1)
                    launch_cap_max = longitudinal_cfg.get('launch_throttle_cap_max', 0.6)
                    launch_rearm_hysteresis = longitudinal_cfg.get('launch_throttle_rearm_hysteresis', 0.5)
                    launch_stop_hold_seconds = longitudinal_cfg.get('launch_throttle_stop_hold_seconds', 0.5)
                    launch_rearm_speed = float(launch_stop_threshold) + float(launch_rearm_hysteresis)
                    if launch_ramp_seconds > 0.0:
                        if current_speed > launch_rearm_speed:
                            self.launch_throttle_ramp_armed = True
                        if current_speed <= launch_stop_threshold:
                            if self.launch_stop_candidate_start_time is None:
                                self.launch_stop_candidate_start_time = float(timestamp)
                        else:
                            self.launch_stop_candidate_start_time = None
                        if self.launch_throttle_ramp_start_time is None and self.launch_throttle_ramp_armed:
                            if (
                                current_speed <= launch_stop_threshold
                                and control_command.get('throttle', 0.0) > 0.0
                                and self.launch_stop_candidate_start_time is not None
                                and (float(timestamp) - self.launch_stop_candidate_start_time) >= float(launch_stop_hold_seconds)
                            ):
                                self.launch_throttle_ramp_start_time = float(timestamp)
                                self.launch_throttle_ramp_armed = False
                        if self.launch_throttle_ramp_start_time is not None:
                            elapsed = max(0.0, float(timestamp) - self.launch_throttle_ramp_start_time)
                            if elapsed >= float(launch_ramp_seconds):
                                self.launch_throttle_ramp_start_time = None
                            else:
                                ratio = min(1.0, elapsed / float(launch_ramp_seconds))
                                launch_cap = float(launch_cap_min + (launch_cap_max - launch_cap_min) * ratio)
                                launch_cap_active = True
                                if control_command.get('throttle', 0.0) > 0.0:
                                    control_command['throttle'] = min(control_command['throttle'], launch_cap)
                control_command['launch_throttle_cap'] = launch_cap
                control_command['launch_throttle_cap_active'] = launch_cap_active

                # Additional safety: Ensure we don't exceed speed limit
                # CRITICAL FIX: Use proportional brake instead of hard 0.8 to prevent sudden stops
                if current_speed > max_speed:  # Over limit (shouldn't happen with prevention)
                    control_command['throttle'] = 0.0
                    # Use proportional brake (already set by brake_override above, but ensure minimum)
                    # Don't override if brake_override is already set (emergency brake case)
                    if brake_override == 0.0:
                        speed_excess = current_speed - max_speed
                        control_command['brake'] = max(control_command.get('brake', 0.0), min(0.8, 0.4 + (speed_excess / 2.0)))
                
                # NEW: Safety bounds check for lateral error
                # Critical: Prevent divergence and enable recovery
                # 89.4% steering direction errors and 8.95m max error indicate need for safety bounds
                # FIXED: lateral_error is now always in control_command (from vehicle_controller.py)
                lateral_error_abs = abs(control_command.get('lateral_error', 0.0))
                
                # CRITICAL: Emergency stop when car goes out of bounds
                # When perception fails and we keep using stale data, the car can drive off the road
                # Calculate threshold accounting for car width and allowed outside distance
                # Formula: threshold = (lane_width/2 - car_width/2 + allowed_outside_lane)
                # This means: car center can be this far from lane center before emergency stop
                # For 7m lane, 1.85m car, 1.0m allowed: threshold = 3.5 - 0.925 + 1.0 = 3.575m
                lane_width = (
                    self.last_gt_lane_width
                    if self.last_gt_lane_width is not None
                    else self.safety_config.get('lane_width', 7.0)
                )
                car_width = self.safety_config.get('car_width', 1.85)
                allowed_outside_lane = self.safety_config.get('allowed_outside_lane', 1.0)
                out_of_bounds_threshold = (lane_width / 2.0) - (car_width / 2.0) + allowed_outside_lane
                use_gt_lane_boundary_stop = bool(
                    self.safety_config.get('emergency_stop_use_gt_lane_boundaries', False)
                )
                gt_lane_boundary_margin = float(
                    self.safety_config.get('emergency_stop_gt_lane_boundary_margin', 0.05)
                )
                def _first_present(keys):
                    for key in keys:
                        if key in vehicle_state_dict and vehicle_state_dict.get(key) is not None:
                            return vehicle_state_dict.get(key)
                    return None

                gt_left_lane_line_x = _first_present(
                    [
                        'groundTruthLeftLaneLineX',
                        'ground_truth_left_lane_line_x',
                        'groundTruthLeftLaneX',
                        'ground_truth_left_lane_x',
                    ]
                )
                gt_right_lane_line_x = _first_present(
                    [
                        'groundTruthRightLaneLineX',
                        'ground_truth_right_lane_line_x',
                        'groundTruthRightLaneX',
                        'ground_truth_right_lane_x',
                    ]
                )
                gt_boundary_offroad_left = False
                gt_boundary_offroad_right = False
                gt_boundary_offroad = False
                if use_gt_lane_boundary_stop:
                    try:
                        if gt_left_lane_line_x is not None and gt_right_lane_line_x is not None:
                            gt_left_lane_line_x = float(gt_left_lane_line_x)
                            gt_right_lane_line_x = float(gt_right_lane_line_x)
                            gt_bounds_available = (
                                abs(gt_left_lane_line_x) > 1e-6
                                or abs(gt_right_lane_line_x) > 1e-6
                            )
                            if gt_bounds_available:
                                gt_boundary_offroad_left = (
                                    gt_left_lane_line_x > gt_lane_boundary_margin
                                )
                                gt_boundary_offroad_right = (
                                    gt_right_lane_line_x < -gt_lane_boundary_margin
                                )
                                gt_boundary_offroad = (
                                    gt_boundary_offroad_left or gt_boundary_offroad_right
                                )
                    except (TypeError, ValueError):
                        gt_boundary_offroad = False
                
                # Also check if perception is frozen (stopped updating)
                # Distinguish between Unity pause (time gap) vs perception failure (no time gap)
                perception_frozen = hasattr(self, 'perception_frozen_frames') and self.perception_frozen_frames > 5
                perception_failed = False
                
                if perception_frozen:
                    # Check if it's a Unity pause (time gap) or perception failure (no time gap)
                    if hasattr(self, 'last_timestamp') and hasattr(self, 'last_timestamp_before_freeze'):
                        if self.last_timestamp_before_freeze is not None:
                            dt_since_freeze = timestamp - self.last_timestamp_before_freeze
                            # If time gap is small (<0.1s per frame), Unity is still running but perception died
                            # If time gap is large (>0.5s per frame), Unity paused
                            avg_dt_per_frame = dt_since_freeze / self.perception_frozen_frames if self.perception_frozen_frames > 0 else 0
                            if avg_dt_per_frame < 0.1:
                                perception_failed = True  # Perception died (Unity still sending frames)
                
                # CRITICAL: Track if emergency stop was triggered to prevent overwriting
                emergency_stop_triggered = False
                emergency_stop_release_speed = float(
                    self.safety_config.get('emergency_stop_release_speed', 0.2)
                )
                
                # Check if emergency condition exists
                is_emergency_condition = (lateral_error_abs > emergency_stop_error or 
                                        lateral_error_abs > out_of_bounds_threshold or 
                                        perception_failed or
                                        gt_boundary_offroad)

                # Teleport/jump guard: skip emergency stop briefly after a position jump
                if (
                    not self.emergency_stop_latched
                    and (teleport_guard_active or self.post_jump_cooldown_frames > 0)
                    and is_emergency_condition
                ):
                    logger.warning(
                        f"[Frame {self.frame_count}] [TELEPORT GUARD] Emergency stop suppressed during "
                        f"post-jump cooldown (jump={self.last_teleport_distance:.2f}m, "
                        f"dt={self.last_teleport_dt if self.last_teleport_dt is not None else 'N/A'}s, "
                        f"cooldown_frames={self.post_jump_cooldown_frames})."
                    )
                    is_emergency_condition = False

                if self.emergency_stop_latched and current_speed <= emergency_stop_release_speed:
                    logger.info(
                        f"[Frame {self.frame_count}] Emergency stop latch released at "
                        f"{current_speed:.3f} m/s (threshold {emergency_stop_release_speed:.3f} m/s)"
                    )
                    self.emergency_stop_latched = False
                    self.emergency_stop_latched_since_wall_time = None
                
                # Reset logged flag if emergency condition cleared
                if not is_emergency_condition and not self.emergency_stop_latched:
                    if self.emergency_stop_logged:
                        logger.info(f"Emergency stop condition cleared. Lateral error: {lateral_error_abs:.3f}m")
                    self.emergency_stop_logged = False
                    self.emergency_stop_type = None

                if self.emergency_stop_latched and current_speed > emergency_stop_release_speed:
                    control_command = {
                        'steering': 0.0,
                        'throttle': 0.0,
                        'brake': 1.0,
                        'lateral_error': control_command.get('lateral_error', 0.0),
                        'emergency_stop': True,
                    }
                    emergency_stop_triggered = True
                elif lateral_error_abs > emergency_stop_error:
                    # Emergency stop: Error exceeds 3.0m - stop immediately
                    if not self.emergency_stop_logged or self.emergency_stop_type != 'lateral_error_exceeded':
                        logger.error(f"[Frame {self.frame_count}] EMERGENCY STOP: Lateral error {lateral_error_abs:.3f}m exceeds {emergency_stop_error}m threshold!")
                        self.emergency_stop_logged = True
                        self.emergency_stop_type = 'lateral_error_exceeded'
                    self.emergency_stop_latched = True
                    if self.emergency_stop_latched_since_wall_time is None:
                        self.emergency_stop_latched_since_wall_time = time.time()
                    control_command = {'steering': 0.0, 'throttle': 0.0, 'brake': 1.0, 'lateral_error': control_command.get('lateral_error', 0.0), 'emergency_stop': True}
                    emergency_stop_triggered = True
                    # Reset PID to prevent further divergence
                    self.controller.lateral_controller.reset()
                elif lateral_error_abs > out_of_bounds_threshold or perception_failed or gt_boundary_offroad:
                    # Out of bounds or perception failed - emergency stop
                    gt_offroad_type = None
                    if gt_boundary_offroad_left and gt_boundary_offroad_right:
                        gt_offroad_type = 'gt_both_offroad'
                    elif gt_boundary_offroad_left:
                        gt_offroad_type = 'gt_left_offroad'
                    elif gt_boundary_offroad_right:
                        gt_offroad_type = 'gt_right_offroad'
                    emergency_type = (
                        'perception_failed'
                        if perception_failed
                        else (gt_offroad_type if gt_offroad_type is not None else 'out_of_bounds')
                    )
                    if not self.emergency_stop_logged or self.emergency_stop_type != emergency_type:
                        if perception_failed:
                            logger.error(f"[Frame {self.frame_count}] EMERGENCY STOP: Perception FAILED! Frozen for {self.perception_frozen_frames} frames "
                                       f"(Unity still running, perception processing stopped)")
                            self.emergency_stop_type = 'perception_failed'
                        elif gt_offroad_type is not None:
                            logger.error(
                                f"[Frame {self.frame_count}] EMERGENCY STOP: GT lane-boundary offroad "
                                f"(left_x={gt_left_lane_line_x:.3f}m, right_x={gt_right_lane_line_x:.3f}m, "
                                f"margin={gt_lane_boundary_margin:.3f}m, type={gt_offroad_type})"
                            )
                            self.emergency_stop_type = gt_offroad_type
                        else:
                            logger.error(f"[Frame {self.frame_count}] EMERGENCY STOP: Car out of bounds! Lateral error {lateral_error_abs:.3f}m exceeds {out_of_bounds_threshold}m threshold!")
                            self.emergency_stop_type = 'out_of_bounds'
                        logger.error(f"[Frame {self.frame_count}]   Stopping vehicle to prevent further off-road driving")
                        self.emergency_stop_logged = True
                    self.emergency_stop_latched = True
                    if self.emergency_stop_latched_since_wall_time is None:
                        self.emergency_stop_latched_since_wall_time = time.time()
                    control_command = {'steering': 0.0, 'throttle': 0.0, 'brake': 1.0, 'lateral_error': control_command.get('lateral_error', 0.0), 'emergency_stop': True}
                    emergency_stop_triggered = True
                    # Reset PID to prevent further divergence
                    self.controller.lateral_controller.reset()
                
                # CRITICAL: If emergency stop was triggered, skip all other control modifications
                if not emergency_stop_triggered:
                    if lateral_error_abs > max_lateral_error:
                        # Hard limit: Error exceeds 2.0m - aggressive correction
                        logger.warning(f"[Frame {self.frame_count}] HARD LIMIT: Lateral error {lateral_error_abs:.3f}m exceeds {max_lateral_error}m - aggressive correction")
                        # Increase steering gain temporarily for recovery
                        # CRITICAL FIX: Use config max_steering instead of hardcoded 0.5
                        max_steering = self.controller.lateral_controller.max_steering
                        steering_original = control_command.get('steering', 0.0)
                        control_command['steering'] = np.clip(steering_original * 1.5, -max_steering, max_steering)  # 50% more aggressive
                        # Reduce throttle to slow down during recovery
                        control_command['throttle'] = control_command.get('throttle', 0.0) * 0.5
                    elif lateral_error_abs > recovery_lateral_error:
                        # Recovery mode: Error exceeds 1.0m - initiate recovery
                        logger.info(f"RECOVERY MODE: Lateral error {lateral_error_abs:.3f}m exceeds {recovery_lateral_error}m - initiating recovery")
                        # Slightly increase steering for recovery
                        # CRITICAL FIX: Use config max_steering instead of hardcoded 0.5
                        max_steering = self.controller.lateral_controller.max_steering
                        steering_original = control_command.get('steering', 0.0)
                        control_command['steering'] = np.clip(steering_original * 1.2, -max_steering, max_steering)  # 20% more aggressive
                        # Reduce throttle slightly
                        control_command['throttle'] = control_command.get('throttle', 0.0) * 0.8
        
        # 4. Send control command to Unity
        # CRITICAL: Don't send control if ground truth follower is active
        # Ground truth follower will send its own control command
        if not (hasattr(self, '_gt_control_override') and self._gt_control_override is not None):
            # Only send control if NOT in ground truth mode
            self.bridge.set_control_command(
                steering=control_command['steering'],
                throttle=control_command['throttle'],
                brake=control_command['brake']
            )
        else:
            # Ground truth mode active - ground truth follower will send control
            logger.debug("Skipping AV stack control send (ground truth mode active)")
        self.last_steering_command = float(control_command.get('steering', 0.0))
        
        # 4b. Send trajectory data for visualization (if available)
        if reference_point is not None and trajectory and trajectory.points:
            # Convert trajectory points to list format for JSON
            trajectory_points_list = []
            # Send more points for better visualization (Unity will limit if needed)
            for point in trajectory.points[:30]:  # Limit to first 30 points for performance
                trajectory_points_list.append([float(point.x), float(point.y), float(point.heading)])
            
            # Reference point
            ref_point_list = [
                float(reference_point.get('x', 0.0)),
                float(reference_point.get('y', 0.0)),
                float(reference_point.get('heading', 0.0)),
                float(reference_point.get('velocity', 8.0))
            ]
            
            # Use lateral error from controller (already calculated correctly)
            # This ensures Unity sees the same error value the controller uses
            # FIXED: Previously recalculated lateral error here, but this gave different
            # values (mean 0.1979m vs controller's 0.0218m), causing trajectory line to be yellow
            lateral_error = control_command.get('lateral_error', 0.0)
            
            # Ensure lateral_error is a valid float (handle None or invalid values)
            if lateral_error is None or not isinstance(lateral_error, (int, float)):
                lateral_error = 0.0
            lateral_error = float(lateral_error)
            
            # Debug: Log lateral error periodically to verify it's being sent correctly
            if self.frame_count % 30 == 0:  # Every ~1 second
                logger.debug(f"Trajectory visualization: lateral_error={lateral_error:.4f}m (abs={abs(lateral_error):.4f}m)")
            
            perception_left = left_lane_line_x if left_lane_line_x is not None else None
            perception_right = right_lane_line_x if right_lane_line_x is not None else None
            perception_valid = perception_left is not None and perception_right is not None
            perception_center = None
            if perception_valid:
                perception_center = (float(perception_left) + float(perception_right)) / 2.0

            # Send to bridge for Unity visualization
            self.bridge.set_trajectory_data(
                trajectory_points=trajectory_points_list,
                reference_point=ref_point_list,
                lateral_error=lateral_error,
                perception_left_lane_x=perception_left,
                perception_right_lane_x=perception_right,
                perception_center_x=perception_center,
                perception_lookahead_m=reference_lookahead,
                perception_valid=perception_valid,
            )
        
        # 5. Record data if enabled
        if self.recorder:
            topdown_frame_meta: dict | None = None
            if topdown_frame_data is None:
                try:
                    td_data = self.bridge.get_latest_camera_frame_with_metadata(
                        camera_id="top_down"
                    )
                    if td_data is not None:
                        td_image, td_ts, td_id, td_meta = td_data
                        topdown_frame_data = (td_image, td_ts, td_id)
                        topdown_frame_meta = td_meta
                    else:
                        topdown_frame_data = None
                except Exception:
                    topdown_frame_data = None
            self._record_frame(
                image,
                timestamp,
                vehicle_state_dict,
                perception_output,
                trajectory,
                control_command,
                speed_limit,
                runtime_reference_point=reference_point,
                trajectory_source=trajectory_source_active,
                camera_frame_id=camera_frame_id,
                camera_frame_meta=camera_frame_meta,
                topdown_frame_data=topdown_frame_data,
                topdown_frame_meta=topdown_frame_meta,
                speed_horizon_guardrail_diag=speed_horizon_guardrail_diag,
            )

        # Update teleport guard countdown and vehicle position tracking
        if self.teleport_guard_frames_remaining > 0:
            self.teleport_guard_frames_remaining -= 1
        if self.post_jump_cooldown_frames > 0:
            self.post_jump_cooldown_frames -= 1
        self.last_vehicle_position = current_position
        self.last_vehicle_timestamp = timestamp

        process_duration = time.time() - process_start
        if process_duration > 0.1:
            unity_frame = vehicle_state_dict.get('unityFrameCount') or vehicle_state_dict.get('unity_frame_count')
            unity_time = vehicle_state_dict.get('unityTime') or vehicle_state_dict.get('unity_time')
            logger.warning(
                "[FRAME_PROCESS_SLOW] duration=%.3fs perception=%.3fs frame=%s unity_frame=%s unity_time=%s",
                process_duration,
                perception_duration,
                self.frame_count,
                unity_frame if unity_frame is not None else "n/a",
                f"{unity_time:.3f}" if unity_time is not None else "n/a",
            )
        
        # Log status periodically (every 30 frames = ~1 second at 30 FPS)
        if self.frame_count % 30 == 0:
            heading = self._extract_heading(vehicle_state_dict)
            num_lanes = sum(1 for c in lane_coeffs if c is not None)
            speed = vehicle_state_dict.get('speed', 0)
            
            # Get reference point for debugging (if available)
            ref_info = ""
            if reference_point is not None:
                ref_x = reference_point.get('x', 0)
                ref_y = reference_point.get('y', 0)
                ref_heading = reference_point.get('heading', 0)
                
                # RUNTIME VALIDATION: Check reference point coordinate system
                if abs(ref_x) > 100.0:
                    logger.error(
                        f"⚠️  CRITICAL: ref_x looks like PIXELS ({ref_x:.1f}), not METERS! "
                        f"Expected < 10m, got {ref_x:.1f}. Coordinate conversion may not be applied!"
                    )
                if abs(ref_heading) > np.pi / 2:  # > 90°
                    logger.warning(
                        f"⚠️  ref_heading is extreme: {np.degrees(ref_heading):.1f}° "
                        f"(expected ~0° for straight road). Coordinate system may be wrong!"
                    )
                
                # Also log lateral error for debugging
                vehicle_pos = self._extract_position(vehicle_state_dict)
                dx = ref_x - vehicle_pos[0]
                dy = ref_y - vehicle_pos[1]
                current_heading = self._extract_heading(vehicle_state_dict)
                cos_h = np.cos(-current_heading)
                sin_h = np.sin(-current_heading)
                lateral_error = dx * cos_h - dy * sin_h  # X component in vehicle frame
                
                # RUNTIME VALIDATION: Check lateral error is reasonable
                if abs(lateral_error) > 10.0:
                    logger.warning(
                        f"⚠️  Lateral error is very large: {lateral_error:.3f}m "
                        f"(expected < 5m). This may indicate coordinate system issue!"
                    )
                
                # VALIDATION: Check steering direction (warn if wrong)
                # Steering should be OPPOSITE sign of lateral_error
                # If lateral_error > 0 (car LEFT of ref), steering should be < 0 (steer LEFT)
                # If lateral_error < 0 (car RIGHT of ref), steering should be > 0 (steer RIGHT)
                if abs(lateral_error) > 0.1 and abs(control_command.get('steering', 0)) > 0.05:
                    steering = control_command.get('steering', 0)
                    # Check if steering and error have same sign (wrong direction)
                    if (lateral_error > 0 and steering > 0) or (lateral_error < 0 and steering < 0):
                        logger.warning(
                            f"⚠️  Steering direction warning: "
                            f"lateral_error={lateral_error:.3f}m, steering={steering:.3f} "
                            f"(same sign - may be wrong direction!)"
                        )
                ref_info = f", Ref=({ref_x:.2f}, {ref_y:.2f}, {np.degrees(ref_heading):.1f}°), LatErr={lateral_error:.3f}"
            
            logger.info(
                f"Frame {self.frame_count}: "
                f"Lanes={num_lanes}, "
                f"Conf={confidence:.2f}, "
                f"Speed={speed:.2f}m/s, "
                f"Heading={np.degrees(heading):.1f}°"
                f"{ref_info}, "
                f"Steering={control_command['steering']:.3f}, "
                f"Throttle={control_command['throttle']:.3f}, "
                f"Brake={control_command['brake']:.3f}"
            )
        
        # Log every frame at DEBUG level (for detailed analysis)
        logger.debug(
            f"Frame {self.frame_count}: "
            f"Lanes={sum(1 for c in lane_coeffs if c is not None)}, "
            f"Conf={confidence:.3f}, "
            f"Speed={vehicle_state_dict.get('speed', 0):.3f}, "
            f"Steering={control_command['steering']:.3f}"
        )
    
    def _extract_heading(self, vehicle_state: dict) -> float:
        """Extract heading from vehicle state (quaternion to yaw angle in radians).
        
        Unity uses left-handed coordinate system with Y-up.
        Yaw is rotation around Y-axis (vertical axis).
        """
        rotation = vehicle_state.get('rotation', {})
        if isinstance(rotation, dict):
            # Unity sends quaternion as {x, y, z, w}
            x = rotation.get('x', 0.0)
            y = rotation.get('y', 0.0)
            z = rotation.get('z', 0.0)
            w = rotation.get('w', 1.0)
            
            # Convert quaternion to Euler angles
            # For Unity: Roll (X), Pitch (Z), Yaw (Y)
            # Yaw around Y-axis: atan2(2*(w*y + x*z), 1 - 2*(y*y + z*z))
            yaw = np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z))
            
            return yaw
        return 0.0
    
    def _extract_position(self, vehicle_state: dict) -> np.ndarray:
        """Extract position from vehicle state."""
        position = vehicle_state.get('position', {})
        if isinstance(position, dict):
            return np.array([position.get('x', 0.0), position.get('z', 0.0)])
        return np.array([0.0, 0.0])

    def _smooth_path_curvature(self, raw_curvature: float, current_speed: float, timestamp: float) -> float:
        """Smooth curvature using a distance-based EMA window."""
        if not self.curvature_smoothing_enabled:
            return float(raw_curvature)
        if not isinstance(raw_curvature, (int, float)) or math.isnan(float(raw_curvature)):
            return 0.0
        if self.smoothed_path_curvature is None or self.last_curvature_time is None:
            self.smoothed_path_curvature = float(raw_curvature)
            self.last_curvature_time = float(timestamp)
            return self.smoothed_path_curvature
        dt = float(timestamp) - float(self.last_curvature_time)
        if dt <= 0.0:
            dt = 1e-3
        min_speed = max(0.0, float(self.curvature_smoothing_min_speed))
        distance = max(float(current_speed), min_speed) * dt
        alpha = curvature_smoothing_alpha(distance, float(self.curvature_smoothing_window_m))
        self.smoothed_path_curvature = (
            alpha * float(raw_curvature)
            + (1.0 - alpha) * float(self.smoothed_path_curvature)
        )
        self.last_curvature_time = float(timestamp)
        return float(self.smoothed_path_curvature)

    @staticmethod
    def _apply_speed_limits(
        base_speed: float,
        speed_limit: float,
        path_curvature: float,
        max_lateral_accel: float,
        min_curve_speed: float = 0.0,
        current_speed: float | None = None,
        min_speed_limit_ratio: float = 0.0,
        min_speed_floor: float = 0.0,
        min_speed_curvature_max: float = 0.0,
        curvature_limit_min_abs: float = 0.0,
    ) -> float:
        """Clamp speed using map limits and curvature-based lateral acceleration."""
        adjusted_speed = base_speed

        if isinstance(speed_limit, (int, float)) and speed_limit > 0:
            adjusted_speed = min(adjusted_speed, float(speed_limit))

        if (
            isinstance(path_curvature, (int, float))
            and abs(path_curvature) > max(curvature_limit_min_abs, 1e-6)
        ):
            if isinstance(max_lateral_accel, (int, float)) and max_lateral_accel > 0:
                curve_speed = (max_lateral_accel / abs(path_curvature)) ** 0.5
                if min_curve_speed > 0:
                    curve_speed = max(curve_speed, min_curve_speed)
                adjusted_speed = min(adjusted_speed, curve_speed)
                if current_speed is not None and current_speed > 0.0:
                    current_lat_accel = (current_speed ** 2) * abs(path_curvature)
                    if current_lat_accel > max_lateral_accel and adjusted_speed < current_speed:
                        logger.debug(
                            "Lateral accel cap active: "
                            f"lat_accel={current_lat_accel:.2f} m/s^2 "
                            f"cap={max_lateral_accel:.2f} m/s^2 "
                            f"curve_speed={curve_speed:.2f} m/s"
                        )

        if isinstance(speed_limit, (int, float)) and speed_limit > 0.0:
            if min_speed_limit_ratio > 0.0 or min_speed_floor > 0.0:
                allow_floor = True
                if isinstance(path_curvature, (int, float)) and min_speed_curvature_max > 0.0:
                    allow_floor = abs(path_curvature) <= min_speed_curvature_max
                if allow_floor:
                    dynamic_floor = max(
                        float(min_speed_floor),
                        float(speed_limit) * float(min_speed_limit_ratio),
                    )
                    adjusted_speed = max(adjusted_speed, dynamic_floor)

        return float(adjusted_speed)

    def _compute_dynamic_speed_floor(
        self, speed_limit: float, path_curvature: float
    ) -> float | None:
        ratio = float(self.trajectory_config.get('min_speed_limit_ratio', 0.0))
        floor = float(self.trajectory_config.get('min_speed_floor', 0.0))
        max_curv = float(self.trajectory_config.get('min_speed_curvature_max', 0.0))
        if ratio <= 0.0 and floor <= 0.0:
            return None
        if not isinstance(speed_limit, (int, float)) or float(speed_limit) <= 0.0:
            return None
        if isinstance(path_curvature, (int, float)) and max_curv > 0.0:
            if abs(float(path_curvature)) > max_curv:
                return None
        return max(floor, float(speed_limit) * ratio)

    def _smooth_speed_limit(self, raw_limit: float, timestamp: float) -> float:
        """Smooth speed limit changes to avoid step inputs."""
        if not isinstance(raw_limit, (int, float)) or raw_limit <= 0.0:
            self.last_speed_limit = None
            self.last_speed_limit_time = None
            return 0.0

        slew_rate = self.trajectory_config.get('speed_limit_slew_rate', 0.0)
        slew_rate_up = self.trajectory_config.get('speed_limit_slew_rate_up', slew_rate)
        slew_rate_down = self.trajectory_config.get('speed_limit_slew_rate_down', slew_rate)
        if (slew_rate <= 0.0 and slew_rate_up <= 0.0 and slew_rate_down <= 0.0) or \
           self.last_speed_limit is None or self.last_speed_limit_time is None:
            self.last_speed_limit = float(raw_limit)
            self.last_speed_limit_time = float(timestamp)
            return float(raw_limit)

        dt = max(1e-3, float(timestamp) - float(self.last_speed_limit_time))
        if float(raw_limit) >= float(self.last_speed_limit):
            rate = float(slew_rate_up)
        else:
            rate = float(slew_rate_down)
        if rate <= 0.0:
            smoothed = float(raw_limit)
        else:
            smoothed = _slew_limit_value(float(self.last_speed_limit), float(raw_limit), rate, dt)
        self.last_speed_limit = smoothed
        self.last_speed_limit_time = float(timestamp)
        return smoothed

    def _apply_target_speed_ramp(self, desired_speed: float, current_speed: float,
                                 timestamp: float) -> Tuple[float, bool, bool]:
        """Smooth target speed changes and ramp after stops."""
        if self.last_target_speed is None or self.last_target_speed_time is None:
            self.last_target_speed = float(desired_speed)
            self.last_target_speed_time = float(timestamp)
            return float(desired_speed), False, False

        dt = max(1e-3, float(timestamp) - float(self.last_target_speed_time))
        longitudinal_cfg = self.control_config.get('longitudinal', {})
        rate_up = float(longitudinal_cfg.get('target_speed_slew_rate_up', 0.0))
        rate_down = float(longitudinal_cfg.get('target_speed_slew_rate_down', 0.0))
        slew_error_min = float(longitudinal_cfg.get('target_speed_slew_error_min', 0.5))
        slew_error_max = float(longitudinal_cfg.get('target_speed_slew_error_max', 3.0))
        rate_up_min = float(longitudinal_cfg.get('target_speed_slew_rate_up_min', rate_up))
        rate_up_max = float(longitudinal_cfg.get('target_speed_slew_rate_up_max', rate_up))
        rate_down_min = float(longitudinal_cfg.get('target_speed_slew_rate_down_min', rate_down))
        rate_down_max = float(longitudinal_cfg.get('target_speed_slew_rate_down_max', rate_down))
        stop_threshold = float(longitudinal_cfg.get('target_speed_stop_threshold', 0.5))
        ramp_seconds = float(longitudinal_cfg.get('target_speed_restart_ramp_seconds', 0.0))

        ramped_speed, self.restart_ramp_start_time, ramp_active = _apply_restart_ramp(
            desired_speed,
            current_speed,
            self.restart_ramp_start_time,
            timestamp,
            ramp_seconds,
            stop_threshold,
        )
        speed_error = abs(float(desired_speed) - float(current_speed))
        if slew_error_max > slew_error_min:
            ratio = np.clip(
                (speed_error - slew_error_min) / (slew_error_max - slew_error_min),
                0.0,
                1.0,
            )
        else:
            ratio = 1.0 if speed_error > 0.0 else 0.0
        dynamic_rate_up = rate_up_min + ratio * (rate_up_max - rate_up_min)
        dynamic_rate_down = rate_down_min + ratio * (rate_down_max - rate_down_min)

        adjusted = _apply_target_speed_slew(
            float(self.last_target_speed),
            float(ramped_speed),
            dynamic_rate_up,
            dynamic_rate_down,
            dt,
        )
        slew_active = abs(adjusted - float(ramped_speed)) > 1e-6
        self.last_target_speed = adjusted
        self.last_target_speed_time = float(timestamp)
        return adjusted, ramp_active, slew_active

    def _extract_oracle_points_xy(self, vehicle_state_dict: dict) -> Optional[np.ndarray]:
        """Extract oracle points from Unity payload as [N, 2] array."""
        oracle_raw = vehicle_state_dict.get(
            "oracleTrajectoryXY", vehicle_state_dict.get("oracle_trajectory_xy")
        )
        if oracle_raw is None:
            return None
        try:
            arr = np.asarray(oracle_raw, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return None
        if arr.size < 4 or arr.size % 2 != 0:
            return None
        arr2 = arr.reshape(-1, 2)
        finite = np.isfinite(arr2[:, 0]) & np.isfinite(arr2[:, 1])
        arr2 = arr2[finite]
        return arr2 if arr2.shape[0] >= 2 else None

    def _build_oracle_reference_point(
        self, oracle_points_xy: np.ndarray, lookahead_m: float, target_speed: float
    ) -> Optional[dict]:
        """Build control reference point from oracle trajectory samples."""
        if oracle_points_xy is None or oracle_points_xy.shape[0] < 2:
            return None
        x = oracle_points_xy[:, 0].astype(float)
        y = oracle_points_xy[:, 1].astype(float)
        valid = np.isfinite(x) & np.isfinite(y) & (y >= 0.0)
        if np.sum(valid) < 2:
            return None
        x = x[valid]
        y = y[valid]
        idx = int(np.argmin(np.abs(y - float(lookahead_m))))
        i0 = max(0, idx - 1)
        i1 = min(len(x) - 1, idx + 1)
        if i1 == i0:
            heading = 0.0
        else:
            dx = float(x[i1] - x[i0])
            dy = float(y[i1] - y[i0])
            heading = float(np.arctan2(dx, dy)) if abs(dy) > 1e-6 else 0.0
        return {
            "x": float(x[idx]),
            "y": float(y[idx]),
            "heading": heading,
            "velocity": float(target_speed),
            "curvature": 0.0,
            "method": "oracle",
            "oracle_ref_index": idx,
        }
    
    def _record_frame(
        self,
        image: np.ndarray,
        timestamp: float,
        vehicle_state_dict: dict,
        perception_output: PerceptionOutput,
        trajectory,
        control_command: dict,
        speed_limit: float = 0.0,
        runtime_reference_point: Optional[dict] = None,
        trajectory_source: str = "planner",
        camera_frame_id: int | None = None,
        camera_frame_meta: dict | None = None,
        topdown_frame_data: tuple[np.ndarray, float, int | None] | None = None,
        topdown_frame_meta: dict | None = None,
        speed_horizon_guardrail_diag: Optional[dict] = None,
    ):
        """Record frame data."""
        # Create camera frame
        camera_frame = CameraFrame(
            image=image,
            timestamp=timestamp,
            frame_id=camera_frame_id if camera_frame_id is not None else self.frame_count
        )

        camera_topdown_frame = None
        if topdown_frame_data is not None:
            topdown_image, topdown_timestamp, topdown_frame_id = topdown_frame_data
            camera_topdown_frame = CameraFrame(
                image=topdown_image,
                timestamp=topdown_timestamp,
                frame_id=(
                    topdown_frame_id
                    if topdown_frame_id is not None
                    else camera_frame.frame_id
                ),
                camera_id="top_down",
            )

        # Stream consume-point lag diagnostics (instrumentation-only).
        unity_time_value = vehicle_state_dict.get(
            'unityTime', vehicle_state_dict.get('unity_time')
        )
        stream_front_unity_dt_ms = 0.0
        stream_topdown_unity_dt_ms = 0.0
        stream_topdown_front_dt_ms = 0.0
        stream_front_source_timestamp = float(timestamp)
        stream_topdown_source_timestamp = (
            float(camera_topdown_frame.timestamp) if camera_topdown_frame is not None else 0.0
        )
        if unity_time_value is not None:
            try:
                stream_front_unity_dt_ms = float(timestamp - float(unity_time_value)) * 1000.0
            except (TypeError, ValueError):
                stream_front_unity_dt_ms = 0.0
            if camera_topdown_frame is not None:
                try:
                    stream_topdown_unity_dt_ms = (
                        float(camera_topdown_frame.timestamp - float(unity_time_value)) * 1000.0
                    )
                except (TypeError, ValueError):
                    stream_topdown_unity_dt_ms = 0.0
        if camera_topdown_frame is not None:
            stream_topdown_front_dt_ms = float(camera_topdown_frame.timestamp - timestamp) * 1000.0

        last_front_id = getattr(self, "_last_recorded_front_frame_id", None)
        last_topdown_id = getattr(self, "_last_recorded_topdown_frame_id", None)
        last_front_ts = getattr(self, "_last_recorded_front_timestamp", None)
        last_topdown_ts = getattr(self, "_last_recorded_topdown_timestamp", None)
        stream_front_frame_id_delta = 0.0
        stream_topdown_frame_id_delta = 0.0
        stream_topdown_front_frame_id_delta = 0.0
        stream_front_timestamp_reused = 0.0
        stream_topdown_timestamp_reused = 0.0
        stream_front_timestamp_non_monotonic = 0.0
        stream_topdown_timestamp_non_monotonic = 0.0
        stream_front_negative_frame_delta = 0.0
        stream_topdown_negative_frame_delta = 0.0
        stream_front_frame_id_reused = 0.0
        stream_topdown_frame_id_reused = 0.0
        stream_front_clock_jump = 0.0
        stream_topdown_clock_jump = 0.0
        if camera_frame_id is not None and last_front_id is not None:
            stream_front_frame_id_delta = float(camera_frame_id - last_front_id)
            if stream_front_frame_id_delta < 0.0:
                stream_front_negative_frame_delta = 1.0
            if abs(stream_front_frame_id_delta) < 1e-6:
                stream_front_frame_id_reused = 1.0
        if camera_topdown_frame is not None:
            topdown_id = camera_topdown_frame.frame_id
            if topdown_id is not None and last_topdown_id is not None:
                stream_topdown_frame_id_delta = float(topdown_id - last_topdown_id)
                if stream_topdown_frame_id_delta < 0.0:
                    stream_topdown_negative_frame_delta = 1.0
                if abs(stream_topdown_frame_id_delta) < 1e-6:
                    stream_topdown_frame_id_reused = 1.0
            if camera_frame_id is not None and topdown_id is not None:
                stream_topdown_front_frame_id_delta = float(topdown_id - camera_frame_id)
            self._last_recorded_topdown_frame_id = topdown_id
        if camera_frame_id is not None:
            self._last_recorded_front_frame_id = camera_frame_id

        if last_front_ts is not None:
            dt_front = float(timestamp) - float(last_front_ts)
            if abs(dt_front) < 1e-9:
                stream_front_timestamp_reused = 1.0
            if dt_front < 0.0:
                stream_front_timestamp_non_monotonic = 1.0
            if dt_front > 0.2:
                stream_front_clock_jump = 1.0
        self._last_recorded_front_timestamp = float(timestamp)
        if camera_topdown_frame is not None:
            topdown_ts = float(camera_topdown_frame.timestamp)
            if last_topdown_ts is not None:
                dt_topdown = topdown_ts - float(last_topdown_ts)
                if abs(dt_topdown) < 1e-9:
                    stream_topdown_timestamp_reused = 1.0
                if dt_topdown < 0.0:
                    stream_topdown_timestamp_non_monotonic = 1.0
                if dt_topdown > 0.2:
                    stream_topdown_clock_jump = 1.0
            self._last_recorded_topdown_timestamp = topdown_ts

        # Bridge freshness/queue diagnostics at consume point.
        front_latest_age_ms = 0.0
        front_queue_depth = 0.0
        front_drop_count = 0.0
        front_decode_in_flight = 0.0
        topdown_latest_age_ms = 0.0
        topdown_queue_depth = 0.0
        topdown_drop_count = 0.0
        topdown_decode_in_flight = 0.0
        front_last_realtime_s = 0.0
        topdown_last_realtime_s = 0.0
        stream_front_timestamp_minus_realtime_ms = 0.0
        stream_topdown_timestamp_minus_realtime_ms = 0.0
        if camera_frame_meta is not None:
            try:
                front_latest_age_ms = float(camera_frame_meta.get("latest_age_ms") or 0.0)
                front_queue_depth = float(camera_frame_meta.get("queue_depth") or 0.0)
                front_drop_count = float(camera_frame_meta.get("drop_count") or 0.0)
                front_decode_in_flight = 1.0 if camera_frame_meta.get("decode_in_flight") else 0.0
                front_last_realtime_s = float(
                    camera_frame_meta.get("last_realtime_since_startup") or 0.0
                )
                if front_last_realtime_s > 0.0:
                    stream_front_timestamp_minus_realtime_ms = (
                        float(timestamp) - front_last_realtime_s
                    ) * 1000.0
            except (TypeError, ValueError):
                pass
        if topdown_frame_meta is not None:
            try:
                topdown_latest_age_ms = float(topdown_frame_meta.get("latest_age_ms") or 0.0)
                topdown_queue_depth = float(topdown_frame_meta.get("queue_depth") or 0.0)
                topdown_drop_count = float(topdown_frame_meta.get("drop_count") or 0.0)
                topdown_decode_in_flight = 1.0 if topdown_frame_meta.get("decode_in_flight") else 0.0
                topdown_last_realtime_s = float(
                    topdown_frame_meta.get("last_realtime_since_startup") or 0.0
                )
                if topdown_last_realtime_s > 0.0 and camera_topdown_frame is not None:
                    stream_topdown_timestamp_minus_realtime_ms = (
                        float(camera_topdown_frame.timestamp) - topdown_last_realtime_s
                    ) * 1000.0
            except (TypeError, ValueError):
                pass
        
        # Create vehicle state
        position = self._extract_position(vehicle_state_dict)
        # Extract rotation quaternion properly
        rotation_dict = vehicle_state_dict.get('rotation', {})
        if isinstance(rotation_dict, dict):
            rotation = np.array([
                rotation_dict.get('x', 0.0),
                rotation_dict.get('y', 0.0),
                rotation_dict.get('z', 0.0),
                rotation_dict.get('w', 1.0)
            ])
        else:
            rotation = np.array([0, 0, 0, 1])
        
        # Extract ground truth lane positions if available
        # DEBUG: Log what we're receiving to verify Unity is sending data
        if self.frame_count % 30 == 0:  # Log every 30 frames
            gt_keys = [k for k in vehicle_state_dict.keys() if 'ground' in k.lower() or 'truth' in k.lower()]
            if gt_keys:
                logger.info(f"[GT DEBUG] Ground truth keys found: {gt_keys}")
                for key in gt_keys:
                    logger.info(f"[GT DEBUG]   {key} = {vehicle_state_dict.get(key, 'NOT FOUND')}")
            else:
                logger.warning(f"[GT DEBUG] No ground truth keys found in vehicle_state_dict. "
                             f"Available keys: {list(vehicle_state_dict.keys())[:10]}")
        
        # Try both camelCase and snake_case field names
        # FIXED: Check for ground truth data (try both camelCase and snake_case)
        gt_left_lane_line_x = vehicle_state_dict.get('groundTruthLeftLaneLineX') or vehicle_state_dict.get('ground_truth_left_lane_line_x') or vehicle_state_dict.get('groundTruthLeftLaneX') or vehicle_state_dict.get('ground_truth_left_lane_x', 0.0)  # Backward compatibility
        gt_right_lane_line_x = vehicle_state_dict.get('groundTruthRightLaneLineX') or vehicle_state_dict.get('ground_truth_right_lane_line_x') or vehicle_state_dict.get('groundTruthRightLaneX') or vehicle_state_dict.get('ground_truth_right_lane_x', 0.0)  # Backward compatibility
        gt_lane_center_x = vehicle_state_dict.get('groundTruthLaneCenterX') or vehicle_state_dict.get('ground_truth_lane_center_x', 0.0)
        gt_path_curvature = vehicle_state_dict.get('groundTruthPathCurvature') or vehicle_state_dict.get('ground_truth_path_curvature', 0.0)
        gt_desired_heading = vehicle_state_dict.get('groundTruthDesiredHeading') or vehicle_state_dict.get('ground_truth_desired_heading', 0.0)

        # Cache ground-truth lane width for safety bounds (track-accurate width).
        if gt_left_lane_line_x or gt_right_lane_line_x:
            gt_lane_width = float(gt_right_lane_line_x - gt_left_lane_line_x)
            if 1.0 <= gt_lane_width <= 20.0:
                self.last_gt_lane_width = gt_lane_width
        
        # Log if we got ground truth data (or if it's missing)
        if self.frame_count % 30 == 0:  # Log every 30 frames
            if gt_left_lane_line_x != 0.0 or gt_right_lane_line_x != 0.0:
                logger.info(f"[GT DEBUG] Ground truth received: left={gt_left_lane_line_x:.3f}m, "
                           f"right={gt_right_lane_line_x:.3f}m, center={gt_lane_center_x:.3f}m")
            else:
                logger.warning(f"[GT DEBUG] Ground truth is 0.0 - check Unity: "
                             f"Is GroundTruthReporter enabled? Is it attached to the car? "
                             f"Vehicle state keys: {list(vehicle_state_dict.keys())}")
        
        # NEW: Extract camera calibration data (try both camelCase and snake_case)
        # CRITICAL: Use 'is not None' check instead of 'or' to handle 0.0 values correctly
        # 'or' treats 0.0 as falsy, which would use the default -1.0 even if Unity sent 0.0
        camera_8m_screen_y = vehicle_state_dict.get('camera8mScreenY')
        if camera_8m_screen_y is None:
            camera_8m_screen_y = vehicle_state_dict.get('camera_8m_screen_y', -1.0)
        camera_lookahead_screen_y = vehicle_state_dict.get('cameraLookaheadScreenY')
        if camera_lookahead_screen_y is None:
            camera_lookahead_screen_y = vehicle_state_dict.get('camera_lookahead_screen_y', -1.0)
        ground_truth_lookahead_distance = vehicle_state_dict.get('groundTruthLookaheadDistance')
        if ground_truth_lookahead_distance is None:
            ground_truth_lookahead_distance = vehicle_state_dict.get('ground_truth_lookahead_distance', 8.0)
        
        # NEW: Extract camera FOV data (what Unity actually uses)
        camera_field_of_view = vehicle_state_dict.get('cameraFieldOfView')
        if camera_field_of_view is None:
            camera_field_of_view = vehicle_state_dict.get('camera_field_of_view', 0.0)
        
        camera_horizontal_fov = vehicle_state_dict.get('cameraHorizontalFOV')
        if camera_horizontal_fov is None:
            camera_horizontal_fov = vehicle_state_dict.get('camera_horizontal_fov', 0.0)
        
        # Debug: Log camera calibration data periodically
        if self.frame_count % 60 == 0:  # Every 60 frames
            if camera_8m_screen_y > 0:
                logger.info(f"[CAMERA CALIB] camera_8m_screen_y = {camera_8m_screen_y:.1f}px (valid, from Unity)")
            else:
                logger.warning(f"[CAMERA CALIB] camera_8m_screen_y = {camera_8m_screen_y:.1f} (not calculated - camera not available or old recording)")
            
            # Log FOV information (critical for understanding Unity's behavior)
            if camera_field_of_view > 0:
                logger.info(f"[CAMERA FOV] Unity fieldOfView = {camera_field_of_view:.2f}° (vertical), "
                           f"calculated horizontal = {camera_horizontal_fov:.2f}°")
            
            # NEW: Log camera position and forward for debugging alignment
            camera_pos_x = vehicle_state_dict.get('cameraPosX', 0.0)
            camera_pos_y = vehicle_state_dict.get('cameraPosY', 0.0)
            camera_pos_z = vehicle_state_dict.get('cameraPosZ', 0.0)
            camera_forward_x = vehicle_state_dict.get('cameraForwardX', 0.0)
            camera_forward_y = vehicle_state_dict.get('cameraForwardY', 0.0)
            camera_forward_z = vehicle_state_dict.get('cameraForwardZ', 0.0)
            if camera_pos_x != 0.0 or camera_pos_y != 0.0 or camera_pos_z != 0.0:
                logger.info(f"[CAMERA POS] position=({camera_pos_x:.3f}, {camera_pos_y:.3f}, {camera_pos_z:.3f}), "
                           f"forward=({camera_forward_x:.3f}, {camera_forward_y:.3f}, {camera_forward_z:.3f})")
                logger.info(f"[CAMERA FOV] Config camera_fov = 110.0° - "
                           f"{'matches horizontal' if abs(camera_horizontal_fov - 110.0) < 1.0 else 'does NOT match horizontal'}")
        
        # CRITICAL: Log every frame to debug why recording shows 0.0
        if self.frame_count < 5:  # Log first 5 frames to debug
            logger.info(f"[CAMERA CALIB DEBUG] Frame {self.frame_count}: camera_8m_screen_y = {camera_8m_screen_y:.1f}, "
                       f"raw dict value = {vehicle_state_dict.get('camera8mScreenY')}, "
                       f"snake_case value = {vehicle_state_dict.get('camera_8m_screen_y')}")
        
        vehicle_state = VehicleState(
            timestamp=timestamp,
            position=np.array([position[0], vehicle_state_dict.get('position', {}).get('y', 0.0), position[1]]),
            rotation=rotation,  # Proper quaternion extraction
            velocity=np.array([0, 0, vehicle_state_dict.get('speed', 0.0)]),
            angular_velocity=np.array([0, 0, 0]),
            speed=vehicle_state_dict.get('speed', 0.0),
            steering_angle=vehicle_state_dict.get('steeringAngle', 0.0),
            steering_angle_actual=vehicle_state_dict.get(
                'steeringAngleActual', vehicle_state_dict.get('steering_angle_actual', 0.0)
            ),
            steering_input=vehicle_state_dict.get(
                'steeringInput', vehicle_state_dict.get('steering_input', 0.0)
            ),
            desired_steer_angle=vehicle_state_dict.get(
                'desiredSteerAngle', vehicle_state_dict.get('desired_steer_angle', 0.0)
            ),
            motor_torque=vehicle_state_dict.get('motorTorque', 0.0),
            brake_torque=vehicle_state_dict.get('brakeTorque', 0.0),
            unity_time=vehicle_state_dict.get('unityTime', vehicle_state_dict.get('unity_time', 0.0)),
            unity_frame_count=vehicle_state_dict.get('unityFrameCount', vehicle_state_dict.get('unity_frame_count', 0)),
            unity_delta_time=vehicle_state_dict.get('unityDeltaTime', vehicle_state_dict.get('unity_delta_time', 0.0)),
            unity_smooth_delta_time=vehicle_state_dict.get('unitySmoothDeltaTime', vehicle_state_dict.get('unity_smooth_delta_time', 0.0)),
            unity_unscaled_delta_time=vehicle_state_dict.get('unityUnscaledDeltaTime', vehicle_state_dict.get('unity_unscaled_delta_time', 0.0)),
            unity_time_scale=vehicle_state_dict.get('unityTimeScale', vehicle_state_dict.get('unity_time_scale', 1.0)),
            ground_truth_left_lane_line_x=gt_left_lane_line_x,
            ground_truth_right_lane_line_x=gt_right_lane_line_x,
            ground_truth_lane_center_x=gt_lane_center_x,
            ground_truth_path_curvature=gt_path_curvature,
            ground_truth_desired_heading=gt_desired_heading,
            camera_8m_screen_y=camera_8m_screen_y,  # NEW: Camera calibration data
            camera_lookahead_screen_y=camera_lookahead_screen_y,
            ground_truth_lookahead_distance=ground_truth_lookahead_distance,
            oracle_trajectory_world_xyz=np.asarray(
                vehicle_state_dict.get(
                    'oracleTrajectoryWorldXYZ',
                    vehicle_state_dict.get('oracle_trajectory_world_xyz', []),
                ),
                dtype=np.float32,
            ),
            oracle_trajectory_screen_xy=np.asarray(
                vehicle_state_dict.get(
                    'oracleTrajectoryScreenXY',
                    vehicle_state_dict.get('oracle_trajectory_screen_xy', []),
                ),
                dtype=np.float32,
            ),
            right_lane_fiducials_vehicle_true_xy=np.asarray(
                vehicle_state_dict.get(
                    'rightLaneFiducialsVehicleTrueXY',
                    vehicle_state_dict.get('right_lane_fiducials_vehicle_true_xy', []),
                ),
                dtype=np.float32,
            ),
            right_lane_fiducials_vehicle_monotonic_xy=np.asarray(
                vehicle_state_dict.get(
                    'rightLaneFiducialsVehicleMonotonicXY',
                    vehicle_state_dict.get('right_lane_fiducials_vehicle_monotonic_xy', []),
                ),
                dtype=np.float32,
            ),
            right_lane_fiducials_world_xyz=np.asarray(
                vehicle_state_dict.get(
                    'rightLaneFiducialsWorldXYZ',
                    vehicle_state_dict.get('right_lane_fiducials_world_xyz', []),
                ),
                dtype=np.float32,
            ),
            right_lane_fiducials_vehicle_xy=np.asarray(
                vehicle_state_dict.get(
                    'rightLaneFiducialsVehicleXY',
                    vehicle_state_dict.get(
                        'right_lane_fiducials_vehicle_xy',
                        vehicle_state_dict.get(
                            'rightLaneFiducialsVehicleTrueXY',
                            vehicle_state_dict.get('right_lane_fiducials_vehicle_true_xy', []),
                        ),
                    ),
                ),
                dtype=np.float32,
            ),
            right_lane_fiducials_screen_xy=np.asarray(
                vehicle_state_dict.get(
                    'rightLaneFiducialsScreenXY',
                    vehicle_state_dict.get('right_lane_fiducials_screen_xy', []),
                ),
                dtype=np.float32,
            ),
            right_lane_fiducials_point_count=int(
                vehicle_state_dict.get(
                    'rightLaneFiducialsPointCount',
                    vehicle_state_dict.get('right_lane_fiducials_point_count', 0),
                )
            ),
            right_lane_fiducials_horizon_meters=float(
                vehicle_state_dict.get(
                    'rightLaneFiducialsHorizonMeters',
                    vehicle_state_dict.get('right_lane_fiducials_horizon_meters', 0.0),
                )
            ),
            right_lane_fiducials_spacing_meters=float(
                vehicle_state_dict.get(
                    'rightLaneFiducialsSpacingMeters',
                    vehicle_state_dict.get('right_lane_fiducials_spacing_meters', 0.0),
                )
            ),
            right_lane_fiducials_enabled=bool(
                vehicle_state_dict.get(
                    'rightLaneFiducialsEnabled',
                    vehicle_state_dict.get('right_lane_fiducials_enabled', False),
                )
            ),
            camera_field_of_view=camera_field_of_view,  # NEW: Camera FOV data
            camera_horizontal_fov=camera_horizontal_fov,  # NEW: Camera horizontal FOV
            camera_pos_x=vehicle_state_dict.get('cameraPosX', 0.0),  # NEW: Camera position
            camera_pos_y=vehicle_state_dict.get('cameraPosY', 0.0),
            camera_pos_z=vehicle_state_dict.get('cameraPosZ', 0.0),
            camera_forward_x=vehicle_state_dict.get('cameraForwardX', 0.0),  # NEW: Camera forward
            camera_forward_y=vehicle_state_dict.get('cameraForwardY', 0.0),
            camera_forward_z=vehicle_state_dict.get('cameraForwardZ', 0.0),
            topdown_camera_pos_x=vehicle_state_dict.get(
                'topDownCameraPosX',
                vehicle_state_dict.get('topdown_camera_pos_x', 0.0),
            ),
            topdown_camera_pos_y=vehicle_state_dict.get(
                'topDownCameraPosY',
                vehicle_state_dict.get('topdown_camera_pos_y', 0.0),
            ),
            topdown_camera_pos_z=vehicle_state_dict.get(
                'topDownCameraPosZ',
                vehicle_state_dict.get('topdown_camera_pos_z', 0.0),
            ),
            topdown_camera_forward_x=vehicle_state_dict.get(
                'topDownCameraForwardX',
                vehicle_state_dict.get('topdown_camera_forward_x', 0.0),
            ),
            topdown_camera_forward_y=vehicle_state_dict.get(
                'topDownCameraForwardY',
                vehicle_state_dict.get('topdown_camera_forward_y', 0.0),
            ),
            topdown_camera_forward_z=vehicle_state_dict.get(
                'topDownCameraForwardZ',
                vehicle_state_dict.get('topdown_camera_forward_z', 0.0),
            ),
            topdown_camera_orthographic_size=vehicle_state_dict.get(
                'topDownCameraOrthographicSize',
                vehicle_state_dict.get('topdown_camera_orthographic_size', 0.0),
            ),
            topdown_camera_field_of_view=vehicle_state_dict.get(
                'topDownCameraFieldOfView',
                vehicle_state_dict.get('topdown_camera_field_of_view', 0.0),
            ),
            stream_front_unity_dt_ms=stream_front_unity_dt_ms,
            stream_topdown_unity_dt_ms=stream_topdown_unity_dt_ms,
            stream_topdown_front_dt_ms=stream_topdown_front_dt_ms,
            stream_topdown_front_frame_id_delta=stream_topdown_front_frame_id_delta,
            stream_front_frame_id_delta=stream_front_frame_id_delta,
            stream_topdown_frame_id_delta=stream_topdown_frame_id_delta,
            stream_front_latest_age_ms=front_latest_age_ms,
            stream_front_queue_depth=front_queue_depth,
            stream_front_drop_count=front_drop_count,
            stream_front_decode_in_flight=front_decode_in_flight,
            stream_topdown_latest_age_ms=topdown_latest_age_ms,
            stream_topdown_queue_depth=topdown_queue_depth,
            stream_topdown_drop_count=topdown_drop_count,
            stream_topdown_decode_in_flight=topdown_decode_in_flight,
            stream_front_last_realtime_s=front_last_realtime_s,
            stream_topdown_last_realtime_s=topdown_last_realtime_s,
            stream_front_timestamp_minus_realtime_ms=stream_front_timestamp_minus_realtime_ms,
            stream_topdown_timestamp_minus_realtime_ms=stream_topdown_timestamp_minus_realtime_ms,
            stream_front_source_timestamp=stream_front_source_timestamp,
            stream_topdown_source_timestamp=stream_topdown_source_timestamp,
            stream_front_timestamp_reused=stream_front_timestamp_reused,
            stream_topdown_timestamp_reused=stream_topdown_timestamp_reused,
            stream_front_timestamp_non_monotonic=stream_front_timestamp_non_monotonic,
            stream_topdown_timestamp_non_monotonic=stream_topdown_timestamp_non_monotonic,
            stream_front_negative_frame_delta=stream_front_negative_frame_delta,
            stream_topdown_negative_frame_delta=stream_topdown_negative_frame_delta,
            stream_front_frame_id_reused=stream_front_frame_id_reused,
            stream_topdown_frame_id_reused=stream_topdown_frame_id_reused,
            stream_front_clock_jump=stream_front_clock_jump,
            stream_topdown_clock_jump=stream_topdown_clock_jump,
            # NEW: Debug fields for diagnosing ground truth offset issues
            road_center_at_car_x=vehicle_state_dict.get('roadCenterAtCarX', 0.0),
            road_center_at_car_y=vehicle_state_dict.get('roadCenterAtCarY', 0.0),
            road_center_at_car_z=vehicle_state_dict.get('roadCenterAtCarZ', 0.0),
            road_center_at_lookahead_x=vehicle_state_dict.get('roadCenterAtLookaheadX', 0.0),
            road_center_at_lookahead_y=vehicle_state_dict.get('roadCenterAtLookaheadY', 0.0),
            road_center_at_lookahead_z=vehicle_state_dict.get('roadCenterAtLookaheadZ', 0.0),
            road_center_reference_t=vehicle_state_dict.get('roadCenterReferenceT', 0.0),
            road_frame_lateral_offset=vehicle_state_dict.get('roadFrameLateralOffset', 0.0),
            road_heading_deg=vehicle_state_dict.get('roadHeadingDeg', 0.0),
            car_heading_deg=vehicle_state_dict.get('carHeadingDeg', 0.0),
            heading_delta_deg=vehicle_state_dict.get('headingDeltaDeg', 0.0),
            road_frame_lane_center_offset=vehicle_state_dict.get('roadFrameLaneCenterOffset', 0.0),
            road_frame_lane_center_error=vehicle_state_dict.get('roadFrameLaneCenterError', 0.0),
            vehicle_frame_lookahead_offset=vehicle_state_dict.get('vehicleFrameLookaheadOffset', 0.0),
            gt_rotation_debug_valid=vehicle_state_dict.get(
                'gtRotationDebugValid',
                vehicle_state_dict.get('gt_rotation_debug_valid', False),
            ),
            gt_rotation_used_road_frame=vehicle_state_dict.get(
                'gtRotationUsedRoadFrame',
                vehicle_state_dict.get('gt_rotation_used_road_frame', False),
            ),
            gt_rotation_rejected_road_frame_hop=vehicle_state_dict.get(
                'gtRotationRejectedRoadFrameHop',
                vehicle_state_dict.get('gt_rotation_rejected_road_frame_hop', False),
            ),
            gt_rotation_reference_heading_deg=vehicle_state_dict.get(
                'gtRotationReferenceHeadingDeg',
                vehicle_state_dict.get('gt_rotation_reference_heading_deg', 0.0),
            ),
            gt_rotation_road_frame_heading_deg=vehicle_state_dict.get(
                'gtRotationRoadFrameHeadingDeg',
                vehicle_state_dict.get('gt_rotation_road_frame_heading_deg', 0.0),
            ),
            gt_rotation_input_heading_deg=vehicle_state_dict.get(
                'gtRotationInputHeadingDeg',
                vehicle_state_dict.get('gt_rotation_input_heading_deg', 0.0),
            ),
            gt_rotation_road_vs_ref_delta_deg=vehicle_state_dict.get(
                'gtRotationRoadVsRefDeltaDeg',
                vehicle_state_dict.get('gt_rotation_road_vs_ref_delta_deg', 0.0),
            ),
            gt_rotation_applied_delta_deg=vehicle_state_dict.get(
                'gtRotationAppliedDeltaDeg',
                vehicle_state_dict.get('gt_rotation_applied_delta_deg', 0.0),
            ),
            speed_limit=speed_limit,
            speed_limit_preview=vehicle_state_dict.get(
                'speedLimitPreview', vehicle_state_dict.get('speed_limit_preview', 0.0)
            ),
            speed_limit_preview_distance=vehicle_state_dict.get(
                'speedLimitPreviewDistance',
                vehicle_state_dict.get('speed_limit_preview_distance', 0.0),
            ),
            speed_limit_preview_min_distance=vehicle_state_dict.get(
                'speedLimitPreviewMinDistance',
                vehicle_state_dict.get('speed_limit_preview_min_distance', 0.0),
            ),
            speed_limit_preview_mid=vehicle_state_dict.get(
                'speedLimitPreviewMid',
                vehicle_state_dict.get('speed_limit_preview_mid', 0.0),
            ),
            speed_limit_preview_mid_distance=vehicle_state_dict.get(
                'speedLimitPreviewMidDistance',
                vehicle_state_dict.get('speed_limit_preview_mid_distance', 0.0),
            ),
            speed_limit_preview_mid_min_distance=vehicle_state_dict.get(
                'speedLimitPreviewMidMinDistance',
                vehicle_state_dict.get('speed_limit_preview_mid_min_distance', 0.0),
            ),
            speed_limit_preview_long=vehicle_state_dict.get(
                'speedLimitPreviewLong',
                vehicle_state_dict.get('speed_limit_preview_long', 0.0),
            ),
            speed_limit_preview_long_distance=vehicle_state_dict.get(
                'speedLimitPreviewLongDistance',
                vehicle_state_dict.get('speed_limit_preview_long_distance', 0.0),
            ),
            speed_limit_preview_long_min_distance=vehicle_state_dict.get(
                'speedLimitPreviewLongMinDistance',
                vehicle_state_dict.get('speed_limit_preview_long_min_distance', 0.0),
            ),
        )
        
        # Create control command with metadata
        control_cmd = ControlCommand(
            timestamp=timestamp,
            steering=control_command['steering'],
            throttle=control_command['throttle'],
            brake=control_command['brake'],
            steering_before_limits=control_command.get('steering_before_limits'),
            throttle_before_limits=control_command.get('throttle_before_limits'),
            brake_before_limits=control_command.get('brake_before_limits'),
            accel_feedforward=control_command.get('accel_feedforward'),
            brake_feedforward=control_command.get('brake_feedforward'),
            longitudinal_accel_capped=bool(control_command.get('longitudinal_accel_capped', False)),
            longitudinal_jerk_capped=bool(control_command.get('longitudinal_jerk_capped', False)),
            pid_integral=control_command.get('pid_integral'),
            pid_derivative=control_command.get('pid_derivative'),
            pid_error=control_command.get('total_error'),  # Total error before PID
            lateral_error=control_command.get('lateral_error'),
            heading_error=control_command.get('heading_error'),
            total_error=control_command.get('total_error'),
            total_error_scaled=control_command.get('total_error_scaled'),
            path_curvature_input=control_command.get('path_curvature_input'),
            feedforward_steering=control_command.get('feedforward_steering'),
            feedback_steering=control_command.get('feedback_steering'),
            straight_sign_flip_override_active=control_command.get(
                'straight_sign_flip_override_active'
            ),
            straight_sign_flip_triggered=control_command.get(
                'straight_sign_flip_triggered'
            ),
            straight_sign_flip_trigger_error=control_command.get(
                'straight_sign_flip_trigger_error'
            ),
            straight_sign_flip_frames_remaining=control_command.get(
                'straight_sign_flip_frames_remaining'
            ),
            is_straight=control_command.get('is_straight'),
            is_control_straight_proxy=control_command.get('is_control_straight_proxy'),
            curve_upcoming=control_command.get('curve_upcoming'),
            curve_at_car=control_command.get('curve_at_car'),
            curve_at_car_distance_remaining_m=control_command.get(
                'curve_at_car_distance_remaining_m'
            ),
            is_road_straight=control_command.get('is_road_straight'),
            road_curvature_valid=control_command.get('road_curvature_valid'),
            road_curvature_abs=control_command.get('road_curvature_abs'),
            road_curvature_source=control_command.get('road_curvature_source'),
            straight_oscillation_rate=control_command.get('straight_oscillation_rate'),
            tuned_deadband=control_command.get('tuned_deadband'),
            tuned_error_smoothing_alpha=control_command.get('tuned_error_smoothing_alpha'),
            # Diagnostic fields for tracking stale perception usage
            using_stale_perception=perception_output.using_stale_data if perception_output else False,
            stale_perception_reason=perception_output.stale_data_reason if perception_output else None,
            emergency_stop=bool(control_command.get('emergency_stop', False)),
            target_speed_raw=control_command.get('target_speed_raw'),
            target_speed_post_limits=control_command.get('target_speed_post_limits'),
            target_speed_planned=control_command.get('target_speed_planned'),
            target_speed_final=control_command.get('target_speed_final'),
            target_speed_slew_active=bool(control_command.get('target_speed_slew_active', False)),
            target_speed_ramp_active=bool(control_command.get('target_speed_ramp_active', False)),
            curve_mode_speed_cap_active=bool(
                control_command.get('curve_mode_speed_cap_active', False)
            ),
            curve_mode_speed_cap_clamped=bool(
                control_command.get('curve_mode_speed_cap_clamped', False)
            ),
            curve_mode_speed_cap_value=control_command.get('curve_mode_speed_cap_value'),
            speed_governor_active_limiter=control_command.get('speed_governor_active_limiter', 'none'),
            speed_governor_comfort_speed=control_command.get('speed_governor_comfort_speed'),
            speed_governor_preview_speed=control_command.get('speed_governor_preview_speed'),
            speed_governor_horizon_speed=control_command.get('speed_governor_horizon_speed'),
            launch_throttle_cap=control_command.get('launch_throttle_cap'),
            launch_throttle_cap_active=bool(control_command.get('launch_throttle_cap_active', False)),
            steering_pre_rate_limit=control_command.get('steering_pre_rate_limit'),
            steering_post_rate_limit=control_command.get('steering_post_rate_limit'),
            steering_post_jerk_limit=control_command.get('steering_post_jerk_limit'),
            steering_post_sign_flip=control_command.get('steering_post_sign_flip'),
            steering_post_hard_clip=control_command.get('steering_post_hard_clip'),
            steering_post_smoothing=control_command.get('steering_post_smoothing'),
            steering_rate_limited_active=bool(control_command.get('steering_rate_limited_active', False)),
            steering_jerk_limited_active=bool(control_command.get('steering_jerk_limited_active', False)),
            steering_hard_clip_active=bool(control_command.get('steering_hard_clip_active', False)),
            steering_smoothing_active=bool(control_command.get('steering_smoothing_active', False)),
            steering_rate_limited_delta=control_command.get('steering_rate_limited_delta'),
            steering_jerk_limited_delta=control_command.get('steering_jerk_limited_delta'),
            steering_hard_clip_delta=control_command.get('steering_hard_clip_delta'),
            steering_smoothing_delta=control_command.get('steering_smoothing_delta'),
            steering_rate_limit_base_from_error=control_command.get('steering_rate_limit_base_from_error'),
            steering_rate_limit_curve_scale=control_command.get('steering_rate_limit_curve_scale'),
            steering_rate_limit_curve_metric_abs=control_command.get('steering_rate_limit_curve_metric_abs'),
            steering_rate_limit_curve_metric_source=control_command.get('steering_rate_limit_curve_metric_source'),
            steering_rate_limit_curve_min=control_command.get('steering_rate_limit_curve_min'),
            steering_rate_limit_curve_max=control_command.get('steering_rate_limit_curve_max'),
            steering_rate_limit_scale_min=control_command.get('steering_rate_limit_scale_min'),
            steering_rate_limit_curve_regime_code=control_command.get('steering_rate_limit_curve_regime_code'),
            steering_rate_limit_after_curve=control_command.get('steering_rate_limit_after_curve'),
            steering_rate_limit_after_floor=control_command.get('steering_rate_limit_after_floor'),
            steering_rate_limit_effective=control_command.get('steering_rate_limit_effective'),
            steering_rate_limit_requested_delta=control_command.get('steering_rate_limit_requested_delta'),
            steering_rate_limit_margin=control_command.get('steering_rate_limit_margin'),
            steering_rate_limit_unlock_delta_needed=control_command.get('steering_rate_limit_unlock_delta_needed'),
            curve_entry_assist_active=bool(control_command.get('curve_entry_assist_active', False)),
            curve_entry_assist_triggered=bool(control_command.get('curve_entry_assist_triggered', False)),
            curve_entry_assist_rearm_frames_remaining=control_command.get(
                'curve_entry_assist_rearm_frames_remaining'
            ),
            dynamic_curve_authority_active=bool(
                control_command.get('dynamic_curve_authority_active', False)
            ),
            dynamic_curve_rate_request_delta=control_command.get(
                'dynamic_curve_rate_request_delta'
            ),
            dynamic_curve_rate_deficit=control_command.get('dynamic_curve_rate_deficit'),
            dynamic_curve_rate_boost=control_command.get('dynamic_curve_rate_boost'),
            dynamic_curve_jerk_boost_factor=control_command.get(
                'dynamic_curve_jerk_boost_factor'
            ),
            dynamic_curve_lateral_accel_est_g=control_command.get(
                'dynamic_curve_lateral_accel_est_g'
            ),
            dynamic_curve_lateral_jerk_est_gps=control_command.get(
                'dynamic_curve_lateral_jerk_est_gps'
            ),
            dynamic_curve_lateral_jerk_est_smoothed_gps=control_command.get(
                'dynamic_curve_lateral_jerk_est_smoothed_gps'
            ),
            dynamic_curve_speed_scale=control_command.get('dynamic_curve_speed_scale'),
            dynamic_curve_comfort_scale=control_command.get('dynamic_curve_comfort_scale'),
            dynamic_curve_comfort_accel_gate=control_command.get(
                'dynamic_curve_comfort_accel_gate'
            ),
            dynamic_curve_comfort_jerk_penalty=control_command.get(
                'dynamic_curve_comfort_jerk_penalty'
            ),
            dynamic_curve_rate_boost_cap_effective=control_command.get(
                'dynamic_curve_rate_boost_cap_effective'
            ),
            dynamic_curve_jerk_boost_cap_effective=control_command.get(
                'dynamic_curve_jerk_boost_cap_effective'
            ),
            dynamic_curve_hard_clip_boost=control_command.get('dynamic_curve_hard_clip_boost'),
            dynamic_curve_hard_clip_boost_cap_effective=control_command.get(
                'dynamic_curve_hard_clip_boost_cap_effective'
            ),
            dynamic_curve_hard_clip_limit_effective=control_command.get(
                'dynamic_curve_hard_clip_limit_effective'
            ),
            dynamic_curve_entry_governor_active=control_command.get(
                'dynamic_curve_entry_governor_active'
            ),
            dynamic_curve_entry_governor_scale=control_command.get(
                'dynamic_curve_entry_governor_scale'
            ),
            dynamic_curve_authority_deficit_streak=control_command.get(
                'dynamic_curve_authority_deficit_streak'
            ),
            curve_entry_schedule_active=bool(control_command.get('curve_entry_schedule_active', False)),
            curve_entry_schedule_triggered=bool(control_command.get('curve_entry_schedule_triggered', False)),
            curve_entry_schedule_handoff_triggered=bool(
                control_command.get('curve_entry_schedule_handoff_triggered', False)
            ),
            curve_entry_schedule_frames_remaining=control_command.get(
                'curve_entry_schedule_frames_remaining'
            ),
            curve_commit_mode_active=bool(control_command.get('curve_commit_mode_active', False)),
            curve_commit_mode_triggered=bool(
                control_command.get('curve_commit_mode_triggered', False)
            ),
            curve_commit_mode_handoff_triggered=bool(
                control_command.get('curve_commit_mode_handoff_triggered', False)
            ),
            curve_commit_mode_frames_remaining=control_command.get(
                'curve_commit_mode_frames_remaining'
            ),
            steering_jerk_limit_effective=control_command.get('steering_jerk_limit_effective'),
            steering_jerk_curve_scale=control_command.get('steering_jerk_curve_scale'),
            steering_jerk_limit_requested_rate_delta=control_command.get('steering_jerk_limit_requested_rate_delta'),
            steering_jerk_limit_allowed_rate_delta=control_command.get('steering_jerk_limit_allowed_rate_delta'),
            steering_jerk_limit_margin=control_command.get('steering_jerk_limit_margin'),
            steering_jerk_limit_unlock_rate_delta_needed=control_command.get('steering_jerk_limit_unlock_rate_delta_needed'),
            steering_authority_gap=control_command.get('steering_authority_gap'),
            steering_transfer_ratio=control_command.get('steering_transfer_ratio'),
            steering_first_limiter_stage_code=control_command.get('steering_first_limiter_stage_code'),
            curve_unwind_active=bool(control_command.get('curve_unwind_active', False)),
            curve_unwind_frames_remaining=control_command.get('curve_unwind_frames_remaining'),
            curve_unwind_progress=control_command.get('curve_unwind_progress'),
            curve_unwind_rate_scale=control_command.get('curve_unwind_rate_scale'),
            curve_unwind_jerk_scale=control_command.get('curve_unwind_jerk_scale'),
            curve_unwind_integral_decay_applied=control_command.get(
                'curve_unwind_integral_decay_applied'
            ),
            turn_feasibility_active=bool(control_command.get('turn_feasibility_active', False)),
            turn_feasibility_infeasible=bool(control_command.get('turn_feasibility_infeasible', False)),
            turn_feasibility_curvature_abs=control_command.get('turn_feasibility_curvature_abs'),
            turn_feasibility_speed_mps=control_command.get('turn_feasibility_speed_mps'),
            turn_feasibility_required_lat_accel_g=control_command.get(
                'turn_feasibility_required_lat_accel_g'
            ),
            turn_feasibility_comfort_limit_g=control_command.get('turn_feasibility_comfort_limit_g'),
            turn_feasibility_peak_limit_g=control_command.get('turn_feasibility_peak_limit_g'),
            turn_feasibility_selected_limit_g=control_command.get('turn_feasibility_selected_limit_g'),
            turn_feasibility_guardband_g=control_command.get('turn_feasibility_guardband_g'),
            turn_feasibility_margin_g=control_command.get('turn_feasibility_margin_g'),
            turn_feasibility_speed_limit_mps=control_command.get('turn_feasibility_speed_limit_mps'),
            turn_feasibility_speed_delta_mps=control_command.get('turn_feasibility_speed_delta_mps'),
            turn_feasibility_use_peak_bound=bool(
                control_command.get('turn_feasibility_use_peak_bound', True)
            ),
            pp_alpha=control_command.get('pp_alpha'),
            pp_lookahead_distance=control_command.get('pp_lookahead_distance'),
            pp_geometric_steering=control_command.get('pp_geometric_steering'),
            pp_feedback_steering=control_command.get('pp_feedback_steering'),
            pp_ref_jump_clamped=bool(control_command.get('pp_ref_jump_clamped', 0) > 0.5),
            pp_stale_hold_active=bool(control_command.get('pp_stale_hold_active', 0) > 0.5),
            pp_pipeline_bypass_active=bool(control_command.get('pp_pipeline_bypass_active', 0) > 0.5),
        )
        
        # Create trajectory output
        # Include reference point as first point for analysis
        trajectory_points = None
        velocities = None
        ref_point = None  # Initialize ref_point
        # FIXED: Check if trajectory exists AND has points (not just truthy check)
        if trajectory is not None and hasattr(trajectory, 'points') and len(trajectory.points) > 0:
            confidence = perception_output.confidence if perception_output else None
            # Get reference point for recording
            # Extract lane_coeffs from perception_output for direct computation
            lane_coeffs_for_ref = None
            if perception_output and perception_output.lane_line_coefficients is not None:
                # Convert numpy array back to list of arrays
                # Handle both 2D array (shape (N, 3)) and array of arrays
                coeffs_array = perception_output.lane_line_coefficients
                if coeffs_array.ndim == 2:
                    # 2D array: convert each row to array
                    lane_coeffs_for_ref = [np.array(row) for row in coeffs_array]
                else:
                    # Array of arrays: convert to list
                    lane_coeffs_for_ref = [coeffs for coeffs in coeffs_array]
            
            reference_lookahead = float(self.trajectory_config.get('reference_lookahead', 8.0))
            runtime_ref_diag = runtime_reference_point if isinstance(runtime_reference_point, dict) else {}
            dynamic_horizon_applied = float(
                runtime_ref_diag.get("diag_dynamic_effective_horizon_applied", 0.0)
            )
            if dynamic_horizon_applied > 0.5:
                reference_lookahead = float(
                    runtime_ref_diag.get(
                        "diag_dynamic_effective_horizon_m",
                        reference_lookahead,
                    )
                )
            # Get lane positions from perception_output for most accurate reference point
            lane_positions_for_ref = None
            if perception_output:
                lane_positions_for_ref = {
                    'left_lane_line_x': perception_output.left_lane_line_x,
                    'right_lane_line_x': perception_output.right_lane_line_x
                }
            
            if runtime_reference_point is not None:
                ref_point = dict(runtime_reference_point)
            else:
                ref_point = self.trajectory_planner.get_reference_point(
                    trajectory, 
                    lookahead=reference_lookahead,
                    lane_coeffs=lane_coeffs_for_ref,  # Pass lane coefficients for direct computation
                    lane_positions=lane_positions_for_ref,  # Preferred: use vehicle coords
                    use_direct=True,  # Use direct midpoint computation
                    timestamp=timestamp,
                    confidence=confidence,
                )
            
            # Build trajectory points array with reference point first
            points_list = []
            vel_list = []
            
            # Add reference point first (for drift analysis)
            if ref_point:
                points_list.append([ref_point['x'], ref_point['y'], ref_point['heading']])
                vel_list.append(ref_point.get('velocity', 8.0))
            
            # Add other trajectory points
            for p in trajectory.points:
                points_list.append([p.x, p.y, p.heading])
                vel_list.append(p.velocity)
            
            trajectory_points = np.array(points_list) if points_list else None
            velocities = np.array(vel_list) if vel_list else None
        
        # Extract debug info from reference point
        ref_point_method = None
        perception_center_x = None
        if ref_point:
            ref_point_method = ref_point.get('method')  # 'lane_positions', 'lane_coeffs', or 'trajectory'
            perception_center_x = ref_point.get('perception_center_x')  # Perception center before trajectory calc
        oracle_points = None
        oracle_point_count = None
        oracle_horizon_meters = None
        oracle_point_spacing_meters = None
        oracle_samples_enabled = False
        oracle_raw = vehicle_state_dict.get(
            'oracleTrajectoryXY',
            vehicle_state_dict.get('oracle_trajectory_xy'),
        )
        if oracle_raw is not None:
            try:
                oracle_arr = np.asarray(oracle_raw, dtype=np.float32).reshape(-1)
                if oracle_arr.size >= 2 and oracle_arr.size % 2 == 0:
                    oracle_points = oracle_arr.reshape(-1, 2)
            except (ValueError, TypeError):
                oracle_points = None
        oracle_point_count = int(
            vehicle_state_dict.get(
                'oraclePointCount',
                vehicle_state_dict.get(
                    'oracle_point_count',
                    oracle_points.shape[0] if oracle_points is not None else 0,
                ),
            )
        )
        oracle_horizon_meters = float(
            vehicle_state_dict.get(
                'oracleHorizonMeters',
                vehicle_state_dict.get('oracle_horizon_meters', 0.0),
            )
        )
        oracle_point_spacing_meters = float(
            vehicle_state_dict.get(
                'oraclePointSpacingMeters',
                vehicle_state_dict.get('oracle_point_spacing_meters', 0.0),
            )
        )
        oracle_samples_enabled = bool(
            vehicle_state_dict.get(
                'oracleSamplesEnabled',
                vehicle_state_dict.get('oracle_samples_enabled', False),
            )
        )
        traj_diag = self.trajectory_planner.get_last_generation_diagnostics()
        ref_diag = {}
        ref_getter = getattr(self.trajectory_planner, 'get_last_reference_diagnostics', None)
        if callable(ref_getter):
            ref_diag = ref_getter() or {}
        ref_point_diag = ref_point if isinstance(ref_point, dict) else {}
        if not isinstance(speed_horizon_guardrail_diag, dict):
            speed_horizon_guardrail_diag = {}
        
        trajectory_output = TrajectoryOutput(
            timestamp=timestamp,
            trajectory_points=trajectory_points,
            trajectory_source=str(trajectory_source),
            oracle_points=oracle_points,
            oracle_point_count=oracle_point_count,
            oracle_horizon_meters=oracle_horizon_meters,
            oracle_point_spacing_meters=oracle_point_spacing_meters,
            oracle_samples_enabled=oracle_samples_enabled,
            velocities=velocities,
            curvature=(
                float(ref_point.get("curvature", 0.0))
                if isinstance(ref_point, dict)
                else None
            ),
            reference_point=ref_point,  # Store reference point for analysis
            reference_point_method=ref_point_method,  # NEW: Track which method was used
            perception_center_x=perception_center_x,  # NEW: Store perception center for comparison
            diag_available=traj_diag.get('diag_available'),
            diag_generated_by_fallback=traj_diag.get('diag_generated_by_fallback'),
            diag_points_generated=traj_diag.get('diag_points_generated'),
            diag_x_clip_count=traj_diag.get('diag_x_clip_count'),
            diag_pre_y0=traj_diag.get('diag_pre_y0'),
            diag_pre_y1=traj_diag.get('diag_pre_y1'),
            diag_pre_y2=traj_diag.get('diag_pre_y2'),
            diag_post_y0=traj_diag.get('diag_post_y0'),
            diag_post_y1=traj_diag.get('diag_post_y1'),
            diag_post_y2=traj_diag.get('diag_post_y2'),
            diag_used_provided_distance0=traj_diag.get('diag_used_provided_distance0'),
            diag_used_provided_distance1=traj_diag.get('diag_used_provided_distance1'),
            diag_used_provided_distance2=traj_diag.get('diag_used_provided_distance2'),
            diag_post_minus_pre_y0=traj_diag.get('diag_post_minus_pre_y0'),
            diag_post_minus_pre_y1=traj_diag.get('diag_post_minus_pre_y1'),
            diag_post_minus_pre_y2=traj_diag.get('diag_post_minus_pre_y2'),
            diag_preclip_x0=traj_diag.get('diag_preclip_x0'),
            diag_preclip_x1=traj_diag.get('diag_preclip_x1'),
            diag_preclip_x2=traj_diag.get('diag_preclip_x2'),
            diag_preclip_x_abs_max=traj_diag.get('diag_preclip_x_abs_max'),
            diag_preclip_x_abs_p95=traj_diag.get('diag_preclip_x_abs_p95'),
            diag_preclip_mean_12_20m_lane_source_x=traj_diag.get('diag_preclip_mean_12_20m_lane_source_x'),
            diag_preclip_mean_12_20m_distance_scale_delta_x=traj_diag.get('diag_preclip_mean_12_20m_distance_scale_delta_x'),
            diag_preclip_mean_12_20m_camera_offset_delta_x=traj_diag.get('diag_preclip_mean_12_20m_camera_offset_delta_x'),
            diag_preclip_abs_mean_12_20m_lane_source_x=traj_diag.get('diag_preclip_abs_mean_12_20m_lane_source_x'),
            diag_preclip_abs_mean_12_20m_distance_scale_delta_x=traj_diag.get('diag_preclip_abs_mean_12_20m_distance_scale_delta_x'),
            diag_preclip_abs_mean_12_20m_camera_offset_delta_x=traj_diag.get('diag_preclip_abs_mean_12_20m_camera_offset_delta_x'),
            diag_heading_zero_gate_active=ref_diag.get('diag_heading_zero_gate_active', ref_point_diag.get('diag_heading_zero_gate_active')),
            diag_small_heading_gate_active=ref_diag.get('diag_small_heading_gate_active', ref_point_diag.get('diag_small_heading_gate_active')),
            diag_multi_lookahead_active=ref_diag.get('diag_multi_lookahead_active', ref_point_diag.get('diag_multi_lookahead_active')),
            diag_smoothing_jump_reject=ref_diag.get('diag_smoothing_jump_reject', ref_point_diag.get('diag_smoothing_jump_reject')),
            diag_ref_x_rate_limit_active=ref_diag.get('diag_ref_x_rate_limit_active', ref_point_diag.get('diag_ref_x_rate_limit_active')),
            diag_raw_ref_x=ref_diag.get('diag_raw_ref_x', ref_point_diag.get('diag_raw_ref_x')),
            diag_smoothed_ref_x=ref_diag.get('diag_smoothed_ref_x', ref_point_diag.get('diag_smoothed_ref_x')),
            diag_ref_x_suppression_abs=ref_diag.get('diag_ref_x_suppression_abs', ref_point_diag.get('diag_ref_x_suppression_abs')),
            diag_raw_ref_heading=ref_diag.get('diag_raw_ref_heading', ref_point_diag.get('diag_raw_ref_heading')),
            diag_smoothed_ref_heading=ref_diag.get('diag_smoothed_ref_heading', ref_point_diag.get('diag_smoothed_ref_heading')),
            diag_heading_suppression_abs=ref_diag.get('diag_heading_suppression_abs', ref_point_diag.get('diag_heading_suppression_abs')),
            diag_smoothing_alpha=ref_diag.get('diag_smoothing_alpha', ref_point_diag.get('diag_smoothing_alpha')),
            diag_smoothing_alpha_x=ref_diag.get('diag_smoothing_alpha_x', ref_point_diag.get('diag_smoothing_alpha_x')),
            diag_multi_lookahead_heading_base=ref_diag.get('diag_multi_lookahead_heading_base', ref_point_diag.get('diag_multi_lookahead_heading_base')),
            diag_multi_lookahead_heading_far=ref_diag.get('diag_multi_lookahead_heading_far', ref_point_diag.get('diag_multi_lookahead_heading_far')),
            diag_multi_lookahead_heading_blended=ref_diag.get('diag_multi_lookahead_heading_blended', ref_point_diag.get('diag_multi_lookahead_heading_blended')),
            diag_multi_lookahead_blend_alpha=ref_diag.get('diag_multi_lookahead_blend_alpha', ref_point_diag.get('diag_multi_lookahead_blend_alpha')),
            diag_dynamic_effective_horizon_m=ref_diag.get('diag_dynamic_effective_horizon_m', ref_point_diag.get('diag_dynamic_effective_horizon_m')),
            diag_dynamic_effective_horizon_base_m=ref_diag.get('diag_dynamic_effective_horizon_base_m', ref_point_diag.get('diag_dynamic_effective_horizon_base_m')),
            diag_dynamic_effective_horizon_min_m=ref_diag.get('diag_dynamic_effective_horizon_min_m', ref_point_diag.get('diag_dynamic_effective_horizon_min_m')),
            diag_dynamic_effective_horizon_max_m=ref_diag.get('diag_dynamic_effective_horizon_max_m', ref_point_diag.get('diag_dynamic_effective_horizon_max_m')),
            diag_dynamic_effective_horizon_speed_scale=ref_diag.get('diag_dynamic_effective_horizon_speed_scale', ref_point_diag.get('diag_dynamic_effective_horizon_speed_scale')),
            diag_dynamic_effective_horizon_curvature_scale=ref_diag.get('diag_dynamic_effective_horizon_curvature_scale', ref_point_diag.get('diag_dynamic_effective_horizon_curvature_scale')),
            diag_dynamic_effective_horizon_confidence_scale=ref_diag.get('diag_dynamic_effective_horizon_confidence_scale', ref_point_diag.get('diag_dynamic_effective_horizon_confidence_scale')),
            diag_dynamic_effective_horizon_final_scale=ref_diag.get('diag_dynamic_effective_horizon_final_scale', ref_point_diag.get('diag_dynamic_effective_horizon_final_scale')),
            diag_dynamic_effective_horizon_speed_mps=ref_diag.get('diag_dynamic_effective_horizon_speed_mps', ref_point_diag.get('diag_dynamic_effective_horizon_speed_mps')),
            diag_dynamic_effective_horizon_curvature_abs=ref_diag.get('diag_dynamic_effective_horizon_curvature_abs', ref_point_diag.get('diag_dynamic_effective_horizon_curvature_abs')),
            diag_dynamic_effective_horizon_confidence_used=ref_diag.get('diag_dynamic_effective_horizon_confidence_used', ref_point_diag.get('diag_dynamic_effective_horizon_confidence_used')),
            diag_dynamic_effective_horizon_limiter_code=ref_diag.get('diag_dynamic_effective_horizon_limiter_code', ref_point_diag.get('diag_dynamic_effective_horizon_limiter_code')),
            diag_dynamic_effective_horizon_applied=ref_diag.get('diag_dynamic_effective_horizon_applied', ref_point_diag.get('diag_dynamic_effective_horizon_applied')),
            diag_speed_horizon_guardrail_active=speed_horizon_guardrail_diag.get('diag_speed_horizon_guardrail_active'),
            diag_speed_horizon_guardrail_margin_m=speed_horizon_guardrail_diag.get('diag_speed_horizon_guardrail_margin_m'),
            diag_speed_horizon_guardrail_horizon_m=speed_horizon_guardrail_diag.get('diag_speed_horizon_guardrail_horizon_m'),
            diag_speed_horizon_guardrail_time_headway_s=speed_horizon_guardrail_diag.get('diag_speed_horizon_guardrail_time_headway_s'),
            diag_speed_horizon_guardrail_margin_buffer_m=speed_horizon_guardrail_diag.get('diag_speed_horizon_guardrail_margin_buffer_m'),
            diag_speed_horizon_guardrail_allowed_speed_mps=speed_horizon_guardrail_diag.get('diag_speed_horizon_guardrail_allowed_speed_mps'),
            diag_speed_horizon_guardrail_target_speed_before_mps=speed_horizon_guardrail_diag.get('diag_speed_horizon_guardrail_target_speed_before_mps'),
            diag_speed_horizon_guardrail_target_speed_after_mps=speed_horizon_guardrail_diag.get('diag_speed_horizon_guardrail_target_speed_after_mps'),
            diag_preclip_abs_mean_0_8m=traj_diag.get('diag_preclip_abs_mean_0_8m'),
            diag_preclip_abs_mean_8_12m=traj_diag.get('diag_preclip_abs_mean_8_12m'),
            diag_preclip_abs_mean_12_20m=traj_diag.get('diag_preclip_abs_mean_12_20m'),
            diag_postclip_x0=traj_diag.get('diag_postclip_x0'),
            diag_postclip_x1=traj_diag.get('diag_postclip_x1'),
            diag_postclip_x2=traj_diag.get('diag_postclip_x2'),
            diag_postclip_abs_mean_0_8m=traj_diag.get('diag_postclip_abs_mean_0_8m'),
            diag_postclip_abs_mean_8_12m=traj_diag.get('diag_postclip_abs_mean_8_12m'),
            diag_postclip_abs_mean_12_20m=traj_diag.get('diag_postclip_abs_mean_12_20m'),
            diag_postclip_near_clip_frac_12_20m=traj_diag.get('diag_postclip_near_clip_frac_12_20m'),
            diag_first_segment_y0_gt_y1_pre=traj_diag.get('diag_first_segment_y0_gt_y1_pre'),
            diag_first_segment_y0_gt_y1_post=traj_diag.get('diag_first_segment_y0_gt_y1_post'),
            diag_inversion_introduced_after_conversion=traj_diag.get('diag_inversion_introduced_after_conversion'),
            diag_far_band_contribution_limited_active=traj_diag.get('diag_far_band_contribution_limited_active'),
            diag_far_band_contribution_limit_start_m=traj_diag.get('diag_far_band_contribution_limit_start_m'),
            diag_far_band_contribution_limit_gain=traj_diag.get('diag_far_band_contribution_limit_gain'),
            diag_far_band_contribution_scale_mean_12_20m=traj_diag.get('diag_far_band_contribution_scale_mean_12_20m'),
            diag_far_band_contribution_limited_frac_12_20m=traj_diag.get('diag_far_band_contribution_limited_frac_12_20m'),
        )
        
        # Allow ground truth follower to override control command for recording
        if hasattr(self, '_gt_control_override') and self._gt_control_override is not None:
            control_cmd = self._gt_control_override
        
        # Get Unity feedback (if available)
        unity_feedback = None
        if self.bridge:
            try:
                feedback_dict = self.bridge.get_latest_unity_feedback()
                if feedback_dict:
                    from data.formats.data_format import UnityFeedback
                    unity_feedback = UnityFeedback(
                        timestamp=feedback_dict.get('timestamp', timestamp),
                        ground_truth_mode_active=feedback_dict.get('ground_truth_mode_active', False),
                        control_command_received=feedback_dict.get('control_command_received', False),
                        actual_steering_applied=feedback_dict.get('actual_steering_applied'),
                        actual_throttle_applied=feedback_dict.get('actual_throttle_applied'),
                        actual_brake_applied=feedback_dict.get('actual_brake_applied'),
                        ground_truth_data_available=feedback_dict.get('ground_truth_data_available', False),
                        ground_truth_reporter_enabled=feedback_dict.get('ground_truth_reporter_enabled', False),
                        path_curvature_calculated=feedback_dict.get('path_curvature_calculated', False),
                        car_controller_mode=feedback_dict.get('car_controller_mode'),
                        av_control_enabled=feedback_dict.get('av_control_enabled', False),
                        unity_frame_count=feedback_dict.get('unity_frame_count'),
                        unity_time=feedback_dict.get('unity_time')
                    )
            except Exception:
                # Unity feedback is optional - don't fail if it's not available
                pass
        
        # Create recording frame
        frame = RecordingFrame(
            timestamp=timestamp,
            frame_id=self.frame_count,
            camera_frame=camera_frame,
            camera_topdown_frame=camera_topdown_frame,
            vehicle_state=vehicle_state,
            control_command=control_cmd,
            perception_output=perception_output,
            trajectory_output=trajectory_output,
            unity_feedback=unity_feedback
        )
        
        self.recorder.record_frame(frame)
    
    def stop(self):
        """Stop AV stack."""
        self.running = False
        
        # CRITICAL: Signal Unity to exit play mode gracefully before stopping
        # This ensures Unity exits play mode when AV stack stops (duration expiry, frame limit, etc.)
        try:
            logger.info("Signaling Unity to exit play mode...")
            self.bridge.signal_shutdown()
        except Exception as e:
            logger.warning(f"Error signaling Unity shutdown: {e}")
        
        if self.recorder:
            logger.info(f"Closing data recorder: {self.recorder.output_file}")
            self.recorder.close()
            logger.info(f"Recorded {self.frame_count} frames")
        
        logger.info(f"AV Stack stopped (processed {self.frame_count} frames)")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run AV Stack')
    parser.add_argument('--bridge_url', type=str, default='http://localhost:8000',
                       help='Unity bridge server URL')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained perception model')
    parser.add_argument('--record', action='store_true', default=True,
                       help='Record data during operation (default: True)')
    parser.add_argument('--no-record', dest='record', action='store_false',
                       help='Disable data recording')
    parser.add_argument('--recording_dir', type=str, default='data/recordings',
                       help='Directory for recordings')
    parser.add_argument('--use-cv', action='store_true',
                       help='Force CV-based perception (override default segmentation)')
    parser.add_argument(
        '--segmentation-checkpoint',
        type=str,
        default='data/segmentation_dataset/checkpoints/segnet_best.pt',
        help='Segmentation checkpoint path (default: segnet_best.pt)',
    )
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum number of frames to process')
    parser.add_argument('--duration', type=float, default=None,
                       help='Maximum duration in seconds (similar to ground_truth_follower.py)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration YAML file (default: config/av_stack_config.yaml)')
    
    args = parser.parse_args()
    
    use_segmentation = not args.use_cv
    if use_segmentation and not Path(args.segmentation_checkpoint).exists():
        parser.error(
            f"Segmentation checkpoint not found: {args.segmentation_checkpoint}. "
            "Provide --segmentation-checkpoint or run with --use-cv."
        )
    
    # Create and run AV stack
    av_stack = AVStack(
        bridge_url=args.bridge_url,
        model_path=args.model_path,
        record_data=args.record,
        config_path=args.config,
        recording_dir=args.recording_dir,
        use_segmentation=use_segmentation,
        segmentation_model_path=args.segmentation_checkpoint,
    )
    
    av_stack.run(max_frames=args.max_frames, duration=args.duration)


if __name__ == "__main__":
    main()

