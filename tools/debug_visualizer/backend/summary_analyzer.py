"""
Summary analyzer for recording-level metrics.
Extracted from analyze_drive_overall.py for use in debug visualizer.
"""

import math
import sys
import json
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np
import yaml
from scipy.fft import fft, fftfreq

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from trajectory.utils import smooth_curvature_distance

G_MPS2 = 9.80665
LOW_VISIBILITY_STALE_REASONS = {"left_lane_low_visibility", "right_lane_low_visibility"}


def _build_sign_mismatch_event(data: Dict, start_frame: int, end_frame: int) -> Dict:
    """Build a straight sign-mismatch event record with a classified root cause."""
    duration_frames = end_frame - start_frame + 1
    duration_seconds = safe_float(
        data['time'][end_frame] - data['time'][start_frame]
    ) if data.get('time') is not None and len(data['time']) > end_frame else 0.0

    err_series = data.get('total_error_scaled') if data.get('total_error_scaled') is not None else data.get('total_error')
    steer_series = data.get('steering')
    fb_series = data.get('feedback_steering')
    before_series = data.get('steering_before_limits')

    err_slice = err_series[start_frame:end_frame + 1] if err_series is not None else None
    steer_slice = steer_series[start_frame:end_frame + 1] if steer_series is not None else None
    fb_slice = fb_series[start_frame:end_frame + 1] if fb_series is not None else None
    before_slice = before_series[start_frame:end_frame + 1] if before_series is not None else None

    lanes = data.get('num_lanes_detected')
    lanes_slice = lanes[start_frame:end_frame + 1] if lanes is not None else None
    stale_ctrl = data.get('using_stale_perception')
    stale_ctrl_slice = stale_ctrl[start_frame:end_frame + 1] if stale_ctrl is not None else None
    stale_perc = data.get('using_stale_data')
    stale_perc_slice = stale_perc[start_frame:end_frame + 1] if stale_perc is not None else None
    override = data.get('straight_sign_flip_override_active')
    override_slice = override[start_frame:end_frame + 1] if override is not None else None

    stale_rate = 0.0
    if stale_ctrl_slice is not None:
        stale_rate = max(stale_rate, safe_float(float((stale_ctrl_slice > 0).mean()) * 100.0))
    if stale_perc_slice is not None:
        stale_rate = max(stale_rate, safe_float(float((stale_perc_slice > 0).mean()) * 100.0))
    lanes_min = int(lanes_slice.min()) if lanes_slice is not None and lanes_slice.size > 0 else None

    root_cause = "unknown"
    if stale_rate > 0.0 or (lanes_min is not None and lanes_min < 2):
        root_cause = "perception_stale_or_missing"
    elif (
        err_slice is not None and fb_slice is not None and before_slice is not None
        and np.sign(np.mean(err_slice)) == np.sign(np.mean(fb_slice))
        and np.sign(np.mean(before_slice)) != np.sign(np.mean(err_slice))
    ):
        root_cause = "rate_or_jerk_limit"
    elif (
        err_slice is not None and before_slice is not None and steer_slice is not None
        and np.sign(np.mean(before_slice)) == np.sign(np.mean(err_slice))
        and np.sign(np.mean(steer_slice)) != np.sign(np.mean(err_slice))
    ):
        root_cause = "smoothing"
    elif (
        err_slice is not None and fb_slice is not None
        and np.sign(np.mean(fb_slice)) != np.sign(np.mean(err_slice))
    ):
        root_cause = "error_computation_or_sign"

    override_rate = 0.0
    if override_slice is not None and override_slice.size > 0:
        override_rate = safe_float(float((override_slice > 0).mean()) * 100.0)

    return {
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "duration_frames": int(duration_frames),
        "duration_seconds": safe_float(duration_seconds),
        "root_cause": root_cause,
        "stale_rate_percent": safe_float(stale_rate),
        "lanes_min": lanes_min,
        "override_rate_percent": safe_float(override_rate),
    }


def safe_float(value, default=0.0):
    """Convert value to float, handling NaN and inf."""
    if value is None:
        return default
    if isinstance(value, (np.floating, np.integer)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return default
    return float(value)


def _load_config() -> dict:
    config_path = REPO_ROOT / "config" / "av_stack_config.yaml"
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _detect_control_mode(data):
    pp_geo = data.get('pp_geometric_steering')
    if pp_geo is not None and np.any(np.abs(pp_geo) > 1e-6):
        return 'pure_pursuit'
    return 'pid'

def _pp_feedback_gain(data):
    pp_fb = data.get('pp_feedback_steering')
    pp_geo = data.get('pp_geometric_steering')
    if pp_fb is not None and pp_geo is not None:
        mask = np.abs(pp_geo) > 0.01
        if np.any(mask):
            return float(np.mean(np.abs(pp_fb[mask]) / np.abs(pp_geo[mask])))
    return None

def _pp_mean_ld(data):
    ld = data.get('pp_lookahead_distance')
    if ld is not None:
        valid = ld[ld > 0.1]
        if len(valid) > 0:
            return float(np.mean(valid))
    return None

def _pp_jump_count(data):
    jc = data.get('pp_ref_jump_clamped')
    if jc is not None:
        return int(np.sum(jc > 0.5))
    return 0


def analyze_recording_summary(recording_path: Path, analyze_to_failure: bool = False, h5_file=None) -> Dict:
    """
    Analyze a recording and return summary metrics.
    
    Args:
        recording_path: Path to HDF5 recording file
        analyze_to_failure: If True, only analyze up to the point where car went out of lane and stayed out
        
    Returns:
        Dictionary with summary metrics and recommendations
    """
    data = {}
    
    try:
        with h5py.File(recording_path, 'r') as f:
            # Load timestamps
            if 'vehicle_state/timestamp' in f:
                data['timestamps'] = np.array(f['vehicle_state/timestamp'][:])
                data['speed'] = np.array(f['vehicle_state/speed'][:]) if 'vehicle_state/speed' in f else None
                data['speed_limit'] = (
                    np.array(f['vehicle_state/speed_limit'][:]) if 'vehicle_state/speed_limit' in f else None
                )
            elif 'vehicle/timestamps' in f:
                data['timestamps'] = np.array(f['vehicle/timestamps'][:])
                data['speed'] = np.array(f['vehicle/speed'][:]) if 'vehicle/speed' in f else None
                data['speed_limit'] = (
                    np.array(f['vehicle/speed_limit'][:]) if 'vehicle/speed_limit' in f else None
                )
            else:
                data['timestamps'] = np.array(f['control/timestamp'][:])
                data['speed'] = np.array(f['vehicle/speed'][:]) if 'vehicle/speed' in f else None
                data['speed_limit'] = (
                    np.array(f['vehicle/speed_limit'][:]) if 'vehicle/speed_limit' in f else None
                )
            
            # Control data
            data['steering'] = np.array(f['control/steering'][:])
            data['lateral_error'] = np.array(f['control/lateral_error'][:]) if 'control/lateral_error' in f else None
            data['heading_error'] = np.array(f['control/heading_error'][:]) if 'control/heading_error' in f else None
            data['total_error'] = np.array(f['control/total_error'][:]) if 'control/total_error' in f else None
            data['total_error_scaled'] = (
                np.array(f['control/total_error_scaled'][:])
                if 'control/total_error_scaled' in f else None
            )
            data['pid_integral'] = np.array(f['control/pid_integral'][:]) if 'control/pid_integral' in f else None
            data['emergency_stop'] = np.array(f['control/emergency_stop'][:]) if 'control/emergency_stop' in f else None
            data['path_curvature_input'] = (
                np.array(f['control/path_curvature_input'][:])
                if 'control/path_curvature_input' in f else None
            )
            data['is_straight'] = np.array(f['control/is_straight'][:]) if 'control/is_straight' in f else None
            data['straight_oscillation_rate'] = (
                np.array(f['control/straight_oscillation_rate'][:])
                if 'control/straight_oscillation_rate' in f else None
            )
            data['tuned_deadband'] = (
                np.array(f['control/tuned_deadband'][:]) if 'control/tuned_deadband' in f else None
            )
            data['tuned_error_smoothing_alpha'] = (
                np.array(f['control/tuned_error_smoothing_alpha'][:])
                if 'control/tuned_error_smoothing_alpha' in f else None
            )
            data['steering_before_limits'] = (
                np.array(f['control/steering_before_limits'][:])
                if 'control/steering_before_limits' in f else None
            )
            data['feedback_steering'] = (
                np.array(f['control/feedback_steering'][:])
                if 'control/feedback_steering' in f else None
            )
            data['feedforward_steering'] = (
                np.array(f['control/feedforward_steering'][:])
                if 'control/feedforward_steering' in f else None
            )
            data['using_stale_perception'] = (
                np.array(f['control/using_stale_perception'][:])
                if 'control/using_stale_perception' in f else None
            )
            data['straight_sign_flip_override_active'] = (
                np.array(f['control/straight_sign_flip_override_active'][:])
                if 'control/straight_sign_flip_override_active' in f else None
            )
            # Pure Pursuit telemetry
            data['pp_alpha'] = (
                np.array(f['control/pp_alpha'][:])
                if 'control/pp_alpha' in f else None
            )
            data['pp_lookahead_distance'] = (
                np.array(f['control/pp_lookahead_distance'][:])
                if 'control/pp_lookahead_distance' in f else None
            )
            data['pp_geometric_steering'] = (
                np.array(f['control/pp_geometric_steering'][:])
                if 'control/pp_geometric_steering' in f else None
            )
            data['pp_feedback_steering'] = (
                np.array(f['control/pp_feedback_steering'][:])
                if 'control/pp_feedback_steering' in f else None
            )
            data['pp_ref_jump_clamped'] = (
                np.array(f['control/pp_ref_jump_clamped'][:])
                if 'control/pp_ref_jump_clamped' in f else None
            )
            data['pp_stale_hold_active'] = (
                np.array(f['control/pp_stale_hold_active'][:])
                if 'control/pp_stale_hold_active' in f else None
            )
            
            # Trajectory data
            data['ref_x'] = np.array(f['trajectory/reference_point_x'][:]) if 'trajectory/reference_point_x' in f else None
            data['ref_heading'] = np.array(f['trajectory/reference_point_heading'][:]) if 'trajectory/reference_point_heading' in f else None
            data['ref_velocity'] = (
                np.array(f['trajectory/reference_point_velocity'][:])
                if 'trajectory/reference_point_velocity' in f else None
            )
            data['diag_heading_zero_gate_active'] = (
                np.array(f['trajectory/diag_heading_zero_gate_active'][:])
                if 'trajectory/diag_heading_zero_gate_active' in f else None
            )
            data['diag_ref_x_rate_limit_active'] = (
                np.array(f['trajectory/diag_ref_x_rate_limit_active'][:])
                if 'trajectory/diag_ref_x_rate_limit_active' in f else None
            )
            data['reference_point_curvature'] = (
                np.array(f['trajectory/reference_point_curvature'][:])
                if 'trajectory/reference_point_curvature' in f else None
            )
            
            # Perception data
            data['num_lanes_detected'] = np.array(f['perception/num_lanes_detected'][:]) if 'perception/num_lanes_detected' in f else None
            data['confidence'] = np.array(f['perception/confidence'][:]) if 'perception/confidence' in f else None
            data['using_stale_data'] = np.array(f['perception/using_stale_data'][:]) if 'perception/using_stale_data' in f else None
            data['fit_points_left'] = (
                f['perception/fit_points_left'][:] if 'perception/fit_points_left' in f else None
            )
            data['fit_points_right'] = (
                f['perception/fit_points_right'][:] if 'perception/fit_points_right' in f else None
            )
            data['image_width'] = int(f['camera/image_width'][0]) if 'camera/image_width' in f else 640
            data['stale_reason'] = None
            if 'perception/stale_reason' in f:
                stale_reasons = f['perception/stale_reason'][:]
                if len(stale_reasons) > 0:
                    data['stale_reason'] = [s.decode('utf-8') if isinstance(s, bytes) else s for s in stale_reasons]
            data['bad_events'] = None
            if 'perception/perception_bad_events' in f:
                bad_events_raw = f['perception/perception_bad_events'][:]
                data['bad_events'] = [
                    (b.decode('utf-8') if isinstance(b, bytes) else b) for b in bad_events_raw
                ]
            
            # Ground truth (if available)
            data['gt_center'] = np.array(f['ground_truth/lane_center_x'][:]) if 'ground_truth/lane_center_x' in f else None
            data['gt_left'] = np.array(f['ground_truth/left_lane_line_x'][:]) if 'ground_truth/left_lane_line_x' in f else None
            data['gt_right'] = np.array(f['ground_truth/right_lane_line_x'][:]) if 'ground_truth/right_lane_line_x' in f else None
            data['gt_path_curvature'] = (
                np.array(f['ground_truth/path_curvature'][:])
                if 'ground_truth/path_curvature' in f else None
            )

            # Road-frame metrics (if available)
            data['road_frame_lateral_offset'] = (
                np.array(f['vehicle/road_frame_lateral_offset'][:])
                if 'vehicle/road_frame_lateral_offset' in f else None
            )
            data['heading_delta_deg'] = (
                np.array(f['vehicle/heading_delta_deg'][:])
                if 'vehicle/heading_delta_deg' in f else None
            )
            data['road_frame_lane_center_offset'] = (
                np.array(f['vehicle/road_frame_lane_center_offset'][:])
                if 'vehicle/road_frame_lane_center_offset' in f else None
            )
            data['road_frame_lane_center_error'] = (
                np.array(f['vehicle/road_frame_lane_center_error'][:])
                if 'vehicle/road_frame_lane_center_error' in f else None
            )
            data['vehicle_frame_lookahead_offset'] = (
                np.array(f['vehicle/vehicle_frame_lookahead_offset'][:])
                if 'vehicle/vehicle_frame_lookahead_offset' in f else None
            )
            
            # Load lane positions for stability calculation
            data['left_lane_x'] = np.array(f['perception/left_lane_line_x'][:]) if 'perception/left_lane_line_x' in f else None
            data['right_lane_x'] = np.array(f['perception/right_lane_line_x'][:]) if 'perception/right_lane_line_x' in f else None
            data['perception_center'] = None
            if data['left_lane_x'] is not None and data['right_lane_x'] is not None:
                data['perception_center'] = (data['left_lane_x'] + data['right_lane_x']) / 2.0
            elif 'perception/left_lane_x' in f and 'perception/right_lane_x' in f:
                left_lane = np.array(f['perception/left_lane_x'][:])
                right_lane = np.array(f['perception/right_lane_x'][:])
                data['perception_center'] = (left_lane + right_lane) / 2.0
            
            # Unity timing (for hitch detection)
            data['unity_time'] = np.array(f['vehicle/unity_time'][:]) if 'vehicle/unity_time' in f else None
            
            # Calculate time axis
            if len(data['timestamps']) > 0:
                data['time'] = data['timestamps'] - data['timestamps'][0]
            else:
                data['time'] = np.arange(len(data['steering'])) * 0.033
                
    except Exception as e:
        return {"error": str(e)}
    
    # Detect if car went out of lane and stayed out (always detect, but only truncate if requested)
    # IMPORTANT: Use GROUND TRUTH lane center, not perception-based lateral_error
    # Ground truth shows the ACTUAL position of the car relative to the lane center
    # Perception-based error shows what the system THINKS the error is (can be wrong if perception fails)
    failure_frame = None
    emergency_stop_frame = None
    
    # Use ground truth boundaries if available, otherwise fall back to center/perception error
    if data.get('gt_left') is not None and data.get('gt_right') is not None:
        # Out of lane if car (x=0) is not between left/right boundaries.
        gt_left = data['gt_left']
        gt_right = data['gt_right']
        out_of_lane_mask = ~((gt_left <= 0) & (0 <= gt_right))
        error_data = np.where(
            gt_left > 0,
            np.abs(gt_left),
            np.where(gt_right < 0, np.abs(gt_right), 0.0),
        )
        error_source = "ground_truth_boundaries"
    elif data['gt_center'] is not None:
        # Ground truth lane_center_x: position of lane center relative to vehicle center (in vehicle frame)
        # Lateral error = -gt_center_x (negate because gt_center is position, not error)
        gt_lateral_error = -data['gt_center']
        error_data = gt_lateral_error
        error_source = "ground_truth"
        out_of_lane_mask = np.abs(error_data) > 0.5
    elif data['lateral_error'] is not None:
        error_data = data['lateral_error']
        error_source = "perception"
        out_of_lane_mask = np.abs(error_data) > 0.5
    else:
        error_data = None
        out_of_lane_mask = None
    
    if error_data is not None and out_of_lane_mask is not None:
        # Define thresholds
        out_of_lane_threshold = 0.5  # meters (half a meter is a reasonable "out of lane" threshold)
        catastrophic_threshold = 2.0  # meters (catastrophic failure - car is way out)
        min_consecutive_out = 10  # frames (must stay out for this long to be considered "stayed out")
        
        # Strategy: Find the last recovery before a catastrophic failure, then find when error
        # first exceeded threshold after that recovery. This identifies the "start of final failure sequence"
        # rather than just the first time error exceeded threshold in the entire recording.
        
        # First, find if there's a catastrophic failure
        catastrophic_frame = None
        for i, error in enumerate(error_data):
            if abs(error) > catastrophic_threshold:
                catastrophic_frame = i
                break
        
        if catastrophic_frame is not None:
            # Find last recovery (error < 0.5m) before catastrophic failure
            last_recovery_frame = None
            for i in range(catastrophic_frame - 1, -1, -1):
                if abs(error_data[i]) < out_of_lane_threshold:
                    last_recovery_frame = i
                    break
            
            # If we found a recovery, find when error first exceeded threshold after that recovery
            if last_recovery_frame is not None:
                consecutive_out_frames = 0
                for i in range(last_recovery_frame + 1, len(error_data)):
                    if out_of_lane_mask[i]:
                        consecutive_out_frames += 1
                        # If we've been out for enough frames, this is the failure point
                        if consecutive_out_frames >= min_consecutive_out and failure_frame is None:
                            failure_frame = i - (min_consecutive_out - 1)  # Go back to when we first went out
                            break
                    else:
                        consecutive_out_frames = 0
        
        # Fallback: If no catastrophic failure, use simple detection
        if failure_frame is None:
            consecutive_out_frames = 0
            for i, is_out in enumerate(out_of_lane_mask):
                if is_out:
                    consecutive_out_frames += 1
                    if consecutive_out_frames >= min_consecutive_out and failure_frame is None:
                        failure_frame = i - (min_consecutive_out - 1)
                        break
                else:
                    consecutive_out_frames = 0
        
    if data.get('emergency_stop') is not None:
        stop_indices = np.where(data['emergency_stop'] > 0)[0]
        if stop_indices.size > 0:
            emergency_stop_frame = int(stop_indices[0])

    if emergency_stop_frame is not None:
        if failure_frame is None or emergency_stop_frame < failure_frame:
            failure_frame = emergency_stop_frame
            error_source = "emergency_stop"

    # If we found a failure point AND analyze_to_failure is True, truncate data to that point
    if failure_frame is not None and analyze_to_failure:
        # Truncate all arrays to failure_frame
        for key in data:
            if isinstance(data[key], np.ndarray) and len(data[key]) > failure_frame:
                data[key] = data[key][:failure_frame]
    
    # Calculate metrics
    n_frames = len(data['steering'])
    dt = safe_float(np.mean(np.diff(data['time'])) if len(data['time']) > 1 else 0.033)
    duration = safe_float(data['time'][-1] if len(data['time']) > 0 else n_frames * dt)

    # Unity timing gaps (hitch detection)
    unity_time_gap_max = 0.0
    unity_time_gap_count = 0
    if data.get('unity_time') is not None and len(data['unity_time']) > 1:
        unity_time_diffs = np.diff(data['unity_time'])
        if unity_time_diffs.size > 0:
            unity_time_gap_max = safe_float(np.max(unity_time_diffs))
            unity_time_gap_count = int(np.sum(unity_time_diffs > 0.2))
    
    # 1. PATH TRACKING
    lateral_error_rmse = safe_float(np.sqrt(np.mean(data['lateral_error']**2)) if data['lateral_error'] is not None and len(data['lateral_error']) > 0 else 0.0)
    lateral_error_mean = safe_float(np.mean(np.abs(data['lateral_error'])) if data['lateral_error'] is not None and len(data['lateral_error']) > 0 else 0.0)
    lateral_error_max = safe_float(np.max(np.abs(data['lateral_error'])) if data['lateral_error'] is not None and len(data['lateral_error']) > 0 else 0.0)
    lateral_error_p95 = safe_float(np.percentile(np.abs(data['lateral_error']), 95) if data['lateral_error'] is not None and len(data['lateral_error']) > 0 else 0.0)
    
    heading_error_rmse = safe_float(np.sqrt(np.mean(data['heading_error']**2)) if data['heading_error'] is not None and len(data['heading_error']) > 0 else 0.0)
    heading_error_max = safe_float(np.max(np.abs(data['heading_error'])) if data['heading_error'] is not None and len(data['heading_error']) > 0 else 0.0)
    
    # Time in Lane:
    # Primary: Use ground truth boundaries if available (car at x=0 is in lane if left <= 0 <= right)
    # Secondary: Centeredness within ±0.5m (for tuning)
    time_in_lane_centered = safe_float(
        np.sum(np.abs(data['lateral_error']) < 0.5) / n_frames * 100
        if data['lateral_error'] is not None and n_frames > 0 else 0.0
    )
    if data['gt_left'] is not None and data['gt_right'] is not None:
        in_lane_mask = (data['gt_left'] <= 0) & (0 <= data['gt_right'])
        time_in_lane = safe_float(np.sum(in_lane_mask) / n_frames * 100 if n_frames > 0 else 0.0)
    elif data['lateral_error'] is not None:
        time_in_lane = time_in_lane_centered
    else:
        time_in_lane = 0.0
    
    # 2. CONTROL SMOOTHNESS
    steering_rate = np.diff(data['steering']) / np.diff(data['time']) if len(data['steering']) > 1 and len(data['time']) > 1 else np.array([0.0])
    steering_jerk = np.diff(steering_rate) / np.diff(data['time'][1:]) if len(steering_rate) > 1 and len(data['time']) > 2 else np.array([0.0])
    steering_jerk_max = safe_float(np.max(np.abs(steering_jerk)) if len(steering_jerk) > 0 else 0.0)
    steering_rate_max = safe_float(np.max(np.abs(steering_rate)) if len(steering_rate) > 0 else 0.0)
    
    steering_std = safe_float(np.std(data['steering']) if len(data['steering']) > 0 else 0.0)
    steering_smoothness = safe_float(1.0 / (steering_std + 1e-6))

    # Straight-line stability (uses controller diagnostics if available)
    straight_fraction = 0.0
    straight_oscillation_mean = 0.0
    straight_oscillation_max = 0.0
    tuned_deadband_mean = 0.0
    tuned_deadband_max = 0.0
    tuned_smoothing_mean = 0.0
    straight_sign_mismatch_rate = 0.0
    straight_sign_mismatch_events = 0
    straight_sign_mismatch_frames = 0
    straight_sign_mismatch_events_list = []
    if data.get('is_straight') is not None and len(data['is_straight']) > 0:
        straight_mask = data['is_straight'][:n_frames] > 0
        straight_fraction = safe_float(np.sum(straight_mask) / max(1, n_frames) * 100.0)
        if data.get('straight_oscillation_rate') is not None and len(data['straight_oscillation_rate']) > 0:
            straight_rates = data['straight_oscillation_rate'][:n_frames]
            if np.any(straight_mask):
                straight_rates = straight_rates[straight_mask]
            straight_oscillation_mean = safe_float(np.mean(straight_rates)) if straight_rates.size > 0 else 0.0
            straight_oscillation_max = safe_float(np.max(straight_rates)) if straight_rates.size > 0 else 0.0
        if data.get('tuned_deadband') is not None and len(data['tuned_deadband']) > 0:
            tuned_deadband = data['tuned_deadband'][:n_frames]
            tuned_deadband_mean = safe_float(np.mean(tuned_deadband))
            tuned_deadband_max = safe_float(np.max(tuned_deadband))
        if data.get('tuned_error_smoothing_alpha') is not None and len(data['tuned_error_smoothing_alpha']) > 0:
            tuned_smoothing = data['tuned_error_smoothing_alpha'][:n_frames]
            tuned_smoothing_mean = safe_float(np.mean(tuned_smoothing))

        # Detect sign-mismatch events on straights (steering fighting scaled error)
        error_series = data.get('total_error_scaled') if data.get('total_error_scaled') is not None else data.get('total_error')
        if error_series is not None and len(error_series) > 0:
            error_series = error_series[:n_frames]
            steering_series = data['steering'][:n_frames]
            valid_mask = straight_mask & (np.abs(error_series) >= 0.02) & (np.abs(steering_series) >= 0.02)
            mismatch_mask = valid_mask & (np.sign(error_series) != np.sign(steering_series))
            straight_sign_mismatch_frames = int(np.sum(mismatch_mask))
            denom = int(np.sum(valid_mask))
            if denom > 0:
                straight_sign_mismatch_rate = safe_float(straight_sign_mismatch_frames / denom * 100.0)

            # Count contiguous mismatch events (>=3 frames) and classify root cause.
            min_event_len = 3
            current = 0
            event_start = None
            for idx, flag in enumerate(mismatch_mask):
                if flag:
                    if current == 0:
                        event_start = idx
                    current += 1
                else:
                    if current >= min_event_len and event_start is not None:
                        event_end = idx - 1
                        straight_sign_mismatch_events += 1
                        straight_sign_mismatch_events_list.append(
                            _build_sign_mismatch_event(
                                data=data,
                                start_frame=event_start,
                                end_frame=event_end,
                            )
                        )
                    current = 0
                    event_start = None
            if current >= min_event_len and event_start is not None:
                event_end = len(mismatch_mask) - 1
                straight_sign_mismatch_events += 1
                straight_sign_mismatch_events_list.append(
                    _build_sign_mismatch_event(
                        data=data,
                        start_frame=event_start,
                        end_frame=event_end,
                    )
                )
    
    # Oscillation frequency
    oscillation_frequency = 0.0
    if data['lateral_error'] is not None and len(data['lateral_error']) > 10:
        try:
            error_centered = data['lateral_error'] - np.mean(data['lateral_error'])
            fft_vals = fft(error_centered)
            fft_freqs = fftfreq(len(error_centered), dt)
            positive_freqs = fft_freqs[:len(fft_freqs)//2]
            positive_fft = np.abs(fft_vals[:len(fft_vals)//2])
            if len(positive_fft) > 1:
                dominant_idx = np.argmax(positive_fft[1:]) + 1
                oscillation_frequency = safe_float(positive_freqs[dominant_idx] if dominant_idx < len(positive_freqs) else 0.0)
        except:
            oscillation_frequency = 0.0
    oscillation_frequency = safe_float(oscillation_frequency)

    # 2.5 SPEED CONTROL + COMFORT
    speed_error_rmse = 0.0
    speed_error_mean = 0.0
    speed_error_max = 0.0
    speed_overspeed_rate = 0.0
    speed_limit_zero_rate = 0.0
    speed_surge_count = 0
    speed_surge_avg_drop = 0.0
    speed_surge_p95_drop = 0.0
    if data.get('speed') is not None and data.get('ref_velocity') is not None:
        n_speed = min(len(data['speed']), len(data['ref_velocity']))
        if n_speed > 0:
            speed = data['speed'][:n_speed]
            ref_speed = data['ref_velocity'][:n_speed]
            speed_error = speed - ref_speed
            speed_error_rmse = safe_float(np.sqrt(np.mean(speed_error ** 2)))
            speed_error_mean = safe_float(np.mean(np.abs(speed_error)))
            speed_error_max = safe_float(np.max(np.abs(speed_error)))
            overspeed_threshold = 0.5
            speed_overspeed_rate = safe_float(np.sum(speed_error > overspeed_threshold) / n_speed * 100)
    if data.get('speed_limit') is not None and len(data['speed_limit']) > 0:
        speed_limit_zero_rate = safe_float(
            np.sum(data['speed_limit'] <= 0.01) / len(data['speed_limit']) * 100.0
        )

    acceleration_mean = 0.0
    acceleration_max = 0.0
    acceleration_p95 = 0.0
    jerk_mean = 0.0
    jerk_max = 0.0
    jerk_p95 = 0.0
    acceleration_mean_filtered = 0.0
    acceleration_max_filtered = 0.0
    acceleration_p95_filtered = 0.0
    jerk_mean_filtered = 0.0
    jerk_max_filtered = 0.0
    jerk_p95_filtered = 0.0
    lateral_accel_p95 = 0.0
    lateral_jerk_p95 = 0.0
    lateral_jerk_max = 0.0
    config = _load_config()
    traj_cfg = config.get('trajectory', {})
    perception_cfg = config.get('perception', {})
    config_summary = {
        "camera_fov": safe_float(traj_cfg.get("camera_fov", 0.0)),
        "camera_height": safe_float(traj_cfg.get("camera_height", 0.0)),
        "segmentation_fit_min_row_ratio": safe_float(
            perception_cfg.get("segmentation_fit_min_row_ratio", 0.0)
        ),
        "segmentation_fit_max_row_ratio": safe_float(
            perception_cfg.get("segmentation_fit_max_row_ratio", 0.0)
        ),
    }
    curvature_smoothing_enabled = bool(traj_cfg.get('curvature_smoothing_enabled', False))
    curvature_window_m = float(traj_cfg.get('curvature_smoothing_window_m', 12.0))
    curvature_min_speed = float(traj_cfg.get('curvature_smoothing_min_speed', 2.0))

    if data.get('speed') is not None:
        curvature_series = None
        if data.get('gt_path_curvature') is not None:
            curvature_series = data['gt_path_curvature']
        elif data.get('path_curvature_input') is not None:
            curvature_series = data['path_curvature_input']
        if curvature_series is not None:
            n_surge = min(len(data['speed']), len(curvature_series))
            speed_series = data['speed'][:n_surge]
            curvature_series = curvature_series[:n_surge]
            straight_threshold = float(traj_cfg.get('straight_speed_smoothing_curvature_threshold', 0.003))
            min_drop = float(traj_cfg.get('speed_surge_min_drop', 1.0))
            straight_mask = np.abs(curvature_series) <= straight_threshold
            drops = []
            i = 1
            while i < len(speed_series) - 1:
                if not straight_mask[i]:
                    i += 1
                    continue
                if speed_series[i] > speed_series[i - 1] and speed_series[i] >= speed_series[i + 1]:
                    max_idx = i
                    j = i + 1
                    while j < len(speed_series) - 1 and straight_mask[j]:
                        if speed_series[j] <= speed_series[j - 1] and speed_series[j] < speed_series[j + 1]:
                            drop = float(speed_series[max_idx] - speed_series[j])
                            if drop >= min_drop:
                                drops.append(drop)
                            break
                        j += 1
                    i = j
                else:
                    i += 1
            if drops:
                speed_surge_count = len(drops)
                speed_surge_avg_drop = safe_float(np.mean(drops))
                speed_surge_p95_drop = safe_float(np.percentile(drops, 95))

    if data.get('speed') is not None and len(data['speed']) > 1 and len(data['time']) > 1:
        dt_series = np.diff(data['time'])
        dt_series[dt_series <= 0] = dt
        acceleration = np.diff(data['speed']) / dt_series
        if acceleration.size > 0:
            abs_accel = np.abs(acceleration)
            acceleration_mean = safe_float(np.mean(abs_accel))
            acceleration_max = safe_float(np.max(abs_accel))
            acceleration_p95 = safe_float(np.percentile(abs_accel, 95))
        if acceleration.size > 1:
            jerk = np.diff(acceleration) / dt_series[1:]
            if jerk.size > 0:
                abs_jerk = np.abs(jerk)
                jerk_mean = safe_float(np.mean(abs_jerk))
                jerk_max = safe_float(np.max(abs_jerk))
                jerk_p95 = safe_float(np.percentile(abs_jerk, 95))
        # Filtered speed for comfort metrics (reduce derivative noise)
        alpha = 0.7
        filtered_speed = np.empty_like(data['speed'])
        filtered_speed[0] = data['speed'][0]
        for i in range(1, len(data['speed'])):
            filtered_speed[i] = alpha * filtered_speed[i - 1] + (1.0 - alpha) * data['speed'][i]
        filtered_accel = np.diff(filtered_speed) / dt_series
        if filtered_accel.size > 0:
            abs_f_accel = np.abs(filtered_accel)
            acceleration_mean_filtered = safe_float(np.mean(abs_f_accel))
            acceleration_max_filtered = safe_float(np.max(abs_f_accel))
            acceleration_p95_filtered = safe_float(np.percentile(abs_f_accel, 95))
        if filtered_accel.size > 1:
            filtered_jerk = np.diff(filtered_accel) / dt_series[1:]
            if filtered_jerk.size > 0:
                abs_f_jerk = np.abs(filtered_jerk)
                jerk_mean_filtered = safe_float(np.mean(abs_f_jerk))
                jerk_max_filtered = safe_float(np.max(abs_f_jerk))
                jerk_p95_filtered = safe_float(np.percentile(abs_f_jerk, 95))
        curvature = None
        if data.get('gt_path_curvature') is not None:
            curvature = data['gt_path_curvature']
        elif data.get('path_curvature_input') is not None:
            curvature = data['path_curvature_input']
        if curvature is not None:
            n_lat = min(len(curvature), len(data['speed']))
            if n_lat > 1:
                curvature_source = curvature[:n_lat]
                if curvature_smoothing_enabled:
                    curvature_source = np.array(
                        smooth_curvature_distance(
                            curvature_source,
                            data['speed'][:n_lat],
                            data['time'][:n_lat],
                            curvature_window_m,
                            curvature_min_speed,
                        )
                    )
                lat_accel = (data['speed'][:n_lat] ** 2) * curvature_source
                abs_lat_accel = np.abs(lat_accel)
                lateral_accel_p95 = safe_float(np.percentile(abs_lat_accel, 95))
                lat_dt = np.diff(data['time'][:n_lat])
                lat_dt[lat_dt <= 0] = dt
                lat_jerk = np.diff(lat_accel) / lat_dt
                if lat_jerk.size > 0:
                    abs_lat_jerk = np.abs(lat_jerk)
                    lateral_jerk_p95 = safe_float(np.percentile(abs_lat_jerk, 95))
                    lateral_jerk_max = safe_float(np.max(abs_lat_jerk))
    
    # 3. PERCEPTION QUALITY
    lane_detection_rate = safe_float(np.sum(data['num_lanes_detected'] >= 2) / n_frames * 100 if data['num_lanes_detected'] is not None and n_frames > 0 else 0.0)
    perception_confidence_mean = safe_float(np.mean(data['confidence']) if data['confidence'] is not None and len(data['confidence']) > 0 else 0.0)
    single_lane_rate = safe_float(np.sum(data['num_lanes_detected'] < 2) / n_frames * 100 if data['num_lanes_detected'] is not None and n_frames > 0 else 0.0)
    
    # Count different types of perception issues
    perception_jumps_detected = 0
    perception_instability_detected = 0
    perception_extreme_coeffs_detected = 0
    perception_invalid_width_detected = 0
    
    if data['stale_reason'] is not None:
        for r in data['stale_reason']:
            if r:
                r_str = str(r).lower()
                if 'jump' in r_str:
                    perception_jumps_detected += 1
                elif 'instability' in r_str:
                    perception_instability_detected += 1
                elif 'extreme' in r_str or 'coefficient' in r_str:
                    perception_extreme_coeffs_detected += 1
                elif 'width' in r_str or 'invalid' in r_str:
                    perception_invalid_width_detected += 1
    
    stale_perception_rate = safe_float(
        np.sum(data['using_stale_data']) / n_frames * 100
        if data['using_stale_data'] is not None and n_frames > 0
        else 0.0
    )
    stale_raw_rate = stale_perception_rate
    stale_hard_rate = 0.0
    stale_fallback_visibility_rate = 0.0
    if data['using_stale_data'] is not None and n_frames > 0:
        stale_flags = np.asarray(data['using_stale_data'][:n_frames]).astype(bool)
        stale_reasons = data['stale_reason'] if data['stale_reason'] is not None else []
        hard_count = 0
        fallback_count = 0
        for i, is_stale in enumerate(stale_flags):
            if not is_stale:
                continue
            reason = ""
            if i < len(stale_reasons) and stale_reasons[i] is not None:
                reason = str(stale_reasons[i]).strip().lower()
            if reason in LOW_VISIBILITY_STALE_REASONS:
                fallback_count += 1
            else:
                hard_count += 1
        stale_hard_rate = safe_float(hard_count / n_frames * 100)
        stale_fallback_visibility_rate = safe_float(fallback_count / n_frames * 100)
    
    # NEW: Calculate perception stability metrics (lane position/width variance)
    # High variance indicates perception instability even if not caught by stale_data
    perception_stability_score = 100.0  # Start at 100, penalize for instability
    lane_position_variance = 0.0
    lane_width_variance = 0.0
    lane_line_jitter_p95 = 0.0
    lane_line_jitter_p99 = 0.0
    reference_jitter_p95 = 0.0
    reference_jitter_p99 = 0.0
    right_lane_low_visibility_rate = 0.0
    right_lane_edge_contact_rate = 0.0
    left_lane_low_visibility_rate = 0.0
    
    if data['left_lane_x'] is not None and data['right_lane_x'] is not None:
        left_lanes = data['left_lane_x'][:n_frames]
        right_lanes = data['right_lane_x'][:n_frames]
        
        # Calculate lane center and width for each frame
        lane_centers = (left_lanes + right_lanes) / 2.0
        lane_widths = right_lanes - left_lanes
        
        # Filter out invalid values (NaN, inf, or extreme values)
        valid_mask = (np.isfinite(lane_centers) & np.isfinite(lane_widths) & 
                     (lane_widths > 2.0) & (lane_widths < 10.0) &  # Valid width range
                     (np.abs(lane_centers) < 5.0))  # Reasonable center position
        
        if np.sum(valid_mask) > 10:  # Need at least 10 valid frames
            valid_centers = lane_centers[valid_mask]
            valid_widths = lane_widths[valid_mask]
            
            # Calculate variance (higher = less stable)
            lane_position_variance = safe_float(np.var(valid_centers))
            lane_width_variance = safe_float(np.var(valid_widths))

            # Lane-line jitter: frame-to-frame movement (max of left/right)
            if valid_centers.size > 1:
                valid_left = left_lanes[valid_mask]
                valid_right = right_lanes[valid_mask]
                left_jitter = np.abs(np.diff(valid_left))
                right_jitter = np.abs(np.diff(valid_right))
                jitter = np.maximum(left_jitter, right_jitter)
                if jitter.size > 0:
                    lane_line_jitter_p95 = safe_float(np.percentile(jitter, 95))
                    lane_line_jitter_p99 = safe_float(np.percentile(jitter, 99))

        # Lane visibility based on fit points near image edges or too few points
        if data.get('fit_points_left') is not None and data.get('fit_points_right') is not None:
            min_points = 6
            edge_margin = 12
            width = data.get('image_width', 640)
            left_low = 0
            right_low = 0
            right_edge_contact = 0

            def parse_points(raw) -> list:
                if raw is None:
                    return []
                try:
                    s = raw.decode('utf-8') if isinstance(raw, (bytes, bytearray, np.bytes_)) else str(raw)
                    return json.loads(s)
                except Exception:
                    return []

            for i in range(min(n_frames, len(data['fit_points_left']))):
                left_pts = parse_points(data['fit_points_left'][i])
                right_pts = parse_points(data['fit_points_right'][i])

                left_xs = [p[0] for p in left_pts if isinstance(p, (list, tuple)) and len(p) >= 2]
                right_xs = [p[0] for p in right_pts if isinstance(p, (list, tuple)) and len(p) >= 2]

                if len(left_xs) < min_points or (left_xs and min(left_xs) < edge_margin):
                    left_low += 1
                right_low_flag = False
                if data.get('num_lanes_detected') is not None and i < len(data['num_lanes_detected']) and data['num_lanes_detected'][i] < 2:
                    right_low_flag = True
                if len(right_xs) < min_points:
                    right_low_flag = True
                if right_xs and max(right_xs) > (width - edge_margin):
                    right_edge_contact += 1
                if right_low_flag:
                    right_low += 1

            right_lane_low_visibility_rate = safe_float(right_low / n_frames * 100 if n_frames > 0 else 0.0)
            right_lane_edge_contact_rate = safe_float(right_edge_contact / n_frames * 100 if n_frames > 0 else 0.0)
            left_lane_low_visibility_rate = safe_float(left_low / n_frames * 100 if n_frames > 0 else 0.0)
            
            # Penalize high variance (instability)
            # Position variance > 0.1m² indicates instability
            if lane_position_variance > 0.1:
                perception_stability_score -= min(30, (lane_position_variance - 0.1) * 100)
            
            # Width variance > 0.2m² indicates instability
            if lane_width_variance > 0.2:
                perception_stability_score -= min(30, (lane_width_variance - 0.2) * 50)
            
            perception_stability_score = safe_float(max(0, perception_stability_score))
    
    # Reference jitter (frame-to-frame movement)
    if data.get('ref_x') is not None and len(data['ref_x']) > 1:
        ref_jitter = np.abs(np.diff(data['ref_x'][:n_frames]))
        if ref_jitter.size > 0:
            reference_jitter_p95 = safe_float(np.percentile(ref_jitter, 95))
            reference_jitter_p99 = safe_float(np.percentile(ref_jitter, 99))

    # Also penalize for detected instability events
    if perception_instability_detected > 0:
        instability_rate = (perception_instability_detected / n_frames) * 100
        perception_stability_score -= min(40, instability_rate * 2)  # -2 points per % of frames with instability
        perception_stability_score = safe_float(max(0, perception_stability_score))

    # Penalize high lane-line jitter even if not flagged as instability
    if lane_line_jitter_p95 > 0.6:
        perception_stability_score -= min(20, (lane_line_jitter_p95 - 0.6) * 40)
        perception_stability_score = safe_float(max(0, perception_stability_score))
    if right_lane_low_visibility_rate > 10.0:
        perception_stability_score -= min(15, (right_lane_low_visibility_rate - 10.0) * 0.5)
        perception_stability_score = safe_float(max(0, perception_stability_score))
    
    # 4. TRAJECTORY QUALITY
    trajectory_availability = safe_float(np.sum(~np.isnan(data['ref_x'])) / n_frames * 100 if data['ref_x'] is not None and n_frames > 0 else 0.0)
    
    ref_point_accuracy_rmse = 0.0
    if data['ref_x'] is not None and data['gt_center'] is not None:
        valid_mask = ~np.isnan(data['ref_x']) & ~np.isnan(data['gt_center'])
        if np.sum(valid_mask) > 0:
            errors = data['ref_x'][valid_mask] - data['gt_center'][valid_mask]
            ref_point_accuracy_rmse = safe_float(np.sqrt(np.mean(errors**2)) if len(errors) > 0 else 0.0)
    
    # 5. SYSTEM HEALTH
    pid_integral_max = safe_float(np.max(np.abs(data['pid_integral'])) if data['pid_integral'] is not None and len(data['pid_integral']) > 0 else 0.0)
    
    # 6. SAFETY
    # Out-of-lane detection: Check if car is actually outside lane boundaries
    # Car is at x=0 in vehicle frame, so check if 0 is between left_lane_line_x and right_lane_line_x
    if data['gt_left'] is not None and data['gt_right'] is not None:
        # Use ground truth lane boundaries (most accurate)
        gt_left = data['gt_left']
        gt_right = data['gt_right']
        # Car is at x=0 in vehicle frame
        # Out of lane if: NOT (left <= 0 <= right)
        # This means: car is OUT if (left > 0) OR (right < 0)
        out_of_lane_mask = ~((gt_left <= 0) & (0 <= gt_right))
        # Use sustained out-of-lane mask to avoid single-frame noise
        min_consecutive_out = 10
        sustained_mask = np.zeros_like(out_of_lane_mask, dtype=bool)
        run_start = None
        run_len = 0
        for i, is_out in enumerate(out_of_lane_mask):
            if is_out:
                if run_start is None:
                    run_start = i
                run_len += 1
            else:
                if run_len >= min_consecutive_out and run_start is not None:
                    sustained_mask[run_start:i] = True
                run_start = None
                run_len = 0
        if run_len >= min_consecutive_out and run_start is not None:
            sustained_mask[run_start:len(out_of_lane_mask)] = True

        out_of_lane_time = safe_float(
            np.sum(sustained_mask) / n_frames * 100 if n_frames > 0 else 0.0
        )
        error_source = "ground_truth_boundaries"
        
        # For error calculation in events list, use distance outside lane
        def get_error_outside_lane(i):
            if i >= len(gt_left) or i >= len(gt_right):
                return 0.0
            if gt_left[i] > 0:
                # Car is left of left lane line
                return abs(gt_left[i])
            elif gt_right[i] < 0:
                # Car is right of right lane line
                return abs(gt_right[i])
            else:
                # Car is in lane
                return 0.0
        
        error_data_for_events = np.array([get_error_outside_lane(i) for i in range(len(out_of_lane_mask))])
    elif error_data is not None:
        # Fallback: Use distance from lane center with adaptive threshold
        # If we have lane width info, use 50% of half-lane-width as threshold
        # Otherwise use 1.75m (half of typical 3.5m lane)
        if data['gt_left'] is not None and data['gt_right'] is not None:
            gt_left = data['gt_left']
            gt_right = data['gt_right']
            # Calculate lane width for each frame and use 50% of half-width as threshold
            lane_widths = gt_right - gt_left
            # Use median lane width to avoid noise
            median_lane_width = safe_float(np.median(lane_widths) if len(lane_widths) > 0 else 3.5)
            threshold = median_lane_width * 0.5 * 0.5  # 50% of half-lane-width
        else:
            threshold = 1.75  # Half of typical 3.5m lane width
        
        out_of_lane_mask = np.abs(error_data) > threshold
        # Use sustained out-of-lane mask to avoid single-frame noise
        min_consecutive_out = 10
        sustained_mask = np.zeros_like(out_of_lane_mask, dtype=bool)
        run_start = None
        run_len = 0
        for i, is_out in enumerate(out_of_lane_mask):
            if is_out:
                if run_start is None:
                    run_start = i
                run_len += 1
            else:
                if run_len >= min_consecutive_out and run_start is not None:
                    sustained_mask[run_start:i] = True
                run_start = None
                run_len = 0
        if run_len >= min_consecutive_out and run_start is not None:
            sustained_mask[run_start:len(out_of_lane_mask)] = True

        out_of_lane_time = safe_float(
            np.sum(sustained_mask) / n_frames * 100 if n_frames > 0 else 0.0
        )
        error_data_for_events = np.abs(error_data)
        error_source = error_source if 'error_source' in locals() else "perception"
    else:
        out_of_lane_events_count = 0
        out_of_lane_time = 0.0
        out_of_lane_events_list = []
        error_data_for_events = None
    
    # Detect individual out-of-lane events (consecutive frames out of lane)
    if error_data_for_events is not None:
        out_of_lane_events_list = []
        in_event = False
        event_start = None
        event_max_error = 0.0
        
        min_consecutive_out = 10
        for i, is_out in enumerate(out_of_lane_mask):
            if is_out:
                if not in_event:
                    # Start of new event
                    in_event = True
                    event_start = i
                    event_max_error = error_data_for_events[i]
                else:
                    # Continue event, update max error
                    event_max_error = max(event_max_error, error_data_for_events[i])
            else:
                if in_event:
                    # End of event
                    event_duration = i - event_start
                    if event_duration >= min_consecutive_out:
                        out_of_lane_events_list.append({
                            'start_frame': int(event_start),
                            'end_frame': int(i - 1),
                            'duration_frames': int(event_duration),
                            'duration_seconds': safe_float(event_duration * dt),
                            'max_error': safe_float(event_max_error),
                            'error_source': error_source
                        })
                    in_event = False
                    event_start = None
                    event_max_error = 0.0
        
        # Handle event that extends to end of data
        if in_event:
            event_duration = len(out_of_lane_mask) - event_start
            if event_duration >= min_consecutive_out:
                out_of_lane_events_list.append({
                    'start_frame': int(event_start),
                    'end_frame': int(len(out_of_lane_mask) - 1),
                    'duration_frames': int(event_duration),
                    'duration_seconds': safe_float(event_duration * dt),
                    'max_error': safe_float(event_max_error),
                    'error_source': error_source
                })
    else:
        out_of_lane_events_list = []
    
    out_of_lane_events = len(out_of_lane_events_list)

    # 7. TURN BIAS (road-frame offsets)
    turn_bias = None
    alignment_summary = None
    if data.get('road_frame_lateral_offset') is not None:
        curvature_source = None
        if data.get('gt_path_curvature') is not None:
            curvature_source = data['gt_path_curvature']
        elif data.get('path_curvature_input') is not None:
            curvature_source = data['path_curvature_input']

        if curvature_source is not None:
            offset = data['road_frame_lateral_offset']
            heading_delta = data.get('heading_delta_deg')
            steering = data.get('steering')
            gt_center = data.get('gt_center')
            p_center = data.get('perception_center')

            n_bias = len(offset)
            n_bias = min(n_bias, len(curvature_source))
            if steering is not None:
                n_bias = min(n_bias, len(steering))
            if heading_delta is not None:
                n_bias = min(n_bias, len(heading_delta))
            if gt_center is not None:
                n_bias = min(n_bias, len(gt_center))
            if p_center is not None:
                n_bias = min(n_bias, len(p_center))

            offset = offset[:n_bias]
            curvature_source = curvature_source[:n_bias]
            if steering is not None:
                steering = steering[:n_bias]
            if heading_delta is not None:
                heading_delta = heading_delta[:n_bias]
            if gt_center is not None:
                gt_center = gt_center[:n_bias]
            if p_center is not None:
                p_center = p_center[:n_bias]

            curve_threshold = 0.002
            curve_mask = np.abs(curvature_source) >= curve_threshold
            left_mask = curve_mask & (curvature_source > 0)
            right_mask = curve_mask & (curvature_source < 0)

            def summarize_mask(mask):
                if not np.any(mask):
                    return {
                        "frames": 0,
                        "mean": 0.0,
                        "p95_abs": 0.0,
                        "max_abs": 0.0,
                        "outside_rate": 0.0,
                    }
                vals = offset[mask]
                outside = (curvature_source[mask] > 0) & (vals > 0)
                outside |= (curvature_source[mask] < 0) & (vals < 0)
                return {
                    "frames": int(mask.sum()),
                    "mean": safe_float(np.mean(vals)),
                    "p95_abs": safe_float(np.percentile(np.abs(vals), 95)),
                    "max_abs": safe_float(np.max(np.abs(vals))),
                    "outside_rate": safe_float(np.mean(outside) * 100.0),
                }

            def top_turn_frames(mask, max_items=8, min_separation=12):
                output = []
                if not np.any(mask):
                    return output
                candidates = np.where(mask)[0]
                sorted_idx = candidates[np.argsort(np.abs(offset[candidates]))[::-1]]
                for idx in sorted_idx:
                    if len(output) >= max_items:
                        break
                    if any(abs(idx - h["frame"]) < min_separation for h in output):
                        continue
                    curvature_val = float(curvature_source[idx])
                    outside = (curvature_val > 0 and offset[idx] > 0) or (curvature_val < 0 and offset[idx] < 0)
                    lane_center_error_val = None
                    if data.get('road_frame_lane_center_error') is not None:
                        lane_center_error_val = safe_float(
                            float(data['road_frame_lane_center_error'][idx])
                        )
                    output.append({
                        "frame": int(idx),
                        "time": safe_float(data['time'][idx]) if idx < len(data['time']) else 0.0,
                        "road_offset": safe_float(float(offset[idx])),
                        "curvature": safe_float(curvature_val),
                        "steering": safe_float(float(steering[idx])) if steering is not None else None,
                        "heading_delta_deg": safe_float(float(heading_delta[idx])) if heading_delta is not None else None,
                        "gt_center": safe_float(float(gt_center[idx])) if gt_center is not None else None,
                        "perception_center": safe_float(float(p_center[idx])) if p_center is not None else None,
                        "outside": bool(outside),
                        "segment": "left" if curvature_val > 0 else "right",
                        "lane_center_error": lane_center_error_val,
                    })
                return output

            turn_bias = {
                "curve_threshold": safe_float(curve_threshold),
                "left_turn": summarize_mask(left_mask),
                "right_turn": summarize_mask(right_mask),
                "top_left": top_turn_frames(left_mask),
                "top_right": top_turn_frames(right_mask),
            }

            if gt_center is not None and p_center is not None:
                align_diff = p_center - gt_center
                if align_diff.size > 0:
                    alignment_summary = {
                        "perception_vs_gt_mean": safe_float(np.mean(align_diff)),
                        "perception_vs_gt_p95_abs": safe_float(np.percentile(np.abs(align_diff), 95)),
                        "perception_vs_gt_rmse": safe_float(np.sqrt(np.mean(align_diff ** 2))),
                    }
                else:
                    alignment_summary = {
                        "perception_vs_gt_mean": 0.0,
                        "perception_vs_gt_p95_abs": 0.0,
                        "perception_vs_gt_rmse": 0.0,
                    }
            if data.get('road_frame_lane_center_error') is not None:
                lane_center_error = data['road_frame_lane_center_error'][:n_bias]
                if alignment_summary is None:
                    alignment_summary = {}
                alignment_summary.update({
                    "road_frame_lane_center_error_mean": (
                        safe_float(np.mean(lane_center_error)) if lane_center_error.size > 0 else 0.0
                    ),
                    "road_frame_lane_center_error_p95_abs": (
                        safe_float(np.percentile(np.abs(lane_center_error), 95))
                        if lane_center_error.size > 0 else 0.0
                    ),
                    "road_frame_lane_center_error_rmse": (
                        safe_float(np.sqrt(np.mean(lane_center_error ** 2)))
                        if lane_center_error.size > 0 else 0.0
                    ),
                })
            if data.get('vehicle_frame_lookahead_offset') is not None:
                vf_offset = data['vehicle_frame_lookahead_offset'][:n_bias]
                if alignment_summary is None:
                    alignment_summary = {}
                alignment_summary.update({
                    "vehicle_frame_lookahead_offset_mean": (
                        safe_float(np.mean(vf_offset)) if vf_offset.size > 0 else 0.0
                    ),
                    "vehicle_frame_lookahead_offset_p95_abs": (
                        safe_float(np.percentile(np.abs(vf_offset), 95))
                        if vf_offset.size > 0 else 0.0
                    ),
                })
    
    # Layered scoring model (0-100 per layer) with hybrid cap.
    # Deadline preset weights:
    #   Safety 0.32, Trajectory 0.30, Control 0.16, Perception 0.14, LongitudinalComfort 0.08
    # Hybrid cap:
    #   if critical layer red -> cap 59
    #   elif critical layer yellow -> cap 79
    #   else cap 100
    lane_detection_penalty = safe_float(min(20, (100 - lane_detection_rate) * 0.2))
    stale_data_penalty = safe_float(min(15, stale_hard_rate * 0.15))
    perception_instability_penalty = safe_float(max(0, (100 - perception_stability_score) * 0.2))
    perception_instability_penalty = safe_float(min(20, perception_instability_penalty))
    lane_jitter_penalty = safe_float(min(10, max(0.0, lane_line_jitter_p95 - 0.30) * 30.0))
    reference_jitter_penalty = safe_float(min(10, max(0.0, reference_jitter_p95 - 0.15) * 40.0))

    trajectory_lateral_rmse_penalty = safe_float(min(30, lateral_error_rmse * 50))
    trajectory_lateral_p95_penalty = safe_float(min(20, max(0.0, lateral_error_p95 - 0.40) * 35.0))
    trajectory_heading_penalty = safe_float(min(20, max(0.0, np.degrees(heading_error_rmse) - 10.0) * 2.5))

    control_steering_jerk_penalty = safe_float(min(20, steering_jerk_max * 10))
    control_oscillation_penalty = safe_float(
        min(15, max(0.0, oscillation_frequency - 1.0) * 7.0)
    )
    control_sign_mismatch_penalty = safe_float(
        min(12, straight_sign_mismatch_rate * 0.2) if straight_sign_mismatch_rate > 5.0 else 0.0
    )

    longitudinal_accel_penalty = safe_float(
        min(20, max(0.0, (acceleration_p95 / G_MPS2) - 0.25) * 120.0)
    )
    longitudinal_jerk_penalty = safe_float(
        min(20, max(0.0, (jerk_p95 / G_MPS2) - 0.51) * 50.0)
    )

    safety_out_of_lane_penalty = safe_float(min(35, out_of_lane_time * 0.35))
    safety_event_penalty = 0.0
    if out_of_lane_events > 0:
        safety_event_penalty += 25.0
    if emergency_stop_frame is not None:
        safety_event_penalty += 20.0
    safety_event_penalty = safe_float(min(45, safety_event_penalty))

    # 8. SIGNAL INTEGRITY
    signal_integrity_heading_penalty = 0.0
    signal_integrity_rate_limit_penalty = 0.0
    signal_integrity_speed_feasibility_penalty = 0.0
    heading_suppression_rate = 0.0
    rate_limit_saturation_rate = 0.0
    speed_feasibility_violation_frames = 0

    curvature_source = None
    for _cs_key in ('reference_point_curvature', 'path_curvature_input', 'gt_path_curvature'):
        _cs_val = data.get(_cs_key)
        if _cs_val is not None and (not hasattr(_cs_val, '__len__') or len(_cs_val) > 0):
            curvature_source = _cs_val
            break
    if (
        curvature_source is not None
        and data.get('speed') is not None
        and len(curvature_source) > 0
        and len(data['speed']) > 0
    ):
        n_sig = min(n_frames, len(curvature_source), len(data['speed']))
        curvature_arr = np.asarray(curvature_source[:n_sig], dtype=np.float64)
        speed_arr = np.asarray(data['speed'][:n_sig], dtype=np.float64)
        curve_threshold = 0.003
        curve_mask = np.abs(curvature_arr) > curve_threshold
        n_curve = int(np.sum(curve_mask))

        # For heading suppression check, use a lower threshold (0.0005)
        # because suppression happens on curve approach where curvature
        # is still building up (0.001-0.003 range).
        approach_threshold = 0.0005
        approach_mask = np.abs(curvature_arr) > approach_threshold
        n_approach = int(np.sum(approach_mask))

        if n_approach > 0:
            heading_gate = data.get('diag_heading_zero_gate_active')
            rate_limit = data.get('diag_ref_x_rate_limit_active')
            if heading_gate is not None and len(heading_gate) >= n_sig:
                heading_active = np.asarray(heading_gate[:n_sig]) > 0.5
                heading_suppression_rate = safe_float(
                    np.sum(heading_active & approach_mask) / n_approach * 100.0
                )
                if heading_suppression_rate > 20.0:
                    signal_integrity_heading_penalty = safe_float(
                        min(25.0, (heading_suppression_rate - 20.0) * 0.625)
                    )
            if rate_limit is not None and len(rate_limit) >= n_sig:
                rate_limit_active = np.asarray(rate_limit[:n_sig]) > 0.5
                rate_limit_saturation_rate = safe_float(
                    np.sum(rate_limit_active & approach_mask) / n_approach * 100.0
                )
                if rate_limit_saturation_rate > 30.0:
                    signal_integrity_rate_limit_penalty = safe_float(
                        min(20.0, (rate_limit_saturation_rate - 30.0) * 0.4)
                    )

        if n_curve > 0:
            pass  # Speed feasibility uses original curve_mask below

        # Speed feasibility: speed > sqrt(2.45/|curvature|) is a violation
        curvature_eps = 1e-6
        abs_curv = np.abs(curvature_arr)
        v_max = np.where(abs_curv > curvature_eps, np.sqrt(2.45 / abs_curv), np.inf)
        speed_feasibility_violation_frames = int(np.sum(speed_arr > v_max))
        if speed_feasibility_violation_frames > 5:
            capped = min(speed_feasibility_violation_frames, 30)
            signal_integrity_speed_feasibility_penalty = safe_float(
                min(15.0, (capped - 5) * 0.6)
            )

    layer_breakdowns = {
        "Perception": {
            "base_score": 100.0,
            "deductions": [
                {
                    "name": "Lane Detection",
                    "value": lane_detection_penalty,
                    "limit": ">=90%",
                },
                {
                    "name": "Stale Hard Data",
                    "value": stale_data_penalty,
                    "limit": "<10%",
                },
                {
                    "name": "Perception Instability",
                    "value": perception_instability_penalty,
                    "limit": "stability>=80%",
                },
                {
                    "name": "Lane Line Jitter P95",
                    "value": lane_jitter_penalty,
                    "limit": "<=0.30m",
                },
                {
                    "name": "Reference Jitter P95",
                    "value": reference_jitter_penalty,
                    "limit": "<=0.15m",
                },
            ],
        },
        "Trajectory": {
            "base_score": 100.0,
            "deductions": [
                {
                    "name": "Lateral Error RMSE",
                    "value": trajectory_lateral_rmse_penalty,
                    "limit": "<=0.20m",
                },
                {
                    "name": "Lateral Error P95",
                    "value": trajectory_lateral_p95_penalty,
                    "limit": "<=0.40m",
                },
                {
                    "name": "Heading Error RMSE",
                    "value": trajectory_heading_penalty,
                    "limit": "<=10deg",
                },
            ],
        },
        "Control": {
            "base_score": 100.0,
            "deductions": [
                {
                    "name": "Steering Jerk",
                    "value": control_steering_jerk_penalty,
                    "limit": "<=0.50/s^2",
                },
                {
                    "name": "Oscillation Frequency",
                    "value": control_oscillation_penalty,
                    "limit": "<=1.0Hz",
                },
                {
                    "name": "Straight Sign Mismatch",
                    "value": control_sign_mismatch_penalty,
                    "limit": "<=5%",
                },
            ],
        },
        "LongitudinalComfort": {
            "base_score": 100.0,
            "deductions": [
                {
                    "name": "Acceleration P95",
                    "value": longitudinal_accel_penalty,
                    "limit": "<=0.25g",
                },
                {
                    "name": "Jerk P95",
                    "value": longitudinal_jerk_penalty,
                    "limit": "<=0.51g/s",
                },
            ],
        },
        "Safety": {
            "base_score": 100.0,
            "deductions": [
                {
                    "name": "Out Of Lane Time",
                    "value": safety_out_of_lane_penalty,
                    "limit": "<5%",
                },
                {
                    "name": "Out Of Lane / Emergency Events",
                    "value": safety_event_penalty,
                    "limit": "none",
                },
            ],
        },
        "SignalIntegrity": {
            "base_score": 100.0,
            "deductions": [
                {
                    "name": "Heading Suppression Rate (curves)",
                    "value": signal_integrity_heading_penalty,
                    "limit": "<=20%",
                },
                {
                    "name": "Rate Limit Saturation Rate (curves)",
                    "value": signal_integrity_rate_limit_penalty,
                    "limit": "<=30%",
                },
                {
                    "name": "Speed Feasibility Violations",
                    "value": signal_integrity_speed_feasibility_penalty,
                    "limit": "<=5 frames",
                },
            ],
        },
    }

    layer_scores: Dict[str, float] = {}
    for layer_name, layer in layer_breakdowns.items():
        total_deduction = sum(float(d.get("value", 0.0)) for d in layer["deductions"])
        layer_score = safe_float(max(0.0, 100.0 - total_deduction))
        layer["total_deduction"] = safe_float(total_deduction)
        layer["final_score"] = layer_score
        layer_scores[layer_name] = layer_score

    # Scale existing weights by 0.92 to make room for SignalIntegrity 0.08 (sum to 1.0)
    layer_weights = {
        "Safety": 0.32 * 0.92,
        "Trajectory": 0.30 * 0.92,
        "Control": 0.16 * 0.92,
        "Perception": 0.14 * 0.92,
        "LongitudinalComfort": 0.08 * 0.92,
        "SignalIntegrity": 0.08,
    }

    weighted_contributions = {
        layer: safe_float(layer_scores.get(layer, 0.0) * weight)
        for layer, weight in layer_weights.items()
    }
    overall_base = safe_float(sum(weighted_contributions.values()))

    critical_layers = ("Safety", "Trajectory")
    critical_cap = 100.0
    cap_reason = "none"
    critical_layer_colors: Dict[str, str] = {}
    for layer in critical_layers:
        score_val = layer_scores.get(layer, 0.0)
        if score_val < 60.0:
            critical_layer_colors[layer] = "red"
        elif score_val < 80.0:
            critical_layer_colors[layer] = "yellow"
        else:
            critical_layer_colors[layer] = "green"

    if any(color == "red" for color in critical_layer_colors.values()):
        critical_cap = 59.0
        cap_reason = "critical_red_layer"
    elif any(color == "yellow" for color in critical_layer_colors.values()):
        critical_cap = 79.0
        cap_reason = "critical_yellow_layer"

    score = safe_float(min(overall_base, critical_cap))
    
    # Generate recommendations
    recommendations = []
    if lateral_error_rmse > 0.3:
        recommendations.append("Reduce lateral error - check PID gains or trajectory planning")
    if steering_jerk_max > 1.0:
        recommendations.append("Reduce steering jerk - increase rate limiting or reduce PID gains")
    if oscillation_frequency > 2.0:
        recommendations.append("Reduce oscillation - increase damping or reduce proportional gain")
    if lane_detection_rate < 90:
        recommendations.append("Improve lane detection - check perception model or CV fallback")
    if lane_line_jitter_p95 > 0.6 or reference_jitter_p95 > 0.25:
        recommendations.append("Reduce perception jitter - increase temporal smoothing or clamp lane-line deltas")
    if right_lane_low_visibility_rate > 10 or single_lane_rate > 10:
        recommendations.append("Right lane visibility drops - add single-lane fallback or widen camera FOV")
    elif right_lane_edge_contact_rate > 20:
        recommendations.append("Right lane frequently touches image edge on right turns - treat as FOV-limited and rely on single-lane corridor logic")
    if stale_hard_rate > 10:
        recommendations.append("Reduce stale data usage - relax jump detection threshold or improve perception")
    if speed_limit_zero_rate > 10:
        recommendations.append("Speed limit missing - verify Unity track speed limits are sent to the bridge")
    if emergency_stop_frame is not None:
        recommendations.append("Emergency stop triggered - review lateral bounds and recovery logic")
    if pid_integral_max > 0.2:
        recommendations.append("Reduce PID integral accumulation - check integral reset mechanisms")
    if straight_oscillation_mean > 0.2:
        recommendations.append("Straight-line oscillation detected - increase deadband or smoothing")
    if straight_sign_mismatch_events > 0:
        recommendations.append("Steering sign mismatches on straights - relax straight smoothing or rate limits")
    if acceleration_p95 > 2.5:
        recommendations.append("Reduce longitudinal acceleration spikes - tune throttle/brake gains")
    if jerk_p95 > 5.0:
        recommendations.append("Reduce longitudinal jerk - add rate limiting on throttle/brake")
    
    # Key issues
    key_issues = []
    if out_of_lane_events > 0:
        key_issues.append(f"{out_of_lane_events} out-of-lane events")
    if lane_detection_rate < 80:
        key_issues.append(f"Low lane detection rate ({lane_detection_rate:.1f}%)")
    if lane_line_jitter_p95 > 0.6:
        key_issues.append(f"High lane-line jitter (p95={lane_line_jitter_p95:.2f}m)")
    if reference_jitter_p95 > 0.25:
        key_issues.append(f"High reference jitter (p95={reference_jitter_p95:.2f}m)")
    if right_lane_low_visibility_rate > 10:
        key_issues.append(f"Right lane low visibility ({right_lane_low_visibility_rate:.1f}%)")
    elif right_lane_edge_contact_rate > 20:
        key_issues.append(f"Right lane at image edge often ({right_lane_edge_contact_rate:.1f}%)")
    if stale_hard_rate > 20:
        key_issues.append(f"High hard stale data usage ({stale_hard_rate:.1f}%)")
    if speed_limit_zero_rate > 10:
        key_issues.append(f"Speed limit missing ({speed_limit_zero_rate:.1f}%)")
    if emergency_stop_frame is not None:
        key_issues.append(f"Emergency stop at frame {emergency_stop_frame}")
    if steering_jerk_max > 2.0:
        key_issues.append("High steering jerk")
    if straight_oscillation_mean > 0.2:
        key_issues.append("Straight-line oscillation detected")
    if straight_sign_mismatch_events > 0:
        key_issues.append("Steering sign mismatches on straights")
    if acceleration_p95 > 2.5:
        key_issues.append("High longitudinal acceleration")
    if jerk_p95 > 5.0:
        key_issues.append("High longitudinal jerk")
    
    return {
        "executive_summary": {
            "overall_score": safe_float(score),
            "drive_duration": safe_float(duration),
            "total_frames": int(n_frames),
            "success_rate": safe_float(time_in_lane),
            "key_issues": key_issues,
            "failure_detected": failure_frame is not None,
            "failure_frame": int(failure_frame) if failure_frame is not None else None,
            "analyzed_to_failure": analyze_to_failure,
            "failure_detection_source": error_source if error_data is not None else "none",
            "score_breakdown": {
                "base_score": 100.0,
                "lateral_error_penalty": trajectory_lateral_rmse_penalty,
                "steering_jerk_penalty": control_steering_jerk_penalty,
                "lane_detection_penalty": lane_detection_penalty,
                "stale_data_penalty": stale_data_penalty,
                "perception_instability_penalty": perception_instability_penalty,
                "out_of_lane_penalty": safety_out_of_lane_penalty,
                "straight_sign_mismatch_penalty": control_sign_mismatch_penalty,
                "layer_weights": layer_weights,
                "layer_scores": layer_scores,
                "layer_weighted_contributions": weighted_contributions,
                "overall_base_score": overall_base,
                "overall_cap": critical_cap,
                "cap_reason": cap_reason,
                "critical_layer_status": critical_layer_colors,
            },
        },
        "path_tracking": {
            "lateral_error_rmse": safe_float(lateral_error_rmse),
            "lateral_error_mean": safe_float(lateral_error_mean),
            "lateral_error_max": safe_float(lateral_error_max),
            "lateral_error_p95": safe_float(lateral_error_p95),
            "heading_error_rmse": safe_float(heading_error_rmse),
            "heading_error_max": safe_float(heading_error_max),
            "time_in_lane": safe_float(time_in_lane),
            "time_in_lane_centered": safe_float(time_in_lane_centered)
        },
        "layer_scores": layer_scores,
        "layer_score_breakdown": layer_breakdowns,
        "control_mode": _detect_control_mode(data),
        "pp_feedback_gain": _pp_feedback_gain(data),
        "pp_mean_lookahead_distance": _pp_mean_ld(data),
        "pp_ref_jump_clamped_count": _pp_jump_count(data),
        "control_smoothness": {
            "steering_jerk_max": safe_float(steering_jerk_max),
            "steering_rate_max": safe_float(steering_rate_max),
            "steering_smoothness": safe_float(steering_smoothness),
            "oscillation_frequency": safe_float(oscillation_frequency)
        },
        "speed_control": {
            "speed_error_rmse": safe_float(speed_error_rmse),
            "speed_error_mean": safe_float(speed_error_mean),
            "speed_error_max": safe_float(speed_error_max),
            "speed_overspeed_rate": safe_float(speed_overspeed_rate),
            "speed_limit_zero_rate": safe_float(speed_limit_zero_rate),
            "speed_surge_count": int(speed_surge_count),
            "speed_surge_avg_drop": safe_float(speed_surge_avg_drop),
            "speed_surge_p95_drop": safe_float(speed_surge_p95_drop),
            "acceleration_mean": safe_float(acceleration_mean),
            "acceleration_p95": safe_float(acceleration_p95),
            "acceleration_max": safe_float(acceleration_max),
            "jerk_mean": safe_float(jerk_mean),
            "jerk_p95": safe_float(jerk_p95),
            "jerk_max": safe_float(jerk_max),
            "acceleration_mean_filtered": safe_float(acceleration_mean_filtered),
            "acceleration_p95_filtered": safe_float(acceleration_p95_filtered),
            "acceleration_max_filtered": safe_float(acceleration_max_filtered),
            "jerk_mean_filtered": safe_float(jerk_mean_filtered),
            "jerk_p95_filtered": safe_float(jerk_p95_filtered),
            "jerk_max_filtered": safe_float(jerk_max_filtered),
            "lateral_accel_p95": safe_float(lateral_accel_p95),
            "lateral_jerk_p95": safe_float(lateral_jerk_p95),
            "lateral_jerk_max": safe_float(lateral_jerk_max)
        },
        "comfort": {
            "steering_jerk_max": safe_float(steering_jerk_max),
            "acceleration_p95": safe_float(acceleration_p95),
            "jerk_p95": safe_float(jerk_p95),
            "acceleration_p95_filtered": safe_float(acceleration_p95_filtered),
            "jerk_p95_filtered": safe_float(jerk_p95_filtered),
            "jerk_max_filtered": safe_float(jerk_max_filtered),
            "lateral_accel_p95": safe_float(lateral_accel_p95),
            "lateral_jerk_p95": safe_float(lateral_jerk_p95),
            "acceleration_p95_g": safe_float(acceleration_p95 / G_MPS2),
            "acceleration_p95_filtered_g": safe_float(acceleration_p95_filtered / G_MPS2),
            "jerk_p95_gps": safe_float(jerk_p95 / G_MPS2),
            "jerk_p95_filtered_gps": safe_float(jerk_p95_filtered / G_MPS2),
            "lateral_accel_p95_g": safe_float(lateral_accel_p95 / G_MPS2),
            "lateral_jerk_p95_gps": safe_float(lateral_jerk_p95 / G_MPS2),
            "comfort_gate_thresholds_g": {
                "longitudinal_accel_p95_g": 0.25,
                "longitudinal_jerk_p95_gps": 0.51
            },
            "comfort_gate_thresholds_si": {
                "longitudinal_accel_p95_mps2": safe_float(0.25 * G_MPS2),
                "longitudinal_jerk_p95_mps3": safe_float(0.51 * G_MPS2)
            }
        },
        "control_stability": {
            "straight_fraction": safe_float(straight_fraction),
            "straight_oscillation_mean": safe_float(straight_oscillation_mean),
            "straight_oscillation_max": safe_float(straight_oscillation_max),
            "tuned_deadband_mean": safe_float(tuned_deadband_mean),
            "tuned_deadband_max": safe_float(tuned_deadband_max),
            "tuned_smoothing_mean": safe_float(tuned_smoothing_mean),
            "straight_sign_mismatch_rate": safe_float(straight_sign_mismatch_rate),
            "straight_sign_mismatch_events": int(straight_sign_mismatch_events),
            "straight_sign_mismatch_frames": int(straight_sign_mismatch_frames),
            "straight_sign_mismatch_events_list": straight_sign_mismatch_events_list
        },
        "perception_quality": {
            "lane_detection_rate": safe_float(lane_detection_rate),
            "perception_confidence_mean": safe_float(perception_confidence_mean),
            "perception_jumps_detected": int(perception_jumps_detected),
            "perception_instability_detected": int(perception_instability_detected),
            "perception_extreme_coeffs_detected": int(perception_extreme_coeffs_detected),
            "perception_invalid_width_detected": int(perception_invalid_width_detected),
            "stale_perception_rate": safe_float(stale_perception_rate),
            "stale_raw_rate": safe_float(stale_raw_rate),
            "stale_hard_rate": safe_float(stale_hard_rate),
            "stale_fallback_visibility_rate": safe_float(stale_fallback_visibility_rate),
            "perception_stability_score": safe_float(perception_stability_score),
            "lane_position_variance": safe_float(lane_position_variance),
            "lane_width_variance": safe_float(lane_width_variance),
            "lane_line_jitter_p95": safe_float(lane_line_jitter_p95),
            "lane_line_jitter_p99": safe_float(lane_line_jitter_p99),
            "reference_jitter_p95": safe_float(reference_jitter_p95),
            "reference_jitter_p99": safe_float(reference_jitter_p99),
            "single_lane_rate": safe_float(single_lane_rate),
            "right_lane_low_visibility_rate": safe_float(right_lane_low_visibility_rate),
            "right_lane_edge_contact_rate": safe_float(right_lane_edge_contact_rate),
            "left_lane_low_visibility_rate": safe_float(left_lane_low_visibility_rate)
        },
        "trajectory_quality": {
            "trajectory_availability": safe_float(trajectory_availability),
            "ref_point_accuracy_rmse": safe_float(ref_point_accuracy_rmse)
        },
        "turn_bias": turn_bias,
        "alignment_summary": alignment_summary,
        "system_health": {
            "pid_integral_max": safe_float(pid_integral_max),
            "unity_time_gap_max": safe_float(unity_time_gap_max),
            "unity_time_gap_count": int(unity_time_gap_count)
        },
        "safety": {
            "out_of_lane_events": int(out_of_lane_events),
            "out_of_lane_time": safe_float(out_of_lane_time),
            "out_of_lane_events_list": out_of_lane_events_list  # List of individual events with frame numbers
        },
        "recommendations": recommendations,
        "config": config_summary,
        "time_series": {
            "time": data['time'].tolist(),
            "lateral_error": data['lateral_error'].tolist() if data['lateral_error'] is not None else None,
            "steering": data['steering'].tolist(),
            "num_lanes_detected": data['num_lanes_detected'].tolist() if data['num_lanes_detected'] is not None else None,
            "perception_health_score": None  # Will be added if available
        }
    }

