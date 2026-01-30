"""
Summary analyzer for recording-level metrics.
Extracted from analyze_drive_overall.py for use in debug visualizer.
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Optional
from scipy.fft import fft, fftfreq
import math


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
            
            # Trajectory data
            data['ref_x'] = np.array(f['trajectory/reference_point_x'][:]) if 'trajectory/reference_point_x' in f else None
            data['ref_heading'] = np.array(f['trajectory/reference_point_heading'][:]) if 'trajectory/reference_point_heading' in f else None
            data['ref_velocity'] = (
                np.array(f['trajectory/reference_point_velocity'][:])
                if 'trajectory/reference_point_velocity' in f else None
            )
            
            # Perception data
            data['num_lanes_detected'] = np.array(f['perception/num_lanes_detected'][:]) if 'perception/num_lanes_detected' in f else None
            data['confidence'] = np.array(f['perception/confidence'][:]) if 'perception/confidence' in f else None
            data['using_stale_data'] = np.array(f['perception/using_stale_data'][:]) if 'perception/using_stale_data' in f else None
            data['stale_reason'] = None
            if 'perception/stale_reason' in f:
                stale_reasons = f['perception/stale_reason'][:]
                if len(stale_reasons) > 0:
                    data['stale_reason'] = [s.decode('utf-8') if isinstance(s, bytes) else s for s in stale_reasons]
            
            # Ground truth (if available)
            data['gt_center'] = np.array(f['ground_truth/lane_center_x'][:]) if 'ground_truth/lane_center_x' in f else None
            data['gt_left'] = np.array(f['ground_truth/left_lane_line_x'][:]) if 'ground_truth/left_lane_line_x' in f else None
            data['gt_right'] = np.array(f['ground_truth/right_lane_line_x'][:]) if 'ground_truth/right_lane_line_x' in f else None
            data['gt_path_curvature'] = (
                np.array(f['ground_truth/path_curvature'][:])
                if 'ground_truth/path_curvature' in f else None
            )
            
            # Load lane positions for stability calculation
            data['left_lane_x'] = np.array(f['perception/left_lane_line_x'][:]) if 'perception/left_lane_line_x' in f else None
            data['right_lane_x'] = np.array(f['perception/right_lane_line_x'][:]) if 'perception/right_lane_line_x' in f else None
            
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
    lateral_accel_p95 = 0.0
    lateral_jerk_p95 = 0.0
    lateral_jerk_max = 0.0
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
        curvature = None
        if data.get('gt_path_curvature') is not None:
            curvature = data['gt_path_curvature']
        elif data.get('path_curvature_input') is not None:
            curvature = data['path_curvature_input']
        if curvature is not None:
            n_lat = min(len(curvature), len(data['speed']))
            if n_lat > 1:
                lat_accel = (data['speed'][:n_lat] ** 2) * curvature[:n_lat]
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
    
    stale_perception_rate = safe_float(np.sum(data['using_stale_data']) / n_frames * 100 if data['using_stale_data'] is not None and n_frames > 0 else 0.0)
    
    # NEW: Calculate perception stability metrics (lane position/width variance)
    # High variance indicates perception instability even if not caught by stale_data
    perception_stability_score = 100.0  # Start at 100, penalize for instability
    lane_position_variance = 0.0
    lane_width_variance = 0.0
    
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
            
            # Penalize high variance (instability)
            # Position variance > 0.1m² indicates instability
            if lane_position_variance > 0.1:
                perception_stability_score -= min(30, (lane_position_variance - 0.1) * 100)
            
            # Width variance > 0.2m² indicates instability
            if lane_width_variance > 0.2:
                perception_stability_score -= min(30, (lane_width_variance - 0.2) * 50)
            
            perception_stability_score = safe_float(max(0, perception_stability_score))
    
    # Also penalize for detected instability events
    if perception_instability_detected > 0:
        instability_rate = (perception_instability_detected / n_frames) * 100
        perception_stability_score -= min(40, instability_rate * 2)  # -2 points per % of frames with instability
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
    
    # Calculate overall score (0-100)
    score = 100.0
    score -= min(30, lateral_error_rmse * 50)  # Penalize lateral error
    score -= min(20, steering_jerk_max * 10)  # Penalize jerk
    score -= min(20, (100 - lane_detection_rate) * 0.2)  # Penalize detection failures
    score -= min(15, stale_perception_rate * 0.15)  # Penalize stale data
    # NEW: Penalize perception instability (even if not caught by stale_data)
    perception_instability_penalty = safe_float(max(0, (100 - perception_stability_score) * 0.2))  # -0.2 points per stability point lost
    score -= min(20, perception_instability_penalty)  # Cap at 20 points
    score -= min(15, out_of_lane_time * 0.15)  # Penalize out-of-lane
    score = safe_float(max(0, score))
    
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
    if stale_perception_rate > 10:
        recommendations.append("Reduce stale data usage - relax jump detection threshold or improve perception")
    if speed_limit_zero_rate > 10:
        recommendations.append("Speed limit missing - verify Unity track speed limits are sent to the bridge")
    if emergency_stop_frame is not None:
        recommendations.append("Emergency stop triggered - review lateral bounds and recovery logic")
    if pid_integral_max > 0.2:
        recommendations.append("Reduce PID integral accumulation - check integral reset mechanisms")
    if straight_oscillation_mean > 0.2:
        recommendations.append("Straight-line oscillation detected - increase deadband or smoothing")
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
    if stale_perception_rate > 20:
        key_issues.append(f"High stale data usage ({stale_perception_rate:.1f}%)")
    if speed_limit_zero_rate > 10:
        key_issues.append(f"Speed limit missing ({speed_limit_zero_rate:.1f}%)")
    if emergency_stop_frame is not None:
        key_issues.append(f"Emergency stop at frame {emergency_stop_frame}")
    if steering_jerk_max > 2.0:
        key_issues.append("High steering jerk")
    if straight_oscillation_mean > 0.2:
        key_issues.append("Straight-line oscillation detected")
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
                "lateral_error_penalty": safe_float(min(30, lateral_error_rmse * 50)),
                "steering_jerk_penalty": safe_float(min(20, steering_jerk_max * 10)),
                "lane_detection_penalty": safe_float(min(20, (100 - lane_detection_rate) * 0.2)),
                "stale_data_penalty": safe_float(min(15, stale_perception_rate * 0.15)),
                "perception_instability_penalty": safe_float(min(20, perception_instability_penalty)),
                "out_of_lane_penalty": safe_float(min(15, out_of_lane_time * 0.15))
            }
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
            "acceleration_mean": safe_float(acceleration_mean),
            "acceleration_p95": safe_float(acceleration_p95),
            "acceleration_max": safe_float(acceleration_max),
            "jerk_mean": safe_float(jerk_mean),
            "jerk_p95": safe_float(jerk_p95),
            "jerk_max": safe_float(jerk_max),
            "lateral_accel_p95": safe_float(lateral_accel_p95),
            "lateral_jerk_p95": safe_float(lateral_jerk_p95),
            "lateral_jerk_max": safe_float(lateral_jerk_max)
        },
        "comfort": {
            "steering_jerk_max": safe_float(steering_jerk_max),
            "acceleration_p95": safe_float(acceleration_p95),
            "jerk_p95": safe_float(jerk_p95),
            "lateral_accel_p95": safe_float(lateral_accel_p95),
            "lateral_jerk_p95": safe_float(lateral_jerk_p95)
        },
        "control_stability": {
            "straight_fraction": safe_float(straight_fraction),
            "straight_oscillation_mean": safe_float(straight_oscillation_mean),
            "straight_oscillation_max": safe_float(straight_oscillation_max),
            "tuned_deadband_mean": safe_float(tuned_deadband_mean),
            "tuned_deadband_max": safe_float(tuned_deadband_max),
            "tuned_smoothing_mean": safe_float(tuned_smoothing_mean)
        },
        "perception_quality": {
            "lane_detection_rate": safe_float(lane_detection_rate),
            "perception_confidence_mean": safe_float(perception_confidence_mean),
            "perception_jumps_detected": int(perception_jumps_detected),
            "perception_instability_detected": int(perception_instability_detected),
            "perception_extreme_coeffs_detected": int(perception_extreme_coeffs_detected),
            "perception_invalid_width_detected": int(perception_invalid_width_detected),
            "stale_perception_rate": safe_float(stale_perception_rate),
            "perception_stability_score": safe_float(perception_stability_score),
            "lane_position_variance": safe_float(lane_position_variance),
            "lane_width_variance": safe_float(lane_width_variance)
        },
        "trajectory_quality": {
            "trajectory_availability": safe_float(trajectory_availability),
            "ref_point_accuracy_rmse": safe_float(ref_point_accuracy_rmse)
        },
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
        "time_series": {
            "time": data['time'].tolist(),
            "lateral_error": data['lateral_error'].tolist() if data['lateral_error'] is not None else None,
            "steering": data['steering'].tolist(),
            "num_lanes_detected": data['num_lanes_detected'].tolist() if data['num_lanes_detected'] is not None else None,
            "perception_health_score": None  # Will be added if available
        }
    }

