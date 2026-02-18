"""
Trajectory vs Steering Diagnostic Module

Analyzes a recording to determine if issues are in trajectory planning or steering control.
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Optional
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


def analyze_trajectory_vs_steering(
    recording_path: Path,
    analyze_to_failure: bool = False,
    curve_entry_start_distance_m: Optional[float] = None,
    curve_entry_window_distance_m: Optional[float] = None,
) -> Dict:
    """
    Analyze trajectory vs steering to identify which component is failing.
    
    Args:
        recording_path: Path to HDF5 recording file
        analyze_to_failure: If True, only analyze up to the point where car went out of lane and stayed out
        curve_entry_start_distance_m: Optional manual curve-entry start distance from run start (meters)
        curve_entry_window_distance_m: Optional manual curve-entry window length (meters)
        
    Returns:
        Dictionary with diagnostic results
    """
    try:
        with h5py.File(recording_path, 'r') as f:
            # Check data availability
            has_trajectory = "trajectory/reference_point_x" in f
            has_control = "control/steering" in f
            has_ground_truth = "ground_truth/lane_center_x" in f
            has_perception = "perception/left_lane_line_x" in f
            
            if not has_trajectory:
                return {
                    "error": "No trajectory data found. This recording doesn't have trajectory planning data.",
                    "data_availability": {
                        "trajectory": False,
                        "control": has_control,
                        "ground_truth": has_ground_truth,
                        "perception": has_perception
                    }
                }
            
            # Load trajectory data
            ref_x = np.array(f["trajectory/reference_point_x"][:])
            ref_y = np.array(f["trajectory/reference_point_y"][:])
            ref_heading = np.array(f["trajectory/reference_point_heading"][:])
            
            # Load control data
            steering = None
            lateral_error = None
            heading_error = None
            is_straight = None
            path_curvature = None
            gt_curvature = None
            control_timestamps = None
            throttle = None
            brake = None
            target_speed = None
            road_heading_deg = None
            using_stale_perception = None
            steering_pre_rate_limit = None
            steering_post_rate_limit = None
            steering_post_jerk_limit = None
            steering_post_hard_clip = None
            steering_post_smoothing = None
            steering_rate_limited_delta = None
            steering_jerk_limited_delta = None
            steering_hard_clip_delta = None
            steering_smoothing_delta = None
            steering_authority_gap = None
            steering_first_limiter_stage_code = None
            
            if has_control:
                steering = np.array(f["control/steering"][:])
                if "control/lateral_error" in f:
                    lateral_error = np.array(f["control/lateral_error"][:])
                if "control/heading_error" in f:
                    heading_error = np.array(f["control/heading_error"][:])
                if "control/is_straight" in f:
                    is_straight = np.array(f["control/is_straight"][:]).astype(bool)
                if "control/path_curvature_input" in f:
                    path_curvature = np.array(f["control/path_curvature_input"][:])
                if "control/timestamps" in f:
                    control_timestamps = np.array(f["control/timestamps"][:])
                if "control/throttle" in f:
                    throttle = np.array(f["control/throttle"][:])
                if "control/brake" in f:
                    brake = np.array(f["control/brake"][:])
                if "control/target_speed_final" in f:
                    target_speed = np.array(f["control/target_speed_final"][:])
                elif "control/target_speed_planned" in f:
                    target_speed = np.array(f["control/target_speed_planned"][:])
                if "control/using_stale_perception" in f:
                    using_stale_perception = np.array(f["control/using_stale_perception"][:]).astype(bool)
                if "control/steering_pre_rate_limit" in f:
                    steering_pre_rate_limit = np.array(f["control/steering_pre_rate_limit"][:])
                if "control/steering_post_rate_limit" in f:
                    steering_post_rate_limit = np.array(f["control/steering_post_rate_limit"][:])
                if "control/steering_post_jerk_limit" in f:
                    steering_post_jerk_limit = np.array(f["control/steering_post_jerk_limit"][:])
                if "control/steering_post_hard_clip" in f:
                    steering_post_hard_clip = np.array(f["control/steering_post_hard_clip"][:])
                if "control/steering_post_smoothing" in f:
                    steering_post_smoothing = np.array(f["control/steering_post_smoothing"][:])
                if "control/steering_rate_limited_delta" in f:
                    steering_rate_limited_delta = np.array(f["control/steering_rate_limited_delta"][:])
                if "control/steering_jerk_limited_delta" in f:
                    steering_jerk_limited_delta = np.array(f["control/steering_jerk_limited_delta"][:])
                if "control/steering_hard_clip_delta" in f:
                    steering_hard_clip_delta = np.array(f["control/steering_hard_clip_delta"][:])
                if "control/steering_smoothing_delta" in f:
                    steering_smoothing_delta = np.array(f["control/steering_smoothing_delta"][:])
                if "control/steering_authority_gap" in f:
                    steering_authority_gap = np.array(f["control/steering_authority_gap"][:])
                if "control/steering_first_limiter_stage_code" in f:
                    steering_first_limiter_stage_code = np.array(
                        f["control/steering_first_limiter_stage_code"][:]
                    )
            
            # Load ground truth for comparison
            gt_center = None
            if has_ground_truth:
                gt_center = np.array(f["ground_truth/lane_center_x"][:])
                if "ground_truth/path_curvature" in f:
                    gt_curvature = np.array(f["ground_truth/path_curvature"][:])
                    if path_curvature is None:
                        path_curvature = gt_curvature
            
            # Load perception for understanding trajectory source
            perception_center = None
            perception_left = None
            perception_right = None
            if has_perception:
                perception_left = np.array(f["perception/left_lane_line_x"][:])
                perception_right = np.array(f["perception/right_lane_line_x"][:])
                perception_center = (perception_left + perception_right) / 2.0

            speed = None
            if "vehicle/speed" in f:
                speed = np.array(f["vehicle/speed"][:])
            if "vehicle/road_heading_deg" in f:
                road_heading_deg = np.array(f["vehicle/road_heading_deg"][:])
            
            # Align data lengths
            min_len = min(
                len(ref_x),
                len(steering) if steering is not None else len(ref_x),
                len(gt_center) if gt_center is not None else len(ref_x),
                len(perception_center) if perception_center is not None else len(ref_x)
            )
            if speed is not None and len(speed) > 0:
                min_len = min(min_len, len(speed))
            if throttle is not None and len(throttle) > 0:
                min_len = min(min_len, len(throttle))
            if brake is not None and len(brake) > 0:
                min_len = min(min_len, len(brake))
            if target_speed is not None and len(target_speed) > 0:
                min_len = min(min_len, len(target_speed))
            
            ref_x = ref_x[:min_len]
            ref_y = ref_y[:min_len]
            ref_heading = ref_heading[:min_len]
            
            if steering is not None:
                steering = steering[:min_len]
            if lateral_error is not None:
                lateral_error = lateral_error[:min_len]
            if heading_error is not None:
                heading_error = heading_error[:min_len]
            if is_straight is not None:
                is_straight = is_straight[:min_len]
            if path_curvature is not None:
                path_curvature = path_curvature[:min_len]
            if gt_curvature is not None:
                gt_curvature = gt_curvature[:min_len]
            if control_timestamps is not None:
                control_timestamps = control_timestamps[:min_len]
            if gt_center is not None:
                gt_center = gt_center[:min_len]
            if perception_center is not None:
                perception_center = perception_center[:min_len]
            if speed is not None:
                speed = speed[:min_len]
            if throttle is not None:
                throttle = throttle[:min_len]
            if brake is not None:
                brake = brake[:min_len]
            if target_speed is not None:
                target_speed = target_speed[:min_len]
            if using_stale_perception is not None:
                using_stale_perception = using_stale_perception[:min_len]
            if road_heading_deg is not None:
                road_heading_deg = road_heading_deg[:min_len]
            if steering_pre_rate_limit is not None:
                steering_pre_rate_limit = steering_pre_rate_limit[:min_len]
            if steering_post_rate_limit is not None:
                steering_post_rate_limit = steering_post_rate_limit[:min_len]
            if steering_post_jerk_limit is not None:
                steering_post_jerk_limit = steering_post_jerk_limit[:min_len]
            if steering_post_hard_clip is not None:
                steering_post_hard_clip = steering_post_hard_clip[:min_len]
            if steering_post_smoothing is not None:
                steering_post_smoothing = steering_post_smoothing[:min_len]
            if steering_rate_limited_delta is not None:
                steering_rate_limited_delta = steering_rate_limited_delta[:min_len]
            if steering_jerk_limited_delta is not None:
                steering_jerk_limited_delta = steering_jerk_limited_delta[:min_len]
            if steering_hard_clip_delta is not None:
                steering_hard_clip_delta = steering_hard_clip_delta[:min_len]
            if steering_smoothing_delta is not None:
                steering_smoothing_delta = steering_smoothing_delta[:min_len]
            if steering_authority_gap is not None:
                steering_authority_gap = steering_authority_gap[:min_len]
            if steering_first_limiter_stage_code is not None:
                steering_first_limiter_stage_code = steering_first_limiter_stage_code[:min_len]

            # Preserve full aligned series for distance-based feasibility analysis.
            speed_full = speed.copy() if speed is not None else None
            path_curvature_full = path_curvature.copy() if path_curvature is not None else None
            steering_full = steering.copy() if steering is not None else None
            steering_pre_rate_limit_full = (
                steering_pre_rate_limit.copy() if steering_pre_rate_limit is not None else None
            )
            steering_post_smoothing_full = (
                steering_post_smoothing.copy() if steering_post_smoothing is not None else None
            )
            steering_rate_limited_delta_full = (
                steering_rate_limited_delta.copy() if steering_rate_limited_delta is not None else None
            )
            steering_jerk_limited_delta_full = (
                steering_jerk_limited_delta.copy() if steering_jerk_limited_delta is not None else None
            )
            steering_hard_clip_delta_full = (
                steering_hard_clip_delta.copy() if steering_hard_clip_delta is not None else None
            )
            steering_smoothing_delta_full = (
                steering_smoothing_delta.copy() if steering_smoothing_delta is not None else None
            )
            steering_authority_gap_full = (
                steering_authority_gap.copy() if steering_authority_gap is not None else None
            )
            steering_first_limiter_stage_code_full = (
                steering_first_limiter_stage_code.copy()
                if steering_first_limiter_stage_code is not None
                else None
            )
            using_stale_perception_full = (
                using_stale_perception.copy() if using_stale_perception is not None else None
            )
            control_timestamps_full = (
                control_timestamps.copy() if control_timestamps is not None else None
            )
            
            # Detect failure point and truncate if requested (same logic as summary_analyzer)
            failure_frame = None
            if analyze_to_failure:
                # Use ground truth if available, otherwise use perception-based error
                if "ground_truth/lane_center_x" in f:
                    gt_center_full = np.array(f["ground_truth/lane_center_x"][:])
                    gt_error_full = -gt_center_full
                    error_data_full = gt_error_full
                    error_source = "ground_truth"
                elif lateral_error is not None:
                    error_data_full = lateral_error
                    error_source = "perception"
                else:
                    error_data_full = None
                
                if error_data_full is not None:
                    # Same failure detection logic as summary_analyzer
                    out_of_lane_threshold = 0.5
                    catastrophic_threshold = 2.0
                    min_consecutive_out = 10
                    
                    # Find catastrophic failure
                    catastrophic_frame = None
                    for i, error in enumerate(error_data_full):
                        if abs(error) > catastrophic_threshold:
                            catastrophic_frame = i
                            break
                    
                    if catastrophic_frame is not None:
                        # Find last recovery
                        last_recovery_frame = None
                        for i in range(catastrophic_frame - 1, -1, -1):
                            if abs(error_data_full[i]) < out_of_lane_threshold:
                                last_recovery_frame = i
                                break
                        
                        # Find failure frame
                        if last_recovery_frame is not None:
                            consecutive_out_frames = 0
                            for i in range(last_recovery_frame + 1, len(error_data_full)):
                                error = error_data_full[i]
                                if abs(error) > out_of_lane_threshold:
                                    consecutive_out_frames += 1
                                    if consecutive_out_frames >= min_consecutive_out and failure_frame is None:
                                        failure_frame = i - (min_consecutive_out - 1)
                                        break
                                else:
                                    consecutive_out_frames = 0
                    
                    # Fallback: If no catastrophic failure, use simple detection
                    if failure_frame is None:
                        consecutive_out_frames = 0
                        for i, error in enumerate(error_data_full):
                            if abs(error) > out_of_lane_threshold:
                                consecutive_out_frames += 1
                                if consecutive_out_frames >= min_consecutive_out and failure_frame is None:
                                    failure_frame = i - (min_consecutive_out - 1)
                                    break
                            else:
                                consecutive_out_frames = 0
                    
                    # Truncate all arrays to failure_frame
                    if failure_frame is not None and failure_frame < min_len:
                        min_len = failure_frame
                        ref_x = ref_x[:min_len]
                        ref_y = ref_y[:min_len]
                        ref_heading = ref_heading[:min_len]
                        if steering is not None:
                            steering = steering[:min_len]
                        if lateral_error is not None:
                            lateral_error = lateral_error[:min_len]
                        if heading_error is not None:
                            heading_error = heading_error[:min_len]
                        if is_straight is not None:
                            is_straight = is_straight[:min_len]
                        if path_curvature is not None:
                            path_curvature = path_curvature[:min_len]
                        if gt_curvature is not None:
                            gt_curvature = gt_curvature[:min_len]
                        if control_timestamps is not None:
                            control_timestamps = control_timestamps[:min_len]
                        if gt_center is not None:
                            gt_center = gt_center[:min_len]
                        if perception_center is not None:
                            perception_center = perception_center[:min_len]
                        if speed is not None:
                            speed = speed[:min_len]
                        if throttle is not None:
                            throttle = throttle[:min_len]
                        if brake is not None:
                            brake = brake[:min_len]
                        if target_speed is not None:
                            target_speed = target_speed[:min_len]
                        if road_heading_deg is not None:
                            road_heading_deg = road_heading_deg[:min_len]
                        if steering_pre_rate_limit is not None:
                            steering_pre_rate_limit = steering_pre_rate_limit[:min_len]
                        if steering_post_rate_limit is not None:
                            steering_post_rate_limit = steering_post_rate_limit[:min_len]
                        if steering_post_jerk_limit is not None:
                            steering_post_jerk_limit = steering_post_jerk_limit[:min_len]
                        if steering_post_hard_clip is not None:
                            steering_post_hard_clip = steering_post_hard_clip[:min_len]
                        if steering_post_smoothing is not None:
                            steering_post_smoothing = steering_post_smoothing[:min_len]
                        if steering_rate_limited_delta is not None:
                            steering_rate_limited_delta = steering_rate_limited_delta[:min_len]
                        if steering_jerk_limited_delta is not None:
                            steering_jerk_limited_delta = steering_jerk_limited_delta[:min_len]
                        if steering_hard_clip_delta is not None:
                            steering_hard_clip_delta = steering_hard_clip_delta[:min_len]
                        if steering_smoothing_delta is not None:
                            steering_smoothing_delta = steering_smoothing_delta[:min_len]
            
            # ANALYSIS 1: TRAJECTORY REFERENCE POINT
            # Check if we have any data
            if len(ref_x) == 0:
                return {
                    "error": "No trajectory data available after filtering. Recording may be too short or all data was filtered out.",
                    "data_availability": {
                        "trajectory": has_trajectory,
                        "control": has_control,
                        "ground_truth": has_ground_truth,
                        "perception": has_perception
                    }
                }
            
            ref_x_abs = np.abs(ref_x)
            mean_ref_x = safe_float(np.mean(ref_x_abs) if len(ref_x_abs) > 0 else 0.0)
            max_ref_x = safe_float(np.max(ref_x_abs) if len(ref_x_abs) > 0 else 0.0)
            std_ref_x = safe_float(np.std(ref_x) if len(ref_x) > 0 else 0.0)
            
            ref_x_change = np.abs(np.diff(ref_x)) if len(ref_x) > 1 else np.array([])
            mean_change = safe_float(np.mean(ref_x_change) if len(ref_x_change) > 0 else 0.0)
            max_change = safe_float(np.max(ref_x_change) if len(ref_x_change) > 0 else 0.0)
            
            trajectory_quality_score = 100.0
            trajectory_issues = []
            trajectory_warnings = []
            
            if mean_ref_x < 0.1 and max_ref_x < 0.2:
                trajectory_issues.append("Reference point very close to center (not planning ahead on curves)")
                trajectory_quality_score -= 30
            elif mean_ref_x < 0.2:
                trajectory_warnings.append("Reference point somewhat close to center")
                trajectory_quality_score -= 10
            
            if mean_change < 0.01:
                trajectory_issues.append("Reference point barely changes (not adapting to curves)")
                trajectory_quality_score -= 25
            elif mean_change < 0.05:
                trajectory_warnings.append("Reference point changes slowly")
                trajectory_quality_score -= 10
            
            trajectory_quality_score = max(0, trajectory_quality_score)
            
            # ANALYSIS 2: TRAJECTORY vs GROUND TRUTH
            trajectory_accuracy = None
            if gt_center is not None:
                ref_vs_gt = ref_x - gt_center
                mean_error = safe_float(np.mean(ref_vs_gt) if len(ref_vs_gt) > 0 else 0.0)
                rmse = safe_float(np.sqrt(np.mean(ref_vs_gt**2)) if len(ref_vs_gt) > 0 else 0.0)
                max_error = safe_float(np.max(np.abs(ref_vs_gt)) if len(ref_vs_gt) > 0 else 0.0)
                
                trajectory_accuracy = {
                    "mean_error": mean_error,
                    "rmse": rmse,
                    "max_error": max_error
                }
                
                if rmse > 0.5:
                    trajectory_issues.append(f"Reference point far from ground truth (RMSE: {rmse:.3f}m)")
                    trajectory_quality_score -= 40
                elif rmse > 0.2:
                    trajectory_warnings.append(f"Reference point has moderate errors vs ground truth (RMSE: {rmse:.3f}m)")
                    trajectory_quality_score -= 15
            else:
                trajectory_accuracy = {
                    "mean_error": None,
                    "rmse": None,
                    "max_error": None
                }

            # ANALYSIS 2.5: PERCEPTION -> TRAJECTORY ATTRIBUTION
            perception_trajectory_attribution = None
            perception_trajectory_hotspots = []
            # Build time axis early since attribution hotspots use it.
            if control_timestamps is not None and len(control_timestamps) > 0:
                time_axis = control_timestamps - control_timestamps[0]
            else:
                time_axis = np.arange(min_len, dtype=np.float32) * 0.033
            if (
                perception_center is not None
                and len(perception_center) == len(ref_x)
                and len(ref_x) > 5
            ):
                ref_vs_perception = ref_x - perception_center
                ref_vs_perception_rmse = safe_float(
                    np.sqrt(np.mean(ref_vs_perception**2))
                    if len(ref_vs_perception) > 0
                    else 0.0
                )
                ref_vs_perception_bias = safe_float(
                    np.mean(ref_vs_perception) if len(ref_vs_perception) > 0 else 0.0
                )
                ref_vs_perception_corr = None
                if np.std(ref_x) > 1e-6 and np.std(perception_center) > 1e-6:
                    ref_vs_perception_corr = safe_float(
                        np.corrcoef(ref_x, perception_center)[0, 1]
                    )

                # Best lag (in frames) for perception_center -> ref_x coupling.
                best_lag_frames = None
                best_lag_corr = None
                max_lag = 12
                for lag in range(-max_lag, max_lag + 1):
                    if lag < 0:
                        p = perception_center[-lag:]
                        r = ref_x[: len(p)]
                    elif lag > 0:
                        p = perception_center[:-lag]
                        r = ref_x[lag:]
                    else:
                        p = perception_center
                        r = ref_x
                    if len(p) < 10:
                        continue
                    if np.std(p) <= 1e-6 or np.std(r) <= 1e-6:
                        continue
                    c = safe_float(np.corrcoef(p, r)[0, 1])
                    if best_lag_corr is None or abs(c) > abs(best_lag_corr):
                        best_lag_corr = c
                        best_lag_frames = lag

                gain = None
                intercept = None
                if len(ref_x) >= 10 and np.std(perception_center) > 1e-6:
                    try:
                        g, b = np.polyfit(perception_center, ref_x, 1)
                        gain = safe_float(g)
                        intercept = safe_float(b)
                    except Exception:
                        gain = None
                        intercept = None

                gt_perception_rmse = None
                gt_ref_rmse = None
                attribution_label = "unknown"
                if gt_center is not None and len(gt_center) == len(ref_x):
                    gt_perception_rmse = safe_float(
                        np.sqrt(np.mean((perception_center - gt_center) ** 2))
                    )
                    gt_ref_rmse = safe_float(np.sqrt(np.mean((ref_x - gt_center) ** 2)))

                    if gt_ref_rmse < 0.20:
                        attribution_label = "trajectory-close-to-ground-truth"
                    elif (
                        ref_vs_perception_rmse < 0.15
                        and (ref_vs_perception_corr is not None and ref_vs_perception_corr > 0.8)
                        and gt_perception_rmse > 0.20
                    ):
                        attribution_label = "perception-driven-trajectory-error"
                    elif (
                        ref_vs_perception_rmse > 0.25
                        and (ref_vs_perception_corr is not None and ref_vs_perception_corr < 0.5)
                        and gt_perception_rmse < 0.20
                    ):
                        attribution_label = "trajectory-logic-driven-error"
                    else:
                        attribution_label = "mixed-perception-trajectory-coupling"

                    # Hotspots: where perception-vs-GT discrepancy is largest.
                    pg_diff = perception_center - gt_center
                    sorted_idx = np.argsort(np.abs(pg_diff))[::-1]
                    min_separation = 10
                    for idx in sorted_idx:
                        if len(perception_trajectory_hotspots) >= 10:
                            break
                        if abs(pg_diff[idx]) < 0.15:
                            break
                        if any(
                            abs(idx - h["frame"]) < min_separation
                            for h in perception_trajectory_hotspots
                        ):
                            continue
                        curvature_val = float(path_curvature[idx]) if path_curvature is not None else 0.0
                        gt_curvature_val = float(gt_curvature[idx]) if gt_curvature is not None else 0.0
                        is_curve = abs(curvature_val) >= 0.01
                        perception_trajectory_hotspots.append(
                            {
                                "frame": int(idx),
                                "time": safe_float(time_axis[idx]) if idx < len(time_axis) else 0.0,
                                "gt_center_x": safe_float(float(gt_center[idx])),
                                "perception_center_x": safe_float(float(perception_center[idx])),
                                "ref_x": safe_float(float(ref_x[idx])),
                                "perception_vs_gt": safe_float(float(perception_center[idx] - gt_center[idx])),
                                "ref_vs_gt": safe_float(float(ref_x[idx] - gt_center[idx])),
                                "ref_vs_perception": safe_float(float(ref_x[idx] - perception_center[idx])),
                                "curvature": safe_float(curvature_val),
                                "gt_curvature": safe_float(gt_curvature_val),
                                "segment": "curve" if is_curve else "straight",
                            }
                        )

                perception_trajectory_attribution = {
                    "ref_vs_perception_rmse": ref_vs_perception_rmse,
                    "ref_vs_perception_bias": ref_vs_perception_bias,
                    "ref_vs_perception_correlation": ref_vs_perception_corr,
                    "best_lag_frames": int(best_lag_frames) if best_lag_frames is not None else None,
                    "best_lag_correlation": best_lag_corr,
                    "gain_perception_to_ref": gain,
                    "intercept_perception_to_ref": intercept,
                    "gt_perception_rmse": gt_perception_rmse,
                    "gt_ref_rmse": gt_ref_rmse,
                    "attribution_label": attribution_label,
                    "stale_perception_rate_pct": (
                        safe_float(np.mean(using_stale_perception) * 100.0)
                        if using_stale_perception is not None and len(using_stale_perception) > 0
                        else None
                    ),
                }
            
            # Build distance axis from speed and time.
            distance_axis = None
            if speed is not None and len(speed) == min_len:
                dt = np.diff(time_axis, prepend=time_axis[0])
                dt = np.maximum(dt, 0.0)
                if len(dt) > 0:
                    dt[0] = 0.0
                distance_axis = np.cumsum(speed * dt)

            # ANALYSIS 3: STEERING RESPONSE
            control_quality_score = 100.0
            steering_issues = []
            steering_warnings = []
            
            steering_correlation = None
            steering_correlation_scope = "all"
            if steering is not None:
                steering_abs = np.abs(steering)
                mean_steering = safe_float(np.mean(steering_abs) if len(steering_abs) > 0 else 0.0)
                max_steering = safe_float(np.max(steering_abs) if len(steering_abs) > 0 else 0.0)
                std_steering = safe_float(np.std(steering) if len(steering) > 0 else 0.0)
                
                steering_change = np.abs(np.diff(steering)) if len(steering) > 1 else np.array([])
                mean_steering_change = safe_float(np.mean(steering_change) if len(steering_change) > 0 else 0.0)
                max_steering_change = safe_float(np.max(steering_change) if len(steering_change) > 0 else 0.0)
                
                if mean_steering < 0.05 and max_steering < 0.1:
                    steering_warnings.append("Steering is very small (may not be turning enough)")
                    control_quality_score -= 20
                elif mean_steering < 0.1:
                    steering_warnings.append("Steering is somewhat small")
                    control_quality_score -= 10
                
                if mean_steering_change < 0.01:
                    steering_warnings.append("Steering barely changes (not responding to curves)")
                    control_quality_score -= 15
                
                # Check steering vs reference point correlation (prefer straight, low-curvature segments)
                if len(ref_x) == len(steering):
                    corr_ref = ref_x
                    corr_steer = steering
                    corr_mask = None
                    if is_straight is not None:
                        corr_mask = is_straight.copy()
                        steering_correlation_scope = "straight"
                    if path_curvature is not None:
                        curvature_mask = np.abs(path_curvature) < 0.01
                        corr_mask = curvature_mask if corr_mask is None else (corr_mask & curvature_mask)
                        steering_correlation_scope = (
                            "straight_low_curvature" if corr_mask is not None else "low_curvature"
                        )
                    if corr_mask is not None and np.any(corr_mask):
                        corr_ref = ref_x[corr_mask]
                        corr_steer = steering[corr_mask]
                    if len(corr_ref) > 1 and np.std(corr_ref) > 1e-6 and np.std(corr_steer) > 1e-6:
                        correlation = safe_float(np.corrcoef(corr_ref, corr_steer)[0, 1])
                        steering_correlation = correlation
                        
                        if correlation < 0.3:
                            steering_issues.append(
                                f"Low correlation between steering and reference point ({correlation:.3f})"
                            )
                            control_quality_score -= 30
                        elif correlation < 0.5:
                            steering_warnings.append(f"Moderate correlation ({correlation:.3f})")
                            control_quality_score -= 10
            else:
                mean_steering = None
                max_steering = None
                std_steering = None
                mean_steering_change = None
                max_steering_change = None
            
            # ANALYSIS 4: CONTROL ERRORS
            # Use RMSE (Root Mean Squared Error) to match Path Tracking metrics
            # RMSE penalizes large errors more than mean absolute error
            lateral_error_analysis = None
            heading_error_analysis = None
            
            if lateral_error is not None:
                lateral_error_abs = np.abs(lateral_error)
                mean_lateral_error = safe_float(np.mean(lateral_error_abs) if len(lateral_error_abs) > 0 else 0.0)
                max_lateral_error = safe_float(np.max(lateral_error_abs) if len(lateral_error_abs) > 0 else 0.0)
                # RMSE: sqrt(mean(squared errors)) - matches Path Tracking calculation
                lateral_error_rmse = safe_float(np.sqrt(np.mean(lateral_error**2)) if len(lateral_error) > 0 else 0.0)
                
                lateral_error_analysis = {
                    "mean": mean_lateral_error,
                    "rmse": lateral_error_rmse,  # Add RMSE to match Path Tracking
                    "max": max_lateral_error
                }
                
                # Use RMSE for penalties (matches Path Tracking thresholds)
                # RMSE thresholds: Good <0.2m, Acceptable <0.4m, Poor >=0.4m
                if lateral_error_rmse > 0.4:
                    steering_issues.append(f"Large lateral errors (RMSE: {lateral_error_rmse:.3f}m, mean: {mean_lateral_error:.3f}m)")
                    control_quality_score -= 40  # Increased penalty for poor RMSE
                elif lateral_error_rmse > 0.2:
                    steering_warnings.append(f"Moderate lateral errors (RMSE: {lateral_error_rmse:.3f}m, mean: {mean_lateral_error:.3f}m)")
                    control_quality_score -= 20  # Increased penalty for acceptable RMSE
                elif lateral_error_rmse > 0.1:
                    # Small but not negligible
                    control_quality_score -= 5
            
            if heading_error is not None:
                heading_error_abs = np.abs(heading_error)
                mean_heading_error = safe_float(np.mean(heading_error_abs) if len(heading_error_abs) > 0 else 0.0)
                max_heading_error = safe_float(np.max(heading_error_abs) if len(heading_error_abs) > 0 else 0.0)
                
                heading_error_analysis = {
                    "mean_deg": safe_float(np.degrees(mean_heading_error)),
                    "max_deg": safe_float(np.degrees(max_heading_error))
                }
            
            control_quality_score = max(0, control_quality_score)

            # Lateral error hotspots (top-N frames)
            hotspots = []
            if lateral_error is not None and len(lateral_error) > 0:
                abs_error = np.abs(lateral_error)
                sorted_idx = np.argsort(abs_error)[::-1]
                min_separation = 15
                for idx in sorted_idx:
                    if len(hotspots) >= 8:
                        break
                    if abs_error[idx] <= 0.0:
                        break
                    if any(abs(idx - h["frame"]) < min_separation for h in hotspots):
                        continue
                    curvature_val = float(path_curvature[idx]) if path_curvature is not None else 0.0
                    gt_curvature_val = float(gt_curvature[idx]) if gt_curvature is not None else 0.0
                    is_curve = abs(curvature_val) >= 0.01
                    straight_flag = bool(is_straight[idx]) if is_straight is not None else None
                    ref_x_val = float(ref_x[idx]) if ref_x is not None else 0.0
                    steer_val = float(steering[idx]) if steering is not None else 0.0
                    heading_val = float(heading_error[idx]) if heading_error is not None else 0.0
                    sign_mismatch = (
                        ref_x_val != 0.0
                        and steer_val != 0.0
                        and np.sign(ref_x_val) != np.sign(steer_val)
                    )
                    hotspots.append({
                        "frame": int(idx),
                        "time": safe_float(time_axis[idx]) if idx < len(time_axis) else 0.0,
                        "lateral_error": safe_float(float(lateral_error[idx])),
                        "steering": safe_float(steer_val),
                        "ref_x": safe_float(ref_x_val),
                        "heading_error": safe_float(heading_val),
                        "is_straight": straight_flag,
                        "curvature": safe_float(curvature_val),
                        "gt_curvature": safe_float(gt_curvature_val),
                        "segment": "curve" if is_curve else "straight",
                        "sign_mismatch": bool(sign_mismatch),
                    })

            # Lateral error disagreement (control vs ground truth)
            disagreement_hotspots = []
            if gt_center is not None and lateral_error is not None and len(gt_center) > 0:
                gt_error = -gt_center
                n_compare = min(len(gt_error), len(lateral_error))
                diff = (lateral_error[:n_compare] - gt_error[:n_compare])
                abs_diff = np.abs(diff)
                sorted_idx = np.argsort(abs_diff)[::-1]
                min_separation = 10
                for idx in sorted_idx:
                    if len(disagreement_hotspots) >= 10:
                        break
                    if abs_diff[idx] < 0.3:
                        break
                    if any(abs(idx - h["frame"]) < min_separation for h in disagreement_hotspots):
                        continue
                    curvature_val = float(path_curvature[idx]) if path_curvature is not None else 0.0
                    gt_curvature_val = float(gt_curvature[idx]) if gt_curvature is not None else 0.0
                    is_curve = abs(curvature_val) >= 0.01
                    straight_flag = bool(is_straight[idx]) if is_straight is not None else None
                    steer_val = float(steering[idx]) if steering is not None else 0.0
                    disagreement_hotspots.append({
                        "frame": int(idx),
                        "time": safe_float(time_axis[idx]) if idx < len(time_axis) else 0.0,
                        "lateral_error": safe_float(float(lateral_error[idx])),
                        "gt_error": safe_float(float(gt_error[idx])),
                        "diff": safe_float(float(diff[idx])),
                        "steering": safe_float(steer_val),
                        "is_straight": straight_flag,
                        "curvature": safe_float(curvature_val),
                        "gt_curvature": safe_float(gt_curvature_val),
                        "segment": "curve" if is_curve else "straight",
                    })

            # GT vs perception lateral-error side-by-side hotspots
            gt_perception_hotspots = []
            if gt_center is not None and perception_center is not None and len(gt_center) > 0:
                gt_error = -gt_center
                perception_error = -perception_center
                n_compare = min(len(gt_error), len(perception_error))
                diff = (perception_error[:n_compare] - gt_error[:n_compare])
                abs_diff = np.abs(diff)
                sorted_idx = np.argsort(abs_diff)[::-1]
                min_separation = 10
                for idx in sorted_idx:
                    if len(gt_perception_hotspots) >= 10:
                        break
                    if abs_diff[idx] < 0.2:
                        break
                    if any(abs(idx - h["frame"]) < min_separation for h in gt_perception_hotspots):
                        continue
                    curvature_val = float(path_curvature[idx]) if path_curvature is not None else 0.0
                    gt_curvature_val = float(gt_curvature[idx]) if gt_curvature is not None else 0.0
                    is_curve = abs(curvature_val) >= 0.01
                    straight_flag = bool(is_straight[idx]) if is_straight is not None else None
                    gt_perception_hotspots.append({
                        "frame": int(idx),
                        "time": safe_float(time_axis[idx]) if idx < len(time_axis) else 0.0,
                        "gt_center_x": safe_float(float(gt_center[idx])),
                        "perception_center_x": safe_float(float(perception_center[idx])),
                        "gt_error": safe_float(float(gt_error[idx])),
                        "perception_error": safe_float(float(perception_error[idx])),
                        "diff": safe_float(float(diff[idx])),
                        "control_lateral_error": (
                            safe_float(float(lateral_error[idx]))
                            if lateral_error is not None and idx < len(lateral_error)
                            else None
                        ),
                        "is_straight": straight_flag,
                        "curvature": safe_float(curvature_val),
                        "gt_curvature": safe_float(gt_curvature_val),
                        "segment": "curve" if is_curve else "straight",
                    })

            # Longitudinal accel/jerk hotspots (top-N frames)
            accel_hotspots = []
            jerk_hotspots = []
            if speed is not None and len(speed) > 1:
                n_series = min(len(speed), len(time_axis))
                speed_series = speed[:n_series]
                time_series = time_axis[:n_series]
                dt_series = np.diff(time_series)
                accel = np.zeros(n_series, dtype=np.float32)
                valid_dt = dt_series > 1e-6
                accel[1:] = np.divide(
                    np.diff(speed_series),
                    dt_series,
                    out=np.zeros_like(dt_series, dtype=np.float32),
                    where=valid_dt,
                )
                jerk = np.zeros(n_series, dtype=np.float32)
                if n_series > 2:
                    jerk_dt = dt_series[1:]
                    valid_jerk_dt = jerk_dt > 1e-6
                    jerk[2:] = np.divide(
                        np.diff(accel[1:]),
                        jerk_dt,
                        out=np.zeros_like(jerk_dt, dtype=np.float32),
                        where=valid_jerk_dt,
                    )

                def build_hotspots(values, max_items=8, min_separation=15):
                    output = []
                    sorted_idx = np.argsort(np.abs(values))[::-1]
                    for idx in sorted_idx:
                        if len(output) >= max_items:
                            break
                        if abs(values[idx]) <= 0.0:
                            break
                        if any(abs(idx - h["frame"]) < min_separation for h in output):
                            continue
                        curvature_val = float(path_curvature[idx]) if path_curvature is not None and idx < len(path_curvature) else 0.0
                        gt_curvature_val = float(gt_curvature[idx]) if gt_curvature is not None and idx < len(gt_curvature) else 0.0
                        is_curve = abs(curvature_val) >= 0.01
                        output.append({
                            "frame": int(idx),
                            "time": safe_float(time_series[idx]) if idx < len(time_series) else 0.0,
                            "speed": safe_float(float(speed_series[idx])) if idx < len(speed_series) else 0.0,
                            "accel": safe_float(float(accel[idx])) if idx < len(accel) else 0.0,
                            "jerk": safe_float(float(jerk[idx])) if idx < len(jerk) else 0.0,
                            "throttle": safe_float(float(throttle[idx])) if throttle is not None and idx < len(throttle) else None,
                            "brake": safe_float(float(brake[idx])) if brake is not None and idx < len(brake) else None,
                            "target_speed": safe_float(float(target_speed[idx])) if target_speed is not None and idx < len(target_speed) else None,
                            "steering": safe_float(float(steering[idx])) if steering is not None and idx < len(steering) else None,
                            "curvature": safe_float(curvature_val),
                            "gt_curvature": safe_float(gt_curvature_val),
                            "segment": "curve" if is_curve else "straight",
                        })
                    return output

                accel_hotspots = build_hotspots(accel)
                jerk_hotspots = build_hotspots(jerk)

            # ANALYSIS 5: STEERING LIMITER ROOT CAUSE ATTRIBUTION
            steering_limiter_analysis = None
            steering_limiter_hotspots = []
            curve_entry_feasibility = None
            if steering is not None and len(steering) > 0:
                rate_delta = None
                jerk_delta = None
                hard_clip_delta = None
                smoothing_delta = None

                if steering_rate_limited_delta is not None:
                    rate_delta = np.abs(steering_rate_limited_delta)
                elif steering_pre_rate_limit is not None and steering_post_rate_limit is not None:
                    rate_delta = np.abs(steering_pre_rate_limit - steering_post_rate_limit)

                if steering_jerk_limited_delta is not None:
                    jerk_delta = np.abs(steering_jerk_limited_delta)
                elif steering_post_rate_limit is not None and steering_post_jerk_limit is not None:
                    jerk_delta = np.abs(steering_post_rate_limit - steering_post_jerk_limit)

                if steering_hard_clip_delta is not None:
                    hard_clip_delta = np.abs(steering_hard_clip_delta)
                elif steering_post_jerk_limit is not None and steering_post_hard_clip is not None:
                    hard_clip_delta = np.abs(steering_post_jerk_limit - steering_post_hard_clip)

                if steering_smoothing_delta is not None:
                    smoothing_delta = np.abs(steering_smoothing_delta)
                elif steering_post_hard_clip is not None and steering_post_smoothing is not None:
                    smoothing_delta = np.abs(steering_post_hard_clip - steering_post_smoothing)

                stage_map = {}
                if rate_delta is not None:
                    stage_map["rate_limit"] = rate_delta
                if jerk_delta is not None:
                    stage_map["jerk_limit"] = jerk_delta
                if hard_clip_delta is not None:
                    stage_map["hard_clip"] = hard_clip_delta
                if smoothing_delta is not None:
                    stage_map["smoothing"] = smoothing_delta

                dominance_threshold = 1e-4
                dominant_counts = {k: 0 for k in stage_map}
                total_limited_frames = 0
                if stage_map:
                    stacked = np.vstack([v for v in stage_map.values()])
                    max_vals = np.max(stacked, axis=0)
                    max_idx = np.argmax(stacked, axis=0)
                    keys = list(stage_map.keys())
                    for i, max_val in enumerate(max_vals):
                        if max_val <= dominance_threshold:
                            continue
                        total_limited_frames += 1
                        dominant_counts[keys[int(max_idx[i])]] += 1

                    # Build hotspots where limiter impact was highest.
                    min_separation = 12
                    sorted_idx = np.argsort(max_vals)[::-1]
                    for idx in sorted_idx:
                        if len(steering_limiter_hotspots) >= 10:
                            break
                        if max_vals[idx] <= dominance_threshold:
                            break
                        if any(abs(idx - h["frame"]) < min_separation for h in steering_limiter_hotspots):
                            continue
                        dominant_stage = keys[int(max_idx[idx])]
                        steering_limiter_hotspots.append({
                            "frame": int(idx),
                            "time": safe_float(time_axis[idx]) if idx < len(time_axis) else 0.0,
                            "dominant_stage": dominant_stage,
                            "dominant_delta": safe_float(float(max_vals[idx])),
                            "rate_delta": safe_float(float(rate_delta[idx])) if rate_delta is not None else 0.0,
                            "jerk_delta": safe_float(float(jerk_delta[idx])) if jerk_delta is not None else 0.0,
                            "hard_clip_delta": safe_float(float(hard_clip_delta[idx])) if hard_clip_delta is not None else 0.0,
                            "smoothing_delta": safe_float(float(smoothing_delta[idx])) if smoothing_delta is not None else 0.0,
                            "steering": safe_float(float(steering[idx])) if idx < len(steering) else 0.0,
                            "lateral_error": safe_float(float(lateral_error[idx])) if lateral_error is not None and idx < len(lateral_error) else None,
                            "curvature": safe_float(float(path_curvature[idx])) if path_curvature is not None and idx < len(path_curvature) else 0.0,
                        })

                def analyze_limiter_window(start_idx: int, end_idx: int) -> Dict:
                    start_idx = max(0, int(start_idx))
                    end_idx = min(min_len, int(end_idx))
                    if end_idx <= start_idx or not stage_map:
                        return {
                            "start_frame": start_idx,
                            "end_frame": end_idx,
                            "frames": max(0, end_idx - start_idx),
                            "limited_frames": 0,
                            "dominant_stage": "none",
                            "dominant_pct": 0.0,
                            "rate_delta_mean": 0.0,
                            "jerk_delta_mean": 0.0,
                            "hard_clip_delta_mean": 0.0,
                            "smoothing_delta_mean": 0.0,
                        }
                    window_arr = np.vstack([v[start_idx:end_idx] for v in stage_map.values()])
                    window_max = np.max(window_arr, axis=0)
                    window_idx = np.argmax(window_arr, axis=0)
                    limited_mask = window_max > dominance_threshold
                    limited_frames = int(np.sum(limited_mask))
                    counts = {k: 0 for k in stage_map}
                    keys_local = list(stage_map.keys())
                    for i, limited in enumerate(limited_mask):
                        if not limited:
                            continue
                        counts[keys_local[int(window_idx[i])]] += 1
                    dominant_local = "none"
                    dominant_pct_local = 0.0
                    if limited_frames > 0:
                        dominant_local = max(counts, key=counts.get)
                        dominant_pct_local = 100.0 * counts[dominant_local] / max(1, limited_frames)
                    return {
                        "start_frame": start_idx,
                        "end_frame": end_idx,
                        "frames": int(end_idx - start_idx),
                        "limited_frames": limited_frames,
                        "dominant_stage": dominant_local,
                        "dominant_pct": safe_float(dominant_pct_local),
                        "rate_delta_mean": safe_float(np.mean(rate_delta[start_idx:end_idx]) if rate_delta is not None else 0.0),
                        "jerk_delta_mean": safe_float(np.mean(jerk_delta[start_idx:end_idx]) if jerk_delta is not None else 0.0),
                        "hard_clip_delta_mean": safe_float(np.mean(hard_clip_delta[start_idx:end_idx]) if hard_clip_delta is not None else 0.0),
                        "smoothing_delta_mean": safe_float(np.mean(smoothing_delta[start_idx:end_idx]) if smoothing_delta is not None else 0.0),
                    }

                # Phase-specific limiter attribution to isolate curve-entry bottlenecks.
                curve_start = 0
                curve_start_source = "none"
                if road_heading_deg is not None and len(road_heading_deg) > 0:
                    base_heading = float(np.median(road_heading_deg[:min(60, len(road_heading_deg))]))
                    curve_candidates = np.where(np.abs(road_heading_deg - base_heading) >= 2.0)[0]
                    if len(curve_candidates) > 0:
                        curve_start = int(curve_candidates[0])
                        curve_start_source = "road_heading_deg"
                if curve_start_source == "none" and path_curvature is not None and len(path_curvature) > 0:
                    curve_candidates = np.where(np.abs(path_curvature) >= 0.0005)[0]
                    if len(curve_candidates) > 0:
                        curve_start = int(curve_candidates[0])
                        curve_start_source = "path_curvature"
                if curve_start_source == "none" and gt_curvature is not None and len(gt_curvature) > 0:
                    curve_candidates = np.where(np.abs(gt_curvature) >= 0.002)[0]
                    if len(curve_candidates) > 0:
                        curve_start = int(curve_candidates[0])
                        curve_start_source = "ground_truth_curvature"

                entry_frames = 20
                pre_frames = 20
                pre_start = max(0, curve_start - pre_frames)
                pre_end = curve_start
                entry_end = min(min_len, curve_start + entry_frames)
                phase_breakdown = {
                    "curve_start_frame": int(curve_start),
                    "curve_start_source": curve_start_source,
                    "pre_curve": analyze_limiter_window(pre_start, pre_end),
                    "curve_entry": analyze_limiter_window(curve_start, entry_end),
                    "curve_maintain": analyze_limiter_window(entry_end, min_len),
                    "overall": analyze_limiter_window(0, min_len),
                }

                dominant_stage = "none"
                dominant_stage_pct = 0.0
                if total_limited_frames > 0 and dominant_counts:
                    dominant_stage = max(dominant_counts, key=dominant_counts.get)
                    dominant_stage_pct = 100.0 * dominant_counts[dominant_stage] / max(1, total_limited_frames)

                steering_limiter_analysis = {
                    "available": bool(stage_map),
                    "dominant_stage": dominant_stage,
                    "dominant_stage_pct": safe_float(dominant_stage_pct),
                    "total_limited_frames": int(total_limited_frames),
                    "rate_delta_mean": safe_float(np.mean(rate_delta) if rate_delta is not None and len(rate_delta) > 0 else 0.0),
                    "jerk_delta_mean": safe_float(np.mean(jerk_delta) if jerk_delta is not None and len(jerk_delta) > 0 else 0.0),
                    "hard_clip_delta_mean": safe_float(np.mean(hard_clip_delta) if hard_clip_delta is not None and len(hard_clip_delta) > 0 else 0.0),
                    "smoothing_delta_mean": safe_float(np.mean(smoothing_delta) if smoothing_delta is not None and len(smoothing_delta) > 0 else 0.0),
                    "dominant_counts": {k: int(v) for k, v in dominant_counts.items()},
                    "phase_breakdown": phase_breakdown,
                }

                if total_limited_frames > 0:
                    if dominant_stage == "rate_limit" and dominant_stage_pct >= 40.0:
                        steering_issues.append(
                            f"Steering rate limit dominates command shaping ({dominant_stage_pct:.1f}% of limited frames)"
                        )
                    elif dominant_stage == "jerk_limit" and dominant_stage_pct >= 35.0:
                        steering_issues.append(
                            f"Steering jerk limit dominates command shaping ({dominant_stage_pct:.1f}% of limited frames)"
                        )
                    elif dominant_stage == "hard_clip" and dominant_stage_pct >= 25.0:
                        steering_issues.append(
                            f"Hard steering clip dominates ({dominant_stage_pct:.1f}% of limited frames)"
                        )
                    elif dominant_stage == "smoothing" and dominant_stage_pct >= 35.0:
                        steering_warnings.append(
                            f"Steering smoothing dominates command shaping ({dominant_stage_pct:.1f}% of limited frames)"
                        )

                # ANALYSIS 6: CURVE ENTRY FEASIBILITY (full-story root cause)
                if (
                    speed is not None
                    and path_curvature is not None
                    and steering_pre_rate_limit is not None
                    and len(speed) > 0
                    and len(path_curvature) > 0
                    and len(steering_pre_rate_limit) > 0
                ):
                    # Use full aligned series (pre analyze_to_failure truncation) for
                    # distance-based manual windows so run-to-run distances remain comparable.
                    speed_src = speed_full if speed_full is not None else speed
                    path_curvature_src = (
                        path_curvature_full if path_curvature_full is not None else path_curvature
                    )
                    steering_pre_src = (
                        steering_pre_rate_limit_full
                        if steering_pre_rate_limit_full is not None
                        else steering_pre_rate_limit
                    )
                    steering_final = (
                        steering_post_smoothing_full
                        if steering_post_smoothing_full is not None
                        else (steering_full if steering_full is not None else steering)
                    )
                    stale_src = (
                        using_stale_perception_full
                        if using_stale_perception_full is not None
                        else using_stale_perception
                    )
                    rate_src = (
                        steering_rate_limited_delta_full
                        if steering_rate_limited_delta_full is not None
                        else rate_delta
                    )
                    jerk_src = (
                        steering_jerk_limited_delta_full
                        if steering_jerk_limited_delta_full is not None
                        else jerk_delta
                    )
                    clip_src = (
                        steering_hard_clip_delta_full
                        if steering_hard_clip_delta_full is not None
                        else hard_clip_delta
                    )
                    smooth_src = (
                        steering_smoothing_delta_full
                        if steering_smoothing_delta_full is not None
                        else smoothing_delta
                    )
                    authority_gap_src = (
                        steering_authority_gap_full
                        if steering_authority_gap_full is not None
                        else None
                    )
                    first_limiter_stage_src = (
                        steering_first_limiter_stage_code_full
                        if steering_first_limiter_stage_code_full is not None
                        else None
                    )
                    if control_timestamps_full is not None and len(control_timestamps_full) > 0:
                        time_axis_src = control_timestamps_full - control_timestamps_full[0]
                    else:
                        time_axis_src = np.arange(len(speed_src), dtype=np.float32) * 0.033

                    # Prefer fixed/manual distance window for run-to-run comparability.
                    # Defaults are tuned to current s-loop runs where frame~174 is ~30-35m.
                    start_dist_m = (
                        float(curve_entry_start_distance_m)
                        if curve_entry_start_distance_m is not None
                        else 34.0
                    )
                    window_dist_m = (
                        float(curve_entry_window_distance_m)
                        if curve_entry_window_distance_m is not None
                        else 8.0
                    )

                    curve_start2 = 0
                    curve_start_source2 = "distance_manual"
                    entry_start2 = 0
                    entry_end2 = min_len
                    entry_start_distance_used = 0.0
                    entry_end_distance_used = 0.0

                    dt_src = np.diff(time_axis_src, prepend=time_axis_src[0])
                    dt_src = np.maximum(dt_src, 0.0)
                    if len(dt_src) > 0:
                        dt_src[0] = 0.0
                    distance_axis_src = np.cumsum(speed_src * dt_src)

                    if len(distance_axis_src) == len(speed_src) and len(distance_axis_src) > 0:
                        start_dist_m = max(0.0, start_dist_m)
                        window_dist_m = max(0.1, window_dist_m)
                        end_dist_m = start_dist_m + window_dist_m
                        entry_start2 = int(np.searchsorted(distance_axis_src, start_dist_m, side="left"))
                        entry_end2 = int(np.searchsorted(distance_axis_src, end_dist_m, side="left"))
                        src_len = len(speed_src)
                        entry_start2 = int(np.clip(entry_start2, 0, max(0, src_len - 1)))
                        entry_end2 = int(np.clip(entry_end2, entry_start2 + 1, src_len))
                        curve_start2 = entry_start2
                        entry_start_distance_used = safe_float(distance_axis_src[entry_start2])
                        entry_end_distance_used = safe_float(distance_axis_src[max(entry_start2, entry_end2 - 1)])
                    else:
                        # Fallback only if distance cannot be constructed.
                        curve_start_source2 = "fallback_frame_heuristic"
                        if road_heading_deg is not None and len(road_heading_deg) > 0:
                            base_heading = float(np.median(road_heading_deg[:min(60, len(road_heading_deg))]))
                            curve_candidates = np.where(np.abs(road_heading_deg - base_heading) >= 2.0)[0]
                            if len(curve_candidates) > 0:
                                curve_start2 = int(curve_candidates[0])
                                curve_start_source2 = "road_heading_deg"
                        if curve_start_source2 == "fallback_frame_heuristic":
                            curve_candidates = np.where(np.abs(path_curvature) >= 0.0005)[0]
                            if len(curve_candidates) > 0:
                                curve_start2 = int(curve_candidates[0])
                                curve_start_source2 = "path_curvature"
                        entry_frames2 = 20
                        entry_start2 = max(0, int(curve_start2))
                        entry_end2 = min(len(speed_src), entry_start2 + entry_frames2)
                        if entry_end2 <= entry_start2:
                            entry_start2 = max(0, len(speed_src) - max(1, entry_frames2))
                            entry_end2 = len(speed_src)

                    sp_e = speed_src[entry_start2:entry_end2]
                    k_e = np.abs(path_curvature_src[entry_start2:entry_end2])
                    pre_e = np.abs(steering_pre_src[entry_start2:entry_end2])
                    fin_e = np.abs(steering_final[entry_start2:entry_end2])
                    authority_gap_e_precomputed = (
                        authority_gap_src[entry_start2:entry_end2]
                        if authority_gap_src is not None
                        else None
                    )
                    first_limiter_stage_e = (
                        first_limiter_stage_src[entry_start2:entry_end2]
                        if first_limiter_stage_src is not None
                        else None
                    )
                    stale_e = (
                        stale_src[entry_start2:entry_end2]
                        if stale_src is not None
                        else None
                    )

                    if len(sp_e) > 0 and len(k_e) > 0 and len(pre_e) > 0 and len(fin_e) > 0:
                        ay_budget = 1.3
                        ay = np.square(sp_e) * k_e
                        speed_margin = ay_budget - ay
                        speed_limited_pct = 100.0 * float(np.mean(ay > ay_budget))

                        if authority_gap_e_precomputed is not None and len(authority_gap_e_precomputed) == len(pre_e):
                            authority_gap = np.maximum(authority_gap_e_precomputed, 0.0)
                        else:
                            authority_gap = np.maximum(pre_e - fin_e, 0.0)
                        transfer_ratio = np.divide(
                            fin_e,
                            np.maximum(pre_e, 1e-6),
                            out=np.zeros_like(fin_e),
                            where=np.maximum(pre_e, 1e-6) > 0,
                        )
                        transfer_ratio = np.clip(transfer_ratio, 0.0, 1.0)

                        stale_pct = (
                            100.0 * float(np.mean(stale_e))
                            if stale_e is not None and len(stale_e) > 0
                            else 0.0
                        )

                        rate_e = (
                            float(np.mean(rate_src[entry_start2:entry_end2]))
                            if rate_src is not None
                            else 0.0
                        )
                        jerk_e = (
                            float(np.mean(jerk_src[entry_start2:entry_end2]))
                            if jerk_src is not None
                            else 0.0
                        )
                        clip_e = (
                            float(np.mean(clip_src[entry_start2:entry_end2]))
                            if clip_src is not None
                            else 0.0
                        )
                        smooth_e = (
                            float(np.mean(smooth_src[entry_start2:entry_end2]))
                            if smooth_src is not None
                            else 0.0
                        )
                        first_limiter_hit_frame = None
                        first_limiter_hit_stage = "none"
                        if first_limiter_stage_e is not None and len(first_limiter_stage_e) > 0:
                            hit_idx = np.where(first_limiter_stage_e > 0.5)[0]
                            if len(hit_idx) > 0:
                                first_limiter_hit_frame = int(entry_start2 + int(hit_idx[0]))
                                stage_code = int(round(float(first_limiter_stage_e[int(hit_idx[0])])))
                                stage_map = {
                                    1: "rate_limit",
                                    2: "jerk_limit",
                                    3: "hard_clip",
                                    4: "smoothing",
                                }
                                first_limiter_hit_stage = stage_map.get(stage_code, "none")
                        elif rate_src is not None or jerk_src is not None or clip_src is not None or smooth_src is not None:
                            first_hits = []
                            if rate_src is not None:
                                idxs = np.where(rate_src[entry_start2:entry_end2] > 1e-4)[0]
                                if len(idxs) > 0:
                                    first_hits.append((int(idxs[0]), "rate_limit"))
                            if jerk_src is not None:
                                idxs = np.where(jerk_src[entry_start2:entry_end2] > 1e-4)[0]
                                if len(idxs) > 0:
                                    first_hits.append((int(idxs[0]), "jerk_limit"))
                            if clip_src is not None:
                                idxs = np.where(clip_src[entry_start2:entry_end2] > 1e-4)[0]
                                if len(idxs) > 0:
                                    first_hits.append((int(idxs[0]), "hard_clip"))
                            if smooth_src is not None:
                                idxs = np.where(smooth_src[entry_start2:entry_end2] > 1e-4)[0]
                                if len(idxs) > 0:
                                    first_hits.append((int(idxs[0]), "smoothing"))
                            if first_hits:
                                first_offset, first_limiter_hit_stage = min(first_hits, key=lambda x: x[0])
                                first_limiter_hit_frame = int(entry_start2 + first_offset)

                        mean_gap = float(np.mean(authority_gap))
                        speed_limited = speed_limited_pct >= 30.0
                        authority_limited = mean_gap > 0.25
                        perception_limited = stale_pct >= 50.0

                        if speed_limited:
                            primary_classification = "speed-limited"
                        elif authority_limited:
                            primary_classification = "steering-authority-limited"
                        elif perception_limited:
                            primary_classification = "perception-limited"
                        else:
                            primary_classification = "mixed-or-unclear"

                        curve_entry_feasibility = {
                            "curve_start_frame": int(curve_start2),
                            "curve_start_source": curve_start_source2,
                            "entry_start_frame": int(entry_start2),
                            "entry_end_frame": int(entry_end2),
                            "entry_frames": int(max(0, entry_end2 - entry_start2)),
                            "entry_start_distance_target_m": safe_float(start_dist_m),
                            "entry_window_distance_target_m": safe_float(window_dist_m),
                            "entry_start_distance_used_m": safe_float(entry_start_distance_used),
                            "entry_end_distance_used_m": safe_float(entry_end_distance_used),
                            "speed_feasibility": {
                                "ay_budget": safe_float(ay_budget),
                                "ay_mean": safe_float(np.mean(ay)),
                                "ay_max": safe_float(np.max(ay)),
                                "speed_margin_mean": safe_float(np.mean(speed_margin)),
                                "speed_margin_min": safe_float(np.min(speed_margin)),
                                "speed_limited_pct": safe_float(speed_limited_pct),
                            },
                            "steering_authority": {
                                "pre_abs_mean": safe_float(np.mean(pre_e)),
                                "final_abs_mean": safe_float(np.mean(fin_e)),
                                "authority_gap_mean": safe_float(mean_gap),
                                "authority_gap_max": safe_float(np.max(authority_gap)),
                                "transfer_ratio_mean": safe_float(np.mean(transfer_ratio)),
                                "transfer_ratio_min": safe_float(np.min(transfer_ratio)),
                            },
                            "limiter_deltas_entry_mean": {
                                "rate": safe_float(rate_e),
                                "jerk": safe_float(jerk_e),
                                "hard_clip": safe_float(clip_e),
                                "smoothing": safe_float(smooth_e),
                            },
                            "first_limiter_hit_frame": first_limiter_hit_frame,
                            "first_limiter_hit_stage": first_limiter_hit_stage,
                            "stale_perception_pct": safe_float(stale_pct),
                            "primary_classification": primary_classification,
                        }

                        # Surface explicit root-cause signal in issues/warnings.
                        if primary_classification == "steering-authority-limited":
                            steering_issues.append(
                                "Curve-entry feasibility: steering authority-limited "
                                f"(mean pre-final gap={mean_gap:.3f})"
                            )
                        elif primary_classification == "speed-limited":
                            steering_issues.append(
                                "Curve-entry feasibility: speed-limited "
                                f"(v^2*kappa exceeds budget on {speed_limited_pct:.1f}% of entry frames)"
                            )
                        elif primary_classification == "perception-limited":
                            steering_warnings.append(
                                "Curve-entry feasibility: perception-limited "
                                f"(stale perception {stale_pct:.1f}% during entry)"
                            )
            
            # DIAGNOSIS SUMMARY
            # Only flag as an issue if score is actually below threshold (70%)
            # Don't flag just because one is lower than the other if both are good (>= 70%)
            primary_issue = None
            recommendations = []
            
            if trajectory_quality_score < 70:
                primary_issue = "trajectory"
                recommendations.extend([
                    "Check trajectory planner configuration",
                    "Verify perception is giving correct lane positions",
                    "Check reference_lookahead distance",
                    "Review trajectory smoothing parameters"
                ])
            elif control_quality_score < 70:
                primary_issue = "control"
                recommendations.extend([
                    "Increase steering gains (kp) in config",
                    "Check controller is receiving correct reference point",
                    "Verify steering limits are not too restrictive",
                    "Review PID tuning parameters"
                ])
            else:
                # Both scores are >= 70% (good or acceptable)
                # Only suggest improvement if one is significantly lower (>= 10% difference) and below 80%
                if trajectory_quality_score < 80 and (trajectory_quality_score < control_quality_score - 10):
                    primary_issue = "trajectory"
                    recommendations.append("Trajectory planning may need improvement")
                elif control_quality_score < 80 and (control_quality_score < trajectory_quality_score - 10):
                    primary_issue = "control"
                    recommendations.append("Control system may need tuning")
                else:
                    primary_issue = "none"
                    recommendations.append("System appears to be working correctly")

            if steering_limiter_analysis and steering_limiter_analysis.get("total_limited_frames", 0) > 0:
                phase_info = steering_limiter_analysis.get("phase_breakdown", {}) or {}
                entry_info = phase_info.get("curve_entry", {}) if isinstance(phase_info, dict) else {}
                if entry_info and entry_info.get("limited_frames", 0) > 0:
                    dominant_stage = entry_info.get("dominant_stage", "none")
                else:
                    dominant_stage = steering_limiter_analysis.get("dominant_stage", "none")
                if dominant_stage == "rate_limit":
                    recommendations.append("Steering rate limit appears dominant; consider easing curve-phase rate suppression")
                elif dominant_stage == "jerk_limit":
                    recommendations.append("Steering jerk limit appears dominant; consider relaxing jerk limit for turn-in")
                elif dominant_stage == "hard_clip":
                    recommendations.append("Hard steering clip appears dominant; check max steering headroom vs curve demand")
                elif dominant_stage == "smoothing":
                    recommendations.append("Steering smoothing appears dominant; consider reducing smoothing lag during curve entry")

            if curve_entry_feasibility:
                cls = curve_entry_feasibility.get("primary_classification", "mixed-or-unclear")
                if cls == "steering-authority-limited":
                    recommendations.append(
                        "Curve-entry feasibility indicates steering authority bottleneck; reduce early limiter suppression before increasing gains"
                    )
                elif cls == "speed-limited":
                    recommendations.append(
                        "Curve-entry feasibility indicates speed bottleneck; increase curve preview decel/lookahead to lower entry speed"
                    )
                elif cls == "perception-limited":
                    recommendations.append(
                        "Curve-entry feasibility indicates stale perception during entry; improve continuity or fallback hold behavior"
                    )
            
            return {
                "data_availability": {
                    "trajectory": True,
                    "control": has_control,
                    "ground_truth": has_ground_truth,
                    "perception": has_perception
                },
                "trajectory_analysis": {
                    "quality_score": safe_float(trajectory_quality_score),
                    "reference_point_stats": {
                        "mean_abs": mean_ref_x,
                        "max_abs": max_ref_x,
                        "std": std_ref_x,
                        "mean_change": mean_change,
                        "max_change": max_change
                    },
                    "accuracy_vs_ground_truth": trajectory_accuracy,
                    "perception_trajectory_attribution": perception_trajectory_attribution,
                    "perception_trajectory_hotspots": perception_trajectory_hotspots,
                    "issues": trajectory_issues,
                    "warnings": trajectory_warnings
                },
                "control_analysis": {
                    "quality_score": safe_float(control_quality_score),
                    "steering_stats": {
                        "mean_abs": mean_steering,
                        "max_abs": max_steering,
                        "std": std_steering,
                        "mean_change": mean_steering_change,
                        "max_change": max_steering_change
                    },
                    "steering_correlation": steering_correlation,
                    "steering_correlation_scope": steering_correlation_scope,
                    "lateral_error": lateral_error_analysis,
                    "heading_error": heading_error_analysis,
                    "issues": steering_issues,
                    "warnings": steering_warnings,
                    "lateral_error_hotspots": hotspots,
                    "lateral_error_disagreement_hotspots": disagreement_hotspots,
                    "gt_perception_hotspots": gt_perception_hotspots,
                    "accel_hotspots": accel_hotspots,
                    "jerk_hotspots": jerk_hotspots,
                    "steering_limiter_analysis": steering_limiter_analysis,
                    "steering_limiter_hotspots": steering_limiter_hotspots,
                    "curve_entry_feasibility": curve_entry_feasibility,
                },
                "diagnosis": {
                    "primary_issue": primary_issue,
                    "trajectory_score": safe_float(trajectory_quality_score),
                    "control_score": safe_float(control_quality_score),
                    "recommendations": recommendations
                }
            }
            
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}

