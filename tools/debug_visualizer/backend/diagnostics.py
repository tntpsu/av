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


def analyze_trajectory_vs_steering(recording_path: Path, analyze_to_failure: bool = False) -> Dict:
    """
    Analyze trajectory vs steering to identify which component is failing.
    
    Args:
        recording_path: Path to HDF5 recording file
        analyze_to_failure: If True, only analyze up to the point where car went out of lane and stayed out
        
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
            if has_perception:
                perception_left = np.array(f["perception/left_lane_line_x"][:])
                perception_right = np.array(f["perception/right_lane_line_x"][:])
                perception_center = (perception_left + perception_right) / 2.0

            speed = None
            if "vehicle/speed" in f:
                speed = np.array(f["vehicle/speed"][:])
            
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
                        if gt_center is not None:
                            gt_center = gt_center[:min_len]
                        if perception_center is not None:
                            perception_center = perception_center[:min_len]
            
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
            
            # Build time axis (seconds) if timestamps available
            if control_timestamps is not None and len(control_timestamps) > 0:
                time_axis = control_timestamps - control_timestamps[0]
            else:
                time_axis = np.arange(min_len, dtype=np.float32) * 0.033

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
                    "accel_hotspots": accel_hotspots,
                    "jerk_hotspots": jerk_hotspots
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

