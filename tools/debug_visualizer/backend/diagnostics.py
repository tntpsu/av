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
            
            if has_control:
                steering = np.array(f["control/steering"][:])
                if "control/lateral_error" in f:
                    lateral_error = np.array(f["control/lateral_error"][:])
                if "control/heading_error" in f:
                    heading_error = np.array(f["control/heading_error"][:])
            
            # Load ground truth for comparison
            gt_center = None
            if has_ground_truth:
                gt_center = np.array(f["ground_truth/lane_center_x"][:])
            
            # Load perception for understanding trajectory source
            perception_center = None
            if has_perception:
                perception_left = np.array(f["perception/left_lane_line_x"][:])
                perception_right = np.array(f["perception/right_lane_line_x"][:])
                perception_center = (perception_left + perception_right) / 2.0
            
            # Align data lengths
            min_len = min(
                len(ref_x),
                len(steering) if steering is not None else len(ref_x),
                len(gt_center) if gt_center is not None else len(ref_x),
                len(perception_center) if perception_center is not None else len(ref_x)
            )
            
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
            
            # ANALYSIS 3: STEERING RESPONSE
            control_quality_score = 100.0
            steering_issues = []
            steering_warnings = []
            
            steering_correlation = None
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
                
                # Check steering vs reference point correlation
                if len(ref_x) == len(steering):
                    correlation = safe_float(np.corrcoef(ref_x, steering)[0, 1] if len(ref_x) > 1 else 0.0)
                    steering_correlation = correlation
                    
                    if correlation < 0.3:
                        steering_issues.append(f"Low correlation between steering and reference point ({correlation:.3f})")
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
                    "lateral_error": lateral_error_analysis,
                    "heading_error": heading_error_analysis,
                    "issues": steering_issues,
                    "warnings": steering_warnings
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

