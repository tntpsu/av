"""
Issue Detection Module

Automatically detects problematic frames in recordings:
- Extreme polynomial coefficients
- High lateral error
- Perception failures
- Emergency stops
- Heading jumps
"""

import numpy as np
import h5py
import json
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


def _append_sign_mismatch_issue(
    issues: List[Dict],
    start_frame: int,
    end_frame: int,
    total_error: np.ndarray,
    steering: np.ndarray,
    before: Optional[np.ndarray],
    feedback: Optional[np.ndarray],
    stale_ctrl: Optional[np.ndarray],
    stale_perc: Optional[np.ndarray],
    num_lanes: Optional[np.ndarray],
    override: Optional[np.ndarray],
) -> None:
    """Append a straight sign mismatch issue with a classified root cause."""
    err_slice = total_error[start_frame:end_frame + 1]
    steer_slice = steering[start_frame:end_frame + 1]
    before_slice = before[start_frame:end_frame + 1] if before is not None else None
    fb_slice = feedback[start_frame:end_frame + 1] if feedback is not None else None
    stale_rate = 0.0
    if stale_ctrl is not None:
        stale_rate = max(stale_rate, safe_float(float((stale_ctrl[start_frame:end_frame + 1] > 0).mean()) * 100.0))
    if stale_perc is not None:
        stale_rate = max(stale_rate, safe_float(float((stale_perc[start_frame:end_frame + 1] > 0).mean()) * 100.0))
    lanes_min = None
    if num_lanes is not None and num_lanes.size > 0:
        lanes_min = int(num_lanes[start_frame:end_frame + 1].min())
    override_rate = 0.0
    if override is not None:
        override_rate = safe_float(float((override[start_frame:end_frame + 1] > 0).mean()) * 100.0)

    root_cause = "unknown"
    if stale_rate > 0.0 or (lanes_min is not None and lanes_min < 2):
        root_cause = "perception_stale_or_missing"
    elif (
        fb_slice is not None and before_slice is not None
        and np.sign(np.mean(err_slice)) == np.sign(np.mean(fb_slice))
        and np.sign(np.mean(before_slice)) != np.sign(np.mean(err_slice))
    ):
        root_cause = "rate_or_jerk_limit"
    elif (
        before_slice is not None
        and np.sign(np.mean(before_slice)) == np.sign(np.mean(err_slice))
        and np.sign(np.mean(steer_slice)) != np.sign(np.mean(err_slice))
    ):
        root_cause = "smoothing"
    elif (
        fb_slice is not None
        and np.sign(np.mean(fb_slice)) != np.sign(np.mean(err_slice))
    ):
        root_cause = "error_computation_or_sign"

    duration = end_frame - start_frame + 1
    severity = "high" if duration >= 10 else "medium"
    issues.append({
        "frame": int(start_frame),
        "type": "straight_sign_mismatch",
        "severity": severity,
        "description": (
            f"Straight sign mismatch for {duration} frames "
            f"(root_cause={root_cause}, stale={stale_rate:.1f}%, "
            f"lanes_min={lanes_min}, override={override_rate:.1f}%)."
        ),
        "duration": int(duration),
        "root_cause": root_cause,
        "stale_rate_percent": safe_float(stale_rate),
        "lanes_min": lanes_min,
        "override_rate_percent": safe_float(override_rate),
        "end_frame": int(end_frame),
    })


def detect_issues(recording_path: Path, analyze_to_failure: bool = False) -> Dict:
    """
    Detect all issues in a recording.
    
    Args:
        recording_path: Path to HDF5 recording file
        analyze_to_failure: If True, only analyze up to the point where car went out of lane and stayed out
        
    Returns:
        Dictionary with list of issues and summary statistics
    """
    issues = []
    
    try:
        with h5py.File(recording_path, 'r') as f:
            # Check data availability
            has_perception = "perception/left_lane_line_x" in f
            has_control = "control/lateral_error" in f
            has_vehicle = "vehicle_state/heading" in f or "vehicle/heading" in f
            has_ground_truth = "ground_truth/lane_center_x" in f
            
            # Get frame count
            if has_perception:
                num_frames = len(f["perception/left_lane_line_x"])
            elif has_control:
                num_frames = len(f["control/lateral_error"])
            else:
                return {"error": "No data found in recording"}
            
            # Detect failure point if requested
            failure_frame = None
            if analyze_to_failure:
                # Use same logic as summary_analyzer
                if has_ground_truth:
                    gt_center = np.array(f["ground_truth/lane_center_x"][:])
                    gt_lateral_error = -gt_center
                    error_data = gt_lateral_error
                    error_source = "ground_truth"
                elif has_control:
                    error_data = np.array(f["control/lateral_error"][:])
                    error_source = "perception"
                else:
                    error_data = None
                
                if error_data is not None:
                    out_of_lane_threshold = 0.5
                    catastrophic_threshold = 2.0
                    min_consecutive_out = 10
                    
                    error_abs = np.abs(error_data)
                    out_of_lane = error_abs > out_of_lane_threshold
                    catastrophic = error_abs > catastrophic_threshold
                    
                    # Find catastrophic failures
                    catastrophic_frames = np.where(catastrophic)[0]
                    if len(catastrophic_frames) > 0:
                        # Find last recovery before catastrophic failure
                        last_catastrophic = catastrophic_frames[-1]
                        recovery_before = np.where(error_abs[:last_catastrophic] < 0.5)[0]
                        if len(recovery_before) > 0:
                            last_recovery = recovery_before[-1]
                            # Find first frame after recovery where error exceeded threshold and stayed out
                            after_recovery = error_abs[last_recovery+1:]
                            consecutive_out = 0
                            for i, err in enumerate(after_recovery):
                                if err > out_of_lane_threshold:
                                    consecutive_out += 1
                                    if consecutive_out >= min_consecutive_out:
                                        failure_frame = last_recovery + 1 + i - min_consecutive_out + 1
                                        break
                                else:
                                    consecutive_out = 0
                    
                    if failure_frame is not None:
                        num_frames = failure_frame
            
            # 1. DETECT EXTREME POLYNOMIAL COEFFICIENTS
            if has_perception:
                left_coeffs = f["perception/left_lane_line_coeffs"][:] if "perception/left_lane_line_coeffs" in f else None
                right_coeffs = f["perception/right_lane_line_coeffs"][:] if "perception/right_lane_line_coeffs" in f else None
                
                # Get image dimensions (assume standard if not available)
                image_width = 640  # Default
                image_height = 480  # Default
                if "camera/image_width" in f:
                    image_width = int(f["camera/image_width"][0])
                if "camera/image_height" in f:
                    image_height = int(f["camera/image_height"][0])
                
                min_reasonable_x = -image_width
                max_reasonable_x = image_width * 2.0
                
                for frame_idx in range(min(num_frames, len(left_coeffs) if left_coeffs is not None else 0)):
                    if left_coeffs is not None and len(left_coeffs) > frame_idx:
                        coeffs = left_coeffs[frame_idx]
                        if len(coeffs) >= 3:  # Polynomial coefficients
                            # Evaluate at top, middle, bottom
                            y_positions = [0, image_height // 2, image_height - 1]
                            for y in y_positions:
                                x_eval = coeffs[0] * y**2 + coeffs[1] * y + coeffs[2]
                                if x_eval < min_reasonable_x or x_eval > max_reasonable_x:
                                    issues.append({
                                        "frame": int(frame_idx),
                                        "type": "extreme_coefficients",
                                        "severity": "high",
                                        "lane": "left",
                                        "description": f"Extreme polynomial coefficients: x={x_eval:.1f}px at y={y} (expected -{image_width} to {image_width*2})",
                                        "x_value": float(x_eval),
                                        "y_position": int(y)
                                    })
                                    break  # Only report once per frame
                    
                    if right_coeffs is not None and len(right_coeffs) > frame_idx:
                        coeffs = right_coeffs[frame_idx]
                        if len(coeffs) >= 3:  # Polynomial coefficients
                            # Evaluate at top, middle, bottom
                            y_positions = [0, image_height // 2, image_height - 1]
                            for y in y_positions:
                                x_eval = coeffs[0] * y**2 + coeffs[1] * y + coeffs[2]
                                if x_eval < min_reasonable_x or x_eval > max_reasonable_x:
                                    issues.append({
                                        "frame": int(frame_idx),
                                        "type": "extreme_coefficients",
                                        "severity": "high",
                                        "lane": "right",
                                        "description": f"Extreme polynomial coefficients: x={x_eval:.1f}px at y={y} (expected -{image_width} to {image_width*2})",
                                        "x_value": float(x_eval),
                                        "y_position": int(y)
                                    })
                                    break  # Only report once per frame
            
            # 2. DETECT PERCEPTION INSTABILITY (from stale_reason)
            # This catches cases where perception changed (lane polygon moved, width changed)
            # but didn't trigger extreme_coefficients or invalid_width
            # This is the issue type for frame 77 (lane polygon moved, causing control error)
            if has_perception and "perception/stale_reason" in f:
                stale_reasons = f["perception/stale_reason"][:num_frames]
                if len(stale_reasons) > 0:
                    # Decode bytes if needed
                    stale_reasons_list = [s.decode('utf-8') if isinstance(s, bytes) else s for s in stale_reasons]
                    
                    for frame_idx, reason in enumerate(stale_reasons_list):
                        if reason and 'instability' in str(reason).lower():
                            issues.append({
                                "frame": int(frame_idx),
                                "type": "perception_instability",
                                "severity": "high",
                                "description": f"Perception instability detected: {reason}. Lane position/width changed significantly, causing control errors.",
                                "stale_reason": str(reason)
                            })

            # 2.6 DETECT RIGHT-LANE LOW VISIBILITY (lane exits FOV)
            if has_perception and "perception/fit_points_right" in f:
                right_raw = f["perception/fit_points_right"][:num_frames]
                num_lanes = (
                    np.array(f["perception/num_lanes_detected"][:num_frames])
                    if "perception/num_lanes_detected" in f else None
                )
                image_width = int(f["camera/image_width"][0]) if "camera/image_width" in f else 640
                edge_margin = 12
                min_points = 6

                def parse_points(raw) -> list:
                    try:
                        s = raw.decode('utf-8') if isinstance(raw, (bytes, bytearray, np.bytes_)) else str(raw)
                        return json.loads(s)
                    except Exception:
                        return []

                low_flags = []
                for idx in range(len(right_raw)):
                    pts = parse_points(right_raw[idx])
                    xs = [p[0] for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2]
                    low = False
                    if num_lanes is not None and num_lanes[idx] < 2:
                        low = True
                    if len(xs) < min_points:
                        low = True
                    elif xs and max(xs) > (image_width - edge_margin):
                        low = True
                    low_flags.append(low)

                min_event_len = 3
                current = 0
                event_start = None
                for idx, flag in enumerate(low_flags):
                    if flag:
                        if current == 0:
                            event_start = idx
                        current += 1
                    else:
                        if current >= min_event_len and event_start is not None:
                            severity = "high" if current >= 10 else "medium"
                            issues.append({
                                "frame": int(event_start),
                                "type": "right_lane_low_visibility",
                                "severity": severity,
                                "description": f"Right lane low visibility for {current} frames (likely exiting FOV on right turn).",
                                "duration": int(current),
                                "end_frame": int(idx - 1),
                            })
                        current = 0
                        event_start = None
                if current >= min_event_len and event_start is not None:
                    severity = "high" if current >= 10 else "medium"
                    issues.append({
                        "frame": int(event_start),
                        "type": "right_lane_low_visibility",
                        "severity": severity,
                        "description": f"Right lane low visibility for {current} frames (likely exiting FOV on right turn).",
                        "duration": int(current),
                        "end_frame": int(len(low_flags) - 1),
                    })
            
            # 2.5 DETECT LANE-LINE JITTER (large frame-to-frame shifts)
            if has_perception and "perception/left_lane_line_x" in f and "perception/right_lane_line_x" in f:
                left = np.array(f["perception/left_lane_line_x"][:num_frames])
                right = np.array(f["perception/right_lane_line_x"][:num_frames])
                if left.size > 1 and right.size > 1:
                    jitter = np.maximum(np.abs(np.diff(left)), np.abs(np.diff(right)))
                    jitter_threshold = 0.8  # meters per frame
                    min_event_len = 3
                    current = 0
                    event_start = None
                    max_jitter = 0.0
                    for idx, value in enumerate(jitter):
                        if value > jitter_threshold:
                            if current == 0:
                                event_start = idx + 1  # jitter is between idx and idx+1
                                max_jitter = value
                            else:
                                max_jitter = max(max_jitter, value)
                            current += 1
                        else:
                            if current >= min_event_len and event_start is not None:
                                severity = "high" if max_jitter > 1.5 or current >= 10 else "medium"
                                issues.append({
                                    "frame": int(event_start),
                                    "type": "perception_lane_jitter",
                                    "severity": severity,
                                    "description": (
                                        f"Lane-line jitter detected for {current} frames "
                                        f"(max_dX={max_jitter:.2f}m)."
                                    ),
                                    "duration": int(current),
                                    "max_jitter": safe_float(max_jitter)
                                })
                            current = 0
                            event_start = None
                            max_jitter = 0.0
                    if current >= min_event_len and event_start is not None:
                        severity = "high" if max_jitter > 1.5 or current >= 10 else "medium"
                        issues.append({
                            "frame": int(event_start),
                            "type": "perception_lane_jitter",
                            "severity": severity,
                            "description": (
                                f"Lane-line jitter detected for {current} frames "
                                f"(max_dX={max_jitter:.2f}m)."
                            ),
                            "duration": int(current),
                            "max_jitter": safe_float(max_jitter)
                        })
            
            # 3. DETECT HIGH LATERAL ERROR (perception-based - indicates perception issues)
            # NOTE: This uses perception-based error, which can be wrong if perception fails
            # This is useful for detecting perception problems, not actual position errors
            if has_control:
                lateral_error = np.array(f["control/lateral_error"][:num_frames])
                lateral_error_abs = np.abs(lateral_error)
                
                high_error_threshold = 0.5  # meters
                for frame_idx in range(len(lateral_error_abs)):
                    if lateral_error_abs[frame_idx] > high_error_threshold:
                        # Only report if it's a peak (not just part of a continuous high error)
                        is_peak = True
                        if frame_idx > 0 and lateral_error_abs[frame_idx - 1] > high_error_threshold * 0.8:
                            is_peak = False
                        if frame_idx < len(lateral_error_abs) - 1 and lateral_error_abs[frame_idx + 1] > high_error_threshold * 0.8:
                            is_peak = False
                        
                        if is_peak:
                            # Check if this is actually a perception issue (high error but car might be in lane)
                            # If we have ground truth, verify if car is actually out of lane
                            is_actual_out_of_lane = False
                            if has_ground_truth and "ground_truth/left_lane_line_x" in f and "ground_truth/right_lane_line_x" in f:
                                gt_left = f["ground_truth/left_lane_line_x"][frame_idx]
                                gt_right = f["ground_truth/right_lane_line_x"][frame_idx]
                                # Car is at x=0, in lane if (left <= 0 <= right)
                                is_actual_out_of_lane = not (gt_left <= 0 <= gt_right)
                            
                            # Only report if it's a significant error (>0.8m) OR if car is actually out of lane
                            # This filters out false positives from perception errors
                            if lateral_error_abs[frame_idx] > 0.8 or is_actual_out_of_lane:
                                issues.append({
                                    "frame": int(frame_idx),
                                    "type": "high_lateral_error",
                                    "severity": "medium" if lateral_error_abs[frame_idx] < 1.0 else "high",
                                    "description": f"High lateral error: {lateral_error_abs[frame_idx]:.3f}m (perception-based{' - car actually out of lane' if is_actual_out_of_lane else ' - may be perception error'})",
                                    "error_value": float(lateral_error_abs[frame_idx])
                                })
            
            # 4. DETECT PERCEPTION FAILURES (<2 lanes detected)
            if has_perception and "perception/num_lanes_detected" in f:
                num_lanes = np.array(f["perception/num_lanes_detected"][:num_frames])
                
                # Find consecutive frames with <2 lanes
                consecutive_failures = 0
                failure_start = None
                for frame_idx in range(len(num_lanes)):
                    if num_lanes[frame_idx] < 2:
                        if failure_start is None:
                            failure_start = frame_idx
                        consecutive_failures += 1
                    else:
                        if consecutive_failures >= 5:  # Report if 5+ consecutive frames
                            issues.append({
                                "frame": int(failure_start),
                                "type": "perception_failure",
                                "severity": "high",
                                "description": f"Perception failure: <2 lanes detected for {consecutive_failures} frames (starting at frame {failure_start})",
                                "duration": int(consecutive_failures)
                            })
            
            # 4.5 DETECT STRAIGHT SIGN MISMATCH EVENTS (steering vs error)
            if has_control and "control/steering" in f:
                steering = np.array(f["control/steering"][:num_frames])
                total_error = None
                if "control/total_error_scaled" in f:
                    total_error = np.array(f["control/total_error_scaled"][:num_frames])
                elif "control/total_error" in f:
                    total_error = np.array(f["control/total_error"][:num_frames])
                is_straight = np.array(f["control/is_straight"][:num_frames]) if "control/is_straight" in f else None
                before = (
                    np.array(f["control/steering_before_limits"][:num_frames])
                    if "control/steering_before_limits" in f else None
                )
                feedback = (
                    np.array(f["control/feedback_steering"][:num_frames])
                    if "control/feedback_steering" in f else None
                )
                stale_ctrl = (
                    np.array(f["control/using_stale_perception"][:num_frames])
                    if "control/using_stale_perception" in f else None
                )
                stale_perc = (
                    np.array(f["perception/using_stale_data"][:num_frames])
                    if "perception/using_stale_data" in f else None
                )
                override = (
                    np.array(f["control/straight_sign_flip_override_active"][:num_frames])
                    if "control/straight_sign_flip_override_active" in f else None
                )
                if total_error is not None and is_straight is not None:
                    valid = (
                        (is_straight > 0)
                        & (np.abs(total_error) >= 0.02)
                        & (np.abs(steering) >= 0.02)
                    )
                    mismatch = valid & (np.sign(total_error) != np.sign(steering))
                    min_event_len = 3
                    current = 0
                    event_start = None
                    for idx, flag in enumerate(mismatch):
                        if flag:
                            if current == 0:
                                event_start = idx
                            current += 1
                        else:
                            if current >= min_event_len and event_start is not None:
                                _append_sign_mismatch_issue(
                                    issues=issues,
                                    start_frame=event_start,
                                    end_frame=idx - 1,
                                    total_error=total_error,
                                    steering=steering,
                                    before=before,
                                    feedback=feedback,
                                    stale_ctrl=stale_ctrl,
                                    stale_perc=stale_perc,
                                    num_lanes=num_lanes if has_perception else None,
                                    override=override,
                                )
                            current = 0
                            event_start = None
                    if current >= min_event_len and event_start is not None:
                        _append_sign_mismatch_issue(
                            issues=issues,
                            start_frame=event_start,
                            end_frame=len(mismatch) - 1,
                            total_error=total_error,
                            steering=steering,
                            before=before,
                            feedback=feedback,
                            stale_ctrl=stale_ctrl,
                            stale_perc=stale_perc,
                            num_lanes=num_lanes if has_perception else None,
                            override=override,
                        )
                        consecutive_failures = 0
                        failure_start = None
                
                # Check if failure extends to end
                if consecutive_failures >= 5:
                    issues.append({
                        "frame": int(failure_start),
                        "type": "perception_failure",
                        "severity": "high",
                        "description": f"Perception failure: <2 lanes detected for {consecutive_failures} frames (starting at frame {failure_start})",
                        "duration": int(consecutive_failures)
                    })
            
            # 5. DETECT NEGATIVE STEERING/REFERENCE CORRELATION WINDOWS
            if "trajectory/reference_point_x" in f and has_control:
                ref_x = np.array(f["trajectory/reference_point_x"][:num_frames])
                steering = np.array(f["control/steering"][:num_frames])
                is_straight = np.array(f["control/is_straight"][:num_frames]).astype(bool) if "control/is_straight" in f else None
                curvature = None
                if "ground_truth/path_curvature" in f:
                    curvature = np.array(f["ground_truth/path_curvature"][:num_frames])
                elif "control/path_curvature_input" in f:
                    curvature = np.array(f["control/path_curvature_input"][:num_frames])
                window_size = 60  # ~2 seconds at 30 FPS
                min_std = 1e-4
                last_issue_frame = None
                for start in range(0, len(ref_x) - window_size + 1):
                    end = start + window_size
                    window_ref = ref_x[start:end]
                    window_steer = steering[start:end]
                    if is_straight is not None:
                        straight_ratio = float(np.mean(is_straight[start:end]))
                        if straight_ratio < 0.6:
                            continue
                    if curvature is not None:
                        if float(np.mean(np.abs(curvature[start:end]))) >= 0.01:
                            continue
                    if np.std(window_ref) < min_std or np.std(window_steer) < min_std:
                        continue
                    corr = safe_float(np.corrcoef(window_ref, window_steer)[0, 1])
                    if corr < -0.3:
                        # Avoid spamming; enforce a cooldown
                        issue_frame = end - 1
                        if last_issue_frame is None or issue_frame - last_issue_frame >= window_size // 2:
                            severity = "high" if corr < -0.6 else "medium"
                            issues.append({
                                "frame": int(issue_frame),
                                "type": "negative_control_correlation",
                                "severity": severity,
                                "description": (
                                    f"Negative steering vs reference correlation over {window_size} frames "
                                    f"(corr={corr:.3f})."
                                ),
                                "correlation": float(corr),
                                "window_frames": int(window_size),
                            })
                            last_issue_frame = issue_frame
            
            # 6. DETECT SUSTAINED OUT-OF-LANE EVENTS
            # CRITICAL: Use ground truth lane boundaries if available (most accurate)
            # Only use perception-based error as fallback - perception can be wrong!
            if has_ground_truth and "ground_truth/left_lane_line_x" in f and "ground_truth/right_lane_line_x" in f:
                # Use ground truth lane boundaries (most accurate)
                gt_left = np.array(f["ground_truth/left_lane_line_x"][:num_frames])
                gt_right = np.array(f["ground_truth/right_lane_line_x"][:num_frames])
                
                # Car is at x=0 in vehicle frame
                # Out of lane if: NOT (left <= 0 <= right)
                # Calculate distance outside lane for each frame
                out_of_lane_distances = np.zeros(num_frames)
                for i in range(num_frames):
                    if gt_left[i] > 0:
                        # Car is left of left lane line
                        out_of_lane_distances[i] = abs(gt_left[i])
                    elif gt_right[i] < 0:
                        # Car is right of right lane line
                        out_of_lane_distances[i] = abs(gt_right[i])
                    else:
                        # Car is in lane
                        out_of_lane_distances[i] = 0.0
                
                lateral_error_abs = out_of_lane_distances
                error_source = "ground_truth_boundaries"
            elif has_ground_truth:
                # Fallback: Use ground truth center if boundaries not available
                gt_center = np.array(f["ground_truth/lane_center_x"][:num_frames])
                lateral_error_data = -gt_center  # Negative because center_x is position, error is offset
                lateral_error_abs = np.abs(lateral_error_data)
                error_source = "ground_truth"
            elif has_control:
                # Last resort: Use perception-based lateral error (can be wrong if perception fails!)
                lateral_error_data = np.array(f["control/lateral_error"][:num_frames])
                lateral_error_abs = np.abs(lateral_error_data)
                error_source = "perception"
            else:
                lateral_error_abs = None
                error_source = None
            
            if lateral_error_abs is not None:
                # Detect sustained out-of-lane events (>0.5m for 10+ consecutive frames)
                out_of_lane_threshold = 0.5
                catastrophic_threshold = 2.0
                min_consecutive = 10
                
                # First, identify the "final failure" (the one that led to catastrophic failure)
                # This matches the summary analyzer's logic
                final_failure_frame = None
                catastrophic_frame = None
                for i, error in enumerate(lateral_error_abs):
                    if error > catastrophic_threshold:
                        catastrophic_frame = i
                        break
                
                if catastrophic_frame is not None:
                    # Find last recovery before catastrophic failure
                    last_recovery_frame = None
                    for i in range(catastrophic_frame - 1, -1, -1):
                        if lateral_error_abs[i] < out_of_lane_threshold:
                            last_recovery_frame = i
                            break
                    
                    # Find when error first exceeded threshold after that recovery
                    if last_recovery_frame is not None:
                        consecutive_out_frames = 0
                        for i in range(last_recovery_frame + 1, len(lateral_error_abs)):
                            if lateral_error_abs[i] > out_of_lane_threshold:
                                consecutive_out_frames += 1
                                if consecutive_out_frames >= min_consecutive and final_failure_frame is None:
                                    final_failure_frame = i - (min_consecutive - 1)
                                    break
                            else:
                                consecutive_out_frames = 0
                
                # Now detect ALL sustained out-of-lane events (including recoverable ones)
                consecutive_out = 0
                event_start = None
                for frame_idx in range(len(lateral_error_abs)):
                    if lateral_error_abs[frame_idx] > out_of_lane_threshold:
                        if event_start is None:
                            event_start = frame_idx
                        consecutive_out += 1
                    else:
                        if consecutive_out >= min_consecutive:
                            # Report sustained out-of-lane event
                            max_error = np.max(lateral_error_abs[event_start:event_start + consecutive_out])
                            is_final_failure = (final_failure_frame is not None and 
                                              event_start == final_failure_frame)
                            
                            # Only report if it's the final failure OR if it's a significant event (>0.7m)
                            # This reduces noise from minor boundary crossings (<0.7m)
                            # Require substantial error to avoid false positives from minor deviations
                            if is_final_failure or max_error > 0.7:
                                issues.append({
                                    "frame": int(event_start),
                                    "type": "out_of_lane",
                                    "severity": "critical" if is_final_failure or max_error > 1.5 else "high",
                                    "description": f"{'Final failure: ' if is_final_failure else 'Sustained out-of-lane: '}{consecutive_out} frames (max error: {max_error:.3f}m, source: {error_source})",
                                    "duration": int(consecutive_out),
                                    "max_error": float(max_error),
                                    "is_final_failure": is_final_failure
                                })
                        consecutive_out = 0
                        event_start = None
                
                # Check if event extends to end (this is likely the final failure)
                if consecutive_out >= min_consecutive:
                    max_error = np.max(lateral_error_abs[event_start:event_start + consecutive_out])
                    is_final_failure = (final_failure_frame is not None and 
                                      event_start == final_failure_frame) or max_error > catastrophic_threshold
                    # Only report if it's the final failure OR if max_error is substantial (>0.7m)
                    if is_final_failure or max_error > 0.7:
                        issues.append({
                            "frame": int(event_start),
                            "type": "out_of_lane",
                            "severity": "critical",
                            "description": f"{'Final failure: ' if is_final_failure else 'Sustained out-of-lane: '}{consecutive_out} frames to end (max error: {max_error:.3f}m, source: {error_source})",
                            "duration": int(consecutive_out),
                            "max_error": float(max_error),
                            "is_final_failure": is_final_failure
                        })
            
            # 7. DETECT EMERGENCY STOPS
            if has_control and "control/emergency_stop" in f:
                emergency_stops = np.array(f["control/emergency_stop"][:num_frames])
                emergency_frames = np.where(emergency_stops)[0]
                
                for frame_idx in emergency_frames:
                    issues.append({
                        "frame": int(frame_idx),
                        "type": "emergency_stop",
                        "severity": "critical",
                        "description": "Emergency stop triggered"
                    })
            
            # 7. DETECT HEADING JUMPS (0° → 180° or similar large jumps)
            if has_vehicle:
                if "vehicle_state/heading" in f:
                    heading = np.array(f["vehicle_state/heading"][:num_frames])
                elif "vehicle/heading" in f:
                    heading = np.array(f["vehicle/heading"][:num_frames])
                else:
                    heading = None
                
                if heading is not None:
                    # Convert to degrees
                    heading_deg = np.degrees(heading)
                    
                    for frame_idx in range(1, len(heading_deg)):
                        # Check for large jumps (>90 degrees)
                        heading_diff = abs(heading_deg[frame_idx] - heading_deg[frame_idx - 1])
                        # Handle wrap-around (e.g., 179° to -179° = 2° change, not 358°)
                        if heading_diff > 180:
                            heading_diff = 360 - heading_diff
                        
                        if heading_diff > 90:  # Large jump threshold
                            issues.append({
                                "frame": int(frame_idx),
                                "type": "heading_jump",
                                "severity": "high",
                                "description": f"Heading jump: {heading_deg[frame_idx-1]:.1f}° → {heading_deg[frame_idx]:.1f}° (change: {heading_diff:.1f}°)",
                                "previous_heading": float(heading_deg[frame_idx - 1]),
                                "current_heading": float(heading_deg[frame_idx]),
                                "jump_magnitude": float(heading_diff)
                            })
            
            # Sort issues by frame number
            issues.sort(key=lambda x: x["frame"])
            
            # Count by type
            issue_counts = {}
            for issue in issues:
                issue_type = issue["type"]
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
            
            return {
                "issues": issues,
                "summary": {
                    "total_issues": len(issues),
                    "by_type": issue_counts,
                    "by_severity": {
                        "critical": len([i for i in issues if i["severity"] == "critical"]),
                        "high": len([i for i in issues if i["severity"] == "high"]),
                        "medium": len([i for i in issues if i["severity"] == "medium"]),
                        "low": len([i for i in issues if i["severity"] == "low"])
                    }
                },
                "analyze_to_failure": analyze_to_failure,
                "failure_frame": int(failure_frame) if failure_frame is not None else None
            }
            
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}

