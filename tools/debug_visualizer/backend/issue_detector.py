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
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import math

# Add tools/ to path for scoring_registry imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scoring_registry import (  # noqa: E402
    OUT_OF_LANE_THRESHOLD_M,
    CATASTROPHIC_ERROR_M,
    MPC_SOLVE_TIME_BUDGET_MS,
)


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


def _agent_debug_log(hypothesis_id: str, location: str, message: str, data: Dict):
    # region agent log
    try:
        repo_root = Path(__file__).resolve().parents[3]
        log_path = repo_root / ".cursor" / "debug.log"
        payload = {
            "runId": "instability-v1",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass
    # endregion


def build_causal_timeline(issues: List[Dict], failure_frame: Optional[int] = None) -> List[Dict]:
    """Build a phase-ordered causal event timeline from detected issues."""
    phase_by_type = {
        "extreme_coefficients": "perception",
        "perception_instability": "perception",
        "perception_failure": "perception",
        "perception_lane_jitter": "perception",
        "right_lane_low_visibility": "perception",
        "right_lane_edge_contact": "perception",
        "negative_control_correlation": "trajectory",
        "steering_limiter_dominant": "control",
        "straight_sign_mismatch": "control",
        "trajectory_suppressed_curve_entry": "trajectory",
        "speed_exceeded_feasible": "control",
        "mpc_infeasible": "control",
        "mpc_solve_slow": "control",
        "mpc_fallback": "control",
        "high_lateral_error": "downstream",
        "out_of_lane": "downstream",
        "emergency_stop": "downstream",
        "heading_jump": "downstream",
    }
    phase_order = {"perception": 0, "trajectory": 1, "control": 2, "downstream": 3}
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}

    timeline: List[Dict] = []
    for issue in issues:
        issue_type = str(issue.get("type", "unknown"))
        phase = phase_by_type.get(issue_type, "downstream")
        frame = int(issue.get("frame", 0))
        item = {
            "phase": phase,
            "frame": frame,
            "type": issue_type,
            "severity": str(issue.get("severity", "medium")),
            "description": str(issue.get("description", "")),
        }
        if "issue_id" in issue:
            item["issue_id"] = str(issue["issue_id"])
        if "end_frame" in issue:
            item["end_frame"] = int(issue["end_frame"])
        if "duration" in issue:
            item["duration"] = int(issue["duration"])
        timeline.append(item)

    timeline.sort(
        key=lambda x: (
            int(x.get("frame", 0)),
            phase_order.get(str(x.get("phase", "downstream")), 99),
            severity_order.get(str(x.get("severity", "low")), 99),
        )
    )

    # Compact contiguous same-type events into transition-style entries
    # (keep first on-transition frame and extend duration/end frame).
    compacted: List[Dict] = []
    for item in timeline:
        frame = int(item.get("frame", 0))
        end_frame = int(item.get("end_frame", frame))
        if compacted:
            prev = compacted[-1]
            prev_frame = int(prev.get("frame", 0))
            prev_end = int(prev.get("end_frame", prev_frame))
            same_bucket = (
                str(prev.get("type")) == str(item.get("type"))
                and str(prev.get("phase")) == str(item.get("phase"))
                and str(prev.get("severity")) == str(item.get("severity"))
            )
            if same_bucket and frame <= (prev_end + 1):
                new_end = max(prev_end, end_frame)
                prev["end_frame"] = new_end
                prev["duration"] = int(new_end - prev_frame + 1)
                prev["transition_first_frame"] = int(prev_frame)
                continue

        if end_frame > frame:
            item = dict(item)
            item["duration"] = int(end_frame - frame + 1)
            item["transition_first_frame"] = int(frame)
        compacted.append(item)

    timeline = compacted

    if failure_frame is not None:
        timeline.append(
            {
                "phase": "downstream",
                "frame": int(failure_frame),
                "type": "failure_frame",
                "severity": "critical",
                "description": "Final failure frame reached (analyze-to-failure cutoff).",
            }
        )
        timeline.sort(
            key=lambda x: (
                int(x.get("frame", 0)),
                phase_order.get(str(x.get("phase", "downstream")), 99),
                severity_order.get(str(x.get("severity", "low")), 99),
            )
        )

    return timeline


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
        # region agent log
        _agent_debug_log(
            "H0",
            "issue_detector.py:detect_issues:entry",
            "Detect issues entry",
            {
                "recording": str(recording_path),
                "analyze_to_failure": bool(analyze_to_failure),
            },
        )
        # endregion
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
                    out_of_lane_threshold = OUT_OF_LANE_THRESHOLD_M
                    catastrophic_threshold = CATASTROPHIC_ERROR_M
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
                    left_lane_series = (
                        np.array(f["perception/left_lane_line_x"][:num_frames], dtype=float)
                        if "perception/left_lane_line_x" in f
                        else None
                    )
                    right_lane_series = (
                        np.array(f["perception/right_lane_line_x"][:num_frames], dtype=float)
                        if "perception/right_lane_line_x" in f
                        else None
                    )
                    num_lanes_series = (
                        np.array(f["perception/num_lanes_detected"][:num_frames], dtype=float)
                        if "perception/num_lanes_detected" in f
                        else None
                    )
                    perception_ts = (
                        np.array(f["perception/timestamps"][:num_frames], dtype=float)
                        if "perception/timestamps" in f
                        else None
                    )
                    fit_points_right = (
                        f["perception/fit_points_right"][:num_frames]
                        if "perception/fit_points_right" in f
                        else None
                    )
                    fit_points_left = (
                        f["perception/fit_points_left"][:num_frames]
                        if "perception/fit_points_left" in f
                        else None
                    )
                    detection_method_series = (
                        f["perception/detection_method"][:num_frames]
                        if "perception/detection_method" in f
                        else None
                    )
                    actual_left_series = (
                        np.array(f["perception/actual_detected_left_lane_x"][:num_frames], dtype=float)
                        if "perception/actual_detected_left_lane_x" in f
                        else None
                    )
                    actual_right_series = (
                        np.array(f["perception/actual_detected_right_lane_x"][:num_frames], dtype=float)
                        if "perception/actual_detected_right_lane_x" in f
                        else None
                    )
                    instability_width_series = (
                        np.array(f["perception/instability_width_change"][:num_frames], dtype=float)
                        if "perception/instability_width_change" in f
                        else None
                    )
                    instability_center_series = (
                        np.array(f["perception/instability_center_shift"][:num_frames], dtype=float)
                        if "perception/instability_center_shift" in f
                        else None
                    )
                    heading_delta_deg = (
                        np.array(f["vehicle/heading_delta_deg"][:num_frames], dtype=float)
                        if "vehicle/heading_delta_deg" in f
                        else None
                    )
                    road_lateral_offset = (
                        np.array(f["vehicle/road_frame_lateral_offset"][:num_frames], dtype=float)
                        if "vehicle/road_frame_lateral_offset" in f
                        else None
                    )
                    steering_cmd = (
                        np.array(f["control/steering"][:num_frames], dtype=float)
                        if "control/steering" in f
                        else None
                    )
                    control_curvature = (
                        np.array(f["control/path_curvature_input"][:num_frames], dtype=float)
                        if "control/path_curvature_input" in f
                        else None
                    )

                    # Group contiguous instability frames into one event so we don't
                    # flood the issues list with one entry per frame.
                    min_event_len = 1
                    current = 0
                    event_start = None
                    reason_counts: Dict[str, int] = {}
                    instability_windows: List[Dict[str, object]] = []
                    for frame_idx, reason in enumerate(stale_reasons_list):
                        is_instability = bool(reason and 'instability' in str(reason).lower())
                        if is_instability:
                            if current == 0:
                                event_start = frame_idx
                                reason_counts = {}
                            current += 1
                            reason_key = str(reason).strip()
                            reason_counts[reason_key] = reason_counts.get(reason_key, 0) + 1
                        else:
                            if current >= min_event_len and event_start is not None:
                                dominant_reason = max(reason_counts, key=reason_counts.get) if reason_counts else "instability"
                                severity = "high" if current >= 10 else "medium"
                                end_idx = int(frame_idx - 1)
                                window = {"start": int(event_start), "end": end_idx, "duration": int(current)}
                                instability_windows.append(window)
                                issues.append({
                                    "frame": int(event_start),
                                    "type": "perception_instability",
                                    "severity": severity,
                                    "description": (
                                        f"Perception instability period: {current} frames "
                                        f"(dominant reason: {dominant_reason})."
                                    ),
                                    "duration": int(current),
                                    "end_frame": end_idx,
                                    "dominant_stale_reason": dominant_reason,
                                    "stale_reasons": reason_counts,
                                })
                            current = 0
                            event_start = None
                            reason_counts = {}
                    if current >= min_event_len and event_start is not None:
                        dominant_reason = max(reason_counts, key=reason_counts.get) if reason_counts else "instability"
                        severity = "high" if current >= 10 else "medium"
                        end_idx = int(len(stale_reasons_list) - 1)
                        window = {"start": int(event_start), "end": end_idx, "duration": int(current)}
                        instability_windows.append(window)
                        issues.append({
                            "frame": int(event_start),
                            "type": "perception_instability",
                            "severity": severity,
                            "description": (
                                f"Perception instability period: {current} frames "
                                f"(dominant reason: {dominant_reason})."
                            ),
                            "duration": int(current),
                            "end_frame": end_idx,
                            "dominant_stale_reason": dominant_reason,
                            "stale_reasons": reason_counts,
                        })

                    def _parse_points(raw) -> list:
                        try:
                            s = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray, np.bytes_)) else str(raw)
                            return json.loads(s)
                        except Exception:
                            return []

                    instability_diagnostics = []
                    for w in instability_windows[:4]:
                        s = int(w["start"])
                        e = int(w["end"])
                        diag = {"start": s, "end": e, "duration": int(w["duration"])}

                        if left_lane_series is not None and right_lane_series is not None:
                            sl = left_lane_series[s:e + 1]
                            sr = right_lane_series[s:e + 1]
                            sw = sr - sl
                            if sl.size > 0:
                                diag["left_x_start"] = safe_float(sl[0], None)
                                diag["left_x_end"] = safe_float(sl[-1], None)
                                diag["left_dx_peak"] = safe_float(np.max(np.abs(np.diff(sl))) if sl.size > 1 else 0.0)
                                diag["right_x_start"] = safe_float(sr[0], None)
                                diag["right_x_end"] = safe_float(sr[-1], None)
                                diag["right_dx_peak"] = safe_float(np.max(np.abs(np.diff(sr))) if sr.size > 1 else 0.0)
                                diag["lane_width_start"] = safe_float(sw[0], None)
                                diag["lane_width_end"] = safe_float(sw[-1], None)
                                diag["lane_width_shift_abs"] = safe_float(abs(sw[-1] - sw[0]), None)
                                diag["lane_width_jitter_p95"] = safe_float(
                                    np.percentile(np.abs(np.diff(sw)), 95) if sw.size > 1 else 0.0
                                )

                        if num_lanes_series is not None:
                            sn = num_lanes_series[s:e + 1]
                            if sn.size > 0:
                                diag["num_lanes_min"] = safe_float(np.min(sn), None)
                                diag["num_lanes_max"] = safe_float(np.max(sn), None)
                                diag["num_lanes_lt2_frames"] = int(np.sum(sn < 2))

                        if perception_ts is not None:
                            st = perception_ts[s:e + 1]
                            if st.size > 1:
                                dts = np.diff(st)
                                diag["timestamp_repeat_frames"] = int(np.sum(np.isclose(dts, 0.0, atol=1e-9)))
                                diag["timestamp_dt_p95_ms"] = safe_float(np.percentile(dts * 1000.0, 95), None)

                        if fit_points_right is not None:
                            edge_margin = 12
                            image_width = int(f["camera/image_width"][0]) if "camera/image_width" in f else 640
                            edge_hits = 0
                            right_max_x = []
                            left_min_x = []
                            for i in range(s, e + 1):
                                pts = _parse_points(fit_points_right[i])
                                xs = [p[0] for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2]
                                if xs and max(xs) > (image_width - edge_margin):
                                    edge_hits += 1
                                if xs:
                                    right_max_x.append(float(max(xs)))
                                if fit_points_left is not None:
                                    left_pts = _parse_points(fit_points_left[i])
                                    left_xs = [p[0] for p in left_pts if isinstance(p, (list, tuple)) and len(p) >= 2]
                                    if left_xs:
                                        left_min_x.append(float(min(left_xs)))
                            diag["right_edge_contact_frames"] = int(edge_hits)
                            diag["right_edge_contact_rate_pct"] = safe_float(
                                (edge_hits / max(1, (e - s + 1))) * 100.0, None
                            )
                            if right_max_x:
                                diag["right_fit_max_x_shift_px"] = safe_float(
                                    max(right_max_x) - min(right_max_x), None
                                )
                            if left_min_x:
                                diag["left_fit_min_x_shift_px"] = safe_float(
                                    max(left_min_x) - min(left_min_x), None
                                )

                        if detection_method_series is not None:
                            methods = []
                            for i in range(s, e + 1):
                                m = detection_method_series[i]
                                methods.append(
                                    m.decode("utf-8") if isinstance(m, (bytes, bytearray, np.bytes_)) else str(m)
                                )
                            counts: Dict[str, int] = {}
                            for m in methods:
                                key = m.strip().lower()
                                counts[key] = counts.get(key, 0) + 1
                            diag["detection_method_counts"] = counts

                        if actual_left_series is not None and actual_right_series is not None:
                            al = actual_left_series[s:e + 1]
                            ar = actual_right_series[s:e + 1]
                            if al.size > 1 and ar.size > 1:
                                diag["actual_left_dx_peak"] = safe_float(np.max(np.abs(np.diff(al))), None)
                                diag["actual_right_dx_peak"] = safe_float(np.max(np.abs(np.diff(ar))), None)
                                diag["actual_lane_width_shift_abs"] = safe_float(
                                    abs((ar[-1] - al[-1]) - (ar[0] - al[0])), None
                                )

                        if instability_width_series is not None:
                            iw = instability_width_series[s:e + 1]
                            if iw.size > 0:
                                diag["instability_width_change_max"] = safe_float(np.max(iw), None)
                                diag["instability_width_change_start"] = safe_float(iw[0], None)

                        if instability_center_series is not None:
                            ic = instability_center_series[s:e + 1]
                            if ic.size > 0:
                                diag["instability_center_shift_max"] = safe_float(np.max(ic), None)
                                diag["instability_center_shift_start"] = safe_float(ic[0], None)

                        if heading_delta_deg is not None:
                            hd = heading_delta_deg[s:e + 1]
                            if hd.size > 0:
                                diag["heading_delta_abs_p95_deg"] = safe_float(
                                    np.percentile(np.abs(hd), 95), None
                                )
                                diag["heading_delta_peak_deg"] = safe_float(np.max(np.abs(hd)), None)

                        if road_lateral_offset is not None:
                            ro = road_lateral_offset[s:e + 1]
                            if ro.size > 0:
                                diag["road_lateral_offset_shift_abs"] = safe_float(abs(ro[-1] - ro[0]), None)

                        if steering_cmd is not None:
                            st = steering_cmd[s:e + 1]
                            if st.size > 0:
                                diag["steering_abs_p95"] = safe_float(np.percentile(np.abs(st), 95), None)

                        if control_curvature is not None:
                            cc = control_curvature[s:e + 1]
                            if cc.size > 0:
                                diag["path_curvature_abs_p95"] = safe_float(np.percentile(np.abs(cc), 95), None)

                        instability_diagnostics.append(diag)

                    # region agent log
                    _agent_debug_log(
                        "H1",
                        "issue_detector.py:perception_instability:windows",
                        "Instability windows diagnostics",
                        {
                            "recording": str(recording_path),
                            "window_count": int(len(instability_windows)),
                            "windows": instability_diagnostics,
                        },
                    )
                    # endregion

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
                edge_flags = []
                for idx in range(len(right_raw)):
                    pts = parse_points(right_raw[idx])
                    xs = [p[0] for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2]
                    low = False
                    edge_contact = False
                    if num_lanes is not None and num_lanes[idx] < 2:
                        low = True
                    if len(xs) < min_points:
                        low = True
                    elif xs and max(xs) > (image_width - edge_margin):
                        edge_contact = True
                    low_flags.append(low)
                    edge_flags.append(edge_contact)

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

                # Track right-lane edge contact separately from true low-visibility.
                edge_current = 0
                edge_start = None
                for idx, flag in enumerate(edge_flags):
                    if flag:
                        if edge_current == 0:
                            edge_start = idx
                        edge_current += 1
                    else:
                        if edge_current >= min_event_len and edge_start is not None:
                            issues.append({
                                "frame": int(edge_start),
                                "type": "right_lane_edge_contact",
                                "severity": "low",
                                "description": f"Right lane touches image edge for {edge_current} frames (FOV boundary contact).",
                                "duration": int(edge_current),
                                "end_frame": int(idx - 1),
                            })
                        edge_current = 0
                        edge_start = None
                if edge_current >= min_event_len and edge_start is not None:
                    issues.append({
                        "frame": int(edge_start),
                        "type": "right_lane_edge_contact",
                        "severity": "low",
                        "description": f"Right lane touches image edge for {edge_current} frames (FOV boundary contact).",
                        "duration": int(edge_current),
                        "end_frame": int(len(edge_flags) - 1),
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

            # 5.5 DETECT DOMINANT STEERING LIMITER ROOT CAUSE
            if has_control:
                rate_delta = (
                    np.abs(np.array(f["control/steering_rate_limited_delta"][:num_frames]))
                    if "control/steering_rate_limited_delta" in f
                    else None
                )
                jerk_delta = (
                    np.abs(np.array(f["control/steering_jerk_limited_delta"][:num_frames]))
                    if "control/steering_jerk_limited_delta" in f
                    else None
                )
                hard_delta = (
                    np.abs(np.array(f["control/steering_hard_clip_delta"][:num_frames]))
                    if "control/steering_hard_clip_delta" in f
                    else None
                )
                smooth_delta = (
                    np.abs(np.array(f["control/steering_smoothing_delta"][:num_frames]))
                    if "control/steering_smoothing_delta" in f
                    else None
                )
                stage_arrays = []
                stage_names = []
                if rate_delta is not None:
                    stage_arrays.append(rate_delta)
                    stage_names.append("rate_limit")
                if jerk_delta is not None:
                    stage_arrays.append(jerk_delta)
                    stage_names.append("jerk_limit")
                if hard_delta is not None:
                    stage_arrays.append(hard_delta)
                    stage_names.append("hard_clip")
                if smooth_delta is not None:
                    stage_arrays.append(smooth_delta)
                    stage_names.append("smoothing")

                if stage_arrays:
                    stacked = np.vstack(stage_arrays)
                    max_vals = np.max(stacked, axis=0)
                    max_idx = np.argmax(stacked, axis=0)
                    threshold = 1e-4
                    limited = max_vals > threshold
                    limited_count = int(np.sum(limited))
                    if limited_count > 0:
                        counts = {name: 0 for name in stage_names}
                        for i in np.where(limited)[0]:
                            counts[stage_names[int(max_idx[i])]] += 1
                        dominant_stage = max(counts, key=counts.get)
                        dominant_pct = 100.0 * counts[dominant_stage] / max(1, limited_count)
                        severity = "high" if dominant_stage in ("jerk_limit", "hard_clip") else "medium"
                        if dominant_pct >= 35.0:
                            peak_idx = int(np.argmax(max_vals))
                            issues.append({
                                "frame": peak_idx,
                                "type": "steering_limiter_dominant",
                                "severity": severity,
                                "description": (
                                    f"Steering limiter root cause: {dominant_stage} dominates "
                                    f"({dominant_pct:.1f}% of limited frames)."
                                ),
                                "dominant_stage": dominant_stage,
                                "dominant_pct": float(dominant_pct),
                                "limited_frames": limited_count,
                            })
            
            # 6. DETECT SUSTAINED OUT-OF-LANE EVENTS
            # CRITICAL: Use ground truth lane boundaries if available (most accurate)
            # Only use perception-based error as fallback - perception can be wrong!
            centerline_cross_start_frame = None
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

                # Detect centerline crossing onset (crossing left lane boundary) to help
                # users ignore downstream diagnostics after entering unsupported state.
                cross_flags = gt_left > 1e-3
                min_cross_frames = 1
                run = 0
                for i, crossed in enumerate(cross_flags):
                    if crossed:
                        run += 1
                        if run >= min_cross_frames:
                            centerline_cross_start_frame = i - (min_cross_frames - 1)
                            break
                    else:
                        run = 0
                if centerline_cross_start_frame is not None:
                    issues.append({
                        "frame": int(centerline_cross_start_frame),
                        "type": "centerline_cross",
                        "severity": "high",
                        "description": (
                            f"Centerline crossed at frame {centerline_cross_start_frame} "
                            f"(left boundary exceeded for >= {min_cross_frames} frame)."
                        ),
                    })
                
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
                emergency_stops = np.array(f["control/emergency_stop"][:num_frames]).astype(bool)
                if emergency_stops.size > 0:
                    # Transition pattern: report only OFF->ON boundaries and duration.
                    rising = list(np.where((~emergency_stops[:-1]) & emergency_stops[1:])[0] + 1)
                    if emergency_stops[0]:
                        rising = [0] + rising
                    for start_idx in rising:
                        end_idx = start_idx
                        while end_idx + 1 < emergency_stops.size and emergency_stops[end_idx + 1]:
                            end_idx += 1
                        issues.append(
                            {
                                "frame": int(start_idx),
                                "type": "emergency_stop",
                                "severity": "critical",
                                "description": (
                                    f"Emergency stop transitioned ON at frame {int(start_idx)}"
                                ),
                                "end_frame": int(end_idx),
                                "duration": int(end_idx - start_idx + 1),
                            }
                        )
            
            # 7.5 DETECT TRAJECTORY SUPPRESSED CURVE ENTRY
            failure_frame_emergency = None
            if "control/emergency_stop" in f:
                em_stop = np.array(f["control/emergency_stop"][:num_frames], dtype=np.float64)
                if em_stop.size > 0:
                    fail_idx = np.where(em_stop > 0.5)[0]
                    if len(fail_idx) > 0:
                        failure_frame_emergency = int(fail_idx[0])

            if failure_frame_emergency is not None and failure_frame_emergency >= 30:
                heading_zero_gate = (
                    np.array(f["trajectory/diag_heading_zero_gate_active"][:num_frames])
                    if "trajectory/diag_heading_zero_gate_active" in f
                    else None
                )
                curvature = (
                    np.array(f["trajectory/reference_point_curvature"][:num_frames])
                    if "trajectory/reference_point_curvature" in f
                    else None
                )
                rate_limit_active = (
                    np.array(f["trajectory/diag_ref_x_rate_limit_active"][:num_frames])
                    if "trajectory/diag_ref_x_rate_limit_active" in f
                    else None
                )

                pre_fail_start = max(0, failure_frame_emergency - 30)
                pre_fail_end = failure_frame_emergency - 1

                if heading_zero_gate is not None and heading_zero_gate.size > 0 and curvature is not None and curvature.size > 0:
                    n_align = min(len(heading_zero_gate), len(curvature))
                    hzg = heading_zero_gate[:n_align]
                    curv = curvature[:n_align]
                    gate_active = (hzg > 0.5) if np.issubdtype(hzg.dtype, np.floating) else (hzg.astype(bool))
                    in_curve = np.abs(curv) > 0.003
                    overlap = gate_active & in_curve
                    pre_fail_overlap = overlap[pre_fail_start : pre_fail_end + 1]
                    if np.any(pre_fail_overlap):
                        start_idx = int(np.where(overlap)[0][0])
                        end_idx = int(np.where(overlap)[0][-1])
                        issues.append({
                            "issue_id": "trajectory_suppressed_curve_entry",
                            "type": "trajectory_suppressed_curve_entry",
                            "frame": int(start_idx),
                            "end_frame": int(end_idx),
                            "severity": "critical",
                            "description": (
                                "Heading-zero gate active during curve (curvature>0.003) "
                                "in 30 frames before failure."
                            ),
                            "deep_link_target": "diag-section-signal-chain",
                            "focus_id": "diag-focus-signal-suppression",
                        })
                    else:
                        overlap_anywhere = np.where(overlap)[0]
                        if len(overlap_anywhere) > 0:
                            start_idx = int(overlap_anywhere[0])
                            end_idx = int(overlap_anywhere[-1])
                            issues.append({
                                "issue_id": "trajectory_suppressed_curve_entry",
                                "type": "trajectory_suppressed_curve_entry",
                                "frame": int(start_idx),
                                "end_frame": int(end_idx),
                                "severity": "warning",
                                "description": (
                                    "Heading-zero gate active during curve (transient, "
                                    "not in pre-failure window)."
                                ),
                                "deep_link_target": "diag-section-signal-chain",
                                "focus_id": "diag-focus-signal-suppression",
                            })

                if rate_limit_active is not None and rate_limit_active.size > 0:
                    rate_bool = (rate_limit_active > 0.5) if rate_limit_active.dtype == np.float64 else rate_limit_active.astype(bool)
                    run = 0
                    run_start = None
                    rate_limit_emitted_critical = False
                    for i in range(rate_bool.size):
                        if rate_bool[i]:
                            if run == 0:
                                run_start = i
                            run += 1
                        else:
                            if run >= 10 and run_start is not None:
                                in_pre_fail = run_start <= pre_fail_end and (run_start + run - 1) >= pre_fail_start
                                if in_pre_fail and not rate_limit_emitted_critical:
                                    issues.append({
                                        "issue_id": "trajectory_suppressed_curve_entry",
                                        "type": "trajectory_suppressed_curve_entry",
                                        "frame": int(run_start),
                                        "end_frame": int(run_start + run - 1),
                                        "severity": "critical",
                                        "description": (
                                            f"Ref-x rate limit active for {run} consecutive frames "
                                            "during curve approach (30 frames before failure)."
                                        ),
                                        "deep_link_target": "diag-section-signal-chain",
                                        "focus_id": "diag-focus-signal-suppression",
                                    })
                                    rate_limit_emitted_critical = True
                                elif not in_pre_fail:
                                    issues.append({
                                        "issue_id": "trajectory_suppressed_curve_entry",
                                        "type": "trajectory_suppressed_curve_entry",
                                        "frame": int(run_start),
                                        "end_frame": int(run_start + run - 1),
                                        "severity": "warning",
                                        "description": (
                                            f"Ref-x rate limit active for {run} consecutive frames "
                                            "(transient, not in pre-failure window)."
                                        ),
                                        "deep_link_target": "diag-section-signal-chain",
                                        "focus_id": "diag-focus-signal-suppression",
                                    })
                            run = 0
                            run_start = None
                    if run >= 10 and run_start is not None:
                        in_pre_fail = run_start <= pre_fail_end and (run_start + run - 1) >= pre_fail_start
                        if in_pre_fail and not rate_limit_emitted_critical:
                            issues.append({
                                "issue_id": "trajectory_suppressed_curve_entry",
                                "type": "trajectory_suppressed_curve_entry",
                                "frame": int(run_start),
                                "end_frame": int(run_start + run - 1),
                                "severity": "critical",
                                "description": (
                                    f"Ref-x rate limit active for {run} consecutive frames "
                                    "during curve approach (30 frames before failure)."
                                ),
                                "deep_link_target": "diag-section-signal-chain",
                                "focus_id": "diag-focus-signal-suppression",
                            })
                        elif not in_pre_fail:
                            issues.append({
                                "issue_id": "trajectory_suppressed_curve_entry",
                                "type": "trajectory_suppressed_curve_entry",
                                "frame": int(run_start),
                                "end_frame": int(run_start + run - 1),
                                "severity": "warning",
                                "description": (
                                    f"Ref-x rate limit active for {run} consecutive frames "
                                    "(transient, not in pre-failure window)."
                                ),
                                "deep_link_target": "diag-section-signal-chain",
                                "focus_id": "diag-focus-signal-suppression",
                            })

            # 7.6 DETECT SPEED EXCEEDED FEASIBLE
            if "vehicle/speed" in f and "trajectory/reference_point_curvature" in f:
                speed_arr = np.array(f["vehicle/speed"][:num_frames], dtype=np.float64)
                curv_arr = np.array(f["trajectory/reference_point_curvature"][:num_frames], dtype=np.float64)
                if speed_arr.size > 0 and curv_arr.size > 0:
                    n = min(len(speed_arr), len(curv_arr))
                    speed_arr = speed_arr[:n]
                    curv_arr = curv_arr[:n]
                    curv_abs = np.abs(curv_arr)
                    curv_abs_safe = np.where(curv_abs < 1e-9, 1e-9, curv_abs)
                    v_max_feasible = np.sqrt(2.45 / curv_abs_safe)
                    overspeed = (speed_arr > v_max_feasible) & (curv_abs > 0.003)
                    overspeed_frames = np.where(overspeed)[0]
                    if len(overspeed_frames) > 0:
                        ratios = np.where(
                            v_max_feasible > 1e-9,
                            speed_arr / np.maximum(v_max_feasible, 1e-9),
                            0.0,
                        )
                        max_ratio = float(np.max(ratios[overspeed]))
                        start_frame = int(overspeed_frames[0])
                        end_frame = int(overspeed_frames[-1])
                        is_pre_failure = False
                        if failure_frame_emergency is not None:
                            pre_start = max(0, failure_frame_emergency - 30)
                            pre_end = failure_frame_emergency - 1
                            is_pre_failure = np.any((overspeed_frames >= pre_start) & (overspeed_frames <= pre_end))
                        severity = "critical" if is_pre_failure else "warning"
                        issues.append({
                            "issue_id": "speed_exceeded_feasible",
                            "type": "speed_exceeded_feasible",
                            "frame": int(start_frame),
                            "end_frame": int(end_frame),
                            "severity": severity,
                            "description": (
                                f"Speed exceeded feasible limit for curvature "
                                f"(max ratio {max_ratio:.2f}x, 0.25g)."
                            ),
                            "deep_link_target": "diag-section-speed-curvature",
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
            
            # 8. DETECT MPC ISSUES
            if 'control/regime' in f:
                regime = np.array(f['control/regime'][:num_frames])
                mpc_mask = regime >= 0.5

                if 'control/mpc_feasible' in f:
                    feasible = np.array(f['control/mpc_feasible'][:num_frames])
                    infeas_frames = np.where(mpc_mask & (feasible < 0.5))[0]
                    if len(infeas_frames) > 0:
                        issues.append({
                            "frame": int(infeas_frames[0]),
                            "type": "mpc_infeasible",
                            "severity": "warning",
                            "description": (
                                f"MPC QP infeasible on {len(infeas_frames)} frame(s). "
                                f"First at frame {int(infeas_frames[0])}."
                            ),
                            "frames": [int(x) for x in infeas_frames[:20]],
                            "first_frame": int(infeas_frames[0]),
                        })

                if 'control/mpc_solve_time_ms' in f:
                    solve = np.array(f['control/mpc_solve_time_ms'][:num_frames])
                    slow_frames = np.where(mpc_mask & (solve > MPC_SOLVE_TIME_BUDGET_MS))[0]
                    if len(slow_frames) > 0:
                        issues.append({
                            "frame": int(slow_frames[0]),
                            "type": "mpc_solve_slow",
                            "severity": "info",
                            "description": (
                                f"MPC solve time > {MPC_SOLVE_TIME_BUDGET_MS:.0f}ms on {len(slow_frames)} frame(s). "
                                f"Max: {float(np.max(solve[slow_frames])):.1f}ms."
                            ),
                            "frames": [int(x) for x in slow_frames[:20]],
                            "first_frame": int(slow_frames[0]),
                        })

                if 'control/mpc_fallback_active' in f:
                    fallback = np.array(f['control/mpc_fallback_active'][:num_frames])
                    fallback_frames = np.where(fallback > 0.5)[0]
                    if len(fallback_frames) > 0:
                        issues.append({
                            "frame": int(fallback_frames[0]),
                            "type": "mpc_fallback",
                            "severity": "high",
                            "description": (
                                f"MPC fallback to PP active on {len(fallback_frames)} frame(s). "
                                f"First at frame {int(fallback_frames[0])}."
                            ),
                            "frames": [int(x) for x in fallback_frames[:20]],
                            "first_frame": int(fallback_frames[0]),
                        })

            # Grade issues (guarded on field existence)
            if 'vehicle/road_grade' in f:
                road_grade = np.array(f['vehicle/road_grade'][:num_frames])
                speed = np.array(f['vehicle/speed'][:num_frames]) if 'vehicle/speed' in f else None
                speed_limit = np.array(f['vehicle/speed_limit'][:num_frames]) if 'vehicle/speed_limit' in f else None
                pitch_rad = np.array(f['vehicle/pitch_rad'][:num_frames]) if 'vehicle/pitch_rad' in f else None
                grade_comp = (
                    np.array(f['control/grade_compensation_active'][:num_frames])
                    if 'control/grade_compensation_active' in f else None
                )

                # Grade overspeed: speed > target + margin on downhill
                if speed is not None and speed_limit is not None:
                    downhill_overspeed = np.where(
                        (road_grade < -0.02) & (speed > speed_limit + 1.0)
                    )[0]
                    if len(downhill_overspeed) > 0:
                        issues.append({
                            "frame": int(downhill_overspeed[0]),
                            "type": "grade_overspeed",
                            "severity": "high",
                            "description": (
                                f"Downhill overspeed on {len(downhill_overspeed)} frame(s). "
                                f"First at frame {int(downhill_overspeed[0])}."
                            ),
                            "frames": [int(x) for x in downhill_overspeed[:20]],
                            "first_frame": int(downhill_overspeed[0]),
                        })

                # Grade compensation inactive on graded terrain
                if grade_comp is not None:
                    comp_missing = np.where(
                        (np.abs(road_grade) > 0.02) & (grade_comp < 0.5)
                    )[0]
                    if len(comp_missing) > 0:
                        issues.append({
                            "frame": int(comp_missing[0]),
                            "type": "grade_compensation_inactive",
                            "severity": "medium",
                            "description": (
                                f"Grade compensation inactive on {len(comp_missing)} graded frame(s). "
                                f"First at frame {int(comp_missing[0])}."
                            ),
                            "frames": [int(x) for x in comp_missing[:20]],
                            "first_frame": int(comp_missing[0]),
                        })

                # Pitch discontinuity (upstream issue)
                if pitch_rad is not None and len(pitch_rad) > 1:
                    pitch_diff = np.abs(np.diff(pitch_rad))
                    pitch_jumps = np.where(pitch_diff > 0.02)[0]
                    if len(pitch_jumps) > 0:
                        issues.append({
                            "frame": int(pitch_jumps[0]),
                            "type": "grade_pitch_discontinuity",
                            "severity": "info",
                            "description": (
                                f"Pitch discontinuity (>0.02 rad) on {len(pitch_jumps)} frame(s). "
                                f"Max jump: {float(np.max(pitch_diff)):.4f} rad."
                            ),
                            "frames": [int(x) for x in pitch_jumps[:20]],
                            "first_frame": int(pitch_jumps[0]),
                        })

                # Pitch signal dead (Step 3B): pitch_rad ~0 while road_grade has signal
                if pitch_rad is not None and len(road_grade) > 10:
                    pitch_var = float(np.var(pitch_rad))
                    grade_var = float(np.var(road_grade))
                    if pitch_var < 1e-6 and grade_var > 1e-4:
                        issues.append({
                            "frame": 0,
                            "type": "grade_pitch_signal_dead",
                            "severity": "info",
                            "description": (
                                f"Pitch telemetry dead (var={pitch_var:.2e}) while road_grade "
                                f"has signal (var={grade_var:.2e}). roadGrade is authoritative."
                            ),
                            "frames": [0],
                            "first_frame": 0,
                        })

                # Grade throttle saturated (Step 3C)
                if 'control/throttle' in f:
                    ctrl_throttle = np.array(f['control/throttle'][:num_frames])
                    sat_frames = np.where(
                        (ctrl_throttle >= 0.95) & (np.abs(road_grade) > 0.02)
                    )[0]
                    for fr in sat_frames[:20]:  # cap to 20 issues
                        issues.append({
                            "frame": int(fr),
                            "type": "grade_throttle_saturated",
                            "severity": "medium",
                            "message": f"Throttle saturated ({ctrl_throttle[fr]:.2f}) on {abs(road_grade[fr])*100:.1f}% grade",
                        })

                # Grade accel budget starved (Step 3C)
                if 'control/effective_max_accel' in f:
                    eff_accel = np.array(f['control/effective_max_accel'][:num_frames])
                    gravity_load = 9.81 * np.abs(np.sin(road_grade))
                    base_max_accel = 1.2  # default from config
                    budget_ratio = (eff_accel - gravity_load) / max(base_max_accel, 0.1)
                    starved_frames = np.where(
                        (budget_ratio < 0.5) & (np.abs(road_grade) > 0.02)
                    )[0]
                    for fr in starved_frames[:20]:
                        issues.append({
                            "frame": int(fr),
                            "type": "grade_accel_budget_starved",
                            "severity": "medium",
                            "message": f"Accel budget ratio {budget_ratio[fr]:.2f} on {abs(road_grade[fr])*100:.1f}% grade",
                        })

                # Grade planner false cap (Step 3C)
                if 'control/speed_governor_active_limiter_code' in f and 'control/curvature_preview_abs' in f:
                    lim_code = np.array(f['control/speed_governor_active_limiter_code'][:num_frames])
                    curv_prev = np.array(f['control/curvature_preview_abs'][:num_frames])
                    false_cap_frames = np.where(
                        (lim_code == 5) & (curv_prev < 0.003) & (speed < speed_limit - 1.0)
                    )[0]
                    for fr in false_cap_frames[:20]:
                        issues.append({
                            "frame": int(fr),
                            "type": "grade_planner_false_cap",
                            "severity": "medium",
                            "message": f"Speed planner cap active (code=5) on straight (κ={curv_prev[fr]:.4f})",
                        })

            # 8. WHEEL SLIP DIVERGENCE (Step 3D)
            if "vehicle/wheel_sideways_slip" in f:
                ws_slip = np.array(f["vehicle/wheel_sideways_slip"][:num_frames])
                if ws_slip.ndim == 2 and ws_slip.shape[1] == 4:
                    # Flag frames where any wheel has sideways slip > 0.1 on a straight
                    max_slip = np.max(np.abs(ws_slip), axis=1)
                    is_straight_for_slip = True  # default
                    if "control/curvature_primary_abs" in f:
                        curv_abs = np.array(f["control/curvature_primary_abs"][:num_frames])
                        curv_abs = curv_abs[:len(max_slip)]
                        is_straight_for_slip = curv_abs < 0.005
                    slip_divergence = np.where(
                        (max_slip > 0.1) & (is_straight_for_slip if isinstance(is_straight_for_slip, np.ndarray) else True)
                    )[0]
                    for fr in slip_divergence[:30]:
                        issues.append({
                            "frame": int(fr),
                            "type": "wheel_slip_divergence",
                            "severity": "high",
                            "message": (
                                f"Wheel sideways slip {max_slip[fr]:.3f} on straight "
                                f"(FL={ws_slip[fr,0]:.3f} FR={ws_slip[fr,1]:.3f} "
                                f"RL={ws_slip[fr,2]:.3f} RR={ws_slip[fr,3]:.3f})"
                            ),
                        })

            # 9. MPC STEERING DIVERGENCE (Step 3D)
            # When MPC is active (regime >= 0.5), flag frames where MPC steering
            # diverges significantly from PP steering — indicates MPC hunting.
            if "control/regime" in f and "control/steering" in f:
                regime = np.array(f["control/regime"][:num_frames])
                final_steer = np.array(f["control/steering"][:num_frames])
                pp_steer = None
                if "control/steering_post_hard_clip" in f:
                    pp_steer = np.array(f["control/steering_post_hard_clip"][:num_frames])
                if pp_steer is not None:
                    mpc_active = regime >= 0.5
                    steer_div = np.abs(final_steer - pp_steer)
                    # Flag when MPC diverges > 0.05 rad from PP on MPC-active frames
                    mpc_divergent = np.where(mpc_active & (steer_div > 0.05))[0]
                    for fr in mpc_divergent[:30]:
                        issues.append({
                            "frame": int(fr),
                            "type": "mpc_steering_divergence",
                            "severity": "medium",
                            "message": (
                                f"MPC steering diverges from PP by {steer_div[fr]:.3f} rad "
                                f"(MPC={final_steer[fr]:.3f}, PP={pp_steer[fr]:.3f})"
                            ),
                        })
                    # Also detect MPC oscillation: sign changes in MPC steering
                    if np.sum(mpc_active) > 10:
                        mpc_frames = np.where(mpc_active)[0]
                        mpc_steer_vals = final_steer[mpc_frames]
                        sign_changes = np.sum(np.diff(np.sign(mpc_steer_vals)) != 0)
                        osc_rate = sign_changes / len(mpc_frames) if len(mpc_frames) > 0 else 0
                        if osc_rate > 0.3:  # > 30% of frames have sign change = hunting
                            issues.append({
                                "frame": int(mpc_frames[0]),
                                "type": "mpc_oscillation",
                                "severity": "high",
                                "message": (
                                    f"MPC steering oscillation detected: {sign_changes} sign changes "
                                    f"in {len(mpc_frames)} MPC frames ({osc_rate:.1%} rate)"
                                ),
                            })

            # 10. GT BOUNDARY SANITY (Step 3D)
            # Unity GroundTruthReporter can produce corrupt boundary values (5000+ m).
            # Flag frames where GT boundaries are implausible.
            for gt_key in ("vehicle/gt_left_boundary", "vehicle/gt_right_boundary",
                           "vehicle/groundTruthLeftBoundary", "vehicle/groundTruthRightBoundary"):
                if gt_key in f:
                    gt_vals = np.array(f[gt_key][:num_frames])
                    corrupt = np.where(np.abs(gt_vals) > 50.0)[0]
                    for fr in corrupt[:10]:
                        issues.append({
                            "frame": int(fr),
                            "type": "gt_boundary_corrupt",
                            "severity": "critical",
                            "message": (
                                f"GT boundary '{gt_key.split('/')[-1]}' = {gt_vals[fr]:.1f}m "
                                f"(|value| > 50m — likely corrupt)"
                            ),
                        })

            # ── Section 11: CURVE LATE TURN-IN / LOOKAHEAD FLOOR RESCUE (Step 4) ──
            la_arr = np.array(f["control/pp_lookahead_distance"][:num_frames]) \
                if "control/pp_lookahead_distance" in f else None
            kappa_arr = np.array(f["control/curvature_primary_abs"][:num_frames]) \
                if "control/curvature_primary_abs" in f else None
            if la_arr is not None and kappa_arr is not None:
                min_len = min(len(la_arr), len(kappa_arr), num_frames)
                la_arr = la_arr[:min_len]
                kappa_arr = kappa_arr[:min_len]
                in_curve = kappa_arr > 0.005
                if len(la_arr) > 1:
                    la_diff = np.diff(la_arr)
                    # Floor rescue: lookahead drops >2m in one frame while in a curve
                    rescue_frames = np.where((la_diff < -2.0) & in_curve[1:])[0]
                    if len(rescue_frames) >= 2:
                        for fr in rescue_frames[:20]:
                            issues.append({
                                "frame": int(fr),
                                "type": "curve_late_turn_in",
                                "severity": "medium",
                                "message": (
                                    f"Lookahead floor rescue at curve onset: "
                                    f"Δlookahead={la_diff[fr]:.1f}m, "
                                    f"κ={kappa_arr[fr+1]:.4f}. "
                                    f"COMMIT phase firing late — raise "
                                    f"curve_local_commit_distance_ready_m or "
                                    f"enable curve_auto_derive."
                                ),
                            })

            # Sort issues by frame number
            issues.sort(key=lambda x: x["frame"])
            for idx, issue in enumerate(issues):
                if "issue_id" not in issue:
                    issue["issue_id"] = f"{issue.get('type', 'issue')}:{int(issue.get('frame', 0))}:{idx}"
            first_out_of_lane = next(
                (int(i["frame"]) for i in issues if i.get("type") == "out_of_lane"),
                None,
            )
            
            # Count by type
            issue_counts = {}
            for issue in issues:
                issue_type = issue["type"]
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
            
            return {
                "issues": issues,
                "causal_timeline": build_causal_timeline(issues, failure_frame=failure_frame),
                "summary": {
                    "total_issues": len(issues),
                    "road_departure_start_frame": first_out_of_lane,
                    "centerline_cross_start_frame": (
                        int(centerline_cross_start_frame)
                        if centerline_cross_start_frame is not None
                        else None
                    ),
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

