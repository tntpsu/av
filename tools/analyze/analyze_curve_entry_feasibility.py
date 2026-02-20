#!/usr/bin/env python3
"""
Curve-entry feasibility analysis for AV recordings.

This complements phase-to-failure metrics by answering:
- Was speed too high for curvature (v^2 * kappa > lateral accel budget)?
- Did steering authority arrive too slowly (pre-limit vs post-limit gap)?
- Was perception continuity degraded during entry (stale perception)?
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np


def _as_float_array(f: h5py.File, key: str) -> Optional[np.ndarray]:
    if key not in f:
        return None
    return np.asarray(f[key][:], dtype=float)


def _as_bool_array(f: h5py.File, key: str) -> Optional[np.ndarray]:
    if key not in f:
        return None
    return np.asarray(f[key][:]).astype(bool)


def _first_true_idx(mask: Optional[np.ndarray]) -> Optional[int]:
    if mask is None:
        return None
    idx = np.where(mask)[0]
    return int(idx[0]) if len(idx) else None


def _find_centerline_cross(
    signal: np.ndarray,
    initial_sign: int,
    min_abs: float,
    persist_frames: int,
    search_start: int = 0,
) -> Optional[int]:
    if len(signal) == 0 or initial_sign == 0:
        return None
    threshold = max(1e-3, float(min_abs))
    for i in range(max(0, search_start), len(signal) - persist_frames + 1):
        w = signal[i : i + persist_frames]
        if np.any(np.abs(w) < threshold):
            continue
        if np.all(np.sign(w) == -initial_sign):
            return i
    return None


def _find_outside_lane_touch(
    left_line_x: np.ndarray,
    right_line_x: np.ndarray,
    margin: float,
    persist_frames: int,
    search_start: int = 0,
) -> Tuple[Optional[int], str]:
    """Find first outside-lane touch (left/right/both) in vehicle frame."""
    if len(left_line_x) == 0 or len(right_line_x) == 0:
        return None, "none"
    threshold = max(0.0, float(margin))
    persist = max(1, int(persist_frames))
    n = min(len(left_line_x), len(right_line_x))
    left = left_line_x[:n]
    right = right_line_x[:n]
    # Ignore obviously missing samples where both lanes are zero.
    valid = (np.abs(left) > 1e-6) | (np.abs(right) > 1e-6)
    left_touch = left > threshold
    right_touch = right < -threshold
    for i in range(max(0, search_start), n - persist + 1):
        if not np.all(valid[i : i + persist]):
            continue
        w_left = left_touch[i : i + persist]
        w_right = right_touch[i : i + persist]
        if np.all(w_left) and np.all(w_right):
            return i, "outside_lane_both"
        if np.all(w_left):
            return i, "outside_lane_left"
        if np.all(w_right):
            return i, "outside_lane_right"
    return None, "none"


def _safe_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.mean(x))


def _safe_max(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.max(x))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze full curve-entry feasibility from a .h5 recording.")
    parser.add_argument("recording", type=Path, help="Path to .h5 recording")
    parser.add_argument("--curve-start-frame", type=int, default=174, help="Curve entry frame")
    parser.add_argument("--entry-length-frames", type=int, default=25, help="Entry window length")
    parser.add_argument(
        "--centerline-signal",
        choices=["lane_center_error", "lateral_offset"],
        default="lane_center_error",
        help="Signal for centerline-cross detection",
    )
    parser.add_argument(
        "--cross-min-abs",
        type=float,
        default=0.0,
        help="Centerline intrusion min abs (0.0 => any intrusion)",
    )
    parser.add_argument("--cross-persist-frames", type=int, default=1, help="Intrusion persistence")
    parser.add_argument(
        "--first-touch-frame",
        type=int,
        default=None,
        help="Optional manual first-touch frame override (bypasses auto touch detection).",
    )
    parser.add_argument(
        "--failure-event",
        choices=["either_touch", "centerline", "outside_lane"],
        default="either_touch",
        help="Event used to define first touch when --first-touch-frame is not provided.",
    )
    parser.add_argument(
        "--outside-lane-margin",
        type=float,
        default=0.05,
        help="Outside-lane touch margin in meters (vehicle frame).",
    )
    parser.add_argument(
        "--outside-persist-frames",
        type=int,
        default=1,
        help="Persistence frames for outside-lane touch.",
    )
    parser.add_argument(
        "--max-lateral-accel",
        type=float,
        default=1.3,
        help="Feasibility budget in m/s^2 (v^2*kappa check)",
    )
    parser.add_argument(
        "--authority-gap-threshold",
        type=float,
        default=0.25,
        help="Mean |pre-final| steering gap threshold for authority-limited classification",
    )
    parser.add_argument(
        "--stale-threshold-pct",
        type=float,
        default=50.0,
        help="Stale-perception percentage threshold for perception-limited classification",
    )
    parser.add_argument(
        "--telemetry-speed-limited-threshold-pct",
        type=float,
        default=30.0,
        help="Threshold for classifying speed-limited from turn_feasibility_infeasible telemetry",
    )
    args = parser.parse_args()

    if not args.recording.exists():
        raise FileNotFoundError(f"Recording not found: {args.recording}")

    with h5py.File(args.recording, "r") as f:
        ts = _as_float_array(f, "vehicle/timestamps")
        speed = _as_float_array(f, "vehicle/speed")
        kappa = _as_float_array(f, "control/path_curvature_input")
        lce = _as_float_array(f, "vehicle/road_frame_lane_center_error")
        lo = _as_float_array(f, "vehicle/road_frame_lateral_offset")
        emergency_stop = _as_bool_array(f, "control/emergency_stop")
        stale = _as_bool_array(f, "control/using_stale_perception")
        if stale is None:
            stale = _as_bool_array(f, "perception/using_stale_data")
        gt_left = _as_float_array(f, "ground_truth/left_lane_line_x")
        gt_right = _as_float_array(f, "ground_truth/right_lane_line_x")
        perception_left = _as_float_array(f, "perception/left_lane_line_x")
        perception_right = _as_float_array(f, "perception/right_lane_line_x")

        s_pre = _as_float_array(f, "control/steering_pre_rate_limit")
        s_final = _as_float_array(f, "control/steering_post_smoothing")
        if s_final is None:
            s_final = _as_float_array(f, "control/steering")
        d_rate = _as_float_array(f, "control/steering_rate_limited_delta")
        d_jerk = _as_float_array(f, "control/steering_jerk_limited_delta")
        d_clip = _as_float_array(f, "control/steering_hard_clip_delta")
        d_smooth = _as_float_array(f, "control/steering_smoothing_delta")
        turn_feasibility_active = _as_bool_array(f, "control/turn_feasibility_active")
        turn_feasibility_infeasible = _as_bool_array(f, "control/turn_feasibility_infeasible")
        turn_feasibility_margin_g = _as_float_array(f, "control/turn_feasibility_margin_g")
        turn_feasibility_speed_delta_mps = _as_float_array(
            f, "control/turn_feasibility_speed_delta_mps"
        )

    needed = [ts, speed, kappa, s_pre, s_final]
    if any(x is None for x in needed):
        raise KeyError(
            "Missing required datasets. Need: vehicle/timestamps, vehicle/speed, "
            "control/path_curvature_input, control/steering_pre_rate_limit, "
            "control/steering_post_smoothing (or control/steering)."
        )

    n = min(len(ts), len(speed), len(kappa), len(s_pre), len(s_final))
    ts = ts[:n]
    speed = speed[:n]
    kappa = kappa[:n]
    s_pre = s_pre[:n]
    s_final = s_final[:n]
    if stale is not None:
        stale = stale[:n]
    if emergency_stop is not None:
        emergency_stop = emergency_stop[:n]
    if d_rate is not None:
        d_rate = d_rate[:n]
    if d_jerk is not None:
        d_jerk = d_jerk[:n]
    if d_clip is not None:
        d_clip = d_clip[:n]
    if d_smooth is not None:
        d_smooth = d_smooth[:n]
    if turn_feasibility_active is not None:
        turn_feasibility_active = turn_feasibility_active[:n]
    if turn_feasibility_infeasible is not None:
        turn_feasibility_infeasible = turn_feasibility_infeasible[:n]
    if turn_feasibility_margin_g is not None:
        turn_feasibility_margin_g = turn_feasibility_margin_g[:n]
    if turn_feasibility_speed_delta_mps is not None:
        turn_feasibility_speed_delta_mps = turn_feasibility_speed_delta_mps[:n]
    if lce is not None:
        lce = lce[:n]
    if lo is not None:
        lo = lo[:n]
    if gt_left is not None:
        gt_left = gt_left[:n]
    if gt_right is not None:
        gt_right = gt_right[:n]
    if perception_left is not None:
        perception_left = perception_left[:n]
    if perception_right is not None:
        perception_right = perception_right[:n]

    signal = lce if args.centerline_signal == "lane_center_error" else lo
    if signal is None:
        signal = lce if lce is not None else lo
    if signal is None:
        raise KeyError("Missing both road-frame centerline signals")

    init_window = min(60, n)
    init_med = float(np.median(signal[:init_window])) if init_window > 0 else 0.0
    initial_sign = int(np.sign(init_med)) if abs(init_med) >= 1e-3 else (int(np.sign(signal[0])) if n > 0 else 0)

    curve_start = int(np.clip(args.curve_start_frame, 0, max(0, n - 1)))
    cross = _find_centerline_cross(
        signal=signal,
        initial_sign=initial_sign,
        min_abs=args.cross_min_abs,
        persist_frames=max(1, args.cross_persist_frames),
        search_start=curve_start,
    )
    outside_source = "none"
    outside_frame: Optional[int] = None
    outside_reason = "none"
    if gt_left is not None and gt_right is not None:
        outside_frame, outside_reason = _find_outside_lane_touch(
            gt_left,
            gt_right,
            margin=args.outside_lane_margin,
            persist_frames=args.outside_persist_frames,
            search_start=curve_start,
        )
        outside_source = "ground_truth"
    elif perception_left is not None and perception_right is not None:
        outside_frame, outside_reason = _find_outside_lane_touch(
            perception_left,
            perception_right,
            margin=args.outside_lane_margin,
            persist_frames=args.outside_persist_frames,
            search_start=curve_start,
        )
        outside_source = "perception"

    offroad = _first_true_idx(emergency_stop)
    first_touch_reason = "none"
    if args.first_touch_frame is not None:
        first_touch = int(np.clip(args.first_touch_frame, 0, max(0, n - 1)))
        first_touch_reason = "manual_override"
    elif args.failure_event == "centerline":
        first_touch = cross if cross is not None else n
        first_touch_reason = "centerline_cross" if cross is not None else "none"
    elif args.failure_event == "outside_lane":
        first_touch = outside_frame if outside_frame is not None else n
        first_touch_reason = outside_reason if outside_frame is not None else "none"
    else:
        touch_candidates = []
        if cross is not None:
            touch_candidates.append((cross, "centerline_cross"))
        if outside_frame is not None:
            touch_candidates.append((outside_frame, outside_reason))
        if touch_candidates:
            first_touch, first_touch_reason = min(touch_candidates, key=lambda x: x[0])
        else:
            first_touch = n

    fail_candidates = [x for x in [first_touch, offroad] if x is not None and x < n]
    first_failure = min(fail_candidates) if fail_candidates else n

    entry_start = curve_start
    entry_end = min(first_failure, curve_start + max(1, args.entry_length_frames))
    if entry_end <= entry_start:
        entry_end = min(n, curve_start + max(1, args.entry_length_frames))

    sl = slice(entry_start, entry_end)
    speed_e = speed[sl]
    kappa_e = np.abs(kappa[sl])
    ay_e = (speed_e ** 2) * kappa_e
    speed_margin_e = args.max_lateral_accel - ay_e

    pre_abs = np.abs(s_pre[sl])
    final_abs = np.abs(s_final[sl])
    authority_gap = pre_abs - final_abs
    authority_gap = np.maximum(authority_gap, 0.0)
    authority_ratio = np.divide(
        final_abs,
        np.maximum(pre_abs, 1e-6),
        out=np.zeros_like(final_abs),
        where=np.maximum(pre_abs, 1e-6) > 0,
    )

    stale_pct = float(np.mean(stale[sl]) * 100.0) if stale is not None and entry_end > entry_start else 0.0
    rate_mean = _safe_mean(d_rate[sl]) if d_rate is not None else 0.0
    jerk_mean = _safe_mean(d_jerk[sl]) if d_jerk is not None else 0.0
    clip_mean = _safe_mean(d_clip[sl]) if d_clip is not None else 0.0
    smooth_mean = _safe_mean(d_smooth[sl]) if d_smooth is not None else 0.0

    speed_limited_pct = float(np.mean(ay_e > args.max_lateral_accel) * 100.0) if ay_e.size else 0.0
    telemetry_present = (
        turn_feasibility_active is not None
        and turn_feasibility_infeasible is not None
        and turn_feasibility_margin_g is not None
        and turn_feasibility_speed_delta_mps is not None
    )
    telemetry_active_mask = (
        turn_feasibility_active[sl] if telemetry_present else np.zeros(max(0, entry_end - entry_start), dtype=bool)
    )
    telemetry_active_pct = (
        float(np.mean(telemetry_active_mask) * 100.0)
        if telemetry_present and telemetry_active_mask.size
        else 0.0
    )
    telemetry_infeasible_pct = (
        float(np.mean(turn_feasibility_infeasible[sl]) * 100.0)
        if telemetry_present and turn_feasibility_infeasible[sl].size
        else 0.0
    )
    telemetry_margin_mean = (
        _safe_mean(turn_feasibility_margin_g[sl]) if telemetry_present else 0.0
    )
    telemetry_margin_min = (
        float(np.min(turn_feasibility_margin_g[sl]))
        if telemetry_present and turn_feasibility_margin_g[sl].size
        else 0.0
    )
    telemetry_speed_delta_mean = (
        _safe_mean(turn_feasibility_speed_delta_mps[sl]) if telemetry_present else 0.0
    )
    telemetry_speed_delta_max = (
        _safe_max(turn_feasibility_speed_delta_mps[sl]) if telemetry_present else 0.0
    )
    authority_limited = _safe_mean(authority_gap) > args.authority_gap_threshold
    perception_limited = stale_pct >= args.stale_threshold_pct

    # Priority order:
    # 1) turn_feasibility_* telemetry (source-of-truth when present)
    # 2) legacy v^2*kappa feasibility as secondary fallback
    if telemetry_present and telemetry_infeasible_pct >= args.telemetry_speed_limited_threshold_pct:
        primary = "speed-limited"
    elif not telemetry_present and speed_limited_pct >= 30.0:
        primary = "speed-limited"
    elif authority_limited:
        primary = "steering-authority-limited"
    elif perception_limited:
        primary = "perception-limited"
    else:
        primary = "mixed-or-unclear"

    print(f"recording: {args.recording}")
    print(f"frames_total: {n}")
    print(f"curve_entry_window: [{entry_start}, {entry_end}) ({max(0, entry_end-entry_start)} frames)")
    print(
        f"first_touch_frame: {first_touch if first_touch < n else 'none'}"
        f" ({first_touch_reason})"
    )
    print(
        f"touch_events: centerline={cross if cross is not None else 'none'}, "
        f"outside_lane={outside_frame if outside_frame is not None else 'none'} "
        f"(source={outside_source}, reason={outside_reason}, margin={args.outside_lane_margin:.3f}m)"
    )
    print(f"first_failure_frame: {first_failure if first_failure < n else 'none'}")
    print("")
    print("curve_entry_story:")
    print(f"- speed_feasibility: mean(v^2*kappa)={_safe_mean(ay_e):.3f} m/s^2, max={_safe_max(ay_e):.3f} m/s^2, budget={args.max_lateral_accel:.3f}")
    print(f"- speed_margin: mean={_safe_mean(speed_margin_e):.3f}, min={float(np.min(speed_margin_e)) if speed_margin_e.size else 0.0:.3f}")
    print(f"- speed_limited_frames_pct: {speed_limited_pct:.1f}%")
    if telemetry_present:
        print(
            f"- turn_feasibility_telemetry: active_pct={telemetry_active_pct:.1f}%, "
            f"infeasible_pct={telemetry_infeasible_pct:.1f}%, "
            f"margin_g_mean={telemetry_margin_mean:.3f}, margin_g_min={telemetry_margin_min:.3f}, "
            f"speed_delta_mean={telemetry_speed_delta_mean:.3f} m/s, speed_delta_max={telemetry_speed_delta_max:.3f} m/s"
        )
    else:
        print("- turn_feasibility_telemetry: unavailable (legacy recording)")
    print(f"- steering_authority: mean|pre|={_safe_mean(pre_abs):.3f}, mean|final|={_safe_mean(final_abs):.3f}, mean_gap={_safe_mean(authority_gap):.3f}")
    print(f"- steering_transfer_ratio_mean: {_safe_mean(authority_ratio):.3f}")
    print(f"- limiter_deltas_mean: rate={rate_mean:.3f}, jerk={jerk_mean:.3f}, clip={clip_mean:.3f}, smooth={smooth_mean:.3f}")
    print(f"- stale_perception_pct: {stale_pct:.1f}%")
    print("")
    print(f"primary_classification: {primary}")


if __name__ == "__main__":
    main()
