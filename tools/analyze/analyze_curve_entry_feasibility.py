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
from typing import Optional

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

        s_pre = _as_float_array(f, "control/steering_pre_rate_limit")
        s_final = _as_float_array(f, "control/steering_post_smoothing")
        if s_final is None:
            s_final = _as_float_array(f, "control/steering")
        d_rate = _as_float_array(f, "control/steering_rate_limited_delta")
        d_jerk = _as_float_array(f, "control/steering_jerk_limited_delta")
        d_clip = _as_float_array(f, "control/steering_hard_clip_delta")
        d_smooth = _as_float_array(f, "control/steering_smoothing_delta")

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
    if lce is not None:
        lce = lce[:n]
    if lo is not None:
        lo = lo[:n]

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
    offroad = _first_true_idx(emergency_stop)
    fail_candidates = [x for x in [cross, offroad] if x is not None]
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
    authority_limited = _safe_mean(authority_gap) > args.authority_gap_threshold
    perception_limited = stale_pct >= args.stale_threshold_pct

    # Priority order: if speed infeasible often, call it speed-limited first.
    if speed_limited_pct >= 30.0:
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
    print(f"first_failure_frame: {first_failure if first_failure < n else 'none'}")
    print("")
    print("curve_entry_story:")
    print(f"- speed_feasibility: mean(v^2*kappa)={_safe_mean(ay_e):.3f} m/s^2, max={_safe_max(ay_e):.3f} m/s^2, budget={args.max_lateral_accel:.3f}")
    print(f"- speed_margin: mean={_safe_mean(speed_margin_e):.3f}, min={float(np.min(speed_margin_e)) if speed_margin_e.size else 0.0:.3f}")
    print(f"- speed_limited_frames_pct: {speed_limited_pct:.1f}%")
    print(f"- steering_authority: mean|pre|={_safe_mean(pre_abs):.3f}, mean|final|={_safe_mean(final_abs):.3f}, mean_gap={_safe_mean(authority_gap):.3f}")
    print(f"- steering_transfer_ratio_mean: {_safe_mean(authority_ratio):.3f}")
    print(f"- limiter_deltas_mean: rate={rate_mean:.3f}, jerk={jerk_mean:.3f}, clip={clip_mean:.3f}, smooth={smooth_mean:.3f}")
    print(f"- stale_perception_pct: {stale_pct:.1f}%")
    print("")
    print(f"primary_classification: {primary}")


if __name__ == "__main__":
    main()
