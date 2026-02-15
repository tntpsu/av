#!/usr/bin/env python3
"""
Analyze a drive only up to first lane-failure event.

Primary goal:
- Measure how long the car stays in the intended lane before failing.
- Segment metrics into straight, curve entry, and curve-maintenance phases.

Failure definition (first event wins):
1) Centerline crossing (sustained sign flip on selected road-frame signal)
2) Off-road / emergency stop (control/emergency_stop == True)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np


@dataclass
class Phase:
    name: str
    start: int
    end_exclusive: int


def _as_float_array(f: h5py.File, key: str) -> Optional[np.ndarray]:
    if key not in f:
        return None
    return np.asarray(f[key][:], dtype=float)


def _as_bool_array(f: h5py.File, key: str) -> Optional[np.ndarray]:
    if key not in f:
        return None
    return np.asarray(f[key][:]).astype(bool)


def _first_true_idx(mask: np.ndarray) -> Optional[int]:
    idx = np.where(mask)[0]
    return int(idx[0]) if len(idx) else None


def _find_centerline_cross(
    signal: np.ndarray,
    initial_sign: int,
    min_abs: float,
    persist_frames: int,
    search_start: int = 0,
) -> Optional[int]:
    """Find first sustained sign flip relative to initial_sign."""
    n = len(signal)
    if n == 0 or initial_sign == 0:
        return None

    for i in range(max(0, search_start), n - persist_frames + 1):
        window = signal[i : i + persist_frames]
        if np.any(np.abs(window) < min_abs):
            continue
        # Require full persistence of opposite sign in window.
        if np.all(np.sign(window) == -initial_sign):
            return i
    return None


def _format_frame_time(frame: Optional[int], ts: np.ndarray) -> str:
    if frame is None:
        return "n/a"
    if len(ts) == 0 or frame >= len(ts):
        return f"{frame} (t=n/a)"
    return f"{frame} (t={ts[frame]:.2f}s)"


def _phase_stats(
    phase: Phase,
    ts: np.ndarray,
    stale: Optional[np.ndarray],
    ff: Optional[np.ndarray],
    curv: Optional[np.ndarray],
) -> Tuple[int, float, float, float, float, float]:
    s = max(0, phase.start)
    e = max(s, phase.end_exclusive)
    frames = max(0, e - s)
    if frames == 0:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0

    if len(ts) > 1:
        t0 = ts[s]
        t1 = ts[e - 1]
        duration = float(max(0.0, t1 - t0))
    else:
        duration = 0.0

    stale_pct = float(np.mean(stale[s:e]) * 100.0) if stale is not None else float("nan")
    ff_abs_mean = float(np.mean(np.abs(ff[s:e]))) if ff is not None else float("nan")
    curv_abs_mean = float(np.mean(np.abs(curv[s:e]))) if curv is not None else float("nan")
    ff_engaged_pct = (
        float(np.mean(np.abs(ff[s:e]) >= 0.01) * 100.0) if ff is not None else float("nan")
    )
    return frames, duration, stale_pct, ff_abs_mean, curv_abs_mean, ff_engaged_pct


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-to-failure lane analysis for an H5 recording.")
    parser.add_argument("recording", type=Path, help="Path to .h5 recording")
    parser.add_argument(
        "--centerline-signal",
        choices=["lane_center_error", "lateral_offset"],
        default="lane_center_error",
        help="Road-frame signal used to detect centerline crossing",
    )
    parser.add_argument(
        "--curve-start-frame",
        type=int,
        default=None,
        help="Optional manual curve-entry frame override",
    )
    parser.add_argument(
        "--entry-length-frames",
        type=int,
        default=20,
        help="Number of frames used for curve-entry phase",
    )
    parser.add_argument(
        "--heading-entry-threshold-deg",
        type=float,
        default=2.0,
        help="Auto curve-entry threshold on |road_heading_deg - initial_heading|",
    )
    parser.add_argument(
        "--cross-min-abs",
        type=float,
        default=0.15,
        help="Minimum absolute magnitude for centerline crossing detection",
    )
    parser.add_argument(
        "--cross-persist-frames",
        type=int,
        default=4,
        help="Required consecutive opposite-sign frames for crossing detection",
    )
    args = parser.parse_args()

    if not args.recording.exists():
        raise FileNotFoundError(f"Recording not found: {args.recording}")

    with h5py.File(args.recording, "r") as f:
        ts = _as_float_array(f, "vehicle/timestamps")
        if ts is None:
            ts = np.arange(len(f["control/steering"][:]), dtype=float) * (1.0 / 30.0)

        lane_center_error = _as_float_array(f, "vehicle/road_frame_lane_center_error")
        lateral_offset = _as_float_array(f, "vehicle/road_frame_lateral_offset")
        if lane_center_error is None and lateral_offset is None:
            raise KeyError("Missing both vehicle/road_frame_lane_center_error and vehicle/road_frame_lateral_offset")

        signal = lane_center_error if args.centerline_signal == "lane_center_error" else lateral_offset
        if signal is None:
            # Fallback if requested signal missing.
            signal = lane_center_error if lane_center_error is not None else lateral_offset

        stale = _as_bool_array(f, "control/using_stale_perception")
        if stale is None:
            stale = _as_bool_array(f, "perception/using_stale_data")

        ff = _as_float_array(f, "control/feedforward_steering")
        curv = _as_float_array(f, "control/path_curvature_input")
        emergency_stop = _as_bool_array(f, "control/emergency_stop")
        road_heading = _as_float_array(f, "vehicle/road_heading_deg")

    n = len(signal)
    init_window = min(60, n)
    init_med = float(np.median(signal[:init_window])) if init_window > 0 else 0.0
    initial_sign = int(np.sign(init_med)) if abs(init_med) >= 1e-3 else 0
    if initial_sign == 0 and n > 0:
        initial_sign = int(np.sign(signal[0]))

    # Determine curve start.
    if args.curve_start_frame is not None:
        curve_start = int(np.clip(args.curve_start_frame, 0, max(0, n - 1)))
        curve_start_source = "manual"
    else:
        if road_heading is not None and len(road_heading) == n:
            h0 = float(np.median(road_heading[:init_window])) if init_window > 0 else 0.0
            idx = np.where(np.abs(road_heading - h0) >= args.heading_entry_threshold_deg)[0]
            curve_start = int(idx[0]) if len(idx) else 0
            curve_start_source = "auto_heading"
        else:
            curve_start = 0
            curve_start_source = "fallback_0"

    centerline_cross = _find_centerline_cross(
        signal=signal,
        initial_sign=initial_sign,
        min_abs=args.cross_min_abs,
        persist_frames=max(1, args.cross_persist_frames),
        search_start=curve_start,
    )
    offroad_frame = _first_true_idx(emergency_stop) if emergency_stop is not None else None

    # First failure frame.
    fail_candidates = [x for x in [centerline_cross, offroad_frame] if x is not None]
    failure_frame = min(fail_candidates) if fail_candidates else None
    if failure_frame is None:
        analysis_end = n
        failure_reason = "none"
    else:
        analysis_end = failure_frame
        if centerline_cross is not None and failure_frame == centerline_cross:
            failure_reason = "centerline_cross"
        else:
            failure_reason = "offroad_emergency_stop"

    # Build phases only up to failure.
    entry_end = min(analysis_end, curve_start + max(1, args.entry_length_frames))
    phases = [
        Phase("straight", 0, min(curve_start, analysis_end)),
        Phase("curve_entry", min(curve_start, analysis_end), entry_end),
        Phase("curve_maintain", entry_end, analysis_end),
    ]

    print(f"recording: {args.recording}")
    print(f"frames_total: {n}")
    print(f"initial_lane_sign: {initial_sign:+d} (median_first_{init_window}={init_med:.3f})")
    print(f"curve_start: {_format_frame_time(curve_start, ts)} [{curve_start_source}]")
    print(f"centerline_cross: {_format_frame_time(centerline_cross, ts)}")
    print(f"offroad_emergency_stop: {_format_frame_time(offroad_frame, ts)}")
    print(f"first_failure: {_format_frame_time(failure_frame, ts)} reason={failure_reason}")
    print(f"analysis_window: [0, {analysis_end})")
    if len(ts) > 1 and analysis_end > 0:
        print(f"time_in_lane_before_failure: {max(0.0, ts[analysis_end - 1] - ts[0]):.2f}s")
    else:
        print("time_in_lane_before_failure: n/a")

    print("\nphase_metrics (computed only before first failure):")
    print("phase,frames,duration_s,stale_pct,ff_abs_mean,curvature_abs_mean,ff_engaged_pct")
    for ph in phases:
        frames, dur, stale_pct, ff_abs_mean, curv_abs_mean, ff_engaged_pct = _phase_stats(
            ph, ts, stale, ff, curv
        )
        print(
            f"{ph.name},{frames},{dur:.2f},{stale_pct:.1f},{ff_abs_mean:.4f},"
            f"{curv_abs_mean:.5f},{ff_engaged_pct:.1f}"
        )


if __name__ == "__main__":
    main()
