#!/usr/bin/env python3
"""
Signal Chain Diagnostic: D1 + R1
Analyzes a recording to measure:
  - Baseline trajectory-to-steering signal delay (frames)
  - Suppression summary (heading-zeroed, rate-limited, jerk-limited frames)
  - Curvature unit validation (R1: compare _compute_curvature vs vehicle-space)

Usage:
    python tools/analyze/analyze_signal_chain.py <recording_path> [--to-failure]
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np


G_MPS2 = 9.80665


def _safe_array(f: h5py.File, key: str) -> np.ndarray:
    """Read HDF5 dataset as float array, returning empty if missing."""
    if key not in f:
        return np.array([], dtype=np.float64)
    arr = np.asarray(f[key][:], dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr


def analyze_signal_chain(recording_path: str, analyze_to_failure: bool = True) -> dict:
    recording_path = Path(recording_path)
    if not recording_path.exists():
        return {"error": f"Recording not found: {recording_path}"}

    with h5py.File(recording_path, "r") as f:
        timestamps = _safe_array(f, "vehicle/timestamps")
        n_frames = len(timestamps)
        if n_frames < 10:
            return {"error": f"Recording too short: {n_frames} frames"}

        failure_frame = n_frames
        if analyze_to_failure:
            em = _safe_array(f, "control/emergency_stop")
            if len(em) > 0:
                fail_indices = np.where(em > 0.5)[0]
                if len(fail_indices) > 0:
                    failure_frame = int(fail_indices[0])

        scope = slice(0, failure_frame)

        traj_heading_raw = _safe_array(f, "trajectory/diag_raw_ref_heading")[scope]
        traj_heading_smoothed = _safe_array(f, "trajectory/reference_point_heading")[scope]
        traj_curvature = _safe_array(f, "trajectory/reference_point_curvature")[scope]
        traj_ref_x_raw = _safe_array(f, "trajectory/diag_raw_ref_x")[scope]
        traj_ref_x_smoothed = _safe_array(f, "trajectory/diag_smoothed_ref_x")[scope]
        traj_ref_x_suppression = _safe_array(f, "trajectory/diag_ref_x_suppression_abs")[scope]
        heading_zero_gate = _safe_array(f, "trajectory/diag_heading_zero_gate_active")[scope]
        rate_limit_active = _safe_array(f, "trajectory/diag_ref_x_rate_limit_active")[scope]
        jump_reject = _safe_array(f, "trajectory/diag_smoothing_jump_reject")[scope]

        ctrl_feedforward = _safe_array(f, "control/feedforward_steering")[scope]
        ctrl_steering = _safe_array(f, "control/steering")[scope]
        ctrl_steering_before = _safe_array(f, "control/steering_before_limits")[scope]
        ctrl_rate_limited = _safe_array(f, "control/steering_rate_limited_active")[scope]
        ctrl_jerk_limited = _safe_array(f, "control/steering_jerk_limited_active")[scope]
        ctrl_hard_clip = _safe_array(f, "control/steering_hard_clip_active")[scope]
        ctrl_limiter_code = _safe_array(f, "control/steering_first_limiter_stage_code")[scope]
        ctrl_authority_gap = _safe_array(f, "control/steering_authority_gap")[scope]
        ctrl_transfer_ratio = _safe_array(f, "control/steering_transfer_ratio")[scope]

        heading_error = _safe_array(f, "control/heading_error")[scope]
        lateral_error = _safe_array(f, "control/lateral_error")[scope]

        speed = _safe_array(f, "vehicle/speed")[scope]

        curve_upcoming = _safe_array(f, "control/curve_upcoming")[scope]
        curve_at_car = _safe_array(f, "control/curve_at_car")[scope]

    n = min(len(traj_heading_raw), len(ctrl_feedforward), failure_frame)
    if n < 10:
        return {"error": f"Insufficient aligned data: {n} frames"}

    heading_threshold = 0.01  # rad
    ff_threshold = 0.01

    traj_heading_onset = None
    for i in range(n):
        if abs(traj_heading_smoothed[i]) > heading_threshold:
            traj_heading_onset = i
            break

    ctrl_ff_onset = None
    for i in range(n):
        if abs(ctrl_feedforward[i]) > ff_threshold:
            ctrl_ff_onset = i
            break

    signal_delay_frames = None
    if traj_heading_onset is not None and ctrl_ff_onset is not None:
        signal_delay_frames = ctrl_ff_onset - traj_heading_onset

    heading_zero_frames = int(np.sum(heading_zero_gate > 0.5)) if len(heading_zero_gate) > 0 else 0
    rate_limit_frames = int(np.sum(rate_limit_active > 0.5)) if len(rate_limit_active) > 0 else 0
    jump_reject_frames = int(np.sum(jump_reject > 0.5)) if len(jump_reject) > 0 else 0
    ctrl_rate_limited_frames = int(np.sum(ctrl_rate_limited > 0.5)) if len(ctrl_rate_limited) > 0 else 0
    ctrl_jerk_limited_frames = int(np.sum(ctrl_jerk_limited > 0.5)) if len(ctrl_jerk_limited) > 0 else 0
    ctrl_hard_clip_frames = int(np.sum(ctrl_hard_clip > 0.5)) if len(ctrl_hard_clip) > 0 else 0

    first_limiter_frame = None
    for i in range(n):
        if len(ctrl_limiter_code) > i and ctrl_limiter_code[i] > 0:
            first_limiter_frame = i
            break

    max_rate_limit_streak = 0
    current_streak = 0
    for i in range(min(len(rate_limit_active), n)):
        if rate_limit_active[i] > 0.5:
            current_streak += 1
            max_rate_limit_streak = max(max_rate_limit_streak, current_streak)
        else:
            current_streak = 0

    curve_approach_start = None
    curve_approach_end = None
    if len(curve_at_car) > 0:
        for i in range(n):
            if curve_at_car[i] > 0.5:
                curve_approach_end = i
                break
    if len(curve_upcoming) > 0:
        for i in range(n):
            if curve_upcoming[i] > 0.5:
                curve_approach_start = i
                break
    if curve_approach_start is None and traj_heading_onset is not None:
        curve_approach_start = max(0, traj_heading_onset - 30)
    if curve_approach_end is None:
        curve_approach_end = failure_frame

    heading_zero_during_curve = 0
    rate_limit_during_curve = 0
    curve_frames = 0
    if curve_approach_start is not None:
        for i in range(curve_approach_start, min(curve_approach_end, n)):
            curve_frames += 1
            if len(heading_zero_gate) > i and heading_zero_gate[i] > 0.5:
                heading_zero_during_curve += 1
            if len(rate_limit_active) > i and rate_limit_active[i] > 0.5:
                rate_limit_during_curve += 1

    speed_at_curve_entry = None
    if curve_approach_end is not None and curve_approach_end < len(speed):
        speed_at_curve_entry = float(speed[curve_approach_end])

    curvature_at_entry = None
    if curve_approach_end is not None and curve_approach_end < len(traj_curvature):
        curvature_at_entry = float(traj_curvature[curve_approach_end])

    v_max_feasible = None
    if curvature_at_entry is not None and abs(curvature_at_entry) > 1e-6:
        a_lat_max_g = 0.25
        a_lat_max = a_lat_max_g * G_MPS2
        v_max_feasible = float(np.sqrt(a_lat_max / abs(curvature_at_entry)))

    curvature_range = {
        "min": float(np.min(np.abs(traj_curvature))) if len(traj_curvature) > 0 else None,
        "max": float(np.max(np.abs(traj_curvature))) if len(traj_curvature) > 0 else None,
        "mean_during_curve": (
            float(np.mean(np.abs(traj_curvature[curve_approach_start:curve_approach_end])))
            if curve_approach_start is not None and len(traj_curvature) > curve_approach_start
            else None
        ),
    }

    per_frame_window = []
    window_start = max(0, (curve_approach_start or 0) - 10)
    window_end = min(n, (curve_approach_end or n) + 30)
    for i in range(window_start, window_end):
        frame = {"frame": i}
        if i < len(traj_heading_raw):
            frame["traj_heading_raw"] = round(float(traj_heading_raw[i]), 5)
        if i < len(traj_heading_smoothed):
            frame["traj_heading_smoothed"] = round(float(traj_heading_smoothed[i]), 5)
        if i < len(traj_curvature):
            frame["traj_curvature"] = round(float(traj_curvature[i]), 6)
        if i < len(heading_zero_gate):
            frame["heading_zero_gate"] = int(heading_zero_gate[i] > 0.5)
        if i < len(rate_limit_active):
            frame["rate_limit_active"] = int(rate_limit_active[i] > 0.5)
        if i < len(ctrl_feedforward):
            frame["ctrl_feedforward"] = round(float(ctrl_feedforward[i]), 5)
        if i < len(ctrl_steering):
            frame["ctrl_steering"] = round(float(ctrl_steering[i]), 5)
        if i < len(ctrl_limiter_code):
            frame["ctrl_limiter_code"] = int(ctrl_limiter_code[i])
        if i < len(speed):
            frame["speed_mps"] = round(float(speed[i]), 3)
        per_frame_window.append(frame)

    result = {
        "recording": str(recording_path.name),
        "total_frames": n,
        "failure_frame": failure_frame if failure_frame < n_frames else None,
        "signal_chain": {
            "traj_heading_onset_frame": traj_heading_onset,
            "ctrl_feedforward_onset_frame": ctrl_ff_onset,
            "signal_delay_frames": signal_delay_frames,
            "first_limiter_hit_frame": first_limiter_frame,
        },
        "suppression_summary": {
            "heading_zero_gate_total_frames": heading_zero_frames,
            "traj_rate_limit_total_frames": rate_limit_frames,
            "traj_jump_reject_total_frames": jump_reject_frames,
            "ctrl_rate_limited_total_frames": ctrl_rate_limited_frames,
            "ctrl_jerk_limited_total_frames": ctrl_jerk_limited_frames,
            "ctrl_hard_clip_total_frames": ctrl_hard_clip_frames,
            "max_rate_limit_consecutive_streak": max_rate_limit_streak,
        },
        "curve_approach": {
            "curve_approach_start_frame": curve_approach_start,
            "curve_approach_end_frame": curve_approach_end,
            "curve_frames": curve_frames,
            "heading_zero_during_curve_frames": heading_zero_during_curve,
            "heading_zero_during_curve_pct": (
                round(100.0 * heading_zero_during_curve / curve_frames, 1)
                if curve_frames > 0 else None
            ),
            "rate_limit_during_curve_frames": rate_limit_during_curve,
            "rate_limit_during_curve_pct": (
                round(100.0 * rate_limit_during_curve / curve_frames, 1)
                if curve_frames > 0 else None
            ),
        },
        "speed_feasibility": {
            "speed_at_curve_entry_mps": speed_at_curve_entry,
            "curvature_at_entry": curvature_at_entry,
            "v_max_feasible_mps": v_max_feasible,
            "overspeed": (
                speed_at_curve_entry > v_max_feasible
                if speed_at_curve_entry is not None and v_max_feasible is not None
                else None
            ),
        },
        "curvature_range": curvature_range,
        "per_frame_window": per_frame_window,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Signal chain diagnostic (D1 + R1)")
    parser.add_argument("recording", type=str, help="Path to HDF5 recording")
    parser.add_argument("--to-failure", action="store_true", default=True,
                        help="Analyze only up to first failure (default: true)")
    parser.add_argument("--full", action="store_true",
                        help="Analyze full recording (ignore failures)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    args = parser.parse_args()

    result = analyze_signal_chain(args.recording, analyze_to_failure=not args.full)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fp:
            json.dump(result, fp, indent=2, default=str)
        print(f"Wrote: {out_path}")
    else:
        filtered = {k: v for k, v in result.items() if k != "per_frame_window"}
        print(json.dumps(filtered, indent=2, default=str))

    sc = result.get("signal_chain", {})
    ss = result.get("suppression_summary", {})
    ca = result.get("curve_approach", {})
    sf = result.get("speed_feasibility", {})

    print("\n=== D1 Signal Chain Summary ===")
    print(f"  Failure frame: {result.get('failure_frame', 'none')}")
    print(f"  Trajectory heading onset: frame {sc.get('traj_heading_onset_frame', '?')}")
    print(f"  Control feedforward onset: frame {sc.get('ctrl_feedforward_onset_frame', '?')}")
    print(f"  Signal delay: {sc.get('signal_delay_frames', '?')} frames")
    print(f"  First limiter hit: frame {sc.get('first_limiter_hit_frame', '?')}")
    print(f"\n=== Suppression Summary ===")
    print(f"  Heading zeroed: {ss.get('heading_zero_gate_total_frames', 0)} frames total")
    print(f"  Rate limit active: {ss.get('traj_rate_limit_total_frames', 0)} frames total")
    print(f"  Jump rejections: {ss.get('traj_jump_reject_total_frames', 0)} frames")
    print(f"  Ctrl rate limited: {ss.get('ctrl_rate_limited_total_frames', 0)} frames")
    print(f"  Ctrl jerk limited: {ss.get('ctrl_jerk_limited_total_frames', 0)} frames")
    print(f"  Max rate limit streak: {ss.get('max_rate_limit_consecutive_streak', 0)} frames")
    print(f"\n=== Curve Approach ===")
    print(f"  Curve approach: frames {ca.get('curve_approach_start_frame', '?')}-{ca.get('curve_approach_end_frame', '?')}")
    print(f"  Heading zero during curve: {ca.get('heading_zero_during_curve_pct', '?')}%")
    print(f"  Rate limit during curve: {ca.get('rate_limit_during_curve_pct', '?')}%")
    print(f"\n=== Speed Feasibility ===")
    print(f"  Speed at curve entry: {sf.get('speed_at_curve_entry_mps', '?')} m/s")
    print(f"  Curvature at entry: {sf.get('curvature_at_entry', '?')}")
    print(f"  V_max feasible (0.25g): {sf.get('v_max_feasible_mps', '?')} m/s")
    print(f"  Overspeed: {sf.get('overspeed', '?')}")
    print(f"\n=== R1 Curvature Range ===")
    cr = result.get("curvature_range", {})
    print(f"  Min |curvature|: {cr.get('min', '?')}")
    print(f"  Max |curvature|: {cr.get('max', '?')}")
    print(f"  Mean during curve: {cr.get('mean_during_curve', '?')}")

    return 0 if "error" not in result else 1


if __name__ == "__main__":
    sys.exit(main())
