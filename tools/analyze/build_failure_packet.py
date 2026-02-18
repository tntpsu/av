#!/usr/bin/env python3
"""
Build a compact failure packet around first failure.

Includes:
- failure frame + reason
- first limiter-hit frame (actual clipping only)
- windowed telemetry table around failure
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np


@dataclass
class FailureEvent:
    frame: Optional[int]
    reason: str


def _read_float(f: h5py.File, key: str) -> Optional[np.ndarray]:
    if key not in f:
        return None
    return np.asarray(f[key][:], dtype=float)


def _read_bool(f: h5py.File, key: str) -> Optional[np.ndarray]:
    if key not in f:
        return None
    return np.asarray(f[key][:]).astype(bool)


def _first_idx(mask: np.ndarray) -> Optional[int]:
    idx = np.where(mask)[0]
    return int(idx[0]) if len(idx) else None


def _find_centerline_cross(signal: np.ndarray, curve_start: int, min_abs: float = 0.15, persist: int = 4) -> Optional[int]:
    init = float(np.median(signal[: min(60, len(signal))]))
    sign0 = int(np.sign(init)) if abs(init) >= 1e-3 else int(np.sign(signal[0])) if len(signal) else 0
    if sign0 == 0:
        return None
    for i in range(max(0, curve_start), len(signal) - persist + 1):
        w = signal[i : i + persist]
        if np.any(np.abs(w) < min_abs):
            continue
        if np.all(np.sign(w) == -sign0):
            return i
    return None


def _first_limiter_hit(rate: np.ndarray, jerk: np.ndarray, clip: np.ndarray, smooth: np.ndarray) -> Optional[int]:
    mask = (rate > 1e-6) | (jerk > 1e-6) | (clip > 1e-6) | (smooth > 1e-6)
    return _first_idx(mask)


def _pick_failure(
    lane_center_error: Optional[np.ndarray],
    emergency_stop: Optional[np.ndarray],
    curve_start_frame: int,
) -> FailureEvent:
    cross = None
    offroad = _first_idx(emergency_stop) if emergency_stop is not None else None
    if lane_center_error is not None:
        cross = _find_centerline_cross(lane_center_error, curve_start=curve_start_frame)
    candidates = [x for x in [cross, offroad] if x is not None]
    frame = min(candidates) if candidates else None
    if frame is None:
        return FailureEvent(frame=None, reason="none")
    if cross is not None and frame == cross:
        return FailureEvent(frame=frame, reason="centerline_cross")
    return FailureEvent(frame=frame, reason="offroad_emergency_stop")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build pre-failure packet from recording.")
    parser.add_argument("recording", type=Path)
    parser.add_argument("--curve-start-frame", type=int, default=174)
    parser.add_argument("--window-before", type=int, default=30)
    parser.add_argument("--window-after", type=int, default=30)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or (Path("data/reports/failure_packets") / args.recording.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.recording, "r") as f:
        ts = _read_float(f, "vehicle/timestamps")
        if ts is None:
            n_fallback = len(f["control/steering"][:])
            ts = np.arange(n_fallback, dtype=float) * (1.0 / 30.0)
        lane_center_error = _read_float(f, "vehicle/road_frame_lane_center_error")
        emergency_stop = _read_bool(f, "control/emergency_stop")

        rate_delta = _read_float(f, "control/steering_rate_limited_delta")
        jerk_delta = _read_float(f, "control/steering_jerk_limited_delta")
        clip_delta = _read_float(f, "control/steering_hard_clip_delta")
        smooth_delta = _read_float(f, "control/steering_smoothing_delta")
        transfer = _read_float(f, "control/steering_transfer_ratio")
        authority_gap = _read_float(f, "control/steering_authority_gap")
        first_stage = _read_float(f, "control/steering_first_limiter_stage_code")
        schedule_active = _read_float(f, "control/curve_entry_schedule_active")
        commit_active = _read_float(f, "control/curve_commit_mode_active")
        assist_active = _read_float(f, "control/curve_entry_assist_active")

    if any(x is None for x in [rate_delta, jerk_delta, clip_delta, smooth_delta]):
        raise KeyError("Missing limiter delta datasets in recording.")

    failure = _pick_failure(lane_center_error, emergency_stop, curve_start_frame=args.curve_start_frame)
    first_limiter = _first_limiter_hit(rate_delta, jerk_delta, clip_delta, smooth_delta)

    failure_frame = failure.frame if failure.frame is not None else max(0, len(ts) - 1)
    s = max(0, failure_frame - args.window_before)
    e = min(len(ts), failure_frame + args.window_after + 1)

    packet: Dict[str, object] = {
        "recording": str(args.recording),
        "failure_frame": failure.frame,
        "failure_reason": failure.reason,
        "failure_time_s": float(ts[failure.frame]) if failure.frame is not None else None,
        "first_limiter_hit_frame": first_limiter,
        "first_limiter_hit_time_s": float(ts[first_limiter]) if first_limiter is not None else None,
        "window_start": s,
        "window_end_exclusive": e,
    }
    (out_dir / "packet.json").write_text(json.dumps(packet, indent=2))

    header = (
        "frame,time_s,transfer_ratio,authority_gap,rate_clip_delta,jerk_clip_delta,"
        "hard_clip_delta,smoothing_delta,first_stage,entry_schedule_active,commit_active,entry_assist_active"
    )
    rows = [header]
    for i in range(s, e):
        rows.append(
            f"{i},{ts[i]:.3f},"
            f"{(transfer[i] if transfer is not None else np.nan):.6f},"
            f"{(authority_gap[i] if authority_gap is not None else np.nan):.6f},"
            f"{rate_delta[i]:.6f},{jerk_delta[i]:.6f},{clip_delta[i]:.6f},{smooth_delta[i]:.6f},"
            f"{(first_stage[i] if first_stage is not None else np.nan):.1f},"
            f"{(schedule_active[i] if schedule_active is not None else np.nan):.0f},"
            f"{(commit_active[i] if commit_active is not None else np.nan):.0f},"
            f"{(assist_active[i] if assist_active is not None else np.nan):.0f}"
        )
    (out_dir / "window.csv").write_text("\n".join(rows) + "\n")

    summary = [
        "# Failure Packet",
        "",
        f"- recording: `{args.recording}`",
        f"- failure_frame: `{failure.frame}` ({failure.reason})",
        f"- first_limiter_hit_frame: `{first_limiter}`",
        f"- window: `[{s}, {e})`",
        "",
        "Artifacts:",
        "- `packet.json`",
        "- `window.csv`",
    ]
    (out_dir / "README.md").write_text("\n".join(summary) + "\n")
    print(f"Wrote failure packet to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
