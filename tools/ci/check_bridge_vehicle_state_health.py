#!/usr/bin/env python3
"""CI gate for bridge vehicle-state health + minimum movement thresholds."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np


def _count_log_matches(log_path: Path, pattern: str) -> int:
    rx = re.compile(pattern)
    count = 0
    with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if rx.search(line):
                count += 1
    return count


def _load_movement_metrics(recording_path: Path) -> dict[str, float]:
    with h5py.File(recording_path, "r") as f:
        if "vehicle/speed" not in f:
            raise KeyError("Missing dataset vehicle/speed")
        if "vehicle/position" not in f:
            raise KeyError("Missing dataset vehicle/position")

        speed = np.asarray(f["vehicle/speed"][:], dtype=np.float64)
        position = np.asarray(f["vehicle/position"][:], dtype=np.float64)

        if speed.size == 0:
            return {"max_speed_mps": 0.0, "moving_frame_count": 0.0, "distance_m": 0.0}

        if position.ndim != 2 or position.shape[1] != 3:
            raise ValueError(f"Unexpected shape for vehicle/position: {position.shape}")

        deltas = np.diff(position, axis=0)
        distance_m = float(np.linalg.norm(deltas, axis=1).sum()) if deltas.size > 0 else 0.0
        max_speed_mps = float(np.nanmax(speed)) if speed.size > 0 else 0.0

        return {
            "max_speed_mps": max_speed_mps,
            "moving_frame_count": float(speed.size),  # caller computes thresholded moving count
            "distance_m": distance_m,
            "speed_series_count": float(speed.size),
            "speed_series": speed,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-log", required=True, type=Path)
    parser.add_argument("--recording", required=True, type=Path)
    parser.add_argument("--min-max-speed-mps", type=float, default=1.0)
    parser.add_argument("--min-moving-speed-mps", type=float, default=0.5)
    parser.add_argument("--min-moving-frames", type=int, default=120)
    parser.add_argument("--min-distance-m", type=float, default=15.0)
    args = parser.parse_args()

    failures: list[str] = []

    if not args.bridge_log.exists():
        failures.append(f"Bridge log not found: {args.bridge_log}")
    if not args.recording.exists():
        failures.append(f"Recording not found: {args.recording}")
    if failures:
        for msg in failures:
            print(f"[FAIL] {msg}")
        return 1

    state_500_count = _count_log_matches(
        args.bridge_log, r'/api/vehicle/state/latest HTTP/1\.1"\s+500'
    )
    nan_json_count = _count_log_matches(
        args.bridge_log, r"Out of range float values are not JSON compliant: nan"
    )

    metrics = _load_movement_metrics(args.recording)
    speed_series = metrics.pop("speed_series")
    moving_frames = int(np.sum(speed_series >= args.min_moving_speed_mps))
    max_speed = float(metrics["max_speed_mps"])
    distance_m = float(metrics["distance_m"])

    if state_500_count > 0:
        failures.append(f"/api/vehicle/state/latest 500 count = {state_500_count} (expected 0)")
    if nan_json_count > 0:
        failures.append(f"JSON NaN serialization errors = {nan_json_count} (expected 0)")
    if max_speed < args.min_max_speed_mps:
        failures.append(
            f"max speed {max_speed:.3f} m/s < threshold {args.min_max_speed_mps:.3f} m/s"
        )
    if moving_frames < args.min_moving_frames:
        failures.append(
            f"moving frames {moving_frames} < threshold {args.min_moving_frames} "
            f"(speed >= {args.min_moving_speed_mps:.3f} m/s)"
        )
    if distance_m < args.min_distance_m:
        failures.append(f"distance {distance_m:.3f} m < threshold {args.min_distance_m:.3f} m")

    print(
        "[INFO] bridge_500=%d nan_json=%d max_speed=%.3f moving_frames=%d distance=%.3f"
        % (state_500_count, nan_json_count, max_speed, moving_frames, distance_m)
    )

    if failures:
        for msg in failures:
            print(f"[FAIL] {msg}")
        return 1

    print("[PASS] bridge vehicle-state health gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
