#!/usr/bin/env python3
"""
Derive speed targets for a given centeredness goal on tight curves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def _load_array(h5: h5py.File, path: str) -> np.ndarray | None:
    if path in h5:
        return h5[path][:]
    return None


def _summarize_recording(
    recording: Path,
    curvature_threshold: float,
    centered_threshold: float,
) -> dict:
    with h5py.File(recording, "r") as h5:
        curvature = _load_array(h5, "ground_truth/path_curvature")
        lateral_error = _load_array(h5, "control/lateral_error")
        if lateral_error is None:
            lateral_error = _load_array(h5, "vehicle/road_frame_lane_center_error")
        speed = _load_array(h5, "vehicle/speed")

    if curvature is None or lateral_error is None or speed is None:
        raise RuntimeError(f"Missing required datasets in {recording}")

    n = min(len(curvature), len(lateral_error), len(speed))
    curvature = curvature[:n]
    lateral_error = lateral_error[:n]
    speed = speed[:n]

    tight_mask = np.abs(curvature) >= curvature_threshold
    if not np.any(tight_mask):
        return {
            "recording": recording.name,
            "tight_samples": 0,
            "tight_centered": 0.0,
            "tight_speed_mean": 0.0,
            "tight_curv_mean": 0.0,
        }

    tight_speed = speed[tight_mask]
    tight_error = lateral_error[tight_mask]
    tight_curv = np.abs(curvature[tight_mask])
    tight_centered = 100.0 * np.mean(np.abs(tight_error) < centered_threshold)
    return {
        "recording": recording.name,
        "tight_samples": int(tight_mask.sum()),
        "tight_centered": float(tight_centered),
        "tight_speed_mean": float(np.mean(tight_speed)),
        "tight_curv_mean": float(np.mean(tight_curv)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Estimate tight-curve speed target for a centeredness goal."
    )
    parser.add_argument("recordings", nargs="+", help="Recording .h5 files")
    parser.add_argument("--curvature-threshold", type=float, default=0.02)
    parser.add_argument("--centered-threshold", type=float, default=0.5)
    parser.add_argument("--target-centered", type=float, default=80.0)
    args = parser.parse_args()

    summaries = []
    for rec in args.recordings:
        summary = _summarize_recording(
            Path(rec),
            curvature_threshold=args.curvature_threshold,
            centered_threshold=args.centered_threshold,
        )
        summaries.append(summary)

    summaries.sort(key=lambda s: s["tight_speed_mean"])
    print("Tight-curve summaries:")
    for s in summaries:
        print(
            f"  {s['recording']}: centered={s['tight_centered']:.1f}% "
            f"speed_mean={s['tight_speed_mean']:.2f} "
            f"curv_mean={s['tight_curv_mean']:.4f} "
            f"samples={s['tight_samples']}"
        )

    target = args.target_centered
    eligible = [s for s in summaries if s["tight_centered"] >= target]
    if not eligible:
        print(f"\nNo runs hit {target:.1f}% centeredness on tight curves.")
        return 0

    best = max(eligible, key=lambda s: s["tight_speed_mean"])
    max_lat_accel = (best["tight_speed_mean"] ** 2) * best["tight_curv_mean"]
    print("\nRecommended targets:")
    print(
        f"  tight_speed_mean={best['tight_speed_mean']:.2f} m/s "
        f"(from {best['recording']})"
    )
    print(f"  implied max_lateral_accel≈{max_lat_accel:.2f} m/s^2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
