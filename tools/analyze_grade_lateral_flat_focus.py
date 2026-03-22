#!/usr/bin/env python3
"""
Find frame ranges where road grade is in the "flat" bin but |lateral_error| is worst.

Uses the same flat threshold as grade_lateral_v1 (±grade_threshold rise/run).
"High" lateral on flat: |lat| >= max(abs_focus_m floor, p90 of |lat| on flat frames).

Output: JSON (stdout or --out) with merged contiguous ranges + top-N worst flat frames.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def merge_ranges(idxs: np.ndarray) -> list[tuple[int, int]]:
    if idxs.size == 0:
        return []
    idxs = np.sort(np.unique(idxs))
    out: list[tuple[int, int]] = []
    start = prev = int(idxs[0])
    for x in idxs[1:]:
        x = int(x)
        if x == prev + 1:
            prev = x
        else:
            out.append((start, prev))
            start = prev = x
    out.append((start, prev))
    return out


def analyze_flat_focus(
    recording_path: Path,
    *,
    grade_threshold: float = 0.02,
    abs_focus_m: float = 0.25,
    top_n: int = 25,
) -> dict:
    with h5py.File(recording_path, "r") as f:
        if "control/lateral_error" not in f or "vehicle/road_grade" not in f:
            return {"error": "Need control/lateral_error and vehicle/road_grade"}
        n = min(
            int(f["control/lateral_error"].shape[0]),
            int(f["vehicle/road_grade"].shape[0]),
        )
        lat = np.asarray(f["control/lateral_error"][:n], dtype=np.float64)
        grade = np.asarray(f["vehicle/road_grade"][:n], dtype=np.float64)

    abs_lat = np.abs(lat)
    thr_g = float(grade_threshold)
    flat = (grade >= -thr_g) & (grade <= thr_g)
    flat_lat = abs_lat[flat]
    if flat_lat.size == 0:
        return {"error": "No flat-bin frames", "recording": recording_path.name}

    p50 = float(np.percentile(flat_lat, 50))
    p90 = float(np.percentile(flat_lat, 90))
    p95 = float(np.percentile(flat_lat, 95))
    thr = max(float(abs_focus_m), p90)
    bad_flat = flat & (abs_lat >= thr)

    ranges = merge_ranges(np.where(bad_flat)[0])
    flat_idx = np.where(flat)[0]
    order = np.argsort(-abs_lat[flat_idx])[: max(1, int(top_n))]
    top_frames = [
        {"frame": int(flat_idx[i]), "abs_lateral_error_m": round(float(abs_lat[flat_idx[i]]), 6)}
        for i in order
    ]

    return {
        "recording": recording_path.name,
        "frames": int(n),
        "grade_threshold_rise_per_run": thr_g,
        "flat_bin": {
            "frame_count": int(np.sum(flat)),
            "abs_lateral_error_m": {
                "p50": round(p50, 6),
                "p90": round(p90, 6),
                "p95": round(p95, 6),
            },
            "focus_threshold_m": round(thr, 6),
            "note": (
                f"bad_flat uses |lat| >= max({abs_focus_m}, p90_flat) = {thr:.4f} m"
            ),
        },
        "flat_high_lateral_ranges": [
            {"start_frame": a, "end_frame": b, "length_frames": b - a + 1}
            for a, b in ranges
        ],
        f"flat_top_{top_n}_frames_by_abs_lateral": top_frames,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Flat-bin worst lateral frame ranges")
    p.add_argument("recording", type=Path, help="Path to .h5")
    p.add_argument("--grade-threshold", type=float, default=0.02)
    p.add_argument("--abs-focus-m", type=float, default=0.25, help="Floor vs p90 for spike threshold")
    p.add_argument("--top-n", type=int, default=25)
    p.add_argument("--out", type=Path, default=None, help="Write JSON to this path")
    args = p.parse_args()

    out = analyze_flat_focus(
        args.recording,
        grade_threshold=args.grade_threshold,
        abs_focus_m=args.abs_focus_m,
        top_n=args.top_n,
    )
    text = json.dumps(out, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"\nWrote {args.out}", file=sys.stderr)
    return 0 if "error" not in out else 1


if __name__ == "__main__":
    raise SystemExit(main())
