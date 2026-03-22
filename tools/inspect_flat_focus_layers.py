#!/usr/bin/env python3
"""
Perception / trajectory / control slice stats for flat-focus frame ranges.

For each contiguous range from analyze_flat_focus(), reports (within the range):
  - control: mean |lateral_error|, mean |steering|, mean |d(steer)/dt| proxy
  - perception: mean lane width, mean |ref_x − lane_center|
  - trajectory: mean |reference_point_x| (magnitude context)

CLI: python tools/inspect_flat_focus_layers.py <recording.h5> [--json] [--out path.md]
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


def _series(f: h5py.File, path: str, n: int) -> np.ndarray | None:
    if path not in f:
        return None
    d = f[path]
    m = min(int(d.shape[0]), n)
    return np.asarray(d[:m], dtype=np.float64)


def inspect_recording(
    recording_path: Path,
    *,
    grade_threshold: float = 0.02,
    abs_focus_m: float = 0.25,
    top_ranges: int = 12,
) -> dict:
    from analyze_grade_lateral_flat_focus import analyze_flat_focus  # noqa: E402

    summary = analyze_flat_focus(
        recording_path,
        grade_threshold=grade_threshold,
        abs_focus_m=abs_focus_m,
        top_n=25,
    )
    if "error" in summary:
        return summary

    ranges = summary.get("flat_high_lateral_ranges") or []
    ranges = sorted(ranges, key=lambda r: -r.get("length_frames", 0))[:top_ranges]

    with h5py.File(recording_path, "r") as f:
        n = int(summary.get("frames", 0))
        lat = _series(f, "control/lateral_error", n)
        steer = _series(f, "control/steering", n)
        left = _series(f, "perception/left_lane_line_x", n)
        right = _series(f, "perception/right_lane_line_x", n)
        refx = _series(f, "trajectory/reference_point_x", n)

        ts = _series(f, "vehicle/timestamps", n)
        if ts is None:
            ts = _series(f, "control/timestamps", n)
        if ts is None:
            ts = np.arange(n, dtype=np.float64) / 20.0

    if lat is None or steer is None:
        return {**summary, "error": "Missing control/lateral_error or steering"}

    perc_center = None
    if left is not None and right is not None:
        perc_center = (left + right) / 2.0
    traj_vs_perc_m = None
    if refx is not None and perc_center is not None:
        traj_vs_perc_m = np.abs(refx - perc_center)

    dsteer = np.zeros(n, dtype=np.float64)
    if n > 1:
        dt = np.diff(ts)
        dt = np.where(np.abs(dt) < 1e-9, 1.0 / 20.0, dt)
        dsteer[1:] = np.abs(np.diff(steer) / dt)

    slices: list[dict] = []
    for r in ranges:
        a, b = int(r["start_frame"]), int(r["end_frame"])
        sl = slice(a, b + 1)
        row: dict = {
            "start_frame": a,
            "end_frame": b,
            "length_frames": b - a + 1,
            "control": {
                "abs_lateral_error_mean_m": float(np.nanmean(np.abs(lat[sl]))),
                "abs_steering_mean": float(np.nanmean(np.abs(steer[sl]))),
                "abs_dsteer_dt_mean_per_s": float(np.nanmean(dsteer[sl])),
            },
        }
        if left is not None and right is not None:
            row["perception"] = {
                "lane_width_mean_m": float(np.nanmean(np.abs(right[sl] - left[sl]))),
            }
        else:
            row["perception"] = {"lane_width_mean_m": None}
        if traj_vs_perc_m is not None:
            row["trajectory"] = {
                "abs_ref_x_minus_lane_center_mean_m": float(np.nanmean(traj_vs_perc_m[sl])),
            }
        else:
            row["trajectory"] = {"abs_ref_x_minus_lane_center_mean_m": None}
        slices.append(row)

    return {
        "recording": recording_path.name,
        "flat_focus_threshold_m": summary.get("flat_bin", {}).get("focus_threshold_m"),
        "ranges_analyzed": len(slices),
        "slices": slices,
        "interpretation_notes": [
            "Large |lateral_error| with small traj_vs_perc suggests control/loop or timing.",
            "Large traj_vs_perc suggests planner ref vs perception center disagreement.",
            "Large lane width variance (not shown) — check perception stability in PhilViz.",
        ],
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Layer triage for flat-focus frame ranges")
    p.add_argument("recording", type=Path)
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", type=Path, default=None, help="Markdown report path")
    args = p.parse_args()

    out = inspect_recording(args.recording)
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(json.dumps(out, indent=2))

    if args.out:
        lines = [
            f"# Flat-focus layer slice: `{out.get('recording', '')}`\n",
            f"- Focus threshold (|lat| on flat): **{out.get('flat_focus_threshold_m', '—')}** m\n",
            f"- Ranges analyzed: **{out.get('ranges_analyzed', 0)}**\n",
            "\n## Slices (longest flat high-|lat| ranges first)\n",
            "\n| Frames | len | |lat| mean | |steer| mean | |d steer/dt| | lane w | |ref−perc| |\n",
            "|--------|-----|----------|------------|----------------|--------|------------|\n",
        ]
        for s in out.get("slices") or []:
            c = s.get("control") or {}
            perc = (s.get("perception") or {}).get("lane_width_mean_m")
            tr = (s.get("trajectory") or {}).get("abs_ref_x_minus_lane_center_mean_m")
            lines.append(
                f"| {s['start_frame']}–{s['end_frame']} | {s['length_frames']} | "
                f"{c.get('abs_lateral_error_mean_m', 0):.3f} | {c.get('abs_steering_mean', 0):.3f} | "
                f"{c.get('abs_dsteer_dt_mean_per_s', 0):.3f} | "
                f"{perc if perc is not None else '—'} | {tr if tr is not None else '—'} |\n"
            )
        lines.append("\n## Notes\n")
        for n in out.get("interpretation_notes") or []:
            lines.append(f"- {n}\n")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(lines), encoding="utf-8")
        print(f"\nWrote {args.out}", file=sys.stderr)

    return 0 if "error" not in out else 1


if __name__ == "__main__":
    raise SystemExit(main())
