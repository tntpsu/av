#!/usr/bin/env python3
"""
Correlate trajectory pre-clip lateral outliers with under-turn events.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def find_latest_recording() -> Path | None:
    candidates = sorted(Path("data/recordings").glob("recording_*.h5"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def parse_xy(flat: Any, stride: int = 2) -> list[dict[str, float]]:
    arr = np.asarray(flat, dtype=np.float64).reshape(-1)
    out: list[dict[str, float]] = []
    if stride < 2:
        stride = 2
    n = arr.size // stride
    for i in range(n):
        x = float(arr[stride * i])
        y = float(arr[stride * i + 1])
        if math.isfinite(x) and math.isfinite(y):
            out.append({"x": x, "y": y})
    return out


def to_forward_monotonic(points: list[dict[str, float]]) -> list[dict[str, float]]:
    valid = [p for p in points if p["y"] >= 0 and math.isfinite(p["x"]) and math.isfinite(p["y"])]
    if not valid:
        return []
    min_idx = min(range(len(valid)), key=lambda i: valid[i]["y"])
    out: list[dict[str, float]] = []
    last_y = None
    for p in valid[min_idx:]:
        if last_y is not None and p["y"] + 1e-6 < last_y:
            continue
        out.append(p)
        last_y = p["y"]
    return out


def sample_x(points: list[dict[str, float]], y_target: float) -> float | None:
    if len(points) < 2:
        return None
    pts = sorted(points, key=lambda p: p["y"])
    if y_target < pts[0]["y"] or y_target > pts[-1]["y"]:
        return None
    for i in range(1, len(pts)):
        p0, p1 = pts[i - 1], pts[i]
        if p1["y"] < y_target:
            continue
        dy = p1["y"] - p0["y"]
        if abs(dy) < 1e-6:
            return p1["x"]
        t = (y_target - p0["y"]) / dy
        return p0["x"] + (p1["x"] - p0["x"]) * t
    return None


def sample_shape(points: list[dict[str, float]], y: float, h: float = 0.5) -> tuple[float, float] | None:
    x0 = sample_x(points, y - h)
    x1 = sample_x(points, y)
    x2 = sample_x(points, y + h)
    if x0 is None or x1 is None or x2 is None:
        return None
    dxdy = (x2 - x0) / (2.0 * h)
    d2xdy2 = (x2 - 2.0 * x1 + x0) / (h * h)
    heading = math.atan2(dxdy, 1.0)
    denom = (1.0 + dxdy * dxdy) ** 1.5
    if abs(denom) < 1e-8:
        return None
    curvature = d2xdy2 / denom
    return heading, curvature


def ratio(a: int, b: int) -> float | None:
    return (float(a) / float(b)) if b > 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Correlate pre-clip outliers with under-turn events.")
    parser.add_argument("recording", nargs="?", help="Path to recording .h5")
    parser.add_argument("--latest", action="store_true", help="Use latest recording")
    parser.add_argument("--underturn-threshold", type=float, default=0.6, help="Under-turn if ratio10 < threshold")
    parser.add_argument("--outlier-threshold-m", type=float, default=8.0, help="Outlier threshold for pre-clip abs max")
    parser.add_argument("--output-json", type=str, default="", help="Output JSON path")
    args = parser.parse_args()

    rec_path: Path | None
    if args.latest:
        rec_path = find_latest_recording()
    elif args.recording:
        rec_path = Path(args.recording)
    else:
        rec_path = find_latest_recording()
    if rec_path is None or not rec_path.exists():
        raise SystemExit("No recording found.")

    rows: list[dict[str, Any]] = []
    with h5py.File(rec_path, "r") as f:
        t = f["trajectory"]
        has_fullpath_preclip = "diag_preclip_x_abs_max" in t and "diag_preclip_x_abs_p95" in t
        if has_fullpath_preclip:
            n = min(len(t["trajectory_points"]), len(t["oracle_points"]), len(t["diag_preclip_x_abs_max"]), len(t["diag_preclip_x_abs_p95"]))
        else:
            n = min(len(t["trajectory_points"]), len(t["oracle_points"]), len(t["diag_preclip_x0"]), len(t["diag_preclip_x1"]), len(t["diag_preclip_x2"]))
        for i in range(n):
            planner = to_forward_monotonic(parse_xy(t["trajectory_points"][i], stride=3))
            oracle = to_forward_monotonic(parse_xy(t["oracle_points"][i], stride=2))
            ps = sample_shape(planner, 10.0)
            os = sample_shape(oracle, 10.0)
            if ps is None or os is None:
                continue
            _, kp = ps
            _, ko = os
            if not (math.isfinite(kp) and math.isfinite(ko)) or abs(ko) < 1e-4:
                continue
            ratio10 = kp / ko
            under = ratio10 < float(args.underturn_threshold)

            if has_fullpath_preclip:
                preclip_max = float(t["diag_preclip_x_abs_max"][i])
                preclip_p95 = float(t["diag_preclip_x_abs_p95"][i])
                if not (math.isfinite(preclip_max) and math.isfinite(preclip_p95)):
                    continue
            else:
                preclip_vals = np.array(
                    [
                        float(t["diag_preclip_x0"][i]),
                        float(t["diag_preclip_x1"][i]),
                        float(t["diag_preclip_x2"][i]),
                    ],
                    dtype=np.float64,
                )
                preclip_abs = np.abs(preclip_vals[np.isfinite(preclip_vals)])
                if preclip_abs.size == 0:
                    continue
                preclip_max = float(np.max(preclip_abs))
                preclip_p95 = float(np.percentile(preclip_abs, 95))

            rows.append(
                {
                    "frame": i,
                    "underturn10": bool(under),
                    "ratio10": float(ratio10),
                    "preclip_abs_max": preclip_max,
                    "preclip_abs_p95": preclip_p95,
                    "preclip_outlier": preclip_max >= float(args.outlier_threshold_m),
                }
            )

    total = len(rows)
    under = [r for r in rows if r["underturn10"]]
    outlier = [r for r in rows if r["preclip_outlier"]]
    no_outlier = [r for r in rows if not r["preclip_outlier"]]

    p_under_given_outlier = ratio(sum(1 for r in outlier if r["underturn10"]), len(outlier))
    p_under_given_no_outlier = ratio(sum(1 for r in no_outlier if r["underturn10"]), len(no_outlier))
    lift = (
        (p_under_given_outlier / p_under_given_no_outlier)
        if p_under_given_outlier is not None and p_under_given_no_outlier is not None and p_under_given_no_outlier > 1e-9
        else None
    )

    summary = {
        "recording": str(rec_path),
        "frames_analyzed": total,
        "underturn_threshold_ratio10": float(args.underturn_threshold),
        "preclip_outlier_threshold_m": float(args.outlier_threshold_m),
        "preclip_metric_source": "full_path_preclip_abs" if has_fullpath_preclip else "lookahead_triplet_preclip_abs",
        "underturn_rate": ratio(len(under), total),
        "preclip_outlier_rate": ratio(len(outlier), total),
        "p_underturn_given_outlier": p_under_given_outlier,
        "p_underturn_given_no_outlier": p_under_given_no_outlier,
        "underturn_lift_outlier_vs_no_outlier": lift,
        "preclip_abs_max_m": {
            "mean": float(np.mean([r["preclip_abs_max"] for r in rows])) if rows else None,
            "p95": float(np.percentile([r["preclip_abs_max"] for r in rows], 95)) if rows else None,
            "max": float(np.max([r["preclip_abs_max"] for r in rows])) if rows else None,
        },
    }

    if args.output_json:
        out_path = Path(args.output_json)
    else:
        out_path = Path("tmp/analysis") / f"trajectory_preclip_outliers_{rec_path.stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
