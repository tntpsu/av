#!/usr/bin/env python3
"""
Analyze trajectory turn-authority suppressors against oracle mismatch.

This script prefers recorded suppressor diagnostics (when present) and
falls back to proxy flags for older recordings.
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
        p0 = pts[i - 1]
        p1 = pts[i]
        if p1["y"] < y_target:
            continue
        dy = p1["y"] - p0["y"]
        if abs(dy) < 1e-6:
            return float(p1["x"])
        t = (y_target - p0["y"]) / dy
        return float(p0["x"] + (p1["x"] - p0["x"]) * t)
    return None


def sample_shape(points: list[dict[str, float]], y: float, half_window: float = 0.5) -> tuple[float, float] | None:
    x0 = sample_x(points, y - half_window)
    x1 = sample_x(points, y)
    x2 = sample_x(points, y + half_window)
    if x0 is None or x1 is None or x2 is None:
        return None
    dxdy = (x2 - x0) / (2.0 * half_window)
    d2xdy2 = (x2 - 2.0 * x1 + x0) / (half_window * half_window)
    heading_rad = math.atan2(dxdy, 1.0)
    denom = (1.0 + dxdy * dxdy) ** 1.5
    if abs(denom) < 1e-8:
        return None
    curvature = d2xdy2 / denom
    return heading_rad, curvature


def ratio(a: int, b: int) -> float | None:
    return (float(a) / float(b)) if b > 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Correlate trajectory suppressor proxies with under-turn events.")
    parser.add_argument("recording", nargs="?", help="Path to .h5 recording")
    parser.add_argument("--latest", action="store_true", help="Use latest recording in data/recordings")
    parser.add_argument("--underturn-threshold", type=float, default=0.6, help="Under-turn flag if ratio10 < threshold")
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional output JSON path (default: tmp/analysis/trajectory_turn_authority_rootcause_<recording>.json)",
    )
    args = parser.parse_args()

    rec_path: Path | None = None
    if args.latest:
        rec_path = find_latest_recording()
    elif args.recording:
        rec_path = Path(args.recording)
    else:
        latest = find_latest_recording()
        rec_path = latest
    if rec_path is None or not rec_path.exists():
        raise SystemExit("No recording found. Provide a recording path or use --latest.")

    with h5py.File(rec_path, "r") as f:
        traj_pts = f["trajectory/trajectory_points"]
        oracle_pts = f["trajectory/oracle_points"]
        ref_heading = np.asarray(f["trajectory/reference_point_heading"])
        ref_method = f["trajectory/reference_point_method"]
        coeffs = f["perception/lane_line_coefficients"]
        ctrl_curv = np.asarray(f["control/path_curvature_input"])

        n = min(len(traj_pts), len(oracle_pts), len(ref_heading), len(ref_method), len(coeffs), len(ctrl_curv))
        tgrp = f["trajectory"]

        rows: list[dict[str, Any]] = []
        for i in range(n):
            planner = to_forward_monotonic(parse_xy(traj_pts[i], stride=3))
            oracle = to_forward_monotonic(parse_xy(oracle_pts[i], stride=2))
            planner_shape = sample_shape(planner, 10.0)
            oracle_shape = sample_shape(oracle, 10.0)
            if planner_shape is None or oracle_shape is None:
                continue

            _, kp = planner_shape
            _, ko = oracle_shape
            if not (math.isfinite(kp) and math.isfinite(ko)) or abs(ko) < 1e-4:
                continue
            ratio10 = kp / ko

            method_val = ref_method[i]
            if isinstance(method_val, bytes):
                method_val = method_val.decode("utf-8", errors="ignore")
            method = str(method_val)

            heading_rad = float(ref_heading[i]) if math.isfinite(ref_heading[i]) else float("nan")
            small_heading_gate = math.isfinite(heading_rad) and abs(math.degrees(heading_rad)) < 1.0

            a = np.asarray(coeffs[i], dtype=np.float64).reshape(-1)
            heading_zero_gate = False
            if a.size >= 6 and np.isfinite(a).all():
                center_a = 0.5 * (float(a[0]) + float(a[3]))
                heading_zero_gate = abs(center_a) < 0.01

            control_curv = float(ctrl_curv[i]) if math.isfinite(ctrl_curv[i]) else float("nan")
            control_vs_oracle = None
            if math.isfinite(control_curv):
                control_vs_oracle = abs(control_curv) / abs(ko)

            diag_heading_zero = None
            if "diag_heading_zero_gate_active" in tgrp:
                val = float(np.asarray(tgrp["diag_heading_zero_gate_active"])[i])
                if math.isfinite(val):
                    diag_heading_zero = val > 0.5
            diag_small_heading = None
            if "diag_small_heading_gate_active" in tgrp:
                val = float(np.asarray(tgrp["diag_small_heading_gate_active"])[i])
                if math.isfinite(val):
                    diag_small_heading = val > 0.5
            diag_multi_lookahead = None
            if "diag_multi_lookahead_active" in tgrp:
                val = float(np.asarray(tgrp["diag_multi_lookahead_active"])[i])
                if math.isfinite(val):
                    diag_multi_lookahead = val > 0.5
            diag_jump_reject = None
            if "diag_smoothing_jump_reject" in tgrp:
                val = float(np.asarray(tgrp["diag_smoothing_jump_reject"])[i])
                if math.isfinite(val):
                    diag_jump_reject = val > 0.5
            diag_rate_limit = None
            if "diag_ref_x_rate_limit_active" in tgrp:
                val = float(np.asarray(tgrp["diag_ref_x_rate_limit_active"])[i])
                if math.isfinite(val):
                    diag_rate_limit = val > 0.5
            diag_dyn_applied = None
            if "diag_dynamic_effective_horizon_applied" in tgrp:
                val = float(np.asarray(tgrp["diag_dynamic_effective_horizon_applied"])[i])
                if math.isfinite(val):
                    diag_dyn_applied = val > 0.5
            diag_dyn_limiter_code = None
            if "diag_dynamic_effective_horizon_limiter_code" in tgrp:
                val = float(np.asarray(tgrp["diag_dynamic_effective_horizon_limiter_code"])[i])
                if math.isfinite(val):
                    diag_dyn_limiter_code = val

            rows.append(
                {
                    "frame": i,
                    "ratio10": ratio10,
                    "underturn10": bool(ratio10 < args.underturn_threshold),
                    "heading_zero_gate": bool(diag_heading_zero if diag_heading_zero is not None else heading_zero_gate),
                    "small_heading_gate": bool(diag_small_heading if diag_small_heading is not None else small_heading_gate),
                    "multi_lookahead_active": bool(diag_multi_lookahead if diag_multi_lookahead is not None else (method == "multi_lookahead_heading_blend")),
                    "smoothing_jump_reject_active": bool(diag_jump_reject) if diag_jump_reject is not None else False,
                    "ref_x_rate_limit_active": bool(diag_rate_limit) if diag_rate_limit is not None else False,
                    "dynamic_effective_horizon_applied": bool(diag_dyn_applied) if diag_dyn_applied is not None else False,
                    "dynamic_effective_horizon_confidence_limited": bool(diag_dyn_limiter_code == 3.0) if diag_dyn_limiter_code is not None else False,
                    "x_clip_count": float(np.asarray(f["trajectory/diag_x_clip_count"])[i]) if "diag_x_clip_count" in f["trajectory"] else None,
                    "x_clip_any": bool(
                        ("diag_x_clip_count" in f["trajectory"]) and math.isfinite(float(np.asarray(f["trajectory/diag_x_clip_count"])[i])) and float(np.asarray(f["trajectory/diag_x_clip_count"])[i]) > 0.0
                    ),
                    "x_clip_heavy": bool(
                        ("diag_x_clip_count" in f["trajectory"]) and math.isfinite(float(np.asarray(f["trajectory/diag_x_clip_count"])[i])) and float(np.asarray(f["trajectory/diag_x_clip_count"])[i]) >= 8.0
                    ),
                    "control_curv_ratio10": control_vs_oracle,
                    "control_curv_low": bool(control_vs_oracle is not None and control_vs_oracle < args.underturn_threshold),
                }
            )

    total = len(rows)
    under = [r for r in rows if r["underturn10"]]
    under_n = len(under)
    base_under_rate = ratio(under_n, total) or 0.0

    def summarize_flag(flag: str) -> dict[str, Any]:
        flag_rows = [r for r in rows if r[flag]]
        non_flag_rows = [r for r in rows if not r[flag]]
        under_flag = sum(1 for r in flag_rows if r["underturn10"])
        under_non_flag = sum(1 for r in non_flag_rows if r["underturn10"])
        p_flag = ratio(under_flag, len(flag_rows))
        p_non = ratio(under_non_flag, len(non_flag_rows))
        lift = (p_flag / base_under_rate) if (p_flag is not None and base_under_rate > 1e-9) else None
        return {
            "flag_true_frames": len(flag_rows),
            "flag_true_pct": ratio(len(flag_rows), total),
            "underturn_given_flag_true": p_flag,
            "underturn_given_flag_false": p_non,
            "underturn_lift_vs_baseline": lift,
        }

    summary = {
        "recording": str(rec_path),
        "frames_analyzed": total,
        "underturn_threshold_ratio10": args.underturn_threshold,
        "underturn_frames": under_n,
        "underturn_rate": base_under_rate,
        "flags": {
            "heading_zero_gate_active": summarize_flag("heading_zero_gate"),
            "small_heading_gate_active": summarize_flag("small_heading_gate"),
            "multi_lookahead_active": summarize_flag("multi_lookahead_active"),
            "smoothing_jump_reject_active": summarize_flag("smoothing_jump_reject_active"),
            "ref_x_rate_limit_active": summarize_flag("ref_x_rate_limit_active"),
            "dynamic_effective_horizon_applied": summarize_flag("dynamic_effective_horizon_applied"),
            "dynamic_effective_horizon_confidence_limited": summarize_flag("dynamic_effective_horizon_confidence_limited"),
            "x_clip_any": summarize_flag("x_clip_any"),
            "x_clip_heavy": summarize_flag("x_clip_heavy"),
            "control_curvature_low_proxy": summarize_flag("control_curv_low"),
        },
        "ratio10_stats": {
            "mean": float(np.mean([r["ratio10"] for r in rows])) if rows else None,
            "median": float(np.median([r["ratio10"] for r in rows])) if rows else None,
            "p95_abs": float(np.percentile(np.abs([r["ratio10"] for r in rows]), 95)) if rows else None,
        },
    }

    if args.output_json:
        out_path = Path(args.output_json)
    else:
        out_path = Path("tmp/analysis") / f"trajectory_turn_authority_rootcause_{rec_path.stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
