#!/usr/bin/env python3
"""
Run-level trajectory triage with layer/location localization.

Outputs distance-banded error and a coarse root-cause hint:
- preclip_generation_issue
- postclip_farfield_distortion
- farfield_reference_or_generation_mismatch
- nearfield_source_issue
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


def parse_xy(flat: Any, stride: int = 2) -> list[tuple[float, float]]:
    arr = np.asarray(flat, dtype=np.float64).reshape(-1)
    n = arr.size // stride
    out: list[tuple[float, float]] = []
    for i in range(n):
        x = float(arr[stride * i])
        y = float(arr[stride * i + 1])
        if math.isfinite(x) and math.isfinite(y):
            out.append((x, y))
    return out


def to_forward_monotonic(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    valid = [p for p in points if p[1] >= 0.0]
    if not valid:
        return []
    min_idx = min(range(len(valid)), key=lambda i: valid[i][1])
    out: list[tuple[float, float]] = []
    last_y = None
    for p in valid[min_idx:]:
        if last_y is not None and p[1] + 1e-6 < last_y:
            continue
        out.append(p)
        last_y = p[1]
    return out


def sample_x(points: list[tuple[float, float]], y_target: float) -> float | None:
    if len(points) < 2:
        return None
    pts = sorted(points, key=lambda p: p[1])
    if y_target < pts[0][1] or y_target > pts[-1][1]:
        return None
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        if y1 < y_target:
            continue
        dy = y1 - y0
        if abs(dy) < 1e-6:
            return x1
        t = (y_target - y0) / dy
        return x0 + (x1 - x0) * t
    return None


def summarize(vals: list[float]) -> dict[str, float | int | None]:
    if not vals:
        return {"count": 0, "mean": None, "p95": None}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


def analyze_trajectory_layer_localization(rec: Path, clip_limit_m: float = 15.0) -> dict[str, Any]:
    bands = [(0.0, 8.0, [2, 4, 6, 8]), (8.0, 12.0, [8, 9, 10, 11, 12]), (12.0, 20.0, [12, 14, 16, 18, 20])]
    clip_limit = float(clip_limit_m)
    with h5py.File(rec, "r") as f:
        t = f["trajectory"]
        n = min(len(t["trajectory_points"]), len(t["oracle_points"]))
        xclip = np.asarray(t["diag_x_clip_count"][:n], dtype=np.float64) if "diag_x_clip_count" in t else np.zeros(n)
        premax = (
            np.asarray(t["diag_preclip_x_abs_max"][:n], dtype=np.float64)
            if "diag_preclip_x_abs_max" in t
            else np.full(n, np.nan, dtype=np.float64)
        )
        preclip_0_8 = np.asarray(t["diag_preclip_abs_mean_0_8m"][:n], dtype=np.float64) if "diag_preclip_abs_mean_0_8m" in t else np.array([], dtype=np.float64)
        preclip_8_12 = np.asarray(t["diag_preclip_abs_mean_8_12m"][:n], dtype=np.float64) if "diag_preclip_abs_mean_8_12m" in t else np.array([], dtype=np.float64)
        preclip_12_20 = np.asarray(t["diag_preclip_abs_mean_12_20m"][:n], dtype=np.float64) if "diag_preclip_abs_mean_12_20m" in t else np.array([], dtype=np.float64)
        postclip_12_20 = np.asarray(t["diag_postclip_abs_mean_12_20m"][:n], dtype=np.float64) if "diag_postclip_abs_mean_12_20m" in t else np.array([], dtype=np.float64)
        nearclip_12_20 = np.asarray(t["diag_postclip_near_clip_frac_12_20m"][:n], dtype=np.float64) if "diag_postclip_near_clip_frac_12_20m" in t else np.array([], dtype=np.float64)
        comp_lane_abs_12_20 = (
            np.asarray(t["diag_preclip_abs_mean_12_20m_lane_source_x"][:n], dtype=np.float64)
            if "diag_preclip_abs_mean_12_20m_lane_source_x" in t
            else np.array([], dtype=np.float64)
        )
        comp_dist_abs_12_20 = (
            np.asarray(t["diag_preclip_abs_mean_12_20m_distance_scale_delta_x"][:n], dtype=np.float64)
            if "diag_preclip_abs_mean_12_20m_distance_scale_delta_x" in t
            else np.array([], dtype=np.float64)
        )
        comp_offset_abs_12_20 = (
            np.asarray(t["diag_preclip_abs_mean_12_20m_camera_offset_delta_x"][:n], dtype=np.float64)
            if "diag_preclip_abs_mean_12_20m_camera_offset_delta_x" in t
            else np.array([], dtype=np.float64)
        )
        comp_lane_signed_12_20 = (
            np.asarray(t["diag_preclip_mean_12_20m_lane_source_x"][:n], dtype=np.float64)
            if "diag_preclip_mean_12_20m_lane_source_x" in t
            else np.array([], dtype=np.float64)
        )
        comp_dist_signed_12_20 = (
            np.asarray(t["diag_preclip_mean_12_20m_distance_scale_delta_x"][:n], dtype=np.float64)
            if "diag_preclip_mean_12_20m_distance_scale_delta_x" in t
            else np.array([], dtype=np.float64)
        )
        comp_offset_signed_12_20 = (
            np.asarray(t["diag_preclip_mean_12_20m_camera_offset_delta_x"][:n], dtype=np.float64)
            if "diag_preclip_mean_12_20m_camera_offset_delta_x" in t
            else np.array([], dtype=np.float64)
        )
        jump_reject = (
            np.asarray(t["diag_smoothing_jump_reject"][:n], dtype=np.float64)
            if "diag_smoothing_jump_reject" in t
            else np.array([], dtype=np.float64)
        )
        rate_limit = (
            np.asarray(t["diag_ref_x_rate_limit_active"][:n], dtype=np.float64)
            if "diag_ref_x_rate_limit_active" in t
            else np.array([], dtype=np.float64)
        )
        heading_suppr = (
            np.asarray(t["diag_heading_suppression_abs"][:n], dtype=np.float64)
            if "diag_heading_suppression_abs" in t
            else np.array([], dtype=np.float64)
        )
        x_suppr = (
            np.asarray(t["diag_ref_x_suppression_abs"][:n], dtype=np.float64)
            if "diag_ref_x_suppression_abs" in t
            else np.array([], dtype=np.float64)
        )
        dyn_horizon = (
            np.asarray(t["diag_dynamic_effective_horizon_m"][:n], dtype=np.float64)
            if "diag_dynamic_effective_horizon_m" in t
            else np.array([], dtype=np.float64)
        )
        dyn_final_scale = (
            np.asarray(t["diag_dynamic_effective_horizon_final_scale"][:n], dtype=np.float64)
            if "diag_dynamic_effective_horizon_final_scale" in t
            else np.array([], dtype=np.float64)
        )
        dyn_limiter_code = (
            np.asarray(t["diag_dynamic_effective_horizon_limiter_code"][:n], dtype=np.float64)
            if "diag_dynamic_effective_horizon_limiter_code" in t
            else np.array([], dtype=np.float64)
        )
        dyn_applied = (
            np.asarray(t["diag_dynamic_effective_horizon_applied"][:n], dtype=np.float64)
            if "diag_dynamic_effective_horizon_applied" in t
            else np.array([], dtype=np.float64)
        )

        result: dict[str, Any] = {
            "recording": str(rec),
            "frames_analyzed": int(n),
            "clip_limit_m": clip_limit,
            "x_clip_any_rate": float(np.mean(xclip > 0.5)),
            "bands": {},
        }

        for y0, y1, probes in bands:
            key = f"{int(y0)}-{int(y1)}m"
            lat_errs: list[float] = []
            absx_max: list[float] = []
            near_clip_frac: list[float] = []
            for i in range(n):
                planner = to_forward_monotonic(parse_xy(t["trajectory_points"][i], stride=3))
                oracle = to_forward_monotonic(parse_xy(t["oracle_points"][i], stride=2))
                if len(planner) < 2 or len(oracle) < 2:
                    continue
                errs_row: list[float] = []
                absx_row: list[float] = []
                near_row: list[float] = []
                for y in probes:
                    px = sample_x(planner, y)
                    ox = sample_x(oracle, y)
                    if px is not None:
                        absx = abs(px)
                        absx_row.append(absx)
                        near_row.append(1.0 if absx >= (clip_limit - 0.05) else 0.0)
                    if px is not None and ox is not None:
                        errs_row.append(abs(px - ox))
                if errs_row:
                    lat_errs.append(float(np.mean(errs_row)))
                if absx_row:
                    absx_max.append(float(np.max(absx_row)))
                if near_row:
                    near_clip_frac.append(float(np.mean(near_row)))

            result["bands"][key] = {
                "lateral_error_abs_m": summarize(lat_errs),
                "planner_abs_x_max_m": summarize(absx_max),
                "near_clip_fraction": summarize(near_clip_frac),
            }

        e0 = result["bands"]["0-8m"]["lateral_error_abs_m"]["mean"]
        e1 = result["bands"]["8-12m"]["lateral_error_abs_m"]["mean"]
        e2 = result["bands"]["12-20m"]["lateral_error_abs_m"]["mean"]
        near_clip_far = result["bands"]["12-20m"]["near_clip_fraction"]["mean"]
        premax_p95 = float(np.nanpercentile(premax, 95)) if np.isfinite(premax).any() else None
        result["preclip_abs_max_p95_m"] = premax_p95
        if preclip_0_8.size > 0:
            result["planner_source_bands"] = {
                "preclip_abs_mean_0_8m": summarize(preclip_0_8[np.isfinite(preclip_0_8)].tolist()),
                "preclip_abs_mean_8_12m": summarize(preclip_8_12[np.isfinite(preclip_8_12)].tolist()),
                "preclip_abs_mean_12_20m": summarize(preclip_12_20[np.isfinite(preclip_12_20)].tolist()),
                "postclip_abs_mean_12_20m": summarize(postclip_12_20[np.isfinite(postclip_12_20)].tolist()),
                "postclip_near_clip_frac_12_20m": summarize(nearclip_12_20[np.isfinite(nearclip_12_20)].tolist()),
            }
        if (
            comp_lane_abs_12_20.size > 0
            or comp_dist_abs_12_20.size > 0
            or comp_offset_abs_12_20.size > 0
        ):
            result["planner_source_components_12_20m"] = {
                "lane_source_abs_mean_x": summarize(
                    comp_lane_abs_12_20[np.isfinite(comp_lane_abs_12_20)].tolist()
                ),
                "distance_scale_delta_abs_mean_x": summarize(
                    comp_dist_abs_12_20[np.isfinite(comp_dist_abs_12_20)].tolist()
                ),
                "camera_offset_delta_abs_mean_x": summarize(
                    comp_offset_abs_12_20[np.isfinite(comp_offset_abs_12_20)].tolist()
                ),
                "lane_source_signed_mean_x": summarize(
                    comp_lane_signed_12_20[np.isfinite(comp_lane_signed_12_20)].tolist()
                ),
                "distance_scale_delta_signed_mean_x": summarize(
                    comp_dist_signed_12_20[np.isfinite(comp_dist_signed_12_20)].tolist()
                ),
                "camera_offset_delta_signed_mean_x": summarize(
                    comp_offset_signed_12_20[np.isfinite(comp_offset_signed_12_20)].tolist()
                ),
            }
        if jump_reject.size > 0 or rate_limit.size > 0 or heading_suppr.size > 0 or x_suppr.size > 0:
            result["reference_suppression"] = {
                "jump_reject_rate": float(
                    np.mean(jump_reject[np.isfinite(jump_reject)] > 0.5)
                ) if np.isfinite(jump_reject).any() else None,
                "ref_x_rate_limit_rate": float(
                    np.mean(rate_limit[np.isfinite(rate_limit)] > 0.5)
                ) if np.isfinite(rate_limit).any() else None,
                "heading_suppression_abs": summarize(
                    heading_suppr[np.isfinite(heading_suppr)].tolist()
                ),
                "ref_x_suppression_abs": summarize(
                    x_suppr[np.isfinite(x_suppr)].tolist()
                ),
            }
        if (
            dyn_horizon.size > 0
            or dyn_final_scale.size > 0
            or dyn_limiter_code.size > 0
            or dyn_applied.size > 0
        ):
            limiter_valid = dyn_limiter_code[np.isfinite(dyn_limiter_code)]
            result["dynamic_effective_horizon"] = {
                "effective_horizon_m": summarize(
                    dyn_horizon[np.isfinite(dyn_horizon)].tolist()
                ),
                "final_scale": summarize(
                    dyn_final_scale[np.isfinite(dyn_final_scale)].tolist()
                ),
                "applied_rate": float(
                    np.mean(dyn_applied[np.isfinite(dyn_applied)] > 0.5)
                ) if np.isfinite(dyn_applied).any() else None,
                "limiter_code_distribution": {
                    "none_0": int(np.sum(limiter_valid == 0.0)),
                    "speed_1": int(np.sum(limiter_valid == 1.0)),
                    "curvature_2": int(np.sum(limiter_valid == 2.0)),
                    "confidence_3": int(np.sum(limiter_valid == 3.0)),
                } if limiter_valid.size > 0 else None,
            }

        hint = "insufficient_data"
        if all(v is not None for v in [e0, e1, e2]):
            if e2 is not None and e2 > 4.0:
                if near_clip_far is not None and near_clip_far > 0.25:
                    hint = "postclip_farfield_distortion"
                elif premax_p95 is not None and premax_p95 > clip_limit + 0.5:
                    hint = "preclip_generation_issue"
                else:
                    hint = "farfield_reference_or_generation_mismatch"
            elif e0 is not None and e0 > 1.0:
                hint = "nearfield_source_issue"
            else:
                hint = "no_dominant_layer_local_issue"
        result["localization_hint"] = hint

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Trajectory layer/location localization report.")
    parser.add_argument("recording", nargs="?", help="Recording .h5 path")
    parser.add_argument("--latest", action="store_true", help="Use latest recording")
    parser.add_argument("--clip-limit-m", type=float, default=15.0)
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    rec = Path(args.recording) if args.recording else find_latest_recording()
    if args.latest:
        rec = find_latest_recording()
    if rec is None or not rec.exists():
        raise SystemExit("No recording found.")

    result = analyze_trajectory_layer_localization(rec, clip_limit_m=float(args.clip_limit_m))
    out_path = (
        Path(args.output_json)
        if args.output_json
        else Path("tmp/analysis") / f"trajectory_layer_localization_{rec.stem}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

