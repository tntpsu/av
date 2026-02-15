#!/usr/bin/env python3
"""
Correlate visual overlay jump events with stream snap-risk indicators.

This script projects planner/oracle trajectories into image space using the same
core geometry as PhilViz and applies the same planner path draw filtering
(y-window + 140px discontinuity break). It then compares visible break events
against stream sync risk flags.
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


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array(
        [
            a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
            a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
            a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
            a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
        ],
        dtype=np.float64,
    )


def quat_rotate_vec(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    vq = np.array([v[0], v[1], v[2], 0.0], dtype=np.float64)
    q_conj = np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)
    res = quat_multiply(quat_multiply(q, vq), q_conj)
    return res[:3]


def project_vehicle_path_to_image(
    path_points: list[dict[str, float]],
    q: np.ndarray,
    veh_pos: np.ndarray,
    cam_pos: np.ndarray,
    cam_fwd: np.ndarray,
    road_center_y: float,
    width: float,
    height: float,
    hfov_deg: float,
    vfov_deg: float,
    nearfield_enabled: bool = True,
    nearfield_offset_m: float = -0.3,
    nearfield_blend_m: float = 10.0,
) -> list[tuple[float, float]]:
    if len(path_points) == 0:
        return []

    fwd_n = np.linalg.norm(cam_fwd)
    if fwd_n < 1e-9:
        return []
    forward = cam_fwd / fwd_n
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(world_up, forward)
    rn = np.linalg.norm(right)
    if rn < 1e-9:
        return []
    right /= rn
    up = np.cross(forward, right)

    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    fy = (height / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
    cx = width / 2.0
    cy = height / 2.0

    veh_forward_raw = quat_rotate_vec(np.array([0.0, 0.0, 1.0]), q)
    veh_right_raw = quat_rotate_vec(np.array([1.0, 0.0, 0.0]), q)

    veh_forward_2d = np.array([veh_forward_raw[0], 0.0, veh_forward_raw[2]], dtype=np.float64)
    fn = np.linalg.norm(veh_forward_2d)
    if fn < 1e-9:
        return []
    veh_forward_2d /= fn

    veh_right_2d = np.array([veh_right_raw[0], 0.0, veh_right_raw[2]], dtype=np.float64)
    rn2 = np.linalg.norm(veh_right_2d)
    if rn2 < 1e-6:
        veh_right_2d = np.array([veh_forward_2d[2], 0.0, -veh_forward_2d[0]], dtype=np.float64)
        rn2 = np.linalg.norm(veh_right_2d)
        if rn2 < 1e-9:
            return []
    veh_right_2d /= rn2

    projected: list[tuple[float, float]] = []
    for p in path_points:
        x_local = float(p["x"])
        y_local = float(p["y"])
        if not (math.isfinite(x_local) and math.isfinite(y_local)) or y_local < 0:
            continue

        blend_den = max(1e-3, float(nearfield_blend_m))
        near_weight_raw = max(0.0, min(1.0, 1.0 - (max(0.0, y_local) / blend_den)))
        near_weight = near_weight_raw if nearfield_enabled else 0.0
        world_y = road_center_y + nearfield_offset_m * near_weight

        world_pt = np.array(
            [
                veh_pos[0] + veh_right_2d[0] * x_local + veh_forward_2d[0] * y_local,
                world_y,
                veh_pos[2] + veh_right_2d[2] * x_local + veh_forward_2d[2] * y_local,
            ],
            dtype=np.float64,
        )

        rel = world_pt - cam_pos
        x_cam = float(np.dot(rel, right))
        y_cam = float(np.dot(rel, up))
        z_cam = float(np.dot(rel, forward))
        if z_cam <= 0.1:
            continue
        x_img = cx + (x_cam / z_cam) * fx
        y_img = cy - (y_cam / z_cam) * fy
        if math.isfinite(x_img) and math.isfinite(y_img):
            projected.append((x_img, y_img))

    return projected


def max_consecutive_jump_px(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 2:
        return None
    max_jump = 0.0
    prev = points[0]
    for cur in points[1:]:
        jump = math.hypot(cur[0] - prev[0], cur[1] - prev[1])
        if jump > max_jump:
            max_jump = jump
        prev = cur
    return max_jump


def compute_trajectory_min_y(height: float, y8m_actual: float | None) -> float:
    trajectory_min_y = math.floor(height * 0.2)
    if y8m_actual is not None and math.isfinite(y8m_actual) and y8m_actual > 0:
        px_per_meter = (height - y8m_actual) / 8.0
        if px_per_meter > 0:
            y_for_meters = height - (17.5 * px_per_meter)
            trajectory_min_y = max(0, math.floor(y_for_meters))
    return float(trajectory_min_y)


def apply_planner_draw_filter(
    projected: list[tuple[float, float]],
    image_height: float,
    trajectory_min_y: float,
    jump_break_threshold_px: float = 140.0,
) -> tuple[list[tuple[float, float]], int]:
    clipped: list[tuple[float, float]] = []
    prev: tuple[float, float] | None = None
    discontinuity_breaks = 0
    for pt in projected:
        x, y = pt
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        if y < trajectory_min_y or y > image_height:
            continue
        if prev is not None:
            jump_px = math.hypot(x - prev[0], y - prev[1])
            if jump_px > jump_break_threshold_px:
                discontinuity_breaks += 1
                prev = pt
                continue
        clipped.append(pt)
        prev = pt
    return clipped, discontinuity_breaks


def has_large_jump(
    points: list[tuple[float, float]],
    jump_threshold_px: float,
    y_min: float,
    y_max: float,
) -> bool:
    prev: tuple[float, float] | None = None
    for pt in points:
        x, y = pt
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        if y < y_min or y > y_max:
            continue
        if prev is not None:
            if math.hypot(x - prev[0], y - prev[1]) > jump_threshold_px:
                return True
        prev = pt
    return False


def ratio(a: int, b: int) -> float | None:
    return (float(a) / float(b)) if b > 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze correlation between overlay jump events and stream snap risk.")
    parser.add_argument("recording", nargs="?", help="Path to recording .h5")
    parser.add_argument("--latest", action="store_true", help="Use latest recording")
    parser.add_argument("--jump-threshold-px", type=float, default=140.0, help="Jump threshold for visibility risk")
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional output path (default: tmp/analysis/overlay_snap_jump_<recording>.json)",
    )
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

    jump_threshold = float(args.jump_threshold_px)

    rows: list[dict[str, Any]] = []
    with h5py.File(rec_path, "r") as f:
        v = f["vehicle"]
        t = f["trajectory"]
        n = min(
            len(t["trajectory_points"]),
            len(t["oracle_points"]),
            len(v["rotation"]),
            len(v["position"]),
            len(v["camera_pos_x"]),
            len(v["camera_pos_y"]),
            len(v["camera_pos_z"]),
            len(v["camera_forward_x"]),
            len(v["camera_forward_y"]),
            len(v["camera_forward_z"]),
            len(v["camera_horizontal_fov"]),
            len(v["camera_field_of_view"]),
            len(v["road_center_at_car_y"]),
            len(v["stream_front_frame_id_delta"]),
            len(v["stream_front_unity_dt_ms"]),
            len(v["camera_8m_screen_y"]) if "camera_8m_screen_y" in v else 10**12,
        )

        for i in range(n):
            q = np.asarray(v["rotation"][i], dtype=np.float64).reshape(-1)
            pos = np.asarray(v["position"][i], dtype=np.float64).reshape(-1)
            if q.size < 4 or pos.size < 3 or not np.isfinite(q).all() or not np.isfinite(pos).all():
                continue

            cam_pos = np.array(
                [v["camera_pos_x"][i], v["camera_pos_y"][i], v["camera_pos_z"][i]],
                dtype=np.float64,
            )
            cam_fwd = np.array(
                [v["camera_forward_x"][i], v["camera_forward_y"][i], v["camera_forward_z"][i]],
                dtype=np.float64,
            )
            if not np.isfinite(cam_pos).all() or not np.isfinite(cam_fwd).all():
                continue

            hfov = float(v["camera_horizontal_fov"][i])
            vfov = float(v["camera_field_of_view"][i])
            if not (math.isfinite(hfov) and hfov > 1.0):
                hfov = 110.0
            if not (math.isfinite(vfov) and vfov > 1.0):
                vfov = math.degrees(2.0 * math.atan(math.tan(math.radians(hfov) / 2.0) * (480.0 / 640.0)))

            road_y = float(v["road_center_at_car_y"][i])
            if not math.isfinite(road_y):
                road_y = float(pos[1])

            planner = to_forward_monotonic(parse_xy(t["trajectory_points"][i], stride=3))
            oracle = to_forward_monotonic(parse_xy(t["oracle_points"][i], stride=2))

            proj_planner = project_vehicle_path_to_image(
                planner, q[:4], pos[:3], cam_pos, cam_fwd, road_y, 640.0, 480.0, hfov, vfov
            )
            proj_oracle = project_vehicle_path_to_image(
                oracle, q[:4], pos[:3], cam_pos, cam_fwd, road_y, 640.0, 480.0, hfov, vfov
            )

            y8m = float(v["camera_8m_screen_y"][i]) if "camera_8m_screen_y" in v else float("nan")
            y8m_actual = y8m if math.isfinite(y8m) and y8m > 0 else 350.0
            traj_min_y = compute_trajectory_min_y(480.0, y8m_actual)

            planner_drawn, planner_breaks = apply_planner_draw_filter(
                proj_planner, image_height=480.0, trajectory_min_y=traj_min_y, jump_break_threshold_px=float(jump_threshold)
            )
            max_jump_planner_raw = max_consecutive_jump_px(proj_planner)
            max_jump_planner_drawn = max_consecutive_jump_px(planner_drawn)
            max_jump_oracle_raw = max_consecutive_jump_px(proj_oracle)

            max_jump = max(
                [x for x in [max_jump_planner_drawn, max_jump_oracle_raw, max_jump_planner_raw] if x is not None],
                default=None,
            )
            if max_jump is None:
                continue

            fd = float(v["stream_front_frame_id_delta"][i])
            udt = float(v["stream_front_unity_dt_ms"][i])
            snap_risk = (math.isfinite(fd) and fd >= 2.0) or (math.isfinite(udt) and abs(udt) >= 20.0)
            oracle_large_jump = has_large_jump(
                proj_oracle,
                jump_threshold_px=float(jump_threshold),
                y_min=traj_min_y,
                y_max=480.0,
            )
            visible_jump_risk = (not snap_risk) and (planner_breaks > 0 or oracle_large_jump)

            rows.append(
                {
                    "frame": i,
                    "snap_risk": bool(snap_risk),
                    "jump_risk": bool(visible_jump_risk),
                    "max_jump_px": float(max_jump),
                    "planner_discontinuity_breaks": int(planner_breaks),
                    "oracle_large_jump": bool(oracle_large_jump),
                    "planner_drawn_points": len(planner_drawn),
                    "planner_projected_points": len(proj_planner),
                    "max_jump_planner_raw_px": max_jump_planner_raw,
                    "max_jump_planner_drawn_px": max_jump_planner_drawn,
                    "max_jump_oracle_raw_px": max_jump_oracle_raw,
                    "front_frame_delta": fd,
                    "front_unity_dt_ms": udt,
                }
            )

    total = len(rows)
    snap_true = [r for r in rows if r["snap_risk"]]
    snap_false = [r for r in rows if not r["snap_risk"]]
    jump_true = [r for r in rows if r["jump_risk"]]

    p_jump_given_snap = ratio(sum(1 for r in snap_true if r["jump_risk"]), len(snap_true))
    p_jump_given_no_snap = ratio(sum(1 for r in snap_false if r["jump_risk"]), len(snap_false))
    lift = (
        (p_jump_given_snap / p_jump_given_no_snap)
        if p_jump_given_snap is not None and p_jump_given_no_snap is not None and p_jump_given_no_snap > 1e-9
        else None
    )

    summary = {
        "recording": str(rec_path),
        "frames_analyzed": total,
        "jump_threshold_px": jump_threshold,
        "snap_risk_rate": ratio(len(snap_true), total),
        "jump_risk_rate": ratio(len(jump_true), total),
        "p_jump_given_snap_risk": p_jump_given_snap,
        "p_jump_given_no_snap_risk": p_jump_given_no_snap,
        "jump_risk_lift_snap_vs_no_snap": lift,
        "planner_breaks": {
            "frames_with_breaks": int(sum(1 for r in rows if r["planner_discontinuity_breaks"] > 0)),
            "mean_breaks_per_frame": float(np.mean([r["planner_discontinuity_breaks"] for r in rows])) if rows else None,
            "p95_breaks_per_frame": float(np.percentile([r["planner_discontinuity_breaks"] for r in rows], 95)) if rows else None,
        },
        "max_jump_px": {
            "mean": float(np.mean([r["max_jump_px"] for r in rows])) if rows else None,
            "p95": float(np.percentile([r["max_jump_px"] for r in rows], 95)) if rows else None,
            "max": float(np.max([r["max_jump_px"] for r in rows])) if rows else None,
        },
    }

    if args.output_json:
        out_path = Path(args.output_json)
    else:
        out_path = Path("tmp/analysis") / f"overlay_snap_jump_{rec_path.stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
