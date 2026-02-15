#!/usr/bin/env python3
"""
Single-frame fiducial reprojection check to isolate geometry from timing.

Compares Unity screen-truth fiducials against reprojections from:
- vehicle_true_xy (preferred)
- vehicle_xy_legacy
- vehicle_monotonic_xy
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


def parse_xy(flat: Any) -> np.ndarray:
    arr = np.asarray(flat, dtype=np.float64).reshape(-1)
    n = arr.size // 2
    if n <= 0:
        return np.empty((0, 2), dtype=np.float64)
    arr = arr[: n * 2].reshape(n, 2)
    good = np.isfinite(arr).all(axis=1)
    return arr[good]


def parse_xyz(flat: Any) -> np.ndarray:
    arr = np.asarray(flat, dtype=np.float64).reshape(-1)
    n = arr.size // 3
    if n <= 0:
        return np.empty((0, 3), dtype=np.float64)
    arr = arr[: n * 3].reshape(n, 3)
    good = np.isfinite(arr).all(axis=1)
    return arr[good]


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
    return quat_multiply(quat_multiply(q, vq), q_conj)[:3]


def project_vehicle_points(
    vehicle_points: np.ndarray,
    q: np.ndarray,
    veh_pos: np.ndarray,
    cam_pos: np.ndarray,
    cam_fwd: np.ndarray,
    ground_y: float,
    width: float,
    height: float,
    hfov_deg: float,
    vfov_deg: float,
    nearfield_enabled: bool,
    nearfield_offset_m: float,
    nearfield_blend_m: float,
) -> np.ndarray:
    if vehicle_points.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    fwd_n = np.linalg.norm(cam_fwd)
    if fwd_n < 1e-9:
        return np.empty((0, 2), dtype=np.float64)
    forward = cam_fwd / fwd_n
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(world_up, forward)
    right_n = np.linalg.norm(right)
    if right_n < 1e-9:
        return np.empty((0, 2), dtype=np.float64)
    right /= right_n
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
        return np.empty((0, 2), dtype=np.float64)
    veh_forward_2d /= fn

    veh_right_2d = np.array([veh_right_raw[0], 0.0, veh_right_raw[2]], dtype=np.float64)
    rn = np.linalg.norm(veh_right_2d)
    if rn < 1e-6:
        veh_right_2d = np.array([veh_forward_2d[2], 0.0, -veh_forward_2d[0]], dtype=np.float64)
        rn = np.linalg.norm(veh_right_2d)
        if rn < 1e-9:
            return np.empty((0, 2), dtype=np.float64)
    veh_right_2d /= rn

    out = []
    blend_den = max(1e-3, float(nearfield_blend_m))
    for x_local, y_local in vehicle_points:
        if y_local < 0.0:
            continue
        near_weight_raw = max(0.0, min(1.0, 1.0 - (max(0.0, y_local) / blend_den)))
        near_weight = near_weight_raw if nearfield_enabled else 0.0
        world_y = ground_y + nearfield_offset_m * near_weight
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
            out.append([np.nan, np.nan])
            continue
        x_img = cx + (x_cam / z_cam) * fx
        y_img = cy - (y_cam / z_cam) * fy
        out.append([x_img, y_img])
    return np.asarray(out, dtype=np.float64)


def project_world_points(
    world_points: np.ndarray,
    cam_pos: np.ndarray,
    cam_fwd: np.ndarray,
    width: float,
    height: float,
    hfov_deg: float,
    vfov_deg: float,
) -> np.ndarray:
    if world_points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    fwd_n = np.linalg.norm(cam_fwd)
    if fwd_n < 1e-9:
        return np.empty((0, 2), dtype=np.float64)
    forward = cam_fwd / fwd_n
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(world_up, forward)
    right_n = np.linalg.norm(right)
    if right_n < 1e-9:
        return np.empty((0, 2), dtype=np.float64)
    right /= right_n
    up = np.cross(forward, right)
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    fy = (height / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
    cx = width / 2.0
    cy = height / 2.0
    out = []
    for wx, wy, wz in world_points:
        rel = np.array([wx - cam_pos[0], wy - cam_pos[1], wz - cam_pos[2]], dtype=np.float64)
        x_cam = float(np.dot(rel, right))
        y_cam = float(np.dot(rel, up))
        z_cam = float(np.dot(rel, forward))
        if z_cam <= 0.1:
            out.append([np.nan, np.nan])
            continue
        x_img = cx + (x_cam / z_cam) * fx
        y_img = cy - (y_cam / z_cam) * fy
        out.append([x_img, y_img])
    return np.asarray(out, dtype=np.float64)


def summarize_errors(projected: np.ndarray, truth: np.ndarray, spacing_m: float) -> dict[str, Any]:
    n = min(len(projected), len(truth))
    if n <= 0:
        return {
            "valid_pairs": 0,
            "total_pairs": 0,
            "mean_err_px": None,
            "max_err_px": None,
            "err_5m_px": None,
            "err_10m_px": None,
            "err_15m_px": None,
        }
    errs = []
    valid = 0
    for i in range(n):
        p = projected[i]
        t = truth[i]
        t_ok = np.isfinite(t).all() and t[0] >= 0.0 and t[1] >= 0.0
        p_ok = np.isfinite(p).all()
        if not (t_ok and p_ok):
            errs.append(np.nan)
            continue
        errs.append(float(np.hypot(p[0] - t[0], p[1] - t[1])))
        valid += 1
    errs_arr = np.asarray(errs, dtype=np.float64)

    def read_at(m: float) -> float | None:
        if spacing_m <= 0:
            return None
        idx = int(round(m / spacing_m))
        if idx < 0 or idx >= errs_arr.size:
            return None
        v = errs_arr[idx]
        return float(v) if np.isfinite(v) else None

    return {
        "valid_pairs": int(valid),
        "total_pairs": int(n),
        "mean_err_px": float(np.nanmean(errs_arr)) if valid > 0 else None,
        "max_err_px": float(np.nanmax(errs_arr)) if valid > 0 else None,
        "err_5m_px": read_at(5.0),
        "err_10m_px": read_at(10.0),
        "err_15m_px": read_at(15.0),
    }


def choose_frame(f: h5py.File, explicit_idx: int | None) -> int:
    cam_n = len(f["camera/timestamps"])
    if explicit_idx is not None:
        return int(max(0, min(cam_n - 1, explicit_idx)))
    curv = np.asarray(f["vehicle/ground_truth_path_curvature"][:], dtype=np.float64) if "vehicle/ground_truth_path_curvature" in f else np.zeros(cam_n)
    has_screen = np.asarray(
        [len(np.asarray(x).reshape(-1)) >= 4 for x in f["vehicle/right_lane_fiducials_screen_xy"][:]],
        dtype=bool,
    ) if "vehicle/right_lane_fiducials_screen_xy" in f else np.zeros(cam_n, dtype=bool)
    mask = has_screen
    if np.any(mask):
        idxs = np.where(mask)[0]
        curv_abs = np.abs(curv[idxs])
        if np.nanmax(curv_abs) > 1e-6:
            pick = idxs[int(np.nanargmax(curv_abs))]
            return int(pick)
        # If curvature telemetry is flat/zero, pick a stable non-startup frame.
        return int(idxs[int(0.7 * (len(idxs) - 1))])
    return int(min(cam_n - 1, max(0, cam_n // 2)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-frame fiducial reprojection sanity check.")
    parser.add_argument("recording", nargs="?", help="Path to recording .h5")
    parser.add_argument("--latest", action="store_true", help="Use latest recording")
    parser.add_argument("--frame-index", type=int, default=None, help="Frame index to analyze")
    parser.add_argument("--nearfield-enabled", action="store_true", help="Enable near-field ground y blend")
    parser.add_argument("--nearfield-offset-m", type=float, default=-0.3)
    parser.add_argument("--nearfield-blend-m", type=float, default=10.0)
    parser.add_argument("--output-json", type=str, default="", help="Optional output path")
    args = parser.parse_args()

    if args.latest:
        rec = find_latest_recording()
    elif args.recording:
        rec = Path(args.recording)
    else:
        rec = find_latest_recording()
    if rec is None or not rec.exists():
        raise SystemExit("No recording found.")

    with h5py.File(rec, "r") as f:
        idx = choose_frame(f, args.frame_index)
        width = float(f["camera/image_width"][0]) if "camera/image_width" in f else 640.0
        height = float(f["camera/image_height"][0]) if "camera/image_height" in f else 480.0
        spacing = float(f["vehicle/right_lane_fiducials_spacing_meters"][idx]) if "vehicle/right_lane_fiducials_spacing_meters" in f else 5.0

        q = np.asarray(f["vehicle/rotation"][idx], dtype=np.float64)
        veh_pos = np.asarray(f["vehicle/position"][idx], dtype=np.float64)
        cam_pos = np.array(
            [
                float(f["vehicle/camera_pos_x"][idx]) if "vehicle/camera_pos_x" in f else 0.0,
                float(f["vehicle/camera_pos_y"][idx]) if "vehicle/camera_pos_y" in f else 0.0,
                float(f["vehicle/camera_pos_z"][idx]) if "vehicle/camera_pos_z" in f else 0.0,
            ],
            dtype=np.float64,
        )
        cam_fwd = np.array(
            [
                float(f["vehicle/camera_forward_x"][idx]) if "vehicle/camera_forward_x" in f else 0.0,
                float(f["vehicle/camera_forward_y"][idx]) if "vehicle/camera_forward_y" in f else -1.0,
                float(f["vehicle/camera_forward_z"][idx]) if "vehicle/camera_forward_z" in f else 0.0,
            ],
            dtype=np.float64,
        )
        hfov = float(f["vehicle/camera_horizontal_fov"][idx]) if "vehicle/camera_horizontal_fov" in f else 110.0
        vfov = float(f["vehicle/camera_field_of_view"][idx]) if "vehicle/camera_field_of_view" in f else (
            2.0 * math.degrees(math.atan(math.tan(math.radians(hfov) / 2.0) * (height / width)))
        )
        ground_y = float(f["vehicle/road_center_at_car_y"][idx]) if "vehicle/road_center_at_car_y" in f else float(veh_pos[1])

        truth = parse_xy(f["vehicle/right_lane_fiducials_screen_xy"][idx]) if "vehicle/right_lane_fiducials_screen_xy" in f else np.empty((0, 2))
        src_world = parse_xyz(f["vehicle/right_lane_fiducials_world_xyz"][idx]) if "vehicle/right_lane_fiducials_world_xyz" in f else np.empty((0, 3))
        src_true = parse_xy(f["vehicle/right_lane_fiducials_vehicle_true_xy"][idx]) if "vehicle/right_lane_fiducials_vehicle_true_xy" in f else np.empty((0, 2))
        src_legacy = parse_xy(f["vehicle/right_lane_fiducials_vehicle_xy"][idx]) if "vehicle/right_lane_fiducials_vehicle_xy" in f else np.empty((0, 2))
        src_mono = parse_xy(f["vehicle/right_lane_fiducials_vehicle_monotonic_xy"][idx]) if "vehicle/right_lane_fiducials_vehicle_monotonic_xy" in f else np.empty((0, 2))

        result = {
            "recording": str(rec),
            "frame_index": int(idx),
            "nearfield_enabled": bool(args.nearfield_enabled),
            "nearfield_offset_m": float(args.nearfield_offset_m),
            "nearfield_blend_m": float(args.nearfield_blend_m),
            "geometry_snapshot": {
                "image_width": width,
                "image_height": height,
                "camera_pos": cam_pos.tolist(),
                "vehicle_pos": veh_pos.tolist(),
                "camera_fwd": cam_fwd.tolist(),
                "camera_hfov_deg": hfov,
                "camera_vfov_deg": vfov,
                "ground_y": ground_y,
                "spacing_m": spacing,
            },
            "availability": {
                "screen_truth_points": int(len(truth)),
                "world_points": int(len(src_world)),
                "vehicle_true_points": int(len(src_true)),
                "vehicle_legacy_points": int(len(src_legacy)),
                "vehicle_monotonic_points": int(len(src_mono)),
            },
            "sources": {},
        }

        if len(src_world) > 0:
            proj_world = project_world_points(
                src_world,
                cam_pos=cam_pos,
                cam_fwd=cam_fwd,
                width=width,
                height=height,
                hfov_deg=hfov,
                vfov_deg=vfov,
            )
            stats_world = summarize_errors(proj_world, truth, spacing)
            stats_world["available"] = True
            result["sources"]["world_xyz"] = stats_world
        else:
            result["sources"]["world_xyz"] = {"available": False}

        for name, src in [
            ("vehicle_true_xy", src_true),
            ("vehicle_xy_legacy", src_legacy),
            ("vehicle_monotonic_xy", src_mono),
        ]:
            if len(src) == 0:
                result["sources"][name] = {"available": False}
                continue
            proj = project_vehicle_points(
                src,
                q=q,
                veh_pos=veh_pos,
                cam_pos=cam_pos,
                cam_fwd=cam_fwd,
                ground_y=ground_y,
                width=width,
                height=height,
                hfov_deg=hfov,
                vfov_deg=vfov,
                nearfield_enabled=bool(args.nearfield_enabled),
                nearfield_offset_m=float(args.nearfield_offset_m),
                nearfield_blend_m=float(args.nearfield_blend_m),
            )
            stats = summarize_errors(proj, truth, spacing)
            stats["available"] = True
            result["sources"][name] = stats

    out_path = Path(args.output_json) if args.output_json else (
        Path("tmp/analysis") / f"fiducial_reprojection_static_{rec.stem}_f{result['frame_index']}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

