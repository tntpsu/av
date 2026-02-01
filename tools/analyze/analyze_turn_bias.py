#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def _load_center(f: h5py.File) -> np.ndarray | None:
    if "perception/left_lane_line_x" in f and "perception/right_lane_line_x" in f:
        left = np.array(f["perception/left_lane_line_x"][:])
        right = np.array(f["perception/right_lane_line_x"][:])
        return (left + right) / 2.0
    if "perception/left_lane_x" in f and "perception/right_lane_x" in f:
        left = np.array(f["perception/left_lane_x"][:])
        right = np.array(f["perception/right_lane_x"][:])
        return (left + right) / 2.0
    return None


def _load_curvature(f: h5py.File) -> tuple[np.ndarray, str]:
    if "control/path_curvature_input" in f:
        return np.array(f["control/path_curvature_input"][:]), "control/path_curvature_input"
    if "ground_truth/path_curvature" in f:
        return np.array(f["ground_truth/path_curvature"][:]), "ground_truth/path_curvature"
    return np.zeros(0), "missing"


def _safe_array(f: h5py.File, key: str) -> np.ndarray:
    return np.array(f[key][:]) if key in f else np.zeros(0)


def _format_row(
    idx: int,
    t: float | None,
    offset: float,
    curvature: float,
    steering: float | None,
    heading_delta: float | None,
    gt_center: float | None,
    p_center: float | None,
    lane_center_error: float | None,
    vehicle_frame_offset: float | None,
) -> str:
    time_str = f"{t:7.2f}s" if t is not None else "   n/a "
    steer_str = f"{steering:7.3f}" if steering is not None else "   n/a "
    hdg_str = f"{heading_delta:7.2f}" if heading_delta is not None else "   n/a "
    gt_str = f"{gt_center:7.3f}" if gt_center is not None else "   n/a "
    pc_str = f"{p_center:7.3f}" if p_center is not None else "   n/a "
    lane_err_str = f"{lane_center_error:7.3f}" if lane_center_error is not None else "   n/a "
    vf_str = f"{vehicle_frame_offset:7.3f}" if vehicle_frame_offset is not None else "   n/a "
    return (
        f"frame={idx:4d} t={time_str} "
        f"offset={offset:7.3f}m curv={curvature:8.5f} "
        f"steer={steer_str} hdg_delta={hdg_str} "
        f"gt_center={gt_str} p_center={pc_str} lane_err={lane_err_str} vf_off={vf_str}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze turn bias using road-frame offset.")
    parser.add_argument("recording", type=Path, help="Path to HDF5 recording")
    parser.add_argument("--top-n", type=int, default=10, help="Top N frames per turn direction")
    parser.add_argument("--curv-threshold", type=float, default=0.002, help="Curvature threshold")
    args = parser.parse_args()

    with h5py.File(args.recording, "r") as f:
        offset = _safe_array(f, "vehicle/road_frame_lateral_offset")
        heading_delta = _safe_array(f, "vehicle/heading_delta_deg")
        steering = _safe_array(f, "control/steering")
        timestamps = _safe_array(f, "vehicle/timestamps")
        gt_center = _safe_array(f, "ground_truth/lane_center_x")
        p_center = _load_center(f)
        curvature, curv_name = _load_curvature(f)
        lane_center_error = _safe_array(f, "vehicle/road_frame_lane_center_error")
        vehicle_frame_offset = _safe_array(f, "vehicle/vehicle_frame_lookahead_offset")

        n = min(
            len(offset),
            len(curvature),
            len(heading_delta),
            len(steering),
            len(timestamps),
        )
        if lane_center_error.size:
            n = min(n, len(lane_center_error))
        if vehicle_frame_offset.size:
            n = min(n, len(vehicle_frame_offset))
        if n == 0:
            print("No usable data in recording.")
            return
        offset = offset[:n]
        curvature = curvature[:n]
        heading_delta = heading_delta[:n]
        steering = steering[:n]
        timestamps = timestamps[:n]
        if lane_center_error.size:
            lane_center_error = lane_center_error[:n]
        if vehicle_frame_offset.size:
            vehicle_frame_offset = vehicle_frame_offset[:n]
        if gt_center.size:
            gt_center = gt_center[:n]
        if p_center is not None and p_center.size:
            p_center = p_center[:n]

        curve_mask = np.abs(curvature) >= args.curv_threshold
        left_turn = curve_mask & (curvature > 0)
        right_turn = curve_mask & (curvature < 0)

        def top_indices(mask: np.ndarray) -> np.ndarray:
            if not np.any(mask):
                return np.array([], dtype=int)
            candidates = np.where(mask)[0]
            sorted_idx = candidates[np.argsort(-np.abs(offset[candidates]))]
            return sorted_idx[: args.top_n]

        left_idx = top_indices(left_turn)
        right_idx = top_indices(right_turn)

        print(f"Recording: {args.recording.name}")
        print(f"Curvature source: {curv_name}")
        print(f"Curvature threshold: {args.curv_threshold}")
        print(f"Curve frames: {curve_mask.sum()} / {n}")

        print("\nTop frames by |road_frame_lateral_offset| (LEFT turns):")
        if left_idx.size == 0:
            print("  none")
        else:
            for idx in left_idx:
                t = float(timestamps[idx]) if timestamps.size else None
                gt = float(gt_center[idx]) if gt_center.size else None
                pc = float(p_center[idx]) if p_center is not None and p_center.size else None
                print(
                    "  "
                    + _format_row(
                        idx,
                        t,
                        float(offset[idx]),
                        float(curvature[idx]),
                        float(steering[idx]) if steering.size else None,
                        float(heading_delta[idx]) if heading_delta.size else None,
                        gt,
                        pc,
                        float(lane_center_error[idx]) if lane_center_error.size else None,
                        float(vehicle_frame_offset[idx]) if vehicle_frame_offset.size else None,
                    )
                )

        print("\nTop frames by |road_frame_lateral_offset| (RIGHT turns):")
        if right_idx.size == 0:
            print("  none")
        else:
            for idx in right_idx:
                t = float(timestamps[idx]) if timestamps.size else None
                gt = float(gt_center[idx]) if gt_center.size else None
                pc = float(p_center[idx]) if p_center is not None and p_center.size else None
                print(
                    "  "
                    + _format_row(
                        idx,
                        t,
                        float(offset[idx]),
                        float(curvature[idx]),
                        float(steering[idx]) if steering.size else None,
                        float(heading_delta[idx]) if heading_delta.size else None,
                        gt,
                        pc,
                        float(lane_center_error[idx]) if lane_center_error.size else None,
                        float(vehicle_frame_offset[idx]) if vehicle_frame_offset.size else None,
                    )
                )


if __name__ == "__main__":
    main()
