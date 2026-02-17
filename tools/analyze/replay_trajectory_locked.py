#!/usr/bin/env python3
"""
Replay control with trajectory reference point locked from a source recording.

This is a Stage-2 isolation harness:
- Replays frames from an input recording through AVStack
- Overrides trajectory reference point every frame with a locked source recording
- Runs current control stack against the locked trajectory reference
- Saves a new HDF5 recording plus a JSON sidecar summary

The goal is to isolate controller behavior from live trajectory variation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from av_stack import AVStack  # noqa: E402

DEFAULT_SEGMENTATION_CHECKPOINT = (
    REPO_ROOT / "data" / "segmentation_dataset" / "checkpoints" / "segnet_best.pt"
)


def _pick_latest_recording() -> Path:
    recordings = sorted(
        (REPO_ROOT / "data" / "recordings").glob("*.h5"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not recordings:
        raise FileNotFoundError("No recordings found in data/recordings")
    return recordings[0]


def _decode_bytes(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore").strip()
    return str(value).strip()


def _get_len(f: h5py.File, key: str) -> int:
    return int(len(f[key])) if key in f else 0


def _dataset_row_to_list(source_file: h5py.File, key: str, idx: int) -> list[float]:
    if key not in source_file or idx >= len(source_file[key]):
        return []
    row = np.asarray(source_file[key][idx]).reshape(-1)
    if row.size == 0:
        return []
    return row.astype(np.float32).tolist()


def _inject_oracle_and_fiducials(source_file: h5py.File, idx: int, d: Dict[str, object]) -> None:
    # Oracle overlays and reference telemetry.
    oracle_xy = _dataset_row_to_list(source_file, "trajectory/oracle_points", idx)
    if oracle_xy:
        d["oracleTrajectoryXY"] = oracle_xy
        d["oracle_trajectory_xy"] = oracle_xy
    if "trajectory/oracle_point_count" in source_file and idx < len(source_file["trajectory/oracle_point_count"]):
        d["oraclePointCount"] = int(source_file["trajectory/oracle_point_count"][idx])
        d["oracle_point_count"] = int(source_file["trajectory/oracle_point_count"][idx])
    if "trajectory/oracle_horizon_meters" in source_file and idx < len(source_file["trajectory/oracle_horizon_meters"]):
        d["oracleHorizonMeters"] = float(source_file["trajectory/oracle_horizon_meters"][idx])
        d["oracle_horizon_meters"] = float(source_file["trajectory/oracle_horizon_meters"][idx])
    if "trajectory/oracle_point_spacing_meters" in source_file and idx < len(source_file["trajectory/oracle_point_spacing_meters"]):
        d["oraclePointSpacingMeters"] = float(source_file["trajectory/oracle_point_spacing_meters"][idx])
        d["oracle_point_spacing_meters"] = float(source_file["trajectory/oracle_point_spacing_meters"][idx])
    if "trajectory/oracle_samples_enabled" in source_file and idx < len(source_file["trajectory/oracle_samples_enabled"]):
        enabled = bool(source_file["trajectory/oracle_samples_enabled"][idx])
        d["oracleSamplesEnabled"] = enabled
        d["oracle_samples_enabled"] = enabled

    oracle_world = _dataset_row_to_list(source_file, "vehicle/oracle_trajectory_world_xyz", idx)
    if oracle_world:
        d["oracleTrajectoryWorldXYZ"] = oracle_world
        d["oracle_trajectory_world_xyz"] = oracle_world
    oracle_screen = _dataset_row_to_list(source_file, "vehicle/oracle_trajectory_screen_xy", idx)
    if oracle_screen:
        d["oracleTrajectoryScreenXY"] = oracle_screen
        d["oracle_trajectory_screen_xy"] = oracle_screen

    # Right-lane fiducials used by PhilViz overlays and distance-scale anchors.
    for key in (
        "vehicle/right_lane_fiducials_vehicle_true_xy",
        "vehicle/right_lane_fiducials_vehicle_monotonic_xy",
        "vehicle/right_lane_fiducials_world_xyz",
        "vehicle/right_lane_fiducials_vehicle_xy",
        "vehicle/right_lane_fiducials_screen_xy",
    ):
        vals = _dataset_row_to_list(source_file, key, idx)
        if vals:
            d[key.split("/", 1)[1]] = vals
    for key, caster in (
        ("vehicle/right_lane_fiducials_point_count", int),
        ("vehicle/right_lane_fiducials_horizon_meters", float),
        ("vehicle/right_lane_fiducials_spacing_meters", float),
    ):
        if key in source_file and idx < len(source_file[key]):
            d[key.split("/", 1)[1]] = caster(source_file[key][idx])
    if "vehicle/right_lane_fiducials_enabled" in source_file and idx < len(source_file["vehicle/right_lane_fiducials_enabled"]):
        d["right_lane_fiducials_enabled"] = bool(source_file["vehicle/right_lane_fiducials_enabled"][idx])


def _read_scalar_float(lock_file: h5py.File, key: str, idx: int, default: float) -> float:
    if key not in lock_file or idx >= len(lock_file[key]):
        return float(default)
    try:
        v = float(lock_file[key][idx])
    except Exception:
        return float(default)
    return v if np.isfinite(v) else float(default)


def _load_locked_reference(lock_file: h5py.File, idx: int) -> Optional[Dict[str, float]]:
    if "trajectory/reference_point_x" not in lock_file:
        return None
    if idx >= len(lock_file["trajectory/reference_point_x"]):
        return None

    x = float(lock_file["trajectory/reference_point_x"][idx])
    y = float(lock_file["trajectory/reference_point_y"][idx])
    heading = float(lock_file["trajectory/reference_point_heading"][idx])
    velocity = float(lock_file["trajectory/reference_point_velocity"][idx])
    method = ""
    if "trajectory/reference_point_method" in lock_file:
        method = _decode_bytes(lock_file["trajectory/reference_point_method"][idx])
    curvature = 0.0
    if "trajectory/path_curvature" in lock_file and idx < len(lock_file["trajectory/path_curvature"]):
        curvature = float(lock_file["trajectory/path_curvature"][idx])
    perception_center_x = 0.0
    if "trajectory/perception_center_x" in lock_file:
        perception_center_x = float(lock_file["trajectory/perception_center_x"][idx])

    ref = {
        "x": x,
        "y": y,
        "heading": heading,
        "velocity": velocity,
        "method": f"locked:{method}" if method else "locked",
        "curvature": curvature,
        "perception_center_x": perception_center_x,
    }
    # Keep replay diagnostics finite for downstream JSON/visualizer compatibility.
    for key, default in (
        ("diag_smoothing_jump_reject", 0.0),
        ("diag_small_heading_gate_active", 0.0),
        ("diag_heading_zero_gate_active", 0.0),
        ("diag_ref_x_rate_limit_active", 0.0),
        ("diag_multi_lookahead_active", 0.0),
        ("diag_raw_ref_x", x),
        ("diag_raw_ref_heading", heading),
        ("diag_smoothed_ref_x", x),
        ("diag_smoothed_ref_heading", heading),
        ("diag_smoothing_alpha", 0.0),
        ("diag_smoothing_alpha_x", 0.0),
        ("diag_ref_x_suppression_abs", 0.0),
        ("diag_heading_suppression_abs", 0.0),
        ("diag_multi_lookahead_heading_base", heading),
        ("diag_multi_lookahead_heading_far", heading),
        ("diag_multi_lookahead_heading_blended", heading),
        ("diag_multi_lookahead_blend_alpha", 0.0),
        ("diag_dynamic_effective_horizon_m", 0.0),
        ("diag_dynamic_effective_horizon_base_m", 0.0),
        ("diag_dynamic_effective_horizon_min_m", 0.0),
        ("diag_dynamic_effective_horizon_max_m", 0.0),
        ("diag_dynamic_effective_horizon_speed_scale", 1.0),
        ("diag_dynamic_effective_horizon_curvature_scale", 1.0),
        ("diag_dynamic_effective_horizon_confidence_scale", 1.0),
        ("diag_dynamic_effective_horizon_final_scale", 1.0),
        ("diag_dynamic_effective_horizon_speed_mps", 0.0),
        ("diag_dynamic_effective_horizon_curvature_abs", 0.0),
        ("diag_dynamic_effective_horizon_confidence_used", 1.0),
        ("diag_dynamic_effective_horizon_limiter_code", 0.0),
        ("diag_dynamic_effective_horizon_applied", 0.0),
    ):
        ref[key] = _read_scalar_float(lock_file, f"trajectory/{key}", idx, default)
    return ref


def _build_vehicle_state_dict(source_file: h5py.File, idx: int) -> Dict[str, object]:
    d: Dict[str, object] = {
        "position": {
            "x": float(source_file["vehicle/position"][idx, 0]),
            "y": float(source_file["vehicle/position"][idx, 1]),
            "z": float(source_file["vehicle/position"][idx, 2]),
        },
        "rotation": {
            "x": float(source_file["vehicle/rotation"][idx, 0]) if "vehicle/rotation" in source_file else 0.0,
            "y": float(source_file["vehicle/rotation"][idx, 1]) if "vehicle/rotation" in source_file else 0.0,
            "z": float(source_file["vehicle/rotation"][idx, 2]) if "vehicle/rotation" in source_file else 0.0,
            "w": float(source_file["vehicle/rotation"][idx, 3]) if "vehicle/rotation" in source_file else 1.0,
        },
        "velocity": {
            "x": float(source_file["vehicle/velocity"][idx, 0]) if "vehicle/velocity" in source_file else 0.0,
            "y": float(source_file["vehicle/velocity"][idx, 1]) if "vehicle/velocity" in source_file else 0.0,
            "z": float(source_file["vehicle/velocity"][idx, 2]) if "vehicle/velocity" in source_file else 0.0,
        },
        "angularVelocity": {
            "x": float(source_file["vehicle/angular_velocity"][idx, 0]) if "vehicle/angular_velocity" in source_file else 0.0,
            "y": float(source_file["vehicle/angular_velocity"][idx, 1]) if "vehicle/angular_velocity" in source_file else 0.0,
            "z": float(source_file["vehicle/angular_velocity"][idx, 2]) if "vehicle/angular_velocity" in source_file else 0.0,
        },
        "speed": float(source_file["vehicle/speed"][idx]),
        "steeringAngle": float(source_file["vehicle/steering_angle"][idx]),
        "motorTorque": float(source_file["vehicle/motor_torque"][idx]) if "vehicle/motor_torque" in source_file else 0.0,
        "brakeTorque": float(source_file["vehicle/brake_torque"][idx]) if "vehicle/brake_torque" in source_file else 0.0,
    }

    def _copy_vehicle_scalar(dataset_name: str, out_key: str, caster=float) -> None:
        full_key = f"vehicle/{dataset_name}"
        if full_key in source_file and idx < len(source_file[full_key]):
            try:
                d[out_key] = caster(source_file[full_key][idx])
            except (TypeError, ValueError):
                return

    # Preserve front-camera mounting/projection context for replayed frames.
    # Without these fields, main-camera trajectory projection falls back and can appear unmounted.
    _copy_vehicle_scalar("camera_pos_x", "cameraPosX")
    _copy_vehicle_scalar("camera_pos_y", "cameraPosY")
    _copy_vehicle_scalar("camera_pos_z", "cameraPosZ")
    _copy_vehicle_scalar("camera_forward_x", "cameraForwardX")
    _copy_vehicle_scalar("camera_forward_y", "cameraForwardY")
    _copy_vehicle_scalar("camera_forward_z", "cameraForwardZ")
    _copy_vehicle_scalar("camera_field_of_view", "cameraFieldOfView")
    _copy_vehicle_scalar("camera_horizontal_fov", "cameraHorizontalFOV")
    _copy_vehicle_scalar("camera_8m_screen_y", "camera8mScreenY")
    _copy_vehicle_scalar("camera_lookahead_screen_y", "cameraLookaheadScreenY")
    _copy_vehicle_scalar("ground_truth_lookahead_distance", "groundTruthLookaheadDistance")
    _copy_vehicle_scalar("road_center_at_car_x", "roadCenterAtCarX")
    _copy_vehicle_scalar("road_center_at_car_y", "roadCenterAtCarY")
    _copy_vehicle_scalar("road_center_at_car_z", "roadCenterAtCarZ")
    _copy_vehicle_scalar("topdown_camera_pos_x", "topDownCameraPosX")
    _copy_vehicle_scalar("topdown_camera_pos_y", "topDownCameraPosY")
    _copy_vehicle_scalar("topdown_camera_pos_z", "topDownCameraPosZ")
    _copy_vehicle_scalar("topdown_camera_forward_x", "topDownCameraForwardX")
    _copy_vehicle_scalar("topdown_camera_forward_y", "topDownCameraForwardY")
    _copy_vehicle_scalar("topdown_camera_forward_z", "topDownCameraForwardZ")
    _copy_vehicle_scalar("topdown_camera_orthographic_size", "topDownCameraOrthographicSize")
    _copy_vehicle_scalar("topdown_camera_field_of_view", "topDownCameraFieldOfView")

    # Ground truth names vary between older/newer recordings.
    if "ground_truth/left_lane_line_x" in source_file:
        d["groundTruthLeftLaneLineX"] = float(source_file["ground_truth/left_lane_line_x"][idx])
    elif "ground_truth/left_lane_x" in source_file:
        d["groundTruthLeftLaneLineX"] = float(source_file["ground_truth/left_lane_x"][idx])
    if "ground_truth/right_lane_line_x" in source_file:
        d["groundTruthRightLaneLineX"] = float(source_file["ground_truth/right_lane_line_x"][idx])
    elif "ground_truth/right_lane_x" in source_file:
        d["groundTruthRightLaneLineX"] = float(source_file["ground_truth/right_lane_x"][idx])
    if "ground_truth/lane_center_x" in source_file:
        d["groundTruthLaneCenterX"] = float(source_file["ground_truth/lane_center_x"][idx])
    if "ground_truth/path_curvature" in source_file:
        d["groundTruthPathCurvature"] = float(source_file["ground_truth/path_curvature"][idx])
    if "ground_truth/desired_heading" in source_file:
        d["groundTruthDesiredHeading"] = float(source_file["ground_truth/desired_heading"][idx])
    if "vehicle/road_heading_deg" in source_file:
        d["roadHeadingDeg"] = float(source_file["vehicle/road_heading_deg"][idx])
    if "vehicle/vehicle_frame_lookahead_offset" in source_file:
        d["vehicleFrameLookaheadOffset"] = float(source_file["vehicle/vehicle_frame_lookahead_offset"][idx])

    _inject_oracle_and_fiducials(source_file, idx, d)
    return d


def replay_trajectory_locked(
    input_recording: Path,
    lock_recording: Path,
    output_name: Optional[str],
    use_segmentation: bool,
    segmentation_checkpoint: Optional[str],
    disable_vehicle_frame_lookahead_ref: bool,
    lock_latency_ms: float = 0.0,
    lock_latency_frames: int = 0,
    noise_seed: int = 1337,
    noise_std_x: float = 0.0,
    noise_std_heading: float = 0.0,
    noise_std_velocity: float = 0.0,
    noise_std_curvature: float = 0.0,
) -> Dict[str, object]:
    if output_name is None:
        output_name = f"trajectory_locked_{input_recording.stem}"

    with h5py.File(input_recording, "r") as in_f, h5py.File(lock_recording, "r") as lock_f:
        n_input = _get_len(in_f, "camera/images")
        n_lock = _get_len(lock_f, "trajectory/reference_point_x")
        if n_input == 0:
            raise RuntimeError(f"No camera frames found in {input_recording}")
        if n_lock == 0:
            raise RuntimeError(f"No trajectory reference datasets found in {lock_recording}")

        n_frames = min(n_input, n_lock)
        print(f"Input frames: {n_input}, lock frames: {n_lock}, using: {n_frames}")
        ts = np.asarray(in_f["camera/timestamps"][:n_frames], dtype=float)
        dt = float(np.median(np.diff(ts))) if len(ts) > 1 else (1.0 / 30.0)
        fps = 1.0 / max(dt, 1e-3)
        latency_frames_from_ms = int(round((max(0.0, float(lock_latency_ms)) / 1000.0) * fps))
        total_latency_frames = max(0, int(lock_latency_frames) + latency_frames_from_ms)
        rng = np.random.default_rng(int(noise_seed))

        av_stack = AVStack(
            bridge_url="http://localhost:8000",
            record_data=True,
            recording_dir=str(REPO_ROOT / "data" / "recordings"),
            use_segmentation=use_segmentation,
            segmentation_model_path=segmentation_checkpoint,
        )

        # Ensure lock is not immediately overridden by vehicle-frame lookahead.
        if disable_vehicle_frame_lookahead_ref:
            av_stack.trajectory_config["use_vehicle_frame_lookahead_ref"] = False

        av_stack.recorder.recording_name = output_name
        av_stack.recorder.recording_type = "trajectory_locked_replay"
        av_stack.recorder.metadata["recording_type"] = "trajectory_locked_replay"
        av_stack.recorder.metadata["source_recording"] = input_recording.name
        av_stack.recorder.metadata["lock_source_recording"] = lock_recording.name
        av_stack.recorder.metadata["disable_vehicle_frame_lookahead_ref"] = bool(
            disable_vehicle_frame_lookahead_ref
        )
        av_stack.recorder.metadata["lock_latency_ms"] = float(lock_latency_ms)
        av_stack.recorder.metadata["lock_latency_frames"] = int(lock_latency_frames)
        av_stack.recorder.metadata["lock_total_latency_frames"] = int(total_latency_frames)
        av_stack.recorder.metadata["lock_noise_seed"] = int(noise_seed)
        av_stack.recorder.metadata["lock_noise_std_x"] = float(noise_std_x)
        av_stack.recorder.metadata["lock_noise_std_heading"] = float(noise_std_heading)
        av_stack.recorder.metadata["lock_noise_std_velocity"] = float(noise_std_velocity)
        av_stack.recorder.metadata["lock_noise_std_curvature"] = float(noise_std_curvature)
        av_stack.recorder.metadata["perception_mode"] = (
            "segmentation" if use_segmentation else "cv_or_ml"
        )

        original_get_reference_point = av_stack.trajectory_planner.get_reference_point
        original_send = None
        original_get_camera_frame = None
        original_get_camera_frame_with_metadata = None
        if hasattr(av_stack.bridge, "set_control_command"):
            original_send = av_stack.bridge.set_control_command
            av_stack.bridge.set_control_command = lambda *args, **kwargs: True
        if hasattr(av_stack.bridge, "get_latest_camera_frame"):
            original_get_camera_frame = av_stack.bridge.get_latest_camera_frame
        if hasattr(av_stack.bridge, "get_latest_camera_frame_with_metadata"):
            original_get_camera_frame_with_metadata = (
                av_stack.bridge.get_latest_camera_frame_with_metadata
            )

        has_topdown_input = "camera/topdown_images" in in_f and len(in_f["camera/topdown_images"]) > 0
        av_stack.recorder.metadata["topdown_replayed_from_input"] = bool(has_topdown_input)
        frame_holder = {"idx": 0}
        if has_topdown_input and (
            original_get_camera_frame is not None
            or original_get_camera_frame_with_metadata is not None
        ):
            topdown_images = in_f["camera/topdown_images"]
            if "camera/topdown_timestamps" in in_f and len(in_f["camera/topdown_timestamps"]) > 0:
                topdown_timestamps = in_f["camera/topdown_timestamps"]
            else:
                # Fallback for older recordings that don't store dedicated top-down timestamps.
                topdown_timestamps = in_f["camera/timestamps"]

            def _replay_get_latest_camera_frame(*args, **kwargs):
                camera_id = kwargs.get("camera_id")
                if camera_id is None and len(args) >= 1:
                    camera_id = args[0]
                if camera_id in ("top_down", "topdown"):
                    i = min(frame_holder["idx"], len(topdown_images) - 1)
                    return topdown_images[i], float(topdown_timestamps[i]), None
                if original_get_camera_frame is None:
                    return None
                return original_get_camera_frame(*args, **kwargs)

            def _replay_get_latest_camera_frame_with_metadata(*args, **kwargs):
                camera_id = kwargs.get("camera_id")
                if camera_id is None and len(args) >= 1:
                    camera_id = args[0]
                if camera_id in ("top_down", "topdown"):
                    i = min(frame_holder["idx"], len(topdown_images) - 1)
                    topdown_ts = float(topdown_timestamps[i])
                    # Keep metadata minimal but consistent with bridge contract.
                    meta = {
                        "queue_depth": 0.0,
                        "queue_capacity": 0.0,
                        "drop_count": 0.0,
                        "decode_in_flight": False,
                        "latest_age_ms": 0.0,
                        "last_arrival_time": None,
                        "last_realtime_since_startup": topdown_ts,
                        "last_unscaled_time": None,
                    }
                    return topdown_images[i], topdown_ts, None, meta
                if original_get_camera_frame_with_metadata is None:
                    return None
                return original_get_camera_frame_with_metadata(*args, **kwargs)

            av_stack.bridge.get_latest_camera_frame = _replay_get_latest_camera_frame
            if original_get_camera_frame_with_metadata is not None:
                av_stack.bridge.get_latest_camera_frame_with_metadata = (
                    _replay_get_latest_camera_frame_with_metadata
                )

        missing_locked_ref_count = 0
        latency_clamped_frames = 0
        control_vs_source_steering_abs_diff: list[float] = []
        control_vs_source_throttle_abs_diff: list[float] = []
        control_vs_source_brake_abs_diff: list[float] = []

        try:
            for idx in range(n_frames):
                frame_holder["idx"] = idx
                image = in_f["camera/images"][idx]
                timestamp = float(in_f["camera/timestamps"][idx])
                vehicle_state_dict = _build_vehicle_state_dict(in_f, idx)
                lock_idx = idx - total_latency_frames
                if lock_idx < 0:
                    latency_clamped_frames += 1
                    lock_idx = 0
                locked_ref = _load_locked_reference(lock_f, lock_idx)

                if locked_ref is None:
                    missing_locked_ref_count += 1

                    def _fallback_get_reference_point(*args, **kwargs):
                        return original_get_reference_point(*args, **kwargs)

                    av_stack.trajectory_planner.get_reference_point = _fallback_get_reference_point
                else:
                    # Deterministic perturbations for stress testing.
                    if noise_std_x > 0.0:
                        locked_ref["x"] = float(locked_ref["x"] + rng.normal(0.0, noise_std_x))
                    if noise_std_heading > 0.0:
                        locked_ref["heading"] = float(
                            locked_ref["heading"] + rng.normal(0.0, noise_std_heading)
                        )
                    if noise_std_velocity > 0.0:
                        locked_ref["velocity"] = float(
                            max(0.0, locked_ref["velocity"] + rng.normal(0.0, noise_std_velocity))
                        )
                    if noise_std_curvature > 0.0:
                        locked_ref["curvature"] = float(
                            locked_ref["curvature"] + rng.normal(0.0, noise_std_curvature)
                        )

                    def _locked_get_reference_point(*args, **kwargs):
                        return dict(locked_ref)

                    av_stack.trajectory_planner.get_reference_point = _locked_get_reference_point

                av_stack._process_frame(image, timestamp, vehicle_state_dict)

                if "control/steering" in in_f and idx < len(in_f["control/steering"]):
                    source_steering = float(in_f["control/steering"][idx])
                    replay_steering = float(av_stack.last_steering_command or 0.0)
                    control_vs_source_steering_abs_diff.append(abs(replay_steering - source_steering))
                if "control/throttle" in in_f and idx < len(in_f["control/throttle"]):
                    source_throttle = float(in_f["control/throttle"][idx])
                    replay_throttle = float(getattr(av_stack, "last_throttle_command", 0.0) or 0.0)
                    control_vs_source_throttle_abs_diff.append(abs(replay_throttle - source_throttle))
                if "control/brake" in in_f and idx < len(in_f["control/brake"]):
                    source_brake = float(in_f["control/brake"][idx])
                    replay_brake = float(getattr(av_stack, "last_brake_command", 0.0) or 0.0)
                    control_vs_source_brake_abs_diff.append(abs(replay_brake - source_brake))

                if idx % 30 == 0:
                    print(f"Processed {idx}/{n_frames} frames...")
        finally:
            av_stack.trajectory_planner.get_reference_point = original_get_reference_point
            if original_send is not None:
                av_stack.bridge.set_control_command = original_send
            if original_get_camera_frame is not None:
                av_stack.bridge.get_latest_camera_frame = original_get_camera_frame
            if original_get_camera_frame_with_metadata is not None:
                av_stack.bridge.get_latest_camera_frame_with_metadata = (
                    original_get_camera_frame_with_metadata
                )
            av_stack.recorder.close()

    output_path = Path(av_stack.recorder.output_file)
    summary = {
        "input_recording": str(input_recording),
        "lock_recording": str(lock_recording),
        "output_recording": str(output_path),
        "frames_processed": int(n_frames),
        "missing_locked_reference_frames": int(missing_locked_ref_count),
        "latency": {
            "lock_latency_ms": float(lock_latency_ms),
            "lock_latency_frames_arg": int(lock_latency_frames),
            "derived_fps": float(fps),
            "derived_dt_s": float(dt),
            "latency_frames_from_ms": int(latency_frames_from_ms),
            "total_latency_frames": int(total_latency_frames),
            "latency_clamped_frames": int(latency_clamped_frames),
        },
        "noise": {
            "seed": int(noise_seed),
            "std_x": float(noise_std_x),
            "std_heading": float(noise_std_heading),
            "std_velocity": float(noise_std_velocity),
            "std_curvature": float(noise_std_curvature),
        },
        "steering_abs_diff_mean_vs_source": float(np.mean(control_vs_source_steering_abs_diff))
        if control_vs_source_steering_abs_diff
        else None,
        "steering_abs_diff_p95_vs_source": float(np.quantile(control_vs_source_steering_abs_diff, 0.95))
        if control_vs_source_steering_abs_diff
        else None,
        "throttle_abs_diff_mean_vs_source": float(np.mean(control_vs_source_throttle_abs_diff))
        if control_vs_source_throttle_abs_diff
        else None,
        "throttle_abs_diff_p95_vs_source": float(np.quantile(control_vs_source_throttle_abs_diff, 0.95))
        if control_vs_source_throttle_abs_diff
        else None,
        "brake_abs_diff_mean_vs_source": float(np.mean(control_vs_source_brake_abs_diff))
        if control_vs_source_brake_abs_diff
        else None,
        "brake_abs_diff_p95_vs_source": float(np.quantile(control_vs_source_brake_abs_diff, 0.95))
        if control_vs_source_brake_abs_diff
        else None,
    }

    summary_path = (
        REPO_ROOT
        / "tmp"
        / "analysis"
        / f"{output_path.stem}_trajectory_locked_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    summary["summary_json"] = str(summary_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay control with trajectory reference locked from source recording."
    )
    parser.add_argument(
        "input_recording",
        nargs="?",
        default=None,
        help="Input recording for camera + vehicle state (default: latest)",
    )
    parser.add_argument(
        "--lock-recording",
        default=None,
        help="Recording to lock reference points from (default: same as input)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output recording name (default: trajectory_locked_<input_stem>)",
    )
    parser.add_argument(
        "--use-cv",
        action="store_true",
        help="Force CV-based perception (override default segmentation).",
    )
    parser.add_argument(
        "--use-segmentation",
        action="store_true",
        help="Deprecated compatibility flag; segmentation is already the default.",
    )
    parser.add_argument(
        "--segmentation-checkpoint",
        default=str(DEFAULT_SEGMENTATION_CHECKPOINT),
        help="Segmentation checkpoint path (default: segnet_best.pt).",
    )
    parser.add_argument(
        "--disable-vehicle-frame-lookahead-ref",
        action="store_true",
        help="Force disable vehicle-frame lookahead override during replay.",
    )
    parser.add_argument(
        "--lock-latency-ms",
        type=float,
        default=0.0,
        help="Delay locked reference by this many milliseconds (converted to frames).",
    )
    parser.add_argument(
        "--lock-latency-frames",
        type=int,
        default=0,
        help="Additional delay of locked reference in frames.",
    )
    parser.add_argument("--noise-seed", type=int, default=1337)
    parser.add_argument("--noise-std-x", type=float, default=0.0)
    parser.add_argument("--noise-std-heading", type=float, default=0.0)
    parser.add_argument("--noise-std-velocity", type=float, default=0.0)
    parser.add_argument("--noise-std-curvature", type=float, default=0.0)
    args = parser.parse_args()

    if args.use_cv and args.use_segmentation:
        print("Error: --use-cv and --use-segmentation cannot be combined")
        return 2
    use_segmentation = not bool(args.use_cv)
    if use_segmentation and not Path(args.segmentation_checkpoint).exists():
        print(
            f"Error: segmentation checkpoint not found: {args.segmentation_checkpoint}. "
            "Provide --segmentation-checkpoint or run with --use-cv."
        )
        return 2

    input_recording = Path(args.input_recording) if args.input_recording else _pick_latest_recording()
    if not input_recording.exists():
        print(f"Error: input recording not found: {input_recording}")
        return 2
    lock_recording = Path(args.lock_recording) if args.lock_recording else input_recording
    if not lock_recording.exists():
        print(f"Error: lock recording not found: {lock_recording}")
        return 2

    summary = replay_trajectory_locked(
        input_recording=input_recording,
        lock_recording=lock_recording,
        output_name=args.output,
        use_segmentation=use_segmentation,
        segmentation_checkpoint=args.segmentation_checkpoint,
        disable_vehicle_frame_lookahead_ref=bool(args.disable_vehicle_frame_lookahead_ref),
        lock_latency_ms=float(args.lock_latency_ms),
        lock_latency_frames=int(args.lock_latency_frames),
        noise_seed=int(args.noise_seed),
        noise_std_x=float(args.noise_std_x),
        noise_std_heading=float(args.noise_std_heading),
        noise_std_velocity=float(args.noise_std_velocity),
        noise_std_curvature=float(args.noise_std_curvature),
    )
    print("\nTrajectory-locked replay complete.")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
