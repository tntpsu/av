#!/usr/bin/env python3
"""
Replay with control commands locked from a source recording.

Stage-3 isolation harness:
- Replays frames from an input recording through AVStack
- Replaces controller output with locked control series from lock recording
- Records a new HDF5 output and JSON summary

Use this to isolate the impact of upstream perception/trajectory changes when
the control policy output is held fixed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

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

    # Ground truth names vary between old/new recordings.
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

    _inject_oracle_and_fiducials(source_file, idx, d)
    return d


def replay_control_locked(
    input_recording: Path,
    lock_recording: Path,
    output_name: str | None,
    use_segmentation: bool,
    segmentation_checkpoint: str | None,
) -> Dict[str, Any]:
    if output_name is None:
        output_name = f"control_locked_{input_recording.stem}"

    with h5py.File(input_recording, "r") as in_f, h5py.File(lock_recording, "r") as lock_f:
        n_input = _get_len(in_f, "camera/images")
        n_lock = min(
            _get_len(lock_f, "control/steering"),
            _get_len(lock_f, "control/throttle"),
            _get_len(lock_f, "control/brake"),
        )
        if n_input == 0:
            raise RuntimeError(f"No camera frames found in {input_recording}")
        if n_lock == 0:
            raise RuntimeError(f"No lock control datasets found in {lock_recording}")

        n_frames = min(n_input, n_lock)
        print(f"Input frames: {n_input}, lock control frames: {n_lock}, using: {n_frames}")

        # Cache lock/source controls for post-safety attribution.
        lock_steering = np.asarray(lock_f["control/steering"][:n_frames], dtype=float)
        lock_throttle = np.asarray(lock_f["control/throttle"][:n_frames], dtype=float)
        lock_brake = np.asarray(lock_f["control/brake"][:n_frames], dtype=float)
        source_steering = (
            np.asarray(in_f["control/steering"][:n_frames], dtype=float)
            if "control/steering" in in_f and len(in_f["control/steering"]) >= n_frames
            else None
        )
        source_throttle = (
            np.asarray(in_f["control/throttle"][:n_frames], dtype=float)
            if "control/throttle" in in_f and len(in_f["control/throttle"]) >= n_frames
            else None
        )
        source_brake = (
            np.asarray(in_f["control/brake"][:n_frames], dtype=float)
            if "control/brake" in in_f and len(in_f["control/brake"]) >= n_frames
            else None
        )

        av_stack = AVStack(
            bridge_url="http://localhost:8000",
            record_data=True,
            recording_dir=str(REPO_ROOT / "data" / "recordings"),
            use_segmentation=use_segmentation,
            segmentation_model_path=segmentation_checkpoint,
        )
        av_stack.recorder.recording_name = output_name
        av_stack.recorder.recording_type = "control_locked_replay"
        av_stack.recorder.metadata["recording_type"] = "control_locked_replay"
        av_stack.recorder.metadata["source_recording"] = input_recording.name
        av_stack.recorder.metadata["control_lock_source_recording"] = lock_recording.name
        av_stack.recorder.metadata["perception_mode"] = (
            "segmentation" if use_segmentation else "cv_or_ml"
        )

        original_compute_control = av_stack.controller.compute_control
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

        frame_holder = {"idx": 0}
        has_topdown_input = "camera/topdown_images" in in_f and len(in_f["camera/topdown_images"]) > 0
        av_stack.recorder.metadata["topdown_replayed_from_input"] = bool(has_topdown_input)
        if has_topdown_input and (
            original_get_camera_frame is not None
            or original_get_camera_frame_with_metadata is not None
        ):
            topdown_images = in_f["camera/topdown_images"]
            if "camera/topdown_timestamps" in in_f and len(in_f["camera/topdown_timestamps"]) > 0:
                topdown_timestamps = in_f["camera/topdown_timestamps"]
            else:
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

        def _locked_compute_control(*args, **kwargs):
            i = frame_holder["idx"]
            return {
                "steering": float(lock_steering[i]),
                "throttle": float(lock_throttle[i]),
                "brake": float(lock_brake[i]),
                "lateral_error": 0.0,
                "heading_error": 0.0,
                "total_error": 0.0,
                "pid_integral": 0.0,
                "pid_derivative": 0.0,
            }

        try:
            av_stack.controller.compute_control = _locked_compute_control

            for idx in range(n_frames):
                frame_holder["idx"] = idx
                image = in_f["camera/images"][idx]
                timestamp = float(in_f["camera/timestamps"][idx])
                vehicle_state_dict = _build_vehicle_state_dict(in_f, idx)
                av_stack._process_frame(image, timestamp, vehicle_state_dict)

                if idx % 30 == 0:
                    print(f"Processed {idx}/{n_frames} frames...")
        finally:
            av_stack.controller.compute_control = original_compute_control
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
    with h5py.File(output_path, "r") as out_f:
        n_out = min(
            n_frames,
            _get_len(out_f, "control/steering"),
            _get_len(out_f, "control/throttle"),
            _get_len(out_f, "control/brake"),
        )
        out_steering = np.asarray(out_f["control/steering"][:n_out], dtype=float)
        out_throttle = np.asarray(out_f["control/throttle"][:n_out], dtype=float)
        out_brake = np.asarray(out_f["control/brake"][:n_out], dtype=float)
        out_emergency = (
            np.asarray(out_f["control/emergency_stop"][:n_out], dtype=bool)
            if "control/emergency_stop" in out_f
            else np.zeros(n_out, dtype=bool)
        )

    eps = 1e-4
    d_steer_lock = np.abs(out_steering - lock_steering[:n_out])
    d_throttle_lock = np.abs(out_throttle - lock_throttle[:n_out])
    d_brake_lock = np.abs(out_brake - lock_brake[:n_out])
    override_steer = d_steer_lock > eps
    override_throttle = d_throttle_lock > eps
    override_brake = d_brake_lock > eps
    override_any = override_steer | override_throttle | override_brake
    throttle_cut = (
        (lock_throttle[:n_out] > eps)
        & (out_throttle <= eps)
        & (out_brake > lock_brake[:n_out] + eps)
    )
    brake_increase = out_brake > (lock_brake[:n_out] + eps)

    def _mean(arr: np.ndarray) -> float | None:
        return float(np.mean(arr)) if arr.size else None

    def _p95(arr: np.ndarray) -> float | None:
        return float(np.quantile(arr, 0.95)) if arr.size else None

    d_steer_src = None
    d_throttle_src = None
    d_brake_src = None
    if source_steering is not None:
        d_steer_src = np.abs(out_steering - source_steering[:n_out])
    if source_throttle is not None:
        d_throttle_src = np.abs(out_throttle - source_throttle[:n_out])
    if source_brake is not None:
        d_brake_src = np.abs(out_brake - source_brake[:n_out])

    summary = {
        "input_recording": str(input_recording),
        "lock_recording": str(lock_recording),
        "output_recording": str(output_path),
        "frames_processed": int(n_out),
        "steering_abs_diff_mean_vs_lock": _mean(d_steer_lock),
        "steering_abs_diff_p95_vs_lock": _p95(d_steer_lock),
        "throttle_abs_diff_mean_vs_lock": _mean(d_throttle_lock),
        "throttle_abs_diff_p95_vs_lock": _p95(d_throttle_lock),
        "brake_abs_diff_mean_vs_lock": _mean(d_brake_lock),
        "brake_abs_diff_p95_vs_lock": _p95(d_brake_lock),
        "steering_abs_diff_mean_vs_input_source": _mean(d_steer_src) if d_steer_src is not None else None,
        "steering_abs_diff_p95_vs_input_source": _p95(d_steer_src) if d_steer_src is not None else None,
        "throttle_abs_diff_mean_vs_input_source": _mean(d_throttle_src) if d_throttle_src is not None else None,
        "throttle_abs_diff_p95_vs_input_source": _p95(d_throttle_src) if d_throttle_src is not None else None,
        "brake_abs_diff_mean_vs_input_source": _mean(d_brake_src) if d_brake_src is not None else None,
        "brake_abs_diff_p95_vs_input_source": _p95(d_brake_src) if d_brake_src is not None else None,
        "post_safety_override": {
            "epsilon": eps,
            "steering_override_frames": int(np.sum(override_steer)),
            "throttle_override_frames": int(np.sum(override_throttle)),
            "brake_override_frames": int(np.sum(override_brake)),
            "any_control_override_frames": int(np.sum(override_any)),
            "override_pct": float(np.mean(override_any) * 100.0) if n_out > 0 else 0.0,
            "emergency_stop_frames": int(np.sum(out_emergency)),
            "override_while_emergency_frames": int(np.sum(override_any & out_emergency)),
            "throttle_cut_with_brake_increase_frames": int(np.sum(throttle_cut)),
            "brake_increase_frames": int(np.sum(brake_increase)),
        },
    }
    summary_path = (
        REPO_ROOT
        / "tmp"
        / "analysis"
        / f"{output_path.stem}_control_locked_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    summary["summary_json"] = str(summary_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay AVStack with control outputs locked from source recording."
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
        help="Recording to lock control outputs from (default: same as input)",
    )
    parser.add_argument("--output", default=None, help="Output recording name")
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

    summary = replay_control_locked(
        input_recording=input_recording,
        lock_recording=lock_recording,
        output_name=args.output,
        use_segmentation=use_segmentation,
        segmentation_checkpoint=args.segmentation_checkpoint,
    )
    print("\nControl-locked replay complete.")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
