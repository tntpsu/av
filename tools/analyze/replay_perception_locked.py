#!/usr/bin/env python3
"""
Replay with perception output locked from a source recording.

Stage-1 isolation harness:
- Replays frames from an input recording through AVStack
- Overrides self.perception.detect() every frame with locked perception from a source recording
- Lets trajectory planning and control run live against the locked lane detections
- Records a new HDF5 output and JSON summary

The goal is to isolate trajectory and control behavior from live perception variation.
If replay output matches source, then trajectory/control are deterministic given the
same perception input. If they diverge, the trajectory planner or controller has
state-dependent behaviour beyond what perception alone determines.

Lane polynomial coefficients are stored per frame as a flattened float32 vlen array
(format: [a0, b0, c0, a1, b1, c1, ...] for n_lanes × degree-2 polynomials).
num_lanes_detected and left/right scalar positions are used to reconstruct the
[left_coeffs_or_None, right_coeffs_or_None] list expected by the planner.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SEGMENTATION_CHECKPOINT = (
    REPO_ROOT / "data" / "segmentation_dataset" / "checkpoints" / "segnet_best.pt"
)

# Degree of polynomial used by the CV/segmentation perception pipeline.
# All existing detectors fit degree-2 polynomials, giving 3 coefficients per lane.
_POLY_DEGREE = 2
_COEFFS_PER_LANE = _POLY_DEGREE + 1  # 3


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


def _load_locked_perception(
    lock_file: h5py.File,
    idx: int,
) -> Optional[tuple]:
    """
    Load and reconstruct perception output for a given frame from a source HDF5 recording.

    Returns a 4-tuple (lane_coeffs, confidence, detection_method, num_lanes_detected)
    matching the extended av_stack detect() format, or None if perception datasets are
    absent in the lock file.

    lane_coeffs is reconstructed as [left_coeffs_or_None, right_coeffs_or_None].
    The flat stored coefficient array is split using num_lanes_detected and, for the
    single-lane case, the left/right scalar positions disambiguate which slot to fill.
    """
    # Check that this recording has perception data at all.
    if "perception/confidence" not in lock_file:
        return None
    if idx >= _get_len(lock_file, "perception/confidence"):
        return None

    # --- scalars ----------------------------------------------------------
    confidence = 1.0
    try:
        v = float(lock_file["perception/confidence"][idx])
        confidence = v if np.isfinite(v) else 1.0
    except Exception:
        pass

    detection_method = "cv"
    if "perception/detection_method" in lock_file and idx < len(lock_file["perception/detection_method"]):
        try:
            detection_method = _decode_bytes(lock_file["perception/detection_method"][idx])
        except Exception:
            pass

    num_lanes_detected = 0
    if "perception/num_lanes_detected" in lock_file and idx < len(lock_file["perception/num_lanes_detected"]):
        try:
            num_lanes_detected = int(lock_file["perception/num_lanes_detected"][idx])
        except Exception:
            pass

    # Scalar x positions used to assign single-lane detections to left vs right.
    left_x: Optional[float] = None
    right_x: Optional[float] = None
    if "perception/left_lane_line_x" in lock_file and idx < len(lock_file["perception/left_lane_line_x"]):
        try:
            v = float(lock_file["perception/left_lane_line_x"][idx])
            if np.isfinite(v):
                left_x = v
        except Exception:
            pass
    if "perception/right_lane_line_x" in lock_file and idx < len(lock_file["perception/right_lane_line_x"]):
        try:
            v = float(lock_file["perception/right_lane_line_x"][idx])
            if np.isfinite(v):
                right_x = v
        except Exception:
            pass

    # --- reconstruct lane_coeffs ------------------------------------------
    # Storage format: flat float32 vlen row, concatenation of non-None coefficient
    # arrays in left-before-right order.  Each lane uses degree-2 polynomials
    # (_COEFFS_PER_LANE = 3 coefficients).
    lane_coeffs: List[Optional[np.ndarray]] = [None, None]

    if "perception/lane_line_coefficients" in lock_file and idx < len(lock_file["perception/lane_line_coefficients"]):
        try:
            flat = np.asarray(lock_file["perception/lane_line_coefficients"][idx], dtype=np.float32)
        except Exception:
            flat = np.array([], dtype=np.float32)

        if len(flat) > 0 and num_lanes_detected > 0:
            # Infer coefficients per lane from stored length; fall back to degree-2 default.
            n_per_lane = len(flat) // num_lanes_detected
            if n_per_lane < 1:
                n_per_lane = _COEFFS_PER_LANE

            chunks = [
                flat[i * n_per_lane : (i + 1) * n_per_lane]
                for i in range(num_lanes_detected)
                if (i + 1) * n_per_lane <= len(flat)
            ]

            if len(chunks) >= 2:
                lane_coeffs[0] = chunks[0]  # left
                lane_coeffs[1] = chunks[1]  # right
            elif len(chunks) == 1:
                # Single lane: use scalar positions to assign slot.
                right_valid = right_x is not None
                left_valid = left_x is not None
                if right_valid and not left_valid:
                    lane_coeffs[1] = chunks[0]  # only right lane detected
                else:
                    lane_coeffs[0] = chunks[0]  # only left lane (default)

    return lane_coeffs, confidence, detection_method, num_lanes_detected


def replay_perception_locked(
    input_recording: Path,
    lock_recording: Path,
    output_name: Optional[str],
    use_segmentation: bool,
    segmentation_checkpoint: Optional[str],
) -> Dict[str, Any]:
    from av_stack import AVStack  # Imported lazily for lightweight helper tests.

    if output_name is None:
        output_name = f"perception_locked_{input_recording.stem}"

    with h5py.File(input_recording, "r") as in_f, h5py.File(lock_recording, "r") as lock_f:
        n_input = _get_len(in_f, "camera/images")
        n_lock = _get_len(lock_f, "perception/confidence")
        if n_input == 0:
            raise RuntimeError(f"No camera frames found in {input_recording}")
        if n_lock == 0:
            raise RuntimeError(
                f"No perception datasets found in {lock_recording}. "
                "Ensure the lock recording was made with record_data=True."
            )

        n_frames = min(n_input, n_lock)
        print(f"Input frames: {n_input}, lock perception frames: {n_lock}, using: {n_frames}")

        av_stack = AVStack(
            bridge_url="http://localhost:8000",
            record_data=True,
            recording_dir=str(REPO_ROOT / "data" / "recordings"),
            use_segmentation=use_segmentation,
            segmentation_model_path=segmentation_checkpoint,
        )
        av_stack.recorder.recording_name = output_name
        av_stack.recorder.recording_type = "perception_locked_replay"
        av_stack.recorder.metadata["recording_type"] = "perception_locked_replay"
        av_stack.recorder.metadata["source_recording"] = input_recording.name
        av_stack.recorder.metadata["perception_lock_source_recording"] = lock_recording.name
        av_stack.recorder.metadata["perception_mode"] = (
            "segmentation" if use_segmentation else "cv_or_ml"
        )

        original_detect = av_stack.perception.detect
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

        missing_locked_perception_count = 0
        fallback_to_live_count = 0
        control_vs_source_steering_abs_diff: list[float] = []
        control_vs_source_throttle_abs_diff: list[float] = []
        control_vs_source_brake_abs_diff: list[float] = []

        # av_stack checks len(perception_result) == 4 to accept the extended format.
        # We always return a 4-tuple so detection_method and num_lanes are preserved
        # exactly without touching perception.last_detection_method per frame.
        locked_perception_holder: dict = {"result": None}

        def _locked_detect(image: np.ndarray):
            result = locked_perception_holder["result"]
            if result is None:
                # Lock data unavailable for this frame — fall back to live perception.
                return original_detect(image)
            return result  # (lane_coeffs, confidence, detection_method, num_lanes_detected)

        try:
            av_stack.perception.detect = _locked_detect

            for idx in range(n_frames):
                frame_holder["idx"] = idx
                image = in_f["camera/images"][idx]
                timestamp = float(in_f["camera/timestamps"][idx])
                vehicle_state_dict = _build_vehicle_state_dict(in_f, idx)

                locked = _load_locked_perception(lock_f, idx)
                if locked is None:
                    missing_locked_perception_count += 1
                    fallback_to_live_count += 1
                    locked_perception_holder["result"] = None
                else:
                    locked_perception_holder["result"] = locked

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
            av_stack.perception.detect = original_detect
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
        "missing_locked_perception_frames": int(missing_locked_perception_count),
        "fallback_to_live_perception_frames": int(fallback_to_live_count),
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
        / f"{output_path.stem}_perception_locked_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    summary["summary_json"] = str(summary_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay AVStack with perception outputs locked from source recording."
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
        help="Recording to lock perception outputs from (default: same as input)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output recording name (default: perception_locked_<input_stem>)",
    )
    parser.add_argument(
        "--use-cv",
        action="store_true",
        help="Force CV-based perception for live fallback frames.",
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

    summary = replay_perception_locked(
        input_recording=input_recording,
        lock_recording=lock_recording,
        output_name=args.output,
        use_segmentation=use_segmentation,
        segmentation_checkpoint=args.segmentation_checkpoint,
    )
    print("\nPerception-locked replay complete.")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
