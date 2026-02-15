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

    return {
        "x": x,
        "y": y,
        "heading": heading,
        "velocity": velocity,
        "method": f"locked:{method}" if method else "locked",
        "curvature": curvature,
        "perception_center_x": perception_center_x,
    }


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
        if hasattr(av_stack.bridge, "set_control_command"):
            original_send = av_stack.bridge.set_control_command
            av_stack.bridge.set_control_command = lambda *args, **kwargs: True
        if hasattr(av_stack.bridge, "get_latest_camera_frame"):
            original_get_camera_frame = av_stack.bridge.get_latest_camera_frame

        has_topdown_input = "camera/topdown_images" in in_f and len(in_f["camera/topdown_images"]) > 0
        av_stack.recorder.metadata["topdown_replayed_from_input"] = bool(has_topdown_input)
        frame_holder = {"idx": 0}
        if has_topdown_input and original_get_camera_frame is not None:
            topdown_images = in_f["camera/topdown_images"]
            cam_timestamps = in_f["camera/timestamps"]

            def _replay_get_latest_camera_frame(*args, **kwargs):
                camera_id = kwargs.get("camera_id")
                if camera_id is None and len(args) >= 1:
                    camera_id = args[0]
                if camera_id in ("top_down", "topdown"):
                    i = min(frame_holder["idx"], len(topdown_images) - 1)
                    return topdown_images[i], float(cam_timestamps[i]), None
                return original_get_camera_frame(*args, **kwargs)

            av_stack.bridge.get_latest_camera_frame = _replay_get_latest_camera_frame

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
