"""
Replay perception stack against a recording using ground truth.

This is a PERCEPTION REPLAY TEST - it:
- Loads frames from any existing recording
- Replays perception stack (lane detection) with CURRENT code
- Recalculates heading from perception
- Compares against ground truth data in the recording
- Records results to a NEW recording

This allows testing perception/heading fixes without needing Unity running.
The car is NOT controlled - we just replay the perception/heading calculation.

By default uses the latest recording, but you can specify any recording file.
Works with any recording type (manual drive, GT drive, normal AV stack, etc.)

Usage:
    python tools/replay_perception.py [recording_file] [--output name]
    python tools/replay_perception.py --list  # List available recordings
"""

import sys
import h5py
import numpy as np
import json
from pathlib import Path
import time
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from av_stack import AVStack
from data.recorder import DataRecorder
from data.formats.data_format import CameraFrame, VehicleState, ControlCommand


def replay_perception(
    input_recording: str,
    output_name: str = None,
    reference_lookahead: float = 7.0,
    use_segmentation: bool = False,
    segmentation_checkpoint: str = None,
):
    """
    Replay perception stack against a recording using ground truth.
    
    Args:
        input_recording: Path to existing HDF5 recording
        output_name: Name for new recording (default: perception_replay_<original_name>)
        reference_lookahead: Lookahead distance to use (default: 7.0m to match visualizer tuning)
    """
    print(f"Loading recording: {input_recording}")
    
    # Load existing recording
    with h5py.File(input_recording, 'r') as f:
        num_frames = len(f["camera/images"])
        print(f"Found {num_frames} frames")
        
        # Create new recording name
        if output_name is None:
            output_name = f"perception_replay_{Path(input_recording).stem}"
        
        print(f"Will create new recording: {output_name}")
        
        # Initialize AV stack (this has the fixed coordinate conversion)
        # We don't need bridge since we're using recorded data
        av_stack = AVStack(
            bridge_url="http://localhost:8000",  # Not used, but required
            record_data=True,  # Enable recording
            recording_dir="data/recordings",
            use_segmentation=use_segmentation,
            segmentation_model_path=segmentation_checkpoint,
        )
        
        # CRITICAL: Override reference_lookahead to match visualizer tuning
        # The visualizer needs 7m to align lines, so replay should use the same
        # This ensures replay results match what you see in the visualizer
        av_stack.trajectory_config['reference_lookahead'] = reference_lookahead
        print(f"Using reference_lookahead = {reference_lookahead}m (to match visualizer tuning)")
        
        # Override recording name and type
        av_stack.recorder.recording_name = output_name
        av_stack.recorder.recording_type = "perception_replay"
        av_stack.recorder.metadata["recording_type"] = "perception_replay"
        av_stack.recorder.metadata["source_recording"] = Path(input_recording).name
        av_stack.recorder.metadata["reference_lookahead"] = reference_lookahead  # Record the tuning used
        av_stack.recorder.metadata["perception_mode"] = "segmentation" if use_segmentation else "cv_or_ml"
        
        print("\nReplaying perception stack against recording...")
        print("="*80)
        
        for i in range(num_frames):
            # Load frame data
            image = f["camera/images"][i]
            timestamp = float(f["camera/timestamps"][i])
            frame_id = int(f["camera/frame_ids"][i])
            
            # Load vehicle state (with all required fields)
            vehicle_state_dict = {
                'position': {
                    'x': float(f["vehicle/position"][i, 0]),
                    'y': float(f["vehicle/position"][i, 1]),
                    'z': float(f["vehicle/position"][i, 2])
                },
                'rotation': {
                    'x': float(f["vehicle/rotation"][i, 0]) if "vehicle/rotation" in f and len(f["vehicle/rotation"][i]) > 0 else 0.0,
                    'y': float(f["vehicle/rotation"][i, 1]) if "vehicle/rotation" in f and len(f["vehicle/rotation"][i]) > 1 else 0.0,
                    'z': float(f["vehicle/rotation"][i, 2]) if "vehicle/rotation" in f and len(f["vehicle/rotation"][i]) > 2 else 0.0,
                    'w': float(f["vehicle/rotation"][i, 3]) if "vehicle/rotation" in f and len(f["vehicle/rotation"][i]) > 3 else 1.0
                },
                'velocity': {
                    'x': float(f["vehicle/velocity"][i, 0]) if "vehicle/velocity" in f and len(f["vehicle/velocity"][i]) > 0 else 0.0,
                    'y': float(f["vehicle/velocity"][i, 1]) if "vehicle/velocity" in f and len(f["vehicle/velocity"][i]) > 1 else 0.0,
                    'z': float(f["vehicle/velocity"][i, 2]) if "vehicle/velocity" in f and len(f["vehicle/velocity"][i]) > 2 else 0.0
                },
                'angularVelocity': {
                    'x': float(f["vehicle/angular_velocity"][i, 0]) if "vehicle/angular_velocity" in f and len(f["vehicle/angular_velocity"][i]) > 0 else 0.0,
                    'y': float(f["vehicle/angular_velocity"][i, 1]) if "vehicle/angular_velocity" in f and len(f["vehicle/angular_velocity"][i]) > 1 else 0.0,
                    'z': float(f["vehicle/angular_velocity"][i, 2]) if "vehicle/angular_velocity" in f and len(f["vehicle/angular_velocity"][i]) > 2 else 0.0
                },
                'speed': float(f["vehicle/speed"][i]),
                'steeringAngle': float(f["vehicle/steering_angle"][i]),
                'motorTorque': float(f["vehicle/motor_torque"][i]) if "vehicle/motor_torque" in f else 0.0,
                'brakeTorque': float(f["vehicle/brake_torque"][i]) if "vehicle/brake_torque" in f else 0.0,
                # Ground truth (with backward compatibility for old field names)
                'groundTruthLeftLaneLineX': float(f["ground_truth/left_lane_line_x"][i]) if "ground_truth/left_lane_line_x" in f else float(f["ground_truth/left_lane_x"][i]),
                'groundTruthRightLaneLineX': float(f["ground_truth/right_lane_line_x"][i]) if "ground_truth/right_lane_line_x" in f else float(f["ground_truth/right_lane_x"][i]),
                'groundTruthLaneCenterX': float(f["ground_truth/lane_center_x"][i])
            }

            # Ground truth path curvature/heading if present
            if "ground_truth/path_curvature" in f:
                vehicle_state_dict['groundTruthPathCurvature'] = float(f["ground_truth/path_curvature"][i])
            if "ground_truth/desired_heading" in f:
                vehicle_state_dict['groundTruthDesiredHeading'] = float(f["ground_truth/desired_heading"][i])
            
            # Add camera calibration data if available
            if "vehicle/camera_8m_screen_y" in f:
                vehicle_state_dict['camera8mScreenY'] = float(f["vehicle/camera_8m_screen_y"][i])
            if "vehicle/camera_field_of_view" in f:
                vehicle_state_dict['cameraFieldOfView'] = float(f["vehicle/camera_field_of_view"][i])
            if "vehicle/camera_horizontal_fov" in f:
                vehicle_state_dict['cameraHorizontalFOV'] = float(f["vehicle/camera_horizontal_fov"][i])
            
            # Add camera position and forward direction if available
            if "vehicle/camera_pos_x" in f:
                vehicle_state_dict['cameraPosition'] = {
                    'x': float(f["vehicle/camera_pos_x"][i]),
                    'y': float(f["vehicle/camera_pos_y"][i]) if "vehicle/camera_pos_y" in f else 0.0,
                    'z': float(f["vehicle/camera_pos_z"][i]) if "vehicle/camera_pos_z" in f else 0.0
                }
            if "vehicle/camera_forward_x" in f:
                vehicle_state_dict['cameraForward'] = {
                    'x': float(f["vehicle/camera_forward_x"][i]),
                    'y': float(f["vehicle/camera_forward_y"][i]) if "vehicle/camera_forward_y" in f else 0.0,
                    'z': float(f["vehicle/camera_forward_z"][i]) if "vehicle/camera_forward_z" in f else 0.0
                }
            
            # Process frame with FIXED code
            # This will run perception and trajectory planning with the fixed coordinate conversion
            try:
                # Mock the bridge to prevent it from trying to send commands
                original_send = None
                if hasattr(av_stack.bridge, 'set_control_command'):
                    original_send = av_stack.bridge.set_control_command
                    av_stack.bridge.set_control_command = lambda *args, **kwargs: True
                
                av_stack._process_frame(image, timestamp, vehicle_state_dict)
                
                # Restore original method
                if original_send:
                    av_stack.bridge.set_control_command = original_send
                    
            except Exception as e:
                print(f"Error processing frame {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            if i % 30 == 0:
                print(f"Processed {i}/{num_frames} frames...")
        
        print(f"\n✓ Perception replay complete!")
        print(f"  New recording: {av_stack.recorder.output_file}")
        print(f"  Frames: {num_frames}")
        
        # Close recorder
        av_stack.recorder.close()
        
        return av_stack.recorder.output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Replay perception stack against a recording using ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Replay perception on latest recording
  python tools/replay_perception.py
  
  # Replay perception on specific recording
  python tools/replay_perception.py data/recordings/recording_20251231_120044.h5
  
  # Replay perception on GT drive recording
  python tools/replay_perception.py data/recordings/gt_test.h5
  
  # List available recordings
  python tools/replay_perception.py --list
        """
    )
    parser.add_argument("input_recording", nargs='?', default=None,
                       help="Path to input recording (default: latest)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output recording name")
    parser.add_argument("--use-segmentation", action="store_true",
                       help="Use segmentation model for perception replay")
    parser.add_argument("--segmentation-checkpoint", type=str, default=None,
                       help="Segmentation checkpoint path (required with --use-segmentation)")
    parser.add_argument("--reference-lookahead", type=float, default=7.0,
                       help="Reference lookahead distance in meters (default: 7.0m to match visualizer tuning)")
    parser.add_argument("--list", action="store_true",
                       help="List available recordings and exit")
    
    args = parser.parse_args()
    
    # List recordings if requested
    if args.list:
        from tools.list_recordings import list_recordings
        list_recordings()
        sys.exit(0)
    
    # Find input recording
    if args.input_recording:
        input_file = args.input_recording
        if not Path(input_file).exists():
            print(f"Error: Recording not found: {input_file}")
            sys.exit(1)
    else:
        recordings_dir = Path("data/recordings")
        recordings = sorted(recordings_dir.glob("*.h5"), 
                          key=lambda p: p.stat().st_mtime, reverse=True)
        if not recordings:
            print("No recordings found!")
            sys.exit(1)
        
        # Prefer non-replay recordings (replay ones are less useful to replay again)
        non_replay = []
        for rec in recordings:
            try:
                with h5py.File(rec, 'r') as f:
                    if "metadata" in f.attrs:
                        metadata = json.loads(f.attrs["metadata"])
                        if metadata.get("recording_type") != "perception_replay":
                            non_replay.append(rec)
                    else:
                        non_replay.append(rec)  # Old recordings without metadata
            except:
                non_replay.append(rec)  # If we can't read it, include it
        
        # Use first non-replay, or fall back to latest
        if non_replay:
            input_file = str(non_replay[0])
            print(f"Using latest non-replay recording: {Path(input_file).name}")
        else:
            input_file = str(recordings[0])
            print(f"Using latest recording: {Path(input_file).name}")
        
        print(f"  (You can specify a different recording: python tools/replay_perception.py <path>)")
        print(f"  (List recordings: python tools/replay_perception.py --list)\n")
    
    if args.use_segmentation and not args.segmentation_checkpoint:
        print("Error: --segmentation-checkpoint is required with --use-segmentation")
        sys.exit(1)
    
    output_file = replay_perception(
        input_file,
        args.output,
        args.reference_lookahead,
        use_segmentation=args.use_segmentation,
        segmentation_checkpoint=args.segmentation_checkpoint,
    )
    print(f"\nNow analyze the new recording:")
    print(f"  python tools/analyze/analyze_perception_questions.py {output_file}")

