"""
Reprocess an existing recording with the current (fixed) code.

This is a PERCEPTION REPLAY TEST - it:
- Loads frames from any existing recording
- Reprocesses each frame with CURRENT code (with fixes applied)
- Recalculates perception (lane detection) 
- Recalculates heading (from perception)
- Records results to a NEW recording

This allows testing fixes (like coordinate conversion) without needing Unity running.
The car is NOT controlled - we just replay the perception/heading calculation.

By default uses the latest recording, but you can specify any recording file.
Works with any recording type (manual drive, ground truth follower, normal AV stack, etc.)
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


def reprocess_recording(input_recording: str, output_name: str = None):
    """
    Reprocess an existing recording with current code.
    
    Args:
        input_recording: Path to existing HDF5 recording
        output_name: Name for new recording (default: reprocessed_<original_name>)
    """
    print(f"Loading recording: {input_recording}")
    
    # Load existing recording
    with h5py.File(input_recording, 'r') as f:
        num_frames = len(f["camera/images"])
        print(f"Found {num_frames} frames")
        
        # Create new recording name
        if output_name is None:
            output_name = f"reprocessed_{Path(input_recording).stem}"
        
        print(f"Will create new recording: {output_name}")
        
        # Initialize AV stack (this has the fixed coordinate conversion)
        # We don't need bridge since we're using recorded data
        av_stack = AVStack(
            bridge_url="http://localhost:8000",  # Not used, but required
            record_data=True,  # Enable recording
            recording_dir="data/recordings"
        )
        
        # Override recording name and type
        av_stack.recorder.recording_name = output_name
        av_stack.recorder.recording_type = "reprocessed"
        av_stack.recorder.metadata["recording_type"] = "reprocessed"
        av_stack.recorder.metadata["source_recording"] = Path(input_recording).name
        
        print("\nReprocessing frames with fixed coordinate conversion...")
        print("="*80)
        
        for i in range(num_frames):
            # Load frame data
            image = f["camera/images"][i]
            timestamp = float(f["camera/timestamps"][i])
            frame_id = int(f["camera/frame_ids"][i])
            
            # Load vehicle state
            vehicle_state_dict = {
                'position': {
                    'x': float(f["vehicle/position"][i, 0]),
                    'y': float(f["vehicle/position"][i, 1]),
                    'z': float(f["vehicle/position"][i, 2])
                },
                'speed': float(f["vehicle/speed"][i]),
                'steeringAngle': float(f["vehicle/steering_angle"][i]),
                # Ground truth
                'groundTruthLeftLaneX': float(f["ground_truth/left_lane_x"][i]),
                'groundTruthRightLaneX': float(f["ground_truth/right_lane_x"][i]),
                'groundTruthLaneCenterX': float(f["ground_truth/lane_center_x"][i])
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
        
        print(f"\n✓ Reprocessing complete!")
        print(f"  New recording: {av_stack.recorder.output_file}")
        print(f"  Frames: {num_frames}")
        
        # Close recorder
        av_stack.recorder.close()
        
        return av_stack.recorder.output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Reprocess recording with fixed code (perception replay test)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Reprocess latest recording
  python tools/reprocess_recording.py
  
  # Reprocess specific recording
  python tools/reprocess_recording.py data/recordings/recording_20251231_120044.h5
  
  # Reprocess ground truth follower recording
  python tools/reprocess_recording.py data/recordings/gt_perception_test.h5
  
  # List available recordings
  python tools/list_recordings.py
        """
    )
    parser.add_argument("input_recording", nargs='?', default=None,
                       help="Path to input recording (default: latest)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output recording name")
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
        
        # Prefer non-reprocessed recordings (reprocessed ones are less useful to reprocess again)
        non_reprocessed = []
        for rec in recordings:
            try:
                with h5py.File(rec, 'r') as f:
                    if "metadata" in f.attrs:
                        metadata = json.loads(f.attrs["metadata"])
                        if metadata.get("recording_type") != "reprocessed":
                            non_reprocessed.append(rec)
                    else:
                        non_reprocessed.append(rec)  # Old recordings without metadata
            except:
                non_reprocessed.append(rec)  # If we can't read it, include it
        
        # Use first non-reprocessed, or fall back to latest
        if non_reprocessed:
            input_file = str(non_reprocessed[0])
            print(f"Using latest non-reprocessed recording: {Path(input_file).name}")
        else:
            input_file = str(recordings[0])
            print(f"Using latest recording: {Path(input_file).name}")
        
        print(f"  (You can specify a different recording: python tools/reprocess_recording.py <path>)")
        print(f"  (List recordings: python tools/reprocess_recording.py --list)\n")
    
    output_file = reprocess_recording(input_file, args.output)
    print(f"\nNow analyze the new recording:")
    print(f"  python tools/analyze_perception_questions.py {output_file}")

