"""
Script for collecting training data from Unity simulation.
Can be used for manual driving or automated data collection.
"""

import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bridge.client import UnityBridgeClient
from data.recorder import DataRecorder
from data.formats.data_format import CameraFrame, VehicleState, ControlCommand, RecordingFrame


def collect_data(bridge_url: str, output_dir: str, duration: float = 60.0, 
                manual_mode: bool = False):
    """
    Collect data from Unity simulation.
    
    Args:
        bridge_url: Unity bridge server URL
        output_dir: Directory to save recordings
        duration: Collection duration in seconds
        manual_mode: If True, wait for manual control; if False, use current AV commands
    """
    print(f"Starting data collection...")
    print(f"Duration: {duration} seconds")
    print(f"Mode: {'Manual' if manual_mode else 'Automated'}")
    
    # Connect to bridge
    bridge = UnityBridgeClient(bridge_url)
    
    if not bridge.health_check():
        print("ERROR: Bridge server is not available!")
        print("Please start the bridge server: python -m bridge.server")
        return
    
    # Create recorder
    recorder = DataRecorder(output_dir)
    
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < duration:
            # Get camera frame
            frame_data = bridge.get_latest_camera_frame()
            if frame_data is None:
                time.sleep(0.01)
                continue
            
            image, timestamp, camera_frame_id = frame_data
            
            # Get vehicle state
            vehicle_state_dict = bridge.get_latest_vehicle_state()
            if vehicle_state_dict is None:
                time.sleep(0.01)
                continue
            
            # Get control command (current command being sent to Unity)
            # In manual mode, this would come from Unity's manual input
            
            # Create recording frame
            camera_frame = CameraFrame(
                image=image,
                timestamp=timestamp,
                frame_id=camera_frame_id if camera_frame_id is not None else frame_count
            )
            
            # Create vehicle state
            position = vehicle_state_dict.get('position', {})
            vehicle_state = VehicleState(
                timestamp=timestamp,
                position=[position.get('x', 0), position.get('y', 0), position.get('z', 0)],
                rotation=[0, 0, 0, 1],  # Simplified
                velocity=[0, 0, vehicle_state_dict.get('speed', 0.0)],
                angular_velocity=[0, 0, 0],
                speed=vehicle_state_dict.get('speed', 0.0),
                steering_angle=vehicle_state_dict.get('steeringAngle', 0.0),
                motor_torque=vehicle_state_dict.get('motorTorque', 0.0),
                brake_torque=vehicle_state_dict.get('brakeTorque', 0.0)
            )
            
            frame = RecordingFrame(
                timestamp=timestamp,
                frame_id=frame_count,
                camera_frame=camera_frame,
                vehicle_state=vehicle_state
            )
            
            recorder.record_frame(frame)
            frame_count += 1
            
            # Print progress
            elapsed = time.time() - start_time
            if frame_count % 30 == 0:
                print(f"Collected {frame_count} frames ({elapsed:.1f}s / {duration:.1f}s)")
            
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\nData collection interrupted")
    finally:
        recorder.close()
        print(f"\nData collection complete: {frame_count} frames saved")


def main():
    parser = argparse.ArgumentParser(description='Collect training data from Unity')
    parser.add_argument('--bridge_url', type=str, default='http://localhost:8000',
                       help='Unity bridge server URL')
    parser.add_argument('--output_dir', type=str, default='data/recordings',
                       help='Output directory for recordings')
    parser.add_argument('--duration', type=float, default=60.0,
                       help='Collection duration in seconds')
    parser.add_argument('--manual', action='store_true',
                       help='Manual driving mode (requires Unity manual controls)')
    
    args = parser.parse_args()
    
    collect_data(
        bridge_url=args.bridge_url,
        output_dir=args.output_dir,
        duration=args.duration,
        manual_mode=args.manual
    )


if __name__ == "__main__":
    main()

