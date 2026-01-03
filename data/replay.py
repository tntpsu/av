"""
Data replay utility for AV stack recordings.
Allows replaying recorded data for debugging and visualization.
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Optional, Iterator, Tuple
import cv2


class DataReplay:
    """Replay recorded AV stack data."""
    
    def __init__(self, recording_file: str):
        """
        Initialize data replay.
        
        Args:
            recording_file: Path to HDF5 recording file
        """
        self.recording_file = Path(recording_file)
        if not self.recording_file.exists():
            raise FileNotFoundError(f"Recording file not found: {recording_file}")
        
        self.h5_file = h5py.File(self.recording_file, 'r')
        self._load_metadata()
    
    def _load_metadata(self):
        """Load recording metadata."""
        if "metadata" in self.h5_file.attrs:
            import json
            self.metadata = json.loads(self.h5_file.attrs["metadata"])
        else:
            self.metadata = {}
    
    def get_camera_frames(self) -> Iterator[Tuple[np.ndarray, float, int]]:
        """
        Get camera frames iterator.
        
        Yields:
            Tuple of (image, timestamp, frame_id)
        """
        if "camera/images" not in self.h5_file:
            return
        
        images = self.h5_file["camera/images"]
        timestamps = self.h5_file["camera/timestamps"]
        frame_ids = self.h5_file["camera/frame_ids"]
        
        for i in range(len(images)):
            yield images[i], float(timestamps[i]), int(frame_ids[i])
    
    def get_vehicle_states(self) -> Iterator[dict]:
        """
        Get vehicle states iterator.
        
        Yields:
            Dictionary with vehicle state data
        """
        if "vehicle/timestamps" not in self.h5_file:
            return
        
        timestamps = self.h5_file["vehicle/timestamps"]
        positions = self.h5_file["vehicle/position"]
        rotations = self.h5_file["vehicle/rotation"]
        velocities = self.h5_file["vehicle/velocity"]
        angular_velocities = self.h5_file["vehicle/angular_velocity"]
        speeds = self.h5_file["vehicle/speed"]
        steering_angles = self.h5_file["vehicle/steering_angle"]
        motor_torques = self.h5_file["vehicle/motor_torque"]
        brake_torques = self.h5_file["vehicle/brake_torque"]
        
        for i in range(len(timestamps)):
            yield {
                "timestamp": float(timestamps[i]),
                "position": positions[i],
                "rotation": rotations[i],
                "velocity": velocities[i],
                "angular_velocity": angular_velocities[i],
                "speed": float(speeds[i]),
                "steering_angle": float(steering_angles[i]),
                "motor_torque": float(motor_torques[i]),
                "brake_torque": float(brake_torques[i])
            }
    
    def get_control_commands(self) -> Iterator[dict]:
        """
        Get control commands iterator.
        
        Yields:
            Dictionary with control command data
        """
        if "control/timestamps" not in self.h5_file:
            return
        
        timestamps = self.h5_file["control/timestamps"]
        steerings = self.h5_file["control/steering"]
        throttles = self.h5_file["control/throttle"]
        brakes = self.h5_file["control/brake"]
        
        for i in range(len(timestamps)):
            yield {
                "timestamp": float(timestamps[i]),
                "steering": float(steerings[i]),
                "throttle": float(throttles[i]),
                "brake": float(brakes[i])
            }
    
    def get_synchronized_frames(self) -> Iterator[dict]:
        """
        Get synchronized frames (camera + vehicle state + control).
        
        Yields:
            Dictionary with all synchronized data
        """
        # Simple synchronization by timestamp
        camera_data = list(self.get_camera_frames())
        vehicle_data = list(self.get_vehicle_states())
        control_data = list(self.get_control_commands())
        
        # Match by closest timestamp
        for cam_img, cam_ts, cam_id in camera_data:
            # Find closest vehicle state
            vehicle_state = None
            if vehicle_data:
                vehicle_state = min(
                    vehicle_data,
                    key=lambda v: abs(v["timestamp"] - cam_ts)
                )
            
            # Find closest control command
            control_command = None
            if control_data:
                control_command = min(
                    control_data,
                    key=lambda c: abs(c["timestamp"] - cam_ts)
                )
            
            yield {
                "camera": {
                    "image": cam_img,
                    "timestamp": cam_ts,
                    "frame_id": cam_id
                },
                "vehicle_state": vehicle_state,
                "control_command": control_command
            }
    
    def visualize(self, fps: int = 30):
        """
        Visualize recording with OpenCV.
        
        Args:
            fps: Playback frames per second
        """
        import time
        
        frame_time = 1.0 / fps
        
        for frame_data in self.get_synchronized_frames():
            img = frame_data["camera"]["image"]
            
            # Draw vehicle state info
            if frame_data["vehicle_state"]:
                vs = frame_data["vehicle_state"]
                info_text = [
                    f"Speed: {vs['speed']:.2f} m/s",
                    f"Steering: {vs['steering_angle']:.2f} deg",
                    f"Position: ({vs['position'][0]:.2f}, {vs['position'][2]:.2f})"
                ]
                
                for i, text in enumerate(info_text):
                    cv2.putText(
                        img, text, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                    )
            
            # Draw control command info
            if frame_data["control_command"]:
                cc = frame_data["control_command"]
                control_text = [
                    f"Steering Cmd: {cc['steering']:.2f}",
                    f"Throttle Cmd: {cc['throttle']:.2f}",
                    f"Brake Cmd: {cc['brake']:.2f}"
                ]
                
                for i, text in enumerate(control_text):
                    cv2.putText(
                        img, text, (10, 150 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )
            
            cv2.imshow("Replay", img)
            
            if cv2.waitKey(int(frame_time * 1000)) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    def get_statistics(self) -> dict:
        """Get statistics about the recording."""
        stats = {
            "file": str(self.recording_file),
            "metadata": self.metadata
        }
        
        if "camera/images" in self.h5_file:
            stats["camera_frames"] = len(self.h5_file["camera/images"])
        
        if "vehicle/timestamps" in self.h5_file:
            stats["vehicle_states"] = len(self.h5_file["vehicle/timestamps"])
        
        if "control/timestamps" in self.h5_file:
            stats["control_commands"] = len(self.h5_file["control/timestamps"])
        
        return stats
    
    def close(self):
        """Close the replay file."""
        self.h5_file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m data.replay <recording_file.h5>")
        sys.exit(1)
    
    recording_file = sys.argv[1]
    
    with DataReplay(recording_file) as replay:
        stats = replay.get_statistics()
        print("Recording Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\nStarting visualization (press 'q' to quit)...")
        replay.visualize()

