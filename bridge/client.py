"""
Python client helper for Unity bridge communication.
Provides convenient methods for AV stack components to interact with Unity.
"""

import requests
import numpy as np
from PIL import Image
import base64
import io
from typing import Optional, Dict, Tuple
import time


class UnityBridgeClient:
    """Client for communicating with Unity bridge server."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize Unity bridge client.
        
        Args:
            base_url: Base URL of the bridge server
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
    
    def get_latest_camera_frame(
        self,
        camera_id: str = "front_center"
    ) -> Optional[Tuple[np.ndarray, float, Optional[int]]]:
        """
        Get latest camera frame from Unity.
        
        Returns:
            Tuple of (image_array, timestamp) or None if not available
        """
        frame = self.get_latest_camera_frame_with_metadata(camera_id=camera_id)
        if frame is None:
            return None
        image, timestamp, frame_id, _meta = frame
        return image, timestamp, frame_id

    def get_latest_camera_frame_with_metadata(
        self,
        camera_id: str = "front_center"
    ) -> Optional[Tuple[np.ndarray, float, Optional[int], Dict]]:
        """Get latest camera frame plus bridge freshness metadata."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/camera/latest",
                params={"camera_id": camera_id},
                timeout=0.5,
            )
            response.raise_for_status()

            data = response.json()
            img_base64 = data["image"]
            timestamp = data.get("timestamp", time.time())
            frame_id = data.get("frame_id")

            img_bytes = base64.b64decode(img_base64)
            img = Image.open(io.BytesIO(img_bytes))
            img_array = np.array(img)

            meta = {
                "queue_depth": data.get("queue_depth"),
                "queue_capacity": data.get("queue_capacity"),
                "drop_count": data.get("drop_count"),
                "decode_in_flight": data.get("decode_in_flight"),
                "latest_age_ms": data.get("latest_age_ms"),
                "last_arrival_time": data.get("last_arrival_time"),
                "last_realtime_since_startup": data.get("last_realtime_since_startup"),
                "last_unscaled_time": data.get("last_unscaled_time"),
            }
            return img_array, timestamp, frame_id, meta
        except requests.exceptions.Timeout:
            return None
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            return None
        except requests.RequestException:
            return None

    def get_next_camera_frame(
        self,
        camera_id: str = "front_center"
    ) -> Optional[Tuple[np.ndarray, float, Optional[int]]]:
        """
        Get next queued camera frame from Unity (FIFO order).

        Returns:
            Tuple of (image_array, timestamp, frame_id) or None if unavailable
        """
        try:
            response = self.session.get(
                f"{self.base_url}/api/camera/next",
                params={"camera_id": camera_id},
                timeout=0.5,
            )
            response.raise_for_status()

            data = response.json()
            img_base64 = data["image"]
            timestamp = data.get("timestamp", time.time())
            frame_id = data.get("frame_id")

            img_bytes = base64.b64decode(img_base64)
            img = Image.open(io.BytesIO(img_bytes))
            img_array = np.array(img)

            return img_array, timestamp, frame_id
        except requests.exceptions.Timeout:
            return None
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            return None
        except requests.RequestException:
            return None
    
    def get_latest_vehicle_state(self) -> Optional[Dict]:
        """
        Get latest vehicle state from Unity.
        
        Returns:
            Vehicle state dictionary or None if not available
        """
        try:
            response = self.session.get(f"{self.base_url}/api/vehicle/state/latest", timeout=0.5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            return None
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            return None
        except requests.RequestException:
            return None

    def get_next_vehicle_state(self) -> Optional[Dict]:
        """
        Get next queued vehicle state from Unity bridge (FIFO order).

        Returns:
            Vehicle state dictionary or None if unavailable
        """
        try:
            response = self.session.get(f"{self.base_url}/api/vehicle/state/next", timeout=0.5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            return None
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            return None
        except requests.RequestException:
            return None
    
    def get_latest_unity_feedback(self) -> Optional[Dict]:
        """
        Get latest Unity feedback/status data.
        
        Returns:
            Unity feedback dictionary or None if not available
        """
        try:
            response = self.session.get(f"{self.base_url}/api/unity/feedback/latest", timeout=0.5)
            response.raise_for_status()
            data = response.json()
            return data if data else None
        except requests.exceptions.Timeout:
            return None
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            return None
        except requests.RequestException:
            return None
    
    def _build_control_command(
        self,
        steering: float,
        throttle: float,
        brake: float,
        ground_truth_mode: bool = False,
        ground_truth_speed: float = 5.0,
        randomize_start: bool = False,
        randomize_request_id: int = 0,
        randomize_seed: int | None = None,
    ) -> dict:
        command = {
            "steering": float(steering),
            "throttle": float(throttle),
            "brake": float(brake),
            "ground_truth_mode": bool(ground_truth_mode),
            "ground_truth_speed": float(ground_truth_speed),
            "randomize_start": bool(randomize_start),
            "randomize_request_id": int(randomize_request_id),
        }
        if randomize_seed is not None:
            command["randomize_seed"] = int(randomize_seed)
        return command

    def set_control_command(
        self,
        steering: float,
        throttle: float,
        brake: float,
        ground_truth_mode: bool = False,
        ground_truth_speed: float = 5.0,
        randomize_start: bool = False,
        randomize_request_id: int = 0,
        randomize_seed: int | None = None,
    ) -> bool:
        """
        Set control command for Unity vehicle.
        
        Args:
            steering: Steering angle (-1.0 to 1.0)
            throttle: Throttle (-1.0 to 1.0, negative = reverse)
            brake: Brake (0.0 to 1.0)
            ground_truth_mode: Enable direct velocity control for precise path following
            ground_truth_speed: Speed for ground truth mode (m/s)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            command = self._build_control_command(
                steering=steering,
                throttle=throttle,
                brake=brake,
                ground_truth_mode=ground_truth_mode,
                ground_truth_speed=ground_truth_speed,
                randomize_start=randomize_start,
                randomize_request_id=randomize_request_id,
                randomize_seed=randomize_seed,
            )
            
            response = self.session.post(
                f"{self.base_url}/api/vehicle/control",
                json=command
            )
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            print(f"Error setting control command: {e}")
            return False
    
    def request_unity_play(self) -> bool:
        """
        Request Unity to start playing (enter play mode).
        Only works when Unity Editor is open.
        
        Returns:
            True if request was sent successfully, False otherwise
        """
        try:
            response = self.session.post(f"{self.base_url}/api/unity/play", timeout=2.0)
            response.raise_for_status()
            result = response.json()
            if result.get("status") == "play_requested":
                return True
            return False
        except requests.RequestException as e:
            print(f"Error requesting Unity play: {e}")
            return False
    
    def set_trajectory_data(
        self,
        trajectory_points: list,
        reference_point: list,
        lateral_error: float,
        perception_left_lane_x: float | None = None,
        perception_right_lane_x: float | None = None,
        perception_center_x: float | None = None,
        perception_lookahead_m: float | None = None,
        perception_valid: bool | None = None,
    ) -> bool:
        """
        Set trajectory data for Unity visualization.
        
        Args:
            trajectory_points: List of [x, y, heading] points
            reference_point: [x, y, heading, velocity] reference point
            lateral_error: Current lateral error for color coding
        
        Returns:
            True if successful, False otherwise
        """
        try:
            trajectory_data = {
                "trajectory_points": trajectory_points,
                "reference_point": reference_point,
                "lateral_error": float(lateral_error),
                "timestamp": time.time()
            }
            if perception_left_lane_x is not None:
                trajectory_data["perception_left_lane_x"] = float(perception_left_lane_x)
            if perception_right_lane_x is not None:
                trajectory_data["perception_right_lane_x"] = float(perception_right_lane_x)
            if perception_center_x is not None:
                trajectory_data["perception_center_x"] = float(perception_center_x)
            if perception_lookahead_m is not None:
                trajectory_data["perception_lookahead_m"] = float(perception_lookahead_m)
            if perception_valid is not None:
                trajectory_data["perception_valid"] = bool(perception_valid)
            
            response = self.session.post(
                f"{self.base_url}/api/trajectory",
                json=trajectory_data
            )
            response.raise_for_status()
            return True
        except requests.RequestException:
            # Don't log errors - trajectory visualization is optional
            return False
    
    def health_check(self) -> bool:
        """
        Check if bridge server is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/api/health", timeout=1.0)
            response.raise_for_status()
            return True
        except requests.RequestException:
            return False
    
    def signal_shutdown(self) -> bool:
        """
        Signal to Unity that the AV stack is shutting down.
        Unity will exit play mode when it detects this signal.
        
        Returns:
            True if signal was sent successfully, False otherwise
        """
        try:
            response = self.session.post(f"{self.base_url}/api/shutdown", timeout=2.0)
            response.raise_for_status()
            result = response.json()
            if result.get("status") == "shutdown":
                return True
            return False
        except requests.RequestException as e:
            print(f"Error signaling shutdown: {e}")
            return False


if __name__ == "__main__":
    # Test client
    client = UnityBridgeClient()
    
    if client.health_check():
        print("Bridge server is healthy")
        
        # Get camera frame
        frame_data = client.get_latest_camera_frame()
        if frame_data:
            img, timestamp = frame_data
            print(f"Got camera frame: {img.shape}, timestamp: {timestamp}")
        
        # Get vehicle state
        state = client.get_latest_vehicle_state()
        if state:
            print(f"Got vehicle state: speed={state.get('speed', 0):.2f} m/s")
        
        # Set control command
        client.set_control_command(steering=0.1, throttle=0.5, brake=0.0)
        print("Set control command")
    else:
        print("Bridge server is not available")

