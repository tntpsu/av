"""
Ground truth following test driver.

This script collects ground truth data by driving the car along the ground truth
lane center path. It uses ground truth lane positions from Unity to compute steering
commands, allowing the car to follow the center line automatically.

Purpose:
    1. Drive car along ground truth path (using GT steering)
    2. Record all data: camera frames, vehicle state, ground truth lane positions
    3. Later replay this data with replay_perception.py to test your perception stack

Why steering is needed:
    Unity is a physics simulation - the car won't move without control commands.
    This script computes steering from ground truth lane center to make the car
    follow the path, while recording everything for later analysis.

Workflow:
    1. Run this script: python tools/ground_truth_follower.py --duration 60
    2. Car follows ground truth center line and records data
    3. Replay data: python tools/replay_perception.py <recording_file>
    4. Compare your perception output vs ground truth

Usage:
    python tools/ground_truth_follower.py [--duration 60] [--output recording_name]
"""

import argparse
import time
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import sys
import subprocess
import threading
import signal
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from av_stack import AVStack
from data.recorder import DataRecorder
from data.formats.data_format import ControlCommand
from typing import Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from bridge.client import UnityBridgeClient
except ImportError:
    # Fallback if bridge.client is not available
    UnityBridgeClient = None

# Configure logging to both console and file for debugging
log_dir = Path(__file__).parent.parent / 'tmp' / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / 'ground_truth_follower.log'


def configure_logging(level: int) -> logging.Logger:
    file_handler = logging.FileHandler(str(log_file), mode='a')
    file_handler.setLevel(level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[console_handler, file_handler],
        force=True
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to console and file: {log_file}")
    file_handler.flush()
    return logger


logger = configure_logging(logging.ERROR)

def _normalize_target_lane(target_lane: str) -> str:
    lane = (target_lane or "right").strip().lower()
    if lane not in {"right", "left", "center"}:
        raise ValueError(f"Invalid target lane: {target_lane!r}. Use right, left, or center.")
    return lane


def compute_target_lane_center(
    left_lane_line_x: Optional[float],
    right_lane_line_x: Optional[float],
    fallback_center: float,
    target_lane: str,
    single_lane_width_threshold_m: float = 5.0,
) -> float:
    """
    Compute the desired lane center (right/left/center) from ground-truth lane lines.

    Args:
        left_lane_line_x: Left lane line lateral position in vehicle coords.
        right_lane_line_x: Right lane line lateral position in vehicle coords.
        fallback_center: Road center fallback if lane lines are unavailable.
        target_lane: "right", "left", or "center".
    """
    if left_lane_line_x is None or right_lane_line_x is None:
        return fallback_center
    if not np.isfinite(left_lane_line_x) or not np.isfinite(right_lane_line_x):
        return fallback_center
    lane_width = right_lane_line_x - left_lane_line_x
    if lane_width <= 1e-6:
        return fallback_center
    road_center = (left_lane_line_x + right_lane_line_x) / 2.0
    if lane_width < single_lane_width_threshold_m:
        return road_center
    target_lane = _normalize_target_lane(target_lane)
    if target_lane == "right":
        return road_center + (lane_width * 0.25)
    if target_lane == "left":
        return road_center - (lane_width * 0.25)
    return road_center


def resolve_segmentation_settings(
    use_cv: bool, segmentation_checkpoint: Optional[str]
) -> tuple[bool, Optional[str]]:
    use_segmentation = not use_cv
    if use_segmentation:
        if not segmentation_checkpoint:
            raise FileNotFoundError(
                "Segmentation checkpoint not provided. Use --segmentation-checkpoint or --use-cv."
            )
        if not Path(segmentation_checkpoint).exists():
            raise FileNotFoundError(
                f"Segmentation checkpoint not found: {segmentation_checkpoint}"
            )
    return use_segmentation, segmentation_checkpoint


def resolve_lateral_correction_gain(
    target_lane: str,
    provided_gain: Optional[float],
    default_center: float = 0.2,
    default_offset: float = 0.8,
) -> float:
    target_lane = _normalize_target_lane(target_lane)
    if provided_gain is not None:
        return float(provided_gain)
    return default_offset if target_lane != "center" else default_center


class GroundTruthFollower:
    """
    Drive car using ground truth data to collect ground truth recordings.
    
    This class:
    - Uses ground truth lane center from Unity to compute steering commands
    - Drives the car along the ground truth path (Unity needs steering to move)
    - Records all data: camera frames, vehicle state, ground truth lane positions
    - Records perception output for comparison
    
    The recorded data can then be replayed with replay_perception.py to test
    your perception stack against ground truth without needing Unity running.
    """
    
    def __init__(
        self,
        bridge_url: str = "http://localhost:8000",
        target_speed: float = 5.0,  # Constant speed
        lookahead: float = 8.0,
        start_bridge_server: bool = True,
        max_safe_speed: float = 10.0,  # Maximum safe speed
        speed_kp: float = 0.15,  # Speed control proportional gain
        speed_ki: float = 0.02,  # Speed control integral gain
        speed_kd: float = 0.05,  # Speed control derivative gain
        random_start: bool = False,
        random_seed: Optional[int] = None,
        constant_speed: bool = False,
        target_lane: str = "right",
        use_segmentation: bool = True,
        segmentation_model_path: Optional[str] = None,
        single_lane_width_threshold_m: float = 5.0,
        lateral_correction_gain: Optional[float] = None,
    ):
        """
        Initialize ground truth follower.
        
        Args:
            bridge_url: Unity bridge URL
            target_speed: Constant target speed (m/s)
            lookahead: Lookahead distance for ground truth following
            start_bridge_server: If True, start bridge server automatically
            max_safe_speed: Maximum safe speed before emergency brake (m/s)
            speed_kp: Speed control proportional gain (tuned for Unity physics)
            speed_ki: Speed control integral gain
            speed_kd: Speed control derivative gain
        """
        if UnityBridgeClient is None:
            raise ImportError("UnityBridgeClient not available. Check bridge/client.py")
        self.bridge = UnityBridgeClient(bridge_url)
        self.target_speed = target_speed
        self.lookahead = lookahead
        self.max_safe_speed = max_safe_speed
        self.bridge_server_process = None
        self.start_bridge_server = start_bridge_server
        self.constant_speed = constant_speed
        self.target_lane = _normalize_target_lane(target_lane)
        self.single_lane_width_threshold_m = float(single_lane_width_threshold_m)
        
        # Speed control PID parameters (tuned for Unity physics)
        # Unity: motorForce=400, brake uses damping + velocity reduction
        # These gains are tuned for smooth speed control at ~5 m/s
        self.speed_kp = speed_kp
        self.speed_ki = speed_ki
        self.speed_kd = speed_kd
        self.speed_integral = 0.0
        self.prev_speed_error = 0.0
        self.prev_speed_time = None
        
        # Initialize AV stack (for perception and recording)
        self.av_stack = AVStack(
            bridge_url=bridge_url,
            record_data=True,
            recording_dir="data/recordings",
            use_segmentation=use_segmentation,
            segmentation_model_path=segmentation_model_path,
        )
        
        # Mark as ground truth drive recording (for creating test recordings)
        if self.av_stack.recorder:
            self.av_stack.recorder.recording_type = "gt_drive"
            self.av_stack.recorder.metadata["recording_type"] = "gt_drive"
        
        # Speed-adaptive steering control
        # Base gain - will be adjusted based on speed
        self.kp_steering_base = 1.5  # Base proportional gain
        self.kp_steering = self.kp_steering_base  # Will be updated based on speed
        
        # Speed-dependent gain adjustment
        # At higher speeds, reduce gain to prevent oversteering
        # At lower speeds, can use higher gain for better tracking
        self.speed_gain_factor = 0.8  # Reduce gain by this factor per m/s above base speed
        self.base_speed = 3.0  # Base speed for gain calculation (m/s)
        
        # EXACT STEERING CALCULATION: Car physical properties (from Unity)
        # These are defaults and will be overridden by Unity vehicle state when available.
        self.wheelbase = 2.5  # meters (estimated from car length)
        self.max_steer_angle_deg = 30.0  # degrees (Unity default)
        self.max_steer_angle_rad = np.deg2rad(self.max_steer_angle_deg)
        self.unity_fixed_delta_time = 0.02  # Typical Unity FixedUpdate rate (50 Hz)
        self.vehicle_params_source = "defaults"
        
        # Steering calculation mode
        self.use_exact_steering = True  # Use exact calculation from path geometry
        self.use_proportional_steering = False  # Fallback to proportional if exact fails
        
        # Optional derivative term for stability (helps prevent overshoot)
        self.kd_steering = 0.3  # Reduced derivative gain
        self.prev_lane_center = 0.0
        self.prev_steering_time = None
        
        # Lookahead distance - look ahead along the path based on speed
        # Higher speed = longer lookahead to anticipate curves
        self.lookahead_base = 5.0  # Base lookahead distance (meters)
        self.lookahead_speed_factor = 1.5  # Additional lookahead per m/s
        
        # Sign correction: if car steers away from center, flip this to -1
        # Positive = use lane_center directly (center to right → steer right)
        # Negative = negate lane_center (center to right → steer left)
        self.steering_sign = 1.0
        
        # Steering smoothing - reduced for faster response
        self.last_steering = 0.0
        self.steering_smoothing = 0.3  # Lower = more responsive (was 0.7, too slow)
        
        # Deadband - reduced to allow corrections when lane gets straight
        self.steering_deadband = 0.02  # Reduced from 0.05m - allow smaller corrections
        
        # EXACT STEERING CALCULATION: Car physical properties (from Unity)
        # Car scale: {x: 1.85, y: 1, z: 4} -> car length ~4m
        # Typical wheelbase is ~60-70% of car length
        self.wheelbase = 2.5  # meters (estimated from car length)
        self.max_steer_angle_deg = 30.0  # degrees (from Unity: maxSteerAngle = 30f)
        self.max_steer_angle_rad = np.deg2rad(self.max_steer_angle_deg)
        
        # Unity ground truth mode: steerAngle = steerInput * maxSteerAngle
        # Then rotates by: steerAngle * Time.fixedDeltaTime (typically 0.02s)
        # So: rotation_rate = steerInput * maxSteerAngle * (1 / fixedDeltaTime)
        # For exact calculation, we need to know Unity's fixedDeltaTime
        self.unity_fixed_delta_time = 0.02  # Typical Unity FixedUpdate rate (50 Hz)
        
        # Steering calculation mode
        # ALWAYS use exact calculation - we have all the information we need!
        self.use_exact_steering = True  # Use exact calculation from path geometry
        
        # Lateral error correction gain (small, just to stay centered)
        # Since path curvature handles the main steering, this is just a small correction
        self.lateral_correction_gain = resolve_lateral_correction_gain(
            self.target_lane,
            lateral_correction_gain,
        )

        # Curvature-based speed reduction and saturation fallback
        self.min_target_speed = 2.0
        self.curvature_speed_gain = 1.5  # Higher = reduce speed more on curves
        self.steering_saturation_threshold = 0.9
        self.steering_saturation_frames = 0
        self.steering_saturation_limit = 3
        self._speed_scale = 1.0
        
        # Randomized start support
        self.random_start = random_start
        self.random_seed = random_seed
        self.randomize_request_id = 0
        self.randomize_pending = False
        
        # Start bridge server if requested
        if self.start_bridge_server:
            self._start_bridge_server()
    
    def _start_bridge_server(self):
        """Start the bridge server in a separate process."""
        try:
            # Check if server is already running
            try:
                if self.bridge.health_check():
                    logger.info("Bridge server is already running")
                    return
            except:
                pass  # Server not running, continue to start it
            
            logger.info("Starting bridge server...")
            # Start bridge server as subprocess
            project_root = Path(__file__).parent.parent
            server_script = project_root / "bridge" / "server.py"
            
            # Log bridge server output to file for debugging
            log_dir = project_root / 'tmp' / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            bridge_log = log_dir / 'bridge_server.log'
            
            self.bridge_server_process = subprocess.Popen(
                [sys.executable, "-m", "bridge.server"],
                cwd=str(project_root),
                stdout=open(bridge_log, 'a'),
                stderr=subprocess.STDOUT
            )
            
            # Wait a moment for server to start
            time.sleep(2)
            
            # Verify it's running
            try:
                if self.bridge.health_check():
                    logger.info("✓ Bridge server started successfully")
                else:
                    logger.warning("Bridge server may not have started correctly")
                    logger.warning(f"Check logs: {bridge_log}")
            except Exception as e:
                logger.warning(f"Could not verify bridge server: {e}")
                logger.warning(f"Check logs: {bridge_log}")
        except Exception as e:
            logger.error(f"Failed to start bridge server: {e}", exc_info=True)
            logger.error("You may need to start it manually: python -m bridge.server")
    
    def _stop_bridge_server(self):
        """Stop the bridge server if we started it."""
        if self.bridge_server_process:
            try:
                logger.info("Stopping bridge server...")
                self.bridge_server_process.terminate()
                self.bridge_server_process.wait(timeout=5)
                logger.info("✓ Bridge server stopped")
            except subprocess.TimeoutExpired:
                logger.warning("Bridge server didn't stop gracefully, killing...")
                self.bridge_server_process.kill()
            except Exception as e:
                logger.warning(f"Error stopping bridge server: {e}")

    @staticmethod
    def _get_state_value(vehicle_state: dict, keys: list[str], default=None):
        for key in keys:
            if key in vehicle_state and vehicle_state[key] is not None:
                return vehicle_state[key]
        nested = vehicle_state.get("ground_truth")
        if isinstance(nested, dict):
            for key in keys:
                if key in nested and nested[key] is not None:
                    return nested[key]
        return default

    def _get_target_lane_center(self, vehicle_state: dict) -> float:
        lane_center = self._get_state_value(
            vehicle_state,
            ["groundTruthLaneCenterX", "ground_truth_lane_center_x", "lane_center_x"],
            0.0,
        )
        left_lane_line_x = self._get_state_value(
            vehicle_state,
            [
                "groundTruthLeftLaneLineX",
                "ground_truth_left_lane_line_x",
                "groundTruthLeftLaneX",
                "ground_truth_left_lane_x",
            ],
            None,
        )
        right_lane_line_x = self._get_state_value(
            vehicle_state,
            [
                "groundTruthRightLaneLineX",
                "ground_truth_right_lane_line_x",
                "groundTruthRightLaneX",
                "ground_truth_right_lane_x",
            ],
            None,
        )
        return compute_target_lane_center(
            left_lane_line_x,
            right_lane_line_x,
            float(lane_center),
            self.target_lane,
            self.single_lane_width_threshold_m,
        )
    
    def _calculate_exact_steering(self, vehicle_state: dict, current_speed: float) -> float:
        """
        Calculate EXACT steering angle using bicycle model and path geometry.
        
        This is the ONLY steering calculation we need - no tuning, no guessing!
        
        Formula:
            steering_angle = atan(wheelbase * path_curvature)
            steerInput = steering_angle / maxSteerAngle
        
        Plus a small lateral correction to stay centered in the lane.
        
        Args:
            vehicle_state: Vehicle state dictionary
            current_speed: Current vehicle speed (m/s) - not used, but kept for compatibility
            
        Returns:
            Exact steering command (-1.0 to 1.0)
        """
        # Get path curvature from Unity (this is the EXACT path geometry)
        path_curvature = (
            vehicle_state.get('groundTruthPathCurvature') or
            vehicle_state.get('ground_truth_path_curvature') or
            0.0
        )
        
        # Get lateral error (to stay centered in target lane)
        lane_center = self._get_target_lane_center(vehicle_state)
        
        # EXACT STEERING FROM PATH CURVATURE (bicycle model)
        # This is the mathematically exact steering needed to follow the path
        if abs(path_curvature) > 1e-6:
            # Calculate exact steering angle from curvature
            # Bicycle model: steering_angle = atan(wheelbase * curvature)
            steering_angle_rad = np.arctan(self.wheelbase * path_curvature)
            
            # Convert to Unity steering input [-1, 1]
            # Unity: steerInput * maxSteerAngle = actual steering angle
            # So: steerInput = steering_angle_rad / maxSteerAngle_rad
            exact_steering = steering_angle_rad / self.max_steer_angle_rad
            
            # Add small lateral correction to stay centered in lane
            # This is just a small adjustment - path curvature handles the main steering
            lateral_correction = self.lateral_correction_gain * lane_center
            exact_steering = exact_steering + lateral_correction
            
            # Clip to valid range
            exact_steering = np.clip(exact_steering, -1.0, 1.0)
            
            return exact_steering
        else:
            # Straight path (curvature = 0) - only need lateral correction
            # For straight path, just steer toward center
            if abs(lane_center) > self.steering_deadband:
                exact_steering = self.lateral_correction_gain * lane_center
                exact_steering = np.clip(exact_steering, -1.0, 1.0)
                return exact_steering
            else:
                return 0.0

    def _update_vehicle_params(self, vehicle_state: dict) -> None:
        """Update vehicle parameters from Unity state if available."""
        try:
            max_steer = float(vehicle_state.get("maxSteerAngle", 0.0))
            wheelbase = float(vehicle_state.get("wheelbaseMeters", 0.0))
            fixed_dt = float(vehicle_state.get("fixedDeltaTime", 0.0))
        except (TypeError, ValueError):
            return
        
        if max_steer > 0.0:
            self.max_steer_angle_deg = max_steer
            self.max_steer_angle_rad = np.deg2rad(self.max_steer_angle_deg)
        if wheelbase > 0.0:
            self.wheelbase = wheelbase
        if fixed_dt > 0.0:
            self.unity_fixed_delta_time = fixed_dt
        
        if max_steer > 0.0 or wheelbase > 0.0 or fixed_dt > 0.0:
            self.vehicle_params_source = "unity_state"

    @staticmethod
    def _compute_speed_scale(curvature: float, gain: float, min_scale: float) -> float:
        """Compute a speed scale factor based on path curvature."""
        if curvature <= 0.0:
            return 1.0
        scale = 1.0 - (curvature * gain)
        return float(np.clip(scale, min_scale, 1.0))
    
    def get_ground_truth_steering(self, vehicle_state: dict) -> float:
        """
        Get steering command from ground truth.
        
        Uses speed-adaptive proportional-derivative control with lookahead.
        Steering gain is adjusted based on speed: higher speed = lower gain.
        Uses lookahead to anticipate curves and adjust steering accordingly.
        
        Args:
            vehicle_state: Vehicle state dictionary (from AV stack)
            
        Returns:
            Steering command (-1.0 to 1.0)
        """
        try:
            # Get current speed for adaptive control
            current_speed = vehicle_state.get('speed', 0.0) or vehicle_state.get('velocity', {}).get('magnitude', 0.0) or 0.0
            if isinstance(current_speed, (list, np.ndarray)):
                current_speed = np.linalg.norm(current_speed) if len(current_speed) > 0 else 0.0
            current_speed = max(0.1, float(current_speed))  # Minimum 0.1 m/s to avoid division by zero
            
            # Update vehicle parameters from Unity state (max steer, wheelbase, fixed dt)
            self._update_vehicle_params(vehicle_state)
            
            # Adjust gain based on speed
            # At higher speeds, reduce gain to prevent oversteering
            # Formula: kp = base_kp * (1 - speed_factor * (speed - base_speed) / base_speed)
            # Clamp to reasonable range (0.3 to 1.5x base)
            speed_diff = current_speed - self.base_speed
            speed_factor = 1.0 - self.speed_gain_factor * speed_diff / max(self.base_speed, 0.1)
            speed_factor = np.clip(speed_factor, 0.3, 1.5)  # Clamp between 0.3 and 1.5
            self.kp_steering = self.kp_steering_base * speed_factor
            
            # Calculate lookahead distance based on speed
            # Higher speed = longer lookahead to anticipate curves
            lookahead_distance = self.lookahead_base + self.lookahead_speed_factor * current_speed
            
            # Ground truth comes from Unity in vehicle_state_dict
            lane_center = self._get_target_lane_center(vehicle_state)
            
            # NEW: Get path heading for path-based steering
            desired_heading = (
                vehicle_state.get('groundTruthDesiredHeading') or
                vehicle_state.get('ground_truth_desired_heading') or
                vehicle_state.get('ground_truth', {}).get('desired_heading') or
                None
            )
            
            # Get current car heading
            current_heading = vehicle_state.get('heading', 0.0)
            if isinstance(current_heading, dict):
                # If heading is in a dict, extract it
                current_heading = current_heading.get('y', 0.0) if 'y' in current_heading else 0.0
            
            # Compute heading error if we have desired heading
            heading_error = None
            if desired_heading is not None:
                heading_error = desired_heading - current_heading
                # Normalize to [-180, 180]
                while heading_error > 180:
                    heading_error -= 360
                while heading_error < -180:
                    heading_error += 360
            
            # SELF-DIAGNOSTIC: Check if steering is correct
            current_time = time.time()
            # (We'll call this after computing steering, but need to prepare data)
            
            # CALCULATE EXACT STEERING - that's all we need!
            # We have all the information: path curvature, car properties, physics model
            # No tuning, no guessing, no auto-correction needed!
            raw_steering = self._calculate_exact_steering(vehicle_state, current_speed)
            
            # Calculate steering angle in degrees for recording
            calculated_steering_angle_deg = raw_steering * self.max_steer_angle_deg
            
            # Get path curvature and lateral correction for recording
            path_curvature = vehicle_state.get('groundTruthPathCurvature', 0.0) or vehicle_state.get('ground_truth_path_curvature', 0.0)
            lateral_correction = self.lateral_correction_gain * lane_center
            
            # Curvature-based speed reduction (applied in speed controller)
            self._speed_scale = self._compute_speed_scale(abs(path_curvature), self.curvature_speed_gain, 0.4)
            
            # Saturation fallback: reduce speed if steering saturates
            if abs(raw_steering) >= self.steering_saturation_threshold:
                self.steering_saturation_frames += 1
            else:
                self.steering_saturation_frames = max(0, self.steering_saturation_frames - 1)
            
            if self.steering_saturation_frames >= self.steering_saturation_limit:
                self._speed_scale = min(self._speed_scale, 0.6)

            # Light smoothing to prevent sudden changes (optional, but helps stability)
            steering = (self.steering_smoothing * raw_steering + 
                       (1.0 - self.steering_smoothing) * self.last_steering)
            self.last_steering = steering
            
            # Store for recording (will be added to ControlCommand)
            self._last_raw_steering = raw_steering
            self._last_calculated_steering_angle_deg = calculated_steering_angle_deg
            self._last_lateral_correction = lateral_correction
            self._last_path_curvature_input = path_curvature
            
            # Debug logging (every 10 frames for better visibility)
            if not hasattr(self, '_debug_frame_count'):
                self._debug_frame_count = 0
            self._debug_frame_count += 1
            if self._debug_frame_count % 10 == 0:
                speed_factor = getattr(self, '_last_speed_factor', 1.0)
                # Log exact steering calculation
                path_curvature = vehicle_state.get('groundTruthPathCurvature', 0.0)
                logger.info(f"GT Steering [EXACT]: speed={current_speed:.2f}m/s, "
                           f"curvature={path_curvature:.4f} (1/m), "
                           f"lane_center={lane_center:.3f}m, "
                           f"steering={steering:.3f} (raw={raw_steering:.3f}), "
                           f"speed_scale={self._speed_scale:.2f}, params={self.vehicle_params_source}")
                
                # CRITICAL: Verify coordinate system interpretation
                if abs(lane_center) > 0.1:  # Only log when there's a significant offset
                    expected_direction = "RIGHT" if lane_center > 0 else "LEFT"
                    steering_direction = "RIGHT" if steering > 0 else "LEFT"
                    logger.info(f"  → Car is {abs(lane_center):.2f}m {'LEFT' if lane_center > 0 else 'RIGHT'} of center, "
                               f"steering {steering_direction} (expected: {expected_direction})")
            
            return steering
        except Exception as e:
            logger.error(f"Error getting ground truth steering: {e}")
            return 0.0
    
    def _wait_for_bridge(self, max_retries: int = 10, initial_delay: float = 1.0) -> bool:
        """Wait for bridge server to be available with exponential backoff."""
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                if self.bridge.health_check():
                    return True
            except:
                pass
            if attempt % 3 == 0:  # Log every 3 attempts
                logger.info(f"Waiting for bridge server... (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
            delay = min(delay * 1.5, 5.0)  # Exponential backoff, max 5s
        return False
    
    def _wait_for_first_frame(self, max_retries: int = 30, initial_delay: float = 0.5) -> bool:
        """Wait for first frame from Unity with exponential backoff."""
        delay = initial_delay
        for attempt in range(max_retries):
            frame_data = self.bridge.get_latest_camera_frame()
            if frame_data is not None:
                return True
            if attempt % 5 == 0:  # Log every 5 attempts
                logger.info(f"Waiting for Unity frames... (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
            delay = min(delay * 1.2, 2.0)  # Exponential backoff, max 2s
        return False
    
    def run(self, duration: float = 60.0, recording_name: Optional[str] = None, auto_play: bool = False):
        """
        Run ground truth following test.
        
        Args:
            duration: Test duration in seconds
            recording_name: Optional name for recording
            auto_play: If True, request Unity to start playing automatically (requires Unity Editor)
        """
        logger.info(f"Starting ground truth following test (duration: {duration}s)")
        logger.info(f"Target speed: {self.target_speed} m/s")
        logger.info(f"Lookahead: {self.lookahead} m")
        logger.info("Unity reminder: Ensure the CarController 'Ground Truth Mode' checkbox is enabled "
                    "if you want GT mode active before the first control command.")
        
        if self.random_start:
            self.randomize_request_id = int(time.time() * 1000)
            self.randomize_pending = True
            logger.info(
                "Random start enabled: will request Unity to randomize oval start position."
            )
        
        # Request Unity to start playing if requested
        if auto_play:
            logger.info("Requesting Unity to start playing...")
            if self.bridge.request_unity_play():
                logger.info("Play request sent to Unity. Waiting for Unity to enter play mode...")
                # Wait a bit for Unity to enter play mode
                time.sleep(2.0)
            else:
                logger.warning("Failed to send play request to Unity. Unity may not be ready yet.")
        
        # Override recording name if provided (recreate recorder so filename is correct)
        if recording_name and self.av_stack.recorder:
            try:
                output_dir = str(self.av_stack.recorder.output_dir)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                full_name = f"{recording_name}_{timestamp}"
                # Close the original recorder (created at AVStack init)
                self.av_stack.recorder.close()
                self.av_stack.recorder = DataRecorder(
                    output_dir=output_dir,
                    recording_name=full_name,
                    recording_type="gt_drive",
                )
                self.av_stack.recorder.metadata["recording_type"] = "gt_drive"
                logger.info(f"Using recording name: {full_name}")
            except Exception as e:
                logger.error(f"Failed to reset recorder name: {e}", exc_info=True)
        
        # Wait for bridge connection (like normal AV stack)
        logger.info("Waiting for Unity bridge...")
        if not self._wait_for_bridge(max_retries=10, initial_delay=1.0):
            logger.error("Bridge server is not available!")
            logger.error("Please start Unity first, then run this script.")
            return
        
        logger.info("Bridge connected. Waiting for first frame...")
        if not self._wait_for_first_frame(max_retries=30, initial_delay=0.5):
            logger.warning("No frames received from Unity. Starting anyway...")
        else:
            logger.info("First frame received! Starting ground truth following...")
        
        # Enable ground truth mode in Unity (direct velocity control)
        logger.info("Enabling ground truth mode (direct velocity control) in Unity...")
        # This will be set via control commands with ground_truth_mode flag
        
        # Safety: Check initial vehicle state
        initial_state = self.bridge.get_latest_vehicle_state()
        if initial_state:
            initial_speed = initial_state.get('speed', 0.0)
            if initial_speed > self.max_safe_speed:
                logger.warning(f"Initial speed {initial_speed:.2f} m/s is high (max: {self.max_safe_speed:.2f} m/s).")
                logger.warning("Will apply brake to slow down. If car doesn't slow, reset Unity.")
            elif initial_speed > self.target_speed:
                logger.info(f"Initial speed {initial_speed:.2f} m/s. Will brake to target speed {self.target_speed:.2f} m/s.")
        
        start_time = time.time()
        frame_count = 0
        last_frame_time = time.time()
        consecutive_failures = 0
        max_consecutive_failures = 100  # ~3 seconds at 30ms sleep
        no_frame_timeout = 5.0  # If no frames for 5 seconds, warn
        
        try:
            while time.time() - start_time < duration:
                # Check for connection health
                try:
                    health_ok = self.bridge.health_check()
                except Exception as e:
                    logger.warning(f"Bridge health check exception: {e}")
                    health_ok = False
                
                if not health_ok:
                    consecutive_failures += 1
                    
                    # Check if bridge server process is still alive (if we started it)
                    if self.bridge_server_process is not None:
                        if self.bridge_server_process.poll() is not None:
                            # Process has died
                            logger.error(f"Bridge server process died! Exit code: {self.bridge_server_process.returncode}")
                            logger.error("This usually means the bridge server crashed.")
                            logger.error("Check logs: tmp/logs/bridge_server.log")
                            
                            # Try to restart if we started it
                            if self.start_bridge_server and consecutive_failures <= 3:
                                logger.info("Attempting to restart bridge server...")
                                self._start_bridge_server()
                                time.sleep(2)  # Wait for server to start
                                # Check again
                                try:
                                    if self.bridge.health_check():
                                        logger.info("✓ Bridge server restarted successfully!")
                                        consecutive_failures = 0
                                        continue
                                except:
                                    pass
                    
                    if consecutive_failures == 1:
                        logger.warning("Bridge health check failed! Bridge server may have crashed.")
                    elif consecutive_failures % 10 == 0:
                        logger.warning(f"Bridge health check failed ({consecutive_failures} consecutive failures)")
                    
                    if consecutive_failures > max_consecutive_failures:
                        logger.error("Too many consecutive failures. Bridge server may have crashed.")
                        logger.error("Check if bridge server is running: python -m bridge.server")
                        logger.error("Or check logs: tmp/logs/bridge_server.log")
                        logger.error("Stopping ground truth follower.")
                        break
                    time.sleep(0.1)
                    continue
                
                consecutive_failures = 0  # Reset on successful health check
                
                # Get current frame
                frame_data = self.bridge.get_latest_camera_frame()
                if frame_data is None:
                    # Check if we've been waiting too long for frames
                    if time.time() - last_frame_time > no_frame_timeout:
                        logger.warning(f"No frames received for {no_frame_timeout}s. Unity may have stopped.")
                        logger.warning("Check Unity console for errors.")
                    time.sleep(0.01)
                    continue
                
                last_frame_time = time.time()  # Update last frame time
                image, timestamp, _ = frame_data
                
                # Get vehicle state
                vehicle_state = self.bridge.get_latest_vehicle_state()
                if vehicle_state is None:
                    logger.warning("Got frame but no vehicle state. Retrying...")
                    time.sleep(0.01)
                    continue
                
                # Safety checks before processing
                current_speed = vehicle_state.get('speed', 0.0)
                lane_center = self._get_target_lane_center(vehicle_state)
                
                # Safety: If speed is very high, apply maximum brake (don't stop script)
                if current_speed > self.max_safe_speed:
                    logger.warning(f"Speed {current_speed:.2f} m/s exceeds safe limit {self.max_safe_speed:.2f} m/s! Applying maximum brake.")
                    # Will be handled in speed control below
                
                # Compute steering from ground truth lane center
                # This makes the car follow the ground truth path (Unity needs steering commands to move)
                steering = self.get_ground_truth_steering(vehicle_state)
                
                # PID speed control: maintain target speed (with curvature/saturation scaling)
                # Tuned for Unity physics: motorForce=400, brake uses damping + velocity reduction
                effective_target_speed = max(self.min_target_speed, self.target_speed * self._speed_scale)
                speed_error = effective_target_speed - current_speed
                
                if self.constant_speed:
                    throttle = 1.0
                    brake = 0.0
                # Emergency: speed way too high - maximum brake
                elif current_speed > self.max_safe_speed:
                    throttle = 0.0
                    brake = 1.0
                    # Reset PID to prevent windup
                    self.speed_integral = 0.0
                    self.prev_speed_error = speed_error
                else:
                    # PID control
                    dt = 1.0 / 30.0  # Assume ~30 Hz update rate
                    
                    # Proportional term
                    p_term = self.speed_kp * speed_error
                    
                    # Integral term (with anti-windup)
                    self.speed_integral += speed_error * dt
                    # Limit integral to prevent windup
                    max_integral = 2.0  # Max integral contribution
                    self.speed_integral = np.clip(self.speed_integral, -max_integral, max_integral)
                    i_term = self.speed_ki * self.speed_integral
                    
                    # Derivative term
                    if self.prev_speed_time is not None:
                        actual_dt = time.time() - self.prev_speed_time
                        if actual_dt > 0:
                            d_error = (speed_error - self.prev_speed_error) / actual_dt
                            d_term = self.speed_kd * d_error
                        else:
                            d_term = 0.0
                    else:
                        d_term = 0.0
                    
                    # PID output
                    pid_output = p_term + i_term + d_term
                    
                    # Convert PID output to throttle/brake
                    # Positive output = need to accelerate (throttle)
                    # Negative output = need to decelerate (brake)
                    if pid_output > 0:
                        throttle = np.clip(pid_output, 0.0, 0.6)  # Max throttle 0.6 for smooth control
                        brake = 0.0
                    else:
                        throttle = 0.0
                        brake = np.clip(abs(pid_output), 0.0, 1.0)  # Max brake 1.0
                    
                    # Update for next iteration
                    self.prev_speed_error = speed_error
                    self.prev_speed_time = time.time()
                
                # Store ground truth control for recording
                # We'll modify the recording to use this instead of computed control
                if not hasattr(self.av_stack, '_gt_control_override'):
                    self.av_stack._gt_control_override = None
                
                # Create ControlCommand object for override (matches what AV stack expects)
                # Include exact steering calculation breakdown for verification
                control_command_obj = ControlCommand(
                    timestamp=timestamp,
                    steering=float(steering),
                    throttle=float(throttle),
                    brake=float(brake),
                    calculated_steering_angle_deg=getattr(self, '_last_calculated_steering_angle_deg', None),
                    raw_steering=getattr(self, '_last_raw_steering', None),
                    lateral_correction=getattr(self, '_last_lateral_correction', None),
                    path_curvature_input=getattr(self, '_last_path_curvature_input', None)
                )
                self.av_stack._gt_control_override = control_command_obj
                
                # Also create dict for sending to Unity
                control_command_dict = {
                    'steering': float(steering),
                    'throttle': float(throttle),
                    'brake': float(brake)
                }
                if self.randomize_pending:
                    control_command_dict['randomize_start'] = True
                    control_command_dict['randomize_request_id'] = self.randomize_request_id
                    if self.random_seed is not None:
                        control_command_dict['randomize_seed'] = int(self.random_seed)
                    logger.info(
                        "Requesting random start: request_id=%s, seed=%s",
                        self.randomize_request_id,
                        self.random_seed,
                    )
                
                # Process frame through AV stack to run perception and record data
                # This computes perception output (lane detection) and records everything
                # We override the control command with ground truth steering to follow GT path
                # The recorded perception output can be compared to GT later
                try:
                    # Process frame normally (perception, trajectory, recording)
                    # This will compute control but we'll override it
                    self.av_stack._process_frame(
                        image,
                        timestamp,
                        vehicle_state
                    )
                    # Keep AVStack frame counter in sync to avoid repeated "Frame 0" logs.
                    self.av_stack.frame_count += 1
                    
                    # Send our ground truth control command (not the computed one)
                    # Enable ground truth mode for direct velocity control
                    control_success = self.bridge.set_control_command(
                        control_command_dict['steering'],
                        control_command_dict['throttle'],
                        control_command_dict['brake'],
                        ground_truth_mode=True,
                        ground_truth_speed=effective_target_speed,
                        randomize_start=control_command_dict.get('randomize_start', False),
                        randomize_request_id=control_command_dict.get('randomize_request_id', 0),
                        randomize_seed=control_command_dict.get('randomize_seed')
                    )
                    if control_success and self.randomize_pending:
                        self.randomize_pending = False
                    
                    if not control_success:
                        logger.warning("Failed to send control command to Unity. Connection may be lost.")
                    
                    # Clear override
                    self.av_stack._gt_control_override = None
                except KeyboardInterrupt:
                    # Re-raise keyboard interrupt to allow clean shutdown
                    raise
                except Exception as e:
                    logger.error(f"Error processing frame {frame_count}: {e}", exc_info=True)
                    # Check for critical errors that should stop the script
                    error_str = str(e).lower()
                    if any(keyword in error_str for keyword in ["recorder", "h5", "hdf5", "file", "disk", "memory"]):
                        logger.error("Critical error detected (recording/file system). Stopping.")
                        break
                    # For other errors, continue but log
                    logger.warning(f"Continuing after error (frame {frame_count})...")
                    # CRITICAL: Still send control command with ground truth mode enabled!
                    # If we don't, Unity will disable ground truth mode when it receives
                    # a command without ground_truth_mode=True
                    try:
                        # Use the dict version for sending to Unity
                        if 'control_command_dict' in locals():
                            self.bridge.set_control_command(
                                control_command_dict['steering'],
                                control_command_dict['throttle'],
                                control_command_dict['brake'],
                                ground_truth_mode=True,  # CRITICAL: Keep ground truth mode enabled!
                                ground_truth_speed=effective_target_speed,
                                randomize_start=control_command_dict.get('randomize_start', False),
                                randomize_request_id=control_command_dict.get('randomize_request_id', 0),
                                randomize_seed=control_command_dict.get('randomize_seed')
                            )
                    except:
                        pass
                    continue
                
                frame_count += 1
                
                # Log progress
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"Frame {frame_count}, elapsed: {elapsed:.1f}s, "
                              f"speed: {current_speed:.2f} m/s, steering: {steering:.3f}")
                
                # Sleep to maintain ~30 fps
                time.sleep(1.0 / 30.0)
        
        except KeyboardInterrupt:
            logger.info("Test interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
            logger.error("This may indicate Unity stopped or connection was lost.")
        
        finally:
            # Signal Unity that we're shutting down (so it can exit play mode)
            try:
                logger.info("Signaling Unity to stop (shutdown signal)...")
                self.bridge.signal_shutdown()
            except Exception as e:
                logger.warning(f"Failed to send shutdown signal to Unity: {e}")
            
            # Disable ground truth mode before cleanup
            try:
                logger.info("Disabling ground truth mode (returning to physics-based control)...")
                self.bridge.set_control_command(0.0, 0.0, 0.0, ground_truth_mode=False, ground_truth_speed=0.0)
            except:
                pass  # Ignore errors during cleanup
            
            elapsed = time.time() - start_time
            logger.info(f"\nTest completed:")
            logger.info(f"  Duration: {elapsed:.1f}s")
            logger.info(f"  Frames: {frame_count}")
            if self.av_stack.recorder:
                logger.info(f"  Recording: {self.av_stack.recorder.output_file}")
            
            # Close recorder
            if self.av_stack.recorder:
                try:
                    self.av_stack.recorder.close()
                except Exception as e:
                    logger.error(f"Error closing recorder: {e}")
            
            # Stop bridge server if we started it
            self._stop_bridge_server()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Ground truth following test")
    parser.add_argument("--duration", type=float, default=60.0,
                       help="Test duration in seconds (default: 60)")
    parser.add_argument("--speed", type=float, default=5.0,
                       help="Target speed in m/s (default: 5.0)")
    parser.add_argument("--output", type=str, default=None,
                       help="Recording name (default: auto-generated)")
    parser.add_argument("--bridge-url", type=str, default="http://localhost:8000",
                       help="Unity bridge URL (default: http://localhost:8000)")
    parser.add_argument("--auto-play", action="store_true",
                       help="Automatically start Unity play mode (requires Unity Editor to be open)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging for debugging")
    parser.add_argument("--random-start", action="store_true",
                       help="Randomize car start position on the oval")
    parser.add_argument("--random-seed", type=int, default=None,
                       help="Optional seed for random start reproducibility")
    parser.add_argument("--constant-speed", action="store_true",
                       help="Disable GT speed PID and move at constant speed")
    parser.add_argument("--target-lane", type=str, default="right",
                       choices=["right", "left", "center"],
                       help="Target lane center to follow (default: right)")
    parser.add_argument("--single-lane-width-threshold", type=float, default=5.0,
                       help="Lane width below this is treated as a single lane (default: 5.0m)")
    parser.add_argument("--lateral-correction-gain", type=float, default=None,
                       help="Optional lateral correction gain override")
    parser.add_argument("--use-cv", action="store_true",
                       help="Force CV-based perception (override default segmentation)")
    parser.add_argument(
        "--segmentation-checkpoint",
        type=str,
        default="data/segmentation_dataset/checkpoints/segnet_best.pt",
        help="Segmentation checkpoint path (default: segnet_best.pt)",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        global logger
        logger = configure_logging(logging.INFO)

    try:
        use_segmentation, seg_checkpoint = resolve_segmentation_settings(
            args.use_cv, args.segmentation_checkpoint
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))
        return 1

    follower = GroundTruthFollower(
        bridge_url=args.bridge_url,
        target_speed=args.speed,
        lookahead=8.0,
        max_safe_speed=10.0,
        speed_kp=0.15,  # Tuned for Unity: motorForce=400, smooth at 5 m/s
        speed_ki=0.02,  # Small integral for steady-state accuracy
        speed_kd=0.05,  # Derivative for damping oscillations
        random_start=args.random_start,
        random_seed=args.random_seed,
        constant_speed=args.constant_speed,
        target_lane=args.target_lane,
        use_segmentation=use_segmentation,
        segmentation_model_path=seg_checkpoint,
        single_lane_width_threshold_m=args.single_lane_width_threshold,
        lateral_correction_gain=args.lateral_correction_gain,
    )
    
    try:
        follower.run(duration=args.duration, recording_name=args.output, auto_play=args.auto_play)
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

