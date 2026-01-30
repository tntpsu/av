"""
Main AV stack integration script.
Connects all components: perception, trajectory planning, and control.
"""

import time
import math
import numpy as np
from typing import Optional, Tuple
import sys
from pathlib import Path
import logging
import yaml
from dataclasses import dataclass
import cv2

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from bridge.client import UnityBridgeClient
from perception.inference import LaneDetectionInference
from trajectory.inference import TrajectoryPlanningInference
from trajectory.speed_planner import SpeedPlanner, SpeedPlannerConfig
from control.pid_controller import VehicleController
from data.recorder import DataRecorder
from data.formats.data_format import CameraFrame, VehicleState, ControlCommand, PerceptionOutput, TrajectoryOutput, RecordingFrame

# Configure logging
# Ensure tmp/logs directory exists
log_dir = Path(__file__).parent / 'tmp' / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / 'av_stack.log'

logging.basicConfig(
    level=logging.ERROR,  # Changed from INFO to ERROR to reduce expensive print operations
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(log_file))
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ControlConfig:
    """Control configuration parameters."""
    lateral_kp: float = 0.3
    lateral_ki: float = 0.0
    lateral_kd: float = 0.1
    lateral_max_steering: float = 0.5
    lateral_deadband: float = 0.02
    lateral_heading_weight: float = 0.5
    lateral_lateral_weight: float = 0.5
    lateral_error_clip: float = np.pi / 4
    lateral_integral_limit: float = 0.3
    
    longitudinal_kp: float = 0.3
    longitudinal_ki: float = 0.05
    longitudinal_kd: float = 0.02
    longitudinal_target_speed: float = 8.0
    longitudinal_max_speed: float = 10.0
    longitudinal_speed_smoothing: float = 0.7
    longitudinal_speed_deadband: float = 0.1
    longitudinal_throttle_limit_threshold: float = 0.8
    longitudinal_throttle_reduction_factor: float = 0.3
    longitudinal_brake_aggression: float = 3.0


@dataclass
class TrajectoryConfig:
    """Trajectory planning configuration parameters."""
    lookahead_distance: float = 20.0
    point_spacing: float = 1.0
    target_speed: float = 8.0
    reference_lookahead: float = 8.0
    image_width: float = 640.0
    image_height: float = 480.0
    camera_fov: float = 75.0
    camera_height: float = 1.2
    bias_correction_threshold: float = 10.0


@dataclass
class SafetyConfig:
    """Safety configuration parameters."""
    max_speed: float = 10.0
    emergency_brake_threshold: float = 2.0
    speed_prevention_threshold: float = 0.85
    speed_prevention_brake_threshold: float = 0.9
    speed_prevention_brake_amount: float = 0.2
    lane_width: float = 7.0  # Lane width in meters
    car_width: float = 1.85  # Car width in meters
    allowed_outside_lane: float = 1.0  # Allowed distance outside lane before emergency stop (meters)


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file or use defaults."""
    if config_path is None:
        config_path = Path(__file__).parent / "config" / "av_stack_config.yaml"
    else:
        config_path = Path(config_path)
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    else:
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return {}


def _slew_limit_value(previous: float, target: float, max_rate: float, dt: float) -> float:
    """Slew-limit a value by max_rate per second."""
    if max_rate <= 0.0 or dt <= 0.0:
        return float(target)
    max_delta = max_rate * dt
    return float(previous + np.clip(target - previous, -max_delta, max_delta))


def _apply_speed_limit_preview(base_speed: float, preview_limit: float,
                               preview_distance: float, max_decel: float) -> float:
    """Clamp speed so we can meet an upcoming limit with a comfortable decel."""
    if preview_limit <= 0.0 or preview_distance <= 0.0 or max_decel <= 0.0:
        return float(base_speed)
    max_allowed = math.sqrt(max((preview_limit ** 2) + (2.0 * max_decel * preview_distance), 0.0))
    return float(min(base_speed, max_allowed))


def _apply_target_speed_slew(previous: float, target: float, rate_up: float,
                             rate_down: float, dt: float) -> float:
    """Slew-limit target speed with asymmetric up/down rates."""
    if dt <= 0.0:
        return float(target)
    adjusted = float(target)
    if rate_up > 0.0 and adjusted > previous:
        adjusted = min(adjusted, previous + (rate_up * dt))
    if rate_down > 0.0 and adjusted < previous:
        adjusted = max(adjusted, previous - (rate_down * dt))
    return float(adjusted)


def _apply_restart_ramp(desired_speed: float, current_speed: float,
                        ramp_start_time: Optional[float], timestamp: float,
                        ramp_seconds: float, stop_threshold: float) -> Tuple[float, Optional[float], bool]:
    """Ramp target speed up after coming to a stop."""
    if ramp_seconds <= 0.0:
        return float(desired_speed), None, False
    if current_speed <= stop_threshold and desired_speed > 0.0:
        if ramp_start_time is None:
            ramp_start_time = float(timestamp)
        elapsed = max(0.0, float(timestamp) - float(ramp_start_time))
        ramp_limit = float(desired_speed) * min(1.0, elapsed / ramp_seconds)
        return float(min(desired_speed, ramp_limit)), ramp_start_time, True
    return float(desired_speed), None, False


def _is_teleport_jump(
    previous_position: Optional[np.ndarray],
    current_position: Optional[np.ndarray],
    distance_threshold: float,
) -> Tuple[bool, float]:
    """Detect sudden position jumps that indicate Unity teleport/discontinuity."""
    if previous_position is None or current_position is None:
        return False, 0.0
    distance = float(np.linalg.norm(current_position - previous_position))
    return distance > distance_threshold, distance


class AVStack:
    """Main AV stack integrating all components."""
    
    def __init__(self, bridge_url: str = "http://localhost:8000",
                 model_path: Optional[str] = None,
                 record_data: bool = True,  # Enable by default for comprehensive logging
                 recording_dir: str = "data/recordings",
                 config_path: Optional[str] = None,
                 use_segmentation: bool = False,
                 segmentation_model_path: Optional[str] = None):
        """
        Initialize AV stack.
        
        Args:
            bridge_url: URL of Unity bridge server
            model_path: Path to trained perception model
            record_data: Whether to record data (default: True for comprehensive logging)
            recording_dir: Directory for recordings
        """
        # Load configuration
        config = load_config(config_path)
        control_cfg = config.get('control', {})
        trajectory_cfg = config.get('trajectory', {})
        safety_cfg = config.get('safety', {})
        
        # Bridge client
        self.bridge = UnityBridgeClient(bridge_url)
        
        # Perception
        self.perception = LaneDetectionInference(
            model_path=model_path,
            segmentation_model_path=segmentation_model_path,
            segmentation_mode=use_segmentation,
        )
        
        # Trajectory planning - Load from config
        lateral_cfg = control_cfg.get('lateral', {})
        longitudinal_cfg = control_cfg.get('longitudinal', {})
        
        self.trajectory_planner = TrajectoryPlanningInference(
            planner_type="rule_based",
            lookahead_distance=trajectory_cfg.get('lookahead_distance', 20.0),
            target_speed=trajectory_cfg.get('target_speed', 8.0),
            image_width=trajectory_cfg.get('image_width', 640.0),
            image_height=trajectory_cfg.get('image_height', 480.0),
            reference_smoothing=trajectory_cfg.get('reference_smoothing', 0.95),  # Increased to 0.95
            lane_smoothing_alpha=trajectory_cfg.get('lane_smoothing_alpha', 0.7),  # NEW: Lane coefficient smoothing
            camera_fov=trajectory_cfg.get('camera_fov', 75.0),
            camera_height=trajectory_cfg.get('camera_height', 1.2),
            bias_correction_threshold=trajectory_cfg.get('bias_correction_threshold', 10.0),
            camera_offset_x=trajectory_cfg.get('camera_offset_x', 0.0),  # NEW: Camera offset correction
            distance_scaling_factor=trajectory_cfg.get('distance_scaling_factor', 0.875)  # NEW: Distance scaling (7/8 = 0.875)
        )
        
        # Control - Load from config
        lateral_kp = lateral_cfg.get('kp', 0.3)
        lateral_ki = lateral_cfg.get('ki', 0.0)
        lateral_kd = lateral_cfg.get('kd', 0.1)
        
        # Log controller parameters for debugging
        logger.info(f"[CONTROLLER] Lateral control parameters: kp={lateral_kp}, ki={lateral_ki}, kd={lateral_kd}")
        
        self.controller = VehicleController(
            lateral_kp=lateral_kp,
            lateral_ki=lateral_ki,
            lateral_kd=lateral_kd,
            longitudinal_kp=longitudinal_cfg.get('kp', 0.3),
            longitudinal_ki=longitudinal_cfg.get('ki', 0.05),
            longitudinal_kd=longitudinal_cfg.get('kd', 0.02),
            lookahead_distance=trajectory_cfg.get('reference_lookahead', 8.0),
            target_speed=longitudinal_cfg.get('target_speed', 8.0),
            max_speed=safety_cfg.get('max_speed', 10.0),
            throttle_rate_limit=longitudinal_cfg.get('throttle_rate_limit', 0.08),
            brake_rate_limit=longitudinal_cfg.get('brake_rate_limit', 0.15),
            throttle_smoothing_alpha=longitudinal_cfg.get('throttle_smoothing_alpha', 0.6),
            speed_smoothing_alpha=longitudinal_cfg.get('speed_smoothing', 0.6),
            min_throttle_when_accel=longitudinal_cfg.get('min_throttle_when_accel', 0.0),
            max_steering=lateral_cfg.get('max_steering', 0.5),
            lateral_deadband=lateral_cfg.get('deadband', 0.02),
            lateral_heading_weight=lateral_cfg.get('heading_weight', 0.5),
            lateral_lateral_weight=lateral_cfg.get('lateral_weight', 0.5),
            lateral_error_clip=lateral_cfg.get('error_clip', np.pi / 4),
            lateral_integral_limit=lateral_cfg.get('integral_limit', 0.10),  # FIXED: Read from config (default 0.10)
            steering_smoothing_alpha=lateral_cfg.get('steering_smoothing_alpha', 0.7),
            curve_feedforward_gain=lateral_cfg.get('curve_feedforward_gain', 1.0),
            curve_feedforward_threshold=lateral_cfg.get('curve_feedforward_threshold', 0.02),
            curve_feedforward_gain_min=lateral_cfg.get('curve_feedforward_gain_min', 1.0),
            curve_feedforward_gain_max=lateral_cfg.get('curve_feedforward_gain_max', 1.0),
            curve_feedforward_curvature_min=lateral_cfg.get('curve_feedforward_curvature_min', 0.005),
            curve_feedforward_curvature_max=lateral_cfg.get('curve_feedforward_curvature_max', 0.03),
            curve_feedforward_curvature_clamp=lateral_cfg.get('curve_feedforward_curvature_clamp', 0.03),
            straight_curvature_threshold=lateral_cfg.get('straight_curvature_threshold', 0.01),
            steering_rate_curvature_min=lateral_cfg.get('steering_rate_curvature_min', 0.005),
            steering_rate_curvature_max=lateral_cfg.get('steering_rate_curvature_max', 0.03),
            steering_rate_scale_min=lateral_cfg.get('steering_rate_scale_min', 0.5)
        )
        
        # Store config for use in _process_frame
        self.config = config
        self.control_config = control_cfg
        self.trajectory_config = trajectory_cfg
        self.safety_config = safety_cfg

        # Speed planner (jerk-limited) configuration
        speed_planner_cfg = trajectory_cfg.get('speed_planner', {})
        self.speed_planner_enabled = bool(speed_planner_cfg.get('enabled', False))
        self.speed_planner = None
        if self.speed_planner_enabled:
            default_dt = float(speed_planner_cfg.get('default_dt', 1.0 / 30.0))
            planner_config = SpeedPlannerConfig(
                max_accel=float(speed_planner_cfg.get('max_accel', 2.0)),
                max_decel=float(speed_planner_cfg.get('max_decel', 3.0)),
                max_jerk=float(speed_planner_cfg.get('max_jerk', 2.0)),
                min_speed=float(speed_planner_cfg.get('min_speed', 0.0)),
                launch_speed_floor=float(speed_planner_cfg.get('launch_speed_floor', 0.0)),
                launch_speed_floor_threshold=float(speed_planner_cfg.get('launch_speed_floor_threshold', 0.0)),
                reset_gap_seconds=float(speed_planner_cfg.get('reset_gap_seconds', 0.5)),
                sync_speed_threshold=float(speed_planner_cfg.get('sync_speed_threshold', 3.0)),
                default_dt=default_dt,
            )
            self.speed_planner = SpeedPlanner(planner_config)
        
        # Data recording (enabled by default for comprehensive logging)
        self.recorder = None
        if record_data:
            self.recorder = DataRecorder(recording_dir)
            logger.info(f"Data recording enabled: {self.recorder.output_file}")
        else:
            logger.info("Data recording disabled")
        
        # State
        self.running = False
        self.frame_count = 0
        self.target_fps = 30.0  # Match Unity's frame rate
        self.frame_interval = 1.0 / self.target_fps

        # Initialize previous lane history for stale-data fallback paths
        self.previous_left_lane_x = None
        self.previous_right_lane_x = None
        
        # Track if emergency stop has been logged (to prevent repeated messages)
        self.emergency_stop_logged = False
        self.emergency_stop_type = None  # Track which type of emergency stop (for logging)

        # Teleport/jump guard state
        self.last_vehicle_position = None
        self.last_vehicle_timestamp = None
        self.teleport_guard_frames_remaining = 0
        self.last_teleport_distance = None
        self.last_teleport_dt = None
        self.post_jump_cooldown_frames = 0
        
        # NEW: Perception health monitoring
        self.consecutive_bad_detection_frames = 0  # Track consecutive frames with <2 lanes
        self.perception_health_history = []  # Recent detection quality (True/False for good/bad)
        self.perception_health_history_size = 60  # Track last 60 frames (~2 seconds at 30 FPS)
        self.perception_health_score = 1.0  # Current health score (1.0 = perfect, 0.0 = failed)
        self.perception_health_status = "healthy"  # "healthy", "degraded", "poor", "critical"
        self.original_target_speed = trajectory_cfg.get('target_speed', 8.0)  # Store original target speed for recovery
        # Speed limit smoothing state
        self.last_speed_limit: Optional[float] = None
        self.last_speed_limit_time: Optional[float] = None
        # Target speed smoothing state
        self.last_target_speed: Optional[float] = None
        self.last_target_speed_time: Optional[float] = None
        self.restart_ramp_start_time: Optional[float] = None
        # Launch throttle ramp state
        self.launch_throttle_ramp_start_time: Optional[float] = None
        self.launch_throttle_ramp_armed: bool = True
        self.launch_stop_candidate_start_time: Optional[float] = None
    
    def run(self, max_frames: Optional[int] = None, duration: Optional[float] = None):
        """
        Run AV stack main loop.
        
        Args:
            max_frames: Maximum number of frames to process (None for infinite)
            duration: Maximum duration in seconds (None for infinite)
        """
        logger.info("Starting AV Stack...")
        
        # Check bridge connection with retry
        if not self._wait_for_bridge(max_retries=10, initial_delay=1.0):
            logger.error("Bridge server is not available after retries!")
            logger.error("Please start the bridge server: python -m bridge.server")
            return
        
        logger.info("Bridge server connected")
        
        # Wait for first frame from Unity (with exponential backoff)
        logger.info("Waiting for Unity to send first frame...")
        if not self._wait_for_first_frame(max_retries=30, initial_delay=0.5):
            logger.warning("No frames received from Unity. Starting anyway...")
        else:
            logger.info("First frame received! Starting main loop...")
        
        self.running = True
        self.frame_count = 0
        camera_wait_start: Optional[float] = None
        last_camera_wall_time: Optional[float] = None
        last_camera_timestamp: Optional[float] = None
        last_loop_time: Optional[float] = None
        last_frame_time = time.time()
        start_time = time.time()
        
        try:
            while self.running:
                loop_start = time.time()
                # Check frame limit
                if max_frames and self.frame_count >= max_frames:
                    logger.info(f"Reached frame limit: {max_frames}")
                    break
                
                # Check duration limit
                if duration is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= duration:
                        logger.info(f"Reached duration limit: {duration}s")
                        break
                
                # Rate limiting: maintain target FPS
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < self.frame_interval:
                    time.sleep(self.frame_interval - elapsed)
                
                last_frame_time = time.time()
                
                # Get latest camera frame
                camera_fetch_start = time.time()
                frame_data = self.bridge.get_latest_camera_frame()
                camera_fetch_duration = time.time() - camera_fetch_start
                if camera_fetch_duration > 0.1:
                    logger.warning(
                        "[CAMERA_FETCH_SLOW] duration=%.3fs frame=%s",
                        camera_fetch_duration,
                        self.frame_count,
                    )
                if frame_data is None:
                    if camera_wait_start is None:
                        camera_wait_start = time.time()
                    # No frame available - don't spam, just wait
                    time.sleep(self.frame_interval)
                    continue
                if camera_wait_start is not None:
                    wait_duration = time.time() - camera_wait_start
                    if wait_duration > 0.2:
                        logger.warning(
                            "[CAMERA_WAIT_GAP] duration=%.3fs frame=%s",
                            wait_duration,
                            self.frame_count,
                        )
                    camera_wait_start = None
                
                image, timestamp = frame_data
                now = time.time()
                if last_camera_wall_time is not None and last_camera_timestamp is not None:
                    wall_gap = now - last_camera_wall_time
                    ts_gap = timestamp - last_camera_timestamp
                    if ts_gap > 0.2:
                        logger.warning(
                            "[CAMERA_TIMESTAMP_GAP] ts_gap=%.3fs wall_gap=%.3fs frame=%s",
                            ts_gap,
                            wall_gap,
                            self.frame_count,
                        )
                    drift = ts_gap - wall_gap
                    if abs(drift) > 0.2:
                        logger.warning(
                            "[CAMERA_TIME_DRIFT] ts_gap=%.3fs wall_gap=%.3fs drift=%.3fs frame=%s",
                            ts_gap,
                            wall_gap,
                            drift,
                            self.frame_count,
                        )
                last_camera_wall_time = now
                last_camera_timestamp = timestamp
                
                # Get vehicle state
                vehicle_state_dict = self.bridge.get_latest_vehicle_state()
                if vehicle_state_dict is None:
                    time.sleep(self.frame_interval)
                    continue
                
                # Process frame
                self._process_frame(image, timestamp, vehicle_state_dict)
                
                self.frame_count += 1

                if last_loop_time is not None:
                    loop_duration = time.time() - loop_start
                    if loop_duration > 0.2:
                        logger.warning(
                            "[LOOP_SLOW] duration=%.3fs frame=%s",
                            loop_duration,
                            self.frame_count,
                        )
                last_loop_time = loop_start
        
        except KeyboardInterrupt:
            logger.info("\nStopping AV Stack...")
        finally:
            self.stop()
    
    def _wait_for_bridge(self, max_retries: int = 10, initial_delay: float = 1.0) -> bool:
        """Wait for bridge server to be available with exponential backoff."""
        delay = initial_delay
        for attempt in range(max_retries):
            if self.bridge.health_check():
                return True
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
                logger.info(f"First frame received after {attempt + 1} attempts")
                return True
            if attempt % 5 == 0:  # Log every 5 attempts
                logger.info(f"Waiting for Unity frames... (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
            delay = min(delay * 1.2, 2.0)  # Exponential backoff, max 2s
        return False
    
    def _process_frame(self, image: np.ndarray, timestamp: float, vehicle_state_dict: dict):
        """
        Process a single frame through the AV stack.
        
        Args:
            image: Camera image
            timestamp: Frame timestamp
            vehicle_state_dict: Vehicle state dictionary
        """
        process_start = time.time()
        # Track timestamp for perception frozen detection
        # CRITICAL: Save previous timestamp BEFORE updating (we need it for comparisons)
        prev_timestamp = self.last_timestamp if hasattr(self, 'last_timestamp') else None
        # Update last_timestamp at the END of processing (after all timestamp comparisons)

        # Teleport/jump guard: detect sudden vehicle position discontinuities
        current_position = self._extract_position(vehicle_state_dict)
        teleport_distance_threshold = self.safety_config.get('teleport_distance_threshold', 2.0)
        teleport_detected, teleport_distance = _is_teleport_jump(
            self.last_vehicle_position,
            current_position,
            teleport_distance_threshold,
        )
        if teleport_detected:
            dt = None
            if self.last_vehicle_timestamp is not None:
                dt = timestamp - self.last_vehicle_timestamp
            self.teleport_guard_frames_remaining = int(
                self.safety_config.get('teleport_guard_frames', 1)
            )
            self.last_teleport_distance = teleport_distance
            self.last_teleport_dt = dt
            self.controller.reset()
            if hasattr(self.trajectory_planner, 'last_smoothed_ref_point'):
                self.trajectory_planner.last_smoothed_ref_point = None
            if hasattr(self.trajectory_planner, 'smoothed_bias_history'):
                self.trajectory_planner.smoothed_bias_history = []
            if hasattr(self.trajectory_planner, 'last_timestamp'):
                self.trajectory_planner.last_timestamp = None
            self.post_jump_cooldown_frames = int(
                self.safety_config.get('post_jump_cooldown_frames', 30)
            )
            logger.warning(
                f"[Frame {self.frame_count}] [TELEPORT GUARD] Position jump detected "
                f"({teleport_distance:.2f}m, dt={dt if dt is not None else 'N/A'}s). "
                "Resetting controllers and skipping emergency stop for this frame."
            )
        teleport_guard_active = self.teleport_guard_frames_remaining > 0
        
        # 1. Perception: Detect lanes
        # detect() now returns: (lane_coeffs, confidence, detection_method, num_lanes_detected)
        perception_start = time.time()
        perception_result = self.perception.detect(image)
        perception_duration = time.time() - perception_start
        if len(perception_result) == 4:
            lane_coeffs, confidence, detection_method, num_lanes_detected = perception_result
        else:
            # Fallback for old format (2 values)
            lane_coeffs, confidence = perception_result
            num_lanes_detected = sum(1 for c in lane_coeffs if c is not None)
            # Determine detection method (check if CV fallback was used)
            detection_method = "ml"  # Default
            if hasattr(self.perception, 'last_detection_method'):
                detection_method = self.perception.last_detection_method
            elif not hasattr(self.perception, 'model_trained') or not self.perception.model_trained:
                detection_method = "cv"  # Likely using CV fallback
        
        # DEBUG: Save visualization frames periodically to diagnose road edge detection
        frame_count = getattr(self, '_frame_count', 0)
        self._frame_count = frame_count + 1
        enable_debug_vis = (frame_count % 30 == 0)  # Every 30 frames (1 second at 30fps)
        
        # Check if we should create debug visualization
        # Try both direct cv_detector access and through inference
        should_debug = enable_debug_vis and detection_method == "cv"
        cv_detector = None
        if hasattr(self.perception, 'cv_detector'):
            cv_detector = self.perception.cv_detector
        elif hasattr(self.perception, 'detector') and hasattr(self.perception.detector, 'cv_detector'):
            cv_detector = self.perception.detector.cv_detector
        
        # DEBUG: Log why debug visualization might not run
        if enable_debug_vis:
            logger.info(f"[DEBUG VIS] Frame {frame_count}: enable_debug_vis={enable_debug_vis}, "
                       f"detection_method={detection_method}, has_cv_detector={cv_detector is not None}, "
                       f"should_debug={should_debug}")
        
        # Store fit_points from debug_info (if available)
        fit_points_left = None
        fit_points_right = None
        
        if should_debug and cv_detector is not None:
            try:
                logger.info(f"[DEBUG VIS] Creating debug visualization for frame {frame_count}")
                # Re-detect with debug info
                # detect() returns (lane_coeffs, debug_info) when return_debug=True
                result = cv_detector.detect(image, return_debug=True)
                logger.info(f"[DEBUG VIS] detect() returned type: {type(result)}, length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
                if isinstance(result, tuple) and len(result) == 2:
                    cv_lanes, debug_info = result
                    logger.info(f"[DEBUG VIS] Unpacked: cv_lanes type={type(cv_lanes)}, debug_info type={type(debug_info)}")
                    # Extract fit_points from debug_info
                    if isinstance(debug_info, dict) and 'fit_points' in debug_info:
                        fit_points = debug_info['fit_points']
                        if isinstance(fit_points, dict):
                            fit_points_left = fit_points.get('left')
                            fit_points_right = fit_points.get('right')
                else:
                    # Fallback if return format is different
                    logger.warning(f"[DEBUG VIS] Unexpected return format: {type(result)}")
                    cv_lanes = result if isinstance(result, list) else [None, None]
                    debug_info = {}
            except Exception as e:
                logger.error(f"[DEBUG VIS] Error getting debug info: {e}")
                debug_info = {}
                fit_points_left = None
                fit_points_right = None
        else:
            # Even if not creating debug vis, try to get fit_points for recording
            # Only do this if CV detector is available and we're using CV method
            if detection_method == "cv" and cv_detector is not None:
                try:
                    result = cv_detector.detect(image, return_debug=True)
                    if isinstance(result, tuple) and len(result) == 2:
                        _, debug_info = result
                        if isinstance(debug_info, dict) and 'fit_points' in debug_info:
                            fit_points = debug_info['fit_points']
                            if isinstance(fit_points, dict):
                                fit_points_left = fit_points.get('left')
                                fit_points_right = fit_points.get('right')
                except Exception as e:
                    logger.debug(f"Failed to get fit_points: {e}")
                
                # Create visualization
                try:
                    vis_image = cv_detector.visualize_detection(
                        image, cv_lanes, debug_info=debug_info
                    )
                    
                    # Save to debug directory
                    debug_dir = Path('tmp/debug_visualizations')
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    save_path = debug_dir / f'frame_{frame_count:06d}.png'
                    
                    # DEBUG: Also save intermediate images to diagnose detection
                    logger.info(f"[DEBUG VIS] debug_info keys: {list(debug_info.keys()) if debug_info else 'empty'}")
                    if debug_info:
                        # Save masked edges (what HoughLinesP sees)
                        logger.info(f"[DEBUG VIS] Checking masked_edges: {'masked_edges' in debug_info}, "
                                   f"value is None: {debug_info.get('masked_edges') is None if 'masked_edges' in debug_info else 'key missing'}")
                        if 'masked_edges' in debug_info and debug_info['masked_edges'] is not None:
                            edges_path = debug_dir / f'frame_{frame_count:06d}_edges.png'
                            cv2.imwrite(str(edges_path), debug_info['masked_edges'])
                        
                        # Save yellow mask (yellow lane detection - primary for yellow lanes)
                        if 'yellow_mask' in debug_info and debug_info['yellow_mask'] is not None:
                            yellow_path = debug_dir / f'frame_{frame_count:06d}_yellow_mask.png'
                            cv2.imwrite(str(yellow_path), debug_info['yellow_mask'])
                        
                        # Save combined (edges + color)
                        if 'combined' in debug_info and debug_info['combined'] is not None:
                            combined_path = debug_dir / f'frame_{frame_count:06d}_combined.png'
                            cv2.imwrite(str(combined_path), debug_info['combined'])
                        
                        # Log detection stats
                        num_lines = debug_info.get('num_lines_detected', 0)
                        left_count = debug_info.get('left_lines_count', 0)
                        right_count = debug_info.get('right_lines_count', 0)
                        skipped_short = debug_info.get('skipped_short', 0)
                        skipped_center = debug_info.get('skipped_center', 0)
                        validation_failures = debug_info.get('validation_failures', {})
                        
                        logger.info(f"[DEBUG VIS] Frame {frame_count}: {num_lines} lines detected by HoughLinesP")
                        logger.info(f"[DEBUG VIS] Frame {frame_count}: Left={left_count}, Right={right_count}, "
                                   f"Skipped: short={skipped_short}, center={skipped_center}")
                        if validation_failures:
                            logger.info(f"[DEBUG VIS] Frame {frame_count}: Validation failures: {validation_failures}")
                    
                    cv2.imwrite(str(save_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                    
                    # Also create histogram of line x-positions to check for bimodal peaks
                    if debug_info and 'all_lines' in debug_info and debug_info['all_lines'] is not None:
                        line_centers = []
                        for line in debug_info['all_lines']:
                            x1, y1, x2, y2 = line[0]
                            line_center_x = (x1 + x2) / 2
                            line_centers.append(line_center_x)
                        
                        if line_centers:
                            import matplotlib
                            matplotlib.use('Agg')  # Non-interactive backend
                            import matplotlib.pyplot as plt
                            
                            plt.figure(figsize=(10, 6))
                            plt.hist(line_centers, bins=50, edgecolor='black')
                            plt.axvline(image.shape[1] // 2, color='r', linestyle='--', label='Image Center')
                            plt.xlabel('Line Center X Position (pixels)')
                            plt.ylabel('Frequency')
                            plt.title(f'Detected Line Positions (Frame {frame_count})')
                            plt.legend()
                            plt.grid(True, alpha=0.3)
                            
                            hist_path = debug_dir / f'line_histogram_{frame_count:06d}.png'
                            plt.savefig(hist_path, dpi=100, bbox_inches='tight')
                            plt.close()
                            
                            logger.info(f"Saved debug visualization: {save_path}, histogram: {hist_path}")
                except Exception as e:
                    logger.warning(f"Failed to create debug visualization: {e}")
        
        # Calculate lane line positions at lookahead distance (for trajectory centering verification)
        left_lane_line_x = None
        right_lane_line_x = None
        
        # Initialize diagnostic tracking variables (available for all code paths)
        using_stale_data = False
        stale_data_reason = None
        left_jump_magnitude = None
        right_jump_magnitude = None
        jump_threshold_used = None
        # NEW: Diagnostic fields for perception instability (initialized to None, set only when instability detected)
        actual_detected_left_lane_x = None
        actual_detected_right_lane_x = None
        instability_width_change = None
        instability_center_shift = None
        
        reference_lookahead = self.trajectory_config.get('reference_lookahead', 8.0)
        
        # Extract camera_8m_screen_y early so we can use it for lane evaluation
        camera_8m_screen_y = vehicle_state_dict.get('camera8mScreenY')
        if camera_8m_screen_y is None:
            camera_8m_screen_y = vehicle_state_dict.get('camera_8m_screen_y', -1.0)
        camera_lookahead_screen_y = vehicle_state_dict.get('cameraLookaheadScreenY')
        if camera_lookahead_screen_y is None:
            camera_lookahead_screen_y = vehicle_state_dict.get('camera_lookahead_screen_y', -1.0)
        ground_truth_lookahead_distance = vehicle_state_dict.get('groundTruthLookaheadDistance')
        if ground_truth_lookahead_distance is None:
            ground_truth_lookahead_distance = vehicle_state_dict.get('ground_truth_lookahead_distance', 8.0)

        speed_limit = vehicle_state_dict.get('speedLimit')
        if speed_limit is None:
            speed_limit = vehicle_state_dict.get('speed_limit', 0.0)
        speed_limit_preview = vehicle_state_dict.get('speedLimitPreview')
        if speed_limit_preview is None:
            speed_limit_preview = vehicle_state_dict.get('speed_limit_preview', 0.0)
        speed_limit_preview_distance = vehicle_state_dict.get('speedLimitPreviewDistance')
        if speed_limit_preview_distance is None:
            speed_limit_preview_distance = vehicle_state_dict.get(
                'speed_limit_preview_distance',
                self.trajectory_config.get('speed_limit_preview_distance', 0.0),
            )
        camera_lookahead_screen_y = vehicle_state_dict.get('cameraLookaheadScreenY')
        if camera_lookahead_screen_y is None:
            camera_lookahead_screen_y = vehicle_state_dict.get('camera_lookahead_screen_y', -1.0)
        ground_truth_lookahead_distance = vehicle_state_dict.get('groundTruthLookaheadDistance')
        if ground_truth_lookahead_distance is None:
            ground_truth_lookahead_distance = vehicle_state_dict.get('ground_truth_lookahead_distance', 8.0)
        camera_lookahead_screen_y = vehicle_state_dict.get('cameraLookaheadScreenY')
        if camera_lookahead_screen_y is None:
            camera_lookahead_screen_y = vehicle_state_dict.get('camera_lookahead_screen_y', -1.0)
        ground_truth_lookahead_distance = vehicle_state_dict.get('groundTruthLookaheadDistance')
        if ground_truth_lookahead_distance is None:
            ground_truth_lookahead_distance = vehicle_state_dict.get('ground_truth_lookahead_distance', 8.0)
        camera_lookahead_screen_y = vehicle_state_dict.get('cameraLookaheadScreenY')
        if camera_lookahead_screen_y is None:
            camera_lookahead_screen_y = vehicle_state_dict.get('camera_lookahead_screen_y', -1.0)
        camera_lookahead_screen_y = vehicle_state_dict.get('cameraLookaheadScreenY')
        if camera_lookahead_screen_y is None:
            camera_lookahead_screen_y = vehicle_state_dict.get('camera_lookahead_screen_y', -1.0)
        ground_truth_lookahead_distance = vehicle_state_dict.get('groundTruthLookaheadDistance')
        if ground_truth_lookahead_distance is None:
            ground_truth_lookahead_distance = vehicle_state_dict.get('ground_truth_lookahead_distance', 8.0)
        camera_lookahead_screen_y = vehicle_state_dict.get('cameraLookaheadScreenY')
        if camera_lookahead_screen_y is None:
            camera_lookahead_screen_y = vehicle_state_dict.get('camera_lookahead_screen_y', -1.0)
        lookahead_distance_from_unity = vehicle_state_dict.get('groundTruthLookaheadDistance')
        if lookahead_distance_from_unity is None:
            lookahead_distance_from_unity = vehicle_state_dict.get('ground_truth_lookahead_distance')
        if isinstance(lookahead_distance_from_unity, (int, float)) and lookahead_distance_from_unity > 0:
            reference_lookahead = float(lookahead_distance_from_unity)
        
        # DEBUG: Log what we extracted (first few frames only)
        if self.frame_count < 3:
            logger.info(f"[COORD DEBUG] Frame {self.frame_count}: Extracted camera_8m_screen_y={camera_8m_screen_y} "
                       f"(from dict keys: {list(vehicle_state_dict.keys())[:10]}...)")
        
        # DEBUG: Log lane_coeffs format for debugging
        if self.frame_count == 0:
            logger.debug(f"[Frame {self.frame_count}] [DEBUG] lane_coeffs type={type(lane_coeffs)}, len={len(lane_coeffs) if hasattr(lane_coeffs, '__len__') else 'N/A'}")
            if hasattr(lane_coeffs, '__len__') and len(lane_coeffs) > 0:
                logger.debug(f"[Frame {self.frame_count}] [DEBUG] lane_coeffs[0] type={type(lane_coeffs[0])}, value={lane_coeffs[0] if lane_coeffs[0] is not None else 'None'}")
            if hasattr(lane_coeffs, '__len__') and len(lane_coeffs) > 1:
                logger.debug(f"[Frame {self.frame_count}] [DEBUG] lane_coeffs[1] type={type(lane_coeffs[1])}, value={lane_coeffs[1] if lane_coeffs[1] is not None else 'None'}")
            logger.debug(f"[Frame {self.frame_count}] [DEBUG] Condition check: len >= 2: {len(lane_coeffs) >= 2 if hasattr(lane_coeffs, '__len__') else False}, [0] not None: {lane_coeffs[0] is not None if hasattr(lane_coeffs, '__len__') and len(lane_coeffs) > 0 else False}, [1] not None: {lane_coeffs[1] is not None if hasattr(lane_coeffs, '__len__') and len(lane_coeffs) > 1 else False}")
        
        if len(lane_coeffs) >= 2 and lane_coeffs[0] is not None and lane_coeffs[1] is not None:
            try:
                # Get trajectory planner's conversion parameters
                planner = self.trajectory_planner.planner if hasattr(self.trajectory_planner, 'planner') else self.trajectory_planner
                image_height = planner.image_height if hasattr(planner, 'image_height') else 480.0
                image_width = planner.image_width if hasattr(planner, 'image_width') else 640.0
                pixel_to_meter_y = planner.pixel_to_meter_y if hasattr(planner, 'pixel_to_meter_y') else 0.02
                
                # Ensure pixel_to_meter_y is a numeric value (not a Mock in tests)
                if not isinstance(pixel_to_meter_y, (int, float)):
                    pixel_to_meter_y = 0.02
                
                # CRITICAL FIX: Use actual 8m position from Unity FIRST (most accurate)
                # Unity's WorldToScreenPoint tells us exactly where 8m appears in the image
                # This is more accurate than estimating from camera model
                # Use the already-extracted camera_8m_screen_y (extracted earlier at line 432)
                # DEBUG: Log what we're checking
                logger.debug(f"[COORD DEBUG] Checking camera_8m_screen_y: value={camera_8m_screen_y}, type={type(camera_8m_screen_y)}, >0={camera_8m_screen_y > 0 if camera_8m_screen_y is not None else 'None'}")
                if camera_lookahead_screen_y is not None and camera_lookahead_screen_y > 0:
                    # Use Unity's lookahead screen Y (most accurate for current lookahead)
                    y_image_at_lookahead = camera_lookahead_screen_y
                    logger.info(
                        f"[COORD DEBUG] ✅ Using Unity's camera_lookahead_screen_y={camera_lookahead_screen_y:.1f}px "
                        f"(lookahead={reference_lookahead:.2f}m) for lane evaluation"
                    )
                elif camera_8m_screen_y is not None and camera_8m_screen_y > 0 and abs(reference_lookahead - 8.0) < 0.5:
                    # Use Unity's actual 8m position (most accurate for 8m lookahead)
                    y_image_at_lookahead = camera_8m_screen_y
                    logger.info(f"[COORD DEBUG] ✅ Using Unity's camera_8m_screen_y={camera_8m_screen_y:.1f}px for lane evaluation")
                else:
                    # Fallback: Estimate y position from reference_lookahead distance
                    # Use inverse of the coordinate conversion fallback logic
                    # At bottom (y=height): distance ≈ 1.5m
                    # At top (y=0): distance ≈ far
                    # For 8m: y_normalized = 1.5 / 8.0 = 0.1875
                    # y_from_bottom = 0.1875 * image_height = 90 pixels
                    # y_pixels = image_height - 90 = 390 pixels
                    base_distance = 1.5  # meters at bottom
                    if reference_lookahead < base_distance:
                        y_image_at_lookahead = image_height
                    else:
                        y_normalized = base_distance / reference_lookahead
                        y_from_bottom = y_normalized * image_height
                        y_image_at_lookahead = image_height - y_from_bottom
                    logger.warning(f"[COORD DEBUG] camera_8m_screen_y not available ({camera_8m_screen_y}), using fallback y={y_image_at_lookahead:.1f}px")
                
                y_image_at_lookahead = max(0, min(image_height - 1, int(y_image_at_lookahead)))
                
                # Get lane x positions in image coordinates
                # CRITICAL FIX: Lane detection may return lanes in wrong order
                # Always check and swap if needed to ensure left < right
                lane0_coeffs = lane_coeffs[0]
                lane1_coeffs = lane_coeffs[1]
                
                # CRITICAL: Use the ACTUAL distance from Unity, not a fixed value
                # Unity's camera_8m_screen_y tells us where 8m appears in the image
                # If we're evaluating at that y position, the distance IS 8m
                # Don't use a fixed 7m - use the actual distance Unity reports
                if camera_lookahead_screen_y > 0:
                    conversion_distance = reference_lookahead
                    logger.info(f"[COORD DEBUG] Using Unity lookahead distance: {reference_lookahead:.2f}m")
                elif camera_8m_screen_y > 0 and abs(reference_lookahead - 8.0) < 0.5 and abs(y_image_at_lookahead - camera_8m_screen_y) < 5.0:
                    conversion_distance = 8.0  # Actual distance from Unity
                    logger.info(f"[COORD DEBUG] Using Unity's actual distance: 8.0m (at y={camera_8m_screen_y:.1f}px)")
                else:
                    conversion_distance = reference_lookahead
                    logger.warning(f"[COORD DEBUG] Using reference_lookahead={reference_lookahead}m (Unity distance not available)")
                
                logger.info(f"[COORD DEBUG] Evaluating at y={y_image_at_lookahead}px, "
                           f"using fixed distance={conversion_distance:.2f}m for coordinate conversion")
                
                # Determine which is left and which is right (smaller x = left in image coords)
                # Check at bottom of image (closest to vehicle) for most reliable ordering
                x0_at_bottom = np.polyval(lane0_coeffs, image_height)
                x1_at_bottom = np.polyval(lane1_coeffs, image_height)
                
                # Also check at lookahead distance for consistency
                x0_at_lookahead = np.polyval(lane0_coeffs, y_image_at_lookahead)
                x1_at_lookahead = np.polyval(lane1_coeffs, y_image_at_lookahead)
                
                # Use both checks to be more robust
                # If both agree, use that ordering. If they disagree, use bottom (closer to vehicle)
                if x0_at_bottom < x1_at_bottom and x0_at_lookahead < x1_at_lookahead:
                    # Consistent: lane0 is left, lane1 is right
                    actual_left_coeffs = lane0_coeffs
                    actual_right_coeffs = lane1_coeffs
                elif x0_at_bottom > x1_at_bottom and x0_at_lookahead > x1_at_lookahead:
                    # Consistent: lane0 is right, lane1 is left (swap needed)
                    actual_left_coeffs = lane1_coeffs
                    actual_right_coeffs = lane0_coeffs
                else:
                    # Inconsistent: use bottom position (closer to vehicle, more reliable)
                    if x0_at_bottom < x1_at_bottom:
                        actual_left_coeffs = lane0_coeffs
                        actual_right_coeffs = lane1_coeffs
                    else:
                        actual_left_coeffs = lane1_coeffs
                        actual_right_coeffs = lane0_coeffs
                
                # Get x positions at lookahead in image coordinates
                left_x_image = np.polyval(actual_left_coeffs, y_image_at_lookahead)
                right_x_image = np.polyval(actual_right_coeffs, y_image_at_lookahead)
                
                # CRITICAL FIX: Verify ordering is correct
                # Left lane should have smaller x (left side of image), right should have larger x
                if left_x_image > right_x_image:
                    logger.warning(f"[COORD DEBUG] Lane ordering reversed at y={y_image_at_lookahead}px! "
                                 f"left_x={left_x_image:.1f}px > right_x={right_x_image:.1f}px. Swapping.")
                    # Swap the positions
                    left_x_image, right_x_image = right_x_image, left_x_image
                    # Also swap the coefficients for consistency
                    actual_left_coeffs, actual_right_coeffs = actual_right_coeffs, actual_left_coeffs
                
                # FIXED: Don't clamp to image bounds - allow lanes outside image to be converted
                # This allows us to see true detected positions even if they're beyond image edges
                # The coordinate conversion will handle out-of-bounds values correctly
                # Only log a warning if lanes are way outside reasonable bounds
                if left_x_image < -100 or left_x_image > image_width + 100:
                    logger.warning(f"[COORD DEBUG] Left lane x={left_x_image:.1f}px is way outside image bounds (0-{image_width})")
                if right_x_image < -100 or right_x_image > image_width + 100:
                    logger.warning(f"[COORD DEBUG] Right lane x={right_x_image:.1f}px is way outside image bounds (0-{image_width})")
                
                # Final check: Ensure left < right after clamping
                if left_x_image >= right_x_image:
                    logger.warning(f"[COORD DEBUG] After clamping, lanes still reversed! "
                                 f"left_x={left_x_image:.1f}px, right_x={right_x_image:.1f}px. "
                                 f"Using center of image as fallback.")
                    # Fallback: Use center of image ± half lane width
                    image_center = image_width / 2.0
                    lane_half_width_px = 100  # Approximate half lane width in pixels at lookahead
                    left_x_image = image_center - lane_half_width_px
                    right_x_image = image_center + lane_half_width_px
                
                # Convert from image coordinates to vehicle coordinates
                # CRITICAL FIX: Use calculated distance for conversion (corresponds to y position)
                # This ensures pixel-to-meter conversion matches the actual distance
                logger.info(f"[COORD DEBUG] Input: left_x_image={left_x_image:.1f}, right_x_image={right_x_image:.1f}, "
                           f"y_image_at_lookahead={y_image_at_lookahead:.1f}, conversion_distance={conversion_distance:.2f}m")
                # Use Unity's reported horizontal FOV if available (most accurate)
                camera_horizontal_fov = vehicle_state_dict.get('cameraHorizontalFOV')
                if camera_horizontal_fov is None:
                    camera_horizontal_fov = vehicle_state_dict.get('camera_horizontal_fov', None)
                
                # DEBUG: Log before coordinate conversion (frame 0 only)
                if self.frame_count == 0:
                    logger.debug(f"[Frame {self.frame_count}] [DEBUG] About to call _convert_image_to_vehicle_coords")
                    logger.debug(f"[Frame {self.frame_count}] [DEBUG] left_x_image={left_x_image:.1f}, right_x_image={right_x_image:.1f}, y={y_image_at_lookahead:.1f}, dist={conversion_distance:.2f}m")
                
                left_x_vehicle, _ = planner._convert_image_to_vehicle_coords(
                    left_x_image, y_image_at_lookahead, lookahead_distance=conversion_distance,
                    horizontal_fov_override=camera_horizontal_fov if camera_horizontal_fov and camera_horizontal_fov > 0 else None
                )
                right_x_vehicle, _ = planner._convert_image_to_vehicle_coords(
                    right_x_image, y_image_at_lookahead, lookahead_distance=conversion_distance,
                    horizontal_fov_override=camera_horizontal_fov if camera_horizontal_fov and camera_horizontal_fov > 0 else None
                )
                calculated_lane_width = right_x_vehicle - left_x_vehicle
                
                # DEBUG: Log after coordinate conversion (frame 0 only)
                if self.frame_count == 0:
                    logger.debug(f"[Frame {self.frame_count}] [DEBUG] Coordinate conversion SUCCESS!")
                    logger.debug(f"[Frame {self.frame_count}] [DEBUG] left_x_vehicle={left_x_vehicle:.3f}m, right_x_vehicle={right_x_vehicle:.3f}m, width={calculated_lane_width:.3f}m")
                
                logger.info(f"[COORD DEBUG] Result: left_x_vehicle={left_x_vehicle:.3f}m, "
                           f"right_x_vehicle={right_x_vehicle:.3f}m, lane_width={calculated_lane_width:.3f}m")
                
                # FINAL SAFETY CHECK: Ensure left < right after coordinate conversion
                # Coordinate conversion should preserve order, but check just in case
                if left_x_vehicle > right_x_vehicle:
                    logger.warning(f"[COORD DEBUG] Order reversed after conversion! Swapping: "
                                 f"left={left_x_vehicle:.3f}m, right={right_x_vehicle:.3f}m")
                    left_x_vehicle, right_x_vehicle = right_x_vehicle, left_x_vehicle
                    calculated_lane_width = right_x_vehicle - left_x_vehicle
                
                # CRITICAL FIX: Validate lane width in vehicle coordinates (meters)
                # This catches perception failures that weren't caught in pixel space
                # Example: Frame 100 had width=1.712m (should be ~7.0m) - this would catch it!
                min_lane_width = 2.0  # Minimum reasonable lane width (meters)
                max_lane_width = 10.0  # Maximum reasonable lane width (meters)
                
                if calculated_lane_width < min_lane_width or calculated_lane_width > max_lane_width:
                    logger.error(f"[Frame {self.frame_count}] [PERCEPTION VALIDATION] ❌ INVALID LANE WIDTH: {calculated_lane_width:.3f}m "
                               f"(expected {min_lane_width}-{max_lane_width}m)")
                    logger.error(f"[Frame {self.frame_count}]   Left lane: {left_x_vehicle:.3f}m, Right lane: {right_x_vehicle:.3f}m")
                    
                    # CRITICAL: On first frame, we have no previous data to fall back to
                    # If width is invalid but not catastrophically wrong, still use it
                    is_first_frame = (self.frame_count == 0)
                    if is_first_frame:
                        # First frame: Be EXTREMELY lenient - we have no previous data to fall back to
                        # Only reject if width is catastrophically wrong (<0.1m or >20m)
                        # Even if width is outside normal range (2-10m), use it on first frame
                        catastrophic_min = 0.1  # Less than 0.1m is definitely wrong (was 0.5m)
                        catastrophic_max = 20.0  # More than 20m is definitely wrong (was 15.0m)
                        if calculated_lane_width < catastrophic_min or calculated_lane_width > catastrophic_max:
                            logger.error(f"[Frame {self.frame_count}]   FIRST FRAME: Catastrophically wrong width ({calculated_lane_width:.3f}m) - rejecting!")
                            left_lane_line_x = None
                            right_lane_line_x = None
                        else:
                            # Width is unusual but not catastrophic - ALWAYS use it on first frame
                            # Better to have some data (even if imperfect) than 0.0
                            if calculated_lane_width < 2.0 or calculated_lane_width > 10.0:
                                logger.warning(f"[Frame {self.frame_count}] ⚠️  FIRST FRAME: Invalid width ({calculated_lane_width:.3f}m) but not catastrophic. Using detection anyway (better than 0.0).")
                            else:
                                logger.info(f"[Frame {self.frame_count}] ✓ FIRST FRAME: Valid width ({calculated_lane_width:.3f}m)")
                            left_lane_line_x = float(left_x_vehicle)
                            right_lane_line_x = float(right_x_vehicle)
                            # Not using stale data - this is the actual detection (even if imperfect)
                            using_stale_data = False
                            stale_data_reason = None
                    else:
                        # Subsequent frames: Reject and use stale data
                        logger.error(f"[Frame {self.frame_count}]   This is a perception failure - rejecting detection!")
                        left_lane_line_x = None
                        right_lane_line_x = None
                        # Track that we're using stale data due to invalid width
                        if self.previous_left_lane_x is not None and self.previous_right_lane_x is not None:
                            using_stale_data = True
                            stale_data_reason = "invalid_width"
                            left_lane_line_x = float(self.previous_left_lane_x)
                            right_lane_line_x = float(self.previous_right_lane_x)
                            logger.warning(f"[Frame {self.frame_count}] Using STALE perception data (reason: {stale_data_reason}) - invalid lane width detected")
                else:
                    # NEW: Validate polynomial coefficients to catch extreme curves
                    # Even if width at lookahead is valid, polynomial can produce extreme values elsewhere
                    # This catches cases like frame 398 where width=7.4m is valid but polynomial goes way outside image
                    image_height = planner.image_height if hasattr(planner, 'image_height') else 480.0
                    image_width = planner.image_width if hasattr(planner, 'image_width') else 640.0
                    
                    # Evaluate polynomial at the lookahead distance where we actually use it
                    # CRITICAL: Don't validate at top of image (y=0) - extrapolation there can be extreme on curves
                    # We only use the polynomial at the lookahead distance (y_image_at_lookahead), so validate there
                    # Also check bottom of image (where lanes are closest to vehicle) for consistency
                    is_first_frame = (self.frame_count == 0)
                    
                    # CRITICAL FIX: Only validate where we actually use the polynomial
                    # The lookahead distance is where we evaluate for coordinate conversion
                    # Top of image (y=0) is extrapolation and can be extreme on curves - that's OK
                    # Bottom of image (y=image_height-1) may have missing dashed lines, causing incorrect extrapolation
                    # We only use the polynomial at the lookahead distance, so only validate there
                    y_check_positions = [y_image_at_lookahead]  # Only check where we actually use it
                    
                    # Use reasonable thresholds - allow some extrapolation for curves
                    max_reasonable_x = image_width * 2.5  # Allow up to 2.5x image width (for curves)
                    min_reasonable_x = -image_width * 1.5  # Allow some negative (for curves)
                    
                    # On first frame, be even more lenient since we have no previous data
                    if is_first_frame:
                        max_reasonable_x = image_width * 3.0  # More lenient for first frame
                        min_reasonable_x = -image_width * 2.0
                    
                    extreme_coeffs_detected = False
                    for lane_idx, lane_coeffs_item in enumerate(lane_coeffs):
                        if lane_coeffs_item is not None:
                            # Convert to numpy array and ensure it's 1D
                            lane_coeffs_array = np.asarray(lane_coeffs_item)
                            if lane_coeffs_array.ndim == 1 and len(lane_coeffs_array) >= 3:
                                # Evaluate polynomial: x = a*y^2 + b*y + c
                                for y_check in y_check_positions:
                                    x_eval = lane_coeffs_array[0] * y_check * y_check + lane_coeffs_array[1] * y_check + lane_coeffs_array[2]
                                    if x_eval < min_reasonable_x or x_eval > max_reasonable_x:
                                        extreme_coeffs_detected = True
                                        logger.error(f"[Frame {self.frame_count}] [PERCEPTION VALIDATION] ❌ EXTREME POLYNOMIAL COEFFICIENTS! "
                                                   f"Lane {lane_idx} at y={y_check}px: x={x_eval:.1f}px (expected {min_reasonable_x:.0f} to {max_reasonable_x:.0f}px)")
                                        logger.error(f"[Frame {self.frame_count}]   Coefficients: a={lane_coeffs_array[0]:.6f}, b={lane_coeffs_array[1]:.6f}, c={lane_coeffs_array[2]:.6f}")
                                        if is_first_frame:
                                            logger.warning(f"[Frame {self.frame_count}]   NOTE: First frame - validation relaxed, but this is still extreme!")
                                        break
                                if extreme_coeffs_detected:
                                    break
                    
                    if extreme_coeffs_detected:
                        # Reject this detection - polynomial produces extreme values
                        logger.error(f"[Frame {self.frame_count}]   This is a perception failure - rejecting detection!")
                        
                        # CRITICAL: On first frame, we have no previous data to fall back to
                        # If validation fails on first frame, we need to either:
                        # 1. Accept the detection anyway (risky but better than no data)
                        # 2. Use a default/straight trajectory
                        # For now, we'll log a warning but still try to use the detection if width is reasonable
                        if is_first_frame:
                            logger.warning(f"[Frame {self.frame_count}] ⚠️  FIRST FRAME: Extreme coefficients detected but no previous data available!")
                            logger.warning(f"[Frame {self.frame_count}]   Will attempt to use detection anyway - width check will catch truly bad detections")
                            # Don't set to None - let it continue to see if width check passes
                            # The width check below will catch truly bad detections
                            extreme_coeffs_detected = False  # Allow it through, but width check will catch it
                        else:
                            # Subsequent frames: Reject and use stale data
                            left_lane_line_x = None
                            right_lane_line_x = None
                            # Track that we're using stale data due to extreme coefficients
                            if self.previous_left_lane_x is not None and self.previous_right_lane_x is not None:
                                using_stale_data = True
                                stale_data_reason = "extreme_coefficients"
                                left_lane_line_x = float(self.previous_left_lane_x)
                                right_lane_line_x = float(self.previous_right_lane_x)
                                logger.warning(f"[Frame {self.frame_count}] Using STALE perception data (reason: {stale_data_reason}) - extreme polynomial coefficients detected")
                    else:
                        # CRITICAL FIX: Check for sudden jumps in lane positions (temporal validation)
                        # If lane position jumps >2m from previous frame, it's likely a false detection
                        if not hasattr(self, 'previous_left_lane_x'):
                            self.previous_left_lane_x = None
                            self.previous_right_lane_x = None
                        
                        # CRITICAL FIX: On first frame (no previous data), accept current detection immediately
                        # Don't skip the entire block - we need to assign values!
                        if self.previous_left_lane_x is None or self.previous_right_lane_x is None:
                            # First frame or no previous data - accept current detection
                            if self.frame_count == 0:
                                logger.debug(f"[Frame {self.frame_count}] [DEBUG] First frame - assigning values: left={left_x_vehicle:.3f}m, right={right_x_vehicle:.3f}m")
                            left_lane_line_x = float(left_x_vehicle)
                            right_lane_line_x = float(right_x_vehicle)
                            if self.frame_count == 0:
                                logger.debug(f"[Frame {self.frame_count}] [DEBUG] After assignment: left_lane_line_x={left_lane_line_x}, right_lane_line_x={right_lane_line_x}")
                            
                            # Update previous values for next frame
                            self.previous_left_lane_x = left_x_vehicle
                            self.previous_right_lane_x = right_x_vehicle
                        elif self.previous_left_lane_x is not None and self.previous_right_lane_x is not None:
                            left_jump = abs(left_x_vehicle - self.previous_left_lane_x)
                            right_jump = abs(right_x_vehicle - self.previous_right_lane_x)
                            # RECOMMENDED: Increased from 2.0m to 3.5m to handle curves better
                            # On curves, lane positions can change by 2-3m between frames when car is turning
                            # 2.0m was too strict and caused false rejections on curves (e.g., frame 201: 3.094m jump)
                            # NEW: Adaptive threshold based on path curvature for even better curve handling
                            path_curvature = vehicle_state_dict.get('groundTruthPathCurvature') or vehicle_state_dict.get('ground_truth_path_curvature', 0.0)
                            abs_curvature = abs(path_curvature)
                            if abs_curvature > 0.1:  # On a curve
                                max_jump_threshold = 4.5  # meters - more lenient for curves (increased from 3.5m)
                            else:  # On straight road
                                max_jump_threshold = 3.5  # meters - standard threshold for straights
                            
                            # CRITICAL FIX: Check for timestamp gap (Unity pause) OR frozen timestamp before applying jump detection
                            # After a Unity pause, large jumps are expected and should be accepted
                            # Also detect frozen timestamps (dt = 0.0) which indicates Unity stopped updating
                            has_time_gap = False
                            has_frozen_timestamp = False
                            if prev_timestamp is None:
                                dt = 0.0  # First frame - no previous timestamp
                            else:
                                dt = timestamp - prev_timestamp
                                if dt > 0.5:  # Large time gap = Unity paused
                                    has_time_gap = True
                                    logger.info(f"[PERCEPTION VALIDATION] Unity pause detected (time gap: {dt:.3f}s) - relaxing jump detection")
                                    # After Unity pause, accept larger jumps (6.0m instead of 3.5m)
                                    max_jump_threshold = 6.0
                                elif abs(dt) < 0.001:  # Timestamp frozen (dt ≈ 0) = Unity stopped updating
                                    has_frozen_timestamp = True
                                    logger.warning(f"[PERCEPTION VALIDATION] ⚠️  FROZEN TIMESTAMP detected (dt={dt:.6f}s) - Unity stopped updating timestamps!")
                                    logger.warning("  This means Unity paused/froze but Python kept processing the same frame")
                                    logger.warning("  Perception will return identical results until Unity resumes")
                                    # Treat frozen timestamp like a time gap - accept any changes when Unity resumes
                                    # But for now, we're still processing the same frame, so values will be identical
                                    # Don't update last_timestamp - keep it as the last valid timestamp
                                    # This will help detect when Unity resumes (timestamp will jump)
                            
                            # CRITICAL: Detect if perception values are frozen (same as previous frame)
                            # This could indicate:
                            # 1. Unity paused (no new frames) - check timestamp gap
                            # 2. Perception died (no new detections, but Unity still sending frames)
                            # 3. Bridge disconnected (no frames at all)
                            # CRITICAL FIX: Only mark as frozen if timestamp is actually frozen OR values unchanged for 3+ frames
                            # This prevents false positives from legitimate repeated detections (1-2 frames)
                            values_match = (abs(left_x_vehicle - self.previous_left_lane_x) < 0.001 and 
                                          abs(right_x_vehicle - self.previous_right_lane_x) < 0.001)
                            
                            # Check timestamp to distinguish actual freeze from legitimate repeated detections
                            if prev_timestamp is None:
                                dt = 0.0  # First frame - no previous timestamp
                            else:
                                dt = timestamp - prev_timestamp
                            timestamp_frozen = abs(dt) < 0.001  # Timestamp not updating = Unity frozen
                            
                            # Track consecutive frames with matching values
                            if not hasattr(self, 'perception_frozen_frames'):
                                self.perception_frozen_frames = 0
                            
                            if values_match:
                                self.perception_frozen_frames += 1
                            else:
                                self.perception_frozen_frames = 0  # Reset counter if values change
                            
                            # Only mark as frozen if timestamp is frozen OR values unchanged for 3+ frames
                            # This prevents false positives from legitimate repeated detections (1-2 frames)
                            perception_frozen = timestamp_frozen or (values_match and self.perception_frozen_frames >= 3)
                            
                            if perception_frozen:
                                # Perception values are frozen - diagnose the cause
                                # Note: perception_frozen_frames counter already updated above
                                if not hasattr(self, 'last_timestamp_before_freeze'):
                                    self.last_timestamp_before_freeze = None
                                
                                # Track that we're using stale data due to frozen perception
                                if not using_stale_data:  # Only set if not already set by jump detection
                                    using_stale_data = True
                                    stale_data_reason = "frozen"
                                    if timestamp_frozen:
                                        logger.warning(f"[Frame {self.frame_count}] Using STALE perception data (reason: {stale_data_reason}) - timestamp frozen (dt={dt:.6f}s)")
                                    else:
                                        logger.warning(f"[Frame {self.frame_count}] Using STALE perception data (reason: {stale_data_reason}) - perception values unchanged for {self.perception_frozen_frames} frames")
                                
                                # Check timestamp to distinguish Unity pause vs perception failure
                                if not hasattr(self, 'last_timestamp_before_freeze'):
                                    self.last_timestamp_before_freeze = None
                                
                                if self.perception_frozen_frames == 1 or self.last_timestamp_before_freeze is None:
                                    # First frozen frame - record previous timestamp
                                    self.last_timestamp_before_freeze = prev_timestamp if prev_timestamp is not None else timestamp
                                
                                # Diagnose cause
                                if timestamp_frozen:  # Frozen timestamp (dt ≈ 0) = Unity stopped updating timestamps
                                    if self.perception_frozen_frames == 1:
                                        logger.warning(f"[PERCEPTION VALIDATION] ⚠️  FROZEN TIMESTAMP detected (dt={dt:.6f}s) - Unity stopped updating!")
                                        logger.warning("  Unity paused/froze but Python kept processing the same frame")
                                        logger.warning("  Perception will return identical results until Unity resumes")
                                    # Don't update last_timestamp - keep it as the last valid timestamp
                                    # This helps detect when Unity resumes (timestamp will jump)
                                elif dt > 0.5:  # Large time gap = Unity paused
                                    if self.perception_frozen_frames == 1:
                                        logger.warning(f"[PERCEPTION VALIDATION] ⚠️  Unity paused (time gap: {dt:.3f}s) - perception values frozen")
                                    # Don't update last_timestamp here - update at end of function
                                elif dt < 0.1:  # Normal time gap but values frozen = perception died
                                    if self.perception_frozen_frames > 3:
                                        logger.error(f"[PERCEPTION VALIDATION] ❌ PERCEPTION FAILED! "
                                                   f"Values unchanged for {self.perception_frozen_frames} frames "
                                                   f"(Unity still sending frames, dt={dt:.3f}s)")
                                        logger.error(f"  Left: {left_x_vehicle:.3f}m, Right: {right_x_vehicle:.3f}m")
                                        logger.error("  This indicates perception processing has stopped - emergency stop needed")
                                    # Don't update last_timestamp here - update at end of function
                                else:
                                    # Medium time gap - could be either
                                    if self.perception_frozen_frames > 3:
                                        logger.warning(f"[PERCEPTION VALIDATION] ⚠️  Perception values frozen for {self.perception_frozen_frames} frames "
                                                     f"(time gap: {dt:.3f}s)")
                                    # Don't update last_timestamp here - update at end of function
                            else:
                                # Perception is updating - check if we just recovered from frozen timestamp
                                if hasattr(self, 'perception_frozen_frames') and self.perception_frozen_frames > 0:
                                    # Check if timestamp jumped (Unity resumed)
                                    if prev_timestamp is not None:
                                        recovery_dt = timestamp - prev_timestamp
                                        if abs(recovery_dt) > 0.001:  # Timestamp is updating again
                                            logger.info(f"[PERCEPTION VALIDATION] ✓ Unity resumed! Timestamp updated (dt={recovery_dt:.3f}s) after {self.perception_frozen_frames} frozen frames")
                                
                                # Reset frozen counter
                                if hasattr(self, 'perception_frozen_frames'):
                                    if self.perception_frozen_frames > 0:
                                        logger.info(f"[PERCEPTION VALIDATION] ✓ Perception recovered after {self.perception_frozen_frames} frozen frames")
                                    self.perception_frozen_frames = 0
                                # Don't update last_timestamp here - update at end of function
                            
                            if left_jump > max_jump_threshold or right_jump > max_jump_threshold:
                                if has_time_gap:
                                    # After Unity pause, accept the jump (it's expected)
                                    logger.info("[PERCEPTION VALIDATION] Large jump after Unity pause - accepting detection")
                                    logger.info(f"  Left lane jump: {left_jump:.3f}m (threshold: {max_jump_threshold}m)")
                                    logger.info(f"  Right lane jump: {right_jump:.3f}m (threshold: {max_jump_threshold}m)")
                                    logger.info(f"  Previous: left={self.previous_left_lane_x:.3f}m, right={self.previous_right_lane_x:.3f}m")
                                    logger.info(f"  Current: left={left_x_vehicle:.3f}m, right={right_x_vehicle:.3f}m")
                                    # Accept the new detection after Unity pause
                                    left_lane_line_x = float(left_x_vehicle)
                                    right_lane_line_x = float(right_x_vehicle)
                                    # Update previous values
                                    self.previous_left_lane_x = left_x_vehicle
                                    self.previous_right_lane_x = right_x_vehicle
                                else:
                                    if detection_method == "segmentation":
                                        # Segmentation outputs can change more between frames without being wrong.
                                        # Accept the new detection to avoid freezing on stale data.
                                        left_lane_line_x = float(left_x_vehicle)
                                        right_lane_line_x = float(right_x_vehicle)
                                        self.previous_left_lane_x = left_x_vehicle
                                        self.previous_right_lane_x = right_x_vehicle
                                        logger.info(
                                            f"[Frame {self.frame_count}] [PERCEPTION VALIDATION] Segmentation jump accepted "
                                            f"(left={left_jump:.3f}m, right={right_jump:.3f}m, threshold={max_jump_threshold:.3f}m)"
                                        )
                                        # Skip stale handling for segmentation
                                        using_stale_data = False
                                        stale_data_reason = None
                                    else:
                                        # Normal jump detection (no time gap) - reject if too large
                                        logger.error(f"[PERCEPTION VALIDATION] ❌ SUDDEN LANE JUMP DETECTED! (Frame {self.frame_count})")
                                        logger.error(f"  Left lane jump: {left_jump:.3f}m (threshold: {max_jump_threshold}m)")
                                        logger.error(f"  Right lane jump: {right_jump:.3f}m (threshold: {max_jump_threshold}m)")
                                        logger.error(f"  Previous: left={self.previous_left_lane_x:.3f}m, right={self.previous_right_lane_x:.3f}m")
                                        logger.error(f"  Current: left={left_x_vehicle:.3f}m, right={right_x_vehicle:.3f}m")
                                        logger.error("  Rejecting detection - using previous frame's values")
                                        
                                        # Use previous frame's values instead of current (failed) detection
                                        left_lane_line_x = float(self.previous_left_lane_x)
                                        right_lane_line_x = float(self.previous_right_lane_x)
                                        # Track that we're using stale data due to jump detection
                                        using_stale_data = True
                                        stale_data_reason = "jump_detection"
                                        left_jump_magnitude = left_jump
                                        right_jump_magnitude = right_jump
                                        jump_threshold_used = max_jump_threshold
                                        logger.warning(f"[Frame {self.frame_count}] Using STALE perception data (reason: {stale_data_reason}) - jump detected: left={left_jump:.3f}m, right={right_jump:.3f}m (threshold={jump_threshold_used:.3f}m)")
                            else:
                                # NEW: Detect lane width/center changes that indicate perception instability
                                # Even if jump detection didn't trigger, small changes can cause control errors
                                # If lane width or center changes significantly, use stale data to prevent control oscillations
                                lane_center_current = (left_x_vehicle + right_x_vehicle) / 2.0
                                lane_width_current = calculated_lane_width
                                
                                if self.previous_left_lane_x is not None and self.previous_right_lane_x is not None:
                                    lane_center_previous = (self.previous_left_lane_x + self.previous_right_lane_x) / 2.0
                                    lane_width_previous = self.previous_right_lane_x - self.previous_left_lane_x
                                    
                                    # NEW: Adaptive thresholds based on path curvature
                                    # Get path curvature from vehicle state (ground truth if available)
                                    path_curvature = vehicle_state_dict.get('groundTruthPathCurvature') or vehicle_state_dict.get('ground_truth_path_curvature', 0.0)
                                    abs_curvature = abs(path_curvature)
                                    
                                    # On curves (|curvature| > 0.1 1/m), lane positions change naturally
                                    # Use more lenient thresholds to avoid rejecting legitimate curve-induced changes
                                    if abs_curvature > 0.1:  # On a curve
                                        width_change_threshold = 1.5  # meters - much more lenient for curves (increased from 1.3m to 1.5m)
                                        center_shift_threshold = 1.2  # meters - more lenient for curves (increased from 0.8m to 1.2m)
                                        threshold_mode = "curve"
                                    else:  # On straight road
                                        width_change_threshold = 0.3  # meters - strict for straights (catches noise)
                                        center_shift_threshold = 0.2  # meters - strict for straights (catches noise)
                                        threshold_mode = "straight"
                                    
                                    # Detect significant lane width change (threshold depends on curvature)
                                    width_change = abs(lane_width_current - lane_width_previous)
                                    
                                    # Detect significant lane center shift (threshold depends on curvature)
                                    center_shift = abs(lane_center_current - lane_center_previous)
                                    
                                    # CRITICAL FIX: Skip instability check on first frame with previous data (frame 1)
                                    # On frame 1, this is the first comparison, and small changes are expected
                                    # due to normal perception variation. Only check instability from frame 2 onwards.
                                    skip_instability_check = (self.frame_count == 1)  # Skip on frame 1 only
                                    
                                    if not skip_instability_check and (width_change > width_change_threshold or center_shift > center_shift_threshold):
                                        if detection_method == "segmentation":
                                            # Segmentation can have larger natural shifts without being wrong.
                                            # Accept the new detection to avoid freezing.
                                            left_lane_line_x = float(left_x_vehicle)
                                            right_lane_line_x = float(right_x_vehicle)
                                            self.previous_left_lane_x = left_x_vehicle
                                            self.previous_right_lane_x = right_x_vehicle
                                            logger.info(
                                                f"[Frame {self.frame_count}] [PERCEPTION VALIDATION] Segmentation instability accepted "
                                                f"(width_change={width_change:.3f}m, center_shift={center_shift:.3f}m)"
                                            )
                                            using_stale_data = False
                                            stale_data_reason = None
                                        else:
                                            # Perception instability detected - use stale data to prevent control oscillations
                                            logger.warning(f"[Frame {self.frame_count}] [PERCEPTION VALIDATION] ⚠️  PERCEPTION INSTABILITY DETECTED!")
                                            logger.warning(f"  Width change: {width_change:.3f}m (threshold: {width_change_threshold}m [{threshold_mode}])")
                                            logger.warning(f"  Center shift: {center_shift:.3f}m (threshold: {center_shift_threshold}m [{threshold_mode}])")
                                            logger.warning(f"  Path curvature: {path_curvature:.4f} 1/m (|curvature|={abs_curvature:.4f})")
                                            logger.warning("  Using stale data to prevent control oscillations")
                                            
                                            using_stale_data = True
                                            stale_data_reason = "perception_instability"
                                            # Store actual detected values for debugging (even though we reject them)
                                            actual_detected_left_lane_x = float(left_x_vehicle)
                                            actual_detected_right_lane_x = float(right_x_vehicle)
                                            instability_width_change = width_change
                                            instability_center_shift = center_shift
                                            # Use stale values for control
                                            left_lane_line_x = float(self.previous_left_lane_x)
                                            right_lane_line_x = float(self.previous_right_lane_x)
                                            # CRITICAL FIX: Don't update previous values when instability detected
                                            # This allows recovery: next frame will compare against stale values
                                        # If next frame's detection is within thresholds vs stale values, recovery happens
                                    else:
                                        # Valid detection - use it (either first valid detection or recovery from instability)
                                        if using_stale_data and stale_data_reason == "perception_instability":
                                            logger.info(f"[Frame {self.frame_count}] [PERCEPTION VALIDATION] ✅ RECOVERED from perception instability!")
                                            logger.info(f"  Width change: {width_change:.3f}m (threshold: {width_change_threshold}m [{threshold_mode}])")
                                            logger.info(f"  Center shift: {center_shift:.3f}m (threshold: {center_shift_threshold}m [{threshold_mode}])")
                                            logger.info("  Accepting new detection and updating previous values")
                                        
                                        left_lane_line_x = float(left_x_vehicle)
                                        right_lane_line_x = float(right_x_vehicle)
                                        
                                        # Update previous values for next frame (enables recovery)
                                        self.previous_left_lane_x = left_x_vehicle
                                        self.previous_right_lane_x = right_x_vehicle
                                else:
                                    # First frame or no previous data - accept current detection
                                    if self.frame_count == 0:
                                        logger.debug(f"[Frame {self.frame_count}] [DEBUG] First frame - assigning values: left={left_x_vehicle:.3f}m, right={right_x_vehicle:.3f}m")
                                    left_lane_line_x = float(left_x_vehicle)
                                    right_lane_line_x = float(right_x_vehicle)
                                    if self.frame_count == 0:
                                        logger.debug(f"[Frame {self.frame_count}] [DEBUG] After assignment: left_lane_line_x={left_lane_line_x}, right_lane_line_x={right_lane_line_x}")
                                    
                                    # Update previous values for next frame
                                    self.previous_left_lane_x = left_x_vehicle
                                    self.previous_right_lane_x = right_x_vehicle
            except Exception as e:  # CHANGED: Catch ALL exceptions, not just AttributeError/TypeError/ValueError
                # If planner is a Mock or conversion fails, skip lane data calculation
                # This is acceptable in unit tests
                # CRITICAL: Log the exception so we can diagnose why coordinate conversion failed
                import traceback
                logger.error(f"[Frame {self.frame_count}] [COORD CONVERSION] Exception during coordinate conversion: {type(e).__name__}: {e}")
                logger.error(f"[Frame {self.frame_count}]   Traceback:\n{traceback.format_exc()}")
                logger.error(f"[Frame {self.frame_count}]   This means left_lane_line_x and right_lane_line_x will remain None (converted to 0.0 in recorder)")
                logger.error(f"[Frame {self.frame_count}]   If overlays look good, this is a bug - coordinate conversion should work!")
                
                # DEBUG: Log what we have
                logger.error(f"[Frame {self.frame_count}]   DEBUG: lane_coeffs type={type(lane_coeffs)}, len={len(lane_coeffs) if hasattr(lane_coeffs, '__len__') else 'N/A'}")
                if len(lane_coeffs) >= 2:
                    logger.error(f"[Frame {self.frame_count}]   DEBUG: lane_coeffs[0] type={type(lane_coeffs[0])}, shape={lane_coeffs[0].shape if hasattr(lane_coeffs[0], 'shape') else 'N/A'}")
                    logger.error(f"[Frame {self.frame_count}]   DEBUG: lane_coeffs[1] type={type(lane_coeffs[1])}, shape={lane_coeffs[1].shape if hasattr(lane_coeffs[1], 'shape') else 'N/A'}")
                logger.error(f"[Frame {self.frame_count}]   DEBUG: planner type={type(planner) if 'planner' in locals() else 'not defined'}")
                if 'planner' in locals():
                    logger.error(f"[Frame {self.frame_count}]   DEBUG: planner has image_height={hasattr(planner, 'image_height')}, image_width={hasattr(planner, 'image_width')}")
                    logger.error(f"[Frame {self.frame_count}]   DEBUG: planner has _convert_image_to_vehicle_coords={hasattr(planner, '_convert_image_to_vehicle_coords')}")
                
                # On first frame, try to use a fallback if we have valid coefficients
                if self.frame_count == 0:
                    logger.warning(f"[Frame {self.frame_count}] ⚠️  FIRST FRAME: Coordinate conversion failed but overlays look good!")
                    logger.warning(f"[Frame {self.frame_count}]   This suggests a bug in coordinate conversion, not perception")
                    # Don't set to None - leave as None so the check at line 1013 can catch it
                    # But log a warning so we know what happened
                pass
        elif num_lanes_detected == 1:
            # FIX: Handle single-lane-line detection case
            # When only 1 lane line is detected, estimate the other lane line based on standard lane width
            # CRITICAL: Check if we have valid coefficients (not None)
            valid_coeffs = [c for c in lane_coeffs if c is not None]
            if len(valid_coeffs) >= 1:
                try:
                    planner = self.trajectory_planner.planner if hasattr(self.trajectory_planner, 'planner') else self.trajectory_planner
                    image_height = planner.image_height if hasattr(planner, 'image_height') else 480.0
                    image_width = planner.image_width if hasattr(planner, 'image_width') else 640.0
                    reference_lookahead = planner.lookahead_distance if hasattr(planner, 'lookahead_distance') else 8.0
                    
                    # Evaluate single lane line at lookahead
                    y_image_at_lookahead = 350  # Within detected range
                    single_lane_coeffs = valid_coeffs[0]  # Use first valid coefficient
                    single_x_image = np.polyval(single_lane_coeffs, y_image_at_lookahead)
                    
                    # Convert to vehicle coordinates
                    conversion_distance = reference_lookahead
                    # Use Unity's reported horizontal FOV if available
                    camera_horizontal_fov = vehicle_state_dict.get('cameraHorizontalFOV')
                    if camera_horizontal_fov is None:
                        camera_horizontal_fov = vehicle_state_dict.get('camera_horizontal_fov', None)
                    
                    single_x_vehicle, _ = planner._convert_image_to_vehicle_coords(
                        single_x_image, y_image_at_lookahead, lookahead_distance=conversion_distance,
                        horizontal_fov_override=camera_horizontal_fov if camera_horizontal_fov and camera_horizontal_fov > 0 else None
                    )
                    
                    # Estimate other lane based on standard lane width (3.5m per lane, 7m total)
                    standard_lane_width = 7.0  # meters
                    lane_half_width = standard_lane_width / 2.0
                    
                    # Determine if detected lane is left or right based on position relative to image center
                    image_center_x = image_width / 2.0
                    if single_x_image < image_center_x:
                        # Detected lane line is on left side - it's the left lane line
                        left_lane_line_x = float(single_x_vehicle)
                        right_lane_line_x = float(single_x_vehicle + standard_lane_width)
                    else:
                        # Detected lane line is on right side - it's the right lane line
                        left_lane_line_x = float(single_x_vehicle - standard_lane_width)
                        right_lane_line_x = float(single_x_vehicle)
                    
                    logger.warning(f"[SINGLE LANE] Only 1 lane line detected. Detected at x={single_x_vehicle:.3f}m. "
                                 f"Estimated: left={left_lane_line_x:.3f}m, right={right_lane_line_x:.3f}m")
                except (AttributeError, TypeError, ValueError) as e:
                    logger.warning(f"[SINGLE LANE] Failed to process single lane line: {e}")
                    # Leave as 0.0 (default)
                    pass
            else:
                # num_lanes_detected = 1 but no valid coefficients - detection failed
                logger.warning(f"[SINGLE LANE] num_lanes_detected=1 but no valid coefficients! "
                             f"lane_coeffs={lane_coeffs}. Detection may have failed.")
                # Leave as 0.0 (default)
        
        # CRITICAL FIX: If both lanes are 0.0 (perception failed), set to None
        # This ensures reference_point will be None, preventing movement
        if left_lane_line_x is not None and right_lane_line_x is not None:
            if abs(left_lane_line_x) < 0.001 and abs(right_lane_line_x) < 0.001:
                logger.warning("[PERCEPTION] Perception failed (both lanes = 0.0) - setting to None to prevent movement")
                left_lane_line_x = None
                right_lane_line_x = None
        
        # Record perception output with diagnostic fields
        # NEW: Perception health monitoring
        # Track detection quality (good = 2+ lanes detected)
        is_good_detection = num_lanes_detected >= 2
        
        # NEW: Check for bad events that should reduce health score immediately
        bad_events = []
        
        # Check 1: Insufficient lanes
        if not is_good_detection:
            bad_events.append("insufficient_lanes")
        
        # Check 2: Extreme polynomial coefficients (if we have coefficients)
        # CRITICAL: Only check where we actually use the polynomial (lookahead distance)
        # Don't check top of image (y=0) - extrapolation there can be extreme on curves
        if lane_coeffs is not None and len(lane_coeffs) >= 2:
            # Get image dimensions (try to get from planner, fallback to defaults)
            try:
                planner = self.trajectory_planner.planner if hasattr(self.trajectory_planner, 'planner') else self.trajectory_planner
                image_height = planner.image_height if hasattr(planner, 'image_height') else 480.0
                image_width = planner.image_width if hasattr(planner, 'image_width') else 640.0
            except:
                image_height = 480.0
                image_width = 640.0
            
            # Get lookahead y position - use camera_8m_screen_y if available (extracted earlier in _process_frame)
            # Otherwise use middle of image as fallback
            if camera_8m_screen_y is not None and camera_8m_screen_y > 0:
                health_y_image_at_lookahead = int(camera_8m_screen_y)
            else:
                # Fallback: Use middle of image (typical lookahead position)
                health_y_image_at_lookahead = int(image_height * 0.7)  # Bottom 30% of image
            
            # Only check at lookahead distance and bottom (where we actually use the polynomial)
            # Only check at lookahead distance (where we actually use the polynomial)
            # Don't check bottom - missing dashed lines can cause incorrect extrapolation there
            health_check_y_positions = [health_y_image_at_lookahead]
            
            # Use same thresholds as validation
            health_max_reasonable_x = image_width * 2.5
            health_min_reasonable_x = -image_width * 1.5
            
            for lane_idx, lane_coeffs_item in enumerate(lane_coeffs):
                # Check if lane_coeffs_item is a valid array (not None, not a scalar)
                if lane_coeffs_item is not None:
                    # Convert to numpy array if needed and check shape
                    lane_coeffs_array = np.asarray(lane_coeffs_item)
                    if lane_coeffs_array.ndim == 1 and len(lane_coeffs_array) >= 3:
                        for y_check in health_check_y_positions:
                            x_eval = lane_coeffs_array[0] * y_check * y_check + lane_coeffs_array[1] * y_check + lane_coeffs_array[2]
                            if x_eval < health_min_reasonable_x or x_eval > health_max_reasonable_x:
                                bad_events.append("extreme_coefficients")
                                break
                        if "extreme_coefficients" in bad_events:
                            break
        
        # Check 3: Using stale data (indicates perception issues)
        if using_stale_data:
            bad_events.append("stale_data")
        
        # Track health: good detection AND no bad events = truly good
        is_truly_good = is_good_detection and len(bad_events) == 0
        
        # DEBUG: Log health monitoring on early frames to diagnose issues
        if self.frame_count <= 5:
            logger.debug(f"[Frame {self.frame_count}] [HEALTH DEBUG] is_good_detection={is_good_detection}, num_lanes={num_lanes_detected}, bad_events={bad_events}, is_truly_good={is_truly_good}")
        
        self.perception_health_history.append(is_truly_good)
        if len(self.perception_health_history) > self.perception_health_history_size:
            self.perception_health_history.pop(0)
        
        # Update consecutive bad detection frames
        if is_truly_good:
            self.consecutive_bad_detection_frames = 0
        else:
            self.consecutive_bad_detection_frames += 1
            # DEBUG: Log why frame is marked as bad
            if self.frame_count <= 5:
                logger.debug(f"[Frame {self.frame_count}] [HEALTH DEBUG] Frame marked as BAD: consecutive_bad={self.consecutive_bad_detection_frames}, is_good_detection={is_good_detection}, bad_events={bad_events}")
        
        # Calculate health score (fraction of recent frames with truly good detection)
        # NEW: Penalize frames with bad events even if detection rate is good
        if len(self.perception_health_history) > 0:
            good_detection_count = sum(self.perception_health_history)
            base_score = good_detection_count / len(self.perception_health_history)
            
            # Apply immediate penalty for current frame bad events
            # Each bad event reduces score by 0.1 (can stack)
            penalty = min(0.3, len(bad_events) * 0.1)  # Max 0.3 penalty
            self.perception_health_score = max(0.0, base_score - penalty)
        else:
            self.perception_health_score = 1.0 if len(bad_events) == 0 else 0.7
        
        # Determine health status
        if self.perception_health_score >= 0.9:
            self.perception_health_status = "healthy"
        elif self.perception_health_score >= 0.7:
            self.perception_health_status = "degraded"
        elif self.perception_health_score >= 0.5:
            self.perception_health_status = "poor"
        else:
            self.perception_health_status = "critical"
        
        # Log health degradation warnings
        if self.consecutive_bad_detection_frames >= 10 and self.consecutive_bad_detection_frames % 10 == 0:
            logger.warning(f"[Frame {self.frame_count}] ⚠️  Perception health degraded: {self.consecutive_bad_detection_frames} consecutive bad frames, "
                         f"health score: {self.perception_health_score:.2f} ({self.perception_health_status})")
        
        # CRITICAL: Update last_timestamp at the END of processing (after all timestamp comparisons)
        self.last_timestamp = timestamp

        # Fallback: if lane positions are missing but we have previous values, use stale data
        # This prevents left/right/width from collapsing to 0.0 when validation rejects positions.
        if (left_lane_line_x is None or right_lane_line_x is None) and \
           hasattr(self, 'previous_left_lane_x') and hasattr(self, 'previous_right_lane_x') and \
           self.previous_left_lane_x is not None and self.previous_right_lane_x is not None:
            using_stale_data = True
            stale_data_reason = stale_data_reason or "missing_lane_positions"
            left_lane_line_x = float(self.previous_left_lane_x)
            right_lane_line_x = float(self.previous_right_lane_x)
            logger.warning(
                f"[Frame {self.frame_count}] Using STALE perception data (reason: {stale_data_reason}) - "
                "lane positions missing after validation"
            )
        
        perception_output = PerceptionOutput(
            timestamp=timestamp,
            lane_line_coefficients=np.array([c for c in lane_coeffs if c is not None]) if any(c is not None for c in lane_coeffs) else None,
            confidence=confidence,
            detection_method=detection_method,
            num_lanes_detected=num_lanes_detected,
            left_lane_line_x=left_lane_line_x,
            right_lane_line_x=right_lane_line_x,
            # Diagnostic fields for tracking stale data usage
            using_stale_data=using_stale_data,
            stale_data_reason=stale_data_reason,
            left_jump_magnitude=left_jump_magnitude,
            right_jump_magnitude=right_jump_magnitude,
            jump_threshold=jump_threshold_used,
            # NEW: Diagnostic fields for perception instability
            actual_detected_left_lane_x=actual_detected_left_lane_x,
            actual_detected_right_lane_x=actual_detected_right_lane_x,
            instability_width_change=instability_width_change,
            instability_center_shift=instability_center_shift,
            # NEW: Perception health monitoring fields
            consecutive_bad_detection_frames=self.consecutive_bad_detection_frames,
            perception_health_score=self.perception_health_score,
            perception_health_status=self.perception_health_status,
            # NEW: Points used for polynomial fitting (for debug visualization)
            fit_points_left=fit_points_left,
            fit_points_right=fit_points_right
        )
        
        # NEW: Apply gradual slowdown based on perception health
        # Reduce target speed when perception health degrades (but not emergency stop)
        if self.perception_health_status == "healthy":
            # Full speed
            adjusted_target_speed = self.original_target_speed
        elif self.perception_health_status == "degraded":
            # Reduce to 80% of target speed
            adjusted_target_speed = self.original_target_speed * 0.8
        elif self.perception_health_status == "poor":
            # Reduce to 50% of target speed
            adjusted_target_speed = self.original_target_speed * 0.5
        else:  # critical
            # Reduce to 30% of target speed (very slow, but still moving)
            adjusted_target_speed = self.original_target_speed * 0.3

        # Apply speed limits from map + curvature
        path_curvature = vehicle_state_dict.get('groundTruthPathCurvature')
        if path_curvature is None:
            path_curvature = vehicle_state_dict.get('ground_truth_path_curvature', 0.0)
        max_lateral_accel = self.trajectory_config.get('max_lateral_accel', 2.5)
        min_curve_speed = self.trajectory_config.get('min_curve_speed', 0.0)
        base_speed = adjusted_target_speed
        smoothed_speed_limit = self._smooth_speed_limit(speed_limit, timestamp)
        adjusted_target_speed = self._apply_speed_limits(
            adjusted_target_speed,
            speed_limit=smoothed_speed_limit,
            path_curvature=path_curvature,
            max_lateral_accel=max_lateral_accel,
            min_curve_speed=min_curve_speed,
        )
        preview_decel = self.trajectory_config.get('speed_limit_preview_decel', 0.0)
        if isinstance(speed_limit_preview, (int, float)) and speed_limit_preview > 0:
            preview_target = _apply_speed_limit_preview(
                adjusted_target_speed,
                float(speed_limit_preview),
                float(speed_limit_preview_distance),
                float(preview_decel),
            )
            adjusted_target_speed = min(adjusted_target_speed, preview_target)
        target_speed_post_limits = adjusted_target_speed
        current_speed = vehicle_state_dict.get('speed', 0.0)
        target_speed_planned = None
        if self.speed_planner_enabled and self.speed_planner is not None:
            planned_speed, planned_accel, planned_jerk, planner_active = self.speed_planner.step(
                adjusted_target_speed,
                current_speed=current_speed,
                timestamp=timestamp,
            )
            adjusted_target_speed = planned_speed
            target_speed_planned = planned_speed
            target_speed_ramp_active = False
            target_speed_slew_active = False
        else:
            adjusted_target_speed, target_speed_ramp_active, target_speed_slew_active = (
                self._apply_target_speed_ramp(adjusted_target_speed, current_speed, timestamp)
            )
        target_speed_final = adjusted_target_speed
        if adjusted_target_speed < base_speed and self.frame_count % 30 == 0:
            curve_speed = None
            if isinstance(path_curvature, (int, float)) and abs(path_curvature) > 1e-6 and max_lateral_accel > 0:
                curve_speed = (max_lateral_accel / abs(path_curvature)) ** 0.5
                if min_curve_speed > 0:
                    curve_speed = max(curve_speed, min_curve_speed)
            logger.info(
                f"[Frame {self.frame_count}] Speed limit applied: "
                f"base={base_speed:.2f} m/s, "
                f"map_limit={float(speed_limit) if speed_limit else 0.0:.2f} m/s, "
                f"curve_limit={curve_speed:.2f} m/s"
                if curve_speed is not None
                else f"[Frame {self.frame_count}] Speed limit applied: "
                     f"base={base_speed:.2f} m/s, "
                     f"map_limit={float(speed_limit) if speed_limit else 0.0:.2f} m/s"
            )
        
        # Update trajectory planner's target speed (if it supports dynamic updates)
        if hasattr(self.trajectory_planner, 'planner') and hasattr(self.trajectory_planner.planner, 'target_speed'):
            self.trajectory_planner.planner.target_speed = adjusted_target_speed
        elif hasattr(self.trajectory_planner, 'target_speed'):
            self.trajectory_planner.target_speed = adjusted_target_speed
        
        # Log speed adjustment if health is degraded
        if self.perception_health_status != "healthy" and self.frame_count % 30 == 0:  # Log every second
            logger.warning(f"[Frame {self.frame_count}] ⚠️  Perception health-based speed reduction: "
                         f"{self.perception_health_status} -> target speed: {adjusted_target_speed:.2f} m/s "
                         f"(original: {self.original_target_speed:.2f} m/s)")
        
        # 2. Trajectory Planning: Plan path
        trajectory = self.trajectory_planner.plan(lane_coeffs, vehicle_state_dict)
        
        # Get current speed for safety checks (check BEFORE computing control)
        max_speed = self.safety_config.get('max_speed', 10.0)
        emergency_threshold = self.safety_config.get('emergency_brake_threshold', 2.0)
        prevention_threshold = self.safety_config.get('speed_prevention_threshold', 0.85)
        prevention_brake_threshold = self.safety_config.get('speed_prevention_brake_threshold', 0.9)
        prevention_brake_amount = self.safety_config.get('speed_prevention_brake_amount', 0.2)
        
        # NEW: Safety bounds for lateral error (hard limits)
        # Critical: 89.4% steering direction errors and 8.95m max error indicate need for safety bounds
        max_lateral_error = self.safety_config.get('max_lateral_error', 2.0)  # Hard limit: 2.0m
        recovery_lateral_error = self.safety_config.get('recovery_lateral_error', 1.0)  # Recovery trigger: 1.0m
        emergency_stop_error = self.safety_config.get('emergency_stop_error', 3.0)  # Emergency stop: 3.0m
        
        # Initialize reference_point for logging
        reference_point = None
        
        # Speed prevention: Check speed BEFORE computing control to prevent acceleration
        # Emergency brake if speed is dangerously high
        # FIXED: Still compute steering during braking for safety (steer while braking)
        # CRITICAL FIX: Make emergency brake more gradual to prevent stop-start cycle
        emergency_brake = False
        brake_override = 0.0
        if current_speed > max_speed * emergency_threshold:
            # Emergency brake: Speed is dangerously high (> 1.5x max = 15 m/s)
            # Use proportional braking instead of full brake to prevent sudden stops
            speed_excess = current_speed - (max_speed * emergency_threshold)
            # Proportional brake: 0.6 base + up to 0.4 more for extreme overspeed
            brake_override = min(1.0, 0.6 + (speed_excess / 5.0))
            emergency_brake = True
            logger.warning(f"Emergency brake! Speed: {current_speed:.2f} m/s, Brake: {brake_override:.2f}")
        elif current_speed > max_speed:
            # Over speed limit - proportional braking (not full brake)
            speed_excess = current_speed - max_speed
            # Gradual braking: 0.4 base + proportional to excess (max 0.8 total)
            brake_override = min(0.8, 0.4 + (speed_excess / 2.0))
            logger.warning(f"Speed limit exceeded! Speed: {current_speed:.2f} m/s, Brake: {brake_override:.2f}")
        
        # Always compute steering (even during braking) for safety
        # Get reference point - use configurable lookahead
        # NEW: Use direct midpoint computation (simpler, more accurate)
        reference_lookahead = self.trajectory_config.get('reference_lookahead', 8.0)
        # Pass timestamp for jump detection (handles Unity pauses)
        reference_point = self.trajectory_planner.get_reference_point(
            trajectory, 
            lookahead=reference_lookahead,
            lane_coeffs=lane_coeffs,  # Pass lane coefficients for direct computation
            lane_positions={'left_lane_line_x': left_lane_line_x, 'right_lane_line_x': right_lane_line_x},  # Preferred: use vehicle coords
            use_direct=True,  # Use direct midpoint computation
            timestamp=timestamp  # Pass timestamp for time gap detection
        )
        
        if reference_point is None:
            # No valid trajectory - stop vehicle safely
            # FIXED: Still compute steering from last known reference or use zero
            if current_speed > 1.0:
                # Apply brake to slow down (proportional to speed)
                brake_override = max(brake_override, min(1.0, 0.3 + (current_speed / 10.0)))
            
            # Use zero steering when no trajectory (safety fallback)
            control_command = {
                'steering': 0.0, 
                'throttle': 0.0, 
                'brake': brake_override,
                'lateral_error': 0.0,
                'heading_error': 0.0,
                'total_error': 0.0,
                'pid_integral': 0.0,
                'pid_derivative': 0.0
            }
        else:
                # Post-jump cooldown: reduce aggressiveness for a short window after teleport
                if self.post_jump_cooldown_frames > 0:
                    cooldown_scale = self.safety_config.get('post_jump_speed_scale', 0.5)
                    reference_point['velocity'] = reference_point.get('velocity', 8.0) * cooldown_scale

                # Limit reference velocity to prevent requesting high speeds
                if reference_point.get('velocity', 8.0) > max_speed:
                    reference_point['velocity'] = max_speed * 0.9  # 90% of max for safety
                
                # 3. Control: Compute control commands
                current_state = {
                    'heading': self._extract_heading(vehicle_state_dict),
                    'speed': current_speed,
                    'position': self._extract_position(vehicle_state_dict)
                }
                
                # Get control command with metadata for recording
                control_command = self.controller.compute_control(
                    current_state, reference_point, return_metadata=True
                )
                control_command['target_speed_raw'] = base_speed
                control_command['target_speed_post_limits'] = target_speed_post_limits
                control_command['target_speed_planned'] = target_speed_planned
                control_command['target_speed_final'] = target_speed_final
                control_command['target_speed_slew_active'] = target_speed_slew_active
                control_command['target_speed_ramp_active'] = target_speed_ramp_active
                
                # If emergency braking was triggered, override throttle/brake but keep steering
                if brake_override > 0:
                    control_command['throttle'] = 0.0
                    control_command['brake'] = brake_override
                    # Keep steering for safety (steer while braking)
                
                # Post-jump cooldown: clamp steering/throttle to avoid oscillation
                if self.post_jump_cooldown_frames > 0:
                    steering_scale = self.safety_config.get('post_jump_steering_scale', 0.6)
                    control_command['steering'] = np.clip(
                        control_command.get('steering', 0.0) * steering_scale,
                        -self.controller.lateral_controller.max_steering,
                        self.controller.lateral_controller.max_steering,
                    )
                    control_command['throttle'] = control_command.get('throttle', 0.0) * cooldown_scale
                
                # Speed prevention: Limit throttle when approaching speed limit
                # FIXED: Start prevention earlier (at 70% instead of 85%)
                longitudinal_cfg = self.control_config.get('longitudinal', {})
                throttle_limit_threshold = longitudinal_cfg.get('throttle_limit_threshold', 0.70)  # FIXED: was 0.8
                throttle_reduction_factor = longitudinal_cfg.get('throttle_reduction_factor', 0.2)  # FIXED: was 0.3
                
                # Start prevention earlier - at 70% of max speed (7.0 m/s for 10 m/s max)
                if current_speed > max_speed * prevention_threshold:
                    # Aggressively reduce throttle to prevent exceeding limit
                    # FIXED: More aggressive reduction
                    throttle_reduction = 1.0 - ((current_speed - max_speed * prevention_threshold) / (max_speed * (1.0 - prevention_threshold)))
                    throttle_reduction = max(0.0, min(1.0, throttle_reduction))
                    control_command['throttle'] = control_command.get('throttle', 0.0) * throttle_reduction * throttle_reduction_factor
                    
                    # If still accelerating, apply light brake (FIXED: start at 80% instead of 90%)
                    if current_speed > max_speed * prevention_brake_threshold:
                        control_command['throttle'] = 0.0
                        # FIXED: More aggressive brake amount (was 0.2, now 0.4)
                        control_command['brake'] = max(control_command.get('brake', 0.0), prevention_brake_amount)
                
                # Launch throttle ramp: cap throttle during stop -> start transitions
                launch_cap_active = False
                launch_cap = 0.0
                if not self.speed_planner_enabled:
                    launch_stop_threshold = longitudinal_cfg.get('launch_throttle_stop_threshold', 0.5)
                    launch_ramp_seconds = longitudinal_cfg.get('launch_throttle_ramp_seconds', 0.0)
                    launch_cap_min = longitudinal_cfg.get('launch_throttle_cap_min', 0.1)
                    launch_cap_max = longitudinal_cfg.get('launch_throttle_cap_max', 0.6)
                    launch_rearm_hysteresis = longitudinal_cfg.get('launch_throttle_rearm_hysteresis', 0.5)
                    launch_stop_hold_seconds = longitudinal_cfg.get('launch_throttle_stop_hold_seconds', 0.5)
                    launch_rearm_speed = float(launch_stop_threshold) + float(launch_rearm_hysteresis)
                    if launch_ramp_seconds > 0.0:
                        if current_speed > launch_rearm_speed:
                            self.launch_throttle_ramp_armed = True
                        if current_speed <= launch_stop_threshold:
                            if self.launch_stop_candidate_start_time is None:
                                self.launch_stop_candidate_start_time = float(timestamp)
                        else:
                            self.launch_stop_candidate_start_time = None
                        if self.launch_throttle_ramp_start_time is None and self.launch_throttle_ramp_armed:
                            if (
                                current_speed <= launch_stop_threshold
                                and control_command.get('throttle', 0.0) > 0.0
                                and self.launch_stop_candidate_start_time is not None
                                and (float(timestamp) - self.launch_stop_candidate_start_time) >= float(launch_stop_hold_seconds)
                            ):
                                self.launch_throttle_ramp_start_time = float(timestamp)
                                self.launch_throttle_ramp_armed = False
                        if self.launch_throttle_ramp_start_time is not None:
                            elapsed = max(0.0, float(timestamp) - self.launch_throttle_ramp_start_time)
                            if elapsed >= float(launch_ramp_seconds):
                                self.launch_throttle_ramp_start_time = None
                            else:
                                ratio = min(1.0, elapsed / float(launch_ramp_seconds))
                                launch_cap = float(launch_cap_min + (launch_cap_max - launch_cap_min) * ratio)
                                launch_cap_active = True
                                if control_command.get('throttle', 0.0) > 0.0:
                                    control_command['throttle'] = min(control_command['throttle'], launch_cap)
                control_command['launch_throttle_cap'] = launch_cap
                control_command['launch_throttle_cap_active'] = launch_cap_active

                # Additional safety: Ensure we don't exceed speed limit
                # CRITICAL FIX: Use proportional brake instead of hard 0.8 to prevent sudden stops
                if current_speed > max_speed:  # Over limit (shouldn't happen with prevention)
                    control_command['throttle'] = 0.0
                    # Use proportional brake (already set by brake_override above, but ensure minimum)
                    # Don't override if brake_override is already set (emergency brake case)
                    if brake_override == 0.0:
                        speed_excess = current_speed - max_speed
                        control_command['brake'] = max(control_command.get('brake', 0.0), min(0.8, 0.4 + (speed_excess / 2.0)))
                
                # NEW: Safety bounds check for lateral error
                # Critical: Prevent divergence and enable recovery
                # 89.4% steering direction errors and 8.95m max error indicate need for safety bounds
                # FIXED: lateral_error is now always in control_command (from vehicle_controller.py)
                lateral_error_abs = abs(control_command.get('lateral_error', 0.0))
                
                # CRITICAL: Emergency stop when car goes out of bounds
                # When perception fails and we keep using stale data, the car can drive off the road
                # Calculate threshold accounting for car width and allowed outside distance
                # Formula: threshold = (lane_width/2 - car_width/2 + allowed_outside_lane)
                # This means: car center can be this far from lane center before emergency stop
                # For 7m lane, 1.85m car, 1.0m allowed: threshold = 3.5 - 0.925 + 1.0 = 3.575m
                lane_width = self.safety_config.get('lane_width', 7.0)
                car_width = self.safety_config.get('car_width', 1.85)
                allowed_outside_lane = self.safety_config.get('allowed_outside_lane', 1.0)
                out_of_bounds_threshold = (lane_width / 2.0) - (car_width / 2.0) + allowed_outside_lane
                
                # Also check if perception is frozen (stopped updating)
                # Distinguish between Unity pause (time gap) vs perception failure (no time gap)
                perception_frozen = hasattr(self, 'perception_frozen_frames') and self.perception_frozen_frames > 5
                perception_failed = False
                
                if perception_frozen:
                    # Check if it's a Unity pause (time gap) or perception failure (no time gap)
                    if hasattr(self, 'last_timestamp') and hasattr(self, 'last_timestamp_before_freeze'):
                        if self.last_timestamp_before_freeze is not None:
                            dt_since_freeze = timestamp - self.last_timestamp_before_freeze
                            # If time gap is small (<0.1s per frame), Unity is still running but perception died
                            # If time gap is large (>0.5s per frame), Unity paused
                            avg_dt_per_frame = dt_since_freeze / self.perception_frozen_frames if self.perception_frozen_frames > 0 else 0
                            if avg_dt_per_frame < 0.1:
                                perception_failed = True  # Perception died (Unity still sending frames)
                
                # CRITICAL: Track if emergency stop was triggered to prevent overwriting
                emergency_stop_triggered = False
                
                # Check if emergency condition exists
                is_emergency_condition = (lateral_error_abs > emergency_stop_error or 
                                        lateral_error_abs > out_of_bounds_threshold or 
                                        perception_failed)

                # Teleport/jump guard: skip emergency stop briefly after a position jump
                if (teleport_guard_active or self.post_jump_cooldown_frames > 0) and is_emergency_condition:
                    logger.warning(
                        f"[Frame {self.frame_count}] [TELEPORT GUARD] Emergency stop suppressed during "
                        f"post-jump cooldown (jump={self.last_teleport_distance:.2f}m, "
                        f"dt={self.last_teleport_dt if self.last_teleport_dt is not None else 'N/A'}s, "
                        f"cooldown_frames={self.post_jump_cooldown_frames})."
                    )
                    is_emergency_condition = False
                
                # Reset logged flag if emergency condition cleared
                if not is_emergency_condition:
                    if self.emergency_stop_logged:
                        logger.info(f"Emergency stop condition cleared. Lateral error: {lateral_error_abs:.3f}m")
                    self.emergency_stop_logged = False
                    self.emergency_stop_type = None
                
                if lateral_error_abs > emergency_stop_error:
                    # Emergency stop: Error exceeds 3.0m - stop immediately
                    if not self.emergency_stop_logged or self.emergency_stop_type != 'lateral_error_exceeded':
                        logger.error(f"[Frame {self.frame_count}] EMERGENCY STOP: Lateral error {lateral_error_abs:.3f}m exceeds {emergency_stop_error}m threshold!")
                        self.emergency_stop_logged = True
                        self.emergency_stop_type = 'lateral_error_exceeded'
                    control_command = {'steering': 0.0, 'throttle': 0.0, 'brake': 1.0, 'lateral_error': control_command.get('lateral_error', 0.0), 'emergency_stop': True}
                    emergency_stop_triggered = True
                    # Reset PID to prevent further divergence
                    self.controller.lateral_controller.reset()
                elif lateral_error_abs > out_of_bounds_threshold or perception_failed:
                    # Out of bounds or perception failed - emergency stop
                    if not self.emergency_stop_logged or self.emergency_stop_type != ('perception_failed' if perception_failed else 'out_of_bounds'):
                        if perception_failed:
                            logger.error(f"[Frame {self.frame_count}] EMERGENCY STOP: Perception FAILED! Frozen for {self.perception_frozen_frames} frames "
                                       f"(Unity still running, perception processing stopped)")
                            self.emergency_stop_type = 'perception_failed'
                        else:
                            logger.error(f"[Frame {self.frame_count}] EMERGENCY STOP: Car out of bounds! Lateral error {lateral_error_abs:.3f}m exceeds {out_of_bounds_threshold}m threshold!")
                            self.emergency_stop_type = 'out_of_bounds'
                        logger.error(f"[Frame {self.frame_count}]   Stopping vehicle to prevent further off-road driving")
                        self.emergency_stop_logged = True
                    control_command = {'steering': 0.0, 'throttle': 0.0, 'brake': 1.0, 'lateral_error': control_command.get('lateral_error', 0.0), 'emergency_stop': True}
                    emergency_stop_triggered = True
                    # Reset PID to prevent further divergence
                    self.controller.lateral_controller.reset()
                
                # CRITICAL: If emergency stop was triggered, skip all other control modifications
                if not emergency_stop_triggered:
                    if lateral_error_abs > max_lateral_error:
                        # Hard limit: Error exceeds 2.0m - aggressive correction
                        logger.warning(f"[Frame {self.frame_count}] HARD LIMIT: Lateral error {lateral_error_abs:.3f}m exceeds {max_lateral_error}m - aggressive correction")
                        # Increase steering gain temporarily for recovery
                        # CRITICAL FIX: Use config max_steering instead of hardcoded 0.5
                        max_steering = self.controller.lateral_controller.max_steering
                        steering_original = control_command.get('steering', 0.0)
                        control_command['steering'] = np.clip(steering_original * 1.5, -max_steering, max_steering)  # 50% more aggressive
                        # Reduce throttle to slow down during recovery
                        control_command['throttle'] = control_command.get('throttle', 0.0) * 0.5
                    elif lateral_error_abs > recovery_lateral_error:
                        # Recovery mode: Error exceeds 1.0m - initiate recovery
                        logger.info(f"RECOVERY MODE: Lateral error {lateral_error_abs:.3f}m exceeds {recovery_lateral_error}m - initiating recovery")
                        # Slightly increase steering for recovery
                        # CRITICAL FIX: Use config max_steering instead of hardcoded 0.5
                        max_steering = self.controller.lateral_controller.max_steering
                        steering_original = control_command.get('steering', 0.0)
                        control_command['steering'] = np.clip(steering_original * 1.2, -max_steering, max_steering)  # 20% more aggressive
                        # Reduce throttle slightly
                        control_command['throttle'] = control_command.get('throttle', 0.0) * 0.8
        
        # 4. Send control command to Unity
        # CRITICAL: Don't send control if ground truth follower is active
        # Ground truth follower will send its own control command
        if not (hasattr(self, '_gt_control_override') and self._gt_control_override is not None):
            # Only send control if NOT in ground truth mode
            self.bridge.set_control_command(
                steering=control_command['steering'],
                throttle=control_command['throttle'],
                brake=control_command['brake']
            )
        else:
            # Ground truth mode active - ground truth follower will send control
            logger.debug("Skipping AV stack control send (ground truth mode active)")
        
        # 4b. Send trajectory data for visualization (if available)
        if reference_point is not None and trajectory and trajectory.points:
            # Convert trajectory points to list format for JSON
            trajectory_points_list = []
            # Send more points for better visualization (Unity will limit if needed)
            for point in trajectory.points[:30]:  # Limit to first 30 points for performance
                trajectory_points_list.append([float(point.x), float(point.y), float(point.heading)])
            
            # Reference point
            ref_point_list = [
                float(reference_point.get('x', 0.0)),
                float(reference_point.get('y', 0.0)),
                float(reference_point.get('heading', 0.0)),
                float(reference_point.get('velocity', 8.0))
            ]
            
            # Use lateral error from controller (already calculated correctly)
            # This ensures Unity sees the same error value the controller uses
            # FIXED: Previously recalculated lateral error here, but this gave different
            # values (mean 0.1979m vs controller's 0.0218m), causing trajectory line to be yellow
            lateral_error = control_command.get('lateral_error', 0.0)
            
            # Ensure lateral_error is a valid float (handle None or invalid values)
            if lateral_error is None or not isinstance(lateral_error, (int, float)):
                lateral_error = 0.0
            lateral_error = float(lateral_error)
            
            # Debug: Log lateral error periodically to verify it's being sent correctly
            if self.frame_count % 30 == 0:  # Every ~1 second
                logger.debug(f"Trajectory visualization: lateral_error={lateral_error:.4f}m (abs={abs(lateral_error):.4f}m)")
            
            # Send to bridge for Unity visualization
            self.bridge.set_trajectory_data(
                trajectory_points=trajectory_points_list,
                reference_point=ref_point_list,
                lateral_error=lateral_error
            )
        
        # 5. Record data if enabled
        if self.recorder:
            self._record_frame(
                image, timestamp, vehicle_state_dict,
                perception_output, trajectory, control_command,
                speed_limit
            )

        # Update teleport guard countdown and vehicle position tracking
        if self.teleport_guard_frames_remaining > 0:
            self.teleport_guard_frames_remaining -= 1
        if self.post_jump_cooldown_frames > 0:
            self.post_jump_cooldown_frames -= 1
        self.last_vehicle_position = current_position
        self.last_vehicle_timestamp = timestamp

        process_duration = time.time() - process_start
        if process_duration > 0.1:
            unity_frame = vehicle_state_dict.get('unityFrameCount') or vehicle_state_dict.get('unity_frame_count')
            unity_time = vehicle_state_dict.get('unityTime') or vehicle_state_dict.get('unity_time')
            logger.warning(
                "[FRAME_PROCESS_SLOW] duration=%.3fs perception=%.3fs frame=%s unity_frame=%s unity_time=%s",
                process_duration,
                perception_duration,
                self.frame_count,
                unity_frame if unity_frame is not None else "n/a",
                f"{unity_time:.3f}" if unity_time is not None else "n/a",
            )
        
        # Log status periodically (every 30 frames = ~1 second at 30 FPS)
        if self.frame_count % 30 == 0:
            heading = self._extract_heading(vehicle_state_dict)
            num_lanes = sum(1 for c in lane_coeffs if c is not None)
            speed = vehicle_state_dict.get('speed', 0)
            
            # Get reference point for debugging (if available)
            ref_info = ""
            if reference_point is not None:
                ref_x = reference_point.get('x', 0)
                ref_y = reference_point.get('y', 0)
                ref_heading = reference_point.get('heading', 0)
                
                # RUNTIME VALIDATION: Check reference point coordinate system
                if abs(ref_x) > 100.0:
                    logger.error(
                        f"⚠️  CRITICAL: ref_x looks like PIXELS ({ref_x:.1f}), not METERS! "
                        f"Expected < 10m, got {ref_x:.1f}. Coordinate conversion may not be applied!"
                    )
                if abs(ref_heading) > np.pi / 2:  # > 90°
                    logger.warning(
                        f"⚠️  ref_heading is extreme: {np.degrees(ref_heading):.1f}° "
                        f"(expected ~0° for straight road). Coordinate system may be wrong!"
                    )
                
                # Also log lateral error for debugging
                vehicle_pos = self._extract_position(vehicle_state_dict)
                dx = ref_x - vehicle_pos[0]
                dy = ref_y - vehicle_pos[1]
                current_heading = self._extract_heading(vehicle_state_dict)
                cos_h = np.cos(-current_heading)
                sin_h = np.sin(-current_heading)
                lateral_error = dx * cos_h - dy * sin_h  # X component in vehicle frame
                
                # RUNTIME VALIDATION: Check lateral error is reasonable
                if abs(lateral_error) > 10.0:
                    logger.warning(
                        f"⚠️  Lateral error is very large: {lateral_error:.3f}m "
                        f"(expected < 5m). This may indicate coordinate system issue!"
                    )
                
                # VALIDATION: Check steering direction (warn if wrong)
                # Steering should be OPPOSITE sign of lateral_error
                # If lateral_error > 0 (car LEFT of ref), steering should be < 0 (steer LEFT)
                # If lateral_error < 0 (car RIGHT of ref), steering should be > 0 (steer RIGHT)
                if abs(lateral_error) > 0.1 and abs(control_command.get('steering', 0)) > 0.05:
                    steering = control_command.get('steering', 0)
                    # Check if steering and error have same sign (wrong direction)
                    if (lateral_error > 0 and steering > 0) or (lateral_error < 0 and steering < 0):
                        logger.warning(
                            f"⚠️  Steering direction warning: "
                            f"lateral_error={lateral_error:.3f}m, steering={steering:.3f} "
                            f"(same sign - may be wrong direction!)"
                        )
                ref_info = f", Ref=({ref_x:.2f}, {ref_y:.2f}, {np.degrees(ref_heading):.1f}°), LatErr={lateral_error:.3f}"
            
            logger.info(
                f"Frame {self.frame_count}: "
                f"Lanes={num_lanes}, "
                f"Conf={confidence:.2f}, "
                f"Speed={speed:.2f}m/s, "
                f"Heading={np.degrees(heading):.1f}°"
                f"{ref_info}, "
                f"Steering={control_command['steering']:.3f}, "
                f"Throttle={control_command['throttle']:.3f}, "
                f"Brake={control_command['brake']:.3f}"
            )
        
        # Log every frame at DEBUG level (for detailed analysis)
        logger.debug(
            f"Frame {self.frame_count}: "
            f"Lanes={sum(1 for c in lane_coeffs if c is not None)}, "
            f"Conf={confidence:.3f}, "
            f"Speed={vehicle_state_dict.get('speed', 0):.3f}, "
            f"Steering={control_command['steering']:.3f}"
        )
    
    def _extract_heading(self, vehicle_state: dict) -> float:
        """Extract heading from vehicle state (quaternion to yaw angle in radians).
        
        Unity uses left-handed coordinate system with Y-up.
        Yaw is rotation around Y-axis (vertical axis).
        """
        rotation = vehicle_state.get('rotation', {})
        if isinstance(rotation, dict):
            # Unity sends quaternion as {x, y, z, w}
            x = rotation.get('x', 0.0)
            y = rotation.get('y', 0.0)
            z = rotation.get('z', 0.0)
            w = rotation.get('w', 1.0)
            
            # Convert quaternion to Euler angles
            # For Unity: Roll (X), Pitch (Z), Yaw (Y)
            # Yaw around Y-axis: atan2(2*(w*y + x*z), 1 - 2*(y*y + z*z))
            yaw = np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z))
            
            return yaw
        return 0.0
    
    def _extract_position(self, vehicle_state: dict) -> np.ndarray:
        """Extract position from vehicle state."""
        position = vehicle_state.get('position', {})
        if isinstance(position, dict):
            return np.array([position.get('x', 0.0), position.get('z', 0.0)])
        return np.array([0.0, 0.0])

    @staticmethod
    def _apply_speed_limits(
        base_speed: float,
        speed_limit: float,
        path_curvature: float,
        max_lateral_accel: float,
        min_curve_speed: float = 0.0,
    ) -> float:
        """Clamp speed using map limits and curvature-based lateral acceleration."""
        adjusted_speed = base_speed

        if isinstance(speed_limit, (int, float)) and speed_limit > 0:
            adjusted_speed = min(adjusted_speed, float(speed_limit))

        if isinstance(path_curvature, (int, float)) and abs(path_curvature) > 1e-6:
            if isinstance(max_lateral_accel, (int, float)) and max_lateral_accel > 0:
                curve_speed = (max_lateral_accel / abs(path_curvature)) ** 0.5
                if min_curve_speed > 0:
                    curve_speed = max(curve_speed, min_curve_speed)
                adjusted_speed = min(adjusted_speed, curve_speed)

        return float(adjusted_speed)

    def _smooth_speed_limit(self, raw_limit: float, timestamp: float) -> float:
        """Smooth speed limit changes to avoid step inputs."""
        if not isinstance(raw_limit, (int, float)) or raw_limit <= 0.0:
            self.last_speed_limit = None
            self.last_speed_limit_time = None
            return 0.0

        slew_rate = self.trajectory_config.get('speed_limit_slew_rate', 0.0)
        if slew_rate <= 0.0 or self.last_speed_limit is None or self.last_speed_limit_time is None:
            self.last_speed_limit = float(raw_limit)
            self.last_speed_limit_time = float(timestamp)
            return float(raw_limit)

        dt = max(1e-3, float(timestamp) - float(self.last_speed_limit_time))
        smoothed = _slew_limit_value(float(self.last_speed_limit), float(raw_limit), float(slew_rate), dt)
        self.last_speed_limit = smoothed
        self.last_speed_limit_time = float(timestamp)
        return smoothed

    def _apply_target_speed_ramp(self, desired_speed: float, current_speed: float,
                                 timestamp: float) -> Tuple[float, bool, bool]:
        """Smooth target speed changes and ramp after stops."""
        if self.last_target_speed is None or self.last_target_speed_time is None:
            self.last_target_speed = float(desired_speed)
            self.last_target_speed_time = float(timestamp)
            return float(desired_speed), False, False

        dt = max(1e-3, float(timestamp) - float(self.last_target_speed_time))
        longitudinal_cfg = self.control_config.get('longitudinal', {})
        rate_up = float(longitudinal_cfg.get('target_speed_slew_rate_up', 0.0))
        rate_down = float(longitudinal_cfg.get('target_speed_slew_rate_down', 0.0))
        stop_threshold = float(longitudinal_cfg.get('target_speed_stop_threshold', 0.5))
        ramp_seconds = float(longitudinal_cfg.get('target_speed_restart_ramp_seconds', 0.0))

        ramped_speed, self.restart_ramp_start_time, ramp_active = _apply_restart_ramp(
            desired_speed,
            current_speed,
            self.restart_ramp_start_time,
            timestamp,
            ramp_seconds,
            stop_threshold,
        )
        adjusted = _apply_target_speed_slew(
            float(self.last_target_speed),
            float(ramped_speed),
            rate_up,
            rate_down,
            dt,
        )
        slew_active = abs(adjusted - float(ramped_speed)) > 1e-6
        self.last_target_speed = adjusted
        self.last_target_speed_time = float(timestamp)
        return adjusted, ramp_active, slew_active
    
    def _record_frame(self, image: np.ndarray, timestamp: float,
                     vehicle_state_dict: dict, perception_output: PerceptionOutput,
                     trajectory, control_command: dict, speed_limit: float = 0.0):
        """Record frame data."""
        # Create camera frame
        camera_frame = CameraFrame(
            image=image,
            timestamp=timestamp,
            frame_id=self.frame_count
        )
        
        # Create vehicle state
        position = self._extract_position(vehicle_state_dict)
        # Extract rotation quaternion properly
        rotation_dict = vehicle_state_dict.get('rotation', {})
        if isinstance(rotation_dict, dict):
            rotation = np.array([
                rotation_dict.get('x', 0.0),
                rotation_dict.get('y', 0.0),
                rotation_dict.get('z', 0.0),
                rotation_dict.get('w', 1.0)
            ])
        else:
            rotation = np.array([0, 0, 0, 1])
        
        # Extract ground truth lane positions if available
        # DEBUG: Log what we're receiving to verify Unity is sending data
        if self.frame_count % 30 == 0:  # Log every 30 frames
            gt_keys = [k for k in vehicle_state_dict.keys() if 'ground' in k.lower() or 'truth' in k.lower()]
            if gt_keys:
                logger.info(f"[GT DEBUG] Ground truth keys found: {gt_keys}")
                for key in gt_keys:
                    logger.info(f"[GT DEBUG]   {key} = {vehicle_state_dict.get(key, 'NOT FOUND')}")
            else:
                logger.warning(f"[GT DEBUG] No ground truth keys found in vehicle_state_dict. "
                             f"Available keys: {list(vehicle_state_dict.keys())[:10]}")
        
        # Try both camelCase and snake_case field names
        # FIXED: Check for ground truth data (try both camelCase and snake_case)
        gt_left_lane_line_x = vehicle_state_dict.get('groundTruthLeftLaneLineX') or vehicle_state_dict.get('ground_truth_left_lane_line_x') or vehicle_state_dict.get('groundTruthLeftLaneX') or vehicle_state_dict.get('ground_truth_left_lane_x', 0.0)  # Backward compatibility
        gt_right_lane_line_x = vehicle_state_dict.get('groundTruthRightLaneLineX') or vehicle_state_dict.get('ground_truth_right_lane_line_x') or vehicle_state_dict.get('groundTruthRightLaneX') or vehicle_state_dict.get('ground_truth_right_lane_x', 0.0)  # Backward compatibility
        gt_lane_center_x = vehicle_state_dict.get('groundTruthLaneCenterX') or vehicle_state_dict.get('ground_truth_lane_center_x', 0.0)
        gt_path_curvature = vehicle_state_dict.get('groundTruthPathCurvature') or vehicle_state_dict.get('ground_truth_path_curvature', 0.0)
        gt_desired_heading = vehicle_state_dict.get('groundTruthDesiredHeading') or vehicle_state_dict.get('ground_truth_desired_heading', 0.0)
        
        # Log if we got ground truth data (or if it's missing)
        if self.frame_count % 30 == 0:  # Log every 30 frames
            if gt_left_lane_line_x != 0.0 or gt_right_lane_line_x != 0.0:
                logger.info(f"[GT DEBUG] Ground truth received: left={gt_left_lane_line_x:.3f}m, "
                           f"right={gt_right_lane_line_x:.3f}m, center={gt_lane_center_x:.3f}m")
            else:
                logger.warning(f"[GT DEBUG] Ground truth is 0.0 - check Unity: "
                             f"Is GroundTruthReporter enabled? Is it attached to the car? "
                             f"Vehicle state keys: {list(vehicle_state_dict.keys())}")
        
        # NEW: Extract camera calibration data (try both camelCase and snake_case)
        # CRITICAL: Use 'is not None' check instead of 'or' to handle 0.0 values correctly
        # 'or' treats 0.0 as falsy, which would use the default -1.0 even if Unity sent 0.0
        camera_8m_screen_y = vehicle_state_dict.get('camera8mScreenY')
        if camera_8m_screen_y is None:
            camera_8m_screen_y = vehicle_state_dict.get('camera_8m_screen_y', -1.0)
        camera_lookahead_screen_y = vehicle_state_dict.get('cameraLookaheadScreenY')
        if camera_lookahead_screen_y is None:
            camera_lookahead_screen_y = vehicle_state_dict.get('camera_lookahead_screen_y', -1.0)
        ground_truth_lookahead_distance = vehicle_state_dict.get('groundTruthLookaheadDistance')
        if ground_truth_lookahead_distance is None:
            ground_truth_lookahead_distance = vehicle_state_dict.get('ground_truth_lookahead_distance', 8.0)
        
        # NEW: Extract camera FOV data (what Unity actually uses)
        camera_field_of_view = vehicle_state_dict.get('cameraFieldOfView')
        if camera_field_of_view is None:
            camera_field_of_view = vehicle_state_dict.get('camera_field_of_view', 0.0)
        
        camera_horizontal_fov = vehicle_state_dict.get('cameraHorizontalFOV')
        if camera_horizontal_fov is None:
            camera_horizontal_fov = vehicle_state_dict.get('camera_horizontal_fov', 0.0)
        
        # Debug: Log camera calibration data periodically
        if self.frame_count % 60 == 0:  # Every 60 frames
            if camera_8m_screen_y > 0:
                logger.info(f"[CAMERA CALIB] camera_8m_screen_y = {camera_8m_screen_y:.1f}px (valid, from Unity)")
            else:
                logger.warning(f"[CAMERA CALIB] camera_8m_screen_y = {camera_8m_screen_y:.1f} (not calculated - camera not available or old recording)")
            
            # Log FOV information (critical for understanding Unity's behavior)
            if camera_field_of_view > 0:
                logger.info(f"[CAMERA FOV] Unity fieldOfView = {camera_field_of_view:.2f}° (vertical), "
                           f"calculated horizontal = {camera_horizontal_fov:.2f}°")
            
            # NEW: Log camera position and forward for debugging alignment
            camera_pos_x = vehicle_state_dict.get('cameraPosX', 0.0)
            camera_pos_y = vehicle_state_dict.get('cameraPosY', 0.0)
            camera_pos_z = vehicle_state_dict.get('cameraPosZ', 0.0)
            camera_forward_x = vehicle_state_dict.get('cameraForwardX', 0.0)
            camera_forward_y = vehicle_state_dict.get('cameraForwardY', 0.0)
            camera_forward_z = vehicle_state_dict.get('cameraForwardZ', 0.0)
            if camera_pos_x != 0.0 or camera_pos_y != 0.0 or camera_pos_z != 0.0:
                logger.info(f"[CAMERA POS] position=({camera_pos_x:.3f}, {camera_pos_y:.3f}, {camera_pos_z:.3f}), "
                           f"forward=({camera_forward_x:.3f}, {camera_forward_y:.3f}, {camera_forward_z:.3f})")
                logger.info(f"[CAMERA FOV] Config camera_fov = 110.0° - "
                           f"{'matches horizontal' if abs(camera_horizontal_fov - 110.0) < 1.0 else 'does NOT match horizontal'}")
        
        # CRITICAL: Log every frame to debug why recording shows 0.0
        if self.frame_count < 5:  # Log first 5 frames to debug
            logger.info(f"[CAMERA CALIB DEBUG] Frame {self.frame_count}: camera_8m_screen_y = {camera_8m_screen_y:.1f}, "
                       f"raw dict value = {vehicle_state_dict.get('camera8mScreenY')}, "
                       f"snake_case value = {vehicle_state_dict.get('camera_8m_screen_y')}")
        
        vehicle_state = VehicleState(
            timestamp=timestamp,
            position=np.array([position[0], vehicle_state_dict.get('position', {}).get('y', 0.0), position[1]]),
            rotation=rotation,  # Proper quaternion extraction
            velocity=np.array([0, 0, vehicle_state_dict.get('speed', 0.0)]),
            angular_velocity=np.array([0, 0, 0]),
            speed=vehicle_state_dict.get('speed', 0.0),
            steering_angle=vehicle_state_dict.get('steeringAngle', 0.0),
            motor_torque=vehicle_state_dict.get('motorTorque', 0.0),
            brake_torque=vehicle_state_dict.get('brakeTorque', 0.0),
            unity_time=vehicle_state_dict.get('unityTime', vehicle_state_dict.get('unity_time', 0.0)),
            unity_frame_count=vehicle_state_dict.get('unityFrameCount', vehicle_state_dict.get('unity_frame_count', 0)),
            unity_delta_time=vehicle_state_dict.get('unityDeltaTime', vehicle_state_dict.get('unity_delta_time', 0.0)),
            unity_smooth_delta_time=vehicle_state_dict.get('unitySmoothDeltaTime', vehicle_state_dict.get('unity_smooth_delta_time', 0.0)),
            unity_unscaled_delta_time=vehicle_state_dict.get('unityUnscaledDeltaTime', vehicle_state_dict.get('unity_unscaled_delta_time', 0.0)),
            unity_time_scale=vehicle_state_dict.get('unityTimeScale', vehicle_state_dict.get('unity_time_scale', 1.0)),
            ground_truth_left_lane_line_x=gt_left_lane_line_x,
            ground_truth_right_lane_line_x=gt_right_lane_line_x,
            ground_truth_lane_center_x=gt_lane_center_x,
            ground_truth_path_curvature=gt_path_curvature,
            ground_truth_desired_heading=gt_desired_heading,
            camera_8m_screen_y=camera_8m_screen_y,  # NEW: Camera calibration data
            camera_lookahead_screen_y=camera_lookahead_screen_y,
            ground_truth_lookahead_distance=ground_truth_lookahead_distance,
            camera_field_of_view=camera_field_of_view,  # NEW: Camera FOV data
            camera_horizontal_fov=camera_horizontal_fov,  # NEW: Camera horizontal FOV
            camera_pos_x=vehicle_state_dict.get('cameraPosX', 0.0),  # NEW: Camera position
            camera_pos_y=vehicle_state_dict.get('cameraPosY', 0.0),
            camera_pos_z=vehicle_state_dict.get('cameraPosZ', 0.0),
            camera_forward_x=vehicle_state_dict.get('cameraForwardX', 0.0),  # NEW: Camera forward
            camera_forward_y=vehicle_state_dict.get('cameraForwardY', 0.0),
            camera_forward_z=vehicle_state_dict.get('cameraForwardZ', 0.0),
            # NEW: Debug fields for diagnosing ground truth offset issues
            road_center_at_car_x=vehicle_state_dict.get('roadCenterAtCarX', 0.0),
            road_center_at_car_y=vehicle_state_dict.get('roadCenterAtCarY', 0.0),
            road_center_at_car_z=vehicle_state_dict.get('roadCenterAtCarZ', 0.0),
            road_center_at_lookahead_x=vehicle_state_dict.get('roadCenterAtLookaheadX', 0.0),
            road_center_at_lookahead_y=vehicle_state_dict.get('roadCenterAtLookaheadY', 0.0),
            road_center_at_lookahead_z=vehicle_state_dict.get('roadCenterAtLookaheadZ', 0.0),
            road_center_reference_t=vehicle_state_dict.get('roadCenterReferenceT', 0.0),
            speed_limit=speed_limit,
            speed_limit_preview=vehicle_state_dict.get(
                'speedLimitPreview', vehicle_state_dict.get('speed_limit_preview', 0.0)
            ),
            speed_limit_preview_distance=vehicle_state_dict.get(
                'speedLimitPreviewDistance',
                vehicle_state_dict.get('speed_limit_preview_distance', 0.0),
            ),
        )
        
        # Create control command with metadata
        control_cmd = ControlCommand(
            timestamp=timestamp,
            steering=control_command['steering'],
            throttle=control_command['throttle'],
            brake=control_command['brake'],
            steering_before_limits=control_command.get('steering_before_limits'),
            throttle_before_limits=control_command.get('throttle_before_limits'),
            brake_before_limits=control_command.get('brake_before_limits'),
            pid_integral=control_command.get('pid_integral'),
            pid_derivative=control_command.get('pid_derivative'),
            pid_error=control_command.get('total_error'),  # Total error before PID
            lateral_error=control_command.get('lateral_error'),
            heading_error=control_command.get('heading_error'),
            total_error=control_command.get('total_error'),
            path_curvature_input=control_command.get('path_curvature_input'),
            is_straight=control_command.get('is_straight'),
            straight_oscillation_rate=control_command.get('straight_oscillation_rate'),
            tuned_deadband=control_command.get('tuned_deadband'),
            tuned_error_smoothing_alpha=control_command.get('tuned_error_smoothing_alpha'),
            # Diagnostic fields for tracking stale perception usage
            using_stale_perception=perception_output.using_stale_data if perception_output else False,
            stale_perception_reason=perception_output.stale_data_reason if perception_output else None,
            emergency_stop=bool(control_command.get('emergency_stop', False)),
            target_speed_raw=control_command.get('target_speed_raw'),
            target_speed_post_limits=control_command.get('target_speed_post_limits'),
            target_speed_planned=control_command.get('target_speed_planned'),
            target_speed_final=control_command.get('target_speed_final'),
            target_speed_slew_active=bool(control_command.get('target_speed_slew_active', False)),
            target_speed_ramp_active=bool(control_command.get('target_speed_ramp_active', False)),
            launch_throttle_cap=control_command.get('launch_throttle_cap'),
            launch_throttle_cap_active=bool(control_command.get('launch_throttle_cap_active', False))
        )
        
        # Create trajectory output
        # Include reference point as first point for analysis
        trajectory_points = None
        velocities = None
        ref_point = None  # Initialize ref_point
        # FIXED: Check if trajectory exists AND has points (not just truthy check)
        if trajectory is not None and hasattr(trajectory, 'points') and len(trajectory.points) > 0:
            confidence = perception_output.confidence if perception_output else None
            # Get reference point for recording
            # Extract lane_coeffs from perception_output for direct computation
            lane_coeffs_for_ref = None
            if perception_output and perception_output.lane_line_coefficients is not None:
                # Convert numpy array back to list of arrays
                # Handle both 2D array (shape (N, 3)) and array of arrays
                coeffs_array = perception_output.lane_line_coefficients
                if coeffs_array.ndim == 2:
                    # 2D array: convert each row to array
                    lane_coeffs_for_ref = [np.array(row) for row in coeffs_array]
                else:
                    # Array of arrays: convert to list
                    lane_coeffs_for_ref = [coeffs for coeffs in coeffs_array]
            
            reference_lookahead = self.trajectory_config.get('reference_lookahead', 8.0)
            # Get lane positions from perception_output for most accurate reference point
            lane_positions_for_ref = None
            if perception_output:
                lane_positions_for_ref = {
                    'left_lane_line_x': perception_output.left_lane_line_x,
                    'right_lane_line_x': perception_output.right_lane_line_x
                }
            
            ref_point = self.trajectory_planner.get_reference_point(
                trajectory, 
                lookahead=reference_lookahead,
                lane_coeffs=lane_coeffs_for_ref,  # Pass lane coefficients for direct computation
                lane_positions=lane_positions_for_ref,  # Preferred: use vehicle coords
                use_direct=True,  # Use direct midpoint computation
                timestamp=timestamp,
                confidence=confidence,
            )
            
            # Build trajectory points array with reference point first
            points_list = []
            vel_list = []
            
            # Add reference point first (for drift analysis)
            if ref_point:
                points_list.append([ref_point['x'], ref_point['y'], ref_point['heading']])
                vel_list.append(ref_point.get('velocity', 8.0))
            
            # Add other trajectory points
            for p in trajectory.points:
                points_list.append([p.x, p.y, p.heading])
                vel_list.append(p.velocity)
            
            trajectory_points = np.array(points_list) if points_list else None
            velocities = np.array(vel_list) if vel_list else None
        
        # Extract debug info from reference point
        ref_point_method = None
        perception_center_x = None
        if ref_point:
            ref_point_method = ref_point.get('method')  # 'lane_positions', 'lane_coeffs', or 'trajectory'
            perception_center_x = ref_point.get('perception_center_x')  # Perception center before trajectory calc
        
        trajectory_output = TrajectoryOutput(
            timestamp=timestamp,
            trajectory_points=trajectory_points,
            velocities=velocities,
            reference_point=ref_point,  # Store reference point for analysis
            reference_point_method=ref_point_method,  # NEW: Track which method was used
            perception_center_x=perception_center_x  # NEW: Store perception center for comparison
        )
        
        # Allow ground truth follower to override control command for recording
        if hasattr(self, '_gt_control_override') and self._gt_control_override is not None:
            control_cmd = self._gt_control_override
        
        # Get Unity feedback (if available)
        unity_feedback = None
        if self.bridge:
            try:
                feedback_dict = self.bridge.get_latest_unity_feedback()
                if feedback_dict:
                    from data.formats.data_format import UnityFeedback
                    unity_feedback = UnityFeedback(
                        timestamp=feedback_dict.get('timestamp', timestamp),
                        ground_truth_mode_active=feedback_dict.get('ground_truth_mode_active', False),
                        control_command_received=feedback_dict.get('control_command_received', False),
                        actual_steering_applied=feedback_dict.get('actual_steering_applied'),
                        actual_throttle_applied=feedback_dict.get('actual_throttle_applied'),
                        actual_brake_applied=feedback_dict.get('actual_brake_applied'),
                        ground_truth_data_available=feedback_dict.get('ground_truth_data_available', False),
                        ground_truth_reporter_enabled=feedback_dict.get('ground_truth_reporter_enabled', False),
                        path_curvature_calculated=feedback_dict.get('path_curvature_calculated', False),
                        car_controller_mode=feedback_dict.get('car_controller_mode'),
                        av_control_enabled=feedback_dict.get('av_control_enabled', False),
                        unity_frame_count=feedback_dict.get('unity_frame_count'),
                        unity_time=feedback_dict.get('unity_time')
                    )
            except Exception:
                # Unity feedback is optional - don't fail if it's not available
                pass
        
        # Create recording frame
        frame = RecordingFrame(
            timestamp=timestamp,
            frame_id=self.frame_count,
            camera_frame=camera_frame,
            vehicle_state=vehicle_state,
            control_command=control_cmd,
            perception_output=perception_output,
            trajectory_output=trajectory_output,
            unity_feedback=unity_feedback
        )
        
        self.recorder.record_frame(frame)
    
    def stop(self):
        """Stop AV stack."""
        self.running = False
        
        # CRITICAL: Signal Unity to exit play mode gracefully before stopping
        # This ensures Unity exits play mode when AV stack stops (duration expiry, frame limit, etc.)
        try:
            logger.info("Signaling Unity to exit play mode...")
            self.bridge.signal_shutdown()
        except Exception as e:
            logger.warning(f"Error signaling Unity shutdown: {e}")
        
        if self.recorder:
            logger.info(f"Closing data recorder: {self.recorder.output_file}")
            self.recorder.close()
            logger.info(f"Recorded {self.frame_count} frames")
        
        logger.info(f"AV Stack stopped (processed {self.frame_count} frames)")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run AV Stack')
    parser.add_argument('--bridge_url', type=str, default='http://localhost:8000',
                       help='Unity bridge server URL')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained perception model')
    parser.add_argument('--record', action='store_true', default=True,
                       help='Record data during operation (default: True)')
    parser.add_argument('--no-record', dest='record', action='store_false',
                       help='Disable data recording')
    parser.add_argument('--recording_dir', type=str, default='data/recordings',
                       help='Directory for recordings')
    parser.add_argument('--use-cv', action='store_true',
                       help='Force CV-based perception (override default segmentation)')
    parser.add_argument(
        '--segmentation-checkpoint',
        type=str,
        default='data/segmentation_dataset/checkpoints/segnet_best.pt',
        help='Segmentation checkpoint path (default: segnet_best.pt)',
    )
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum number of frames to process')
    parser.add_argument('--duration', type=float, default=None,
                       help='Maximum duration in seconds (similar to ground_truth_follower.py)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration YAML file (default: config/av_stack_config.yaml)')
    
    args = parser.parse_args()
    
    use_segmentation = not args.use_cv
    if use_segmentation and not Path(args.segmentation_checkpoint).exists():
        parser.error(
            f"Segmentation checkpoint not found: {args.segmentation_checkpoint}. "
            "Provide --segmentation-checkpoint or run with --use-cv."
        )
    
    # Create and run AV stack
    av_stack = AVStack(
        bridge_url=args.bridge_url,
        model_path=args.model_path,
        record_data=args.record,
        config_path=args.config,
        recording_dir=args.recording_dir,
        use_segmentation=use_segmentation,
        segmentation_model_path=args.segmentation_checkpoint,
    )
    
    av_stack.run(max_frames=args.max_frames, duration=args.duration)


if __name__ == "__main__":
    main()

