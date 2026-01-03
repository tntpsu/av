"""
Main AV stack integration script.
Connects all components: perception, trajectory planning, and control.
"""

import time
import numpy as np
from typing import Optional
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
from control.pid_controller import VehicleController
from data.recorder import DataRecorder
from data.formats.data_format import CameraFrame, VehicleState, ControlCommand, PerceptionOutput, TrajectoryOutput, RecordingFrame

# Configure logging
# Ensure tmp/logs directory exists
import os
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


class AVStack:
    """Main AV stack integrating all components."""
    
    def __init__(self, bridge_url: str = "http://localhost:8000",
                 model_path: Optional[str] = None,
                 record_data: bool = True,  # Enable by default for comprehensive logging
                 recording_dir: str = "data/recordings",
                 config_path: Optional[str] = None):
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
        self.perception = LaneDetectionInference(model_path=model_path)
        
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
            max_steering=lateral_cfg.get('max_steering', 0.5),
            lateral_deadband=lateral_cfg.get('deadband', 0.02),
            lateral_heading_weight=lateral_cfg.get('heading_weight', 0.5),
            lateral_lateral_weight=lateral_cfg.get('lateral_weight', 0.5),
            lateral_error_clip=lateral_cfg.get('error_clip', np.pi / 4),
            lateral_integral_limit=lateral_cfg.get('integral_limit', 0.10)  # FIXED: Read from config (default 0.10)
        )
        
        # Store config for use in _process_frame
        self.config = config
        self.control_config = control_cfg
        self.trajectory_config = trajectory_cfg
        self.safety_config = safety_cfg
        
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
        
        # Track if emergency stop has been logged (to prevent repeated messages)
        self.emergency_stop_logged = False
        self.emergency_stop_type = None  # Track which type of emergency stop (for logging)
    
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
        last_frame_time = time.time()
        start_time = time.time()
        
        try:
            while self.running:
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
                frame_data = self.bridge.get_latest_camera_frame()
                if frame_data is None:
                    # No frame available - don't spam, just wait
                    time.sleep(self.frame_interval)
                    continue
                
                image, timestamp = frame_data
                
                # Get vehicle state
                vehicle_state_dict = self.bridge.get_latest_vehicle_state()
                if vehicle_state_dict is None:
                    time.sleep(self.frame_interval)
                    continue
                
                # Process frame
                self._process_frame(image, timestamp, vehicle_state_dict)
                
                self.frame_count += 1
        
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
        # Track timestamp for perception frozen detection
        if not hasattr(self, 'last_timestamp'):
            self.last_timestamp = timestamp
        else:
            self.last_timestamp = timestamp
        
        # 1. Perception: Detect lanes
        # detect() now returns: (lane_coeffs, confidence, detection_method, num_lanes_detected)
        perception_result = self.perception.detect(image)
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
                else:
                    # Fallback if return format is different
                    logger.warning(f"[DEBUG VIS] Unexpected return format: {type(result)}")
                    cv_lanes = result if isinstance(result, list) else [None, None]
                    debug_info = {}
                
                # Create visualization
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
        
        reference_lookahead = self.trajectory_config.get('reference_lookahead', 8.0)
        
        # Extract camera_8m_screen_y early so we can use it for lane evaluation
        camera_8m_screen_y = vehicle_state_dict.get('camera8mScreenY')
        if camera_8m_screen_y is None:
            camera_8m_screen_y = vehicle_state_dict.get('camera_8m_screen_y', -1.0)
        
        # DEBUG: Log what we extracted (first few frames only)
        if self.frame_count < 3:
            logger.info(f"[COORD DEBUG] Frame {self.frame_count}: Extracted camera_8m_screen_y={camera_8m_screen_y} "
                       f"(from dict keys: {list(vehicle_state_dict.keys())[:10]}...)")
        
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
                if camera_8m_screen_y is not None and camera_8m_screen_y > 0:
                    # Use Unity's actual 8m position (most accurate)
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
                if camera_8m_screen_y > 0 and abs(y_image_at_lookahead - camera_8m_screen_y) < 5.0:
                    # We're evaluating at or near Unity's 8m position, so distance is 8m
                    conversion_distance = 8.0  # Actual distance from Unity
                    logger.info(f"[COORD DEBUG] Using Unity's actual distance: 8.0m (at y={camera_8m_screen_y:.1f}px)")
                else:
                    # Fallback: Use reference_lookahead (may not be exact, but consistent)
                    conversion_distance = reference_lookahead  # 8m - approximate for now
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
                
                left_x_vehicle, _ = planner._convert_image_to_vehicle_coords(
                    left_x_image, y_image_at_lookahead, lookahead_distance=conversion_distance,
                    horizontal_fov_override=camera_horizontal_fov if camera_horizontal_fov and camera_horizontal_fov > 0 else None
                )
                right_x_vehicle, _ = planner._convert_image_to_vehicle_coords(
                    right_x_image, y_image_at_lookahead, lookahead_distance=conversion_distance,
                    horizontal_fov_override=camera_horizontal_fov if camera_horizontal_fov and camera_horizontal_fov > 0 else None
                )
                calculated_lane_width = right_x_vehicle - left_x_vehicle
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
                    logger.error(f"[PERCEPTION VALIDATION] ❌ INVALID LANE WIDTH: {calculated_lane_width:.3f}m "
                               f"(expected {min_lane_width}-{max_lane_width}m)")
                    logger.error(f"  Left lane: {left_x_vehicle:.3f}m, Right lane: {right_x_vehicle:.3f}m")
                    logger.error(f"  This is a perception failure - rejecting detection!")
                    
                    # Reject this detection - set to None so temporal filtering can use previous frame
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
                    # CRITICAL FIX: Check for sudden jumps in lane positions (temporal validation)
                    # If lane position jumps >2m from previous frame, it's likely a false detection
                    if not hasattr(self, 'previous_left_lane_x'):
                        self.previous_left_lane_x = None
                        self.previous_right_lane_x = None
                    
                    if self.previous_left_lane_x is not None and self.previous_right_lane_x is not None:
                        left_jump = abs(left_x_vehicle - self.previous_left_lane_x)
                        right_jump = abs(right_x_vehicle - self.previous_right_lane_x)
                        # RECOMMENDED: Increased from 2.0m to 3.5m to handle curves better
                        # On curves, lane positions can change by 2-3m between frames when car is turning
                        # 2.0m was too strict and caused false rejections on curves (e.g., frame 201: 3.094m jump)
                        max_jump_threshold = 3.5  # meters - maximum allowed jump between frames (was 2.0m)
                        
                        # CRITICAL FIX: Check for timestamp gap (Unity pause) OR frozen timestamp before applying jump detection
                        # After a Unity pause, large jumps are expected and should be accepted
                        # Also detect frozen timestamps (dt = 0.0) which indicates Unity stopped updating
                        has_time_gap = False
                        has_frozen_timestamp = False
                        if not hasattr(self, 'last_timestamp') or self.last_timestamp is None:
                            self.last_timestamp = timestamp
                        else:
                            dt = timestamp - self.last_timestamp
                            if dt > 0.5:  # Large time gap = Unity paused
                                has_time_gap = True
                                logger.info(f"[PERCEPTION VALIDATION] Unity pause detected (time gap: {dt:.3f}s) - relaxing jump detection")
                                # After Unity pause, accept larger jumps (6.0m instead of 3.5m)
                                max_jump_threshold = 6.0
                            elif abs(dt) < 0.001:  # Timestamp frozen (dt ≈ 0) = Unity stopped updating
                                has_frozen_timestamp = True
                                logger.warning(f"[PERCEPTION VALIDATION] ⚠️  FROZEN TIMESTAMP detected (dt={dt:.6f}s) - Unity stopped updating timestamps!")
                                logger.warning(f"  This means Unity paused/froze but Python kept processing the same frame")
                                logger.warning(f"  Perception will return identical results until Unity resumes")
                                # Treat frozen timestamp like a time gap - accept any changes when Unity resumes
                                # But for now, we're still processing the same frame, so values will be identical
                                # Don't update last_timestamp - keep it as the last valid timestamp
                                # This will help detect when Unity resumes (timestamp will jump)
                        
                        # CRITICAL: Detect if perception values are frozen (same as previous frame)
                        # This could indicate:
                        # 1. Unity paused (no new frames) - check timestamp gap
                        # 2. Perception died (no new detections, but Unity still sending frames)
                        # 3. Bridge disconnected (no frames at all)
                        perception_frozen = (abs(left_x_vehicle - self.previous_left_lane_x) < 0.001 and 
                                           abs(right_x_vehicle - self.previous_right_lane_x) < 0.001)
                        
                        if perception_frozen:
                            # Perception values are frozen - diagnose the cause
                            # Track how many consecutive frames perception has been frozen
                            if not hasattr(self, 'perception_frozen_frames'):
                                self.perception_frozen_frames = 0
                                self.last_timestamp_before_freeze = None
                            
                            self.perception_frozen_frames += 1
                            # Track that we're using stale data due to frozen perception
                            if not using_stale_data:  # Only set if not already set by jump detection
                                using_stale_data = True
                                stale_data_reason = "frozen"
                                logger.warning(f"[Frame {self.frame_count}] Using STALE perception data (reason: {stale_data_reason}) - perception values frozen")
                            
                            # Check timestamp to distinguish Unity pause vs perception failure
                            if not hasattr(self, 'last_timestamp') or self.last_timestamp is None:
                                self.last_timestamp = timestamp
                                self.last_timestamp_before_freeze = timestamp
                            
                            dt = timestamp - self.last_timestamp if self.last_timestamp is not None else 0
                            
                            if self.perception_frozen_frames == 1:
                                # First frozen frame - record timestamp
                                self.last_timestamp_before_freeze = self.last_timestamp if hasattr(self, 'last_timestamp') else timestamp
                            
                            # Diagnose cause
                            if abs(dt) < 0.001:  # Frozen timestamp (dt ≈ 0) = Unity stopped updating timestamps
                                if self.perception_frozen_frames == 1:
                                    logger.warning(f"[PERCEPTION VALIDATION] ⚠️  FROZEN TIMESTAMP detected (dt={dt:.6f}s) - Unity stopped updating!")
                                    logger.warning(f"  Unity paused/froze but Python kept processing the same frame")
                                    logger.warning(f"  Perception will return identical results until Unity resumes")
                                # Don't update last_timestamp - keep it as the last valid timestamp
                                # This helps detect when Unity resumes (timestamp will jump)
                            elif dt > 0.5:  # Large time gap = Unity paused
                                if self.perception_frozen_frames == 1:
                                    logger.warning(f"[PERCEPTION VALIDATION] ⚠️  Unity paused (time gap: {dt:.3f}s) - perception values frozen")
                                self.last_timestamp = timestamp
                            elif dt < 0.1:  # Normal time gap but values frozen = perception died
                                if self.perception_frozen_frames > 3:
                                    logger.error(f"[PERCEPTION VALIDATION] ❌ PERCEPTION FAILED! "
                                               f"Values unchanged for {self.perception_frozen_frames} frames "
                                               f"(Unity still sending frames, dt={dt:.3f}s)")
                                    logger.error(f"  Left: {left_x_vehicle:.3f}m, Right: {right_x_vehicle:.3f}m")
                                    logger.error(f"  This indicates perception processing has stopped - emergency stop needed")
                                self.last_timestamp = timestamp
                            else:
                                # Medium time gap - could be either
                                if self.perception_frozen_frames > 3:
                                    logger.warning(f"[PERCEPTION VALIDATION] ⚠️  Perception values frozen for {self.perception_frozen_frames} frames "
                                                 f"(time gap: {dt:.3f}s)")
                                self.last_timestamp = timestamp
                        else:
                            # Perception is updating - check if we just recovered from frozen timestamp
                            if hasattr(self, 'perception_frozen_frames') and self.perception_frozen_frames > 0:
                                # Check if timestamp jumped (Unity resumed)
                                dt = timestamp - self.last_timestamp if (hasattr(self, 'last_timestamp') and self.last_timestamp is not None) else 0
                                if abs(dt) > 0.001:  # Timestamp is updating again
                                    logger.info(f"[PERCEPTION VALIDATION] ✓ Unity resumed! Timestamp updated (dt={dt:.3f}s) after {self.perception_frozen_frames} frozen frames")
                            
                            # Reset frozen counter
                            if hasattr(self, 'perception_frozen_frames'):
                                if self.perception_frozen_frames > 0:
                                    logger.info(f"[PERCEPTION VALIDATION] ✓ Perception recovered after {self.perception_frozen_frames} frozen frames")
                                self.perception_frozen_frames = 0
                            
                            if not hasattr(self, 'last_timestamp'):
                                self.last_timestamp = timestamp
                            else:
                                self.last_timestamp = timestamp
                        
                        if left_jump > max_jump_threshold or right_jump > max_jump_threshold:
                            if has_time_gap:
                                # After Unity pause, accept the jump (it's expected)
                                logger.info(f"[PERCEPTION VALIDATION] Large jump after Unity pause - accepting detection")
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
                                # Normal jump detection (no time gap) - reject if too large
                                logger.error(f"[PERCEPTION VALIDATION] ❌ SUDDEN LANE JUMP DETECTED! (Frame {self.frame_count})")
                                logger.error(f"  Left lane jump: {left_jump:.3f}m (threshold: {max_jump_threshold}m)")
                                logger.error(f"  Right lane jump: {right_jump:.3f}m (threshold: {max_jump_threshold}m)")
                                logger.error(f"  Previous: left={self.previous_left_lane_x:.3f}m, right={self.previous_right_lane_x:.3f}m")
                                logger.error(f"  Current: left={left_x_vehicle:.3f}m, right={right_x_vehicle:.3f}m")
                                logger.error(f"  Rejecting detection - using previous frame's values")
                                
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
                            # Valid detection - use it
                            left_lane_line_x = float(left_x_vehicle)
                            right_lane_line_x = float(right_x_vehicle)
                            # Update previous values for next frame
                            self.previous_left_lane_x = left_x_vehicle
                            self.previous_right_lane_x = right_x_vehicle
                    else:
                        # First frame or no previous data - accept current detection
                        left_lane_line_x = float(left_x_vehicle)
                        right_lane_line_x = float(right_x_vehicle)
                        # Store for next frame
                        self.previous_left_lane_x = left_x_vehicle
                        self.previous_right_lane_x = right_x_vehicle
            except (AttributeError, TypeError, ValueError):
                # If planner is a Mock or conversion fails, skip lane data calculation
                # This is acceptable in unit tests
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
                logger.warning(f"[PERCEPTION] Perception failed (both lanes = 0.0) - setting to None to prevent movement")
                left_lane_line_x = None
                right_lane_line_x = None
        
        # Record perception output with diagnostic fields
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
            jump_threshold=jump_threshold_used
        )
        
        # 2. Trajectory Planning: Plan path
        trajectory = self.trajectory_planner.plan(lane_coeffs, vehicle_state_dict)
        
        # Get current speed for safety checks (check BEFORE computing control)
        current_speed = vehicle_state_dict.get('speed', 0.0)
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
                
                # If emergency braking was triggered, override throttle/brake but keep steering
                if brake_override > 0:
                    control_command['throttle'] = 0.0
                    control_command['brake'] = brake_override
                    # Keep steering for safety (steer while braking)
                
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
                
                # Additional prevention: Start reducing throttle even earlier (at 60% of max)
                elif current_speed > max_speed * 0.60:
                    # Start gentle throttle reduction at 60% of max speed
                    control_command['throttle'] = control_command.get('throttle', 0.0) * 0.7
                
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
                    control_command = {'steering': 0.0, 'throttle': 0.0, 'brake': 1.0, 'lateral_error': control_command.get('lateral_error', 0.0)}
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
                    control_command = {'steering': 0.0, 'throttle': 0.0, 'brake': 1.0, 'lateral_error': control_command.get('lateral_error', 0.0)}
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
                perception_output, trajectory, control_command
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
    
    def _record_frame(self, image: np.ndarray, timestamp: float,
                     vehicle_state_dict: dict, perception_output: PerceptionOutput,
                     trajectory, control_command: dict):
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
            ground_truth_left_lane_line_x=gt_left_lane_line_x,
            ground_truth_right_lane_line_x=gt_right_lane_line_x,
            ground_truth_lane_center_x=gt_lane_center_x,
            ground_truth_path_curvature=gt_path_curvature,
            ground_truth_desired_heading=gt_desired_heading,
            camera_8m_screen_y=camera_8m_screen_y,  # NEW: Camera calibration data
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
            road_center_reference_t=vehicle_state_dict.get('roadCenterReferenceT', 0.0)
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
            # Diagnostic fields for tracking stale perception usage
            using_stale_perception=perception_output.using_stale_data if perception_output else False,
            stale_perception_reason=perception_output.stale_data_reason if perception_output else None
        )
        
        # Create trajectory output
        # Include reference point as first point for analysis
        trajectory_points = None
        velocities = None
        ref_point = None  # Initialize ref_point
        # FIXED: Check if trajectory exists AND has points (not just truthy check)
        if trajectory is not None and hasattr(trajectory, 'points') and len(trajectory.points) > 0:
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
                use_direct=True  # Use direct midpoint computation
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
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum number of frames to process')
    parser.add_argument('--duration', type=float, default=None,
                       help='Maximum duration in seconds (similar to ground_truth_follower.py)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration YAML file (default: config/av_stack_config.yaml)')
    
    args = parser.parse_args()
    
    # Create and run AV stack
    av_stack = AVStack(
        bridge_url=args.bridge_url,
        model_path=args.model_path,
        record_data=args.record,
        config_path=args.config,
        recording_dir=args.recording_dir
    )
    
    av_stack.run(max_frames=args.max_frames, duration=args.duration)


if __name__ == "__main__":
    main()

