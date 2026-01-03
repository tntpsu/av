"""
Data format definitions for AV stack recordings.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class CameraFrame:
    """Camera frame data."""
    image: np.ndarray  # RGB image array
    timestamp: float
    frame_id: int
    camera_id: str = "front_center"


@dataclass
class VehicleState:
    """Vehicle state data."""
    timestamp: float
    position: np.ndarray  # [x, y, z]
    rotation: np.ndarray  # [x, y, z, w] quaternion
    velocity: np.ndarray  # [vx, vy, vz]
    angular_velocity: np.ndarray  # [wx, wy, wz]
    speed: float
    steering_angle: float
    motor_torque: float
    brake_torque: float
    # Ground truth lane positions (optional, from Unity)
    ground_truth_left_lane_line_x: float = 0.0  # Left lane line (painted marking) position
    ground_truth_right_lane_line_x: float = 0.0  # Right lane line (painted marking) position
    ground_truth_lane_center_x: float = 0.0
    # Ground truth path information (for exact steering calculation verification)
    ground_truth_path_curvature: float = 0.0  # Path curvature (1/meters) - CRITICAL for verification!
    ground_truth_desired_heading: float = 0.0  # Desired heading (degrees) - for heading error analysis
    camera_8m_screen_y: float = -1.0  # Actual screen y pixel where 8m appears (from Unity's WorldToScreenPoint) - for distance calibration
    # NEW: Camera FOV values from Unity
    camera_field_of_view: float = -1.0 # Unity's Camera.fieldOfView (vertical FOV)
    camera_horizontal_fov: float = -1.0 # Calculated horizontal FOV from Unity
    # NEW: Camera position and forward direction from Unity (for debugging alignment)
    camera_pos_x: float = 0.0  # Camera position X (world coords)
    camera_pos_y: float = 0.0  # Camera position Y (world coords)
    camera_pos_z: float = 0.0  # Camera position Z (world coords)
    camera_forward_x: float = 0.0  # Camera forward X (normalized)
    camera_forward_y: float = 0.0  # Camera forward Y (normalized)
    camera_forward_z: float = 0.0  # Camera forward Z (normalized)
    # NEW: Debug fields for diagnosing ground truth offset issues
    road_center_at_car_x: float = 0.0  # Road center X at car's location (world coords)
    road_center_at_car_y: float = 0.0  # Road center Y at car's location (world coords)
    road_center_at_car_z: float = 0.0  # Road center Z at car's location (world coords)
    road_center_at_lookahead_x: float = 0.0  # Road center X at 8m lookahead (world coords)
    road_center_at_lookahead_y: float = 0.0  # Road center Y at 8m lookahead (world coords)
    road_center_at_lookahead_z: float = 0.0  # Road center Z at 8m lookahead (world coords)
    road_center_reference_t: float = 0.0  # Parameter t on road path for reference point


@dataclass
class ControlCommand:
    """Control command data."""
    timestamp: float
    steering: float  # -1.0 to 1.0
    throttle: float  # -1.0 to 1.0
    brake: float     # 0.0 to 1.0
    # Before/after limits (for analysis)
    steering_before_limits: Optional[float] = None
    throttle_before_limits: Optional[float] = None
    brake_before_limits: Optional[float] = None
    # PID internal state
    pid_integral: Optional[float] = None  # Lateral PID integral term
    pid_derivative: Optional[float] = None  # Lateral PID derivative term
    pid_error: Optional[float] = None  # Total error before PID
    # Error breakdown
    lateral_error: Optional[float] = None
    heading_error: Optional[float] = None
    total_error: Optional[float] = None
    # Exact steering calculation breakdown (for ground truth verification)
    calculated_steering_angle_deg: Optional[float] = None  # Steering angle in degrees (steerInput * maxSteerAngle)
    raw_steering: Optional[float] = None  # Steering before smoothing
    lateral_correction: Optional[float] = None  # Lateral correction term (k * lane_center)
    path_curvature_input: Optional[float] = None  # Path curvature used in calculation (1/meters)
    # NEW: Diagnostic fields for tracking stale data usage
    using_stale_perception: bool = False  # True if control is using stale perception data
    stale_perception_reason: Optional[str] = None  # Reason: "jump_detection", "perception_failure", "frozen", etc.


@dataclass
class PerceptionOutput:
    """Perception model output."""
    timestamp: float
    lane_lines: Optional[np.ndarray] = None  # Lane line coordinates
    lane_line_coefficients: Optional[np.ndarray] = None  # Polynomial coefficients
    confidence: Optional[float] = None
    detection_method: Optional[str] = None  # "ml" or "cv" (method used)
    num_lanes_detected: Optional[int] = None  # Number of lanes detected
    left_lane_line_x: Optional[float] = None  # Left lane line (painted marking) x position at lookahead (vehicle coords, meters)
    right_lane_line_x: Optional[float] = None  # Right lane line (painted marking) x position at lookahead (vehicle coords, meters)
    # NEW: Diagnostic fields for tracking stale data usage
    using_stale_data: bool = False  # True if we're using previous frame's values instead of current detection
    stale_data_reason: Optional[str] = None  # Reason for using stale data: "jump_detection", "perception_failure", "invalid_width", "frozen", etc.
    left_jump_magnitude: Optional[float] = None  # Magnitude of left lane jump (if jump detected)
    right_jump_magnitude: Optional[float] = None  # Magnitude of right lane jump (if jump detected)
    jump_threshold: Optional[float] = None  # Threshold used for jump detection


@dataclass
class TrajectoryOutput:
    """Trajectory planning output."""
    timestamp: float
    trajectory_points: Optional[np.ndarray] = None  # [N, 3] (x, y, heading)
    velocities: Optional[np.ndarray] = None  # [N] velocities at each point
    curvature: Optional[float] = None
    reference_point: Optional[Dict] = None  # Reference point dict with x, y, heading, velocity
    # NEW: Debug fields for tracking reference point calculation method
    reference_point_method: Optional[str] = None  # "lane_positions", "lane_coeffs", "trajectory", or None
    perception_center_x: Optional[float] = None  # Perception center before trajectory calculation (for comparison)


@dataclass
class UnityFeedback:
    """Unity feedback/status data (sent from Unity to Python)."""
    timestamp: float
    # Control status
    ground_truth_mode_active: bool = False  # Is ground truth mode actually active?
    control_command_received: bool = False  # Did Unity receive the control command?
    actual_steering_applied: Optional[float] = None  # Actual steering Unity applied (degrees)
    actual_throttle_applied: Optional[float] = None  # Actual throttle Unity applied
    actual_brake_applied: Optional[float] = None  # Actual brake Unity applied
    # Ground truth data status
    ground_truth_data_available: bool = False  # Is ground truth data being calculated?
    ground_truth_reporter_enabled: bool = False  # Is GroundTruthReporter enabled?
    path_curvature_calculated: bool = False  # Is path curvature being calculated?
    # Errors/warnings
    unity_errors: Optional[str] = None  # Unity errors (if any)
    unity_warnings: Optional[str] = None  # Unity warnings (if any)
    # Internal state (for debugging)
    car_controller_mode: Optional[str] = None  # "ground_truth" or "physics"
    av_control_enabled: bool = False  # Is AV control enabled?
    # Frame info
    unity_frame_count: Optional[int] = None  # Unity's frame count
    unity_time: Optional[float] = None  # Unity's Time.time


@dataclass
class RecordingFrame:
    """Complete frame of recorded data."""
    timestamp: float
    frame_id: int
    camera_frame: Optional[CameraFrame] = None
    vehicle_state: Optional[VehicleState] = None
    control_command: Optional[ControlCommand] = None
    perception_output: Optional[PerceptionOutput] = None
    trajectory_output: Optional[TrajectoryOutput] = None
    unity_feedback: Optional[UnityFeedback] = None  # NEW: Unity feedback data
    metadata: Optional[Dict[str, Any]] = None

