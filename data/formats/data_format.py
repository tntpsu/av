"""
Data format definitions for AV stack recordings.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
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
    steering_angle_actual: float = 0.0
    steering_input: float = 0.0
    desired_steer_angle: float = 0.0
    unity_time: float = 0.0
    unity_frame_count: int = 0
    unity_delta_time: float = 0.0
    unity_smooth_delta_time: float = 0.0
    unity_unscaled_delta_time: float = 0.0
    unity_time_scale: float = 1.0
    # Ground truth lane positions (optional, from Unity)
    ground_truth_left_lane_line_x: float = 0.0  # Left lane line (painted marking) position
    ground_truth_right_lane_line_x: float = 0.0  # Right lane line (painted marking) position
    ground_truth_lane_center_x: float = 0.0
    # Ground truth path information (for exact steering calculation verification)
    ground_truth_path_curvature: float = 0.0  # Path curvature (1/meters) - CRITICAL for verification!
    ground_truth_desired_heading: float = 0.0  # Desired heading (degrees) - for heading error analysis
    camera_8m_screen_y: float = -1.0  # Actual screen y pixel where 8m appears (from Unity's WorldToScreenPoint) - for distance calibration
    camera_lookahead_screen_y: float = -1.0  # Screen y pixel for ground truth lookahead distance
    ground_truth_lookahead_distance: float = 8.0  # Lookahead distance used for ground truth
    oracle_trajectory_world_xyz: Optional[np.ndarray] = None  # Flattened [x0,y0,z0,x1,y1,z1,...] world frame
    oracle_trajectory_screen_xy: Optional[np.ndarray] = None  # Flattened [u0,v0,u1,v1,...] image frame
    right_lane_fiducials_vehicle_xy: Optional[np.ndarray] = None  # Flattened [x0,y0,x1,y1,...]
    right_lane_fiducials_vehicle_true_xy: Optional[np.ndarray] = None  # Flattened [x0,y0,x1,y1,...] true vehicle frame
    right_lane_fiducials_vehicle_monotonic_xy: Optional[np.ndarray] = None  # Flattened [x0,y0,x1,y1,...] monotonic y=distanceAhead
    right_lane_fiducials_world_xyz: Optional[np.ndarray] = None  # Flattened [x0,y0,z0,x1,y1,z1,...] world frame
    right_lane_fiducials_screen_xy: Optional[np.ndarray] = None  # Flattened [u0,v0,u1,v1,...] in image coords
    right_lane_fiducials_point_count: int = 0
    right_lane_fiducials_horizon_meters: float = 0.0
    right_lane_fiducials_spacing_meters: float = 0.0
    right_lane_fiducials_enabled: bool = False
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
    # Top-down camera calibration/projection (for top-down overlay diagnostics)
    topdown_camera_pos_x: float = 0.0
    topdown_camera_pos_y: float = 0.0
    topdown_camera_pos_z: float = 0.0
    topdown_camera_forward_x: float = 0.0
    topdown_camera_forward_y: float = 0.0
    topdown_camera_forward_z: float = 0.0
    topdown_camera_orthographic_size: float = 0.0
    topdown_camera_field_of_view: float = 0.0
    # Stream consume-point lag diagnostics (instrumentation-only)
    stream_front_unity_dt_ms: float = 0.0
    stream_topdown_unity_dt_ms: float = 0.0
    stream_topdown_front_dt_ms: float = 0.0
    stream_topdown_front_frame_id_delta: float = 0.0
    stream_front_frame_id_delta: float = 0.0
    stream_topdown_frame_id_delta: float = 0.0
    stream_front_latest_age_ms: float = 0.0
    stream_front_queue_depth: float = 0.0
    stream_front_drop_count: float = 0.0
    stream_front_decode_in_flight: float = 0.0
    stream_topdown_latest_age_ms: float = 0.0
    stream_topdown_queue_depth: float = 0.0
    stream_topdown_drop_count: float = 0.0
    stream_topdown_decode_in_flight: float = 0.0
    stream_front_last_realtime_s: float = 0.0
    stream_topdown_last_realtime_s: float = 0.0
    stream_front_timestamp_minus_realtime_ms: float = 0.0
    stream_topdown_timestamp_minus_realtime_ms: float = 0.0
    # Stream clock-integrity diagnostics (instrumentation-only)
    stream_front_source_timestamp: float = 0.0
    stream_topdown_source_timestamp: float = 0.0
    stream_front_timestamp_reused: float = 0.0
    stream_topdown_timestamp_reused: float = 0.0
    stream_front_timestamp_non_monotonic: float = 0.0
    stream_topdown_timestamp_non_monotonic: float = 0.0
    stream_front_negative_frame_delta: float = 0.0
    stream_topdown_negative_frame_delta: float = 0.0
    stream_front_frame_id_reused: float = 0.0
    stream_topdown_frame_id_reused: float = 0.0
    stream_front_clock_jump: float = 0.0
    stream_topdown_clock_jump: float = 0.0
    # NEW: Debug fields for diagnosing ground truth offset issues
    road_center_at_car_x: float = 0.0  # Road center X at car's location (world coords)
    road_center_at_car_y: float = 0.0  # Road center Y at car's location (world coords)
    road_center_at_car_z: float = 0.0  # Road center Z at car's location (world coords)
    road_center_at_lookahead_x: float = 0.0  # Road center X at 8m lookahead (world coords)
    road_center_at_lookahead_y: float = 0.0  # Road center Y at 8m lookahead (world coords)
    road_center_at_lookahead_z: float = 0.0  # Road center Z at 8m lookahead (world coords)
    road_center_reference_t: float = 0.0  # Parameter t on road path for reference point
    # NEW: Road-frame alignment metrics (for diagnosing outside-of-lane bias)
    road_frame_lateral_offset: float = 0.0  # Lateral offset from road center (m, +right)
    road_heading_deg: float = 0.0  # Road tangent heading (deg)
    car_heading_deg: float = 0.0  # Car heading (deg)
    heading_delta_deg: float = 0.0  # Car - road heading delta (deg, -180..180)
    road_frame_lane_center_offset: float = 0.0  # Lookahead road center offset in road frame (m, +right)
    road_frame_lane_center_error: float = 0.0  # Car offset vs lookahead center (m, +right)
    vehicle_frame_lookahead_offset: float = 0.0  # Lookahead road center offset in vehicle frame (m, +right)
    # GT rotation input telemetry (exact Unity inputs used for target rotation).
    gt_rotation_debug_valid: bool = False
    gt_rotation_used_road_frame: bool = False
    gt_rotation_rejected_road_frame_hop: bool = False
    gt_rotation_reference_heading_deg: float = 0.0
    gt_rotation_road_frame_heading_deg: float = 0.0
    gt_rotation_input_heading_deg: float = 0.0
    gt_rotation_road_vs_ref_delta_deg: float = 0.0
    gt_rotation_applied_delta_deg: float = 0.0
    speed_limit: float = 0.0  # Speed limit at current reference point (m/s)
    speed_limit_preview: float = 0.0  # Speed limit at preview distance ahead (m/s)
    speed_limit_preview_distance: float = 0.0  # Preview distance used for speed limit (m)
    speed_limit_preview_min_distance: float = 0.0  # Distance to min limit in preview window (m)
    speed_limit_preview_mid: float = 0.0  # Speed limit at mid preview distance (m/s)
    speed_limit_preview_mid_distance: float = 0.0  # Mid preview distance (m)
    speed_limit_preview_mid_min_distance: float = 0.0  # Distance to min limit in mid window (m)
    speed_limit_preview_long: float = 0.0  # Speed limit at long preview distance (m/s)
    speed_limit_preview_long_distance: float = 0.0  # Long preview distance (m)
    speed_limit_preview_long_min_distance: float = 0.0  # Distance to min limit in long window (m)


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
    accel_feedforward: Optional[float] = None
    brake_feedforward: Optional[float] = None
    longitudinal_accel_capped: bool = False
    longitudinal_jerk_capped: bool = False
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
    feedforward_steering: Optional[float] = None  # Feedforward steering command
    feedback_steering: Optional[float] = None  # Feedback steering command from PID/Stanley
    total_error_scaled: Optional[float] = None  # Total error after scaling/deadband
    straight_sign_flip_override_active: Optional[bool] = None  # Override active for sign flip on straights
    straight_sign_flip_triggered: Optional[bool] = None  # Trigger condition became true this frame
    straight_sign_flip_trigger_error: Optional[float] = None  # |total_error_scaled| at trigger check
    straight_sign_flip_frames_remaining: Optional[int] = None  # Override frames remaining after update
    # NEW: Diagnostic fields for tracking stale data usage
    using_stale_perception: bool = False  # True if control is using stale perception data
    stale_perception_reason: Optional[str] = None  # Reason: "jump_detection", "perception_failure", "frozen", etc.
    # Straight-away stability metrics / tuning
    is_straight: Optional[bool] = None
    straight_oscillation_rate: Optional[float] = None
    tuned_deadband: Optional[float] = None
    tuned_error_smoothing_alpha: Optional[float] = None
    emergency_stop: bool = False
    # Target speed diagnostics
    target_speed_raw: Optional[float] = None
    target_speed_post_limits: Optional[float] = None
    target_speed_planned: Optional[float] = None
    target_speed_final: Optional[float] = None
    target_speed_slew_active: bool = False
    target_speed_ramp_active: bool = False
    # Launch throttle ramp diagnostics
    launch_throttle_cap: Optional[float] = None
    launch_throttle_cap_active: bool = False
    # Steering limiter waterfall diagnostics (for root-cause attribution)
    steering_pre_rate_limit: Optional[float] = None
    steering_post_rate_limit: Optional[float] = None
    steering_post_jerk_limit: Optional[float] = None
    steering_post_sign_flip: Optional[float] = None
    steering_post_hard_clip: Optional[float] = None
    steering_post_smoothing: Optional[float] = None
    steering_rate_limited_active: bool = False
    steering_jerk_limited_active: bool = False
    steering_hard_clip_active: bool = False
    steering_smoothing_active: bool = False
    steering_rate_limited_delta: Optional[float] = None
    steering_jerk_limited_delta: Optional[float] = None
    steering_hard_clip_delta: Optional[float] = None
    steering_smoothing_delta: Optional[float] = None


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
    # NEW: Diagnostic fields for perception instability
    actual_detected_left_lane_x: Optional[float] = None  # Actual detected left lane position (when rejected due to instability)
    actual_detected_right_lane_x: Optional[float] = None  # Actual detected right lane position (when rejected due to instability)
    instability_width_change: Optional[float] = None  # Width change that triggered instability detection
    instability_center_shift: Optional[float] = None  # Center shift that triggered instability detection
    # NEW: Perception health monitoring fields
    consecutive_bad_detection_frames: int = 0  # Number of consecutive frames with <2 lanes detected
    perception_health_score: float = 1.0  # Health score: 1.0 = perfect, 0.0 = failed (based on recent detection rate)
    perception_health_status: str = "healthy"  # "healthy", "degraded", "poor", "critical"
    perception_bad_events: Optional[List[str]] = None  # Reasons that lowered health for this frame
    perception_bad_events_recent: Optional[List[str]] = None  # Most recent bad events in health window
    perception_timestamp_frozen: bool = False  # True if perception timestamp did not advance
    perception_clamp_events: Optional[List[str]] = None  # Clamp/override events for debugging
    reject_reason: Optional[str] = None  # Primary reason lane positions were rejected
    # NEW: Points used for polynomial fitting (for debug visualization)
    fit_points_left: Optional[np.ndarray] = None  # [[x, y], ...] points used for left lane fit
    fit_points_right: Optional[np.ndarray] = None  # [[x, y], ...] points used for right lane fit
    # NEW: Segmentation mask (PNG bytes, label map with 0/1/2 classes)
    segmentation_mask_png: Optional[bytes] = None


@dataclass
class TrajectoryOutput:
    """Trajectory planning output."""
    timestamp: float
    trajectory_points: Optional[np.ndarray] = None  # [N, 3] (x, y, heading)
    oracle_points: Optional[np.ndarray] = None  # [N, 2] oracle centerline samples (x, y)
    oracle_point_count: Optional[int] = None
    oracle_horizon_meters: Optional[float] = None
    oracle_point_spacing_meters: Optional[float] = None
    oracle_samples_enabled: bool = False
    velocities: Optional[np.ndarray] = None  # [N] velocities at each point
    curvature: Optional[float] = None
    reference_point: Optional[Dict] = None  # Reference point dict with x, y, heading, velocity
    trajectory_source: Optional[str] = None  # "planner" or "oracle"
    # NEW: Debug fields for tracking reference point calculation method
    reference_point_method: Optional[str] = None  # "lane_positions", "lane_coeffs", "trajectory", or None
    perception_center_x: Optional[float] = None  # Perception center before trajectory calculation (for comparison)
    # Planner source-isolation diagnostics (instrumentation-only)
    diag_available: Optional[float] = None
    diag_generated_by_fallback: Optional[float] = None
    diag_points_generated: Optional[float] = None
    diag_x_clip_count: Optional[float] = None
    diag_pre_y0: Optional[float] = None
    diag_pre_y1: Optional[float] = None
    diag_pre_y2: Optional[float] = None
    diag_post_y0: Optional[float] = None
    diag_post_y1: Optional[float] = None
    diag_post_y2: Optional[float] = None
    diag_used_provided_distance0: Optional[float] = None
    diag_used_provided_distance1: Optional[float] = None
    diag_used_provided_distance2: Optional[float] = None
    diag_post_minus_pre_y0: Optional[float] = None
    diag_post_minus_pre_y1: Optional[float] = None
    diag_post_minus_pre_y2: Optional[float] = None
    diag_preclip_x0: Optional[float] = None
    diag_preclip_x1: Optional[float] = None
    diag_preclip_x2: Optional[float] = None
    diag_preclip_x_abs_max: Optional[float] = None
    diag_preclip_x_abs_p95: Optional[float] = None
    diag_postclip_x0: Optional[float] = None
    diag_postclip_x1: Optional[float] = None
    diag_postclip_x2: Optional[float] = None
    diag_first_segment_y0_gt_y1_pre: Optional[float] = None
    diag_first_segment_y0_gt_y1_post: Optional[float] = None
    diag_inversion_introduced_after_conversion: Optional[float] = None


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
    camera_topdown_frame: Optional[CameraFrame] = None
    vehicle_state: Optional[VehicleState] = None
    control_command: Optional[ControlCommand] = None
    perception_output: Optional[PerceptionOutput] = None
    trajectory_output: Optional[TrajectoryOutput] = None
    unity_feedback: Optional[UnityFeedback] = None  # NEW: Unity feedback data
    metadata: Optional[Dict[str, Any]] = None

