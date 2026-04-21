"""
Data format definitions for AV stack recordings.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np

from sync_contract import PACKET_MODE_LATEST_PARALLEL


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
    sync_packet_mode: str = PACKET_MODE_LATEST_PARALLEL
    sync_packet_schema_version: int = 0
    sync_packet_id: int = -1
    sync_packet_unity_frame_count: int = -1
    sync_packet_consume_policy: str = ""
    sync_packet_complete: Optional[bool] = None
    sync_packet_fallback_active: Optional[bool] = None
    sync_packet_fallback_reason_code: str = ""
    sync_packet_queue_depth: int = 0
    sync_packet_drop_count: int = 0
    sync_packet_payload_queue_depth: int = 0
    sync_packet_payload_drop_count: int = 0
    sync_packet_orphan_camera_count: int = 0
    sync_packet_orphan_vehicle_count: int = 0
    sync_packet_timeout_count: int = 0
    sync_packet_skipped_unity_frames: int = 0
    sync_packet_age_ms: float = np.nan
    sync_packet_payload_oldest_age_ms: float = np.nan
    sync_packet_payload_bytes: int = 0
    sync_packet_payload_fallback_reason_code: str = ""
    sync_packet_payload_selected_age_ms: float = np.nan
    sync_packet_payload_selected_fresh: Optional[bool] = None
    sync_packet_payload_warn_age_exceeded: Optional[bool] = None
    sync_packet_payload_stale_drop_count: int = 0
    sync_packet_payload_drained_count: int = 0
    sync_packet_payload_max_drained_age_ms: float = np.nan
    sync_packet_payload_selection_source: str = ""
    sync_packet_payload_selection_fallback_active: Optional[bool] = None
    sync_packet_payload_selection_fallback_reason_code: str = ""
    sync_packet_payload_server_queue_depth_after_select: int = 0
    sync_packet_payload_server_oldest_age_ms_after_select: float = np.nan
    sync_packet_selection_result: str = ""
    sync_packet_join_source: str = ""
    sync_packet_join_key_present: Optional[bool] = None
    sync_packet_join_failure_reason_code: str = ""
    sync_packet_join_failure_side_code: str = ""
    sync_packet_selected_failure_contract_reason_code: str = ""
    sync_packet_selected_failure_source_stage_code: str = ""
    sync_packet_source_key_present_camera: Optional[bool] = None
    sync_packet_source_key_present_vehicle: Optional[bool] = None
    sync_packet_selected_packet_key: str = ""
    sync_packet_timeout_event_delta: int = 0
    sync_packet_coherence_pass: Optional[bool] = None
    sync_packet_coherence_reason_code: str = ""
    sync_packet_complete_but_incoherent: Optional[bool] = None
    sync_packet_front_vehicle_time_delta_budget_exceeded: Optional[bool] = None
    sync_packet_front_vehicle_frame_delta_budget_exceeded: Optional[bool] = None
    sync_packet_join_wait_budget_exceeded: Optional[bool] = None
    sync_packet_component_age_budget_exceeded: Optional[bool] = None
    sync_packet_source_context_queue_depth: int = 0
    sync_packet_source_context_dropped_stale_count: int = 0
    sync_packet_source_context_missing_count: int = 0
    sync_packet_source_context_frame_delta: float = np.nan
    sync_packet_source_context_time_delta_ms: float = np.nan
    sync_packet_source_bundle_close_reason: str = ""
    sync_packet_source_bundle_deadline_ms: float = np.nan
    sync_packet_source_bundle_age_ms: float = np.nan
    sync_packet_source_bundle_inflight_count: int = 0
    sync_packet_source_bundle_vehicle_state_built: Optional[bool] = None
    sync_packet_source_bundle_vehicle_state_enqueued: Optional[bool] = None
    sync_packet_source_bundle_vehicle_state_sent: Optional[bool] = None
    sync_packet_source_bundle_camera_requested: Optional[bool] = None
    sync_packet_source_camera_request_attempted: Optional[bool] = None
    sync_packet_source_camera_request_accepted: Optional[bool] = None
    sync_packet_source_camera_request_rejected_reason: str = ""
    sync_packet_source_camera_request_skipped_reason: str = ""
    sync_packet_source_camera_request_disposition_code: str = ""
    sync_packet_source_camera_request_attempt_age_ms: float = np.nan
    sync_packet_source_camera_request_accept_age_ms: float = np.nan
    sync_packet_source_camera_request_queue_depth: int = 0
    sync_packet_source_bundle_active_transport_eligible: Optional[bool] = None
    sync_packet_source_bundle_debug_unbundled_capture: Optional[bool] = None
    sync_packet_camera_capture_contract_reason: str = ""
    sync_packet_source_bundle_camera_sent: Optional[bool] = None
    sync_packet_source_bundle_aborted_before_vehicle_send: Optional[bool] = None
    sync_packet_source_bundle_abort_reason: str = ""
    sync_packet_source_vehicle_send_blocked_by_camera_request: Optional[bool] = None
    sync_packet_source_bundle_superseded_before_send: Optional[bool] = None
    sync_packet_active_camera_excluded_event_delta: int = 0
    sync_packet_active_camera_excluded_reason_code: str = ""
    sync_packet_unbundled_camera_entered_active_path_event_delta: int = 0
    sync_packet_join_wait_ms: float = np.nan
    sync_packet_key_match_count: int = 0
    sync_packet_unity_fallback_count: int = 0
    sync_packet_superseded_camera_count: int = 0
    sync_packet_superseded_vehicle_count: int = 0
    sync_packet_packet_superseded_camera_count: int = 0
    sync_packet_packet_superseded_vehicle_count: int = 0
    sync_front_age_ms: float = np.nan
    sync_vehicle_age_ms: float = np.nan
    sync_front_vehicle_frame_delta: float = np.nan
    sync_front_vehicle_time_delta_ms: float = np.nan
    sync_packet_missing_front: Optional[bool] = None
    sync_packet_missing_vehicle: Optional[bool] = None
    # Ground truth lane positions (optional, from Unity)
    ground_truth_left_lane_line_x: float = 0.0  # Left lane line (painted marking) position
    ground_truth_right_lane_line_x: float = 0.0  # Right lane line (painted marking) position
    ground_truth_lane_center_x: float = 0.0  # Legacy alias: selected lane center at lookahead
    ground_truth_lane_center_x_lookahead: float = 0.0
    ground_truth_lane_center_x_at_car: float = 0.0
    ground_truth_selected_lane_index: int = 0
    ground_truth_ego_lane_index: int = 0
    ground_truth_lane_selection_source: str = ""
    ground_truth_lane_selection_reason: str = ""
    ground_truth_lane_selection_matches_ego: bool = True
    ground_truth_ego_lane_center_x_at_car: float = 0.0
    ground_truth_selected_lane_center_offset_road_frame: float = 0.0
    ground_truth_selected_lane_cross_track_road_frame_at_car: float = 0.0
    ground_truth_ego_lane_center_offset_road_frame: float = 0.0
    ground_truth_ego_lane_cross_track_road_frame_at_car: float = 0.0
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
    # Grade/pitch/roll telemetry (Step 3)
    pitch_rad: float = 0.0       # Vehicle pitch (positive = nose up)
    roll_rad: float = 0.0        # Vehicle roll (positive = right lean)
    road_grade: float = 0.0      # Local road grade (rise/run) from track profile
    # Chassis-ground telemetry (high-rate vehicle-state path)
    chassis_ground_min_clearance_m: float = np.nan
    chassis_ground_effective_min_clearance_m: float = np.nan
    chassis_ground_clearance_m: float = np.nan
    chassis_ground_penetration_m: float = np.nan
    chassis_ground_contact: bool = False
    wheel_grounded_count: int = 0
    wheel_colliders_ready: bool = False
    force_fallback_active: bool = False

    # Per-wheel diagnostics (4 wheels: FL, FR, RL, RR) — Step 3D
    wheel_sideways_slip: Optional[np.ndarray] = None  # shape (4,)
    wheel_forward_slip: Optional[np.ndarray] = None   # shape (4,)
    wheel_contact_force: Optional[np.ndarray] = None   # shape (4,)
    wheel_rpm: Optional[np.ndarray] = None             # shape (4,)
    wheel_sprung_mass: Optional[np.ndarray] = None     # shape (4,)
    wheel_contact_normal_y: Optional[np.ndarray] = None # shape (4,)
    wheel_steer_angle_actual: float = 0.0
    # ACC / forward radar (Step 5 — Phase A data pipeline)
    radar_fwd_detected: float = 0.0        # 1.0 when target detected, 0.0 otherwise
    radar_fwd_distance_m: float = 0.0      # EMA-filtered range to lead vehicle (m)
    radar_fwd_range_rate_mps: float = 0.0  # EMA-filtered range rate (+ = closing)
    radar_fwd_snr: float = 0.0             # Signal-to-noise proxy [0, 1]
    radar_fwd_candidate_present: float = 0.0
    radar_fwd_reject_reason: str = "no_candidate"
    radar_fwd_target_azimuth_deg: float = 0.0
    radar_fwd_target_heading_delta_deg: float = 0.0
    radar_fwd_target_same_lane_confidence: float = -1.0
    radar_fwd_target_lane_offset_m: float = 0.0
    radar_fwd_target_arc_distance_m: float = 0.0
    radar_fwd_association_eligible: float = 0.0
    radar_fwd_track_active: float = 0.0
    radar_fwd_track_source: str = "none"
    radar_fwd_track_age_ms: float = 0.0
    radar_fwd_track_confidence: float = 0.0
    radar_fwd_track_hold_reason: str = "none"
    radar_fwd_track_drop_reason: str = "none"
    lead_collision_detected: bool = False
    lead_collision_override_active: bool = False
    acc_active: float = 0.0               # 1.0 when ACC is engaged
    acc_target_gap_m: float = 0.0         # IDM desired gap (m)
    acc_gap_error_m: float = 0.0          # actual_gap − target_gap (m)
    acc_ttc_s: float = 999.0              # Time-to-collision (s); 999 = no target
    acc_state_code: str = ""
    acc_target_speed_mps: float = 0.0
    acc_request_estop: bool = False
    acc_safety_mode_code: str = "none"
    acc_idm_dynamic_gap_m: float = 0.0
    acc_idm_equilibrium_gap_m: float = 0.0
    acc_idm_accel_mps2: float = 0.0
    acc_lead_speed_estimate_mps: float = 0.0
    acc_closure_reserve_mps: float = 0.0
    acc_convergence_mode: str = "unavailable"
    acc_detection_stable_frames: float = 0.0
    acc_recent_detection_loss: bool = False
    acc_detection_loss_event_delta: float = 0.0
    acc_no_detect_run_length: float = 0.0


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
    longitudinal_limiter_transition_active: bool = False
    longitudinal_limiter_state_code: Optional[float] = None
    longitudinal_accel_cmd_raw: Optional[float] = None
    longitudinal_accel_cmd_smoothed: Optional[float] = None
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
    path_curvature_source_used: Optional[str] = None
    path_curvature_primary_abs: Optional[float] = None
    path_curvature_lane_abs: Optional[float] = None
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
    is_control_straight_proxy: Optional[bool] = None
    curve_upcoming: Optional[bool] = None
    curve_at_car: Optional[bool] = None
    curve_at_car_distance_remaining_m: Optional[float] = None
    curve_phase_source: Optional[str] = None
    curve_phase_use_preview_curvature: Optional[bool] = None
    curve_phase_preview_curvature_abs: Optional[float] = None
    curve_phase_preview_curvature_valid: Optional[bool] = None
    curve_phase_preview_upcoming: Optional[bool] = None
    curve_phase_preview_enter_threshold: Optional[float] = None
    curve_phase_preview_exit_threshold: Optional[float] = None
    curve_scheduler_mode: Optional[str] = None
    curve_phase: Optional[float] = None
    curve_phase_raw: Optional[float] = None
    curve_phase_state: Optional[str] = None
    curve_phase_rearm_event: Optional[bool] = None
    curve_phase_entry_frames: Optional[int] = None
    curve_phase_rearm_hold_frames: Optional[int] = None
    curve_phase_term_preview: Optional[float] = None
    curve_phase_term_path: Optional[float] = None
    curve_phase_term_rise: Optional[float] = None
    curve_phase_term_time: Optional[float] = None
    curve_phase_curvature_rise_abs: Optional[float] = None
    curve_preview_far_upcoming: Optional[bool] = None
    curve_preview_far_phase: Optional[float] = None
    curve_local_phase: Optional[float] = None
    curve_local_phase_raw: Optional[float] = None
    curve_local_state: Optional[str] = None
    curve_local_phase_source: Optional[str] = None
    curve_local_entry_driver: Optional[str] = None
    curve_local_entry_severity: Optional[float] = None
    curve_local_entry_on_effective: Optional[float] = None
    curve_local_phase_distance_start_effective_m: Optional[float] = None
    curve_local_phase_time_start_effective_s: Optional[float] = None
    curve_local_arm_ready: Optional[bool] = None
    curve_local_time_ready: Optional[bool] = None
    curve_local_in_curve_now: Optional[bool] = None
    curve_local_commit_ready: Optional[bool] = None
    curve_local_commit_driver: Optional[str] = None
    curve_local_arm_phase_raw: Optional[float] = None
    curve_local_sustain_phase_raw: Optional[float] = None
    curve_local_path_sustain_active: Optional[bool] = None
    curve_local_distance_ready: Optional[bool] = None
    curve_local_distance_horizon_m: Optional[float] = None
    curve_local_time_horizon_s: Optional[float] = None
    curve_local_reentry_ready: Optional[bool] = None
    curve_activation_blocker_mode: Optional[str] = None
    curve_local_arm_phase_deficit: Optional[float] = None
    curve_local_arm_effect_score: Optional[float] = None
    curve_local_arm_effect_heading_term: Optional[float] = None
    curve_local_arm_effect_lateral_shift_term: Optional[float] = None
    curve_local_arm_effect_time_support_term: Optional[float] = None
    curve_local_dynamic_sustain_effect_score: Optional[float] = None
    curve_preactivation_authority_weight: Optional[float] = None
    curve_preactivation_authority_active: Optional[bool] = None
    curve_preactivation_blocker_mode: Optional[str] = None
    curve_preactivation_preview_weight: Optional[float] = None
    curve_preactivation_speed_weight: Optional[float] = None
    curve_preactivation_curvature_weight: Optional[float] = None
    curve_preactivation_distance_weight: Optional[float] = None
    curve_preactivation_kappa_floor: Optional[float] = None
    curve_preactivation_lookahead_target: Optional[float] = None
    curve_preactivation_speed_cap_target: Optional[float] = None
    curve_local_rearm_cooldown_active: Optional[bool] = None
    curve_local_force_straight_active: Optional[bool] = None
    curve_local_commit_streak_frames: Optional[int] = None
    curve_intent: Optional[float] = None
    curve_intent_raw: Optional[float] = None
    curve_intent_state: Optional[str] = None
    curve_intent_term_preview: Optional[float] = None
    curve_intent_term_path: Optional[float] = None
    curve_intent_term_rise: Optional[float] = None
    curve_intent_confidence: Optional[float] = None
    curve_intent_watchdog_triggered: Optional[bool] = None
    curve_intent_speed_guardrail_active: Optional[bool] = None
    curve_intent_speed_guardrail_cap_mps: Optional[float] = None
    curve_intent_speed_guardrail_confidence: Optional[float] = None
    curve_anticipation_score: Optional[float] = None
    curve_anticipation_score_raw: Optional[float] = None
    curve_anticipation_active: Optional[bool] = None
    curve_anticipation_source: Optional[str] = None
    curve_anticipation_term_curvature: Optional[float] = None
    curve_anticipation_term_heading: Optional[float] = None
    curve_anticipation_term_far_rise: Optional[float] = None
    curve_anticipation_distance_weight: Optional[float] = None
    reference_lookahead_target: Optional[float] = None
    reference_lookahead_target_pre_entry_guard: Optional[float] = None
    reference_lookahead_after_slew: Optional[float] = None
    reference_lookahead_active: Optional[float] = None
    reference_lookahead_owner_nominal_target: Optional[float] = None
    reference_lookahead_owner_commit_band_target: Optional[float] = None
    reference_lookahead_owner_entry_progress: Optional[float] = None
    reference_lookahead_owner_commit_distance_progress: Optional[float] = None
    reference_lookahead_owner_commit_phase_progress: Optional[float] = None
    reference_lookahead_owner_commit_progress: Optional[float] = None
    reference_lookahead_owner_commit_distance_start_effective_m: Optional[float] = None
    reference_lookahead_owner_commit_band_clamp_active: Optional[bool] = None
    reference_lookahead_owner_commit_band_clamp_delta_m: Optional[float] = None
    reference_lookahead_local_gate_weight: Optional[float] = None
    reference_lookahead_owner_mode: Optional[str] = None
    reference_lookahead_entry_weight_source: Optional[str] = None
    reference_lookahead_fallback_active: Optional[bool] = None
    reference_lookahead_entry_shorten_guard_active: Optional[bool] = None
    reference_lookahead_entry_shorten_guard_delta_m: Optional[float] = None
    local_curve_reference_mode: Optional[str] = None
    local_curve_reference_requested_mode: Optional[str] = None
    local_curve_reference_active: Optional[bool] = None
    local_curve_reference_shadow_only: Optional[bool] = None
    local_curve_reference_shadow_promotion_active: Optional[bool] = None
    local_curve_reference_shadow_promotion_weight: Optional[float] = None
    local_curve_reference_shadow_promotion_blend_floor: Optional[float] = None
    local_curve_reference_shadow_promotion_reason: Optional[str] = None
    local_curve_reference_guarded_bounded_active: Optional[bool] = None
    local_curve_reference_guarded_bounded_reason: Optional[str] = None
    local_curve_reference_guarded_bounded_dwell_frames: Optional[int] = None
    local_curve_reference_guarded_bounded_trigger_raw_delta_m: Optional[float] = None
    local_curve_reference_guarded_bounded_exit_raw_delta_m: Optional[float] = None
    local_curve_reference_guarded_bounded_trigger_weight: Optional[float] = None
    local_curve_reference_guarded_bounded_blend_floor: Optional[float] = None
    reference_distractor_guard_active: Optional[bool] = None
    reference_distractor_guard_reason: Optional[str] = None
    reference_distractor_guard_dwell_frames: Optional[int] = None
    reference_distractor_guard_trigger_center_error_m: Optional[float] = None
    reference_distractor_guard_exit_center_error_m: Optional[float] = None
    reference_distractor_guard_center_error_m: Optional[float] = None
    reference_distractor_guard_width_error_m: Optional[float] = None
    reference_distractor_guard_trigger_weight: Optional[float] = None
    reference_distractor_guard_blend_weight: Optional[float] = None
    reference_distractor_guard_expected_center_x_m: Optional[float] = None
    reference_distractor_guard_expected_heading_rad: Optional[float] = None
    reference_distractor_input_guard_active: Optional[bool] = None
    reference_distractor_input_guard_reason: Optional[str] = None
    reference_distractor_input_guard_dwell_frames: Optional[int] = None
    reference_distractor_input_guard_trigger_center_error_m: Optional[float] = None
    reference_distractor_input_guard_exit_center_error_m: Optional[float] = None
    reference_distractor_input_guard_center_error_m: Optional[float] = None
    reference_distractor_input_guard_width_error_m: Optional[float] = None
    reference_distractor_input_guard_trigger_weight: Optional[float] = None
    reference_distractor_input_guard_expected_center_x_m: Optional[float] = None
    reference_distractor_input_guard_synthetic_left_lane_x_m: Optional[float] = None
    reference_distractor_input_guard_synthetic_right_lane_x_m: Optional[float] = None
    reference_distractor_input_guard_synthetic_lane_width_m: Optional[float] = None
    reference_distractor_input_guard_suppressed_lane_coeffs: Optional[bool] = None
    reference_distractor_input_guard_center_history_seeded: Optional[bool] = None
    local_curve_reference_valid: Optional[bool] = None
    local_curve_reference_source: Optional[str] = None
    local_curve_reference_fallback_active: Optional[bool] = None
    local_curve_reference_fallback_reason: Optional[str] = None
    local_curve_reference_blend_weight: Optional[float] = None
    local_curve_reference_progress_weight: Optional[float] = None
    local_curve_reference_arc_curvature_abs: Optional[float] = None
    local_curve_reference_target_x: Optional[float] = None
    local_curve_reference_target_y: Optional[float] = None
    local_curve_reference_target_heading: Optional[float] = None
    local_curve_reference_target_distance_m: Optional[float] = None
    local_curve_reference_vs_planner_delta_m: Optional[float] = None
    local_curve_reference_raw_delta_m: Optional[float] = None
    local_curve_reference_capped_delta_m: Optional[float] = None
    local_curve_reference_cap_active: Optional[bool] = None
    local_curve_reference_curve_direction_sign: Optional[float] = None
    local_curve_reference_curve_progress_ratio: Optional[float] = None
    local_curve_reference_distance_to_curve_start_m: Optional[float] = None
    distance_to_next_curve_start_m: Optional[float] = None
    time_to_next_curve_start_s: Optional[float] = None
    is_road_straight: Optional[bool] = None
    road_curvature_valid: Optional[bool] = None
    road_curvature_abs: Optional[float] = None
    road_curvature_source: Optional[str] = None
    straight_oscillation_rate: Optional[float] = None
    tuned_deadband: Optional[float] = None
    tuned_error_smoothing_alpha: Optional[float] = None
    # Grade-aware lateral smoothing (LateralController): alpha reduction from road grade (0..0.3)
    lateral_grade_damping: Optional[float] = None
    # Blend weight used for lateral error smoothing after grade damping
    lateral_error_smoothing_alpha_effective: Optional[float] = None
    emergency_stop: bool = False
    # Pipeline timing diagnostics (input-ready -> command-sent, monotonic clock domain)
    e2e_front_ready_mono_s: Optional[float] = None
    e2e_vehicle_ready_mono_s: Optional[float] = None
    e2e_inputs_ready_mono_s: Optional[float] = None
    e2e_control_sent_mono_s: Optional[float] = None
    e2e_latency_ms: Optional[float] = None
    e2e_latency_mode: Optional[str] = None
    # Per-layer wall time (ms, perf_counter) — see docs/plans/perf_layer_timings_impl.md
    perf_perception_ms: Optional[float] = None
    perf_planning_ms: Optional[float] = None
    perf_control_ms: Optional[float] = None
    perf_hdf5_write_ms: Optional[float] = None
    perf_wait_input_ms: Optional[float] = None
    teleport_detected: bool = False
    teleport_jump_m: Optional[float] = None
    teleport_guard_suppressed: bool = False
    teleport_continuity_suspect: bool = False
    teleport_guard_reason_code: str = ""
    teleport_dynamic_threshold_m: Optional[float] = None
    teleport_hard_override_threshold_m: Optional[float] = None
    teleport_effective_dt_s: Optional[float] = None
    teleport_unity_dt_s: Optional[float] = None
    # Target speed diagnostics
    target_speed_raw: Optional[float] = None
    target_speed_post_limits: Optional[float] = None
    target_speed_planned: Optional[float] = None
    target_speed_final: Optional[float] = None
    governor_target_speed_mps: Optional[float] = None
    acc_target_speed_mps: Optional[float] = None
    planner_target_speed_applied_mps: Optional[float] = None
    final_longitudinal_target_mps: Optional[float] = None
    final_longitudinal_owner_code: str = ""
    reference_velocity_source_code: str = ""
    target_speed_slew_active: bool = False
    target_speed_ramp_active: bool = False
    reference_velocity_effective: Optional[float] = None
    post_jump_cooldown_active: bool = False
    post_jump_cooldown_frames_remaining: int = 0
    post_jump_reason_code: str = ""
    teleport_expected_motion_m: Optional[float] = None
    teleport_motion_ratio: Optional[float] = None
    # Speed governor diagnostics
    speed_governor_active_limiter: str = "none"
    speed_governor_active_limiter_code: int = 0
    speed_governor_comfort_speed: Optional[float] = None
    speed_governor_preview_speed: Optional[float] = None
    speed_governor_horizon_speed: Optional[float] = None
    speed_governor_curve_cap_speed: Optional[float] = None
    speed_governor_curve_cap_active: Optional[bool] = None
    speed_governor_curve_cap_reason: Optional[str] = None
    speed_governor_curve_cap_margin_mps: Optional[float] = None
    speed_governor_curve_cap_shadow_mode: Optional[bool] = None
    speed_governor_feasibility_backstop_active: Optional[bool] = None
    speed_governor_feasibility_backstop_speed: Optional[float] = None
    speed_governor_cap_tracking_active: bool = False
    speed_governor_cap_tracking_error_mps: Optional[float] = None
    speed_governor_cap_tracking_mode: str = "inactive"
    speed_governor_cap_tracking_mode_code: int = 0
    speed_governor_cap_tracking_recovery_frames: int = 0
    speed_governor_cap_tracking_hard_ceiling_applied: bool = False
    # Velocity profile diagnostics
    velocity_profile_speed_mps: Optional[float] = None
    velocity_profile_active: bool = False
    velocity_profile_station_m: float = 0.0
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
    steering_rate_limit_base_from_error: Optional[float] = None
    steering_rate_limit_curve_scale: Optional[float] = None
    steering_rate_limit_curve_metric_abs: Optional[float] = None
    steering_rate_limit_curve_metric_source: Optional[str] = None
    steering_rate_limit_curve_min: Optional[float] = None
    steering_rate_limit_curve_max: Optional[float] = None
    steering_rate_limit_scale_min: Optional[float] = None
    steering_rate_limit_curve_regime_code: Optional[float] = None
    steering_rate_limit_after_curve: Optional[float] = None
    steering_rate_limit_after_floor: Optional[float] = None
    steering_rate_limit_effective: Optional[float] = None
    steering_rate_limit_effective_raw: Optional[float] = None
    steering_rate_limit_effective_smoothed: Optional[float] = None
    steering_rate_limit_transition_active: bool = False
    steering_rate_limit_requested_delta: Optional[float] = None
    steering_rate_limit_margin: Optional[float] = None
    steering_rate_limit_unlock_delta_needed: Optional[float] = None
    curve_entry_assist_active: bool = False
    curve_entry_assist_triggered: bool = False
    curve_entry_assist_rearm_frames_remaining: Optional[int] = None
    dynamic_curve_authority_active: bool = False
    dynamic_curve_rate_request_delta: Optional[float] = None
    dynamic_curve_rate_deficit: Optional[float] = None
    dynamic_curve_rate_boost: Optional[float] = None
    dynamic_curve_jerk_boost_factor: Optional[float] = None
    dynamic_curve_lateral_accel_est_g: Optional[float] = None
    dynamic_curve_lateral_jerk_est_gps: Optional[float] = None
    dynamic_curve_lateral_jerk_est_smoothed_gps: Optional[float] = None
    dynamic_curve_speed_scale: Optional[float] = None
    dynamic_curve_comfort_scale: Optional[float] = None
    dynamic_curve_comfort_accel_gate: Optional[float] = None
    dynamic_curve_comfort_jerk_penalty: Optional[float] = None
    dynamic_curve_rate_boost_cap_effective: Optional[float] = None
    dynamic_curve_jerk_boost_cap_effective: Optional[float] = None
    dynamic_curve_hard_clip_boost: Optional[float] = None
    dynamic_curve_hard_clip_boost_cap_effective: Optional[float] = None
    dynamic_curve_hard_clip_limit_effective: Optional[float] = None
    dynamic_curve_entry_governor_active: Optional[bool] = None
    dynamic_curve_entry_governor_scale: Optional[float] = None
    dynamic_curve_authority_deficit_streak: Optional[int] = None
    curve_entry_schedule_active: bool = False
    curve_entry_schedule_triggered: bool = False
    curve_entry_schedule_handoff_triggered: bool = False
    curve_entry_schedule_frames_remaining: Optional[int] = None
    curve_commit_mode_active: bool = False
    curve_commit_mode_triggered: bool = False
    curve_commit_mode_handoff_triggered: bool = False
    curve_commit_mode_frames_remaining: Optional[int] = None
    steering_jerk_limit_effective: Optional[float] = None
    steering_jerk_curve_scale: Optional[float] = None
    steering_jerk_limit_requested_rate_delta: Optional[float] = None
    steering_jerk_limit_allowed_rate_delta: Optional[float] = None
    steering_jerk_limit_margin: Optional[float] = None
    steering_jerk_limit_unlock_rate_delta_needed: Optional[float] = None
    # Control attribution helpers for pre-failure triage
    steering_authority_gap: Optional[float] = None
    steering_transfer_ratio: Optional[float] = None
    steering_first_limiter_stage_code: Optional[float] = None  # 0=none,1=rate,2=jerk,3=hard_clip,4=smoothing
    # Explicit unwind policy diagnostics (Phase 2)
    curve_unwind_active: Optional[bool] = None
    curve_unwind_frames_remaining: Optional[int] = None
    curve_unwind_progress: Optional[float] = None
    curve_unwind_rate_scale: Optional[float] = None
    curve_unwind_jerk_scale: Optional[float] = None
    curve_unwind_integral_decay_applied: Optional[float] = None
    # Telemetry-only turn feasibility governor diagnostics (no behavior change in Phase 1)
    turn_feasibility_active: Optional[bool] = None
    turn_feasibility_infeasible: Optional[bool] = None
    turn_feasibility_curvature_abs: Optional[float] = None
    turn_feasibility_speed_mps: Optional[float] = None
    turn_feasibility_required_lat_accel_g: Optional[float] = None
    turn_feasibility_comfort_limit_g: Optional[float] = None
    turn_feasibility_peak_limit_g: Optional[float] = None
    turn_feasibility_selected_limit_g: Optional[float] = None
    turn_feasibility_guardband_g: Optional[float] = None
    turn_feasibility_margin_g: Optional[float] = None
    turn_feasibility_speed_limit_mps: Optional[float] = None
    turn_feasibility_speed_delta_mps: Optional[float] = None
    turn_feasibility_use_peak_bound: Optional[bool] = None
    curvature_primary_abs: Optional[float] = None
    curvature_primary_source: Optional[str] = None
    curvature_map_abs: Optional[float] = None
    curvature_lane_context_abs: Optional[float] = None
    curvature_preview_abs: Optional[float] = None
    curvature_source_diverged: Optional[bool] = None
    curvature_map_authority_lost: Optional[bool] = None
    curvature_source_divergence_abs: Optional[float] = None
    curvature_selection_reason: Optional[str] = None
    map_health_ok: Optional[bool] = None
    track_match_ok: Optional[bool] = None
    map_segment_lookup_success_rate: Optional[float] = None
    map_teleport_skip_count: Optional[int] = None
    map_odometer_jump_rate: Optional[float] = None
    curvature_contract_consistent_controller: Optional[bool] = None
    curvature_contract_consistent_governor: Optional[bool] = None
    curvature_contract_consistent_intent: Optional[bool] = None
    curvature_contract_consistent_all: Optional[bool] = None
    curvature_contract_mismatch_reason: Optional[str] = None
    # Pure Pursuit telemetry
    pp_alpha: Optional[float] = None
    pp_lookahead_distance: Optional[float] = None
    pp_geometric_steering: Optional[float] = None
    pp_feedback_steering: Optional[float] = None
    pp_curve_local_floor_active: bool = False
    pp_curve_local_floor_m: Optional[float] = None
    pp_curve_local_lookahead_pre_floor: Optional[float] = None
    pp_curve_local_lookahead_post_floor: Optional[float] = None
    pp_curve_local_shorten_slew_active: bool = False
    pp_curve_local_shorten_delta_m: Optional[float] = None
    pp_ref_jump_clamped: bool = False
    pp_stale_hold_active: bool = False
    pp_steering_jerk_limited: bool = False
    pp_profile_reversal_detected: bool = False
    pp_profile_reversal_urgency: float = 0.0
    pp_profile_effective_taper: float = 0.0
    pp_effective_steering_rate: float = 0.0
    pp_pipeline_bypass_active: bool = False
    pp_speed_norm_scale: float = 1.0
    pp_map_ff_applied: float = 0.0
    # Lateral-error recovery term (replaces orchestrator post-limiter multiplier)
    lateral_error_recovery_term_applied_rad: float = 0.0
    lateral_error_recovery_smoothstep_weight: float = 0.0
    lateral_error_recovery_e_lat_source: str = 'none'
    lateral_error_recovery_shadow_mode: int = 0
    # MPC telemetry (zero-filled when MPC is inactive)
    mpc_feasible: bool = False
    mpc_solve_time_ms: float = 0.0
    mpc_e_lat: float = 0.0
    mpc_e_heading: float = 0.0
    mpc_kappa_ref: float = 0.0
    mpc_kappa_bias_correction: float = 0.0
    mpc_kappa_bias_ema: float = 0.0
    mpc_kappa_bias_guard_active: bool = False
    mpc_kappa_bias_guard_limit: float = 0.0
    mpc_leff_value: float = 0.0
    mpc_leff_theta: float = 0.0
    mpc_leff_P: float = 0.0
    mpc_leff_innovation: float = 0.0
    mpc_leff_update_count: int = 0
    mpc_tire_cf: float = 0.0
    mpc_tire_cr: float = 0.0
    mpc_tire_ekf_innovation: float = 0.0
    mpc_tire_ekf_P_trace: float = 0.0
    mpc_tire_slip_angle_front: float = 0.0
    mpc_tire_slip_angle_rear: float = 0.0
    mpc_tire_understeer_gradient: float = 0.0
    mpc_dynamic_model_active: int = 0
    mpc_tire_ekf_update_count: int = 0
    mpc_v_y_estimate: float = 0.0
    mpc_yaw_rate_estimate: float = 0.0
    mpc_yaw_rate_measurement: float = 0.0     # rad/s — actual measurement fed to EKF
    mpc_imu_yaw_rate_raw: float = 0.0         # rad/s — sign-corrected IMU before filtering
    mpc_unity_geometry_lf: float = 0.0        # m — Unity ground truth l_f
    mpc_unity_geometry_lr: float = 0.0        # m — Unity ground truth l_r
    mpc_unity_geometry_mass: float = 0.0      # kg — Unity Rigidbody mass
    mpc_unity_geometry_iz: float = 0.0        # kg*m² — Unity yaw inertia
    mpc_unity_geometry_active: bool = False    # True once Unity params applied
    mpc_kappa_active_curve_preserve_ratio: float = 0.0
    mpc_kappa_active_curve_preserve_active: bool = False
    mpc_kappa_active_curve_preserve_weight: float = 0.0
    mpc_kappa_active_mild_curve_authority_active: bool = False
    mpc_kappa_active_mild_curve_authority_weight: float = 0.0
    mpc_kappa_active_mild_curve_authority_ratio: float = 0.0
    mpc_kappa_active_mild_curve_authority_reason: str = ""
    mpc_kappa_active_mild_curve_authority_speed_weight: float = 0.0
    mpc_kappa_active_mild_curve_authority_curvature_weight: float = 0.0
    mpc_kappa_active_mild_curve_authority_gate_weight: float = 0.0
    mpc_fallback_active: bool = False
    mpc_consecutive_failures: int = 0
    mpc_gt_cross_track_m: float = 0.0
    mpc_gt_cross_track_at_car_m: float = 0.0
    mpc_gt_cross_track_road_frame_at_car_m: float = 0.0
    mpc_gt_cross_track_vehicle_frame_at_car_m: float = 0.0
    mpc_gt_cross_track_lookahead_m: float = 0.0
    mpc_gt_cross_track_source_code: str = ""
    mpc_gt_cross_track_control_source_code: str = ""
    mpc_e_lat_reference_source: str = ""
    mpc_e_lat_reference_divergence_m: float = 0.0
    # Frenet-frame MPC reference telemetry (Phase B of greedy-swimming-naur.md)
    mpc_e_lat_frenet_linearized_m: float = 0.0
    mpc_e_lat_shadow_delta_m: float = 0.0
    mpc_e_lat_frenet_shadow_mode: int = 0
    # ACC IDM-accel routing telemetry (acc-idm-accel-plumbing.md — shadow-mode first)
    reference_accel_mps2: float = 0.0             # What's fed to controller (NaN = disabled/shadow)
    reference_accel_source: str = ""              # "acc_idm" | "shadow" | "disabled"
    acc_idm_accel_shadow_mps2: float = 0.0        # IDM accel for logging (always populated when ACC enabled)
    mpc_gt_heading_error_rad: float = 0.0
    mpc_using_ground_truth: float = 0.0
    mpc_kappa_preview_used: bool = False
    mpc_kappa_preview_range: float = 0.0
    regime: int = 0                    # -1=Stanley, 0=PP, 1=LMPC, 2=NMPC
    regime_blend_weight: float = 1.0
    regime_lateral_accel_mps2: float = 0.0       # κ×v² used for regime decision (m/s²)
    regime_lateral_accel_threshold_mps2: float = 0.0  # effective threshold with hysteresis (m/s²)
    stanley_active: float = 0.0        # 1.0 when Stanley formula applied this frame
    stanley_heading_term: float = 0.0  # Heading error component of Stanley steering (rad)
    stanley_crosstrack_term: float = 0.0  # Cross-track arctan component of Stanley steering (rad)
    mpc_recovery_mode_suppressed: bool = False  # True when recovery ×1.2/×1.5 was skipped because MPC is active
    mpc_last_steering_pre_modify: float = 0.0   # Steering before orchestrator post-hoc modifications
    mpc_last_steering_actual: float = 0.0       # Actual steering sent (after all modifications)
    mpc_rate_limiter_active: bool = False       # True when MPC rate limiter clipped a frame-to-frame jump
    mpc_smith_raw_e_lat: float = 0.0           # Measured lateral error before Smith predictor
    mpc_smith_e_lat_predicted: float = 0.0     # Smith-predicted e_lat fed to MPC
    mpc_smith_e_heading_predicted: float = 0.0 # Smith-predicted e_heading fed to MPC
    mpc_delay_frames_used: int = 0             # Number of delay frames used in Smith predictor
    mpc_r_steer_rate_effective: float = 0.0   # Speed-scheduled r_steer_rate used by MPC QP
    grade_compensation_active: float = 0.0     # 1.0 when grade compensation is active (|grade| > 0.001)
    effective_max_accel: float = 0.0          # Grade-adjusted max acceleration (m/s²)
    # Silent e_lat dropout detection: e_lat≈0 but perception NOT flagged stale.
    # This catches cases where the perception layer returns lane center ≈ 0 without
    # triggering the stale hold flag — causing silent drift that isn't rate-limited.
    diag_silent_elat_dropout_active: bool = False  # True when |e_lat|<threshold, stale=0, on curve
    mpc_elat_ramp_active: bool = False             # True when dropout recovery rate-limiter clamped e_lat step
    # NMPC telemetry (zero-filled when NMPC is inactive or regime != NONLINEAR_MPC)
    nmpc_used: float = 0.0                # 1.0 when NMPC ran this frame, 0.0 when LMPC ran as fallback
    nmpc_feasible: bool = False           # NMPC solver returned a valid solution
    nmpc_solve_time_ms: float = 0.0       # SLSQP wall-clock solve time (ms)
    nmpc_cost: float = 0.0               # Optimal NLP cost (for solver health monitoring)
    nmpc_iterations: int = 0             # SLSQP iterations used (spike → near constraint boundary)
    nmpc_fallback_active: bool = False    # True when NMPC failed → LMPC fallback active
    nmpc_consecutive_failures: int = 0   # Consecutive NMPC solver failures
    # Inter-frame control extrapolation diagnostics
    interframe_active: float = 0.0           # 1.0 if inter-frame ran this cycle
    interframe_updates_this_cycle: int = 0   # Count of inter-frame updates since last camera
    interframe_total_count: int = 0          # Cumulative inter-frame update count
    interframe_last_e_lat: float = 0.0       # Last GT e_lat used by inter-frame
    interframe_last_e_heading: float = 0.0   # Last GT heading used by inter-frame
    interframe_dt_actual: float = 0.0        # Actual inter-frame dt (seconds)


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
    # Blind perception detection
    perception_blind: bool = False  # True when both lane positions are zero/None for N consecutive frames
    consecutive_no_detection_frames: int = 0  # Count of consecutive frames with no lane geometry
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
    curvature_preview: Optional[float] = None
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
    diag_preclip_mean_12_20m_lane_source_x: Optional[float] = None
    diag_preclip_mean_12_20m_distance_scale_delta_x: Optional[float] = None
    diag_preclip_mean_12_20m_camera_offset_delta_x: Optional[float] = None
    diag_preclip_abs_mean_12_20m_lane_source_x: Optional[float] = None
    diag_preclip_abs_mean_12_20m_distance_scale_delta_x: Optional[float] = None
    diag_preclip_abs_mean_12_20m_camera_offset_delta_x: Optional[float] = None
    diag_heading_zero_gate_active: Optional[float] = None
    diag_heading_from_history: Optional[float] = None
    diag_curvature_aware_alpha_reduction: Optional[float] = None
    diag_curvature_rate_limit_scale: Optional[float] = None
    diag_small_heading_gate_active: Optional[float] = None
    diag_multi_lookahead_active: Optional[float] = None
    diag_smoothing_jump_reject: Optional[float] = None
    diag_ref_x_rate_limit_active: Optional[float] = None
    diag_raw_ref_x: Optional[float] = None
    diag_smoothed_ref_x: Optional[float] = None
    diag_ref_x_suppression_abs: Optional[float] = None
    diag_raw_ref_heading: Optional[float] = None
    diag_smoothed_ref_heading: Optional[float] = None
    diag_heading_suppression_abs: Optional[float] = None
    diag_smoothing_alpha: Optional[float] = None
    diag_smoothing_alpha_x: Optional[float] = None
    diag_multi_lookahead_heading_base: Optional[float] = None
    diag_multi_lookahead_heading_far: Optional[float] = None
    diag_multi_lookahead_heading_blended: Optional[float] = None
    diag_multi_lookahead_blend_alpha: Optional[float] = None
    diag_dynamic_effective_horizon_m: Optional[float] = None
    diag_dynamic_effective_horizon_base_m: Optional[float] = None
    diag_dynamic_effective_horizon_min_m: Optional[float] = None
    diag_dynamic_effective_horizon_max_m: Optional[float] = None
    diag_dynamic_effective_horizon_speed_scale: Optional[float] = None
    diag_dynamic_effective_horizon_curvature_scale: Optional[float] = None
    diag_dynamic_effective_horizon_confidence_scale: Optional[float] = None
    diag_dynamic_effective_horizon_final_scale: Optional[float] = None
    diag_dynamic_effective_horizon_speed_mps: Optional[float] = None
    diag_dynamic_effective_horizon_curvature_abs: Optional[float] = None
    diag_dynamic_effective_horizon_confidence_used: Optional[float] = None
    diag_dynamic_effective_horizon_limiter_code: Optional[float] = None
    diag_dynamic_effective_horizon_applied: Optional[float] = None
    diag_speed_horizon_guardrail_active: Optional[float] = None
    diag_speed_horizon_guardrail_margin_m: Optional[float] = None
    diag_speed_horizon_guardrail_horizon_m: Optional[float] = None
    diag_speed_horizon_guardrail_time_headway_s: Optional[float] = None
    diag_speed_horizon_guardrail_margin_buffer_m: Optional[float] = None
    diag_speed_horizon_guardrail_allowed_speed_mps: Optional[float] = None
    diag_speed_horizon_guardrail_target_speed_before_mps: Optional[float] = None
    diag_speed_horizon_guardrail_target_speed_after_mps: Optional[float] = None
    diag_preclip_abs_mean_0_8m: Optional[float] = None
    diag_preclip_abs_mean_8_12m: Optional[float] = None
    diag_preclip_abs_mean_12_20m: Optional[float] = None
    diag_postclip_x0: Optional[float] = None
    diag_postclip_x1: Optional[float] = None
    diag_postclip_x2: Optional[float] = None
    diag_postclip_abs_mean_0_8m: Optional[float] = None
    diag_postclip_abs_mean_8_12m: Optional[float] = None
    diag_postclip_abs_mean_12_20m: Optional[float] = None
    diag_postclip_near_clip_frac_12_20m: Optional[float] = None
    diag_first_segment_y0_gt_y1_pre: Optional[float] = None
    diag_first_segment_y0_gt_y1_post: Optional[float] = None
    diag_inversion_introduced_after_conversion: Optional[float] = None
    diag_far_band_contribution_limited_active: Optional[float] = None
    diag_far_band_contribution_limit_start_m: Optional[float] = None
    diag_far_band_contribution_limit_gain: Optional[float] = None
    diag_far_band_contribution_scale_mean_12_20m: Optional[float] = None
    diag_far_band_contribution_limited_frac_12_20m: Optional[float] = None
    # Lane-midpoint curve-contamination clamp diagnostics
    lane_midpoint_clamp_active: Optional[float] = None       # 1.0 when clamp fired this frame
    lane_midpoint_clamp_dist_m: Optional[float] = None       # clamped coord_conversion_distance (m)
    lane_midpoint_clamp_kappa_preview: Optional[float] = None  # preview curvature used for clamp


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
    chassis_ground_min_clearance_m: Optional[float] = None
    chassis_ground_effective_min_clearance_m: Optional[float] = None
    chassis_ground_clearance_m: Optional[float] = None
    chassis_ground_penetration_m: Optional[float] = None
    chassis_ground_contact: bool = False
    wheel_grounded_count: Optional[int] = None
    wheel_colliders_ready: bool = False
    force_fallback_active: bool = False
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
