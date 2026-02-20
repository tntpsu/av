"""
PID controller for vehicle control.
Handles both lateral (steering) and longitudinal (throttle/brake) control.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Union, Any
from pathlib import Path
import math
import yaml


class PIDController:
    """
    PID controller with integral windup protection.
    """
    
    def __init__(self, kp: float, ki: float, kd: float, 
                 integral_limit: Optional[float] = None, output_limit: Optional[Tuple[float, float]] = None):
        """
        Initialize PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            integral_limit: Limit for integral term (anti-windup)
            output_limit: (min, max) output limits
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None
    
    def update(self, error: float, dt: float) -> float:
        """
        Update PID controller.
        
        Args:
            error: Current error
            dt: Time step
        
        Returns:
            Control output
        """
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += error * dt
        if self.integral_limit is not None:
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral
        
        # Derivative term
        if self.prev_time is not None and dt > 0:
            d_error = (error - self.prev_error) / dt
            d_term = self.kd * d_error
        else:
            d_term = 0.0
        
        # Compute output
        output = p_term + i_term + d_term
        
        # Apply output limits
        if self.output_limit is not None:
            output = np.clip(output, self.output_limit[0], self.output_limit[1])
        
        # Update state
        self.prev_error = error
        self.prev_time = (self.prev_time or 0.0) + dt
        
        return output
    
    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None


class LateralController:
    """
    Lateral control (steering) using PID controller.
    """
    
    def __init__(self, kp: float = 0.3, ki: float = 0.0, kd: float = 0.1,
                 lookahead_distance: float = 10.0, max_steering: float = 0.5,
                 deadband: float = 0.02, heading_weight: float = 0.5,
                 lateral_weight: float = 0.5, error_clip: float = np.pi / 4,
                 integral_limit: float = 0.3, steering_smoothing_alpha: float = 0.7,
                 curve_feedforward_gain: float = 1.0, curve_feedforward_threshold: float = 0.02,
                 curve_feedforward_gain_min: float = 1.0, curve_feedforward_gain_max: float = 1.0,
                 curve_feedforward_curvature_min: float = 0.005, curve_feedforward_curvature_max: float = 0.03,
                 curve_feedforward_curvature_clamp: float = 0.03,
                 curve_feedforward_bins: Optional[list] = None,
                 curvature_scale_factor: float = 1.0,  # Scale curvature for feedforward (to match GT)
                 curvature_scale_threshold: float = 0.0005,  # Only scale when curvature > this (avoid noise on straights)
                 curvature_smoothing_alpha: float = 0.7,
                 curvature_transition_threshold: float = 0.01,
                 curvature_transition_alpha: float = 0.3,
                 straight_curvature_threshold: float = 0.01,
                 curve_upcoming_enter_threshold: float = 0.012,
                 curve_upcoming_exit_threshold: float = 0.009,
                 curve_upcoming_on_frames: int = 2,
                 curve_upcoming_off_frames: int = 2,
                 curve_phase_use_distance_track: bool = False,
                 curve_phase_track_name: Optional[str] = None,
                 curve_at_car_distance_min_m: float = 0.0,
                 road_curve_enter_threshold: Optional[float] = None,
                 road_curve_exit_threshold: Optional[float] = None,
                 road_straight_hold_invalid_frames: int = 6,
                 steering_rate_curvature_min: float = 0.005,
                 steering_rate_curvature_max: float = 0.03,
                 steering_rate_scale_min: float = 0.5,
                 curve_rate_floor_moderate_error: float = 0.20,
                 curve_rate_floor_large_error: float = 0.24,
                 straight_sign_flip_error_threshold: float = 0.02,
                 straight_sign_flip_rate: float = 0.2,
                 straight_sign_flip_frames: int = 6,
                 steering_jerk_limit: float = 0.0,
                 steering_jerk_curve_scale_max: float = 1.0,
                 steering_jerk_curve_min: float = 0.003,
                 steering_jerk_curve_max: float = 0.015,
                 curve_entry_assist_enabled: bool = False,
                 curve_entry_assist_error_min: float = 0.30,
                 curve_entry_assist_heading_error_min: float = 0.10,
                 curve_entry_assist_curvature_min: float = 0.010,
                 curve_entry_assist_rate_boost: float = 1.15,
                 curve_entry_assist_jerk_boost: float = 1.10,
                 curve_entry_assist_max_frames: int = 18,
                 curve_entry_assist_watchdog_rate_delta_max: float = 0.22,
                 curve_entry_assist_rearm_frames: int = 20,
                 curve_entry_schedule_enabled: bool = False,
                 curve_entry_schedule_frames: int = 18,
                 curve_entry_schedule_min_rate: float = 0.22,
                 curve_entry_schedule_min_jerk: float = 0.14,
                 curve_entry_schedule_min_hold_frames: int = 12,
                 curve_entry_schedule_min_curve_progress_ratio: float = 0.20,
                 curve_entry_schedule_fallback_only_when_dynamic: bool = False,
                 curve_entry_schedule_fallback_deficit_frames: int = 6,
                 curve_entry_schedule_fallback_rate_deficit_min: float = 0.03,
                 curve_entry_schedule_fallback_rearm_cooldown_frames: int = 18,
                 curve_entry_schedule_handoff_transfer_ratio: float = 0.65,
                 curve_entry_schedule_handoff_error_fall: float = 0.03,
                 dynamic_curve_authority_enabled: bool = True,
                 dynamic_curve_rate_deficit_deadband: float = 0.01,
                 dynamic_curve_rate_boost_gain: float = 1.0,
                 dynamic_curve_rate_boost_max: float = 0.30,
                 dynamic_curve_jerk_boost_gain: float = 4.0,
                 dynamic_curve_jerk_boost_max_factor: float = 3.5,
                 dynamic_curve_hard_clip_boost_gain: float = 1.0,
                 dynamic_curve_hard_clip_boost_max: float = 0.12,
                 dynamic_curve_entry_governor_enabled: bool = True,
                 dynamic_curve_entry_governor_gain: float = 1.2,
                 dynamic_curve_entry_governor_max_scale: float = 1.8,
                 dynamic_curve_entry_governor_stale_floor_scale: float = 1.15,
                 dynamic_curve_entry_governor_exclusive_mode: bool = True,
                 dynamic_curve_entry_governor_anticipatory_enabled: bool = True,
                 dynamic_curve_entry_governor_upcoming_phase_weight: float = 0.55,
                 dynamic_curve_authority_precurve_enabled: bool = True,
                 dynamic_curve_authority_precurve_scale: float = 0.8,
                 dynamic_curve_single_owner_mode: bool = False,
                 dynamic_curve_single_owner_min_rate: float = 0.22,
                 dynamic_curve_single_owner_min_jerk: float = 0.6,
                 dynamic_curve_comfort_lat_accel_comfort_max_g: float = 0.18,
                 dynamic_curve_comfort_lat_accel_peak_max_g: float = 0.25,
                 dynamic_curve_comfort_lat_jerk_comfort_max_gps: float = 0.30,
                 dynamic_curve_lat_jerk_smoothing_alpha: float = 0.25,
                 dynamic_curve_lat_jerk_soft_start_ratio: float = 0.60,
                 dynamic_curve_lat_jerk_soft_floor_scale: float = 0.35,
                 dynamic_curve_speed_low_mps: float = 4.0,
                 dynamic_curve_speed_high_mps: float = 10.0,
                 dynamic_curve_speed_boost_max_scale: float = 1.4,
                 turn_feasibility_governor_enabled: bool = True,
                 turn_feasibility_curvature_min: float = 0.002,
                 turn_feasibility_guardband_g: float = 0.015,
                 turn_feasibility_use_peak_bound: bool = True,
                 curve_unwind_policy_enabled: bool = False,
                 curve_unwind_frames: int = 12,
                 curve_unwind_rate_scale_start: float = 1.0,
                 curve_unwind_rate_scale_end: float = 0.8,
                 curve_unwind_jerk_scale_start: float = 1.0,
                 curve_unwind_jerk_scale_end: float = 0.7,
                 curve_unwind_integral_decay: float = 0.85,
                 curve_commit_mode_enabled: bool = False,
                 curve_commit_mode_max_frames: int = 20,
                 curve_commit_mode_min_rate: float = 0.22,
                 curve_commit_mode_min_jerk: float = 0.14,
                 curve_commit_mode_transfer_ratio_target: float = 0.72,
                 curve_commit_mode_error_fall: float = 0.03,
                 curve_commit_mode_exit_consecutive_frames: int = 4,
                 curve_commit_mode_retrigger_on_dynamic_deficit: bool = True,
                 curve_commit_mode_dynamic_deficit_frames: int = 8,
                 curve_commit_mode_dynamic_deficit_min: float = 0.03,
                 curve_commit_mode_retrigger_cooldown_frames: int = 12,
                 speed_gain_min_speed: float = 4.0,
                 speed_gain_max_speed: float = 10.0,
                 speed_gain_min: float = 1.0,
                 speed_gain_max: float = 1.2,
                 speed_gain_curvature_min: float = 0.002,
                 speed_gain_curvature_max: float = 0.015,
                 control_mode: str = "pid",
                 stanley_k: float = 1.0,
                 stanley_soft_speed: float = 2.0,
                 stanley_heading_weight: float = 1.0,
                 pp_feedback_gain: float = 0.15,
                 pp_min_lookahead: float = 0.5,
                 pp_ref_jump_clamp: float = 0.5,
                 pp_stale_decay: float = 0.98,
                 feedback_gain_min: float = 1.0,
                 feedback_gain_max: float = 1.2,
                 feedback_gain_curvature_min: float = 0.002,
                 feedback_gain_curvature_max: float = 0.015,
                 curvature_stale_hold_seconds: float = 0.30,
                 curvature_stale_hold_min_abs: float = 0.0005,
                 base_error_smoothing_alpha: float = 0.7,
                 heading_error_smoothing_alpha: float = 0.45,
                 straight_window_frames: int = 60,
                 straight_oscillation_high: float = 0.20,
                 straight_oscillation_low: float = 0.05):
        """
        Initialize lateral controller.
        
        Args:
            kp: Proportional gain (reduced for smoother control)
            ki: Integral gain
            kd: Derivative gain (reduced for less oscillation)
            lookahead_distance: Lookahead distance for reference point (meters)
            max_steering: Maximum steering angle (-1.0 to 1.0, reduced for stability)
        """
        self.lookahead_distance = lookahead_distance
        self.max_steering = max_steering
        self.base_deadband = deadband
        self.deadband = deadband
        self.heading_weight = heading_weight
        self.lateral_weight = lateral_weight
        self.error_clip = error_clip
        self.steering_smoothing_alpha = steering_smoothing_alpha
        self.curve_feedforward_gain = curve_feedforward_gain
        self.curve_feedforward_threshold = curve_feedforward_threshold
        self.curve_feedforward_gain_min = curve_feedforward_gain_min
        self.curve_feedforward_gain_max = curve_feedforward_gain_max
        self.curve_feedforward_curvature_min = curve_feedforward_curvature_min
        self.curve_feedforward_curvature_max = curve_feedforward_curvature_max
        self.curve_feedforward_curvature_clamp = curve_feedforward_curvature_clamp
        self.curve_feedforward_bins = curve_feedforward_bins or []
        self.curvature_scale_factor = curvature_scale_factor
        self.curvature_scale_threshold = curvature_scale_threshold
        self.curvature_smoothing_alpha = curvature_smoothing_alpha
        self.curvature_transition_threshold = curvature_transition_threshold
        self.curvature_transition_alpha = curvature_transition_alpha
        self.steering_rate_curvature_min = steering_rate_curvature_min
        self.steering_rate_curvature_max = steering_rate_curvature_max
        self.steering_rate_scale_min = steering_rate_scale_min
        self.curve_rate_floor_moderate_error = max(0.0, float(curve_rate_floor_moderate_error))
        self.curve_rate_floor_large_error = max(0.0, float(curve_rate_floor_large_error))
        self.straight_sign_flip_error_threshold = straight_sign_flip_error_threshold
        self.straight_sign_flip_rate = straight_sign_flip_rate
        self.straight_sign_flip_frames = straight_sign_flip_frames
        self.steering_jerk_limit = steering_jerk_limit
        self.steering_jerk_curve_scale_max = max(1.0, float(steering_jerk_curve_scale_max))
        self.steering_jerk_curve_min = max(1e-6, float(steering_jerk_curve_min))
        self.steering_jerk_curve_max = max(self.steering_jerk_curve_min, float(steering_jerk_curve_max))
        self.curve_entry_assist_enabled = bool(curve_entry_assist_enabled)
        self.curve_entry_assist_error_min = max(0.0, float(curve_entry_assist_error_min))
        self.curve_entry_assist_heading_error_min = max(0.0, float(curve_entry_assist_heading_error_min))
        self.curve_entry_assist_curvature_min = max(0.0, float(curve_entry_assist_curvature_min))
        self.curve_entry_assist_rate_boost = max(1.0, float(curve_entry_assist_rate_boost))
        self.curve_entry_assist_jerk_boost = max(1.0, float(curve_entry_assist_jerk_boost))
        self.curve_entry_assist_max_frames = max(0, int(curve_entry_assist_max_frames))
        self.curve_entry_assist_watchdog_rate_delta_max = max(
            0.05, float(curve_entry_assist_watchdog_rate_delta_max)
        )
        self.curve_entry_assist_rearm_frames = max(0, int(curve_entry_assist_rearm_frames))
        self.curve_entry_schedule_enabled = bool(curve_entry_schedule_enabled)
        self.curve_entry_schedule_frames = max(0, int(curve_entry_schedule_frames))
        self.curve_entry_schedule_min_rate = max(0.0, float(curve_entry_schedule_min_rate))
        self.curve_entry_schedule_min_jerk = max(0.0, float(curve_entry_schedule_min_jerk))
        self.curve_entry_schedule_min_hold_frames = max(0, int(curve_entry_schedule_min_hold_frames))
        self.curve_entry_schedule_min_curve_progress_ratio = float(
            np.clip(curve_entry_schedule_min_curve_progress_ratio, 0.0, 1.0)
        )
        self.curve_entry_schedule_fallback_only_when_dynamic = bool(
            curve_entry_schedule_fallback_only_when_dynamic
        )
        self.curve_entry_schedule_fallback_deficit_frames = max(
            1, int(curve_entry_schedule_fallback_deficit_frames)
        )
        self.curve_entry_schedule_fallback_rate_deficit_min = max(
            0.0, float(curve_entry_schedule_fallback_rate_deficit_min)
        )
        self.curve_entry_schedule_fallback_rearm_cooldown_frames = max(
            0, int(curve_entry_schedule_fallback_rearm_cooldown_frames)
        )
        self.curve_entry_schedule_handoff_transfer_ratio = float(
            np.clip(curve_entry_schedule_handoff_transfer_ratio, 0.0, 1.0)
        )
        self.curve_entry_schedule_handoff_error_fall = max(
            0.0, float(curve_entry_schedule_handoff_error_fall)
        )
        self.dynamic_curve_authority_enabled = bool(dynamic_curve_authority_enabled)
        self.dynamic_curve_rate_deficit_deadband = max(
            0.0, float(dynamic_curve_rate_deficit_deadband)
        )
        self.dynamic_curve_rate_boost_gain = max(0.0, float(dynamic_curve_rate_boost_gain))
        self.dynamic_curve_rate_boost_max = max(0.0, float(dynamic_curve_rate_boost_max))
        self.dynamic_curve_jerk_boost_gain = max(0.0, float(dynamic_curve_jerk_boost_gain))
        self.dynamic_curve_jerk_boost_max_factor = max(
            1.0, float(dynamic_curve_jerk_boost_max_factor)
        )
        self.dynamic_curve_hard_clip_boost_gain = max(
            0.0, float(dynamic_curve_hard_clip_boost_gain)
        )
        self.dynamic_curve_hard_clip_boost_max = max(
            0.0, float(dynamic_curve_hard_clip_boost_max)
        )
        self.dynamic_curve_entry_governor_enabled = bool(dynamic_curve_entry_governor_enabled)
        self.dynamic_curve_entry_governor_gain = max(
            0.0, float(dynamic_curve_entry_governor_gain)
        )
        self.dynamic_curve_entry_governor_max_scale = max(
            1.0, float(dynamic_curve_entry_governor_max_scale)
        )
        self.dynamic_curve_entry_governor_stale_floor_scale = float(
            np.clip(dynamic_curve_entry_governor_stale_floor_scale, 1.0, 2.0)
        )
        self.dynamic_curve_entry_governor_exclusive_mode = bool(
            dynamic_curve_entry_governor_exclusive_mode
        )
        self.dynamic_curve_entry_governor_anticipatory_enabled = bool(
            dynamic_curve_entry_governor_anticipatory_enabled
        )
        self.dynamic_curve_entry_governor_upcoming_phase_weight = float(
            np.clip(dynamic_curve_entry_governor_upcoming_phase_weight, 0.0, 1.0)
        )
        self.dynamic_curve_authority_precurve_enabled = bool(
            dynamic_curve_authority_precurve_enabled
        )
        self.dynamic_curve_authority_precurve_scale = float(
            np.clip(dynamic_curve_authority_precurve_scale, 0.0, 1.0)
        )
        self.dynamic_curve_single_owner_mode = bool(dynamic_curve_single_owner_mode)
        self.dynamic_curve_single_owner_min_rate = max(
            0.0, float(dynamic_curve_single_owner_min_rate)
        )
        self.dynamic_curve_single_owner_min_jerk = max(
            0.0, float(dynamic_curve_single_owner_min_jerk)
        )
        self.dynamic_curve_comfort_lat_accel_comfort_max_g = max(
            1e-3, float(dynamic_curve_comfort_lat_accel_comfort_max_g)
        )
        self.dynamic_curve_comfort_lat_accel_peak_max_g = max(
            self.dynamic_curve_comfort_lat_accel_comfort_max_g,
            float(dynamic_curve_comfort_lat_accel_peak_max_g),
        )
        self.dynamic_curve_comfort_lat_jerk_comfort_max_gps = max(
            1e-3, float(dynamic_curve_comfort_lat_jerk_comfort_max_gps)
        )
        self.dynamic_curve_lat_jerk_smoothing_alpha = float(
            np.clip(dynamic_curve_lat_jerk_smoothing_alpha, 0.0, 1.0)
        )
        self.dynamic_curve_lat_jerk_soft_start_ratio = float(
            np.clip(dynamic_curve_lat_jerk_soft_start_ratio, 0.0, 1.0)
        )
        self.dynamic_curve_lat_jerk_soft_floor_scale = float(
            np.clip(dynamic_curve_lat_jerk_soft_floor_scale, 0.0, 1.0)
        )
        self.dynamic_curve_speed_low_mps = max(0.0, float(dynamic_curve_speed_low_mps))
        self.dynamic_curve_speed_high_mps = max(
            self.dynamic_curve_speed_low_mps + 1e-3,
            float(dynamic_curve_speed_high_mps),
        )
        self.dynamic_curve_speed_boost_max_scale = max(
            1.0, float(dynamic_curve_speed_boost_max_scale)
        )
        self.turn_feasibility_governor_enabled = bool(turn_feasibility_governor_enabled)
        self.turn_feasibility_curvature_min = max(1e-6, float(turn_feasibility_curvature_min))
        self.turn_feasibility_guardband_g = max(0.0, float(turn_feasibility_guardband_g))
        self.turn_feasibility_use_peak_bound = bool(turn_feasibility_use_peak_bound)
        self.curve_unwind_policy_enabled = bool(curve_unwind_policy_enabled)
        self.curve_unwind_frames = max(0, int(curve_unwind_frames))
        self.curve_unwind_rate_scale_start = float(
            np.clip(curve_unwind_rate_scale_start, 0.0, 1.5)
        )
        self.curve_unwind_rate_scale_end = float(
            np.clip(curve_unwind_rate_scale_end, 0.0, 1.5)
        )
        self.curve_unwind_jerk_scale_start = float(
            np.clip(curve_unwind_jerk_scale_start, 0.0, 1.5)
        )
        self.curve_unwind_jerk_scale_end = float(
            np.clip(curve_unwind_jerk_scale_end, 0.0, 1.5)
        )
        self.curve_unwind_integral_decay = float(
            np.clip(curve_unwind_integral_decay, 0.0, 1.0)
        )
        self.curve_commit_mode_enabled = bool(curve_commit_mode_enabled)
        self.curve_commit_mode_max_frames = max(0, int(curve_commit_mode_max_frames))
        self.curve_commit_mode_min_rate = max(0.0, float(curve_commit_mode_min_rate))
        self.curve_commit_mode_min_jerk = max(0.0, float(curve_commit_mode_min_jerk))
        self.curve_commit_mode_transfer_ratio_target = float(
            np.clip(curve_commit_mode_transfer_ratio_target, 0.0, 1.0)
        )
        self.curve_commit_mode_error_fall = max(0.0, float(curve_commit_mode_error_fall))
        self.curve_commit_mode_exit_consecutive_frames = max(
            1, int(curve_commit_mode_exit_consecutive_frames)
        )
        self.curve_commit_mode_retrigger_on_dynamic_deficit = bool(
            curve_commit_mode_retrigger_on_dynamic_deficit
        )
        self.curve_commit_mode_dynamic_deficit_frames = max(
            1, int(curve_commit_mode_dynamic_deficit_frames)
        )
        self.curve_commit_mode_dynamic_deficit_min = max(
            0.0, float(curve_commit_mode_dynamic_deficit_min)
        )
        self.curve_commit_mode_retrigger_cooldown_frames = max(
            0, int(curve_commit_mode_retrigger_cooldown_frames)
        )
        self.speed_gain_min_speed = speed_gain_min_speed
        self.speed_gain_max_speed = speed_gain_max_speed
        self.speed_gain_min = speed_gain_min
        self.speed_gain_max = speed_gain_max
        self.speed_gain_curvature_min = speed_gain_curvature_min
        self.speed_gain_curvature_max = speed_gain_curvature_max
        self.control_mode = control_mode
        self.stanley_k = stanley_k
        self.stanley_soft_speed = stanley_soft_speed
        self.stanley_heading_weight = stanley_heading_weight
        self.pp_feedback_gain = max(0.0, float(pp_feedback_gain))
        self.pp_min_lookahead = max(0.1, float(pp_min_lookahead))
        self.pp_ref_jump_clamp = max(0.1, float(pp_ref_jump_clamp))
        self.pp_stale_decay = float(np.clip(pp_stale_decay, 0.5, 1.0))
        self._pp_last_ref_x = 0.0
        self._pp_last_valid_steering = 0.0
        self._pp_stale_frames = 0
        self.smoothed_path_curvature = 0.0
        self.feedback_gain_min = feedback_gain_min
        self.feedback_gain_max = feedback_gain_max
        self.feedback_gain_curvature_min = feedback_gain_curvature_min
        self.feedback_gain_curvature_max = feedback_gain_curvature_max
        self.curvature_stale_hold_seconds = max(0.0, float(curvature_stale_hold_seconds))
        self.curvature_stale_hold_min_abs = max(0.0, float(curvature_stale_hold_min_abs))
        self._last_valid_path_curvature = 0.0
        self._curvature_hold_frames_remaining = 0
        self._was_using_stale_perception = False
        self.smoothed_steering = None
        self.pid = PIDController(
            kp=kp,
            ki=ki,
            kd=kd,
            integral_limit=integral_limit,  # Use from parameter (can be set from config)
            output_limit=(-max_steering, max_steering)  # Reduced steering range for stability
        )
        # Integral reset tracking
        self.small_error_count = 0
        self.last_error_sign = None
        self.recent_large_error = False  # Track if we recently had large error (on curve)
        self.large_error_frames = 0  # Count consecutive frames with large error
        
        # NEW: Error smoothing to reduce sensitivity to perception noise
        # Smooth lateral + heading errors before feeding to PID to prevent control oscillations from perception instability
        self.smoothed_lateral_error = None
        self.base_error_smoothing_alpha = float(
            np.clip(base_error_smoothing_alpha, 0.0, 1.0)
        )
        self.error_smoothing_alpha = self.base_error_smoothing_alpha
        self.smoothed_heading_error = None
        self.heading_error_smoothing_alpha = float(
            np.clip(heading_error_smoothing_alpha, 0.0, 1.0)
        )

        # Straight-away adaptive tuning (deadband + smoothing)
        self.straight_curvature_threshold = straight_curvature_threshold
        self.curve_upcoming_enter_threshold = max(0.0, float(curve_upcoming_enter_threshold))
        self.curve_upcoming_exit_threshold = max(0.0, float(curve_upcoming_exit_threshold))
        if self.curve_upcoming_exit_threshold > self.curve_upcoming_enter_threshold:
            self.curve_upcoming_exit_threshold = self.curve_upcoming_enter_threshold
        self.curve_upcoming_on_frames = max(1, int(curve_upcoming_on_frames))
        self.curve_upcoming_off_frames = max(1, int(curve_upcoming_off_frames))
        self.curve_phase_use_distance_track = bool(curve_phase_use_distance_track)
        self.curve_phase_track_name = str(curve_phase_track_name or "").strip()
        self.curve_at_car_distance_min_m = max(0.0, float(curve_at_car_distance_min_m))
        self._track_curve_windows: list[tuple[float, float]] = []
        self._track_total_length_m: Optional[float] = None
        if self.curve_phase_use_distance_track and self.curve_phase_track_name:
            self._load_track_curve_windows(self.curve_phase_track_name)
        self._curve_upcoming_state = False
        self._curve_upcoming_on_counter = 0
        self._curve_upcoming_off_counter = 0
        self._curve_at_car_state = False
        self._curve_at_car_distance_remaining = None
        road_enter_threshold = (
            float(straight_curvature_threshold)
            if road_curve_enter_threshold is None
            else float(road_curve_enter_threshold)
        )
        road_exit_threshold = (
            float(straight_curvature_threshold)
            if road_curve_exit_threshold is None
            else float(road_curve_exit_threshold)
        )
        road_enter_threshold = max(0.0, road_enter_threshold)
        road_exit_threshold = max(0.0, road_exit_threshold)
        if road_exit_threshold > road_enter_threshold:
            road_exit_threshold = road_enter_threshold
        self.road_curve_enter_threshold = road_enter_threshold
        self.road_curve_exit_threshold = road_exit_threshold
        self.road_straight_hold_invalid_frames = max(0, int(road_straight_hold_invalid_frames))
        self._road_straight_state = True
        self._road_straight_invalid_frames_remaining = 0
        self.straight_window = max(10, int(straight_window_frames))
        self.straight_oscillation_high = max(0.0, float(straight_oscillation_high))
        self.straight_oscillation_low = max(0.0, float(straight_oscillation_low))
        if self.straight_oscillation_low > self.straight_oscillation_high:
            self.straight_oscillation_low = self.straight_oscillation_high
        self.deadband_min = 0.01
        self.deadband_max = 0.08
        self.smoothing_min = 0.60
        self.smoothing_max = 0.90
        self.base_straight_steering_deadband = 0.022
        self.straight_steering_deadband = self.base_straight_steering_deadband
        self.straight_steering_deadband_max = 0.06
        self._straight_frames = 0
        self._straight_sign_changes = 0
        self._last_straight_sign = 0
        self.straight_oscillation_rate = 0.0
        self._curve_entry_assist_frames_remaining = 0
        self._curve_entry_assist_rearm_remaining = 0
        self._curve_entry_schedule_frames_remaining = 0
        self._curve_entry_schedule_elapsed_frames = 0
        self._curve_entry_schedule_fallback_fired_for_curve = False
        self._curve_entry_schedule_fallback_last_curve_index = None
        self._curve_entry_schedule_fallback_cooldown_remaining = 0
        self._dynamic_authority_deficit_streak = 0
        self._dynamic_commit_deficit_streak = 0
        self._last_lateral_accel_est_g = 0.0
        self._last_lateral_accel_est_initialized = False
        self._smoothed_lateral_jerk_est_gps = 0.0
        self._curve_commit_mode_frames_remaining = 0
        self._curve_commit_retrigger_cooldown_remaining = 0
        self._curve_commit_handoff_streak = 0
        self._curve_unwind_frames_remaining = 0
        self._prev_is_straight = True
        self._prev_entry_phase_active = False
        self._prev_curve_upcoming = False
        self._prev_curve_at_car = False
        self._last_error_magnitude = 0.0
    
    def compute_steering(self, current_heading: float, reference_point: dict,
                        vehicle_position: Optional[np.ndarray] = None,
                        current_speed: Optional[float] = None,
                        road_center_reference_t: Optional[float] = None,
                        dt: Optional[float] = None,
                        return_metadata: bool = False,
                        using_stale_perception: bool = False) -> Union[float, Dict[str, Any]]:
        """
        Compute steering command.
        
        Args:
            current_heading: Current vehicle heading (radians)
            reference_point: Reference point from trajectory (x, y, heading)
            vehicle_position: Current vehicle position [x, y] (optional)
            return_metadata: If True, return dict with steering and internal state
        
        Returns:
            Steering command (-1.0 to 1.0) if return_metadata=False
            Dict with steering, errors, and PID state if return_metadata=True
        """
        if vehicle_position is None:
            vehicle_position = np.array([0.0, 0.0])
        
        # Get reference point data
        ref_x = reference_point['x']  # Already in vehicle frame (meters, lateral offset)
        ref_y = reference_point['y']  # Already in vehicle frame (meters, forward distance)
        desired_heading = reference_point['heading']  # Path heading (used for sign + fallback)
        # desired_heading is the path heading in vehicle frame (used for sign + fallback)
        
        # Compute heading error
        # CRITICAL FIX: On curves, ref_heading represents path curvature, not correction needed
        # This causes conflicts: path curves right (ref_heading > 0) but car heading more right
        # → heading_error < 0 (suggests steer left) conflicts with lateral_error > 0 (steer right)
        #
        # Solution: Calculate heading_error as the heading needed to point TOWARD the reference point
        # This represents the correction needed, not the path curvature
        # Heading needed to point toward reference point
        # This is the correction needed, not the path curvature
        heading_to_ref = np.arctan2(ref_x, ref_y) if ref_y > 0.1 else 0.0
        
        # Use path curvature only for feedforward, not for error calculation
        # For error calculation, use heading_to_ref (correction needed)
        # But blend with ref_heading for smooth curves
        raw_ref_curvature_value = reference_point.get('curvature')
        raw_ref_curvature_source = str(reference_point.get('curvature_source', '') or '')
        road_curvature_valid = bool(
            raw_ref_curvature_value is not None
            and np.isfinite(float(raw_ref_curvature_value))
            and raw_ref_curvature_source.lower() not in {"missing", "placeholder"}
        )
        raw_ref_curvature = float(raw_ref_curvature_value) if road_curvature_valid else 0.0
        path_curvature = raw_ref_curvature
        if path_curvature != 0.0:
            heading_sign = np.sign(desired_heading)
            if heading_sign == 0.0:
                heading_sign = 1.0
            path_curvature = abs(path_curvature) * heading_sign
            if self.curve_feedforward_curvature_clamp and self.curve_feedforward_curvature_clamp > 0.0:
                path_curvature = np.clip(
                    path_curvature,
                    -self.curve_feedforward_curvature_clamp,
                    self.curve_feedforward_curvature_clamp
                )
        dt = float(dt) if dt is not None and np.isfinite(float(dt)) else 0.033
        dt = float(np.clip(dt, 1e-3, 0.25))
        hold_frames = int(round(self.curvature_stale_hold_seconds / dt)) if dt > 0.0 else 0
        if using_stale_perception and not self._was_using_stale_perception:
            self._curvature_hold_frames_remaining = hold_frames
        elif not using_stale_perception:
            self._curvature_hold_frames_remaining = hold_frames

        if abs(path_curvature) >= self.curvature_stale_hold_min_abs and not using_stale_perception:
            self._last_valid_path_curvature = path_curvature

        curvature_hold_active = False
        if (
            using_stale_perception
            and self._curvature_hold_frames_remaining > 0
            and abs(path_curvature) < self.curvature_stale_hold_min_abs
            and abs(self._last_valid_path_curvature) >= self.curvature_stale_hold_min_abs
        ):
            path_curvature = self._last_valid_path_curvature
            curvature_hold_active = True
            self._curvature_hold_frames_remaining -= 1
        elif using_stale_perception:
            if abs(path_curvature) >= self.curvature_stale_hold_min_abs:
                self._last_valid_path_curvature = path_curvature
            self._curvature_hold_frames_remaining = max(0, self._curvature_hold_frames_remaining - 1)
        self._was_using_stale_perception = using_stale_perception

        curve_metric = path_curvature if path_curvature != 0.0 else desired_heading
        curve_metric_source = "path_curvature" if path_curvature != 0.0 else "desired_heading_fallback"
        curve_metric_abs = abs(curve_metric)
        
        # On curves, we need both:
        # 1. Heading to point toward reference (correction)
        # 2. Path curvature (for feedforward)
        # CRITICAL FIX: On curves, use heading_to_ref directly (don't subtract current_heading)
        # Subtracting current_heading causes conflicts: car heading more right → negative heading_error
        # But lateral_error is positive → conflict!
        # Solution: heading_error = heading_to_ref (correction needed to point toward reference)
        # NOTE: heading_to_ref is already computed in vehicle frame.
        # Subtracting current_heading (world frame) causes large sign flips on straights.
        # Use heading_to_ref directly for both straights and curves.
        heading_error = heading_to_ref
        
        # Normalize error to [-pi, pi]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        # NEW: Smooth heading error to reduce jitter-driven steering oscillations
        if self.smoothed_heading_error is None:
            self.smoothed_heading_error = heading_error
        else:
            heading_alpha = self.heading_error_smoothing_alpha
            if abs(curve_metric) < self.straight_curvature_threshold:
                heading_alpha = min(0.9, heading_alpha + 0.2)
            self.smoothed_heading_error = (
                heading_alpha * heading_error +
                (1.0 - heading_alpha) * self.smoothed_heading_error
            )
        heading_error_for_control = self.smoothed_heading_error
        
        # Compute lateral error (cross-track error)
        # CRITICAL FIX: ref_x is already in VEHICLE frame (lateral offset from vehicle center)
        # vehicle_position is in WORLD frame (Unity world coordinates)
        # We CANNOT subtract them directly - they're in different coordinate systems!
        #
        # Solution: Use ref_x directly as lateral error (it's already the lateral offset)
        # For small headings, this is correct. For large headings, we need heading correction.
        
        # FIXED: ref_x is already the lateral offset in vehicle frame
        # We don't need to subtract vehicle_position[0] because:
        #   - ref_x is relative to vehicle center (vehicle frame)
        #   - vehicle_position[0] is absolute world position (world frame)
        #   - They're in different coordinate systems!
        #
        # For zero heading: lateral_error = ref_x (directly)
        # For non-zero heading: need to account for heading effect on lateral offset
        #
        # The heading effect: if vehicle is rotated, the lateral offset needs adjustment
        # But ref_x is already in vehicle frame, so for small headings it's correct
        # For large headings, we might need to account for how heading affects perception
        
        # FIXED: Use ref_x directly as lateral error (for recording/analysis)
        # But we need to think about the sign convention for control:
        # - ref_x > 0 means target is to the RIGHT of vehicle center
        # - If target is RIGHT, car is LEFT of target
        # - To move RIGHT (toward target), need to steer RIGHT (positive steering)
        # - PID: negative error → negative output, but we need positive output
        # - So we need to negate the error OR negate the output
        # - We'll negate total_error before PID to keep lateral_error correct for recording
        base_lateral_error = ref_x  # Keep original sign for recording/analysis
        
        # For large headings, the perception might be affected by vehicle rotation
        # But since ref_x is already in vehicle frame, we use it directly
        # The heading error is handled separately in total_error calculation
        heading_abs = abs(current_heading)
        heading_deg = np.degrees(heading_abs)
        
        # For very large headings, perception might be unreliable
        # But ref_x is still the best estimate we have
        lateral_error = base_lateral_error
        
        # NEW: Smooth lateral error to reduce sensitivity to perception noise
        # Perception instability can cause small changes in lane position that propagate to control
        # Smoothing the error prevents control oscillations from perception noise
        if self.smoothed_lateral_error is None:
            self.smoothed_lateral_error = lateral_error
        else:
            # Exponential smoothing: blend new error with previous smoothed error
            self.smoothed_lateral_error = (self.error_smoothing_alpha * lateral_error + 
                                         (1.0 - self.error_smoothing_alpha) * self.smoothed_lateral_error)
        
        # Use smoothed error for control (but store original for recording/analysis)
        # Steering sign convention: positive ref_x means target is to the RIGHT, so steering should be positive.
        lateral_error_for_control = self.smoothed_lateral_error
        lateral_error_for_recording = lateral_error  # Store original for analysis
        
        # FIXED: Prioritize lateral_error when heading_error conflicts
        # Analysis showed 67% of frames have heading/lateral conflicts causing 24% sign errors
        # Solution: When conflicts occur, prioritize lateral_error (more reliable for steering)
        # Only use heading_error for large headings (>10°) or when errors agree
        heading_abs_deg = abs(np.degrees(heading_error_for_control))
        
        # Check if heading_error and lateral_error have opposite signs (conflict)
        has_conflict = (
            (heading_error_for_control > 0 and lateral_error < 0) or
            (heading_error_for_control < 0 and lateral_error > 0)
        )
        
        if heading_abs_deg >= 10.0:
            # Large headings: use heading error primarily (80% heading, 20% lateral)
            # This prevents heading effect from dominating
            total_error = 0.8 * heading_error_for_control + 0.2 * lateral_error
        elif has_conflict:
            # CRITICAL FIX: When conflicts occur, use ONLY lateral_error (smoothed)
            # On curves, heading_error can conflict with lateral_error
            # Lateral error is more reliable for steering direction
            # Use 100% lateral_error to prevent wrong steering sign
            # This is especially important on curves where heading_error might be noisy
            total_error = lateral_error_for_control
        else:
            # Normal operation: use configurable weights (errors agree)
            # Use smoothed lateral error for control to reduce perception noise sensitivity
            total_error = (
                self.heading_weight * heading_error_for_control +
                self.lateral_weight * lateral_error_for_control
            )
        
        # Curvature-aware feedback gain (boost on gentle curves, neutral on sharp)
        feedback_gain = 1.0
        if self.feedback_gain_curvature_max > self.feedback_gain_curvature_min:
            if curve_metric_abs <= self.feedback_gain_curvature_min:
                feedback_gain = self.feedback_gain_max
            elif curve_metric_abs >= self.feedback_gain_curvature_max:
                feedback_gain = self.feedback_gain_min
            else:
                ratio = (curve_metric_abs - self.feedback_gain_curvature_min) / (
                    self.feedback_gain_curvature_max - self.feedback_gain_curvature_min
                )
                feedback_gain = self.feedback_gain_max + ratio * (
                    self.feedback_gain_min - self.feedback_gain_max
                )
        total_error_raw = total_error
        total_error = total_error_raw * feedback_gain

        # Scale error to prevent overreaction
        total_error = np.clip(total_error, -self.error_clip, self.error_clip)
        stored_total_error = total_error_raw
        
        # Add deadband to prevent constant small corrections
        # CRITICAL FIX: Don't apply deadband when on curves - we need continuous steering
        # On curves, even small errors need correction to maintain path following
        is_on_curve = curve_metric_abs > 0.05  # On a curve (curvature > ~3°)
        if abs(total_error) < self.deadband and not is_on_curve:
            # Only apply deadband on straight roads, not on curves
            total_error = 0.0

        
        # Store errors for metadata (use original, not smoothed, for analysis)
        stored_lateral_error = lateral_error_for_recording  # Store original for analysis
        stored_heading_error = heading_error
        
        # FIXED: More aggressive integral reset to prevent accumulation (was 11.63x increase after 6s)
        # Strategy: Multiple reset mechanisms with stricter thresholds
        
        # Mechanism 1: Reset if error is small for extended period (0.3 seconds = 9 frames)
        # CRITICAL FIX: Don't reset if we recently had large error (transition period)
        # When steering maxes out and error becomes small, we need the integral to maintain steering
        # FIXED: Increased threshold and delay to prevent jerky steering after curves
        if abs(total_error) < 0.1:  # Small error threshold
            # Only count small errors if we're not in transition from large error
            # AND we're not in the middle of a curve-to-straight transition
            in_transition = hasattr(self, 'straight_transition_counter') and self.straight_transition_counter < 15
            if not self.recent_large_error and not in_transition:
                self.small_error_count += 1
                if self.small_error_count > 9:  # Slower reset (0.3s instead of 0.2s) for smoother behavior
                    self.pid.reset()
                    self.small_error_count = 0
            else:
                # In transition period - don't count small errors (protect integral)
                self.small_error_count = 0
        else:
            self.small_error_count = 0
        
        # Mechanism 2: Reset if error is large (immediate reset)
        # CRITICAL FIX: Don't reset if steering would be maxed out OR if we're on a curve
        # When steering maxes out or we're on a curve, error is large, but we need to keep integral
        # to maintain steering through the curve
        # FIXED: Disable resets entirely when on curves to prevent jerky steering
        if abs(total_error) > 0.3:  # Large error (likely on curve)
            # Check if we're on a curve (path curvature > threshold)
            is_on_curve = curve_metric_abs > 0.05  # On a curve (curvature > ~3°)
            
            # Estimate if steering would be maxed out
            estimated_steering = abs(self.pid.kp * total_error + self.pid.integral)
            steering_would_be_maxed = estimated_steering > 0.9 * self.max_steering
            
            # CRITICAL FIX: Don't reset if:
            # 1. Steering is maxed out (we need the integral)
            # 2. We're on a curve (error oscillation is normal, don't reset)
            # Only reset if steering is NOT maxed AND we're NOT on a curve
            if not steering_would_be_maxed and not is_on_curve:
                # Reset immediately on large errors (but only if not maxed and not on curve)
                if abs(self.pid.integral) > 0.03:  # Stricter threshold (was 0.05)
                    self.pid.reset()
                    self.small_error_count = 0
            # If steering IS maxed out OR we're on a curve, keep the integral (don't reset)
        
        # Mechanism 3: Reset on sign change (oscillation detection)
        # CRITICAL FIX: Don't reset on sign change if we need constant steering (curves)
        # On curved roads, error naturally oscillates around zero, but we need constant steering
        # FIXED: Disable resets entirely when on curves to prevent jerky steering
        if self.last_error_sign is not None and np.sign(total_error) != self.last_error_sign:
            # Check if we're on a curve (path curvature > threshold)
            is_on_curve = curve_metric_abs > 0.05  # On a curve (curvature > ~3°)
            
            # CRITICAL FIX: Never reset on sign change when on a curve
            # Error oscillation is normal on curves - we need constant steering, not resets
            if is_on_curve:
                # On curve - don't reset, keep integral to maintain steering
                pass  # Keep integral
            else:
                # Not on curve - check if this is true oscillation (small errors) or overshooting
                error_magnitude = abs(total_error)
                last_error_magnitude = abs(self.last_error) if hasattr(self, 'last_error') else 0.0
                
                # CRITICAL FIX: Don't reset if we just had a large error and are now correcting
                # This prevents reset when steering maxes out and then error becomes small
                was_large_error = last_error_magnitude > 0.3  # Had large error (steering was maxed)
                is_small_error = error_magnitude < 0.15  # Now small error (correcting)
                is_overshooting = (last_error_magnitude > 0.2 and error_magnitude > 0.2)  # Large → large opposite
                
                # Only reset if:
                # 1. Both errors are small (< 0.15) - true oscillation around zero
                # 2. OR we're overshooting (large → large opposite, not correcting)
                # DO NOT reset if: large error → small error (we're successfully correcting!)
                should_reset = False
                if last_error_magnitude < 0.15 and error_magnitude < 0.15:
                    # Both small - true oscillation, safe to reset
                    should_reset = True
                elif is_overshooting:
                    # Large → large opposite - overshooting, reset
                    should_reset = True
                # Otherwise: large → small means we're correcting, KEEP integral!
                
                if should_reset:
                    # Error changed sign and it's true oscillation or overshooting - reset integral
                    self.pid.reset()
                    self.small_error_count = 0
                # Otherwise, keep integral to maintain constant steering
        self.last_error_sign = np.sign(total_error) if abs(total_error) > 0.01 else None
        if not hasattr(self, 'last_error'):
            self.last_error = 0.0
        self.last_error = total_error  # Store for next frame comparison
        
        # Mechanism 4: Periodic reset with integral decay (every 0.67 seconds = 20 frames)
        # CRITICAL FIX: Don't decay if steering is maxed or we recently had large error
        # When steering maxes out or error is large, we need the integral to maintain steering
        if not hasattr(self, 'reset_counter'):
            self.reset_counter = 0
        self.reset_counter += 1
        if self.reset_counter > 20:  # Every 0.67 seconds (was 1 second = 30 frames)
            # Estimate if steering would be maxed out
            estimated_steering = abs(self.pid.kp * total_error + self.pid.integral)
            steering_would_be_maxed = estimated_steering > 0.9 * self.max_steering
            error_is_large = abs(total_error) > 0.3  # On a curve (large error)
            
            # Only apply decay if:
            # 1. Steering is NOT maxed out
            # 2. Error is NOT large (not on a curve)
            # 3. We did NOT recently have large error (not in transition period)
            if not steering_would_be_maxed and not error_is_large and not self.recent_large_error:
                if abs(self.pid.integral) > 0.03:  # Stricter threshold (was 0.05)
                    # NEW: Apply exponential decay instead of full reset (smoother)
                    decay_factor = 0.5  # Reduce integral by 50%
                    self.pid.integral *= decay_factor
            # If steering is maxed, error is large, or we recently had large error, don't decay
            self.reset_counter = 0
        
        # Track if we recently had large error (on a curve)
        # This helps protect integral during transition from large error to small error
        if abs(total_error) > 0.3:
            self.large_error_frames = min(30, self.large_error_frames + 2)  # Build up quickly, max 30 frames
            self.recent_large_error = True
        else:
            # Decay the "recent large error" flag slowly
            # Keep it for ~1 second (30 frames) after error becomes small (transition period)
            # This protects integral when steering maxes out and then error becomes small
            if self.large_error_frames > 0:
                self.large_error_frames = max(0, self.large_error_frames - 1)  # Decay slowly (1 frame per update)
            if self.large_error_frames == 0:
                self.recent_large_error = False
        
        # Mechanism 5: FIXED - More aggressive integral decay to prevent accumulation
        # CRITICAL FIX: Don't decay integral when on a curve (large error) or when steering is maxed
        # When steering maxes out or error is large, we need the integral to maintain steering
        # Also protect during transition from large error to small error (recent_large_error)
        # Estimate if steering would be maxed out
        estimated_steering = abs(self.pid.kp * total_error + self.pid.integral)
        steering_would_be_maxed = estimated_steering > 0.9 * self.max_steering
        error_is_large = abs(total_error) > 0.3  # On a curve (large error)
        
        # Only apply decay if:
        # 1. Steering is NOT maxed out
        # 2. Error is NOT large (not on a curve)
        # 3. We did NOT recently have large error (not in transition period)
        # If steering is maxed, error is large, or we recently had large error, we need the integral
        if not steering_would_be_maxed and not error_is_large and not self.recent_large_error:
            # REVERTED: Back to 0.90 decay (was working better than 0.95)
            # Analysis showed 0.95 decay made things WORSE (13.19x vs 5.54x)
            # 0.90 decay (10% per frame) is stronger and was working better
            if abs(self.pid.integral) > 0.01:  # Restored threshold (was 0.005)
                # Apply decay factor (0.90 = 10% decay per frame, was 0.95 = 5%)
                # Stronger decay prevents accumulation more effectively
                decay_factor = 0.90
                self.pid.integral *= decay_factor
            elif abs(self.pid.integral) > 0.005:  # Light decay threshold
                # Light decay (0.99 = 1% decay per frame) for smaller integrals
                decay_factor = 0.99
                self.pid.integral *= decay_factor
        # If steering IS maxed out OR error is large (on curve), don't decay the integral
        
        # PP telemetry defaults
        pp_alpha = 0.0
        pp_lookahead_distance = 0.0
        pp_geometric_steering = 0.0
        pp_feedback_steering_val = 0.0
        pp_ref_jump_clamped = False
        pp_stale_hold_active = False
        stanley_heading_term = 0.0
        stanley_crosstrack_term = 0.0
        stanley_speed = current_speed if current_speed is not None else 0.0
        total_error_scaled = total_error

        if self.control_mode == "pure_pursuit":
            # Pure Pursuit: compute steering from geometry, not error correction
            pp_ref_x = float(reference_point['x'])
            pp_ref_y = float(reference_point['y'])

            # Ref jump clamping (second safety net after trajectory's jump clamp)
            ref_x_delta = pp_ref_x - self._pp_last_ref_x
            if abs(ref_x_delta) > self.pp_ref_jump_clamp:
                pp_ref_x = self._pp_last_ref_x + np.sign(ref_x_delta) * self.pp_ref_jump_clamp
                pp_ref_jump_clamped = True
            self._pp_last_ref_x = pp_ref_x

            if using_stale_perception:
                self._pp_stale_frames += 1
                pp_stale_hold_active = True
                pp_geometric_steering = self._pp_last_valid_steering * (
                    self.pp_stale_decay ** self._pp_stale_frames
                )
            else:
                self._pp_stale_frames = 0
                ld = np.sqrt(pp_ref_x**2 + pp_ref_y**2)
                ld = max(ld, self.pp_min_lookahead)
                alpha = np.arctan2(pp_ref_x, pp_ref_y)

                wheelbase = 2.5
                steering_rad = np.arctan(2.0 * wheelbase * np.sin(alpha) / ld)
                max_steering_rad = np.radians(30.0)
                pp_geometric_steering = float(
                    np.clip(steering_rad / max_steering_rad, -1.0, 1.0)
                )
                pp_geometric_steering *= self.max_steering
                self._pp_last_valid_steering = pp_geometric_steering
                pp_alpha = float(alpha)
                pp_lookahead_distance = float(ld)

            pp_feedback_steering_val = float(
                self.pid.update(lateral_error_for_control, dt) * self.pp_feedback_gain
            )

            feedforward_steering = pp_geometric_steering
            feedback_steering = pp_feedback_steering_val
            steering_before_limits = pp_geometric_steering + pp_feedback_steering_val
            speed_gain_final = 1.0
            speed_gain_scale = 0.0
            speed_gain_applied = False
            curve_gain_scheduled = 1.0
            curve_feedforward_scale = 1.0
            abs_curvature = curve_metric_abs

        else:
            # PID / Stanley: feedforward from curvature + feedback from error
            wheelbase = 2.5
            if not hasattr(self, 'smoothed_path_curvature'):
                self.smoothed_path_curvature = 0.0
            smoothing_alpha = self.curvature_smoothing_alpha
            if abs(path_curvature - self.smoothed_path_curvature) > self.curvature_transition_threshold:
                smoothing_alpha = min(smoothing_alpha, self.curvature_transition_alpha)
            self.smoothed_path_curvature = (
                smoothing_alpha * path_curvature +
                (1.0 - smoothing_alpha) * self.smoothed_path_curvature
            )

            curvature_preview = float(reference_point.get('curvature_preview', 0.0) or 0.0)
            preview_blend = 0.3
            if (
                abs(curvature_preview) > abs(self.smoothed_path_curvature) + 0.001
                and abs(curvature_preview) > 0.003
            ):
                self.smoothed_path_curvature = (
                    (1.0 - preview_blend) * self.smoothed_path_curvature
                    + preview_blend * curvature_preview
                )

            feedforward_steering = 0.0
            curve_gain_scheduled = 1.0
            abs_curvature = abs(self.smoothed_path_curvature)
            speed_gain_final = 1.0
            speed_gain_scale = 0.0
            speed_gain_applied = False
            if self.curve_feedforward_curvature_min is not None and self.curve_feedforward_curvature_max is not None:
                min_curv = max(1e-6, self.curve_feedforward_curvature_min)
                max_curv = max(min_curv, self.curve_feedforward_curvature_max)
                if abs_curvature <= min_curv:
                    curve_gain_scheduled = self.curve_feedforward_gain_min
                elif abs_curvature >= max_curv:
                    curve_gain_scheduled = self.curve_feedforward_gain_max
                else:
                    ratio = (abs_curvature - min_curv) / (max_curv - min_curv)
                    curve_gain_scheduled = self.curve_feedforward_gain_min + ratio * (
                        self.curve_feedforward_gain_max - self.curve_feedforward_gain_min
                    )

            curve_feedforward_scale = 1.0
            for bin_cfg in self.curve_feedforward_bins:
                if not isinstance(bin_cfg, dict):
                    continue
                min_curv = float(bin_cfg.get('min_curvature', 0.0))
                max_curv = float(bin_cfg.get('max_curvature', float('inf')))
                if abs_curvature < min_curv or abs_curvature >= max_curv:
                    continue
                curve_feedforward_scale = float(bin_cfg.get('gain_scale', 1.0))
                break

            if current_speed is not None and self.speed_gain_max_speed > self.speed_gain_min_speed:
                if current_speed <= self.speed_gain_min_speed:
                    speed_gain = self.speed_gain_min
                elif current_speed >= self.speed_gain_max_speed:
                    speed_gain = self.speed_gain_max
                else:
                    ratio = (current_speed - self.speed_gain_min_speed) / (
                        self.speed_gain_max_speed - self.speed_gain_min_speed
                    )
                    speed_gain = self.speed_gain_min + ratio * (self.speed_gain_max - self.speed_gain_min)
                if self.speed_gain_curvature_max > self.speed_gain_curvature_min:
                    if curve_metric_abs <= self.speed_gain_curvature_min:
                        speed_gain_scale = 1.0
                    elif curve_metric_abs >= self.speed_gain_curvature_max:
                        speed_gain_scale = 0.0
                    else:
                        ratio = (curve_metric_abs - self.speed_gain_curvature_min) / (
                            self.speed_gain_curvature_max - self.speed_gain_curvature_min
                        )
                        speed_gain_scale = 1.0 - ratio
                else:
                    speed_gain_scale = 1.0
                speed_gain_final = 1.0 + (speed_gain - 1.0) * speed_gain_scale
                speed_gain_applied = abs(speed_gain_final - 1.0) > 1e-3

            min_curv_ff = max(1e-6, self.curve_feedforward_curvature_min or 0.0)
            if abs_curvature >= min_curv_ff:
                scaled_curvature = self.smoothed_path_curvature
                if abs(self.smoothed_path_curvature) > self.curvature_scale_threshold:
                    scaled_curvature = self.smoothed_path_curvature * self.curvature_scale_factor
                raw_steering = wheelbase * scaled_curvature
                max_steering_rad = np.radians(30.0)
                feedforward_steering = np.clip(raw_steering / max_steering_rad, -1.0, 1.0)
                feedforward_steering *= self.max_steering
                if abs_curvature >= self.curve_feedforward_threshold:
                    feedforward_steering *= self.curve_feedforward_gain * curve_gain_scheduled
                feedforward_steering *= curve_feedforward_scale
                feedforward_steering = np.clip(feedforward_steering, -self.max_steering, self.max_steering)

            # Track curve-to-straight transition for smooth integral decay
            is_on_curve = curve_metric_abs > 0.05
            if not hasattr(self, 'was_on_curve'):
                self.was_on_curve = False
            if not hasattr(self, 'straight_transition_counter'):
                self.straight_transition_counter = 0

            transition_to_straight = self.was_on_curve and not is_on_curve
            transition_to_curve = not self.was_on_curve and is_on_curve

            if not hasattr(self, '_curve_entry_decay_remaining'):
                self._curve_entry_decay_remaining = 0
            if transition_to_curve:
                self._curve_entry_decay_remaining = 5
            if self._curve_entry_decay_remaining > 0:
                self.pid.integral *= 0.35
                self._curve_entry_decay_remaining -= 1

            if transition_to_straight:
                self.straight_transition_counter = 0

            if not is_on_curve and self.straight_transition_counter < 15:
                if abs(total_error) < 0.15:
                    decay_factor = 0.70
                    self.pid.integral *= decay_factor
                    self.straight_transition_counter += 1
                else:
                    self.straight_transition_counter = 0
            elif is_on_curve:
                self.straight_transition_counter = 0

            self.was_on_curve = is_on_curve

            if self.control_mode == "stanley":
                stanley_heading_term = self.stanley_heading_weight * heading_error
                denom = max(self.stanley_soft_speed, float(stanley_speed))
                stanley_crosstrack_term = np.arctan2(
                    self.stanley_k * lateral_error_for_control,
                    denom,
                )
                feedback_steering = stanley_heading_term + stanley_crosstrack_term
            else:
                feedback_steering = self.pid.update(total_error, dt)

            steering_before_limits = feedforward_steering + feedback_steering
            if speed_gain_final != 1.0:
                steering_before_limits *= speed_gain_final
        steering_pre_rate_limit = steering_before_limits
        steering_post_rate_limit = steering_before_limits
        steering_post_jerk_limit = steering_before_limits
        steering_post_sign_flip = steering_before_limits
        steering_post_hard_clip = steering_before_limits
        steering_post_smoothing = steering_before_limits
        steering_rate_limited_active = False
        steering_jerk_limited_active = False
        steering_hard_clip_active = False
        steering_smoothing_active = False
        steering_rate_limited_delta = 0.0
        steering_jerk_limited_delta = 0.0
        steering_hard_clip_delta = 0.0
        steering_smoothing_delta = 0.0
        steering_rate_limit_base_from_error = 0.0
        steering_rate_limit_curve_scale = 1.0
        steering_rate_limit_curve_metric_abs = float(curve_metric_abs)
        steering_rate_limit_curve_metric_source = curve_metric_source
        steering_rate_limit_curve_min = 0.0
        steering_rate_limit_curve_max = 0.0
        steering_rate_limit_scale_min = 0.0
        steering_rate_limit_curve_regime_code = 0.0
        steering_rate_limit_after_curve = 0.0
        steering_rate_limit_after_floor = 0.0
        steering_rate_limit_effective = 0.0
        steering_rate_limit_requested_delta = 0.0
        steering_rate_limit_margin = 0.0
        steering_rate_limit_unlock_delta_needed = 0.0
        curve_entry_assist_active = False
        curve_entry_assist_triggered = False
        curve_entry_assist_rearm_remaining = int(self._curve_entry_assist_rearm_remaining)
        steering_jerk_limit_requested_rate_delta = 0.0
        steering_jerk_limit_allowed_rate_delta = 0.0
        steering_jerk_limit_margin = 0.0
        steering_jerk_limit_unlock_rate_delta_needed = 0.0
        curve_entry_schedule_active = False
        curve_entry_schedule_triggered = False
        curve_entry_schedule_handoff_triggered = False
        curve_entry_schedule_frames_remaining = int(self._curve_entry_schedule_frames_remaining)
        curve_commit_mode_active = False
        curve_commit_mode_triggered = False
        curve_commit_mode_handoff_triggered = False
        curve_commit_mode_frames_remaining = int(self._curve_commit_mode_frames_remaining)
        dynamic_curve_rate_request_delta = 0.0
        dynamic_curve_rate_deficit = 0.0
        dynamic_curve_rate_boost = 0.0
        dynamic_curve_jerk_boost_factor = 1.0
        dynamic_curve_authority_active = False
        dynamic_curve_lateral_accel_est_g = 0.0
        dynamic_curve_lateral_jerk_est_gps = 0.0
        dynamic_curve_lateral_jerk_est_smoothed_gps = 0.0
        dynamic_curve_speed_scale = 1.0
        dynamic_curve_comfort_scale = 1.0
        dynamic_curve_comfort_accel_gate = 1.0
        dynamic_curve_comfort_jerk_penalty = 1.0
        dynamic_curve_rate_boost_cap_effective = self.dynamic_curve_rate_boost_max
        dynamic_curve_jerk_boost_cap_effective = self.dynamic_curve_jerk_boost_max_factor
        dynamic_curve_hard_clip_boost = 0.0
        dynamic_curve_hard_clip_boost_cap_effective = self.dynamic_curve_hard_clip_boost_max
        dynamic_curve_hard_clip_limit_effective = self.max_steering
        dynamic_curve_entry_governor_active = False
        dynamic_curve_entry_governor_scale = 1.0
        dynamic_curve_entry_governor_rate_scale = 1.0
        dynamic_curve_entry_governor_jerk_scale = 1.0
        steering_authority_gap = 0.0
        steering_transfer_ratio = 1.0
        steering_first_limiter_stage_code = 0.0
        curve_unwind_active = False
        curve_unwind_frames_remaining = int(self._curve_unwind_frames_remaining)
        curve_unwind_progress = 0.0
        curve_unwind_rate_scale = 1.0
        curve_unwind_jerk_scale = 1.0
        curve_unwind_integral_decay_applied = 1.0
        turn_feasibility_active = False
        turn_feasibility_infeasible = False
        turn_feasibility_curvature_abs = 0.0
        turn_feasibility_speed_mps = 0.0
        turn_feasibility_required_lat_accel_g = 0.0
        turn_feasibility_comfort_limit_g = 0.0
        turn_feasibility_peak_limit_g = 0.0
        turn_feasibility_selected_limit_g = 0.0
        turn_feasibility_margin_g = 0.0
        turn_feasibility_speed_limit_mps = 0.0
        turn_feasibility_speed_delta_mps = 0.0
        
        # Get PID internal state
        pid_integral = self.pid.integral
        pid_derivative = 0.0
        if self.pid.prev_time is not None and dt > 0:
            pid_derivative = self.pid.kd * ((total_error - self.pid.prev_error) / dt)
        
        # FIXED: Add steering rate limiting to prevent sudden changes and oscillation
        # CRITICAL FIX: Adaptive rate limiting - allow faster changes when error is large
        # Analysis showed rate limiting prevents recovery from large errors
        # Solution: Increase rate limit when error is large (need quick recovery)
        if not hasattr(self, 'last_steering'):
            self.last_steering = 0.0
        
        # Adaptive rate limiting based on error magnitude
        # Small errors: Slow rate limit (smoothness)
        # Large errors: Fast rate limit (recovery)
        error_magnitude = abs(total_error)
        is_straight = curve_metric_abs < self.straight_curvature_threshold
        is_control_straight_proxy = is_straight
        curve_phase_source = "metric_proxy"
        distance_to_next_curve_start_m = None
        current_curve_progress_ratio = None
        current_curve_index = None
        distance_curve_state = self._compute_distance_curve_state(
            road_center_reference_t=road_center_reference_t,
            lookahead_m=max(0.0, float(ref_y)),
        )
        if distance_curve_state is not None:
            curve_phase_source = "distance_track"
            curve_upcoming = bool(distance_curve_state["curve_upcoming"])
            curve_at_car = bool(distance_curve_state["curve_at_car"])
            distance_to_next_curve_start_m = float(distance_curve_state["distance_to_next_curve_start_m"])
            if distance_curve_state.get("current_curve_progress_ratio") is not None:
                current_curve_progress_ratio = float(
                    distance_curve_state["current_curve_progress_ratio"]
                )
            if distance_curve_state.get("current_curve_index") is not None:
                current_curve_index = int(distance_curve_state["current_curve_index"])
            self._curve_upcoming_state = curve_upcoming
            self._curve_at_car_state = curve_at_car
            self._curve_at_car_distance_remaining = distance_to_next_curve_start_m
        else:
            if self._curve_upcoming_state:
                if curve_metric_abs <= self.curve_upcoming_exit_threshold:
                    self._curve_upcoming_off_counter += 1
                else:
                    self._curve_upcoming_off_counter = 0
                if self._curve_upcoming_off_counter >= self.curve_upcoming_off_frames:
                    self._curve_upcoming_state = False
                    self._curve_upcoming_off_counter = 0
                    self._curve_upcoming_on_counter = 0
            else:
                if curve_metric_abs >= self.curve_upcoming_enter_threshold:
                    self._curve_upcoming_on_counter += 1
                else:
                    self._curve_upcoming_on_counter = 0
                if self._curve_upcoming_on_counter >= self.curve_upcoming_on_frames:
                    self._curve_upcoming_state = True
                    self._curve_upcoming_on_counter = 0
                    self._curve_upcoming_off_counter = 0
            curve_upcoming = bool(self._curve_upcoming_state)
            if curve_upcoming and not self._prev_curve_upcoming:
                ref_distance_ahead = max(0.0, float(ref_y))
                self._curve_at_car_distance_remaining = max(
                    self.curve_at_car_distance_min_m,
                    ref_distance_ahead,
                )
            if not curve_upcoming:
                self._curve_at_car_state = False
                self._curve_at_car_distance_remaining = None
            elif self._curve_at_car_distance_remaining is not None:
                speed_for_distance = (
                    float(current_speed)
                    if current_speed is not None and np.isfinite(float(current_speed))
                    else float(reference_point.get('velocity', 0.0) or 0.0)
                )
                distance_step = max(0.0, speed_for_distance) * dt
                self._curve_at_car_distance_remaining = max(
                    0.0,
                    float(self._curve_at_car_distance_remaining) - float(distance_step),
                )
                if self._curve_at_car_distance_remaining <= 1e-3:
                    self._curve_at_car_state = True
            curve_at_car = bool(self._curve_at_car_state)
            distance_to_next_curve_start_m = (
                float(self._curve_at_car_distance_remaining)
                if self._curve_at_car_distance_remaining is not None
                else None
            )
        road_curvature_abs = float(abs(path_curvature)) if road_curvature_valid else None
        if road_curvature_abs is not None:
            if road_curvature_abs >= self.road_curve_enter_threshold:
                self._road_straight_state = False
            elif road_curvature_abs <= self.road_curve_exit_threshold:
                self._road_straight_state = True
            self._road_straight_invalid_frames_remaining = self.road_straight_hold_invalid_frames
            is_road_straight = self._road_straight_state
        elif self._road_straight_invalid_frames_remaining > 0:
            is_road_straight = self._road_straight_state
            self._road_straight_invalid_frames_remaining -= 1
        else:
            is_road_straight = None
        single_owner_entry_phase = bool(
            self.dynamic_curve_single_owner_mode
            and self.dynamic_curve_entry_governor_enabled
            and (curve_upcoming or curve_at_car)
        )
        if single_owner_entry_phase:
            self._curve_entry_schedule_frames_remaining = 0
            self._curve_entry_schedule_elapsed_frames = 0
            self._curve_commit_mode_frames_remaining = 0
            self._curve_commit_handoff_streak = 0
        if not curve_at_car:
            self._curve_entry_schedule_fallback_fired_for_curve = False
        if self._curve_entry_schedule_fallback_cooldown_remaining > 0:
            self._curve_entry_schedule_fallback_cooldown_remaining -= 1
        if self._curve_commit_retrigger_cooldown_remaining > 0:
            self._curve_commit_retrigger_cooldown_remaining -= 1
        if self.curve_unwind_policy_enabled:
            if curve_at_car:
                self._curve_unwind_frames_remaining = 0
            elif self._prev_curve_at_car and self.curve_unwind_frames > 0:
                self._curve_unwind_frames_remaining = self.curve_unwind_frames
            if self._curve_unwind_frames_remaining > 0:
                curve_unwind_active = True
                curve_unwind_frames_remaining = int(self._curve_unwind_frames_remaining)
                unwind_total = max(1, self.curve_unwind_frames)
                curve_unwind_progress = float(
                    np.clip(
                        (unwind_total - self._curve_unwind_frames_remaining) / unwind_total,
                        0.0,
                        1.0,
                    )
                )
                curve_unwind_rate_scale = float(
                    self.curve_unwind_rate_scale_start
                    + (self.curve_unwind_rate_scale_end - self.curve_unwind_rate_scale_start)
                    * curve_unwind_progress
                )
                curve_unwind_jerk_scale = float(
                    self.curve_unwind_jerk_scale_start
                    + (self.curve_unwind_jerk_scale_end - self.curve_unwind_jerk_scale_start)
                    * curve_unwind_progress
                )
                if abs(self.pid.integral) > 1e-6:
                    self.pid.integral *= self.curve_unwind_integral_decay
                    curve_unwind_integral_decay_applied = self.curve_unwind_integral_decay
                self._curve_unwind_frames_remaining = max(
                    0, self._curve_unwind_frames_remaining - 1
                )
                curve_unwind_frames_remaining = int(self._curve_unwind_frames_remaining)
        using_dynamic_schedule_fallback = (
            self.dynamic_curve_authority_enabled
            and self.curve_entry_schedule_fallback_only_when_dynamic
        )
        schedule_trigger_condition = (not self._prev_curve_at_car) and curve_at_car
        if using_dynamic_schedule_fallback:
            schedule_trigger_condition = (
                curve_at_car
                and self._curve_entry_schedule_frames_remaining <= 0
                and not self._curve_entry_schedule_fallback_fired_for_curve
                and self._curve_entry_schedule_fallback_cooldown_remaining <= 0
                and (
                    self._dynamic_authority_deficit_streak
                    >= self.curve_entry_schedule_fallback_deficit_frames
                )
            )
        if (
            not single_owner_entry_phase
            and
            self.curve_entry_schedule_enabled
            and self.curve_entry_schedule_frames > 0
            and schedule_trigger_condition
        ):
            self._curve_entry_schedule_frames_remaining = self.curve_entry_schedule_frames
            self._curve_entry_schedule_elapsed_frames = 0
            curve_entry_schedule_triggered = True
            if using_dynamic_schedule_fallback:
                self._curve_entry_schedule_fallback_fired_for_curve = True
                self._curve_entry_schedule_fallback_cooldown_remaining = (
                    self.curve_entry_schedule_fallback_rearm_cooldown_frames
                )
                self._dynamic_authority_deficit_streak = 0
                if current_curve_index is not None:
                    self._curve_entry_schedule_fallback_last_curve_index = int(current_curve_index)
        if (
            not single_owner_entry_phase
            and
            self.curve_commit_mode_enabled
            and not self.curve_entry_schedule_enabled
            and not self._prev_curve_at_car
            and curve_at_car
            and self._curve_commit_mode_frames_remaining <= 0
        ):
            self._curve_commit_mode_frames_remaining = self.curve_commit_mode_max_frames
            curve_commit_mode_triggered = self._curve_commit_mode_frames_remaining > 0
        if (
            not single_owner_entry_phase
            and
            self.curve_commit_mode_enabled
            and not self.curve_entry_schedule_enabled
            and curve_at_car
            and self._curve_commit_mode_frames_remaining <= 0
            and error_magnitude >= 0.25
            and error_magnitude > (self._last_error_magnitude + 0.01)
        ):
            self._curve_commit_mode_frames_remaining = self.curve_commit_mode_max_frames
            curve_commit_mode_triggered = self._curve_commit_mode_frames_remaining > 0
        if self._curve_entry_schedule_frames_remaining > 0:
            curve_entry_schedule_active = True
            self._curve_entry_schedule_elapsed_frames += 1
        curve_entry_schedule_frames_remaining = int(self._curve_entry_schedule_frames_remaining)
        dynamic_curve_rate_request_delta = float(abs(steering_before_limits - self.last_steering))
        if error_magnitude > 0.6:
            # Very large error (> 0.6) - allow fast steering changes for recovery
            max_steering_rate = 0.24
        elif error_magnitude > 0.3:
            # Moderate error (0.3-0.6) - moderate rate limit
            max_steering_rate = 0.16
        else:
            # Small error (< 0.3) - slow rate limit for smoothness
            max_steering_rate = 0.08
        steering_rate_limit_base_from_error = float(max_steering_rate)
        
        # Curvature-aware scaling: reduce steering rate on sharper curves to limit jerk
        rate_scale = 1.0
        min_curv = max(1e-6, self.steering_rate_curvature_min)
        max_curv = max(min_curv, self.steering_rate_curvature_max)
        steering_rate_limit_curve_min = float(min_curv)
        steering_rate_limit_curve_max = float(max_curv)
        steering_rate_limit_scale_min = float(max(0.0, self.steering_rate_scale_min))
        if curve_metric_abs <= min_curv:
            rate_scale = 1.0
            steering_rate_limit_curve_regime_code = 0.0
        elif curve_metric_abs >= max_curv:
            rate_scale = max(0.0, self.steering_rate_scale_min)
            steering_rate_limit_curve_regime_code = 2.0
        else:
            ratio = (curve_metric_abs - min_curv) / (max_curv - min_curv)
            min_scale = max(0.0, self.steering_rate_scale_min)
            rate_scale = 1.0 - ratio * (1.0 - min_scale)
            steering_rate_limit_curve_regime_code = 1.0
        steering_rate_limit_curve_scale = float(rate_scale)
        max_steering_rate *= rate_scale
        steering_rate_limit_after_curve = float(max_steering_rate)
        dynamic_curve_entry_governor_anticipatory_active = bool(
            self.dynamic_curve_entry_governor_anticipatory_enabled
            and curve_upcoming
            and not curve_at_car
        )
        entry_phase_active_pre = bool(
            curve_at_car
            or curve_entry_schedule_active
            or self._curve_commit_mode_frames_remaining > 0
            or dynamic_curve_entry_governor_anticipatory_active
        )
        if self.curve_unwind_policy_enabled:
            if entry_phase_active_pre:
                self._curve_unwind_frames_remaining = 0
            elif (
                (self._prev_entry_phase_active or ((not self._prev_is_straight) and is_straight))
                and self.curve_unwind_frames > 0
            ):
                self._curve_unwind_frames_remaining = max(
                    self._curve_unwind_frames_remaining,
                    self.curve_unwind_frames,
                )
        entry_governor_exclusive_active_pre = (
            self.dynamic_curve_entry_governor_enabled
            and self.dynamic_curve_entry_governor_exclusive_mode
            and entry_phase_active_pre
        )
        dynamic_curve_precurve_active = bool(
            self.dynamic_curve_authority_precurve_enabled
            and dynamic_curve_entry_governor_anticipatory_active
        )
        dynamic_curve_context_scale = (
            self.dynamic_curve_authority_precurve_scale if dynamic_curve_precurve_active else 1.0
        )

        # Industry-style mode scheduling:
        # keep straightaway behavior unchanged, but guarantee curve-entry authority
        # when error is already moderate/large so turn-in does not lag.
        if not is_straight:
            if error_magnitude > 0.6:
                max_steering_rate = max(max_steering_rate, self.curve_rate_floor_large_error)
            elif error_magnitude > 0.3:
                max_steering_rate = max(max_steering_rate, self.curve_rate_floor_moderate_error)
        if self._curve_entry_schedule_frames_remaining > 0 and not entry_governor_exclusive_active_pre:
            max_steering_rate = max(max_steering_rate, self.curve_entry_schedule_min_rate)
        if self._curve_commit_mode_frames_remaining > 0:
            curve_commit_mode_active = True
            if not entry_governor_exclusive_active_pre:
                max_steering_rate = max(max_steering_rate, self.curve_commit_mode_min_rate)
        if single_owner_entry_phase:
            max_steering_rate = max(
                max_steering_rate,
                steering_rate_limit_base_from_error,
                self.dynamic_curve_single_owner_min_rate,
            )
        speed_for_comfort = (
            float(current_speed)
            if current_speed is not None and np.isfinite(float(current_speed))
            else float(reference_point.get('velocity', 0.0) or 0.0)
        )
        speed_for_comfort = max(0.0, speed_for_comfort)
        turn_feasibility_curvature_abs = float(curve_metric_abs)
        turn_feasibility_speed_mps = float(speed_for_comfort)
        dynamic_curve_lateral_accel_est_g = float(
            (speed_for_comfort * speed_for_comfort * float(curve_metric_abs)) / 9.81
        )
        turn_feasibility_required_lat_accel_g = float(dynamic_curve_lateral_accel_est_g)
        turn_feasibility_comfort_limit_g = float(self.dynamic_curve_comfort_lat_accel_comfort_max_g)
        turn_feasibility_peak_limit_g = float(self.dynamic_curve_comfort_lat_accel_peak_max_g)
        turn_feasibility_selected_limit_g = float(
            self.dynamic_curve_comfort_lat_accel_peak_max_g
            if self.turn_feasibility_use_peak_bound
            else self.dynamic_curve_comfort_lat_accel_comfort_max_g
        )
        turn_feasibility_active = bool(
            self.turn_feasibility_governor_enabled
            and turn_feasibility_curvature_abs >= self.turn_feasibility_curvature_min
            and (curve_upcoming or curve_at_car or not is_straight)
        )
        if turn_feasibility_active:
            effective_limit_g = max(
                0.0,
                turn_feasibility_selected_limit_g - self.turn_feasibility_guardband_g,
            )
            turn_feasibility_margin_g = float(effective_limit_g - turn_feasibility_required_lat_accel_g)
            if turn_feasibility_curvature_abs > 1e-9 and effective_limit_g > 0.0:
                turn_feasibility_speed_limit_mps = float(
                    np.sqrt((effective_limit_g * 9.81) / turn_feasibility_curvature_abs)
                )
            else:
                turn_feasibility_speed_limit_mps = 0.0
            turn_feasibility_speed_delta_mps = float(
                max(0.0, turn_feasibility_speed_mps - turn_feasibility_speed_limit_mps)
            )
            turn_feasibility_infeasible = bool(turn_feasibility_margin_g < 0.0)
        if dt > 0.0 and self._last_lateral_accel_est_initialized:
            dynamic_curve_lateral_jerk_est_gps = float(
                abs(dynamic_curve_lateral_accel_est_g - self._last_lateral_accel_est_g) / dt
            )
        else:
            dynamic_curve_lateral_jerk_est_gps = 0.0
        self._last_lateral_accel_est_g = dynamic_curve_lateral_accel_est_g
        self._last_lateral_accel_est_initialized = True
        jerk_alpha = self.dynamic_curve_lat_jerk_smoothing_alpha
        dynamic_curve_lateral_jerk_est_smoothed_gps = float(
            jerk_alpha * dynamic_curve_lateral_jerk_est_gps
            + (1.0 - jerk_alpha) * self._smoothed_lateral_jerk_est_gps
        )
        self._smoothed_lateral_jerk_est_gps = dynamic_curve_lateral_jerk_est_smoothed_gps
        speed_ratio = (
            (self.dynamic_curve_speed_high_mps - speed_for_comfort)
            / max(1e-3, self.dynamic_curve_speed_high_mps - self.dynamic_curve_speed_low_mps)
        )
        speed_ratio = float(np.clip(speed_ratio, 0.0, 1.0))
        dynamic_curve_speed_scale = float(
            1.0 + speed_ratio * (self.dynamic_curve_speed_boost_max_scale - 1.0)
        )
        accel_headroom = (
            self.dynamic_curve_comfort_lat_accel_peak_max_g - dynamic_curve_lateral_accel_est_g
        ) / max(
            1e-3,
            self.dynamic_curve_comfort_lat_accel_peak_max_g
            - self.dynamic_curve_comfort_lat_accel_comfort_max_g,
        )
        dynamic_curve_comfort_accel_gate = float(np.clip(accel_headroom, 0.0, 1.0))
        jerk_ratio = (
            dynamic_curve_lateral_jerk_est_smoothed_gps
            / max(1e-3, self.dynamic_curve_comfort_lat_jerk_comfort_max_gps)
        )
        if jerk_ratio <= self.dynamic_curve_lat_jerk_soft_start_ratio:
            dynamic_curve_comfort_jerk_penalty = 1.0
        else:
            jerk_penalty_ratio = (jerk_ratio - self.dynamic_curve_lat_jerk_soft_start_ratio) / max(
                1e-3, 1.0 - self.dynamic_curve_lat_jerk_soft_start_ratio
            )
            dynamic_curve_comfort_jerk_penalty = float(
                np.clip(
                    1.0
                    - jerk_penalty_ratio * (1.0 - self.dynamic_curve_lat_jerk_soft_floor_scale),
                    self.dynamic_curve_lat_jerk_soft_floor_scale,
                    1.0,
                )
            )
        # g_lat is the hard comfort guardrail; jerk only soft-damps authority.
        dynamic_curve_comfort_scale = float(
            np.clip(
                dynamic_curve_comfort_accel_gate * dynamic_curve_comfort_jerk_penalty,
                0.0,
                1.0,
            )
        )
        dynamic_curve_rate_boost_cap_effective = float(
            self.dynamic_curve_rate_boost_max
            * dynamic_curve_speed_scale
            * dynamic_curve_comfort_scale
            * dynamic_curve_context_scale
        )
        dynamic_curve_jerk_boost_cap_effective = float(
            1.0
            + (self.dynamic_curve_jerk_boost_max_factor - 1.0)
            * dynamic_curve_speed_scale
            * dynamic_curve_comfort_scale
            * dynamic_curve_context_scale
        )
        dynamic_curve_hard_clip_boost_cap_effective = float(
            self.dynamic_curve_hard_clip_boost_max
            * dynamic_curve_speed_scale
            * dynamic_curve_comfort_scale
            * dynamic_curve_context_scale
        )
        if self.dynamic_curve_authority_enabled and (curve_at_car or dynamic_curve_precurve_active):
            dynamic_curve_authority_active = True
            raw_deficit = dynamic_curve_rate_request_delta - max_steering_rate
            deficit = max(0.0, raw_deficit - self.dynamic_curve_rate_deficit_deadband)
            dynamic_curve_rate_deficit = float(deficit)
            if dynamic_curve_rate_deficit >= self.curve_entry_schedule_fallback_rate_deficit_min:
                self._dynamic_authority_deficit_streak += 1
            else:
                self._dynamic_authority_deficit_streak = 0
            if dynamic_curve_rate_deficit >= self.curve_commit_mode_dynamic_deficit_min:
                self._dynamic_commit_deficit_streak += 1
            else:
                self._dynamic_commit_deficit_streak = 0
            dynamic_curve_rate_boost = float(
                np.clip(
                    dynamic_curve_rate_deficit * self.dynamic_curve_rate_boost_gain,
                    0.0,
                    dynamic_curve_rate_boost_cap_effective,
                )
            )
            max_steering_rate += dynamic_curve_rate_boost
        else:
            self._dynamic_authority_deficit_streak = 0
            self._dynamic_commit_deficit_streak = 0
        entry_phase_active = bool(entry_phase_active_pre)
        if (
            self.dynamic_curve_entry_governor_enabled
            and dynamic_curve_authority_active
            and entry_phase_active
        ):
            if curve_entry_schedule_active:
                phase_weight = 1.0
            elif self._curve_commit_mode_frames_remaining > 0:
                phase_weight = 0.75
            elif dynamic_curve_entry_governor_anticipatory_active:
                phase_weight = self.dynamic_curve_entry_governor_upcoming_phase_weight
            else:
                phase_weight = 0.6
            demand_ratio = float(
                np.clip(
                    (dynamic_curve_rate_request_delta / max(max_steering_rate, 1e-3)) - 1.0,
                    0.0,
                    1.0,
                )
            )
            base_scale = 1.0 + (
                self.dynamic_curve_entry_governor_gain
                * phase_weight
                * demand_ratio
                * dynamic_curve_comfort_scale
            )
            dynamic_curve_entry_governor_scale = float(
                np.clip(
                    base_scale,
                    1.0,
                    self.dynamic_curve_entry_governor_max_scale,
                )
            )
            if using_stale_perception:
                dynamic_curve_entry_governor_scale = max(
                    dynamic_curve_entry_governor_scale,
                    self.dynamic_curve_entry_governor_stale_floor_scale,
                )
            dynamic_curve_entry_governor_active = dynamic_curve_entry_governor_scale > 1.0 + 1e-6
            dynamic_curve_entry_governor_rate_scale = dynamic_curve_entry_governor_scale
            dynamic_curve_entry_governor_jerk_scale = dynamic_curve_entry_governor_scale
            max_steering_rate *= dynamic_curve_entry_governor_rate_scale
        if (
            not single_owner_entry_phase
            and
            self.curve_commit_mode_enabled
            and self.curve_commit_mode_retrigger_on_dynamic_deficit
            and curve_at_car
            and self._curve_commit_mode_frames_remaining <= 0
            and self._curve_entry_schedule_frames_remaining <= 0
            and self._curve_commit_retrigger_cooldown_remaining <= 0
            and self._dynamic_commit_deficit_streak >= self.curve_commit_mode_dynamic_deficit_frames
        ):
            self._curve_commit_mode_frames_remaining = self.curve_commit_mode_max_frames
            curve_commit_mode_triggered = self._curve_commit_mode_frames_remaining > 0
            self._curve_commit_retrigger_cooldown_remaining = (
                self.curve_commit_mode_retrigger_cooldown_frames
            )
            self._dynamic_commit_deficit_streak = 0
            if self._curve_commit_mode_frames_remaining > 0:
                curve_commit_mode_active = True
                if not entry_governor_exclusive_active_pre:
                    max_steering_rate = max(max_steering_rate, self.curve_commit_mode_min_rate)
        if curve_unwind_active:
            max_steering_rate *= curve_unwind_rate_scale
        steering_rate_limit_after_floor = float(max_steering_rate)

        # Bounded curve-entry authority assist:
        # temporarily boost limits during curve turn-in when error is worsening and limiter pressure exists.
        if self._curve_entry_assist_rearm_remaining > 0:
            self._curve_entry_assist_rearm_remaining -= 1
        if self._curve_entry_assist_frames_remaining > 0:
            self._curve_entry_assist_frames_remaining -= 1
        if (
            self.curve_entry_assist_enabled
            and not entry_governor_exclusive_active_pre
            and not single_owner_entry_phase
        ):
            raw_rate_request = abs(steering_before_limits - self.last_steering)
            error_rising = error_magnitude > (self._last_error_magnitude + 0.02)
            curve_entry_candidate = (
                curve_at_car
                and curve_metric_abs >= self.curve_entry_assist_curvature_min
                and abs(heading_error) >= self.curve_entry_assist_heading_error_min
                and error_magnitude >= self.curve_entry_assist_error_min
                and error_rising
            )
            limiter_pressure = raw_rate_request > max_steering_rate
            watchdog_trip = raw_rate_request > self.curve_entry_assist_watchdog_rate_delta_max
            if watchdog_trip:
                self._curve_entry_assist_frames_remaining = 0
                self._curve_entry_assist_rearm_remaining = self.curve_entry_assist_rearm_frames
            elif (
                self._curve_entry_assist_rearm_remaining <= 0
                and self._curve_entry_assist_frames_remaining <= 0
                and curve_entry_candidate
                and limiter_pressure
            ):
                self._curve_entry_assist_frames_remaining = self.curve_entry_assist_max_frames
                curve_entry_assist_triggered = True
            if self._curve_entry_assist_frames_remaining > 0:
                curve_entry_assist_active = True
                if not dynamic_curve_entry_governor_active:
                    max_steering_rate *= self.curve_entry_assist_rate_boost
            curve_entry_assist_rearm_remaining = int(self._curve_entry_assist_rearm_remaining)

        # Sign-flip override: allow corrections to reverse quickly when error flips sign.
        if not hasattr(self, '_sign_flip_override_frames'):
            self._sign_flip_override_frames = 0
        sign_flip_override_active = False
        sign_flip_triggered = False
        sign_flip_trigger_error = abs(total_error_scaled)
        if (
            abs(total_error_scaled) >= self.straight_sign_flip_error_threshold
            and abs(self.last_steering) > 0.02
            and np.sign(total_error_scaled) != np.sign(self.last_steering)
        ):
            sign_flip_triggered = True
            self._sign_flip_override_frames = max(
                self._sign_flip_override_frames,
                int(self.straight_sign_flip_frames),
            )
        if self._sign_flip_override_frames > 0:
            sign_flip_override_active = True
            max_steering_rate = max(max_steering_rate, self.straight_sign_flip_rate)
            self._sign_flip_override_frames -= 1
        sign_flip_frames_remaining = int(self._sign_flip_override_frames)
        steering_rate_limit_effective = float(max_steering_rate)

        # FIXED: Increased steering rate limit for better curve tracking
        # Required steering for 50m curve is only 0.095 normalized, but need to reach it quickly
        # Increased to 0.2 per frame (6.0 per second at 30 FPS) for more responsive steering
        # This allows reaching required steering (0.095) in ~0.5 frames instead of ~0.6 frames
        # FIXED: On first frame, allow larger initial steering to respond to initial state
        # This is important when starting on a curve - we need immediate steering response
        if hasattr(self, '_steering_rate_limit_active'):
            rate_in = steering_before_limits
            steering_change = steering_before_limits - self.last_steering
            steering_rate_limit_requested_delta = abs(steering_change)
            steering_change = np.clip(steering_change, -max_steering_rate, max_steering_rate)
            steering_before_limits = self.last_steering + steering_change
            steering_rate_limited_delta = abs(steering_before_limits - rate_in)
            steering_rate_limited_active = steering_rate_limited_delta > 1e-6
        else:
            # First frame - allow larger initial steering (up to 0.5) to respond to initial state
            # This is critical when starting on a curve where we need immediate steering
            max_initial_steering_rate = 0.25
            rate_in = steering_before_limits
            steering_change = steering_before_limits - self.last_steering
            steering_rate_limit_requested_delta = abs(steering_change)
            if abs(steering_change) > max_initial_steering_rate:
                steering_change = np.clip(steering_change, -max_initial_steering_rate, max_initial_steering_rate)
                steering_before_limits = self.last_steering + steering_change
            steering_rate_limited_delta = abs(steering_before_limits - rate_in)
            steering_rate_limited_active = steering_rate_limited_delta > 1e-6
            # Activate normal rate limiting for next frame
            self._steering_rate_limit_active = True
        steering_post_rate_limit = steering_before_limits
        steering_rate_limit_margin = steering_rate_limit_effective - steering_rate_limit_requested_delta
        steering_rate_limit_unlock_delta_needed = max(
            0.0, steering_rate_limit_requested_delta - steering_rate_limit_effective
        )
        pre_rate_transfer = abs(steering_post_rate_limit) / max(abs(steering_pre_rate_limit), 1e-6)
        error_recovering_schedule = (
            error_magnitude + self.curve_entry_schedule_handoff_error_fall
        ) <= self._last_error_magnitude
        error_recovering_commit = (
            error_magnitude + self.curve_commit_mode_error_fall
        ) <= self._last_error_magnitude
        if self._curve_entry_schedule_frames_remaining > 0 and not single_owner_entry_phase:
            hold_complete = self._curve_entry_schedule_elapsed_frames >= self.curve_entry_schedule_min_hold_frames
            progress_complete = (
                current_curve_progress_ratio is None
                or current_curve_progress_ratio >= self.curve_entry_schedule_min_curve_progress_ratio
            )
            if hold_complete and progress_complete and (
                pre_rate_transfer >= self.curve_entry_schedule_handoff_transfer_ratio
                or error_recovering_schedule
            ):
                self._curve_entry_schedule_frames_remaining = 0
                self._curve_entry_schedule_elapsed_frames = 0
                curve_entry_schedule_handoff_triggered = True
                self._curve_commit_handoff_streak = 0
                if self.curve_commit_mode_enabled and not is_straight:
                    self._curve_commit_mode_frames_remaining = self.curve_commit_mode_max_frames
                    curve_commit_mode_triggered = self._curve_commit_mode_frames_remaining > 0
            else:
                self._curve_entry_schedule_frames_remaining -= 1
            curve_entry_schedule_frames_remaining = int(
                max(0, self._curve_entry_schedule_frames_remaining)
            )
        if self._curve_commit_mode_frames_remaining > 0 and not single_owner_entry_phase:
            commit_handoff_candidate = (
                pre_rate_transfer >= self.curve_commit_mode_transfer_ratio_target
                and error_recovering_commit
            )
            if commit_handoff_candidate:
                self._curve_commit_handoff_streak += 1
            else:
                self._curve_commit_handoff_streak = 0
            if self._curve_commit_handoff_streak >= self.curve_commit_mode_exit_consecutive_frames:
                self._curve_commit_mode_frames_remaining = 0
                curve_commit_mode_handoff_triggered = True
                self._curve_commit_handoff_streak = 0
            else:
                self._curve_commit_mode_frames_remaining -= 1
            curve_commit_mode_frames_remaining = int(
                max(0, self._curve_commit_mode_frames_remaining)
            )

        # Curvature-aware jerk relaxation: keep straight stability, allow faster turn-in on curves.
        jerk_curve_scale = 1.0
        if curve_metric_abs <= self.steering_jerk_curve_min:
            jerk_curve_scale = 1.0
        elif curve_metric_abs >= self.steering_jerk_curve_max:
            jerk_curve_scale = self.steering_jerk_curve_scale_max
        else:
            jerk_ratio = (
                (curve_metric_abs - self.steering_jerk_curve_min)
                / (self.steering_jerk_curve_max - self.steering_jerk_curve_min)
            )
            jerk_curve_scale = 1.0 + jerk_ratio * (self.steering_jerk_curve_scale_max - 1.0)
        effective_steering_jerk_limit = self.steering_jerk_limit * jerk_curve_scale
        if curve_entry_assist_active:
            if not dynamic_curve_entry_governor_active:
                effective_steering_jerk_limit *= self.curve_entry_assist_jerk_boost
        if dynamic_curve_entry_governor_active:
            effective_steering_jerk_limit *= dynamic_curve_entry_governor_jerk_scale
        if self._curve_entry_schedule_frames_remaining > 0:
            if not entry_governor_exclusive_active_pre:
                effective_steering_jerk_limit = max(
                    effective_steering_jerk_limit,
                    self.curve_entry_schedule_min_jerk,
                )
        if self._curve_commit_mode_frames_remaining > 0:
            if not entry_governor_exclusive_active_pre:
                effective_steering_jerk_limit = max(
                    effective_steering_jerk_limit,
                    self.curve_commit_mode_min_jerk,
                )
        if single_owner_entry_phase:
            effective_steering_jerk_limit = max(
                effective_steering_jerk_limit,
                self.dynamic_curve_single_owner_min_jerk,
            )
        if dynamic_curve_authority_active:
            dynamic_curve_jerk_boost_factor = float(
                np.clip(
                    1.0 + dynamic_curve_rate_deficit * self.dynamic_curve_jerk_boost_gain,
                    1.0,
                    dynamic_curve_jerk_boost_cap_effective,
                )
            )
            effective_steering_jerk_limit *= dynamic_curve_jerk_boost_factor
        if curve_unwind_active:
            effective_steering_jerk_limit *= curve_unwind_jerk_scale

        # Optional jerk limit: constrain change in steering rate per frame.
        if effective_steering_jerk_limit > 0.0 and dt > 0.0 and not sign_flip_override_active:
            jerk_in = steering_before_limits
            if not hasattr(self, 'last_steering_rate'):
                self.last_steering_rate = 0.0
            steering_change = steering_before_limits - self.last_steering
            current_rate = steering_change / dt
            steering_jerk_limit_requested_rate_delta = abs(current_rate - self.last_steering_rate)
            max_rate_delta = effective_steering_jerk_limit * dt
            steering_jerk_limit_allowed_rate_delta = max_rate_delta
            rate_delta = np.clip(
                current_rate - self.last_steering_rate,
                -max_rate_delta,
                max_rate_delta,
            )
            limited_rate = self.last_steering_rate + rate_delta
            steering_before_limits = self.last_steering + limited_rate * dt
            self.last_steering_rate = limited_rate
            steering_jerk_limited_delta = abs(steering_before_limits - jerk_in)
            steering_jerk_limited_active = steering_jerk_limited_delta > 1e-6
        elif dt > 0.0:
            steering_change = steering_before_limits - self.last_steering
            current_rate = steering_change / dt
            prev_rate = getattr(self, 'last_steering_rate', 0.0)
            steering_jerk_limit_requested_rate_delta = abs(current_rate - prev_rate)
            steering_jerk_limit_allowed_rate_delta = (
                effective_steering_jerk_limit * dt if effective_steering_jerk_limit > 0.0 else float('inf')
            )
            self.last_steering_rate = (steering_before_limits - self.last_steering) / dt
        steering_post_jerk_limit = steering_before_limits
        if np.isfinite(steering_jerk_limit_allowed_rate_delta):
            steering_jerk_limit_margin = (
                steering_jerk_limit_allowed_rate_delta - steering_jerk_limit_requested_rate_delta
            )
            steering_jerk_limit_unlock_rate_delta_needed = max(
                0.0,
                steering_jerk_limit_requested_rate_delta - steering_jerk_limit_allowed_rate_delta,
            )
        else:
            steering_jerk_limit_margin = 0.0
            steering_jerk_limit_unlock_rate_delta_needed = 0.0

        if sign_flip_override_active:
            # Ensure we are actually moving toward reversing sign when error flips.
            if np.sign(steering_before_limits) == np.sign(self.last_steering):
                if np.sign(total_error_scaled) != np.sign(self.last_steering):
                    min_step = min(abs(self.last_steering), self.straight_sign_flip_rate)
                    steering_before_limits = (
                        self.last_steering - np.sign(self.last_steering) * min_step
                    )
        steering_post_sign_flip = steering_before_limits

        # Apply additional smoothing to prevent oscillation
        clip_in = steering_before_limits
        if dynamic_curve_authority_active:
            clip_overflow = max(0.0, abs(clip_in) - self.max_steering)
            dynamic_curve_hard_clip_boost = float(
                np.clip(
                    max(
                        dynamic_curve_rate_deficit * self.dynamic_curve_hard_clip_boost_gain,
                        clip_overflow,
                    ),
                    0.0,
                    dynamic_curve_hard_clip_boost_cap_effective,
                )
            )
        dynamic_curve_hard_clip_limit_effective = float(
            np.clip(self.max_steering + dynamic_curve_hard_clip_boost, 0.0, 1.0)
        )
        steering = np.clip(
            steering_before_limits,
            -dynamic_curve_hard_clip_limit_effective,
            dynamic_curve_hard_clip_limit_effective,
        )
        steering_hard_clip_delta = abs(steering - clip_in)
        steering_hard_clip_active = steering_hard_clip_delta > 1e-6
        steering_post_hard_clip = steering
        self.last_steering = steering
        if self.steering_smoothing_alpha is not None:
            smoothing_in = steering
            if sign_flip_override_active:
                # Skip smoothing when reversing sign on straights to avoid stickiness.
                self.smoothed_steering = steering
            else:
                if self.smoothed_steering is None:
                    self.smoothed_steering = steering
                else:
                    alpha = np.clip(self.steering_smoothing_alpha, 0.0, 1.0)
                    self.smoothed_steering = (alpha * steering) + ((1.0 - alpha) * self.smoothed_steering)
            steering = self.smoothed_steering
            steering_smoothing_delta = abs(steering - smoothing_in)
            steering_smoothing_active = steering_smoothing_delta > 1e-6
        steering_post_smoothing = steering
        # Explicit attribution signals for control triage.
        pre_abs = abs(steering_pre_rate_limit)
        final_abs = abs(steering_post_smoothing)
        steering_authority_gap = max(pre_abs - final_abs, 0.0)
        steering_transfer_ratio = np.clip(
            final_abs / max(pre_abs, 1e-6),
            0.0,
            1.0,
        )
        if steering_rate_limited_delta > 1e-6:
            steering_first_limiter_stage_code = 1.0
        elif steering_jerk_limited_delta > 1e-6:
            steering_first_limiter_stage_code = 2.0
        elif steering_hard_clip_delta > 1e-6:
            steering_first_limiter_stage_code = 3.0
        elif steering_smoothing_delta > 1e-6:
            steering_first_limiter_stage_code = 4.0
        else:
            steering_first_limiter_stage_code = 0.0

        # Straight-away adaptive tuning (deadband + smoothing)
        if is_straight:
            steering_sign = np.sign(steering) if abs(steering) > 0.02 else 0
            if self._last_straight_sign != 0 and steering_sign != 0 and steering_sign != self._last_straight_sign:
                self._straight_sign_changes += 1
            if steering_sign != 0:
                self._last_straight_sign = steering_sign
            self._straight_frames += 1

            if self._straight_frames >= self.straight_window:
                self.straight_oscillation_rate = self._straight_sign_changes / max(1, self._straight_frames - 1)
                if self.straight_oscillation_rate > self.straight_oscillation_high:
                    # Too much weave on straight: increase deadband and smoothing
                    self.deadband = min(self.deadband_max, self.deadband + 0.005)
                    self.error_smoothing_alpha = min(self.smoothing_max, self.error_smoothing_alpha + 0.02)
                    self.straight_steering_deadband = min(
                        self.straight_steering_deadband_max,
                        self.straight_steering_deadband + 0.005
                    )
                elif self.straight_oscillation_rate < self.straight_oscillation_low:
                    # Stable: relax toward baseline
                    if self.deadband > self.base_deadband:
                        self.deadband = max(self.base_deadband, self.deadband - 0.003)
                    if self.error_smoothing_alpha > self.base_error_smoothing_alpha:
                        self.error_smoothing_alpha = max(self.base_error_smoothing_alpha, self.error_smoothing_alpha - 0.01)
                    if self.straight_steering_deadband > self.base_straight_steering_deadband:
                        self.straight_steering_deadband = max(
                            self.base_straight_steering_deadband,
                            self.straight_steering_deadband - 0.003
                        )
                # Reset window counters
                self._straight_frames = 0
                self._straight_sign_changes = 0
        else:
            # Off straight: decay toward baseline
            if self.deadband > self.base_deadband:
                self.deadband = max(self.base_deadband, self.deadband - 0.002)
            elif self.deadband < self.base_deadband:
                self.deadband = min(self.base_deadband, self.deadband + 0.001)
            if self.error_smoothing_alpha > self.base_error_smoothing_alpha:
                self.error_smoothing_alpha = max(self.base_error_smoothing_alpha, self.error_smoothing_alpha - 0.005)
            elif self.error_smoothing_alpha < self.base_error_smoothing_alpha:
                self.error_smoothing_alpha = min(self.base_error_smoothing_alpha, self.error_smoothing_alpha + 0.003)
            if self.straight_steering_deadband > self.base_straight_steering_deadband:
                self.straight_steering_deadband = max(
                    self.base_straight_steering_deadband,
                    self.straight_steering_deadband - 0.004
                )
            elif self.straight_steering_deadband < self.base_straight_steering_deadband:
                self.straight_steering_deadband = min(
                    self.base_straight_steering_deadband,
                    self.straight_steering_deadband + 0.002
                )
            self._straight_frames = 0
            self._straight_sign_changes = 0
            self._last_straight_sign = 0
        
        self._last_error_magnitude = error_magnitude
        self._prev_is_straight = is_straight
        self._prev_entry_phase_active = entry_phase_active
        self._prev_curve_upcoming = curve_upcoming
        self._prev_curve_at_car = curve_at_car
        if return_metadata:
            return {
                'steering': steering,
                'steering_before_limits': steering_before_limits,
                'feedforward_steering': feedforward_steering,
                'feedback_steering': feedback_steering,
                'lateral_error': stored_lateral_error,
                'heading_error': stored_heading_error,
                'total_error': stored_total_error,
                'total_error_scaled': total_error_scaled,
                'pid_integral': pid_integral,
                'pid_derivative': pid_derivative,
                'path_curvature_input': path_curvature,
                'path_curvature_raw': raw_ref_curvature,
                'curve_feedforward_gain_scheduled': curve_gain_scheduled,
                'curve_feedforward_scale': curve_feedforward_scale,
                'curve_feedforward_curvature_used': self.smoothed_path_curvature,
                'curve_curvature_hold_active': curvature_hold_active,
                'curve_curvature_hold_frames_remaining': self._curvature_hold_frames_remaining,
                'steering_pre_rate_limit': steering_pre_rate_limit,
                'steering_post_rate_limit': steering_post_rate_limit,
                'steering_post_jerk_limit': steering_post_jerk_limit,
                'steering_post_sign_flip': steering_post_sign_flip,
                'steering_post_hard_clip': steering_post_hard_clip,
                'steering_post_smoothing': steering_post_smoothing,
                'steering_rate_limited_active': steering_rate_limited_active,
                'steering_jerk_limited_active': steering_jerk_limited_active,
                'steering_hard_clip_active': steering_hard_clip_active,
                'steering_smoothing_active': steering_smoothing_active,
                'steering_rate_limited_delta': steering_rate_limited_delta,
                'steering_jerk_limited_delta': steering_jerk_limited_delta,
                'steering_hard_clip_delta': steering_hard_clip_delta,
                'steering_smoothing_delta': steering_smoothing_delta,
                'steering_rate_limit_base_from_error': steering_rate_limit_base_from_error,
                'steering_rate_limit_curve_scale': steering_rate_limit_curve_scale,
                'steering_rate_limit_curve_metric_abs': steering_rate_limit_curve_metric_abs,
                'steering_rate_limit_curve_metric_source': steering_rate_limit_curve_metric_source,
                'steering_rate_limit_curve_min': steering_rate_limit_curve_min,
                'steering_rate_limit_curve_max': steering_rate_limit_curve_max,
                'steering_rate_limit_scale_min': steering_rate_limit_scale_min,
                'steering_rate_limit_curve_regime_code': steering_rate_limit_curve_regime_code,
                'steering_rate_limit_after_curve': steering_rate_limit_after_curve,
                'steering_rate_limit_after_floor': steering_rate_limit_after_floor,
                'steering_rate_limit_effective': steering_rate_limit_effective,
                'steering_rate_limit_requested_delta': steering_rate_limit_requested_delta,
                'steering_rate_limit_margin': steering_rate_limit_margin,
                'steering_rate_limit_unlock_delta_needed': steering_rate_limit_unlock_delta_needed,
                'curve_entry_assist_active': curve_entry_assist_active,
                'curve_entry_assist_triggered': curve_entry_assist_triggered,
                'curve_entry_assist_rearm_frames_remaining': curve_entry_assist_rearm_remaining,
                'dynamic_curve_authority_active': dynamic_curve_authority_active,
                'dynamic_curve_rate_request_delta': dynamic_curve_rate_request_delta,
                'dynamic_curve_rate_deficit': dynamic_curve_rate_deficit,
                'dynamic_curve_rate_boost': dynamic_curve_rate_boost,
                'dynamic_curve_jerk_boost_factor': dynamic_curve_jerk_boost_factor,
                'dynamic_curve_lateral_accel_est_g': dynamic_curve_lateral_accel_est_g,
                'dynamic_curve_lateral_jerk_est_gps': dynamic_curve_lateral_jerk_est_gps,
                'dynamic_curve_lateral_jerk_est_smoothed_gps': (
                    dynamic_curve_lateral_jerk_est_smoothed_gps
                ),
                'dynamic_curve_speed_scale': dynamic_curve_speed_scale,
                'dynamic_curve_comfort_scale': dynamic_curve_comfort_scale,
                'dynamic_curve_comfort_accel_gate': dynamic_curve_comfort_accel_gate,
                'dynamic_curve_comfort_jerk_penalty': dynamic_curve_comfort_jerk_penalty,
                'dynamic_curve_rate_boost_cap_effective': dynamic_curve_rate_boost_cap_effective,
                'dynamic_curve_jerk_boost_cap_effective': dynamic_curve_jerk_boost_cap_effective,
                'dynamic_curve_hard_clip_boost': dynamic_curve_hard_clip_boost,
                'dynamic_curve_hard_clip_boost_cap_effective': (
                    dynamic_curve_hard_clip_boost_cap_effective
                ),
                'dynamic_curve_hard_clip_limit_effective': (
                    dynamic_curve_hard_clip_limit_effective
                ),
                'dynamic_curve_entry_governor_active': dynamic_curve_entry_governor_active,
                'dynamic_curve_entry_governor_scale': dynamic_curve_entry_governor_scale,
                'dynamic_curve_entry_governor_rate_scale': dynamic_curve_entry_governor_rate_scale,
                'dynamic_curve_entry_governor_jerk_scale': dynamic_curve_entry_governor_jerk_scale,
                'dynamic_curve_authority_deficit_streak': int(self._dynamic_authority_deficit_streak),
                'dynamic_curve_commit_deficit_streak': int(self._dynamic_commit_deficit_streak),
                'dynamic_curve_comfort_lat_accel_comfort_max_g': (
                    self.dynamic_curve_comfort_lat_accel_comfort_max_g
                ),
                'dynamic_curve_comfort_lat_accel_peak_max_g': (
                    self.dynamic_curve_comfort_lat_accel_peak_max_g
                ),
                'dynamic_curve_comfort_lat_jerk_comfort_max_gps': (
                    self.dynamic_curve_comfort_lat_jerk_comfort_max_gps
                ),
                'dynamic_curve_lat_jerk_smoothing_alpha': (
                    self.dynamic_curve_lat_jerk_smoothing_alpha
                ),
                'dynamic_curve_lat_jerk_soft_start_ratio': (
                    self.dynamic_curve_lat_jerk_soft_start_ratio
                ),
                'dynamic_curve_lat_jerk_soft_floor_scale': (
                    self.dynamic_curve_lat_jerk_soft_floor_scale
                ),
                'dynamic_curve_hard_clip_boost_gain': self.dynamic_curve_hard_clip_boost_gain,
                'dynamic_curve_hard_clip_boost_max': self.dynamic_curve_hard_clip_boost_max,
                'dynamic_curve_entry_governor_enabled': self.dynamic_curve_entry_governor_enabled,
                'dynamic_curve_entry_governor_gain': self.dynamic_curve_entry_governor_gain,
                'dynamic_curve_entry_governor_max_scale': self.dynamic_curve_entry_governor_max_scale,
                'dynamic_curve_entry_governor_stale_floor_scale': (
                    self.dynamic_curve_entry_governor_stale_floor_scale
                ),
                'dynamic_curve_entry_governor_exclusive_mode': (
                    self.dynamic_curve_entry_governor_exclusive_mode
                ),
                'dynamic_curve_entry_governor_anticipatory_enabled': (
                    self.dynamic_curve_entry_governor_anticipatory_enabled
                ),
                'dynamic_curve_entry_governor_upcoming_phase_weight': (
                    self.dynamic_curve_entry_governor_upcoming_phase_weight
                ),
                'dynamic_curve_entry_governor_anticipatory_active': (
                    dynamic_curve_entry_governor_anticipatory_active
                ),
                'dynamic_curve_authority_precurve_enabled': (
                    self.dynamic_curve_authority_precurve_enabled
                ),
                'dynamic_curve_authority_precurve_scale': (
                    self.dynamic_curve_authority_precurve_scale
                ),
                'dynamic_curve_single_owner_mode': self.dynamic_curve_single_owner_mode,
                'dynamic_curve_single_owner_min_rate': self.dynamic_curve_single_owner_min_rate,
                'dynamic_curve_single_owner_min_jerk': self.dynamic_curve_single_owner_min_jerk,
                'dynamic_curve_single_owner_active': single_owner_entry_phase,
                'dynamic_curve_precurve_active': dynamic_curve_precurve_active,
                'dynamic_curve_context_scale': dynamic_curve_context_scale,
                'curve_entry_schedule_active': curve_entry_schedule_active,
                'curve_entry_schedule_triggered': curve_entry_schedule_triggered,
                'curve_entry_schedule_handoff_triggered': curve_entry_schedule_handoff_triggered,
                'curve_entry_schedule_frames_remaining': curve_entry_schedule_frames_remaining,
                'curve_entry_schedule_min_hold_frames': self.curve_entry_schedule_min_hold_frames,
                'curve_entry_schedule_min_curve_progress_ratio': (
                    self.curve_entry_schedule_min_curve_progress_ratio
                ),
                'curve_entry_schedule_fallback_only_when_dynamic': (
                    self.curve_entry_schedule_fallback_only_when_dynamic
                ),
                'curve_entry_schedule_fallback_deficit_frames': (
                    self.curve_entry_schedule_fallback_deficit_frames
                ),
                'curve_entry_schedule_fallback_rate_deficit_min': (
                    self.curve_entry_schedule_fallback_rate_deficit_min
                ),
                'curve_entry_schedule_fallback_rearm_cooldown_frames': (
                    self.curve_entry_schedule_fallback_rearm_cooldown_frames
                ),
                'curve_entry_schedule_fallback_cooldown_remaining': (
                    self._curve_entry_schedule_fallback_cooldown_remaining
                ),
                'curve_commit_mode_active': curve_commit_mode_active,
                'curve_commit_mode_triggered': curve_commit_mode_triggered,
                'curve_commit_mode_handoff_triggered': curve_commit_mode_handoff_triggered,
                'curve_commit_mode_frames_remaining': curve_commit_mode_frames_remaining,
                'curve_commit_mode_exit_handoff_streak': int(self._curve_commit_handoff_streak),
                'curve_commit_mode_exit_consecutive_frames': (
                    self.curve_commit_mode_exit_consecutive_frames
                ),
                'curve_commit_mode_retrigger_on_dynamic_deficit': (
                    self.curve_commit_mode_retrigger_on_dynamic_deficit
                ),
                'curve_commit_mode_dynamic_deficit_frames': (
                    self.curve_commit_mode_dynamic_deficit_frames
                ),
                'curve_commit_mode_dynamic_deficit_min': (
                    self.curve_commit_mode_dynamic_deficit_min
                ),
                'curve_commit_mode_retrigger_cooldown_frames': (
                    self.curve_commit_mode_retrigger_cooldown_frames
                ),
                'curve_commit_mode_retrigger_cooldown_remaining': (
                    self._curve_commit_retrigger_cooldown_remaining
                ),
                'steering_jerk_limit_effective': effective_steering_jerk_limit,
                'steering_jerk_curve_scale': jerk_curve_scale,
                'steering_jerk_limit_requested_rate_delta': steering_jerk_limit_requested_rate_delta,
                'steering_jerk_limit_allowed_rate_delta': steering_jerk_limit_allowed_rate_delta,
                'steering_jerk_limit_margin': steering_jerk_limit_margin,
                'steering_jerk_limit_unlock_rate_delta_needed': steering_jerk_limit_unlock_rate_delta_needed,
                'steering_authority_gap': steering_authority_gap,
                'steering_transfer_ratio': steering_transfer_ratio,
                'steering_first_limiter_stage_code': steering_first_limiter_stage_code,
                'curve_unwind_active': curve_unwind_active,
                'curve_unwind_frames_remaining': curve_unwind_frames_remaining,
                'curve_unwind_progress': curve_unwind_progress,
                'curve_unwind_rate_scale': curve_unwind_rate_scale,
                'curve_unwind_jerk_scale': curve_unwind_jerk_scale,
                'curve_unwind_integral_decay_applied': curve_unwind_integral_decay_applied,
                'turn_feasibility_active': turn_feasibility_active,
                'turn_feasibility_infeasible': turn_feasibility_infeasible,
                'turn_feasibility_curvature_abs': turn_feasibility_curvature_abs,
                'turn_feasibility_speed_mps': turn_feasibility_speed_mps,
                'turn_feasibility_required_lat_accel_g': turn_feasibility_required_lat_accel_g,
                'turn_feasibility_comfort_limit_g': turn_feasibility_comfort_limit_g,
                'turn_feasibility_peak_limit_g': turn_feasibility_peak_limit_g,
                'turn_feasibility_selected_limit_g': turn_feasibility_selected_limit_g,
                'turn_feasibility_guardband_g': self.turn_feasibility_guardband_g,
                'turn_feasibility_margin_g': turn_feasibility_margin_g,
                'turn_feasibility_speed_limit_mps': turn_feasibility_speed_limit_mps,
                'turn_feasibility_speed_delta_mps': turn_feasibility_speed_delta_mps,
                'turn_feasibility_use_peak_bound': self.turn_feasibility_use_peak_bound,
                'curve_unwind_policy_enabled': self.curve_unwind_policy_enabled,
                'curve_unwind_frames': self.curve_unwind_frames,
                'curve_unwind_rate_scale_start': self.curve_unwind_rate_scale_start,
                'curve_unwind_rate_scale_end': self.curve_unwind_rate_scale_end,
                'curve_unwind_jerk_scale_start': self.curve_unwind_jerk_scale_start,
                'curve_unwind_jerk_scale_end': self.curve_unwind_jerk_scale_end,
                'curve_unwind_integral_decay': self.curve_unwind_integral_decay,
                'speed_gain_final': speed_gain_final,
                'speed_gain_scale': speed_gain_scale,
                'speed_gain_applied': speed_gain_applied,
                'control_mode': self.control_mode,
                'stanley_heading_term': stanley_heading_term,
                'stanley_crosstrack_term': stanley_crosstrack_term,
                'stanley_speed': stanley_speed,
                'pp_alpha': pp_alpha,
                'pp_lookahead_distance': pp_lookahead_distance,
                'pp_geometric_steering': pp_geometric_steering,
                'pp_feedback_steering': pp_feedback_steering_val,
                'pp_ref_jump_clamped': float(pp_ref_jump_clamped),
                'pp_stale_hold_active': float(pp_stale_hold_active),
                'feedback_gain_scheduled': feedback_gain,
                'total_error_scaled': total_error,
                'is_straight': is_straight,
                'is_control_straight_proxy': is_control_straight_proxy,
                'curve_upcoming': curve_upcoming,
                'curve_at_car': curve_at_car,
                'curve_at_car_distance_remaining_m': distance_to_next_curve_start_m,
                'curve_at_car_distance_min_m': self.curve_at_car_distance_min_m,
                'curve_phase_source': curve_phase_source,
                'distance_to_next_curve_start_m': distance_to_next_curve_start_m,
                'current_curve_progress_ratio': current_curve_progress_ratio,
                'curve_upcoming_enter_threshold': self.curve_upcoming_enter_threshold,
                'curve_upcoming_exit_threshold': self.curve_upcoming_exit_threshold,
                'is_road_straight': is_road_straight,
                'road_curvature_valid': road_curvature_valid,
                'road_curvature_abs': road_curvature_abs,
                'road_curvature_source': (
                    raw_ref_curvature_source if raw_ref_curvature_source else 'planner_default'
                ),
                'road_curve_enter_threshold': self.road_curve_enter_threshold,
                'road_curve_exit_threshold': self.road_curve_exit_threshold,
                'road_straight_invalid_hold_frames_remaining': (
                    self._road_straight_invalid_frames_remaining
                ),
                'straight_sign_flip_override_active': sign_flip_override_active,
                'straight_sign_flip_triggered': sign_flip_triggered,
                'straight_sign_flip_trigger_error': sign_flip_trigger_error,
                'straight_sign_flip_frames_remaining': sign_flip_frames_remaining,
                'straight_oscillation_rate': self.straight_oscillation_rate,
                'tuned_deadband': self.deadband,
                'tuned_error_smoothing_alpha': self.error_smoothing_alpha
            }
        return steering
    
    def reset(self):
        """Reset controller."""
        self.pid.reset()
        self.small_error_count = 0
        self.last_error_sign = None
        self._curve_entry_assist_frames_remaining = 0
        self._curve_entry_assist_rearm_remaining = 0
        self._curve_entry_schedule_frames_remaining = 0
        self._curve_entry_schedule_elapsed_frames = 0
        self._curve_entry_schedule_fallback_fired_for_curve = False
        self._curve_entry_schedule_fallback_last_curve_index = None
        self._curve_entry_schedule_fallback_cooldown_remaining = 0
        self._dynamic_authority_deficit_streak = 0
        self._dynamic_commit_deficit_streak = 0
        self._last_lateral_accel_est_g = 0.0
        self._last_lateral_accel_est_initialized = False
        self._smoothed_lateral_jerk_est_gps = 0.0
        self._curve_commit_mode_frames_remaining = 0
        self._curve_commit_retrigger_cooldown_remaining = 0
        self._curve_commit_handoff_streak = 0
        self._curve_unwind_frames_remaining = 0
        self._prev_is_straight = True
        self._prev_entry_phase_active = False
        self._prev_curve_upcoming = False
        self._prev_curve_at_car = False
        self._last_error_magnitude = 0.0
        self._curve_upcoming_state = False
        self._curve_upcoming_on_counter = 0
        self._curve_upcoming_off_counter = 0
        self._curve_at_car_state = False
        self._curve_at_car_distance_remaining = None
        self._road_straight_state = True
        self._road_straight_invalid_frames_remaining = 0
        if hasattr(self, 'last_steering'):
            self.last_steering = 0.0
        if hasattr(self, 'last_steering_rate'):
            self.last_steering_rate = 0.0
        self.smoothed_steering = None

    def _load_track_curve_windows(self, track_name: str) -> None:
        """Load curve windows from tracks/<track_name>.yml."""
        self._track_curve_windows = []
        self._track_total_length_m = None
        track_path = Path(__file__).resolve().parents[1] / "tracks" / f"{track_name}.yml"
        if not track_path.exists():
            return
        try:
            with open(track_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            s = 0.0
            windows: list[tuple[float, float]] = []
            for segment in cfg.get("segments", []) or []:
                seg = segment or {}
                seg_type = str(seg.get("type", "straight")).strip().lower()
                if seg_type == "arc":
                    radius = float(seg.get("radius", 0.0) or 0.0)
                    angle_deg = float(seg.get("angle_deg", seg.get("angle", 0.0)) or 0.0)
                    seg_len = max(0.0, radius * math.radians(abs(angle_deg)))
                    windows.append((s, s + seg_len))
                else:
                    seg_len = max(0.0, float(seg.get("length", 0.0) or 0.0))
                s += seg_len
            if s > 1e-3 and windows:
                self._track_total_length_m = float(s)
                self._track_curve_windows = windows
        except Exception:
            self._track_curve_windows = []
            self._track_total_length_m = None

    def _compute_distance_curve_state(
        self, road_center_reference_t: Optional[float], lookahead_m: float
    ) -> Optional[Dict[str, float]]:
        """Compute curve state from track distance when track windows are available."""
        if (
            not self.curve_phase_use_distance_track
            or self._track_total_length_m is None
            or not self._track_curve_windows
            or road_center_reference_t is None
            or not np.isfinite(float(road_center_reference_t))
        ):
            return None

        total = float(self._track_total_length_m)
        t = float(road_center_reference_t)
        if t > 1.5:
            distance_m = t % total
        else:
            distance_m = (t % 1.0) * total

        in_curve = False
        next_curve_delta = None
        curve_start_m = None
        curve_end_m = None
        current_curve_index = None
        for curve_idx, (start_m, end_m) in enumerate(self._track_curve_windows, start=1):
            if start_m <= distance_m < end_m:
                in_curve = True
                next_curve_delta = 0.0
                curve_start_m = float(start_m)
                curve_end_m = float(end_m)
                current_curve_index = int(curve_idx)
                break
        if not in_curve:
            for start_m, _ in self._track_curve_windows:
                delta = start_m - distance_m
                if delta < 0.0:
                    delta += total
                if next_curve_delta is None or delta < next_curve_delta:
                    next_curve_delta = delta

        if next_curve_delta is None:
            next_curve_delta = 0.0
        upcoming_horizon = max(self.curve_at_car_distance_min_m, float(max(0.0, lookahead_m)))
        curve_upcoming = bool(in_curve or next_curve_delta <= upcoming_horizon)
        curve_at_car = bool(in_curve)
        current_curve_progress_ratio = None
        if in_curve and curve_start_m is not None and curve_end_m is not None:
            curve_len_m = max(1e-6, curve_end_m - curve_start_m)
            current_curve_progress_ratio = float(
                np.clip((distance_m - curve_start_m) / curve_len_m, 0.0, 1.0)
            )
        return {
            "curve_upcoming": float(curve_upcoming),
            "curve_at_car": float(curve_at_car),
            "distance_to_next_curve_start_m": float(next_curve_delta),
            "current_curve_progress_ratio": current_curve_progress_ratio,
            "current_curve_index": current_curve_index if in_curve else None,
        }


class LongitudinalController:
    """
    Longitudinal control (throttle/brake) using PID controller.
    """
    
    def __init__(self, kp: float = 0.3, ki: float = 0.05, kd: float = 0.02,
                 target_speed: float = 8.0, max_speed: float = 10.0,
                 throttle_rate_limit: float = 0.08, brake_rate_limit: float = 0.15,
                 throttle_smoothing_alpha: float = 0.6, speed_smoothing_alpha: float = 0.7,
                 min_throttle_when_accel: float = 0.0, default_dt: float = 1.0 / 30.0,
                 min_throttle_hold: float = 0.12,
                 min_throttle_hold_speed: float = 4.0,
                 max_accel: float = 2.5, max_decel: float = 3.0, max_jerk: float = 2.0,
                 accel_feedforward_gain: float = 1.0, decel_feedforward_gain: float = 1.0,
                 speed_for_jerk_alpha: float = 0.7,
                 jerk_error_min: float = 0.5, jerk_error_max: float = 3.0,
                 max_jerk_min: float = 1.5, max_jerk_max: float = 6.0,
                 jerk_cooldown_frames: int = 0, jerk_cooldown_scale: float = 0.6,
                 speed_error_to_accel_gain: float = 0.5,
                 speed_error_deadband: float = 0.0,
                 speed_error_gain_under: Optional[float] = None,
                 speed_error_gain_over: Optional[float] = None,
                 overspeed_accel_zero_threshold: float = 0.0,
                 overspeed_brake_threshold: float = 0.3,
                 overspeed_brake_min: float = 0.02,
                 overspeed_brake_max: float = 0.2,
                 overspeed_brake_gain: float = 0.2,
                 accel_mode_threshold: float = 0.2,
                 decel_mode_threshold: float = 0.2,
                 speed_error_accel_threshold: float = 0.2,
                 speed_error_brake_threshold: float = -0.2,
                 mode_switch_min_time: float = 0.4,
                 coast_hold_seconds: float = 0.5,
                 coast_throttle_kp: float = 0.1,
                 coast_throttle_max: float = 0.12,
                 straight_throttle_cap: float = 0.45,
                 straight_curvature_threshold: float = 0.003,
                 accel_tracking_enabled: bool = True,
                 accel_tracking_error_scale: float = 1.0,
                 accel_pid_kp: float = 0.4,
                 accel_pid_ki: float = 0.05,
                 accel_pid_kd: float = 0.02,
                 throttle_curve_gamma: float = 1.0,
                 throttle_curve_min: float = 0.0,
                 speed_drag_gain: float = 0.0,
                 accel_target_smoothing_alpha: float = 0.6,
                 continuous_accel_control: bool = False,
                 continuous_accel_deadband: float = 0.05,
                 startup_ramp_seconds: float = 2.5,
                 startup_accel_limit: float = 1.2,
                 startup_speed_threshold: float = 2.0,
                 startup_throttle_cap: float = 0.0,
                 startup_disable_accel_feedforward: bool = True,
                 low_speed_accel_limit: float = 0.0,
                 low_speed_speed_threshold: float = 0.0):
        """
        Initialize longitudinal controller.
        
        Args:
            kp: Proportional gain (reduced for smoother control)
            ki: Integral gain
            kd: Derivative gain
            target_speed: Target speed (m/s)
            max_speed: Maximum allowed speed (m/s) - hard limit
        """
        self.target_speed = target_speed
        self.max_speed = max_speed
        self.last_speed = 0.0  # For speed smoothing
        self.throttle_rate_limit = throttle_rate_limit
        self.brake_rate_limit = brake_rate_limit
        self.throttle_smoothing_alpha = throttle_smoothing_alpha
        self.speed_smoothing_alpha = speed_smoothing_alpha
        self.min_throttle_when_accel = min_throttle_when_accel
        self.min_throttle_hold = min_throttle_hold
        self.min_throttle_hold_speed = min_throttle_hold_speed
        self.default_dt = default_dt
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.max_jerk = max_jerk
        self.jerk_error_min = jerk_error_min
        self.jerk_error_max = jerk_error_max
        self.max_jerk_min = max_jerk_min
        self.max_jerk_max = max_jerk_max
        self.accel_feedforward_gain = accel_feedforward_gain
        self.decel_feedforward_gain = decel_feedforward_gain
        self.speed_for_jerk_alpha = speed_for_jerk_alpha
        self.jerk_cooldown_frames = jerk_cooldown_frames
        self.jerk_cooldown_scale = jerk_cooldown_scale
        self.speed_error_to_accel_gain = speed_error_to_accel_gain
        self.speed_error_deadband = speed_error_deadband
        self.speed_error_gain_under = (
            speed_error_gain_under
            if speed_error_gain_under is not None
            else speed_error_to_accel_gain
        )
        self.speed_error_gain_over = (
            speed_error_gain_over
            if speed_error_gain_over is not None
            else speed_error_to_accel_gain
        )
        self.overspeed_accel_zero_threshold = overspeed_accel_zero_threshold
        self.overspeed_brake_threshold = overspeed_brake_threshold
        self.overspeed_brake_min = overspeed_brake_min
        self.overspeed_brake_max = overspeed_brake_max
        self.overspeed_brake_gain = overspeed_brake_gain
        self.accel_mode_threshold = accel_mode_threshold
        self.decel_mode_threshold = decel_mode_threshold
        self.speed_error_accel_threshold = speed_error_accel_threshold
        self.speed_error_brake_threshold = speed_error_brake_threshold
        self.mode_switch_min_time = mode_switch_min_time
        self.coast_hold_seconds = coast_hold_seconds
        self.coast_throttle_kp = coast_throttle_kp
        self.coast_throttle_max = coast_throttle_max
        self.straight_throttle_cap = straight_throttle_cap
        self.straight_curvature_threshold = straight_curvature_threshold
        self.accel_tracking_enabled = accel_tracking_enabled
        self.accel_tracking_error_scale = accel_tracking_error_scale
        self.accel_pid_kp = accel_pid_kp
        self.accel_pid_ki = accel_pid_ki
        self.accel_pid_kd = accel_pid_kd
        self.throttle_curve_gamma = throttle_curve_gamma
        self.throttle_curve_min = throttle_curve_min
        self.speed_drag_gain = speed_drag_gain
        accel_pid_output_limit = None
        if self.max_accel > 0.0 and self.max_decel > 0.0:
            accel_pid_output_limit = (-self.max_decel, self.max_accel)
        accel_pid_integral_limit = None
        if self.max_accel > 0.0 and self.max_decel > 0.0:
            accel_pid_integral_limit = max(self.max_accel, self.max_decel)
        self.accel_pid = PIDController(
            kp=self.accel_pid_kp,
            ki=self.accel_pid_ki,
            kd=self.accel_pid_kd,
            integral_limit=accel_pid_integral_limit,
            output_limit=accel_pid_output_limit,
        )
        self.accel_target_smoothing_alpha = accel_target_smoothing_alpha
        self.continuous_accel_control = continuous_accel_control
        self.continuous_accel_deadband = continuous_accel_deadband
        self.startup_ramp_seconds = startup_ramp_seconds
        self.startup_accel_limit = startup_accel_limit
        self.startup_speed_threshold = startup_speed_threshold
        self.startup_throttle_cap = startup_throttle_cap
        self.startup_disable_accel_feedforward = startup_disable_accel_feedforward
        self.low_speed_accel_limit = low_speed_accel_limit
        self.low_speed_speed_threshold = low_speed_speed_threshold
        self.startup_elapsed = 0.0
        self.last_throttle = 0.0
        self.last_brake = 0.0
        self.last_throttle_before_limits = 0.0
        self.last_brake_before_limits = 0.0
        self.last_accel_feedforward = 0.0
        self.last_brake_feedforward = 0.0
        self.last_raw_speed = None
        self.last_filtered_speed = None
        self.last_accel = None
        self.last_jerk = None
        self.last_accel_capped = False
        self.last_jerk_capped = False
        self.last_accel_cmd = 0.0
        self.jerk_cooldown_remaining = 0
        self.longitudinal_mode = "coast"
        self.mode_time = self.mode_switch_min_time
        self.coast_hold_remaining = 0.0
        self.smoothed_desired_accel = 0.0
        self.pid_throttle = PIDController(
            kp=kp,
            ki=ki,
            kd=kd,
            integral_limit=0.8,  # Reduced to prevent windup
            output_limit=(0.0, 1.0)  # Throttle range
        )
        self.pid_brake = PIDController(
            kp=kp * 3.0,  # More aggressive braking
            ki=ki * 0.3,
            kd=kd * 3.0,
            integral_limit=1.0,
            output_limit=(0.0, 1.0)  # Brake range
        )
    
    def compute_control(
        self,
        current_speed: float,
        reference_velocity: Optional[float] = None,
        dt: Optional[float] = None,
        reference_accel: Optional[float] = None,
        current_curvature: Optional[float] = None,
        min_speed_floor: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Compute throttle and brake commands with speed smoothing and limiting.
        
        Args:
            current_speed: Current vehicle speed (m/s)
            reference_velocity: Desired velocity from trajectory (m/s)
        
        Returns:
            Tuple of (throttle, brake) commands (0.0 to 1.0)
        """
        if dt is None or dt <= 0.0:
            dt = self.default_dt

        if reference_velocity is None:
            reference_velocity = self.target_speed

        accel_feedforward = 0.0
        brake_feedforward = 0.0
        if reference_accel is not None:
            if reference_accel > 0.0 and self.max_accel > 0.0:
                accel_feedforward = min(reference_accel / self.max_accel, 1.0)
                accel_feedforward *= self.accel_feedforward_gain
            elif reference_accel < 0.0 and self.max_decel > 0.0:
                brake_feedforward = min(-reference_accel / self.max_decel, 1.0)
                brake_feedforward *= self.decel_feedforward_gain
        if self.continuous_accel_control:
            # Accel command already encodes reference_accel; avoid double-adding feedforward.
            accel_feedforward = 0.0
            brake_feedforward = 0.0
        
        # Enforce hard speed limit - check FIRST before any other logic
        if current_speed > self.max_speed:
            # Over speed limit - EMERGENCY BRAKING
            speed_excess = current_speed - self.max_speed
            # Very aggressive: minimum 0.8 brake, up to 1.0 for extreme speeds
            brake = min(1.0, 0.8 + (speed_excess / 5.0))
            throttle = 0.0
            self.pid_throttle.reset()
            # Update brake controller for smooth braking
            self.pid_brake.update(speed_excess, dt)
            self.last_speed = current_speed
            self.last_throttle = throttle
            self.last_brake = brake
            self.last_throttle_before_limits = throttle
            self.last_brake_before_limits = brake
            return throttle, brake
        
        # Also limit reference velocity to prevent requesting high speeds
        if reference_velocity > self.max_speed:
            reference_velocity = self.max_speed * 0.9  # Cap at 90% of max
        
        # Speed smoothing - filter rapid changes
        alpha = self.speed_smoothing_alpha
        smoothed_speed = alpha * self.last_speed + (1 - alpha) * current_speed
        self.last_speed = smoothed_speed
        
        # Compute speed error using smoothed speed
        speed_error = reference_velocity - smoothed_speed
        # Guard: never brake when actual speed is still below target.
        # Smoothing can temporarily report a negative error while we're still under-speed.
        raw_speed_error = reference_velocity - current_speed
        if raw_speed_error > 0.0 and speed_error < 0.0:
            speed_error = raw_speed_error

        measured_accel_for_pid = 0.0
        if self.last_filtered_speed is not None and dt > 0.0:
            measured_accel_for_pid = (smoothed_speed - self.last_filtered_speed) / dt
        
        # Determine desired acceleration (planner accel + accel tracking PID + speed-error bias).
        desired_accel = reference_accel if reference_accel is not None else 0.0
        if self.speed_drag_gain > 0.0:
            desired_accel -= self.speed_drag_gain * float(current_speed)
        if (
            self.continuous_accel_control
            and self.accel_tracking_enabled
            and reference_accel is not None
        ):
            accel_error = reference_accel - measured_accel_for_pid
            desired_accel += self.accel_pid.update(accel_error, dt)
        else:
            self.accel_pid.reset()
        speed_error_effective = speed_error
        if abs(speed_error_effective) < self.speed_error_deadband:
            speed_error_effective = 0.0
        if self.accel_tracking_enabled and reference_accel is not None:
            speed_error_effective *= self.accel_tracking_error_scale
        if speed_error_effective > 0.0:
            desired_accel += self.speed_error_gain_under * speed_error_effective
        else:
            desired_accel += self.speed_error_gain_over * speed_error_effective
        if self.max_accel > 0.0:
            desired_accel = min(desired_accel, self.max_accel)
        if self.max_decel > 0.0:
            desired_accel = max(desired_accel, -self.max_decel)
        if reference_velocity is not None and current_speed < reference_velocity - 0.5:
            # Prevent decel commands when we're clearly under target speed.
            desired_accel = max(desired_accel, 0.0)
        if self.accel_target_smoothing_alpha is not None:
            accel_alpha = np.clip(self.accel_target_smoothing_alpha, 0.0, 1.0)
            self.smoothed_desired_accel = (
                accel_alpha * self.smoothed_desired_accel
                + (1.0 - accel_alpha) * desired_accel
            )
        else:
            self.smoothed_desired_accel = desired_accel

        if self.continuous_accel_control:
            accel_cmd = float(self.smoothed_desired_accel)
            if abs(accel_cmd) < self.continuous_accel_deadband:
                accel_cmd = 0.0

            if (
                self.low_speed_accel_limit > 0.0
                and current_speed < self.low_speed_speed_threshold
            ):
                accel_cmd = min(accel_cmd, self.low_speed_accel_limit)

            # Startup ramp: cap accel early to prevent initial spikes.
            startup_ramp_ratio = 1.0
            startup_cap_active = False
            if current_speed < self.startup_speed_threshold:
                self.startup_elapsed += dt
                startup_ramp_ratio = min(
                    1.0,
                    self.startup_elapsed / max(self.startup_ramp_seconds, 1e-3)
                )
                accel_cmd = min(accel_cmd, self.startup_accel_limit * startup_ramp_ratio)
                startup_cap_active = True
            else:
                self.startup_elapsed = 0.0

            if reference_velocity is not None:
                if raw_speed_error > 0.5:
                    accel_cmd = max(accel_cmd, 0.0)
                elif raw_speed_error < -0.5:
                    accel_cmd = min(accel_cmd, 0.0)

            if self.overspeed_accel_zero_threshold > 0.0:
                if raw_speed_error < -self.overspeed_accel_zero_threshold:
                    accel_cmd = min(accel_cmd, 0.0)

            if self.max_jerk > 0.0 and dt > 0.0:
                speed_error_mag = (
                    abs(float(reference_velocity) - float(current_speed))
                    if reference_velocity is not None
                    else 0.0
                )
                if self.jerk_error_max > self.jerk_error_min:
                    jerk_ratio = np.clip(
                        (speed_error_mag - self.jerk_error_min)
                        / (self.jerk_error_max - self.jerk_error_min),
                        0.0,
                        1.0,
                    )
                else:
                    jerk_ratio = 1.0 if speed_error_mag > 0.0 else 0.0
                dynamic_max_jerk = self.max_jerk_min + jerk_ratio * (
                    self.max_jerk_max - self.max_jerk_min
                )
                accel_cmd = np.clip(
                    accel_cmd,
                    self.last_accel_cmd - (dynamic_max_jerk * dt),
                    self.last_accel_cmd + (dynamic_max_jerk * dt),
                )
                if self.jerk_cooldown_remaining > 0:
                    accel_cmd *= float(self.jerk_cooldown_scale)
                    self.jerk_cooldown_remaining -= 1
                self.last_accel_cmd = accel_cmd
            else:
                self.last_accel_cmd = accel_cmd

            if accel_cmd >= 0.0:
                throttle = (accel_cmd / self.max_accel) if self.max_accel > 0.0 else 0.0
                if self.throttle_curve_gamma != 1.0:
                    throttle = np.clip(throttle, 0.0, 1.0)
                    throttle = throttle ** float(self.throttle_curve_gamma)
                if self.throttle_curve_min > 0.0:
                    throttle = max(throttle, self.throttle_curve_min)
                if (
                    accel_feedforward > 0.0
                    and not (startup_cap_active and self.startup_disable_accel_feedforward)
                ):
                    throttle = np.clip(throttle + accel_feedforward, 0.0, 1.0)
                if not startup_cap_active:
                    if raw_speed_error > 0.2 and current_speed < reference_velocity:
                        throttle = max(throttle, self.min_throttle_when_accel)
                    if (
                        raw_speed_error > 0.2
                        and current_speed < self.min_throttle_hold_speed
                    ):
                        throttle = max(throttle, self.min_throttle_hold)
                if (
                    current_curvature is not None
                    and abs(current_curvature) <= self.straight_curvature_threshold
                ):
                    throttle = min(throttle, self.straight_throttle_cap)
                if startup_cap_active and self.max_accel > 0.0:
                    throttle_cap = (self.startup_accel_limit * startup_ramp_ratio) / self.max_accel
                    throttle = min(throttle, throttle_cap)
                if startup_cap_active and self.startup_throttle_cap > 0.0:
                    throttle = min(throttle, self.startup_throttle_cap)
                brake = 0.0
                self.longitudinal_mode = "accel" if throttle > 0.0 else "coast"
            else:
                throttle = 0.0
                brake = (abs(accel_cmd) / self.max_decel) if self.max_decel > 0.0 else 0.0
                if brake_feedforward > 0.0:
                    brake = np.clip(brake + brake_feedforward, 0.0, 1.0)
                self.longitudinal_mode = "brake" if brake > 0.0 else "coast"

            self.pid_throttle.reset()
            self.pid_brake.reset()

            measured_accel = 0.0
            if self.last_filtered_speed is not None and dt > 0.0:
                measured_accel = (smoothed_speed - self.last_filtered_speed) / dt
        else:
            # Mode selection with hysteresis + dwell to prevent oscillation.
            if reference_accel is not None and self.accel_tracking_enabled:
                # Overspeed must be able to force brake mode even when accel tracking is enabled.
                if raw_speed_error < self.speed_error_brake_threshold:
                    desired_mode = "brake"
                elif self.smoothed_desired_accel > self.accel_mode_threshold:
                    desired_mode = "accel"
                elif self.smoothed_desired_accel < -self.decel_mode_threshold:
                    desired_mode = "brake"
                else:
                    desired_mode = "coast"
            else:
                if raw_speed_error > self.speed_error_accel_threshold and self.smoothed_desired_accel > self.accel_mode_threshold:
                    desired_mode = "accel"
                elif raw_speed_error < self.speed_error_brake_threshold and self.smoothed_desired_accel < -self.decel_mode_threshold:
                    desired_mode = "brake"
                else:
                    desired_mode = "coast"

            if self.coast_hold_remaining > 0.0 and desired_mode == "accel":
                desired_mode = "coast"
            if self.coast_hold_remaining > 0.0:
                desired_mode = "coast"

            if desired_mode != self.longitudinal_mode and self.mode_time < self.mode_switch_min_time:
                desired_mode = self.longitudinal_mode

            if desired_mode != self.longitudinal_mode:
                self.longitudinal_mode = desired_mode
                self.mode_time = 0.0
                if desired_mode == "coast":
                    self.coast_hold_remaining = self.coast_hold_seconds
                if desired_mode == "accel":
                    self.pid_brake.reset()
                elif desired_mode == "brake":
                    self.pid_throttle.reset()
                else:
                    self.pid_throttle.reset()
                    self.pid_brake.reset()
            else:
                self.mode_time += dt
            if self.coast_hold_remaining > 0.0:
                self.coast_hold_remaining = max(0.0, self.coast_hold_remaining - dt)

            measured_accel = 0.0
            if self.last_filtered_speed is not None and dt > 0.0:
                measured_accel = (smoothed_speed - self.last_filtered_speed) / dt

            if self.longitudinal_mode == "accel":
                if self.accel_tracking_enabled:
                    accel_error = self.smoothed_desired_accel - measured_accel
                    base_throttle = self.pid_throttle.update(accel_error, dt)
                else:
                    base_throttle = self.pid_throttle.update(speed_error, dt)
                if raw_speed_error > 0.2 and current_speed < reference_velocity:
                    base_throttle = max(base_throttle, self.min_throttle_when_accel)
                if (
                    raw_speed_error > 0.2
                    and current_speed < self.min_throttle_hold_speed
                ):
                    base_throttle = max(base_throttle, self.min_throttle_hold)
                
                # FIXED: Add initial throttle limiting when starting from rest
                # CRITICAL FIX: Limit throttle to prevent unrealistic acceleration
                # Real cars accelerate at 2-4 m/s², not 500+ m/s²!
                # Prevents aggressive acceleration from 0 m/s (throttle would be 1.0 instantly)
                if smoothed_speed < 2.0:  # Very slow or at rest (increased from 1.0 to 2.0)
                    # Limit initial throttle to prevent aggressive acceleration
                    # Ramp up from 0.2 to full throttle as speed increases (reduced from 0.3)
                    initial_throttle_limit = 0.2 + (smoothed_speed / 2.0) * 0.8  # 0.2 to 1.0 over 0-2 m/s
                    base_throttle = min(base_throttle, initial_throttle_limit)
                
                # ADDITIONAL FIX: Cap maximum throttle to prevent extreme acceleration
                # Even at higher speeds, limit throttle to prevent overspeed
                # This helps prevent the car from accelerating too quickly
                max_throttle_limit = 0.8  # Never exceed 80% throttle (prevents extreme acceleration)
                base_throttle = min(base_throttle, max_throttle_limit)
                
                # Progressive throttle limiting as we approach max speed
                # This prevents overshoot and reduces need for emergency braking
                if smoothed_speed > self.max_speed * 0.80:
                    # Very close to max - aggressive reduction
                    throttle = base_throttle * 0.2
                elif smoothed_speed > self.max_speed * 0.75:
                    # Close to max - moderate reduction
                    throttle = base_throttle * 0.4
                elif smoothed_speed > self.max_speed * 0.65:
                    # Approaching max - light reduction
                    throttle = base_throttle * 0.6
                else:
                    # Normal operation - full throttle
                    throttle = base_throttle
                if accel_feedforward > 0.0:
                    throttle = np.clip(throttle + accel_feedforward, 0.0, 1.0)
                if (
                    current_curvature is not None
                    and abs(current_curvature) <= self.straight_curvature_threshold
                ):
                    throttle = min(throttle, self.straight_throttle_cap)
                brake = 0.0
            elif self.longitudinal_mode == "brake":
                throttle = 0.0
                if self.accel_tracking_enabled:
                    accel_error = self.smoothed_desired_accel - measured_accel
                    brake = self.pid_brake.update(-accel_error, dt)
                else:
                    brake = self.pid_brake.update(-speed_error, dt)
                if brake_feedforward > 0.0:
                    brake = np.clip(brake + brake_feedforward, 0.0, 1.0)
            else:  # Coast/hold: avoid braking and only allow light throttle correction
                throttle = 0.0
                if speed_error > 0.0:
                    throttle = min(self.coast_throttle_max, speed_error * self.coast_throttle_kp)
                brake = 0.0

        accel_capped = False
        jerk_capped = False
        if self.last_raw_speed is not None and dt > 0.0:
            if self.last_filtered_speed is None:
                filtered_speed = current_speed
            else:
                alpha = np.clip(self.speed_for_jerk_alpha, 0.0, 1.0)
                filtered_speed = (alpha * self.last_filtered_speed) + ((1.0 - alpha) * current_speed)
            measured_accel = (filtered_speed - self.last_filtered_speed) / dt if self.last_filtered_speed is not None else 0.0
            if measured_accel > 0.0 and self.max_accel > 0.0 and measured_accel > self.max_accel:
                if throttle > 0.0:
                    scale = min(1.0, self.max_accel / measured_accel)
                    throttle *= scale
                    accel_capped = True
            elif measured_accel < 0.0 and self.max_decel > 0.0 and abs(measured_accel) > self.max_decel:
                if brake > 0.0:
                    scale = min(1.0, self.max_decel / abs(measured_accel))
                    brake *= scale
                    accel_capped = True

            if self.last_accel is not None and self.max_jerk > 0.0:
                measured_jerk = (measured_accel - self.last_accel) / dt
                if measured_jerk > 0.0 and measured_jerk > self.max_jerk:
                    if throttle > 0.0:
                        scale = min(1.0, self.max_jerk / measured_jerk)
                        throttle *= scale
                        jerk_capped = True
                elif measured_jerk < 0.0 and abs(measured_jerk) > self.max_jerk:
                    if brake > 0.0:
                        scale = min(1.0, self.max_jerk / abs(measured_jerk))
                        brake *= scale
                        jerk_capped = True
                self.last_jerk = measured_jerk
            self.last_accel = measured_accel
            self.last_filtered_speed = filtered_speed
        elif self.last_filtered_speed is None:
            # Initialize filtered speed for next step.
            self.last_filtered_speed = current_speed

        self.last_raw_speed = current_speed
        self.last_accel_feedforward = accel_feedforward
        self.last_brake_feedforward = brake_feedforward
        self.last_accel_capped = accel_capped
        self.last_jerk_capped = jerk_capped

        if self.last_jerk_capped and self.jerk_cooldown_frames > 0:
            self.jerk_cooldown_remaining = self.jerk_cooldown_frames

        if self.max_jerk > 0.0 and dt > 0.0 and not self.continuous_accel_control:
            speed_error = abs(float(reference_velocity) - float(current_speed)) if reference_velocity is not None else 0.0
            if self.jerk_error_max > self.jerk_error_min:
                jerk_ratio = np.clip(
                    (speed_error - self.jerk_error_min) / (self.jerk_error_max - self.jerk_error_min),
                    0.0,
                    1.0,
                )
            else:
                jerk_ratio = 1.0 if speed_error > 0.0 else 0.0
            dynamic_max_jerk = self.max_jerk_min + jerk_ratio * (self.max_jerk_max - self.max_jerk_min)
            accel_cmd = (throttle * self.max_accel) - (brake * self.max_decel)
            accel_cmd_limited = np.clip(
                accel_cmd,
                self.last_accel_cmd - (dynamic_max_jerk * dt),
                self.last_accel_cmd + (dynamic_max_jerk * dt),
            )
            if self.jerk_cooldown_remaining > 0:
                accel_cmd_limited *= float(self.jerk_cooldown_scale)
                self.jerk_cooldown_remaining -= 1
            if accel_cmd_limited >= 0.0:
                throttle = min(1.0, accel_cmd_limited / self.max_accel) if self.max_accel > 0.0 else 0.0
                brake = 0.0
            else:
                throttle = 0.0
                brake = min(1.0, abs(accel_cmd_limited) / self.max_decel) if self.max_decel > 0.0 else 0.0
            self.last_accel_cmd = accel_cmd_limited
        elif not self.continuous_accel_control:
            self.last_accel_cmd = (throttle * self.max_accel) - (brake * self.max_decel)
        
        if (
            reference_velocity is not None
            and self.overspeed_brake_max > 0.0
            and raw_speed_error < -self.overspeed_brake_threshold
        ):
            overspeed = abs(raw_speed_error) - self.overspeed_brake_threshold
            brake_target = self.overspeed_brake_min + (self.overspeed_brake_gain * overspeed)
            brake_target = np.clip(
                brake_target,
                self.overspeed_brake_min,
                self.overspeed_brake_max
            )
            brake = max(brake, float(brake_target))
            throttle = 0.0

        self.last_throttle_before_limits = throttle
        self.last_brake_before_limits = brake

        if self.throttle_smoothing_alpha is not None:
            alpha = np.clip(self.throttle_smoothing_alpha, 0.0, 1.0)
            throttle = (alpha * throttle) + ((1.0 - alpha) * self.last_throttle)
        if self.throttle_rate_limit > 0.0:
            throttle = np.clip(
                throttle,
                self.last_throttle - self.throttle_rate_limit,
                self.last_throttle + self.throttle_rate_limit
            )
        if self.brake_rate_limit > 0.0:
            brake = np.clip(
                brake,
                self.last_brake - self.brake_rate_limit,
                self.last_brake + self.brake_rate_limit
            )
        if (
            self.longitudinal_mode == "accel"
            and raw_speed_error > 0.2
            and current_speed < self.min_throttle_hold_speed
        ):
            throttle = max(throttle, self.min_throttle_hold)
        if (
            self.longitudinal_mode == "accel"
            and min_speed_floor is not None
            and current_speed < float(min_speed_floor)
            and raw_speed_error > 0.0
        ):
            throttle = max(throttle, self.min_throttle_hold)
            if (
                current_curvature is not None
                and abs(current_curvature) <= self.straight_curvature_threshold
            ):
                throttle = min(throttle, self.straight_throttle_cap)

        if (
            self.low_speed_speed_threshold > 0.0
            and current_speed < self.low_speed_speed_threshold
            and throttle > 0.0
        ):
            # Avoid brake+throttle overlap at low speed (launch smoothness).
            brake = 0.0
        if throttle > 0.0 and brake > 0.0:
            # Enforce mutual exclusivity to prevent oscillation spikes.
            if raw_speed_error >= 0.0:
                brake = 0.0
            else:
                throttle = 0.0

        self.last_throttle = throttle
        self.last_brake = brake

        return throttle, brake
    
    def reset(self):
        """Reset controllers."""
        self.pid_throttle.reset()
        self.pid_brake.reset()
        self.last_throttle = 0.0
        self.last_brake = 0.0
        self.last_throttle_before_limits = 0.0
        self.last_brake_before_limits = 0.0
        self.longitudinal_mode = "coast"
        self.mode_time = self.mode_switch_min_time
        self.coast_hold_remaining = 0.0
        self.smoothed_desired_accel = 0.0


class VehicleController:
    """
    Combined vehicle controller (lateral + longitudinal).
    """
    
    def __init__(self, lateral_kp: float = 0.3, lateral_ki: float = 0.0, lateral_kd: float = 0.1,
                 longitudinal_kp: float = 0.3, longitudinal_ki: float = 0.05, longitudinal_kd: float = 0.02,
                 lookahead_distance: float = 10.0, target_speed: float = 8.0, max_speed: float = 10.0, max_steering: float = 0.5,
                 lateral_deadband: float = 0.02, lateral_heading_weight: float = 0.5, lateral_lateral_weight: float = 0.5,
                 lateral_error_clip: Optional[float] = None, lateral_integral_limit: float = 0.3,
                 throttle_rate_limit: float = 0.08, brake_rate_limit: float = 0.15,
                 throttle_smoothing_alpha: float = 0.6, speed_smoothing_alpha: float = 0.7,
                 min_throttle_when_accel: float = 0.0,
                 longitudinal_default_dt: float = 1.0 / 30.0,
                 longitudinal_max_accel: float = 2.5,
                 longitudinal_max_decel: float = 3.0,
                 longitudinal_max_jerk: float = 2.0,
                 longitudinal_accel_feedforward_gain: float = 1.0,
                 longitudinal_decel_feedforward_gain: float = 1.0,
                 longitudinal_speed_for_jerk_alpha: float = 0.7,
                 longitudinal_jerk_error_min: float = 0.5,
                 longitudinal_jerk_error_max: float = 3.0,
                 longitudinal_max_jerk_min: float = 1.5,
                 longitudinal_max_jerk_max: float = 6.0,
                 longitudinal_jerk_cooldown_frames: int = 0,
                 longitudinal_jerk_cooldown_scale: float = 0.6,
                 longitudinal_min_throttle_hold: float = 0.12,
                 longitudinal_min_throttle_hold_speed: float = 4.0,
                 longitudinal_speed_error_to_accel_gain: float = 0.5,
                 longitudinal_speed_error_deadband: float = 0.0,
                 longitudinal_speed_error_gain_under: Optional[float] = None,
                 longitudinal_speed_error_gain_over: Optional[float] = None,
                 longitudinal_overspeed_accel_zero_threshold: float = 0.0,
                 longitudinal_overspeed_brake_threshold: float = 0.3,
                 longitudinal_overspeed_brake_min: float = 0.02,
                 longitudinal_overspeed_brake_max: float = 0.2,
                 longitudinal_overspeed_brake_gain: float = 0.2,
                 longitudinal_accel_mode_threshold: float = 0.2,
                 longitudinal_decel_mode_threshold: float = 0.2,
                 longitudinal_speed_error_accel_threshold: float = 0.2,
                 longitudinal_speed_error_brake_threshold: float = -0.2,
                 longitudinal_mode_switch_min_time: float = 0.4,
                 longitudinal_coast_hold_seconds: float = 0.5,
                 longitudinal_coast_throttle_kp: float = 0.1,
                 longitudinal_coast_throttle_max: float = 0.12,
                 longitudinal_straight_throttle_cap: float = 0.45,
                 longitudinal_straight_curvature_threshold: float = 0.003,
                 longitudinal_accel_tracking_enabled: bool = True,
                 longitudinal_accel_tracking_error_scale: float = 1.0,
                 longitudinal_accel_pid_kp: float = 0.4,
                 longitudinal_accel_pid_ki: float = 0.05,
                 longitudinal_accel_pid_kd: float = 0.02,
                 longitudinal_throttle_curve_gamma: float = 1.0,
                 longitudinal_throttle_curve_min: float = 0.0,
                 longitudinal_speed_drag_gain: float = 0.0,
                 longitudinal_accel_target_smoothing_alpha: float = 0.6,
                 longitudinal_continuous_accel_control: bool = False,
                 longitudinal_continuous_accel_deadband: float = 0.05,
                 longitudinal_startup_ramp_seconds: float = 2.5,
                 longitudinal_startup_accel_limit: float = 1.2,
                 longitudinal_startup_speed_threshold: float = 2.0,
                 longitudinal_startup_throttle_cap: float = 0.0,
                 longitudinal_startup_disable_accel_feedforward: bool = True,
                 longitudinal_low_speed_accel_limit: float = 0.0,
                 longitudinal_low_speed_speed_threshold: float = 0.0,
                 steering_smoothing_alpha: float = 0.7,
                 base_error_smoothing_alpha: float = 0.7,
                 heading_error_smoothing_alpha: float = 0.45,
                 straight_window_frames: int = 60,
                 straight_oscillation_high: float = 0.20,
                 straight_oscillation_low: float = 0.05,
                 curve_feedforward_gain: float = 1.0, curve_feedforward_threshold: float = 0.02,
                 curve_feedforward_gain_min: float = 1.0, curve_feedforward_gain_max: float = 1.0,
                 curve_feedforward_curvature_min: float = 0.005, curve_feedforward_curvature_max: float = 0.03,
                 curve_feedforward_curvature_clamp: float = 0.03,
                 curve_feedforward_bins: Optional[list] = None,
                 curvature_scale_factor: float = 1.0,  # Scale curvature for feedforward (to match GT)
                 curvature_scale_threshold: float = 0.0005,  # Only scale when curvature > this (avoid noise on straights)
                 curvature_smoothing_alpha: float = 0.7,
                 curvature_transition_threshold: float = 0.01,
                 curvature_transition_alpha: float = 0.3,
                 straight_curvature_threshold: float = 0.01,
                 curve_upcoming_enter_threshold: float = 0.012,
                 curve_upcoming_exit_threshold: float = 0.009,
                 curve_upcoming_on_frames: int = 2,
                 curve_upcoming_off_frames: int = 2,
                 curve_phase_use_distance_track: bool = False,
                 curve_phase_track_name: Optional[str] = None,
                 curve_at_car_distance_min_m: float = 0.0,
                 road_curve_enter_threshold: Optional[float] = None,
                 road_curve_exit_threshold: Optional[float] = None,
                 road_straight_hold_invalid_frames: int = 6,
                 steering_rate_curvature_min: float = 0.005,
                 steering_rate_curvature_max: float = 0.03,
                 steering_rate_scale_min: float = 0.5,
                 curve_rate_floor_moderate_error: float = 0.20,
                 curve_rate_floor_large_error: float = 0.24,
                 straight_sign_flip_error_threshold: float = 0.02,
                 straight_sign_flip_rate: float = 0.2,
                 straight_sign_flip_frames: int = 6,
                 steering_jerk_limit: float = 0.0,
                 steering_jerk_curve_scale_max: float = 1.0,
                 steering_jerk_curve_min: float = 0.003,
                 steering_jerk_curve_max: float = 0.015,
                 curve_entry_assist_enabled: bool = False,
                 curve_entry_assist_error_min: float = 0.30,
                 curve_entry_assist_heading_error_min: float = 0.10,
                 curve_entry_assist_curvature_min: float = 0.010,
                 curve_entry_assist_rate_boost: float = 1.15,
                 curve_entry_assist_jerk_boost: float = 1.10,
                 curve_entry_assist_max_frames: int = 18,
                 curve_entry_assist_watchdog_rate_delta_max: float = 0.22,
                 curve_entry_assist_rearm_frames: int = 20,
                 curve_entry_schedule_enabled: bool = False,
                 curve_entry_schedule_frames: int = 18,
                 curve_entry_schedule_min_rate: float = 0.22,
                 curve_entry_schedule_min_jerk: float = 0.14,
                 curve_entry_schedule_min_hold_frames: int = 12,
                 curve_entry_schedule_min_curve_progress_ratio: float = 0.20,
                 curve_entry_schedule_fallback_only_when_dynamic: bool = False,
                 curve_entry_schedule_fallback_deficit_frames: int = 6,
                 curve_entry_schedule_fallback_rate_deficit_min: float = 0.03,
                 curve_entry_schedule_fallback_rearm_cooldown_frames: int = 18,
                 curve_entry_schedule_handoff_transfer_ratio: float = 0.65,
                 curve_entry_schedule_handoff_error_fall: float = 0.03,
                 dynamic_curve_authority_enabled: bool = True,
                 dynamic_curve_rate_deficit_deadband: float = 0.01,
                 dynamic_curve_rate_boost_gain: float = 1.0,
                 dynamic_curve_rate_boost_max: float = 0.30,
                 dynamic_curve_jerk_boost_gain: float = 4.0,
                 dynamic_curve_jerk_boost_max_factor: float = 3.5,
                 dynamic_curve_hard_clip_boost_gain: float = 1.0,
                 dynamic_curve_hard_clip_boost_max: float = 0.12,
                 dynamic_curve_entry_governor_enabled: bool = True,
                 dynamic_curve_entry_governor_gain: float = 1.2,
                 dynamic_curve_entry_governor_max_scale: float = 1.8,
                 dynamic_curve_entry_governor_stale_floor_scale: float = 1.15,
                 dynamic_curve_entry_governor_exclusive_mode: bool = True,
                 dynamic_curve_entry_governor_anticipatory_enabled: bool = True,
                 dynamic_curve_entry_governor_upcoming_phase_weight: float = 0.55,
                 dynamic_curve_authority_precurve_enabled: bool = True,
                 dynamic_curve_authority_precurve_scale: float = 0.8,
                 dynamic_curve_single_owner_mode: bool = False,
                 dynamic_curve_single_owner_min_rate: float = 0.22,
                 dynamic_curve_single_owner_min_jerk: float = 0.6,
                 dynamic_curve_comfort_lat_accel_comfort_max_g: float = 0.18,
                 dynamic_curve_comfort_lat_accel_peak_max_g: float = 0.25,
                 dynamic_curve_comfort_lat_jerk_comfort_max_gps: float = 0.30,
                 dynamic_curve_lat_jerk_smoothing_alpha: float = 0.25,
                 dynamic_curve_lat_jerk_soft_start_ratio: float = 0.60,
                 dynamic_curve_lat_jerk_soft_floor_scale: float = 0.35,
                 dynamic_curve_speed_low_mps: float = 4.0,
                 dynamic_curve_speed_high_mps: float = 10.0,
                 dynamic_curve_speed_boost_max_scale: float = 1.4,
                 turn_feasibility_governor_enabled: bool = True,
                 turn_feasibility_curvature_min: float = 0.002,
                 turn_feasibility_guardband_g: float = 0.015,
                 turn_feasibility_use_peak_bound: bool = True,
                 curve_unwind_policy_enabled: bool = False,
                 curve_unwind_frames: int = 12,
                 curve_unwind_rate_scale_start: float = 1.0,
                 curve_unwind_rate_scale_end: float = 0.8,
                 curve_unwind_jerk_scale_start: float = 1.0,
                 curve_unwind_jerk_scale_end: float = 0.7,
                 curve_unwind_integral_decay: float = 0.85,
                 curve_commit_mode_enabled: bool = False,
                 curve_commit_mode_max_frames: int = 20,
                 curve_commit_mode_min_rate: float = 0.22,
                 curve_commit_mode_min_jerk: float = 0.14,
                 curve_commit_mode_transfer_ratio_target: float = 0.72,
                 curve_commit_mode_error_fall: float = 0.03,
                 curve_commit_mode_exit_consecutive_frames: int = 4,
                 curve_commit_mode_retrigger_on_dynamic_deficit: bool = True,
                 curve_commit_mode_dynamic_deficit_frames: int = 8,
                 curve_commit_mode_dynamic_deficit_min: float = 0.03,
                 curve_commit_mode_retrigger_cooldown_frames: int = 12,
                 speed_gain_min_speed: float = 4.0,
                 speed_gain_max_speed: float = 10.0,
                 speed_gain_min: float = 1.0,
                 speed_gain_max: float = 1.2,
                 speed_gain_curvature_min: float = 0.002,
                 speed_gain_curvature_max: float = 0.015,
                 control_mode: str = "pid",
                 stanley_k: float = 1.0,
                 stanley_soft_speed: float = 2.0,
                 stanley_heading_weight: float = 1.0,
                 pp_feedback_gain: float = 0.15,
                 pp_min_lookahead: float = 0.5,
                 pp_ref_jump_clamp: float = 0.5,
                 pp_stale_decay: float = 0.98,
                 feedback_gain_min: float = 1.0,
                 feedback_gain_max: float = 1.2,
                 feedback_gain_curvature_min: float = 0.002,
                 feedback_gain_curvature_max: float = 0.015,
                 curvature_stale_hold_seconds: float = 0.30,
                 curvature_stale_hold_min_abs: float = 0.0005):
        """
        Initialize vehicle controller.
        
        Args:
            lateral_kp: Lateral proportional gain
            lateral_ki: Lateral integral gain
            lateral_kd: Lateral derivative gain
            longitudinal_kp: Longitudinal proportional gain
            longitudinal_ki: Longitudinal integral gain
            longitudinal_kd: Longitudinal derivative gain
            lookahead_distance: Lookahead distance (meters)
            target_speed: Target speed (m/s)
        """
        # Set error_clip default if not provided
        if lateral_error_clip is None:
            import numpy as np
            lateral_error_clip = np.pi / 4
        
        self.lateral_controller = LateralController(
            kp=lateral_kp,
            ki=lateral_ki,
            kd=lateral_kd,
            lookahead_distance=lookahead_distance,
            max_steering=max_steering,
            deadband=lateral_deadband,
            heading_weight=lateral_heading_weight,
            lateral_weight=lateral_lateral_weight,
            error_clip=lateral_error_clip,
            integral_limit=lateral_integral_limit,
            steering_smoothing_alpha=steering_smoothing_alpha,
            base_error_smoothing_alpha=base_error_smoothing_alpha,
            heading_error_smoothing_alpha=heading_error_smoothing_alpha,
            straight_window_frames=straight_window_frames,
            straight_oscillation_high=straight_oscillation_high,
            straight_oscillation_low=straight_oscillation_low,
            curve_feedforward_gain=curve_feedforward_gain,
            curve_feedforward_threshold=curve_feedforward_threshold,
            curve_feedforward_gain_min=curve_feedforward_gain_min,
            curve_feedforward_gain_max=curve_feedforward_gain_max,
            curve_feedforward_curvature_min=curve_feedforward_curvature_min,
            curve_feedforward_curvature_max=curve_feedforward_curvature_max,
            curve_feedforward_curvature_clamp=curve_feedforward_curvature_clamp,
            curve_feedforward_bins=curve_feedforward_bins,
            curvature_scale_factor=curvature_scale_factor,
            curvature_scale_threshold=curvature_scale_threshold,
            curvature_smoothing_alpha=curvature_smoothing_alpha,
            curvature_transition_threshold=curvature_transition_threshold,
            curvature_transition_alpha=curvature_transition_alpha,
            straight_curvature_threshold=straight_curvature_threshold,
            curve_upcoming_enter_threshold=curve_upcoming_enter_threshold,
            curve_upcoming_exit_threshold=curve_upcoming_exit_threshold,
            curve_upcoming_on_frames=curve_upcoming_on_frames,
            curve_upcoming_off_frames=curve_upcoming_off_frames,
            curve_phase_use_distance_track=curve_phase_use_distance_track,
            curve_phase_track_name=curve_phase_track_name,
            curve_at_car_distance_min_m=curve_at_car_distance_min_m,
            road_curve_enter_threshold=road_curve_enter_threshold,
            road_curve_exit_threshold=road_curve_exit_threshold,
            road_straight_hold_invalid_frames=road_straight_hold_invalid_frames,
            steering_rate_curvature_min=steering_rate_curvature_min,
            steering_rate_curvature_max=steering_rate_curvature_max,
            steering_rate_scale_min=steering_rate_scale_min,
            curve_rate_floor_moderate_error=curve_rate_floor_moderate_error,
            curve_rate_floor_large_error=curve_rate_floor_large_error,
            straight_sign_flip_error_threshold=straight_sign_flip_error_threshold,
            straight_sign_flip_rate=straight_sign_flip_rate,
            straight_sign_flip_frames=straight_sign_flip_frames,
            steering_jerk_limit=steering_jerk_limit,
            steering_jerk_curve_scale_max=steering_jerk_curve_scale_max,
            steering_jerk_curve_min=steering_jerk_curve_min,
            steering_jerk_curve_max=steering_jerk_curve_max,
            curve_entry_assist_enabled=curve_entry_assist_enabled,
            curve_entry_assist_error_min=curve_entry_assist_error_min,
            curve_entry_assist_heading_error_min=curve_entry_assist_heading_error_min,
            curve_entry_assist_curvature_min=curve_entry_assist_curvature_min,
            curve_entry_assist_rate_boost=curve_entry_assist_rate_boost,
            curve_entry_assist_jerk_boost=curve_entry_assist_jerk_boost,
            curve_entry_assist_max_frames=curve_entry_assist_max_frames,
            curve_entry_assist_watchdog_rate_delta_max=curve_entry_assist_watchdog_rate_delta_max,
            curve_entry_assist_rearm_frames=curve_entry_assist_rearm_frames,
            curve_entry_schedule_enabled=curve_entry_schedule_enabled,
            curve_entry_schedule_frames=curve_entry_schedule_frames,
            curve_entry_schedule_min_rate=curve_entry_schedule_min_rate,
            curve_entry_schedule_min_jerk=curve_entry_schedule_min_jerk,
            curve_entry_schedule_min_hold_frames=curve_entry_schedule_min_hold_frames,
            curve_entry_schedule_min_curve_progress_ratio=(
                curve_entry_schedule_min_curve_progress_ratio
            ),
            curve_entry_schedule_fallback_only_when_dynamic=(
                curve_entry_schedule_fallback_only_when_dynamic
            ),
            curve_entry_schedule_fallback_deficit_frames=(
                curve_entry_schedule_fallback_deficit_frames
            ),
            curve_entry_schedule_fallback_rate_deficit_min=(
                curve_entry_schedule_fallback_rate_deficit_min
            ),
            curve_entry_schedule_fallback_rearm_cooldown_frames=(
                curve_entry_schedule_fallback_rearm_cooldown_frames
            ),
            curve_entry_schedule_handoff_transfer_ratio=curve_entry_schedule_handoff_transfer_ratio,
            curve_entry_schedule_handoff_error_fall=curve_entry_schedule_handoff_error_fall,
            dynamic_curve_authority_enabled=dynamic_curve_authority_enabled,
            dynamic_curve_rate_deficit_deadband=dynamic_curve_rate_deficit_deadband,
            dynamic_curve_rate_boost_gain=dynamic_curve_rate_boost_gain,
            dynamic_curve_rate_boost_max=dynamic_curve_rate_boost_max,
            dynamic_curve_jerk_boost_gain=dynamic_curve_jerk_boost_gain,
            dynamic_curve_jerk_boost_max_factor=dynamic_curve_jerk_boost_max_factor,
            dynamic_curve_hard_clip_boost_gain=dynamic_curve_hard_clip_boost_gain,
            dynamic_curve_hard_clip_boost_max=dynamic_curve_hard_clip_boost_max,
            dynamic_curve_entry_governor_enabled=dynamic_curve_entry_governor_enabled,
            dynamic_curve_entry_governor_gain=dynamic_curve_entry_governor_gain,
            dynamic_curve_entry_governor_max_scale=dynamic_curve_entry_governor_max_scale,
            dynamic_curve_entry_governor_stale_floor_scale=(
                dynamic_curve_entry_governor_stale_floor_scale
            ),
            dynamic_curve_entry_governor_exclusive_mode=(
                dynamic_curve_entry_governor_exclusive_mode
            ),
            dynamic_curve_entry_governor_upcoming_phase_weight=(
                dynamic_curve_entry_governor_upcoming_phase_weight
            ),
            dynamic_curve_entry_governor_anticipatory_enabled=(
                dynamic_curve_entry_governor_anticipatory_enabled
            ),
            dynamic_curve_authority_precurve_enabled=(
                dynamic_curve_authority_precurve_enabled
            ),
            dynamic_curve_authority_precurve_scale=dynamic_curve_authority_precurve_scale,
            dynamic_curve_single_owner_mode=dynamic_curve_single_owner_mode,
            dynamic_curve_single_owner_min_rate=dynamic_curve_single_owner_min_rate,
            dynamic_curve_single_owner_min_jerk=dynamic_curve_single_owner_min_jerk,
            dynamic_curve_comfort_lat_accel_comfort_max_g=(
                dynamic_curve_comfort_lat_accel_comfort_max_g
            ),
            dynamic_curve_comfort_lat_accel_peak_max_g=(
                dynamic_curve_comfort_lat_accel_peak_max_g
            ),
            dynamic_curve_comfort_lat_jerk_comfort_max_gps=(
                dynamic_curve_comfort_lat_jerk_comfort_max_gps
            ),
            dynamic_curve_lat_jerk_smoothing_alpha=dynamic_curve_lat_jerk_smoothing_alpha,
            dynamic_curve_lat_jerk_soft_start_ratio=dynamic_curve_lat_jerk_soft_start_ratio,
            dynamic_curve_lat_jerk_soft_floor_scale=dynamic_curve_lat_jerk_soft_floor_scale,
            dynamic_curve_speed_low_mps=dynamic_curve_speed_low_mps,
            dynamic_curve_speed_high_mps=dynamic_curve_speed_high_mps,
            dynamic_curve_speed_boost_max_scale=dynamic_curve_speed_boost_max_scale,
            turn_feasibility_governor_enabled=turn_feasibility_governor_enabled,
            turn_feasibility_curvature_min=turn_feasibility_curvature_min,
            turn_feasibility_guardband_g=turn_feasibility_guardband_g,
            turn_feasibility_use_peak_bound=turn_feasibility_use_peak_bound,
            curve_unwind_policy_enabled=curve_unwind_policy_enabled,
            curve_unwind_frames=curve_unwind_frames,
            curve_unwind_rate_scale_start=curve_unwind_rate_scale_start,
            curve_unwind_rate_scale_end=curve_unwind_rate_scale_end,
            curve_unwind_jerk_scale_start=curve_unwind_jerk_scale_start,
            curve_unwind_jerk_scale_end=curve_unwind_jerk_scale_end,
            curve_unwind_integral_decay=curve_unwind_integral_decay,
            curve_commit_mode_enabled=curve_commit_mode_enabled,
            curve_commit_mode_max_frames=curve_commit_mode_max_frames,
            curve_commit_mode_min_rate=curve_commit_mode_min_rate,
            curve_commit_mode_min_jerk=curve_commit_mode_min_jerk,
            curve_commit_mode_transfer_ratio_target=curve_commit_mode_transfer_ratio_target,
            curve_commit_mode_error_fall=curve_commit_mode_error_fall,
            curve_commit_mode_exit_consecutive_frames=curve_commit_mode_exit_consecutive_frames,
            curve_commit_mode_retrigger_on_dynamic_deficit=(
                curve_commit_mode_retrigger_on_dynamic_deficit
            ),
            curve_commit_mode_dynamic_deficit_frames=(
                curve_commit_mode_dynamic_deficit_frames
            ),
            curve_commit_mode_dynamic_deficit_min=(
                curve_commit_mode_dynamic_deficit_min
            ),
            curve_commit_mode_retrigger_cooldown_frames=(
                curve_commit_mode_retrigger_cooldown_frames
            ),
            speed_gain_min_speed=speed_gain_min_speed,
            speed_gain_max_speed=speed_gain_max_speed,
            speed_gain_min=speed_gain_min,
            speed_gain_max=speed_gain_max,
            speed_gain_curvature_min=speed_gain_curvature_min,
            speed_gain_curvature_max=speed_gain_curvature_max,
            control_mode=control_mode,
            stanley_k=stanley_k,
            stanley_soft_speed=stanley_soft_speed,
            stanley_heading_weight=stanley_heading_weight,
            pp_feedback_gain=pp_feedback_gain,
            pp_min_lookahead=pp_min_lookahead,
            pp_ref_jump_clamp=pp_ref_jump_clamp,
            pp_stale_decay=pp_stale_decay,
            feedback_gain_min=feedback_gain_min,
            feedback_gain_max=feedback_gain_max,
            feedback_gain_curvature_min=feedback_gain_curvature_min,
            feedback_gain_curvature_max=feedback_gain_curvature_max,
            curvature_stale_hold_seconds=curvature_stale_hold_seconds,
            curvature_stale_hold_min_abs=curvature_stale_hold_min_abs
        )
        self.longitudinal_controller = LongitudinalController(
            kp=longitudinal_kp,
            ki=longitudinal_ki,
            kd=longitudinal_kd,
            target_speed=target_speed,
            max_speed=max_speed,
            throttle_rate_limit=throttle_rate_limit,
            brake_rate_limit=brake_rate_limit,
            throttle_smoothing_alpha=throttle_smoothing_alpha,
            speed_smoothing_alpha=speed_smoothing_alpha,
            min_throttle_when_accel=min_throttle_when_accel,
            default_dt=longitudinal_default_dt,
            max_accel=longitudinal_max_accel,
            max_decel=longitudinal_max_decel,
            max_jerk=longitudinal_max_jerk,
            accel_feedforward_gain=longitudinal_accel_feedforward_gain,
            decel_feedforward_gain=longitudinal_decel_feedforward_gain,
            speed_for_jerk_alpha=longitudinal_speed_for_jerk_alpha,
            jerk_error_min=longitudinal_jerk_error_min,
            jerk_error_max=longitudinal_jerk_error_max,
            max_jerk_min=longitudinal_max_jerk_min,
            max_jerk_max=longitudinal_max_jerk_max,
            jerk_cooldown_frames=longitudinal_jerk_cooldown_frames,
            jerk_cooldown_scale=longitudinal_jerk_cooldown_scale,
            min_throttle_hold=longitudinal_min_throttle_hold,
            min_throttle_hold_speed=longitudinal_min_throttle_hold_speed,
            speed_error_to_accel_gain=longitudinal_speed_error_to_accel_gain,
            speed_error_deadband=longitudinal_speed_error_deadband,
            speed_error_gain_under=longitudinal_speed_error_gain_under,
            speed_error_gain_over=longitudinal_speed_error_gain_over,
            overspeed_accel_zero_threshold=longitudinal_overspeed_accel_zero_threshold,
            overspeed_brake_threshold=longitudinal_overspeed_brake_threshold,
            overspeed_brake_min=longitudinal_overspeed_brake_min,
            overspeed_brake_max=longitudinal_overspeed_brake_max,
            overspeed_brake_gain=longitudinal_overspeed_brake_gain,
            accel_mode_threshold=longitudinal_accel_mode_threshold,
            decel_mode_threshold=longitudinal_decel_mode_threshold,
            speed_error_accel_threshold=longitudinal_speed_error_accel_threshold,
            speed_error_brake_threshold=longitudinal_speed_error_brake_threshold,
            mode_switch_min_time=longitudinal_mode_switch_min_time,
            coast_hold_seconds=longitudinal_coast_hold_seconds,
            coast_throttle_kp=longitudinal_coast_throttle_kp,
            coast_throttle_max=longitudinal_coast_throttle_max,
            straight_throttle_cap=longitudinal_straight_throttle_cap,
            straight_curvature_threshold=longitudinal_straight_curvature_threshold,
            accel_tracking_enabled=longitudinal_accel_tracking_enabled,
            accel_tracking_error_scale=longitudinal_accel_tracking_error_scale,
            accel_pid_kp=longitudinal_accel_pid_kp,
            accel_pid_ki=longitudinal_accel_pid_ki,
            accel_pid_kd=longitudinal_accel_pid_kd,
            throttle_curve_gamma=longitudinal_throttle_curve_gamma,
            throttle_curve_min=longitudinal_throttle_curve_min,
            speed_drag_gain=longitudinal_speed_drag_gain,
            accel_target_smoothing_alpha=longitudinal_accel_target_smoothing_alpha,
            continuous_accel_control=longitudinal_continuous_accel_control,
            continuous_accel_deadband=longitudinal_continuous_accel_deadband,
            startup_ramp_seconds=longitudinal_startup_ramp_seconds,
            startup_accel_limit=longitudinal_startup_accel_limit,
            startup_speed_threshold=longitudinal_startup_speed_threshold,
            startup_throttle_cap=longitudinal_startup_throttle_cap,
            startup_disable_accel_feedforward=longitudinal_startup_disable_accel_feedforward,
            low_speed_accel_limit=longitudinal_low_speed_accel_limit,
            low_speed_speed_threshold=longitudinal_low_speed_speed_threshold
        )
    
    def compute_control(
        self,
        current_state: dict,
        reference_point: dict,
        return_metadata: bool = False,
        dt: Optional[float] = None,
        reference_accel: Optional[float] = None,
        using_stale_perception: bool = False
    ) -> dict:
        """
        Compute control commands.
        
        Args:
            current_state: Current vehicle state (heading, speed, position)
            reference_point: Reference point from trajectory
            return_metadata: If True, include PID internal state and errors
        
        Returns:
            Control commands (steering, throttle, brake)
            If return_metadata=True, also includes: steering_before_limits, 
            lateral_error, heading_error, total_error, pid_integral, pid_derivative
        """
        # Lateral control
        steering_result = self.lateral_controller.compute_steering(
            current_state.get('heading', 0.0),
            reference_point,
            current_state.get('position'),
            current_state.get('speed', 0.0),
            road_center_reference_t=current_state.get('road_center_reference_t'),
            dt=dt,
            return_metadata=return_metadata,
            using_stale_perception=using_stale_perception
        )
        
        if return_metadata:
            assert isinstance(steering_result, dict)
            steering = steering_result['steering']
            lateral_metadata = steering_result
        else:
            steering = steering_result
            lateral_metadata = {}
        
        # Longitudinal control
        throttle, brake = self.longitudinal_controller.compute_control(
            current_state.get('speed', 0.0),
            reference_point.get('velocity'),
            dt=dt,
            reference_accel=reference_accel,
            current_curvature=reference_point.get('curvature'),
            min_speed_floor=reference_point.get('min_speed_floor')
        )
        
        result = {
            'steering': steering,
            'throttle': throttle,
            'brake': brake
        }
        
        # FIXED: Always include lateral_error in control_command for safety mechanisms
        # Safety mechanisms in av_stack.py need lateral_error to trigger
        lateral_error = lateral_metadata.get('lateral_error', 0.0)
        result['lateral_error'] = lateral_error
        
        if return_metadata:
            result.update(lateral_metadata)
            # For throttle/brake, we don't track before_limits separately yet
            result['throttle_before_limits'] = self.longitudinal_controller.last_throttle_before_limits
            result['brake_before_limits'] = self.longitudinal_controller.last_brake_before_limits
            result['accel_feedforward'] = self.longitudinal_controller.last_accel_feedforward
            result['brake_feedforward'] = self.longitudinal_controller.last_brake_feedforward
            result['longitudinal_accel_capped'] = self.longitudinal_controller.last_accel_capped
            result['longitudinal_jerk_capped'] = self.longitudinal_controller.last_jerk_capped
        
        return result
    
    def reset(self):
        """Reset all controllers."""
        self.lateral_controller.reset()
        self.longitudinal_controller.reset()

