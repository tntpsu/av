#!/usr/bin/env python3
"""
Comprehensive overall drive analysis tool for AV Stack.

This is the PRIMARY analysis tool for evaluating complete drive performance.
Combines industry-standard metrics with project-specific diagnostics.

Metrics included:
- Path tracking accuracy (lateral error, heading error, cross-track error)
- Control smoothness (jerk, steering rate, oscillation)
- Perception quality (detection rate, confidence, stale data)
- Trajectory quality (accuracy, smoothness, availability)
- System health (PID health, error conflicts, stale commands)
- Safety metrics (out-of-lane events, emergency stops)

Usage:
    python tools/analyze_drive_overall.py <recording_file>
    python tools/analyze_drive_overall.py --latest
    python tools/analyze_drive_overall.py --list
"""

import sys
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import yaml
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy import stats
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from trajectory.utils import smooth_curvature_distance
from tools.drive_summary_core import analyze_recording_summary


@dataclass
class DriveMetrics:
    """Overall drive performance metrics."""
    # Executive summary
    drive_duration: float
    total_frames: int
    success_rate: float  # % of time in lane
    
    # Path tracking
    lateral_error_rmse: float
    lateral_error_mean: float
    lateral_error_max: float
    lateral_error_std: float
    lateral_error_p50: float
    lateral_error_p95: float
    heading_error_rmse: float
    heading_error_mean: float
    heading_error_max: float
    time_in_lane: float  # % within lane boundaries (primary)
    time_in_lane_centered: float  # % within ±0.5m of center (secondary)
    
    # Control smoothness
    steering_jerk_mean: float
    steering_jerk_max: float
    steering_rate_mean: float
    steering_rate_max: float
    steering_smoothness: float  # 1/std of steering
    oscillation_frequency: float
    control_effort: float
    straight_frames: int
    straight_oscillation_rate: float
    straight_stability_score: float
    
    # Perception quality
    lane_detection_rate: float
    perception_confidence_mean: float
    perception_confidence_std: float
    perception_jumps_detected: int
    stale_perception_rate: float
    perception_freeze_events: int
    
    # Trajectory quality
    trajectory_availability: float
    ref_point_accuracy_rmse: float
    trajectory_smoothness: float
    path_curvature_consistency: float
    
    # System health
    pid_integral_max: float
    pid_reset_frequency: float
    error_conflict_rate: float
    stale_command_rate: float
    
    # Speed control and comfort
    speed_error_rmse: float
    speed_error_mean: float
    speed_error_max: float
    speed_overspeed_rate: float
    planned_speed_error_rmse: float
    planned_speed_error_mean: float
    planned_speed_error_max: float
    planned_overspeed_rate: float
    acceleration_mean: float
    acceleration_max: float
    acceleration_p95: float
    jerk_mean: float
    jerk_max: float
    jerk_p95: float
    lateral_accel_p95: float
    lateral_jerk_p95: float
    lateral_jerk_max: float
    speed_limit_zero_rate: float
    speed_surge_count: int
    speed_surge_avg_drop: float
    speed_surge_p95_drop: float
    speed_surge_transition_count: int
    speed_surge_transition_avg_drop: float
    speed_surge_transition_p95_drop: float
    speed_surge_oscillation_count: int
    speed_governor_comfort_active_rate: float
    speed_governor_horizon_active_rate: float
    speed_governor_mean_comfort_speed: float
    speed_governor_mean_speed: float
    speed_surge_oscillation_avg_drop: float
    speed_surge_oscillation_p95_drop: float
    
    # Safety
    out_of_lane_events: int
    out_of_lane_time: float


class DriveAnalyzer:
    """Comprehensive drive analysis."""
    
    def __init__(self, recording_path: Path, stop_on_emergency: bool = True):
        """Initialize analyzer."""
        self.recording_path = recording_path
        self.stop_on_emergency = stop_on_emergency
        self.data = {}
        self.metrics = None
        self.config = self._load_config()
        self.emergency_stop_frame = None

    @staticmethod
    def _load_config() -> dict:
        config_path = project_root / "config" / "av_stack_config.yaml"
        if not config_path.exists():
            return {}
        try:
            with config_path.open("r") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}
        
    def load_data(self) -> bool:
        """Load all data from recording."""
        try:
            with h5py.File(self.recording_path, 'r') as f:
                # Vehicle state
                if 'vehicle_state/timestamp' in f:
                    self.data['timestamps'] = np.array(f['vehicle_state/timestamp'][:])
                    self.data['position'] = np.array(f['vehicle_state/position'][:]) if 'vehicle_state/position' in f else None
                    self.data['speed'] = np.array(f['vehicle_state/speed'][:]) if 'vehicle_state/speed' in f else None
                    self.data['heading'] = np.array(f['vehicle_state/heading'][:]) if 'vehicle_state/heading' in f else None
                    self.data['speed_limit'] = (
                        np.array(f['vehicle_state/speed_limit'][:])
                        if 'vehicle_state/speed_limit' in f else None
                    )
                elif 'vehicle/timestamps' in f:
                    self.data['timestamps'] = np.array(f['vehicle/timestamps'][:])
                    self.data['position'] = np.array(f['vehicle/position'][:]) if 'vehicle/position' in f else None
                    self.data['speed'] = np.array(f['vehicle/speed'][:]) if 'vehicle/speed' in f else None
                    self.data['speed_limit'] = (
                        np.array(f['vehicle/speed_limit'][:])
                        if 'vehicle/speed_limit' in f else None
                    )
                else:
                    # Fallback to control timestamps
                    self.data['timestamps'] = np.array(f['control/timestamp'][:])
                
                # Control data
                self.data['steering'] = np.array(f['control/steering'][:])
                self.data['throttle'] = np.array(f['control/throttle'][:]) if 'control/throttle' in f else None
                self.data['brake'] = np.array(f['control/brake'][:]) if 'control/brake' in f else None
                self.data['emergency_stop'] = (
                    np.array(f['control/emergency_stop'][:]) if 'control/emergency_stop' in f else None
                )
                self.data['lateral_error'] = np.array(f['control/lateral_error'][:]) if 'control/lateral_error' in f else None
                self.data['heading_error'] = np.array(f['control/heading_error'][:]) if 'control/heading_error' in f else None
                self.data['total_error'] = np.array(f['control/total_error'][:]) if 'control/total_error' in f else None
                self.data['path_curvature_input'] = np.array(f['control/path_curvature_input'][:]) if 'control/path_curvature_input' in f else None
                self.data['pid_integral'] = np.array(f['control/pid_integral'][:]) if 'control/pid_integral' in f else None
                self.data['pid_derivative'] = np.array(f['control/pid_derivative'][:]) if 'control/pid_derivative' in f else None
                self.data['target_speed_planned'] = (
                    np.array(f['control/target_speed_planned'][:])
                    if 'control/target_speed_planned' in f else None
                )
                
                # Trajectory data
                self.data['ref_x'] = np.array(f['trajectory/reference_point_x'][:]) if 'trajectory/reference_point_x' in f else None
                self.data['ref_y'] = np.array(f['trajectory/reference_point_y'][:]) if 'trajectory/reference_point_y' in f else None
                self.data['ref_heading'] = np.array(f['trajectory/reference_point_heading'][:]) if 'trajectory/reference_point_heading' in f else None
                self.data['ref_velocity'] = (
                    np.array(f['trajectory/reference_point_velocity'][:])
                    if 'trajectory/reference_point_velocity' in f else None
                )
                
                # Perception data
                self.data['left_lane_x'] = np.array(f['perception/left_lane_line_x'][:]) if 'perception/left_lane_line_x' in f else None
                self.data['right_lane_x'] = np.array(f['perception/right_lane_line_x'][:]) if 'perception/right_lane_line_x' in f else None
                self.data['num_lanes_detected'] = np.array(f['perception/num_lanes_detected'][:]) if 'perception/num_lanes_detected' in f else None
                self.data['confidence'] = np.array(f['perception/confidence'][:]) if 'perception/confidence' in f else None
                self.data['using_stale_data'] = np.array(f['perception/using_stale_data'][:]) if 'perception/using_stale_data' in f else None
                self.data['stale_reason'] = None
                if 'perception/stale_reason' in f:
                    # Handle string array
                    stale_reasons = f['perception/stale_reason'][:]
                    if len(stale_reasons) > 0:
                        self.data['stale_reason'] = [s.decode('utf-8') if isinstance(s, bytes) else s for s in stale_reasons]
                
                # Ground truth (if available)
                self.data['gt_left'] = np.array(f['ground_truth/left_lane_line_x'][:]) if 'ground_truth/left_lane_line_x' in f else None
                self.data['gt_right'] = np.array(f['ground_truth/right_lane_line_x'][:]) if 'ground_truth/right_lane_line_x' in f else None
                self.data['gt_center'] = np.array(f['ground_truth/lane_center_x'][:]) if 'ground_truth/lane_center_x' in f else None
                self.data['gt_path_curvature'] = (
                    np.array(f['ground_truth/path_curvature'][:])
                    if 'ground_truth/path_curvature' in f else None
                )
                
                # Calculate time axis
                if len(self.data['timestamps']) > 0:
                    self.data['time'] = self.data['timestamps'] - self.data['timestamps'][0]
                else:
                    self.data['time'] = np.arange(len(self.data['steering'])) * 0.033
                
                return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def calculate_metrics(self) -> DriveMetrics:
        """Calculate all metrics."""
        summary = analyze_recording_summary(
            self.recording_path,
            analyze_to_failure=bool(self.stop_on_emergency),
        )
        if isinstance(summary, dict) and not summary.get("error"):
            executive = summary.get("executive_summary", {})
            path_tracking = summary.get("path_tracking", {})
            control_smoothness = summary.get("control_smoothness", {})
            control_stability = summary.get("control_stability", {})
            perception_quality = summary.get("perception_quality", {})
            trajectory_quality = summary.get("trajectory_quality", {})
            system_health = summary.get("system_health", {})
            speed_control = summary.get("speed_control", {})
            comfort = summary.get("comfort", {})
            safety = summary.get("safety", {})

            def _f(section: Dict, key: str, default: float = 0.0) -> float:
                value = section.get(key, default)
                return float(value) if value is not None else float(default)

            def _i(section: Dict, key: str, default: int = 0) -> int:
                value = section.get(key, default)
                return int(value) if value is not None else int(default)

            if executive.get("failure_detected") and executive.get("failure_detection_source") == "emergency_stop":
                failure_frame = executive.get("failure_frame")
                self.emergency_stop_frame = int(failure_frame) if failure_frame is not None else None

            return DriveMetrics(
                drive_duration=_f(executive, "drive_duration"),
                total_frames=_i(executive, "total_frames"),
                success_rate=_f(executive, "success_rate"),
                lateral_error_rmse=_f(path_tracking, "lateral_error_rmse"),
                lateral_error_mean=_f(path_tracking, "lateral_error_mean"),
                lateral_error_max=_f(path_tracking, "lateral_error_max"),
                lateral_error_std=0.0,
                lateral_error_p50=0.0,
                lateral_error_p95=_f(path_tracking, "lateral_error_p95"),
                heading_error_rmse=_f(path_tracking, "heading_error_rmse"),
                heading_error_mean=0.0,
                heading_error_max=_f(path_tracking, "heading_error_max"),
                time_in_lane=_f(path_tracking, "time_in_lane"),
                time_in_lane_centered=_f(path_tracking, "time_in_lane_centered"),
                steering_jerk_mean=0.0,
                steering_jerk_max=_f(control_smoothness, "steering_jerk_max"),
                steering_rate_mean=0.0,
                steering_rate_max=_f(control_smoothness, "steering_rate_max"),
                steering_smoothness=_f(control_smoothness, "steering_smoothness"),
                oscillation_frequency=_f(control_smoothness, "oscillation_frequency"),
                control_effort=0.0,
                straight_frames=0,
                straight_oscillation_rate=_f(control_stability, "straight_oscillation_mean"),
                straight_stability_score=0.0,
                lane_detection_rate=_f(perception_quality, "lane_detection_rate"),
                perception_confidence_mean=_f(perception_quality, "perception_confidence_mean"),
                perception_confidence_std=0.0,
                perception_jumps_detected=_i(perception_quality, "perception_jumps_detected"),
                stale_perception_rate=_f(perception_quality, "stale_perception_rate"),
                perception_freeze_events=0,
                trajectory_availability=_f(trajectory_quality, "trajectory_availability"),
                ref_point_accuracy_rmse=_f(trajectory_quality, "ref_point_accuracy_rmse"),
                trajectory_smoothness=0.0,
                path_curvature_consistency=0.0,
                pid_integral_max=_f(system_health, "pid_integral_max"),
                pid_reset_frequency=0.0,
                error_conflict_rate=0.0,
                stale_command_rate=0.0,
                speed_error_rmse=_f(speed_control, "speed_error_rmse"),
                speed_error_mean=_f(speed_control, "speed_error_mean"),
                speed_error_max=_f(speed_control, "speed_error_max"),
                speed_overspeed_rate=_f(speed_control, "speed_overspeed_rate"),
                planned_speed_error_rmse=0.0,
                planned_speed_error_mean=0.0,
                planned_speed_error_max=0.0,
                planned_overspeed_rate=0.0,
                acceleration_mean=_f(speed_control, "acceleration_mean"),
                acceleration_max=_f(speed_control, "acceleration_max"),
                acceleration_p95=_f(speed_control, "acceleration_p95"),
                jerk_mean=_f(speed_control, "jerk_mean"),
                jerk_max=_f(speed_control, "jerk_max"),
                jerk_p95=_f(speed_control, "jerk_p95"),
                lateral_accel_p95=_f(comfort, "lateral_accel_p95"),
                lateral_jerk_p95=_f(comfort, "lateral_jerk_p95"),
                lateral_jerk_max=_f(speed_control, "lateral_jerk_max"),
                speed_limit_zero_rate=_f(speed_control, "speed_limit_zero_rate"),
                speed_surge_count=_i(speed_control, "speed_surge_count"),
                speed_surge_avg_drop=_f(speed_control, "speed_surge_avg_drop"),
                speed_surge_p95_drop=_f(speed_control, "speed_surge_p95_drop"),
                speed_surge_transition_count=0,
                speed_surge_transition_avg_drop=0.0,
                speed_surge_transition_p95_drop=0.0,
                speed_surge_oscillation_count=0,
                speed_governor_comfort_active_rate=0.0,
                speed_governor_horizon_active_rate=0.0,
                speed_governor_mean_comfort_speed=0.0,
                speed_governor_mean_speed=0.0,
                speed_surge_oscillation_avg_drop=0.0,
                speed_surge_oscillation_p95_drop=0.0,
                out_of_lane_events=_i(safety, "out_of_lane_events"),
                out_of_lane_time=_f(safety, "out_of_lane_time"),
            )

        if self.stop_on_emergency and self.data.get('emergency_stop') is not None:
            emergency_indices = np.where(self.data['emergency_stop'] > 0)[0]
            if emergency_indices.size > 0:
                self.emergency_stop_frame = int(emergency_indices[0])
                for key in self.data:
                    if isinstance(self.data[key], np.ndarray) and len(self.data[key]) > self.emergency_stop_frame:
                        self.data[key] = self.data[key][:self.emergency_stop_frame]

        n_frames = len(self.data['steering'])
        dt = np.mean(np.diff(self.data['time'])) if len(self.data['time']) > 1 else 0.033
        duration = self.data['time'][-1] if len(self.data['time']) > 0 else n_frames * dt
        if n_frames == 0:
            return DriveMetrics(
                drive_duration=0.0,
                total_frames=0,
                success_rate=0.0,
                lateral_error_rmse=0.0,
                lateral_error_mean=0.0,
                lateral_error_max=0.0,
                lateral_error_std=0.0,
                lateral_error_p50=0.0,
                lateral_error_p95=0.0,
                heading_error_rmse=0.0,
                heading_error_mean=0.0,
                heading_error_max=0.0,
                time_in_lane=0.0,
                time_in_lane_centered=0.0,
                steering_jerk_mean=0.0,
                steering_jerk_max=0.0,
                steering_rate_mean=0.0,
                steering_rate_max=0.0,
                steering_smoothness=0.0,
                oscillation_frequency=0.0,
                control_effort=0.0,
                straight_frames=0,
                straight_oscillation_rate=0.0,
                straight_stability_score=0.0,
                lane_detection_rate=0.0,
                perception_confidence_mean=0.0,
                perception_confidence_std=0.0,
                perception_jumps_detected=0,
                stale_perception_rate=0.0,
                perception_freeze_events=0,
                trajectory_availability=0.0,
                ref_point_accuracy_rmse=0.0,
                trajectory_smoothness=0.0,
                path_curvature_consistency=0.0,
                pid_integral_max=0.0,
                pid_reset_frequency=0.0,
                error_conflict_rate=0.0,
                stale_command_rate=0.0,
                speed_error_rmse=0.0,
                speed_error_mean=0.0,
                speed_error_max=0.0,
                speed_overspeed_rate=0.0,
                planned_speed_error_rmse=0.0,
                planned_speed_error_mean=0.0,
                planned_speed_error_max=0.0,
                planned_overspeed_rate=0.0,
                acceleration_mean=0.0,
                acceleration_max=0.0,
                acceleration_p95=0.0,
                jerk_mean=0.0,
                jerk_max=0.0,
                jerk_p95=0.0,
                lateral_accel_p95=0.0,
                lateral_jerk_p95=0.0,
                lateral_jerk_max=0.0,
                speed_limit_zero_rate=0.0,
                speed_surge_count=0,
                speed_surge_avg_drop=0.0,
                speed_surge_p95_drop=0.0,
                speed_surge_transition_count=0,
                speed_surge_transition_avg_drop=0.0,
                speed_surge_transition_p95_drop=0.0,
                speed_surge_oscillation_count=0,
                speed_surge_oscillation_avg_drop=0.0,
                speed_surge_oscillation_p95_drop=0.0,
                speed_governor_comfort_active_rate=0.0,
                speed_governor_horizon_active_rate=0.0,
                speed_governor_mean_comfort_speed=0.0,
                speed_governor_mean_speed=0.0,
                out_of_lane_events=0,
                out_of_lane_time=0.0
            )
        
        # 1. PATH TRACKING METRICS
        lateral_error_valid = self.data['lateral_error'] is not None and len(self.data['lateral_error']) > 0
        heading_error_valid = self.data['heading_error'] is not None and len(self.data['heading_error']) > 0

        lateral_error_rmse = np.sqrt(np.mean(self.data['lateral_error']**2)) if lateral_error_valid else 0.0
        lateral_error_mean = np.mean(np.abs(self.data['lateral_error'])) if lateral_error_valid else 0.0
        lateral_error_max = np.max(np.abs(self.data['lateral_error'])) if lateral_error_valid else 0.0
        lateral_error_std = np.std(self.data['lateral_error']) if lateral_error_valid else 0.0
        lateral_error_p50 = np.percentile(np.abs(self.data['lateral_error']), 50) if lateral_error_valid else 0.0
        lateral_error_p95 = np.percentile(np.abs(self.data['lateral_error']), 95) if lateral_error_valid else 0.0
        
        heading_error_rmse = np.sqrt(np.mean(self.data['heading_error']**2)) if heading_error_valid else 0.0
        heading_error_mean = np.mean(np.abs(self.data['heading_error'])) if heading_error_valid else 0.0
        heading_error_max = np.max(np.abs(self.data['heading_error'])) if heading_error_valid else 0.0
        
        # Time in lane (primary: within lane boundaries if GT available)
        time_in_lane_centered = (
            np.sum(np.abs(self.data['lateral_error']) < 0.5) / n_frames * 100
            if self.data['lateral_error'] is not None else 0.0
        )
        time_in_lane = time_in_lane_centered
        in_lane_mask = None
        if self.data['gt_left'] is not None and self.data['gt_right'] is not None:
            in_lane_mask = (self.data['gt_left'] <= 0.0) & (0.0 <= self.data['gt_right'])
            time_in_lane = np.sum(in_lane_mask) / n_frames * 100
        
        # 2. CONTROL SMOOTHNESS
        # Steering jerk (rate of change of steering rate)
        steering_rate = np.array([])
        steering_jerk = np.array([])
        if len(self.data['steering']) > 1 and len(self.data['time']) > 1:
            dt = np.diff(self.data['time'])
            valid = dt > 1e-6
            if np.any(valid):
                steering_rate = np.diff(self.data['steering'])[valid] / dt[valid]
                if len(steering_rate) > 1:
                    dt2 = np.diff(self.data['time'][1:])[valid[1:]]
                    valid2 = dt2 > 1e-6
                    if np.any(valid2):
                        steering_jerk = np.diff(steering_rate)[valid2] / dt2[valid2]
        steering_jerk_mean = np.mean(np.abs(steering_jerk)) if len(steering_jerk) > 0 else 0.0
        steering_jerk_max = np.max(np.abs(steering_jerk)) if len(steering_jerk) > 0 else 0.0
        
        steering_rate_mean = np.mean(np.abs(steering_rate)) if len(steering_rate) > 0 else 0.0
        steering_rate_max = np.max(np.abs(steering_rate)) if len(steering_rate) > 0 else 0.0
        
        # Steering smoothness (inverse of std)
        steering_std = np.std(self.data['steering'])
        steering_smoothness = 1.0 / (steering_std + 1e-6)
        
        # Oscillation frequency (FFT analysis)
        oscillation_frequency = 0.0
        if self.data['lateral_error'] is not None and len(self.data['lateral_error']) > 10:
            error_centered = self.data['lateral_error'] - np.mean(self.data['lateral_error'])
            time_diffs = np.diff(self.data['time'])
            valid_diffs = time_diffs[time_diffs > 1e-6]
            dt_mean = float(np.mean(valid_diffs)) if len(valid_diffs) > 0 else 0.0
            if dt_mean > 0.0:
                fft_vals = fft(error_centered)
                fft_freqs = fftfreq(len(error_centered), dt_mean)
                positive_freqs = fft_freqs[:len(fft_freqs)//2]
                positive_fft = np.abs(fft_vals[:len(fft_vals)//2])
                if len(positive_fft) > 1:
                    dominant_idx = np.argmax(positive_fft[1:]) + 1
                    oscillation_frequency = positive_freqs[dominant_idx] if dominant_idx < len(positive_freqs) else 0.0
        
        # Control effort (integral of |steering|)
        control_effort = np.trapezoid(np.abs(self.data['steering']), self.data['time'])
        
        # 3. PERCEPTION QUALITY
        lane_detection_rate = np.sum(self.data['num_lanes_detected'] >= 2) / n_frames * 100 if self.data['num_lanes_detected'] is not None else 0.0
        perception_confidence_mean = np.mean(self.data['confidence']) if self.data['confidence'] is not None else 0.0
        perception_confidence_std = np.std(self.data['confidence']) if self.data['confidence'] is not None else 0.0
        
        # Count perception jumps (stale data due to jump detection)
        perception_jumps_detected = 0
        if self.data['stale_reason'] is not None:
            perception_jumps_detected = sum(1 for r in self.data['stale_reason'] if r and 'jump' in str(r).lower())
        
        stale_perception_rate = np.sum(self.data['using_stale_data']) / n_frames * 100 if self.data['using_stale_data'] is not None else 0.0
        
        # Perception freeze (consecutive frames with same values)
        perception_freeze_events = 0
        if self.data['left_lane_x'] is not None and self.data['right_lane_x'] is not None:
            left_changes = np.sum(np.diff(self.data['left_lane_x']) != 0)
            right_changes = np.sum(np.diff(self.data['right_lane_x']) != 0)
            # If no changes for extended period, it's frozen
            if left_changes == 0 or right_changes == 0:
                perception_freeze_events = 1
        
        # 4. TRAJECTORY QUALITY
        trajectory_availability = np.sum(~np.isnan(self.data['ref_x'])) / n_frames * 100 if self.data['ref_x'] is not None else 0.0
        
        # Reference point accuracy (vs ground truth)
        ref_point_accuracy_rmse = 0.0
        if self.data['ref_x'] is not None and self.data['gt_center'] is not None:
            valid_mask = ~np.isnan(self.data['ref_x']) & ~np.isnan(self.data['gt_center'])
            if np.sum(valid_mask) > 0:
                errors = self.data['ref_x'][valid_mask] - self.data['gt_center'][valid_mask]
                ref_point_accuracy_rmse = np.sqrt(np.mean(errors**2))
        
        # Trajectory smoothness (curvature change rate)
        trajectory_smoothness = 0.0
        if self.data['ref_heading'] is not None:
            curvature_changes = np.diff(self.data['ref_heading'])
            trajectory_smoothness = 1.0 / (np.std(curvature_changes) + 1e-6)

        # Straight-away stability (steering sign changes on straight segments)
        straight_frames = 0
        straight_oscillation_rate = 0.0
        straight_stability_score = 0.0
        curvature_source = self.data['path_curvature_input']
        if curvature_source is None:
            curvature_source = self.data['ref_heading']
        if curvature_source is not None and self.data['steering'] is not None:
            straight_mask = np.abs(curvature_source) < 0.02
            straight_frames = int(np.sum(straight_mask))
            if straight_frames > 2:
                steering_straight = self.data['steering'][straight_mask]
                sign_changes = np.sum(np.sign(steering_straight[1:]) != np.sign(steering_straight[:-1]))
                straight_oscillation_rate = sign_changes / max(1, len(steering_straight) - 1)
                straight_stability_score = max(0.0, 1.0 - min(1.0, straight_oscillation_rate * 4.0))
        
        # Path curvature consistency
        path_curvature_consistency = 1.0 / (np.std(self.data['ref_heading']) + 1e-6) if self.data['ref_heading'] is not None else 0.0
        
        # 5. SYSTEM HEALTH
        pid_integral_max = np.max(np.abs(self.data['pid_integral'])) if self.data['pid_integral'] is not None else 0.0
        
        # PID reset frequency (count sign changes in integral)
        pid_reset_frequency = 0.0
        if self.data['pid_integral'] is not None and len(self.data['pid_integral']) > 1:
            integral_sign_changes = np.sum(np.diff(np.sign(self.data['pid_integral'])) != 0)
            pid_reset_frequency = integral_sign_changes / duration
        
        # Error conflicts (heading vs lateral have opposite signs)
        error_conflict_rate = 0.0
        if self.data['heading_error'] is not None and self.data['lateral_error'] is not None:
            conflicts = (self.data['heading_error'] > 0) & (self.data['lateral_error'] < 0) | \
                       (self.data['heading_error'] < 0) & (self.data['lateral_error'] > 0)
            error_conflict_rate = np.sum(conflicts) / n_frames * 100
        
        # Stale command rate
        stale_command_rate = 0.0
        if 'control/using_stale_perception' in h5py.File(self.recording_path, 'r'):
            with h5py.File(self.recording_path, 'r') as f:
                stale_commands = np.array(f['control/using_stale_perception'][:])
                stale_command_rate = np.sum(stale_commands) / n_frames * 100

        # 5.5 SPEED CONTROL + COMFORT
        speed_error_rmse = 0.0
        speed_error_mean = 0.0
        speed_error_max = 0.0
        speed_overspeed_rate = 0.0
        planned_speed_error_rmse = 0.0
        planned_speed_error_mean = 0.0
        planned_speed_error_max = 0.0
        planned_overspeed_rate = 0.0
        if self.data['speed'] is not None and self.data['ref_velocity'] is not None:
            n_speed = min(len(self.data['speed']), len(self.data['ref_velocity']))
            if n_speed > 0:
                speed = self.data['speed'][:n_speed]
                ref_speed = self.data['ref_velocity'][:n_speed]
                speed_error = speed - ref_speed
                speed_error_rmse = float(np.sqrt(np.mean(speed_error ** 2)))
                speed_error_mean = float(np.mean(np.abs(speed_error)))
                speed_error_max = float(np.max(np.abs(speed_error)))
                overspeed_threshold = 0.5
                speed_overspeed_rate = float(np.sum(speed_error > overspeed_threshold) / n_speed * 100)
        if self.data['speed'] is not None and self.data.get('target_speed_planned') is not None:
            n_plan = min(len(self.data['speed']), len(self.data['target_speed_planned']))
            if n_plan > 0:
                speed = self.data['speed'][:n_plan]
                planned = self.data['target_speed_planned'][:n_plan]
                planned_error = speed - planned
                planned_speed_error_rmse = float(np.sqrt(np.mean(planned_error ** 2)))
                planned_speed_error_mean = float(np.mean(np.abs(planned_error)))
                planned_speed_error_max = float(np.max(np.abs(planned_error)))
                overspeed_threshold = 0.5
                planned_overspeed_rate = float(
                    np.sum(planned_error > overspeed_threshold) / n_plan * 100
                )

        acceleration_mean = 0.0
        acceleration_max = 0.0
        acceleration_p95 = 0.0
        jerk_mean = 0.0
        jerk_max = 0.0
        jerk_p95 = 0.0
        lateral_accel_p95 = 0.0
        lateral_jerk_p95 = 0.0
        lateral_jerk_max = 0.0
        speed_limit_zero_rate = 0.0
        speed_surge_count = 0
        speed_surge_avg_drop = 0.0
        speed_surge_p95_drop = 0.0
        speed_surge_transition_count = 0
        speed_surge_transition_avg_drop = 0.0
        speed_surge_transition_p95_drop = 0.0
        speed_surge_oscillation_count = 0
        speed_surge_oscillation_avg_drop = 0.0
        speed_surge_oscillation_p95_drop = 0.0
        traj_cfg = self.config.get('trajectory', {})
        curvature_smoothing_enabled = bool(traj_cfg.get('curvature_smoothing_enabled', False))
        curvature_window_m = float(traj_cfg.get('curvature_smoothing_window_m', 12.0))
        curvature_min_speed = float(traj_cfg.get('curvature_smoothing_min_speed', 2.0))
        if self.data['speed'] is not None and len(self.data['speed']) > 1 and len(self.data['time']) > 1:
            dt_series = np.diff(self.data['time'])
            positive_dt = dt_series[dt_series > 1e-6]
            fallback_dt = float(np.mean(positive_dt)) if len(positive_dt) > 0 else 0.033
            dt_series[dt_series <= 1e-6] = fallback_dt
            # Filter speed to reduce derivative noise in accel/jerk metrics.
            alpha = 0.7
            filtered_speed = np.empty_like(self.data['speed'])
            filtered_speed[0] = self.data['speed'][0]
            for i in range(1, len(self.data['speed'])):
                filtered_speed[i] = alpha * filtered_speed[i - 1] + (1.0 - alpha) * self.data['speed'][i]
            acceleration = np.diff(filtered_speed) / dt_series
            if acceleration.size > 0:
                abs_accel = np.abs(acceleration)
                acceleration_mean = float(np.mean(abs_accel))
                acceleration_max = float(np.max(abs_accel))
                acceleration_p95 = float(np.percentile(abs_accel, 95))
            if acceleration.size > 1:
                jerk = np.diff(acceleration) / dt_series[1:]
                if jerk.size > 0:
                    abs_jerk = np.abs(jerk)
                    jerk_mean = float(np.mean(abs_jerk))
                    jerk_max = float(np.max(abs_jerk))
                    jerk_p95 = float(np.percentile(abs_jerk, 95))
            curvature = None
            if self.data.get('gt_path_curvature') is not None:
                curvature = self.data['gt_path_curvature']
            elif self.data.get('path_curvature_input') is not None:
                curvature = self.data['path_curvature_input']
            if curvature is not None:
                n_lat = min(len(curvature), len(filtered_speed))
                if n_lat > 1:
                    curvature_source = curvature[:n_lat]
                    if curvature_smoothing_enabled:
                        curvature_source = np.array(
                            smooth_curvature_distance(
                                curvature_source,
                                filtered_speed[:n_lat],
                                self.data['time'][:n_lat],
                                curvature_window_m,
                                curvature_min_speed,
                            )
                        )
                    lat_accel = (filtered_speed[:n_lat] ** 2) * curvature_source
                    abs_lat_accel = np.abs(lat_accel)
                    lateral_accel_p95 = float(np.percentile(abs_lat_accel, 95))
                    lat_dt = np.diff(self.data['time'][:n_lat])
                    lat_positive = lat_dt[lat_dt > 1e-6]
                    lat_fallback = float(np.mean(lat_positive)) if len(lat_positive) > 0 else 0.033
                    lat_dt[lat_dt <= 1e-6] = lat_fallback
                    lat_jerk = np.diff(lat_accel) / lat_dt
                    if lat_jerk.size > 0:
                        abs_lat_jerk = np.abs(lat_jerk)
                        lateral_jerk_p95 = float(np.percentile(abs_lat_jerk, 95))
                        lateral_jerk_max = float(np.max(abs_lat_jerk))
        if self.data.get('speed_limit') is not None and len(self.data['speed_limit']) > 0:
            speed_limit_zero_rate = float(
                np.sum(self.data['speed_limit'] <= 0.01) / len(self.data['speed_limit']) * 100
            )
        if self.data['speed'] is not None:
            curvature_series = None
            if self.data.get('gt_path_curvature') is not None:
                curvature_series = self.data['gt_path_curvature']
            elif self.data.get('path_curvature_input') is not None:
                curvature_series = self.data['path_curvature_input']
            if curvature_series is not None:
                n_surge = min(len(self.data['speed']), len(curvature_series))
                speed_series = self.data['speed'][:n_surge]
                curvature_series = curvature_series[:n_surge]
                straight_threshold = float(traj_cfg.get('straight_speed_smoothing_curvature_threshold', 0.003))
                min_drop = float(traj_cfg.get('speed_surge_min_drop', 1.0))
                straight_mask = np.abs(curvature_series) <= straight_threshold
                drops: list[float] = []
                transition_drops: list[float] = []
                oscillation_drops: list[float] = []
                lookback_seconds = float(traj_cfg.get('speed_surge_transition_lookback_seconds', 1.0))
                i = 1
                while i < len(speed_series) - 1:
                    if not straight_mask[i]:
                        i += 1
                        continue
                    if speed_series[i] > speed_series[i - 1] and speed_series[i] >= speed_series[i + 1]:
                        max_idx = i
                        j = i + 1
                        while j < len(speed_series) - 1 and straight_mask[j]:
                            if speed_series[j] <= speed_series[j - 1] and speed_series[j] < speed_series[j + 1]:
                                drop = float(speed_series[max_idx] - speed_series[j])
                                if drop >= min_drop:
                                    drops.append(drop)
                                    if self.data.get('time') is not None and lookback_seconds > 0.0:
                                        time_series = self.data['time'][:n_surge]
                                        max_time = time_series[max_idx]
                                        min_time = max_time - lookback_seconds
                                        lookback_mask = (
                                            (time_series >= min_time)
                                            & (time_series <= max_time)
                                        )
                                        if np.any(np.abs(curvature_series[lookback_mask]) > straight_threshold):
                                            transition_drops.append(drop)
                                        else:
                                            oscillation_drops.append(drop)
                                break
                            j += 1
                        i = j
                    else:
                        i += 1
                if drops:
                    speed_surge_count = len(drops)
                    speed_surge_avg_drop = float(np.mean(drops))
                    speed_surge_p95_drop = float(np.percentile(drops, 95))
                if transition_drops:
                    speed_surge_transition_count = len(transition_drops)
                    speed_surge_transition_avg_drop = float(np.mean(transition_drops))
                    speed_surge_transition_p95_drop = float(np.percentile(transition_drops, 95))
                if oscillation_drops:
                    speed_surge_oscillation_count = len(oscillation_drops)
                    speed_surge_oscillation_avg_drop = float(np.mean(oscillation_drops))
                    speed_surge_oscillation_p95_drop = float(np.percentile(oscillation_drops, 95))
        
        # 5.6 SPEED GOVERNOR METRICS
        speed_governor_comfort_active_rate = 0.0
        speed_governor_horizon_active_rate = 0.0
        speed_governor_mean_comfort_speed = 0.0
        speed_governor_mean_speed = 0.0
        try:
            with h5py.File(self.recording_path, 'r') as f:
                if 'control/speed_governor_comfort_speed' in f:
                    comfort_arr = np.array(f['control/speed_governor_comfort_speed'][:n_frames])
                    valid = comfort_arr[comfort_arr > 0]
                    if len(valid) > 0:
                        speed_governor_comfort_active_rate = float(len(valid) / n_frames * 100)
                        speed_governor_mean_comfort_speed = float(np.mean(valid))
                if 'control/speed_governor_horizon_speed' in f:
                    horizon_arr = np.array(f['control/speed_governor_horizon_speed'][:n_frames])
                    valid_h = horizon_arr[horizon_arr > 0]
                    if len(valid_h) > 0:
                        speed_governor_horizon_active_rate = float(len(valid_h) / n_frames * 100)
                if self.data['speed'] is not None:
                    speed_governor_mean_speed = float(np.mean(self.data['speed'][:n_frames]))
        except Exception:
            pass

        # 6. SAFETY METRICS
        # Out of lane events (lateral error > 1.0m, typical lane width is ~3.5m, so 1.0m is significant)
        if in_lane_mask is not None:
            out_of_lane_mask = ~in_lane_mask
        else:
            out_of_lane_mask = np.abs(self.data['lateral_error']) > 1.0 if self.data['lateral_error'] is not None else np.zeros(n_frames, dtype=bool)
        out_of_lane_events = len(np.where(np.diff(out_of_lane_mask.astype(int)) > 0)[0])  # Count transitions into out-of-lane
        out_of_lane_time = np.sum(out_of_lane_mask) / n_frames * 100 if self.data['lateral_error'] is not None else 0.0
        
        # Success rate (time in lane)
        success_rate = time_in_lane
        
        return DriveMetrics(
            drive_duration=duration,
            total_frames=n_frames,
            success_rate=success_rate,
            lateral_error_rmse=lateral_error_rmse,
            lateral_error_mean=lateral_error_mean,
            lateral_error_max=lateral_error_max,
            lateral_error_std=lateral_error_std,
            lateral_error_p50=lateral_error_p50,
            lateral_error_p95=lateral_error_p95,
            heading_error_rmse=heading_error_rmse,
            heading_error_mean=heading_error_mean,
            heading_error_max=heading_error_max,
            time_in_lane=time_in_lane,
            time_in_lane_centered=time_in_lane_centered,
            steering_jerk_mean=steering_jerk_mean,
            steering_jerk_max=steering_jerk_max,
            steering_rate_mean=steering_rate_mean,
            steering_rate_max=steering_rate_max,
            steering_smoothness=steering_smoothness,
            oscillation_frequency=oscillation_frequency,
            control_effort=control_effort,
            straight_frames=straight_frames,
            straight_oscillation_rate=straight_oscillation_rate,
            straight_stability_score=straight_stability_score,
            lane_detection_rate=lane_detection_rate,
            perception_confidence_mean=perception_confidence_mean,
            perception_confidence_std=perception_confidence_std,
            perception_jumps_detected=perception_jumps_detected,
            stale_perception_rate=stale_perception_rate,
            perception_freeze_events=perception_freeze_events,
            trajectory_availability=trajectory_availability,
            ref_point_accuracy_rmse=ref_point_accuracy_rmse,
            trajectory_smoothness=trajectory_smoothness,
            path_curvature_consistency=path_curvature_consistency,
            pid_integral_max=pid_integral_max,
            pid_reset_frequency=pid_reset_frequency,
            error_conflict_rate=error_conflict_rate,
            stale_command_rate=stale_command_rate,
            speed_error_rmse=speed_error_rmse,
            speed_error_mean=speed_error_mean,
            speed_error_max=speed_error_max,
            speed_overspeed_rate=speed_overspeed_rate,
            planned_speed_error_rmse=planned_speed_error_rmse,
            planned_speed_error_mean=planned_speed_error_mean,
            planned_speed_error_max=planned_speed_error_max,
            planned_overspeed_rate=planned_overspeed_rate,
            acceleration_mean=acceleration_mean,
            acceleration_max=acceleration_max,
            acceleration_p95=acceleration_p95,
            jerk_mean=jerk_mean,
            jerk_max=jerk_max,
            jerk_p95=jerk_p95,
            lateral_accel_p95=lateral_accel_p95,
            lateral_jerk_p95=lateral_jerk_p95,
            lateral_jerk_max=lateral_jerk_max,
            speed_limit_zero_rate=speed_limit_zero_rate,
            speed_surge_count=speed_surge_count,
            speed_surge_avg_drop=speed_surge_avg_drop,
            speed_surge_p95_drop=speed_surge_p95_drop,
            speed_surge_transition_count=speed_surge_transition_count,
            speed_surge_transition_avg_drop=speed_surge_transition_avg_drop,
            speed_surge_transition_p95_drop=speed_surge_transition_p95_drop,
            speed_surge_oscillation_count=speed_surge_oscillation_count,
            speed_surge_oscillation_avg_drop=speed_surge_oscillation_avg_drop,
            speed_surge_oscillation_p95_drop=speed_surge_oscillation_p95_drop,
            speed_governor_comfort_active_rate=speed_governor_comfort_active_rate,
            speed_governor_horizon_active_rate=speed_governor_horizon_active_rate,
            speed_governor_mean_comfort_speed=speed_governor_mean_comfort_speed,
            speed_governor_mean_speed=speed_governor_mean_speed,
            out_of_lane_events=out_of_lane_events,
            out_of_lane_time=out_of_lane_time
        )
    
    def _print_estop_root_cause(self):
        """Print root cause attribution for emergency stop (Step 3D)."""
        if self.emergency_stop_frame is None:
            return
        fr = self.emergency_stop_frame
        try:
            import h5py
            with h5py.File(self.recording_path, "r") as f:
                # Gather signals at the e-stop frame
                lat_err = float(np.array(f["control/lateral_error"][fr])) if "control/lateral_error" in f and fr < len(f["control/lateral_error"]) else None
                speed = float(np.array(f["vehicle/speed"][fr])) if "vehicle/speed" in f and fr < len(f["vehicle/speed"]) else None
                regime = float(np.array(f["control/regime"][fr])) if "control/regime" in f and fr < len(f["control/regime"]) else None

                # Check GT boundary values for corruption
                gt_corrupt = False
                gt_details = []
                for gt_key in ("vehicle/gt_left_boundary", "vehicle/gt_right_boundary",
                               "vehicle/groundTruthLeftBoundary", "vehicle/groundTruthRightBoundary"):
                    if gt_key in f and fr < len(f[gt_key]):
                        val = float(np.array(f[gt_key][fr]))
                        if abs(val) > 50.0:
                            gt_corrupt = True
                            gt_details.append(f"{gt_key.split('/')[-1]}={val:.1f}m")

                # Determine root cause
                print("   E-Stop Root Cause Attribution:")
                if gt_corrupt:
                    print(f"     → GT BOUNDARY CORRUPT: {', '.join(gt_details)}")
                    print(f"       (values >50m indicate Unity GroundTruthReporter glitch)")
                    if lat_err is not None:
                        print(f"       Actual lat_err at e-stop: {lat_err:.3f}m")
                if lat_err is not None and abs(lat_err) > 1.5:
                    print(f"     → LATERAL DIVERGENCE: lat_err={lat_err:.3f}m (threshold ~1.5m)")
                if regime is not None and regime >= 0.5:
                    regime_name = "LINEAR_MPC" if regime < 1.5 else "NONLINEAR_MPC"
                    print(f"     → REGIME: {regime_name} was active at e-stop frame")
                if speed is not None:
                    print(f"     Speed at e-stop: {speed:.1f} m/s")
                if not gt_corrupt and (lat_err is None or abs(lat_err) <= 1.5):
                    print(f"     → UNKNOWN: lat_err={lat_err}, no GT corruption detected — check recording manually")
        except Exception as e:
            print(f"   E-Stop Root Cause: error reading HDF5 — {e}")

    def print_report(self):
        """Print comprehensive analysis report."""
        if not self.load_data():
            print("Failed to load data from recording")
            return
        
        self.metrics = self.calculate_metrics()
        
        print("=" * 80)
        print("OVERALL DRIVE ANALYSIS REPORT")
        print("=" * 80)
        print(f"Recording: {self.recording_path.name}")
        print()
        
        # 1. EXECUTIVE SUMMARY
        print("1. EXECUTIVE SUMMARY")
        print("-" * 80)
        print(f"   Drive Duration: {self.metrics.drive_duration:.2f} seconds")
        print(f"   Total Frames: {self.metrics.total_frames}")
        print(f"   Success Rate: {self.metrics.success_rate:.1f}% (time in lane)")
        if self.emergency_stop_frame is not None:
            suffix = " (analysis truncated)" if self.stop_on_emergency else ""
            print(f"   Emergency Stop Frame: {self.emergency_stop_frame}{suffix}")
            # E-stop root cause attribution (Step 3D)
            self._print_estop_root_cause()
        print()
        
        # Overall score (weighted combination)
        score = (
            self.metrics.success_rate * 0.4 +
            (100 - min(self.metrics.lateral_error_rmse * 20, 100)) * 0.3 +
            (100 - min(self.metrics.steering_jerk_max * 100, 100)) * 0.2 +
            self.metrics.lane_detection_rate * 0.1
        )
        print(f"   Overall Score: {score:.1f}/100")
        print()
        
        # Key issues
        issues = []
        if self.metrics.success_rate < 80:
            issues.append(f"Low success rate ({self.metrics.success_rate:.1f}%)")
        if self.metrics.lateral_error_rmse > 0.5:
            issues.append(f"High lateral error (RMSE: {self.metrics.lateral_error_rmse:.3f}m)")
        if self.metrics.steering_jerk_max > 1.0:
            issues.append(f"Jerky steering (max jerk: {self.metrics.steering_jerk_max:.3f})")
        if self.metrics.lane_detection_rate < 90:
            issues.append(f"Low detection rate ({self.metrics.lane_detection_rate:.1f}%)")
        if self.metrics.stale_perception_rate > 10:
            issues.append(f"High stale data usage ({self.metrics.stale_perception_rate:.1f}%)")
        if self.metrics.speed_limit_zero_rate > 10:
            issues.append(f"Speed limit missing ({self.metrics.speed_limit_zero_rate:.1f}%)")
        if self.metrics.acceleration_p95 > 2.5:
            issues.append("High longitudinal acceleration")
        if self.metrics.jerk_p95 > 5.0:
            issues.append("High longitudinal jerk")
        if self.emergency_stop_frame is not None:
            issues.append(f"Emergency stop at frame {self.emergency_stop_frame}")
        
        if issues:
            print("   Key Issues:")
            for i, issue in enumerate(issues[:5], 1):
                print(f"     {i}. {issue}")
        else:
            print("   ✓ No major issues detected")
        print()
        
        # 2. PATH TRACKING PERFORMANCE
        print("2. PATH TRACKING PERFORMANCE")
        print("-" * 80)
        print(f"   Lateral Error:")
        print(f"     RMSE: {self.metrics.lateral_error_rmse:.4f} m")
        print(f"     Mean: {self.metrics.lateral_error_mean:.4f} m")
        print(f"     Max:  {self.metrics.lateral_error_max:.4f} m")
        print(f"     Std:  {self.metrics.lateral_error_std:.4f} m")
        print(f"     P50:  {self.metrics.lateral_error_p50:.4f} m")
        print(f"     P95:  {self.metrics.lateral_error_p95:.4f} m")
        print()
        print(f"   Heading Error:")
        print(f"     RMSE: {self.metrics.heading_error_rmse:.4f} rad ({np.degrees(self.metrics.heading_error_rmse):.2f}°)")
        print(f"     Mean: {self.metrics.heading_error_mean:.4f} rad ({np.degrees(self.metrics.heading_error_mean):.2f}°)")
        print(f"     Max:  {self.metrics.heading_error_max:.4f} rad ({np.degrees(self.metrics.heading_error_max):.2f}°)")
        print()
        print(f"   Time in Lane (boundaries): {self.metrics.time_in_lane:.1f}%")
        print(f"   Centeredness (±0.5m): {self.metrics.time_in_lane_centered:.1f}%")
        print()
        
        # 3. CONTROL SMOOTHNESS
        print("3. CONTROL SMOOTHNESS")
        print("-" * 80)
        print(f"   Steering Jerk:")
        print(f"     Mean: {self.metrics.steering_jerk_mean:.4f} per s²")
        print(f"     Max:  {self.metrics.steering_jerk_max:.4f} per s²")
        print()
        print(f"   Steering Rate:")
        print(f"     Mean: {self.metrics.steering_rate_mean:.4f} per s")
        print(f"     Max:  {self.metrics.steering_rate_max:.4f} per s")
        print()
        print(f"   Steering Smoothness: {self.metrics.steering_smoothness:.2f} (higher is better)")
        print(f"   Oscillation Frequency: {self.metrics.oscillation_frequency:.2f} Hz")
        print(f"   Control Effort: {self.metrics.control_effort:.2f}")
        print()
        print(f"   Straight-Away Stability:")
        print(f"     Straight Frames: {self.metrics.straight_frames}")
        print(f"     Steering Sign-Change Rate: {self.metrics.straight_oscillation_rate:.3f}")
        print(f"     Stability Score: {self.metrics.straight_stability_score:.2f} (higher is better)")
        print()
        
        # 4. SPEED CONTROL
        print("4. SPEED CONTROL")
        print("-" * 80)
        print(f"   Speed Tracking Error:")
        print(f"     RMSE: {self.metrics.speed_error_rmse:.3f} m/s")
        print(f"     Mean: {self.metrics.speed_error_mean:.3f} m/s")
        print(f"     Max:  {self.metrics.speed_error_max:.3f} m/s")
        print(f"     Overspeed Rate (>0.5 m/s): {self.metrics.speed_overspeed_rate:.1f}%")
        if self.metrics.planned_speed_error_rmse > 0.0:
            print(f"   Planned Speed Tracking Error:")
            print(f"     RMSE: {self.metrics.planned_speed_error_rmse:.3f} m/s")
            print(f"     Mean: {self.metrics.planned_speed_error_mean:.3f} m/s")
            print(f"     Max:  {self.metrics.planned_speed_error_max:.3f} m/s")
            print(f"     Overspeed Rate (>0.5 m/s): {self.metrics.planned_overspeed_rate:.1f}%")
        print()
        print(f"   Speed Limit Missing: {self.metrics.speed_limit_zero_rate:.1f}%")
        print(f"   Straight Surge Drops (>=1.0 m/s): {self.metrics.speed_surge_count} "
              f"(avg {self.metrics.speed_surge_avg_drop:.2f}, p95 {self.metrics.speed_surge_p95_drop:.2f})")
        print(f"     - Transition surges: {self.metrics.speed_surge_transition_count} "
              f"(avg {self.metrics.speed_surge_transition_avg_drop:.2f}, "
              f"p95 {self.metrics.speed_surge_transition_p95_drop:.2f})")
        print(f"     - Oscillation surges: {self.metrics.speed_surge_oscillation_count} "
              f"(avg {self.metrics.speed_surge_oscillation_avg_drop:.2f}, "
              f"p95 {self.metrics.speed_surge_oscillation_p95_drop:.2f})")
        print()
        
        # 5. COMFORT
        print("5. COMFORT")
        print("-" * 80)
        print(f"   Longitudinal Acceleration:")
        print(f"     Mean: {self.metrics.acceleration_mean:.3f} m/s²")
        print(f"     P95:  {self.metrics.acceleration_p95:.3f} m/s²")
        print(f"     Max:  {self.metrics.acceleration_max:.3f} m/s²")
        print()
        print(f"   Longitudinal Jerk:")
        print(f"     Mean: {self.metrics.jerk_mean:.3f} m/s³")
        print(f"     P95:  {self.metrics.jerk_p95:.3f} m/s³")
        print(f"     Max:  {self.metrics.jerk_max:.3f} m/s³")
        print()
        print(f"   Lateral Accel/Jerk (from curvature):")
        print(f"     Accel P95: {self.metrics.lateral_accel_p95:.3f} m/s²")
        print(f"     Jerk P95:  {self.metrics.lateral_jerk_p95:.3f} m/s³")
        print(f"     Jerk Max:  {self.metrics.lateral_jerk_max:.3f} m/s³")
        print()
        print(f"   Steering Jerk:")
        print(f"     Mean: {self.metrics.steering_jerk_mean:.3f} per s²")
        print(f"     Max:  {self.metrics.steering_jerk_max:.3f} per s²")
        print()
        
        # 6. PERCEPTION QUALITY
        print("6. PERCEPTION QUALITY")
        print("-" * 80)
        print(f"   Lane Detection Rate: {self.metrics.lane_detection_rate:.1f}%")
        print(f"   Confidence:")
        print(f"     Mean: {self.metrics.perception_confidence_mean:.3f}")
        print(f"     Std:  {self.metrics.perception_confidence_std:.3f}")
        print()
        print(f"   Perception Jumps Detected: {self.metrics.perception_jumps_detected}")
        print(f"   Stale Perception Rate: {self.metrics.stale_perception_rate:.1f}%")
        print(f"   Perception Freeze Events: {self.metrics.perception_freeze_events}")
        print()
        
        # 7. TRAJECTORY QUALITY
        print("7. TRAJECTORY QUALITY")
        print("-" * 80)
        print(f"   Trajectory Availability: {self.metrics.trajectory_availability:.1f}%")
        print(f"   Reference Point Accuracy (RMSE): {self.metrics.ref_point_accuracy_rmse:.4f} m")
        print(f"   Trajectory Smoothness: {self.metrics.trajectory_smoothness:.2f} (higher is better)")
        print(f"   Path Curvature Consistency: {self.metrics.path_curvature_consistency:.2f} (higher is better)")
        print()
        
        # 8. SYSTEM HEALTH
        print("8. SYSTEM HEALTH")
        print("-" * 80)
        print(f"   PID Integral Max: {self.metrics.pid_integral_max:.4f}")
        print(f"   PID Reset Frequency: {self.metrics.pid_reset_frequency:.2f} per second")
        print(f"   Error Conflict Rate: {self.metrics.error_conflict_rate:.1f}%")
        print(f"   Stale Command Rate: {self.metrics.stale_command_rate:.1f}%")
        print()
        
        # 9. SAFETY METRICS
        print("9. SAFETY METRICS")
        print("-" * 80)
        print(f"   Out-of-Lane Events: {self.metrics.out_of_lane_events}")
        print(f"   Out-of-Lane Time: {self.metrics.out_of_lane_time:.1f}%")
        print()
        
        # 10. RECOMMENDATIONS
        print("10. RECOMMENDATIONS")
        print("-" * 80)
        recommendations = []
        
        if self.metrics.lateral_error_rmse > 0.3:
            recommendations.append("Reduce lateral error - check PID gains or trajectory planning")
        if self.metrics.steering_jerk_max > 1.0:
            recommendations.append("Reduce steering jerk - increase rate limiting or reduce PID gains")
        if self.metrics.oscillation_frequency > 2.0:
            recommendations.append("Reduce oscillation - increase damping or reduce proportional gain")
        if self.metrics.lane_detection_rate < 90:
            recommendations.append("Improve lane detection - check perception model or CV fallback")
        if self.metrics.stale_perception_rate > 10:
            recommendations.append("Reduce stale data usage - relax jump detection threshold or improve perception")
        if self.metrics.speed_limit_zero_rate > 10:
            recommendations.append("Speed limit missing - verify Unity track speed limits are sent to the bridge")
        if self.metrics.error_conflict_rate > 20:
            recommendations.append("Reduce error conflicts - check heading/lateral error weighting")
        if self.metrics.pid_integral_max > 0.2:
            recommendations.append("Reduce PID integral accumulation - check integral reset mechanisms")
        if self.metrics.acceleration_p95 > 2.5:
            recommendations.append("Reduce longitudinal acceleration spikes - tune throttle/brake gains")
        if self.metrics.jerk_p95 > 5.0:
            recommendations.append("Reduce longitudinal jerk - add rate limiting on throttle/brake")
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        else:
            print("   ✓ System performing well - no major recommendations")
        print()
        
        print("=" * 80)


def list_recordings():
    """List available recordings."""
    recordings_dir = Path("data/recordings")
    if not recordings_dir.exists():
        print("No recordings directory found")
        return
    
    recordings = sorted(recordings_dir.glob("*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not recordings:
        print("No recordings found")
        return
    
    print("Available recordings:")
    print("-" * 80)
    for i, rec in enumerate(recordings[:20], 1):  # Show last 20
        mtime = rec.stat().st_mtime
        from datetime import datetime
        mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {i}. {rec.name} ({mtime_str})")
    print()


def _print_summary_report(recording_path: Path, summary: Dict, analyze_to_failure: bool) -> None:
    """Render a CLI report from the canonical summary contract."""
    if not isinstance(summary, dict) or summary.get("error"):
        print(f"Failed to analyze recording: {summary.get('error', 'unknown error')}")
        return

    executive = summary.get("executive_summary", {})
    path_tracking = summary.get("path_tracking", {})
    control_smoothness = summary.get("control_smoothness", {})
    speed_control = summary.get("speed_control", {})
    comfort = summary.get("comfort", {})
    perception = summary.get("perception_quality", {})
    trajectory_quality = summary.get("trajectory_quality", {})
    system_health = summary.get("system_health", {})
    safety = summary.get("safety", {})
    latency_sync = summary.get("latency_sync", {})
    transport_contract = summary.get("transport_contract", {})
    speed_intent = summary.get("speed_intent", {})
    run_intent = summary.get("run_intent", {})
    wrong_target_contract = summary.get("wrong_target_contract", {})
    ego_lane_contract = summary.get("ego_lane_contract", {})
    lateral_owner_contract = summary.get("lateral_owner_contract", {})
    highway_mild_curve_contract = summary.get("highway_mild_curve_contract", {})
    mpc_gt_cross_track_contract = summary.get("mpc_gt_cross_track_contract", {})
    chassis_ground = summary.get("chassis_ground", {})
    curve_intent_diag = summary.get("curve_intent_diagnostics", {})
    recommendations = summary.get("recommendations", [])

    print("=" * 80)
    print("OVERALL DRIVE ANALYSIS REPORT")
    print("=" * 80)
    print(f"Recording: {recording_path.name}")
    print()

    print("1. EXECUTIVE SUMMARY")
    print("-" * 80)
    print(f"   Drive Duration: {executive.get('drive_duration', 0.0):.2f} seconds")
    print(f"   Total Frames: {int(executive.get('total_frames', 0))}")
    print(f"   Success Rate: {executive.get('success_rate', 0.0):.1f}% (time in lane)")
    print(f"   Overall Score: {executive.get('overall_score', 0.0):.1f}/100")
    if executive.get("failure_detected"):
        failure_frame = executive.get("failure_frame")
        if failure_frame is not None:
            mode = "truncated" if analyze_to_failure else "detected"
            print(f"   Failure Frame: {int(failure_frame)} ({mode})")
    print()

    # Score breakdown — why is the overall score what it is?
    _layer_order = ["Safety", "Trajectory", "Control", "Perception", "LongitudinalComfort", "SignalIntegrity"]
    _layer_scores = summary.get("layer_scores") or {}
    _layer_bd = summary.get("layer_score_breakdown") or {}
    _score_bd = executive.get("score_breakdown") or {}
    _weights = _score_bd.get("layer_weights") or {}
    _contribs = _score_bd.get("layer_weighted_contributions") or {}
    if _layer_scores and _weights:
        print(f"   {'Layer':<22} {'Wt':>5}  {'Score':>6}  {'Contrib':>7}  Deductions")
        print(f"   {'-'*22} {'-'*5}  {'-'*6}  {'-'*7}  {'-'*38}")
        for _layer in _layer_order:
            if _layer not in _layer_scores:
                continue
            _sc = float(_layer_scores[_layer])
            _wt = float(_weights.get(_layer, 0.0)) * 100
            _ct = float(_contribs.get(_layer, 0.0))
            _ded = _layer_bd.get(_layer, {})
            _active = [d for d in (_ded.get("deductions") or []) if float(d.get("value", 0)) > 0.01]
            _ded_str = f"{_active[0]['name']}: -{_active[0]['value']:.1f}" if _active else "(none)"
            _sc_color = ""
            print(f"   {_layer:<22} {_wt:>4.1f}%  {_sc:>6.1f}  {_ct:>7.2f}  {_ded_str}")
        _base = float(_score_bd.get("overall_base_score") or executive.get("overall_score") or 0.0)
        _cap = _score_bd.get("cap_reason", "none")
        _final = float(executive.get("overall_score") or 0.0)
        _cap_note = f" → capped: {_cap}" if _cap and _cap != "none" else ""
        print(f"   {'─'*22} {'─'*5}  {'─'*6}  {'─'*7}")
        print(f"   {'Weighted total':<22}        {'':>6}  {_base:>7.2f}{_cap_note}  → {_final:.1f}/100")
    print()

    key_issues = executive.get("key_issues") or []
    if key_issues:
        print("   Key Issues:")
        for idx, issue in enumerate(key_issues[:5], 1):
            print(f"     {idx}. {issue}")
    else:
        print("   ✓ No major issues detected")
    print()

    print("2. PATH TRACKING PERFORMANCE")
    print("-" * 80)
    print(f"   Lateral Error RMSE: {path_tracking.get('lateral_error_rmse', 0.0):.4f} m")
    print(f"   Lateral Error P95:  {path_tracking.get('lateral_error_p95', 0.0):.4f} m")
    print(f"   Heading Error RMSE: {path_tracking.get('heading_error_rmse', 0.0):.4f} rad")
    print(f"   Time in Lane:       {path_tracking.get('time_in_lane', 0.0):.1f}%")
    print(f"   Centeredness:       {path_tracking.get('time_in_lane_centered', 0.0):.1f}%")
    print()

    print("3. CONTROL SMOOTHNESS")
    print("-" * 80)
    print(f"   Steering Jerk Max: {control_smoothness.get('steering_jerk_max', 0.0):.4f}"
          f"  (raw: {control_smoothness.get('steering_jerk_max_raw', 0.0):.4f})")
    print(f"   Steering Rate Max: {control_smoothness.get('steering_rate_max', 0.0):.4f}")
    print(f"   Steering Smoothness: {control_smoothness.get('steering_smoothness', 0.0):.2f}")
    print(f"   Oscillation Frequency: {control_smoothness.get('oscillation_frequency', 0.0):.2f} Hz")
    print(
        "   Oscillation Zero-Crossing Rate: "
        f"{control_smoothness.get('oscillation_zero_crossing_rate_hz', 0.0):.2f} Hz"
    )
    print(
        "   Oscillation RMS Growth Slope: "
        f"{control_smoothness.get('oscillation_rms_growth_slope_mps', 0.0):.4f} m/s"
    )
    print(
        "   Oscillation RMS (start -> end): "
        f"{control_smoothness.get('oscillation_rms_window_start_m', 0.0):.3f} m -> "
        f"{control_smoothness.get('oscillation_rms_window_end_m', 0.0):.3f} m"
    )
    if control_smoothness.get('oscillation_curve_suppressed'):
        curve_frac = control_smoothness.get('oscillation_curve_fraction', 0.0)
        print(
            f"   Oscillation Amplitude Runaway: N/A (curved track {curve_frac*100:.0f}%"
            f" — RMS growth is tracking bias)"
        )
    else:
        print(
            "   Oscillation Amplitude Runaway: "
            f"{'YES' if control_smoothness.get('oscillation_amplitude_runaway') else 'NO'}"
        )
    if lateral_owner_contract.get("availability") == "available":
        print(
            "   Lateral Owner Contract: "
            f"mode={lateral_owner_contract.get('owner_summary_mode') or 'N/A'}, "
            f"primary={lateral_owner_contract.get('primary_owner_mode') or 'N/A'}, "
            f"healthy={'YES' if lateral_owner_contract.get('authoritative_owner_healthy') else 'NO'}, "
            f"legacyIntentSuppressed={'YES' if lateral_owner_contract.get('suppress_legacy_curve_intent_warnings') else 'NO'}"
        )
    if curve_intent_diag.get("available"):
        proxy_label = "Legacy Curve-Intent Proxy"
        if lateral_owner_contract.get("suppress_legacy_curve_intent_warnings"):
            proxy_label += " (compatibility only)"
        print(
            f"   {proxy_label} Arm Early Rate: "
            f"{curve_intent_diag.get('arm_early_enough_rate', 0.0):.1f}%"
        )
        undercall_rate = curve_intent_diag.get('undercall_frame_rate')
        if undercall_rate is None:
            skip_reason = curve_intent_diag.get('undercall_skipped_reason', 'unknown')
            gt_max = curve_intent_diag.get('gt_max_curvature', 0.0)
            print(
                f"   {proxy_label} Undercall Rate: N/A "
                f"({skip_reason}, GT max κ={gt_max:.4f})"
            )
        else:
            print(
                f"   {proxy_label} Undercall Rate: "
                f"{undercall_rate:.1f}%"
            )
        print(
            f"   {proxy_label} Curvature Ratio P50/P95: "
            f"{curve_intent_diag.get('curvature_ratio_p50') or 0.0:.2f} / "
            f"{curve_intent_diag.get('curvature_ratio_p95') or 0.0:.2f}"
        )
    curve_local_contract = summary.get("curve_local_contract", {})
    if curve_local_contract.get("curve_local_contract_available"):
        limits = curve_local_contract.get("limits", {})
        print(
            "   Curve Preview Active On Straights: "
            f"{float(curve_local_contract.get('curve_preview_far_active_straight_rate', 0.0)):.1f}%"
        )
        print(
            "   Curve Local Active On Straights: "
            f"{float(curve_local_contract.get('curve_local_active_straight_rate', 0.0)):.1f}% "
            f"(<= {float(limits.get('curve_local_active_straight_rate_max_pct', 5.0)):.1f}%)"
        )
        if "curve_local_arm_ready_straight_rate" in curve_local_contract:
            print(
                "   Curve Local Arm-Ready On Straights: "
                f"{float(curve_local_contract.get('curve_local_arm_ready_straight_rate', 0.0)):.1f}%"
            )
        if "curve_local_commit_ready_straight_rate" in curve_local_contract:
            print(
                "   Curve Local Commit-Ready On Straights: "
                f"{float(curve_local_contract.get('curve_local_commit_ready_straight_rate', 0.0)):.1f}%"
            )
        if "curve_local_path_sustain_active_straight_rate" in curve_local_contract:
            print(
                "   Curve Local Path-Sustain On Straights: "
                f"{float(curve_local_contract.get('curve_local_path_sustain_active_straight_rate', 0.0)):.1f}%"
            )
        if "curve_local_arm_without_ready_count" in curve_local_contract:
            print(
                "   Curve Local Arm Without Ready: "
                f"{int(curve_local_contract.get('curve_local_arm_without_ready_count', 0))}"
            )
        if "curve_local_commit_without_ready_count" in curve_local_contract:
            print(
                "   Curve Local Commit Without Ready: "
                f"{int(curve_local_contract.get('curve_local_commit_without_ready_count', 0))}"
            )
        print(
            "   Curve Local Re-entry Without Gate: "
            f"{int(curve_local_contract.get('curve_local_reentry_without_gate_count', 0))}"
        )
        print(
            "   Curve Watchdog Ping-Pong: "
            f"{int(curve_local_contract.get('curve_local_watchdog_pingpong_count', 0))}"
        )
        print(
            "   Curve COMMIT Before Distance-Ready: "
            f"{int(curve_local_contract.get('curve_local_commit_without_distance_ready_count', 0))}"
        )
        print(
            "   Lookahead Collapse Violations: "
            f"{int(curve_local_contract.get('curve_lookahead_collapse_violation_count', 0))}"
        )
        print(
            "   Straight Root-Cause P50s (gate/raw/far/time): "
            f"{(curve_local_contract.get('curve_local_gate_weight_straight_p50') if curve_local_contract.get('curve_local_gate_weight_straight_p50') is not None else float('nan')):.2f} / "
            f"{(curve_local_contract.get('curve_local_phase_raw_straight_p50') if curve_local_contract.get('curve_local_phase_raw_straight_p50') is not None else float('nan')):.2f} / "
            f"{(curve_local_contract.get('curve_preview_far_phase_straight_p50') if curve_local_contract.get('curve_preview_far_phase_straight_p50') is not None else float('nan')):.2f} / "
            f"{(curve_local_contract.get('curve_phase_term_time_straight_p50') if curve_local_contract.get('curve_phase_term_time_straight_p50') is not None else float('nan')):.2f}"
        )
        straight_source = curve_local_contract.get("straight_summary_source")
        straight_delta = curve_local_contract.get("straight_summary_vs_segment_rate_delta_pct")
        if straight_source or straight_delta is not None:
            print(
                "   Straight Summary Source: "
                f"{straight_source or 'unknown'}"
                + (
                    f" (delta vs segment aggregate: {float(straight_delta):.2f}%)"
                    if straight_delta is not None
                    else ""
                )
            )
    turn_in_owner = summary.get("turn_in_owner", {})
    if turn_in_owner.get("availability") == "available":
        print(
            "   Turn-In Owner: "
            f"mode={turn_in_owner.get('owner_mode') or 'N/A'}, "
            f"source={turn_in_owner.get('entry_weight_source') or 'N/A'}, "
            f"fallbackRate={(float(turn_in_owner.get('fallback_active_rate')) if turn_in_owner.get('fallback_active_rate') is not None else float('nan')):.1f}%, "
            f"armWithoutReady={int(turn_in_owner.get('curve_local_arm_without_ready_count', 0))}, "
            f"commitWithoutReady={int(turn_in_owner.get('curve_local_commit_without_ready_count', 0))}, "
            f"onsetVsStartP50={(float(turn_in_owner.get('steering_onset_minus_curve_start_frames_p50')) if turn_in_owner.get('steering_onset_minus_curve_start_frames_p50') is not None else float('nan')):.1f} fr"
        )
    local_curve_reference = summary.get("local_curve_reference", {})
    if local_curve_reference.get("availability") == "available":
        print(
            "   Local Arc Reference: "
            f"mode={local_curve_reference.get('mode') or 'N/A'}, "
            f"requested={local_curve_reference.get('requested_mode') or 'N/A'}, "
            f"source={local_curve_reference.get('source_mode') or 'N/A'}, "
            f"fallbackReason={local_curve_reference.get('fallback_reason_mode') or 'N/A'}, "
            f"activeRate={(float(local_curve_reference.get('active_rate')) if local_curve_reference.get('active_rate') is not None else float('nan')):.1f}%, "
            f"fallbackRate={(float(local_curve_reference.get('fallback_active_rate')) if local_curve_reference.get('fallback_active_rate') is not None else float('nan')):.1f}%, "
            f"shadowPromoteRate={(float(local_curve_reference.get('shadow_promotion_active_rate')) if local_curve_reference.get('shadow_promotion_active_rate') is not None else float('nan')):.1f}%, "
            f"plannerDeltaP50={(float(local_curve_reference.get('planner_delta_p50_m')) if local_curve_reference.get('planner_delta_p50_m') is not None else float('nan')):.2f} m"
        )
    if highway_mild_curve_contract.get("availability") == "available":
        print(
            "   Highway Mild-Curve Contract: "
            f"issue={'YES' if highway_mild_curve_contract.get('issue_detected') else 'NO'}, "
            f"mildOnHighErr={(float(highway_mild_curve_contract.get('mild_curve_present_on_high_error_rate')) if highway_mild_curve_contract.get('mild_curve_present_on_high_error_rate') is not None else float('nan')):.1f}%, "
            f"recognizerInactive={(float(highway_mild_curve_contract.get('curve_recognition_inactive_on_high_error_rate')) if highway_mild_curve_contract.get('curve_recognition_inactive_on_high_error_rate') is not None else float('nan')):.1f}%, "
            f"longLook={(float(highway_mild_curve_contract.get('long_lookahead_on_high_error_rate')) if highway_mild_curve_contract.get('long_lookahead_on_high_error_rate') is not None else float('nan')):.1f}%, "
            f"smallOffset={(float(highway_mild_curve_contract.get('reference_geometry_mismatch_on_high_error_rate')) if highway_mild_curve_contract.get('reference_geometry_mismatch_on_high_error_rate') is not None else float('nan')):.1f}%, "
            f"preactiveMissing={(float(highway_mild_curve_contract.get('preactivation_authority_missing_on_high_error_rate')) if highway_mild_curve_contract.get('preactivation_authority_missing_on_high_error_rate') is not None else float('nan')):.1f}%, "
            f"shadowInsufficient={(float(highway_mild_curve_contract.get('active_state_shadow_insufficient_on_high_error_rate')) if highway_mild_curve_contract.get('active_state_shadow_insufficient_on_high_error_rate') is not None else float('nan')):.1f}%, "
            f"blocker={str(highway_mild_curve_contract.get('curve_activation_blocker_mode_on_underactivated') or highway_mild_curve_contract.get('curve_activation_blocker_mode_on_high_error') or 'N/A')}, "
            f"rearmCycle={(float(highway_mild_curve_contract.get('rearm_cycle_on_high_error_rate')) if highway_mild_curve_contract.get('rearm_cycle_on_high_error_rate') is not None else float('nan')):.1f}%, "
            f"mpcBiasCancel={(float(highway_mild_curve_contract.get('mpc_bias_cancellation_on_high_error_rate')) if highway_mild_curve_contract.get('mpc_bias_cancellation_on_high_error_rate') is not None else float('nan')):.1f}%"
        )
    print()

    print("4. SPEED CONTROL")
    print("-" * 80)
    print(f"   Speed Error RMSE: {speed_control.get('speed_error_rmse', 0.0):.3f} m/s")
    print(f"   Speed Overspeed Rate: {speed_control.get('speed_overspeed_rate', 0.0):.1f}%")
    print(
        "   Overspeed Into Curve Rate: "
        f"{speed_control.get('overspeed_into_curve_rate', 0.0):.1f}%"
    )
    print(
        "   Curve-Cap Active Rate: "
        f"{speed_control.get('curve_cap_active_rate', 0.0):.1f}%"
    )
    print(
        "   Turn Infeasible While Curve-Cap Active: "
        f"{speed_control.get('turn_infeasible_rate_when_curve_cap_active', 0.0):.1f}%"
    )
    print(
        "   Pre-turn Arm Lead (P50/P95): "
        f"{speed_control.get('pre_turn_arm_lead_frames_p50', 0.0):.1f} / "
        f"{speed_control.get('pre_turn_arm_lead_frames_p95', 0.0):.1f} frames"
    )
    print(
        "   Cap Tracking Error (P50/P95/Max): "
        f"{speed_control.get('cap_tracking_error_p50_mps', 0.0):.3f} / "
        f"{speed_control.get('cap_tracking_error_p95_mps', 0.0):.3f} / "
        f"{speed_control.get('cap_tracking_error_max_mps', 0.0):.3f} m/s"
    )
    print(
        "   Frames Above Cap (>0.3 / >1.0 m/s): "
        f"{speed_control.get('frames_above_cap_0p3mps_rate', 0.0):.1f}% / "
        f"{speed_control.get('frames_above_cap_1p0mps_rate', 0.0):.1f}%"
    )
    print(
        "   Cap Recovery Frames (P50/P95): "
        f"{speed_control.get('cap_recovery_frames_p50', 0.0):.1f} / "
        f"{speed_control.get('cap_recovery_frames_p95', 0.0):.1f}"
    )
    print(
        "   Hard Ceiling Applied Rate: "
        f"{speed_control.get('hard_ceiling_applied_rate', 0.0):.2f}%"
    )
    print(f"   Speed Limit Missing: {speed_control.get('speed_limit_zero_rate', 0.0):.1f}%")
    print()

    print("5. COMFORT")
    print("-" * 80)
    print(f"   Accel P95: {comfort.get('acceleration_p95_filtered', 0.0):.3f} m/s²"
          f"  (raw: {comfort.get('acceleration_p95', 0.0):.1f})")
    print(f"   Jerk P95 (filtered):   {comfort.get('jerk_p95_filtered', 0.0):.3f} m/s³"
          f"  (raw: {comfort.get('jerk_p95', 0.0):.1f})")
    print(f"   Jerk P95 (commanded):  {comfort.get('commanded_jerk_p95', 0.0):.3f} m/s³"
          f"  (target ≤6.0 — gate metric)")
    print(f"   Lateral Accel P95: {comfort.get('lateral_accel_p95', 0.0):.3f} m/s²")
    print(f"   Lateral Jerk P95:  {comfort.get('lateral_jerk_p95', 0.0):.3f} m/s³")
    hotspot_attr = comfort.get("hotspot_attribution", {})
    hotspot_entries = hotspot_attr.get("entries") if isinstance(hotspot_attr, dict) else None
    if hotspot_attr.get("availability") == "available" and hotspot_entries:
        nominal_dt = hotspot_attr.get("dt_nominal_ms")
        gap_limit = hotspot_attr.get("dt_gap_threshold_ms")
        nominal_text = f"{float(nominal_dt):.1f} ms" if nominal_dt is not None else "n/a"
        gap_text = f"{float(gap_limit):.1f} ms" if gap_limit is not None else "n/a"
        print(
            "   Longitudinal Hotspot Attribution "
            f"(dt_nominal={nominal_text}, gap>= {gap_text}):"
        )
        counts = hotspot_attr.get("counts_by_attribution") or {}
        high_conf = hotspot_attr.get("high_confidence_rate")
        mismatch = hotspot_attr.get("commanded_vs_measured_mismatch_rate")
        if counts:
            counts_text = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            print(f"     Attribution Rollup: {counts_text}")
        if high_conf is not None:
            print(f"     High-confidence rate: {float(high_conf) * 100.0:.1f}%")
        if mismatch is not None:
            print(f"     Command-vs-measured mismatch rate: {float(mismatch) * 100.0:.1f}%")
        print(
            "     Frame Kind    |Metric|  Jerk(raw/filt/cmd)  dt_prev  "
            "CapX LimX DtIrreg  Attribution"
        )
        for entry in hotspot_entries[:8]:
            frame = int(entry.get("frame", -1))
            metric = str(entry.get("metric", "n/a"))
            metric_abs = float(entry.get("metric_abs", 0.0))
            jerk_raw_val = entry.get("jerk_raw_mps3", entry.get("jerk_mps3"))
            jerk_filt_val = entry.get("jerk_filtered_mps3")
            jerk_cmd_val = entry.get("command_jerk_proxy_mps3")
            dt_prev = entry.get("dt_prev_ms")
            limiter_active = bool(entry.get("limiter_active"))
            limiter_transition = bool(entry.get("limiter_transition"))
            dt_irregular = bool(entry.get("timestamp_irregular_nearby"))
            attribution = str(entry.get("attribution", "unknown"))
            jerk_raw_text = (
                f"{float(jerk_raw_val):+7.2f}" if jerk_raw_val is not None else "   n/a "
            )
            jerk_filt_text = (
                f"{float(jerk_filt_val):+7.2f}" if jerk_filt_val is not None else "   n/a "
            )
            jerk_cmd_text = (
                f"{float(jerk_cmd_val):+7.2f}" if jerk_cmd_val is not None else "   n/a "
            )
            dt_text = f"{float(dt_prev):6.1f}" if dt_prev is not None else "   n/a"
            print(
                f"     {frame:5d} {metric:>5s} {metric_abs:9.2f} "
                f"{jerk_raw_text}/{jerk_filt_text}/{jerk_cmd_text} "
                f"{dt_text} ms  {str(limiter_active)[0]}/{str(limiter_transition)[0]}   "
                f"{'Y' if dt_irregular else 'N'}      {attribution}"
            )
    else:
        print("   Longitudinal Hotspot Attribution: N/A")
    print()

    print("6. PERCEPTION QUALITY")
    print("-" * 80)
    print(f"   Lane Detection Rate: {perception.get('lane_detection_rate', 0.0):.1f}%")
    print(f"   Stale Hard Rate:     {perception.get('stale_hard_rate', 0.0):.1f}%")
    print(f"   Lane Jitter P95:     {perception.get('lane_line_jitter_p95', 0.0):.3f}")
    print()

    print("7. TRAJECTORY QUALITY")
    print("-" * 80)
    print(f"   Trajectory Availability: {trajectory_quality.get('trajectory_availability', 0.0):.1f}%")
    print(f"   Ref Point Accuracy RMSE: {trajectory_quality.get('ref_point_accuracy_rmse', 0.0):.4f} m")
    print()

    print("8. SYSTEM HEALTH")
    print("-" * 80)
    print(f"   PID Integral Max: {system_health.get('pid_integral_max', 0.0):.4f}")
    print(f"   Unity Time Gap Max: {system_health.get('unity_time_gap_max', 0.0):.3f} s")
    print()

    print("9. LATENCY & SYNC")
    print("-" * 80)
    e2e = latency_sync.get("e2e", {})
    e2e_stats = e2e.get("stats_ms", {}) if isinstance(e2e, dict) else {}
    e2e_p95 = e2e_stats.get("p95")
    e2e_limit = e2e.get("limit_p95_ms")
    if e2e.get("availability") == "available" and e2e_p95 is not None:
        e2e_status = "PASS" if e2e.get("pass") else "FAIL"
        e2e_limit_text = f"{float(e2e_limit):.1f} ms" if e2e_limit is not None else "n/a"
        print(
            f"   E2E Control Latency (P95): {float(e2e_p95):.1f} ms "
            f"(limit <= {e2e_limit_text}) [{e2e_status}]"
        )
    else:
        print("   E2E Control Latency (P95): N/A")

    sync_alignment = latency_sync.get("sync_alignment", {})
    dt_cam_traj_p95 = (sync_alignment.get("dt_cam_traj_ms") or {}).get("p95")
    dt_cam_control_p95 = (sync_alignment.get("dt_cam_control_ms") or {}).get("p95")
    misaligned_rate = sync_alignment.get("contract_misaligned_rate")
    window = sync_alignment.get("alignment_window_ms")
    rate_limit = sync_alignment.get("contract_misaligned_rate_limit")
    if sync_alignment.get("availability") == "available":
        sync_status = "PASS" if sync_alignment.get("pass") else "FAIL"
        traj_text = f"{float(dt_cam_traj_p95):.1f} ms" if dt_cam_traj_p95 is not None else "n/a"
        ctrl_text = (
            f"{float(dt_cam_control_p95):.1f} ms" if dt_cam_control_p95 is not None else "n/a"
        )
        rate_text = (
            f"{float(misaligned_rate) * 100.0:.2f}%"
            if misaligned_rate is not None
            else "n/a"
        )
        print(
            "   Sync/Alignment Health: "
            f"dt_cam_traj_p95={traj_text}, dt_cam_control_p95={ctrl_text}, "
            f"misaligned_rate={rate_text} "
            f"(limits: p95 <= {float(window):.1f} ms, rate <= {float(rate_limit) * 100.0:.2f}%) "
            f"[{sync_status}]"
        )
    else:
        print("   Sync/Alignment Health: N/A")
    cadence = latency_sync.get("cadence", {})
    cadence_stats = cadence.get("stats_ms", {}) if isinstance(cadence, dict) else {}
    cadence_p95 = cadence_stats.get("p95")
    cadence_max = cadence_stats.get("max")
    cadence_irregular_rate = cadence.get("irregular_rate")
    cadence_severe_rate = cadence.get("severe_irregular_rate")
    cadence_limits = cadence.get("limits", {}) if isinstance(cadence, dict) else {}
    cadence_irregular_limit = cadence_limits.get("irregular_rate_max")
    cadence_severe_limit = cadence_limits.get("severe_irregular_rate_max")
    cadence_tuning_valid = cadence.get("tuning_valid")
    cadence_tuning_dt_p95_limit = cadence_limits.get("dt_p95_ms_max")
    cadence_tuning_dt_max_limit = cadence_limits.get("dt_max_ms_max")
    if cadence.get("availability") == "available":
        cadence_status = "PASS" if cadence.get("pass") else "FAIL"
        print(
            "   Cadence Health: "
            f"dt_p95={float(cadence_p95):.1f} ms, dt_max={float(cadence_max):.1f} ms, "
            f"irregular_rate={float(cadence_irregular_rate) * 100.0:.2f}%, "
            f"severe_rate={float(cadence_severe_rate) * 100.0:.2f}% "
            f"(limits: irregular <= {float(cadence_irregular_limit) * 100.0:.2f}%, "
            f"severe <= {float(cadence_severe_limit) * 100.0:.2f}%) "
            f"[{cadence_status}]"
        )
        tuning_status = "PASS" if bool(cadence_tuning_valid) else "FAIL"
        if cadence_tuning_dt_p95_limit is not None and cadence_tuning_dt_max_limit is not None:
            print(
                "   Tuning Valid: "
                f"{bool(cadence_tuning_valid)} "
                f"(limits: dt_p95 <= {float(cadence_tuning_dt_p95_limit):.1f} ms, "
                f"dt_max <= {float(cadence_tuning_dt_max_limit):.1f} ms, "
                f"severe <= {float(cadence_severe_limit) * 100.0:.2f}%) "
                f"[{tuning_status}]"
            )
        else:
            print(f"   Tuning Valid: {bool(cadence_tuning_valid)} [{tuning_status}]")
    else:
        print("   Cadence Health: N/A")
    print()

    print("10. TRANSPORT & CONTRACTS")
    print("-" * 80)
    if transport_contract.get("availability") == "available":
        queue_stats = transport_contract.get("packet_queue_depth") or {}
        payload_queue_stats = transport_contract.get("payload_queue_depth") or {}
        skipped_stats = transport_contract.get("skipped_unity_frames") or {}
        delta_stats = transport_contract.get("front_vehicle_time_delta_ms") or {}
        payload_age_stats = transport_contract.get("payload_oldest_age_ms") or {}
        payload_bytes_stats = transport_contract.get("payload_bytes") or {}
        selected_age_stats = transport_contract.get("payload_selected_age_ms") or {}
        drained_stats = transport_contract.get("payload_drained_count") or {}
        max_drained_age_stats = transport_contract.get("payload_max_drained_age_ms") or {}
        queue_after_select_stats = (
            transport_contract.get("payload_server_queue_depth_after_select") or {}
        )
        oldest_after_select_stats = (
            transport_contract.get("payload_server_oldest_age_ms_after_select") or {}
        )
        join_wait_stats = transport_contract.get("join_wait_ms") or {}
        coherence_pass_rate = float(transport_contract.get("coherence_pass_rate") or 0.0)
        complete_but_incoherent_rate = float(
            transport_contract.get("complete_but_incoherent_rate") or 0.0
        )
        source_context_queue_stats = (
            transport_contract.get("source_context_queue_depth") or {}
        )
        source_context_time_delta_stats = (
            transport_contract.get("source_context_time_delta_ms") or {}
        )
        print(
            "   Packet Mode / Schema: "
            f"{transport_contract.get('packet_mode', 'unknown')} / "
            f"{transport_contract.get('packet_schema_version', 0)}"
        )
        print(
            "   Consume Policy / Fresh Rate: "
            f"{transport_contract.get('consume_policy') or 'n/a'} / "
            f"{float(transport_contract.get('payload_selected_fresh_rate') or 0.0):.1f}%"
        )
        print(
            "   Selection Source / Fallback Mode: "
            f"{transport_contract.get('payload_selection_source_mode') or 'n/a'} / "
            f"{transport_contract.get('payload_selection_fallback_reason_mode') or 'none'}"
        )
        print(
            "   Join Source / Key-Present Rate: "
            f"{transport_contract.get('join_source_mode') or 'unknown'} / "
            f"{float(transport_contract.get('join_key_present_rate') or 0.0):.1f}%"
        )
        print(
            "   Join Failure / Side / Selected Contract: "
            f"{transport_contract.get('join_failure_reason_mode') or 'none'} / "
            f"{transport_contract.get('join_failure_side_mode') or 'none'} / "
            f"{transport_contract.get('selected_failure_contract_reason_mode') or 'none'}"
        )
        print(
            "   Selected Stage / Bundle Close: "
            f"{transport_contract.get('selected_failure_source_stage_mode') or 'none'} / "
            f"{transport_contract.get('source_bundle_close_reason_mode') or 'none'}"
        )
        print(
            "   Packet Completeness / Fallback: "
            f"{float(transport_contract.get('packet_completeness_rate') or 0.0):.1f}% / "
            f"{float(transport_contract.get('fallback_active_rate') or 0.0):.1f}%"
        )
        print(
            "   Coherence Pass / Complete-but-Incoherent / Reason: "
            f"{coherence_pass_rate:.1f}% / "
            f"{complete_but_incoherent_rate:.1f}% / "
            f"{transport_contract.get('coherence_reason_mode') or 'coherent'}"
        )
        print(
            "   Queue Depth (p50/p95/max): "
            f"{(queue_stats.get('p50') if queue_stats.get('p50') is not None else float('nan')):.1f} / "
            f"{(queue_stats.get('p95') if queue_stats.get('p95') is not None else float('nan')):.1f} / "
            f"{(queue_stats.get('max') if queue_stats.get('max') is not None else float('nan')):.1f}"
        )
        print(
            "   Payload Queue / Age / Bytes (p95): "
            f"{(payload_queue_stats.get('p95') if payload_queue_stats.get('p95') is not None else float('nan')):.1f} / "
            f"{(payload_age_stats.get('p95') if payload_age_stats.get('p95') is not None else float('nan')):.1f} ms / "
            f"{(payload_bytes_stats.get('p95') if payload_bytes_stats.get('p95') is not None else float('nan')):.0f}"
        )
        print(
            "   Selected Payload Age / Drained / MaxDrainedAge (p95): "
            f"{(selected_age_stats.get('p95') if selected_age_stats.get('p95') is not None else float('nan')):.1f} ms / "
            f"{(drained_stats.get('p95') if drained_stats.get('p95') is not None else float('nan')):.1f} / "
            f"{(max_drained_age_stats.get('p95') if max_drained_age_stats.get('p95') is not None else float('nan')):.1f} ms"
        )
        print(
            "   Queue After Select / Oldest After (p95): "
            f"{(queue_after_select_stats.get('p95') if queue_after_select_stats.get('p95') is not None else float('nan')):.1f} / "
            f"{(oldest_after_select_stats.get('p95') if oldest_after_select_stats.get('p95') is not None else float('nan')):.1f} ms"
        )
        print(
            "   Skipped Unity Frames (p50/p95/max): "
            f"{(skipped_stats.get('p50') if skipped_stats.get('p50') is not None else float('nan')):.1f} / "
            f"{(skipped_stats.get('p95') if skipped_stats.get('p95') is not None else float('nan')):.1f} / "
            f"{(skipped_stats.get('max') if skipped_stats.get('max') is not None else float('nan')):.1f}"
        )
        print(
            "   Join Wait ms (p50/p95/max): "
            f"{(join_wait_stats.get('p50') if join_wait_stats.get('p50') is not None else float('nan')):.1f} / "
            f"{(join_wait_stats.get('p95') if join_wait_stats.get('p95') is not None else float('nan')):.1f} / "
            f"{(join_wait_stats.get('max') if join_wait_stats.get('max') is not None else float('nan')):.1f}"
        )
        print(
            "   Front↔Vehicle Δt ms (p50/p95/max): "
            f"{(delta_stats.get('p50') if delta_stats.get('p50') is not None else float('nan')):.1f} / "
            f"{(delta_stats.get('p95') if delta_stats.get('p95') is not None else float('nan')):.1f} / "
            f"{(delta_stats.get('max') if delta_stats.get('max') is not None else float('nan')):.1f}"
        )
        print(
            "   Source Context Queue / Δt ms (p95): "
            f"{(source_context_queue_stats.get('p95') if source_context_queue_stats.get('p95') is not None else float('nan')):.1f} / "
            f"{(source_context_time_delta_stats.get('p95') if source_context_time_delta_stats.get('p95') is not None else float('nan')):.1f}"
        )
        bundle_deadline_stats = transport_contract.get("source_bundle_deadline_ms") or {}
        bundle_age_stats = transport_contract.get("source_bundle_age_ms") or {}
        bundle_inflight_stats = transport_contract.get("source_bundle_inflight_count") or {}
        print(
            "   Bundle Deadline / Age / Inflight (p95): "
            f"{(bundle_deadline_stats.get('p95') if bundle_deadline_stats.get('p95') is not None else float('nan')):.1f} / "
            f"{(bundle_age_stats.get('p95') if bundle_age_stats.get('p95') is not None else float('nan')):.1f} / "
            f"{(bundle_inflight_stats.get('p95') if bundle_inflight_stats.get('p95') is not None else float('nan')):.1f}"
        )
        print(
            "   Bundle Vehicle Built / Enqueued / Sent: "
            f"{float(transport_contract.get('source_bundle_vehicle_state_built_rate') or 0.0):.1f}% / "
            f"{float(transport_contract.get('source_bundle_vehicle_state_enqueued_rate') or 0.0):.1f}% / "
            f"{float(transport_contract.get('source_bundle_vehicle_state_sent_rate') or 0.0):.1f}%"
        )
        print(
            "   Bundle Camera Requested / Sent / Superseded: "
            f"{float(transport_contract.get('source_bundle_camera_requested_rate') or 0.0):.1f}% / "
            f"{float(transport_contract.get('source_bundle_camera_sent_rate') or 0.0):.1f}% / "
            f"{float(transport_contract.get('source_bundle_superseded_before_send_rate') or 0.0):.1f}%"
        )
        print(
            "   Camera Request Attempted / Accepted / Rejected: "
            f"{float(transport_contract.get('source_camera_request_attempted_rate') or 0.0):.1f}% / "
            f"{float(transport_contract.get('source_camera_request_accepted_rate') or 0.0):.1f}% / "
            f"{transport_contract.get('source_camera_request_rejected_reason_mode') or 'none'}"
        )
        print(
            "   Camera Request Skipped / Disposition: "
            f"{transport_contract.get('source_camera_request_skipped_reason_mode') or 'none'} / "
            f"{transport_contract.get('source_camera_request_disposition_mode') or 'none'}"
        )
        print(
            "   Active Camera Eligible / Debug-Unbundled / Contract: "
            f"{float(transport_contract.get('active_camera_eligible_rate') or 0.0):.1f}% / "
            f"{float(transport_contract.get('debug_unbundled_capture_rate') or 0.0):.1f}% / "
            f"{transport_contract.get('camera_capture_contract_reason_mode') or 'none'}"
        )
        print(
            "   Bundle Abort / Vehicle Send Blocked: "
            f"{float(transport_contract.get('source_bundle_aborted_before_vehicle_send_rate') or 0.0):.1f}% / "
            f"{float(transport_contract.get('source_vehicle_send_blocked_by_camera_request_rate') or 0.0):.1f}% / "
            f"{transport_contract.get('source_bundle_abort_reason_mode') or 'none'}"
        )
        print(
            "   Active Camera Excluded / Reason / UnbundledEntered: "
            f"{float(transport_contract.get('active_camera_excluded_event_rate') or 0.0):.1f}% / "
            f"{transport_contract.get('active_camera_excluded_reason_mode') or 'none'} / "
            f"{float(transport_contract.get('unbundled_camera_entered_active_path_rate') or 0.0):.1f}%"
        )
        print(
            "   Join Counters Max: "
            f"key={int(transport_contract.get('key_match_count_max', 0))} / "
            f"fallback={int(transport_contract.get('unity_fallback_count_max', 0))} / "
            f"supCam={int(transport_contract.get('superseded_camera_count_max', 0))} / "
            f"supVeh={int(transport_contract.get('superseded_vehicle_count_max', 0))}"
        )
        print(
            "   Drop/PayloadDrop/Orphan/Timeout Max: "
            f"{int(transport_contract.get('drop_count_max', 0))} / "
            f"{int(transport_contract.get('payload_drop_count_max', 0))} / "
            f"{int(transport_contract.get('orphan_camera_count_max', 0))}c "
            f"{int(transport_contract.get('orphan_vehicle_count_max', 0))}v / "
            f"{int(transport_contract.get('timeout_count_max', 0))}"
        )
        print(
            "   Post-jump Cooldown / False-transport Rate: "
            f"{float(transport_contract.get('post_jump_cooldown_active_rate') or 0.0):.1f}% / "
            f"{float(transport_contract.get('false_teleport_cooldown_rate') or 0.0):.1f}%"
        )
        print(
            "   Guard Suppressed / Continuity Suspect / Reason: "
            f"{float(transport_contract.get('teleport_guard_suppressed_rate') or 0.0):.1f}% / "
            f"{float(transport_contract.get('teleport_continuity_suspect_rate') or 0.0):.1f}% / "
            f"{str(transport_contract.get('teleport_guard_reason_mode') or 'none')}"
        )
        print(
            "   Warn-Age / Stale-Drop Max: "
            f"{float(transport_contract.get('payload_warn_age_exceeded_rate') or 0.0):.1f}% / "
            f"{int(transport_contract.get('payload_stale_drop_count_max') or 0)}"
        )
    else:
        print("   Transport Contract: N/A")
    print()

    print("11. SPEED INTENT")
    print("-" * 80)
    if speed_intent.get("availability") == "available":
        desired_p50 = (speed_intent.get("desired_target_speed_mps") or {}).get("p50")
        post_limits_p50 = (speed_intent.get("post_limits_target_speed_mps") or {}).get("p50")
        final_p50 = (speed_intent.get("final_target_speed_mps") or {}).get("p50")
        planner_p50 = (speed_intent.get("planner_reference_speed_mps") or {}).get("p50")
        effective_p50 = (speed_intent.get("effective_reference_speed_mps") or {}).get("p50")
        print(
            "   Desired / Post-Limits / Final / Planner / Effective (P50): "
            f"{float(desired_p50 or 0.0):.2f} / "
            f"{float(post_limits_p50 or 0.0):.2f} / "
            f"{float(final_p50 or 0.0):.2f} / "
            f"{float(planner_p50 or 0.0):.2f} / "
            f"{float(effective_p50 or 0.0):.2f} m/s"
        )
        print(
            "   Effective Reason Mode: "
            f"{speed_intent.get('effective_reason_mode') or 'n/a'}"
        )
        print(
            "   Effective Ref Drops: "
            f"{int(speed_intent.get('effective_reference_velocity_drop_count') or 0)} "
            f"({float(speed_intent.get('effective_reference_velocity_drop_rate') or 0.0):.1f}%)"
        )
        reason_counts = speed_intent.get("brake_episode_counts_by_reason") or {}
        if reason_counts:
            reason_text = ", ".join(f"{k}={v}" for k, v in sorted(reason_counts.items()))
            print(f"   Brake Episodes By Reason: {reason_text}")
    else:
        print("   Speed Intent: N/A")
    print()

    print("12. RUN INTENT")
    print("-" * 80)
    if run_intent.get("availability") == "available":
        print(
            "   Mode / Recording / Replay: "
            f"{run_intent.get('mode') or 'unknown'} / "
            f"{run_intent.get('recording_type') or 'unknown'} / "
            f"{run_intent.get('replay_type') or 'unknown'}"
        )
        print(
            "   Track / Policy / Candidate: "
            f"{run_intent.get('track_id') or 'unknown'} / "
            f"{run_intent.get('policy_profile') or 'unknown'} / "
            f"{run_intent.get('candidate_label') or 'unknown'}"
        )
        print(
            "   Road Speed Limit / Configured Target: "
            f"{float(run_intent.get('road_speed_limit_expected_mps') or 0.0):.2f} / "
            f"{float(run_intent.get('run_target_speed_mps') or 0.0):.2f} m/s"
        )
        print(
            "   ACC Active / Lead Distance P50: "
            f"{float(run_intent.get('acc_active_rate_pct') or 0.0):.1f}% / "
            f"{float(run_intent.get('lead_vehicle_distance_p50_m') or 0.0):.1f} m"
        )
        if run_intent.get("intent_mismatch_warning"):
            print(
                "   Intent Warning: "
                f"{run_intent.get('intent_mismatch_warning')}"
            )
    else:
        print("   Run Intent: N/A")
    print()

    print("13. WRONG-TARGET CONTRACT")
    print("-" * 80)
    if wrong_target_contract.get("availability") == "available":
        print(
            "   Scenario / Reject-Only / Contract Pass: "
            f"{wrong_target_contract.get('scenario_class') or 'unknown'} / "
            f"{'YES' if wrong_target_contract.get('expected_reject_only') else 'NO'} / "
            f"{'YES' if wrong_target_contract.get('contract_pass') else 'NO'}"
        )
        print(
            "   Reject Reason / Assoc Eligible / Track Active: "
            f"{wrong_target_contract.get('reject_reason_mode') or 'none'} / "
            f"{float(wrong_target_contract.get('association_eligible_rate_pct') or 0.0):.1f}% / "
            f"{float(wrong_target_contract.get('track_active_rate_pct') or 0.0):.1f}%"
        )
        print(
            "   Raw Detect / Continuity Hold / ACC Contamination: "
            f"{float(wrong_target_contract.get('raw_detect_rate_pct') or 0.0):.1f}% / "
            f"{float(wrong_target_contract.get('continuity_hold_rate_pct') or 0.0):.1f}% / "
            f"{'YES' if wrong_target_contract.get('acc_follow_contamination_detected') else 'NO'}"
        )
        print(
            "   Quality Reference Valid: "
            f"{'YES' if wrong_target_contract.get('quality_reference_valid') else 'NO'}"
        )
        print(
            "   Reference Divergence / RawΔ / BlendWt P50: "
            f"{'YES' if wrong_target_contract.get('reference_divergence_issue_detected') else 'NO'} / "
            f"{float(wrong_target_contract.get('reference_divergence_raw_delta_p50_m') or 0.0):.3f} m / "
            f"{float(wrong_target_contract.get('reference_divergence_blend_weight_p50') or 0.0):.3f}"
        )
        print(
            "   Straight Drift / CenterErr P50 / ExpectedCenter P50 / TriggerWt P50: "
            f"{'YES' if wrong_target_contract.get('straight_reference_drift_issue_detected') else 'NO'} / "
            f"{float(wrong_target_contract.get('straight_reference_drift_center_error_p50_m') or 0.0):.3f} m / "
            f"{float(wrong_target_contract.get('straight_reference_drift_expected_center_x_p50_m') or 0.0):.3f} m / "
            f"{float(wrong_target_contract.get('straight_reference_drift_trigger_weight_p50') or 0.0):.3f}"
        )
        print(
            "   Input Guard Active / SuppressCoeff / CenterErr P50 / WidthErr P50 / TriggerWt P50: "
            f"{float(wrong_target_contract.get('input_guard_active_rate_pct') or 0.0):.1f}% / "
            f"{float(wrong_target_contract.get('input_guard_suppressed_lane_coeffs_rate_pct') or 0.0):.1f}% / "
            f"{float(wrong_target_contract.get('input_guard_center_error_p50_m') or 0.0):.3f} m / "
            f"{float(wrong_target_contract.get('input_guard_width_error_p50_m') or 0.0):.3f} m / "
            f"{float(wrong_target_contract.get('input_guard_trigger_weight_p50') or 0.0):.3f}"
        )
        if ego_lane_contract.get("availability") == "available":
            print(
                "   Ego Lane / Selected Lane / Source / MatchesEgo: "
                f"{ego_lane_contract.get('ego_lane_index_mode') or 'unknown'} / "
                f"{ego_lane_contract.get('selected_lane_index_mode') or 'unknown'} / "
                f"{ego_lane_contract.get('lane_selection_source_mode') or 'unknown'} / "
                f"{float(ego_lane_contract.get('selected_matches_ego_rate_pct') or 0.0):.1f}%"
            )
            print(
                "   Ego-Lane Contract / |Selected-Ego Δ| P50: "
                f"{'YES' if ego_lane_contract.get('issue_detected') else 'NO'} / "
                f"{float(ego_lane_contract.get('selected_vs_ego_cross_track_delta_p50_m') or 0.0):.3f} m"
            )
    else:
        print("   Wrong-Target Contract: N/A")
    print()

    print("14. HIGHWAY MILD-CURVE CONTRACT")
    print("-" * 80)
    if highway_mild_curve_contract.get("availability") == "available":
        print(
            "   Issue Detected: "
            f"{'YES' if highway_mild_curve_contract.get('issue_detected') else 'NO'}"
        )
        print(
            "   Scenario Family: "
            f"{str(highway_mild_curve_contract.get('scenario_family_mode') or 'unknown')}"
        )
        print(
            "   High-Error / Mild-Curve / Underactivated / Active-State: "
            f"{int(highway_mild_curve_contract.get('high_error_frame_count') or 0)} frames / "
            f"{float(highway_mild_curve_contract.get('mild_curve_present_on_high_error_rate') or 0.0):.1f}% / "
            f"{float(highway_mild_curve_contract.get('underactivated_tracking_on_high_error_rate') or 0.0):.1f}% / "
            f"{float(highway_mild_curve_contract.get('active_state_tracking_failure_on_high_error_rate') or 0.0):.1f}%"
        )
        print(
            "   Shadow Insufficient / Shadow Promotion / Requested Mode: "
            f"{float(highway_mild_curve_contract.get('active_state_shadow_insufficient_on_high_error_rate') or 0.0):.1f}% / "
            f"{float(highway_mild_curve_contract.get('active_state_shadow_promotion_active_on_high_error_rate') or 0.0):.1f}% / "
            f"{highway_mild_curve_contract.get('local_curve_reference_requested_mode_on_active_state_failure') or 'N/A'}"
        )
        print(
            "   Recognition Inactive / Long Lookahead / Small Offset: "
            f"{float(highway_mild_curve_contract.get('curve_recognition_inactive_on_high_error_rate') or 0.0):.1f}% / "
            f"{float(highway_mild_curve_contract.get('long_lookahead_on_high_error_rate') or 0.0):.1f}% / "
            f"{float(highway_mild_curve_contract.get('reference_geometry_mismatch_on_high_error_rate') or 0.0):.1f}%"
        )
        print(
            "   Perception/Transport/MPC Overlap: "
            f"{float(highway_mild_curve_contract.get('poor_perception_overlap_on_high_error_rate') or 0.0):.1f}% / "
            f"{float(highway_mild_curve_contract.get('transport_fallback_overlap_on_high_error_rate') or 0.0):.1f}% / "
            f"{float(highway_mild_curve_contract.get('mpc_feasible_on_high_error_rate') or 0.0):.1f}% feasible"
        )
        lat_stats = highway_mild_curve_contract.get("lateral_error_abs_m") or {}
        offset_stats = highway_mild_curve_contract.get("road_frame_lane_center_offset_abs_m") or {}
        ref_curve_stats = highway_mild_curve_contract.get("reference_point_curvature_abs") or {}
        kappa_stats = highway_mild_curve_contract.get("mpc_kappa_ref_abs") or {}
        pp_stats = highway_mild_curve_contract.get("pp_lookahead_distance_m") or {}
        ref_ld_stats = highway_mild_curve_contract.get("reference_lookahead_target_m") or {}
        arm_deficit_stats = highway_mild_curve_contract.get("curve_local_arm_phase_deficit") or {}
        arm_effect_stats = highway_mild_curve_contract.get("curve_local_arm_effect_score") or {}
        arm_heading_stats = highway_mild_curve_contract.get("curve_local_arm_effect_heading_term") or {}
        arm_lateral_stats = highway_mild_curve_contract.get("curve_local_arm_effect_lateral_shift_term") or {}
        arm_time_stats = highway_mild_curve_contract.get("curve_local_arm_effect_time_support_term") or {}
        sustain_stats = highway_mild_curve_contract.get("curve_local_sustain_phase_raw") or {}
        sustain_effect_stats = highway_mild_curve_contract.get("curve_local_dynamic_sustain_effect_score") or {}
        path_term_stats = highway_mild_curve_contract.get("curve_phase_term_path") or {}
        kappa_ratio_stats = highway_mild_curve_contract.get("mpc_kappa_ratio_to_reference") or {}
        bias_stats = highway_mild_curve_contract.get("mpc_kappa_bias_correction") or {}
        print(
            "   Error / Lane Offset P50-P95: "
            f"{float(lat_stats.get('p50') or 0.0):.3f}-{float(lat_stats.get('p95') or 0.0):.3f} m / "
            f"{float(offset_stats.get('p50') or 0.0):.3f}-{float(offset_stats.get('p95') or 0.0):.3f} m"
        )
        print(
            "   Ref κ / MPC κ / PP LD / Ref LD (P50): "
            f"{float(ref_curve_stats.get('p50') or 0.0):.4f} / "
            f"{float(kappa_stats.get('p50') or 0.0):.4f} / "
            f"{float(pp_stats.get('p50') or 0.0):.2f} / "
            f"{float(ref_ld_stats.get('p50') or 0.0):.2f}"
        )
        print(
            "   Blocker / Arm Deficit / Arm Effect (P50): "
            f"{str(highway_mild_curve_contract.get('curve_activation_blocker_mode_on_underactivated') or highway_mild_curve_contract.get('curve_activation_blocker_mode_on_high_error') or 'N/A')} / "
            f"{float(arm_deficit_stats.get('p50') or 0.0):.3f} / "
            f"{float(arm_effect_stats.get('p50') or 0.0):.3f}"
        )
        print(
            "   Arm Effect Terms Heading / Lateral / Time (P50): "
            f"{float(arm_heading_stats.get('p50') or 0.0):.3f} / "
            f"{float(arm_lateral_stats.get('p50') or 0.0):.3f} / "
            f"{float(arm_time_stats.get('p50') or 0.0):.3f}"
        )
        print(
            "   Sustain / REARM / MPC Bias Rates: "
            f"{float(highway_mild_curve_contract.get('sustain_phase_collapse_on_high_error_rate') or 0.0):.1f}% / "
            f"{float(highway_mild_curve_contract.get('rearm_cycle_on_high_error_rate') or 0.0):.1f}% / "
            f"{float(highway_mild_curve_contract.get('mpc_bias_cancellation_on_high_error_rate') or 0.0):.1f}%"
        )
        pre_weight_stats = highway_mild_curve_contract.get("curve_preactivation_authority_weight") or {}
        pre_preview_stats = highway_mild_curve_contract.get("curve_preactivation_preview_weight") or {}
        pre_speed_stats = highway_mild_curve_contract.get("curve_preactivation_speed_weight") or {}
        pre_curv_stats = highway_mild_curve_contract.get("curve_preactivation_curvature_weight") or {}
        pre_dist_stats = highway_mild_curve_contract.get("curve_preactivation_distance_weight") or {}
        pre_kappa_floor_stats = highway_mild_curve_contract.get("curve_preactivation_kappa_floor") or {}
        pre_lookahead_stats = highway_mild_curve_contract.get("curve_preactivation_lookahead_target") or {}
        pre_speed_cap_stats = highway_mild_curve_contract.get("curve_preactivation_speed_cap_target") or {}
        print(
            "   Preactivation Candidate / Active / Missing: "
            f"{float(highway_mild_curve_contract.get('preactivation_candidate_on_high_error_rate') or 0.0):.1f}% / "
            f"{float(highway_mild_curve_contract.get('preactivation_authority_active_on_high_error_rate') or 0.0):.1f}% / "
            f"{float(highway_mild_curve_contract.get('preactivation_authority_missing_on_high_error_rate') or 0.0):.1f}%"
        )
        print(
            "   Preactivation Blocker / Weight / Preview / Speed / Curvature / Distance (P50): "
            f"{str(highway_mild_curve_contract.get('curve_preactivation_blocker_mode_on_high_error') or 'N/A')} / "
            f"{float(pre_weight_stats.get('p50') or 0.0):.3f} / "
            f"{float(pre_preview_stats.get('p50') or 0.0):.3f} / "
            f"{float(pre_speed_stats.get('p50') or 0.0):.3f} / "
            f"{float(pre_curv_stats.get('p50') or 0.0):.3f} / "
            f"{float(pre_dist_stats.get('p50') or 0.0):.3f}"
        )
        print(
            "   Preactivation κ Floor / Lookahead / Speed Cap (P50): "
            f"{float(pre_kappa_floor_stats.get('p50') or 0.0):.4f} / "
            f"{float(pre_lookahead_stats.get('p50') or 0.0):.2f} / "
            f"{float(pre_speed_cap_stats.get('p50') or 0.0):.2f}"
        )
        active_preserve_stats = highway_mild_curve_contract.get(
            "mpc_kappa_active_curve_preserve_weight_active_state"
        ) or {}
        active_cap_speed_stats = highway_mild_curve_contract.get(
            "speed_governor_curve_cap_speed_active_state_m"
        ) or {}
        active_cap_margin_stats = highway_mild_curve_contract.get(
            "speed_governor_curve_cap_margin_active_state_mps"
        ) or {}
        guarded_weight_stats = highway_mild_curve_contract.get(
            "local_curve_reference_guarded_bounded_trigger_weight"
        ) or {}
        guarded_blend_floor_stats = highway_mild_curve_contract.get(
            "local_curve_reference_guarded_bounded_blend_floor"
        ) or {}
        authority_weight_stats = highway_mild_curve_contract.get(
            "mpc_kappa_active_mild_curve_authority_weight_active_state"
        ) or {}
        authority_ratio_stats = highway_mild_curve_contract.get(
            "mpc_kappa_active_mild_curve_authority_ratio_active_state"
        ) or {}
        authority_speed_stats = highway_mild_curve_contract.get(
            "mpc_kappa_active_mild_curve_authority_speed_weight"
        ) or {}
        authority_curvature_stats = highway_mild_curve_contract.get(
            "mpc_kappa_active_mild_curve_authority_curvature_weight"
        ) or {}
        authority_gate_stats = highway_mild_curve_contract.get(
            "mpc_kappa_active_mild_curve_authority_gate_weight"
        ) or {}
        print(
            "   Active-State Failure / Preserve Low / Cap Ineffective: "
            f"{float(highway_mild_curve_contract.get('active_state_tracking_failure_on_high_error_rate') or 0.0):.1f}% / "
            f"{float(highway_mild_curve_contract.get('active_state_preserve_weight_low_on_high_error_rate') or 0.0):.1f}% / "
            f"{float(highway_mild_curve_contract.get('active_state_curve_cap_ineffective_on_high_error_rate') or 0.0):.1f}%"
        )
        print(
            "   Active Authority Active / Speed-Gated / Reason: "
            f"{float(highway_mild_curve_contract.get('active_mild_curve_authority_active_on_high_error_rate') or 0.0):.1f}% / "
            f"{float(highway_mild_curve_contract.get('active_mild_curve_authority_speed_gated_on_high_error_rate') or 0.0):.1f}% / "
            f"{str(highway_mild_curve_contract.get('mpc_active_mild_curve_authority_reason_mode_on_high_error') or 'N/A')}"
        )
        print(
            "   Wrong-Target Divergence / Guarded Active / Guarded Reason: "
            f"{float(highway_mild_curve_contract.get('wrong_target_distractor_reference_divergence_on_high_error_rate') or 0.0):.1f}% / "
            f"{float(highway_mild_curve_contract.get('distractor_guarded_bounded_active_on_high_error_rate') or 0.0):.1f}% / "
            f"{str(highway_mild_curve_contract.get('local_curve_reference_guarded_bounded_reason_mode_on_high_error') or 'N/A')}"
        )
        print(
            "   Active-State Local / Cap Reason / PreserveWt / Cap Speed / Cap Margin (P50): "
            f"{str(highway_mild_curve_contract.get('curve_local_state_mode_on_active_state_failure') or 'N/A')} / "
            f"{str(highway_mild_curve_contract.get('speed_governor_curve_cap_reason_mode_on_active_state_failure') or 'N/A')} / "
            f"{float(active_preserve_stats.get('p50') or 0.0):.3f} / "
            f"{float(active_cap_speed_stats.get('p50') or 0.0):.2f} / "
            f"{float(active_cap_margin_stats.get('p50') or 0.0):.2f}"
        )
        print(
            "   Guarded TriggerWt / Guarded Blend Floor (P50): "
            f"{float(guarded_weight_stats.get('p50') or 0.0):.3f} / "
            f"{float(guarded_blend_floor_stats.get('p50') or 0.0):.3f}"
        )
        print(
            "   Active Authority Weight / Ratio / Speed / Curvature / Gate (P50): "
            f"{float(authority_weight_stats.get('p50') or 0.0):.3f} / "
            f"{float(authority_ratio_stats.get('p50') or 0.0):.3f} / "
            f"{float(authority_speed_stats.get('p50') or 0.0):.3f} / "
            f"{float(authority_curvature_stats.get('p50') or 0.0):.3f} / "
            f"{float(authority_gate_stats.get('p50') or 0.0):.3f}"
        )
        print(
            "   Sustain Raw / Dynamic Sustain / Path / κ Ratio / Bias (P50): "
            f"{float(sustain_stats.get('p50') or 0.0):.3f} / "
            f"{float(sustain_effect_stats.get('p50') or 0.0):.3f} / "
            f"{float(path_term_stats.get('p50') or 0.0):.3f} / "
            f"{float(kappa_ratio_stats.get('p50') or 0.0):.3f} / "
            f"{float(bias_stats.get('p50') or 0.0):.4f}"
        )
    else:
        print("   Highway Mild-Curve Contract: N/A")
    print()

    print("15. MPC GT CROSS-TRACK CONTRACT")
    print("-" * 80)
    if mpc_gt_cross_track_contract.get("availability") == "available":
        print(
            "   Issue Detected: "
            f"{'YES' if mpc_gt_cross_track_contract.get('issue_detected') else 'NO'}"
        )
        print(
            "   Issue Mode: "
            f"{str(mpc_gt_cross_track_contract.get('issue_mode') or 'none')}"
        )
        print(
            "   High-Error Frames / Legacy Semantic Mismatch: "
            f"{int(mpc_gt_cross_track_contract.get('high_error_frame_count') or 0)} / "
            f"{float(mpc_gt_cross_track_contract.get('semantic_mismatch_on_high_error_rate') or 0.0):.1f}%"
        )
        print(
            "   Vehicle-Frame Mismatch / Absolute Mismatch / Straight-Mismatch / Control Source: "
            f"{float(mpc_gt_cross_track_contract.get('vehicle_frame_semantic_mismatch_on_high_error_rate') or 0.0):.1f}% / "
            f"{float(mpc_gt_cross_track_contract.get('absolute_coordinate_mismatch_on_high_error_rate') or 0.0):.1f}% / "
            f"{float(mpc_gt_cross_track_contract.get('semantic_mismatch_on_straight_high_error_rate') or 0.0):.1f}% / "
            f"{str(mpc_gt_cross_track_contract.get('mpc_gt_cross_track_control_source_mode_on_high_error') or 'N/A')}"
        )
        print(
            "   Small Road-Frame / Large Vehicle-Frame / Small At-Car / Large Lookahead / Large At-Car: "
            f"{float(mpc_gt_cross_track_contract.get('small_road_frame_cross_track_on_high_error_rate') or 0.0):.1f}% / "
            f"{float(mpc_gt_cross_track_contract.get('large_vehicle_frame_cross_track_on_high_error_rate') or 0.0):.1f}% / "
            f"{float(mpc_gt_cross_track_contract.get('small_at_car_cross_track_on_high_error_rate') or 0.0):.1f}% / "
            f"{float(mpc_gt_cross_track_contract.get('large_lookahead_cross_track_on_high_error_rate') or 0.0):.1f}% / "
            f"{float(mpc_gt_cross_track_contract.get('large_at_car_cross_track_on_high_error_rate') or 0.0):.1f}%"
        )
        print(
            "   Perception / Transport Overlap: "
            f"{float(mpc_gt_cross_track_contract.get('poor_perception_overlap_on_high_error_rate') or 0.0):.1f}% / "
            f"{float(mpc_gt_cross_track_contract.get('transport_fallback_overlap_on_high_error_rate') or 0.0):.1f}%"
        )
        lat_stats = mpc_gt_cross_track_contract.get("control_lateral_error_abs_m") or {}
        src_stats = mpc_gt_cross_track_contract.get("mpc_gt_cross_track_abs_m") or {}
        at_car_stats = mpc_gt_cross_track_contract.get("mpc_gt_cross_track_at_car_abs_m") or {}
        road_frame_stats = mpc_gt_cross_track_contract.get("mpc_gt_cross_track_road_frame_at_car_abs_m") or {}
        vehicle_frame_stats = mpc_gt_cross_track_contract.get("mpc_gt_cross_track_vehicle_frame_at_car_abs_m") or {}
        lookahead_stats = mpc_gt_cross_track_contract.get("mpc_gt_cross_track_lookahead_abs_m") or {}
        delta_stats = mpc_gt_cross_track_contract.get("gt_cross_track_delta_abs_m") or {}
        vehicle_vs_road_stats = mpc_gt_cross_track_contract.get("vehicle_vs_road_frame_cross_track_delta_abs_m") or {}
        print(
            "   Lateral Error / Used GT / Road-Frame GT / Vehicle-Frame GT / Lookahead GT (P50-P95): "
            f"{float(lat_stats.get('p50') or 0.0):.3f}-{float(lat_stats.get('p95') or 0.0):.3f} / "
            f"{float(src_stats.get('p50') or 0.0):.3f}-{float(src_stats.get('p95') or 0.0):.3f} / "
            f"{float(road_frame_stats.get('p50') or 0.0):.3f}-{float(road_frame_stats.get('p95') or 0.0):.3f} / "
            f"{float(vehicle_frame_stats.get('p50') or 0.0):.3f}-{float(vehicle_frame_stats.get('p95') or 0.0):.3f} / "
            f"{float(at_car_stats.get('p50') or 0.0):.3f}-{float(at_car_stats.get('p95') or 0.0):.3f} / "
            f"{float(lookahead_stats.get('p50') or 0.0):.3f}-{float(lookahead_stats.get('p95') or 0.0):.3f}"
        )
        print(
            "   GT Delta / Vehicle-vs-Road Delta / Curve Local State Mode: "
            f"{float(delta_stats.get('p50') or 0.0):.3f}-{float(delta_stats.get('p95') or 0.0):.3f} / "
            f"{float(vehicle_vs_road_stats.get('p50') or 0.0):.3f}-{float(vehicle_vs_road_stats.get('p95') or 0.0):.3f} / "
            f"{str(mpc_gt_cross_track_contract.get('curve_local_state_mode_on_high_error') or 'N/A')}"
        )
    else:
        print("   MPC GT Cross-Track Contract: N/A")
    print()

    print("16. CHASSIS-GROUND HEALTH")
    print("-" * 80)
    chassis_availability = str(chassis_ground.get("availability", "unavailable")).lower()
    if chassis_availability == "available":
        health = str(chassis_ground.get("health", "UNKNOWN")).upper()
        limits = chassis_ground.get("limits", {})
        contact_rate = chassis_ground.get("contact_rate_pct")
        contact_limit = limits.get("warn_contact_rate_pct_max")
        penetration_max = chassis_ground.get("penetration_max_m")
        penetration_limit = limits.get("warn_penetration_m_max")
        configured_min = chassis_ground.get("configured_min_clearance_m")
        effective_min = chassis_ground.get("effective_min_clearance_m")
        clearance_p05 = chassis_ground.get("clearance_p05_m")
        fallback_rate = chassis_ground.get("force_fallback_rate_pct")
        fallback_limit = limits.get("fallback_warn_rate_pct_max")
        print(f"   Health: {health}")
        if configured_min is not None:
            print(f"   Configured Min Clearance: {float(configured_min):.3f} m")
        if effective_min is not None:
            print(f"   Effective Min Clearance:  {float(effective_min):.3f} m")
        if penetration_max is not None:
            fail_suffix = (
                f" (<= {float(penetration_limit):.3f} m)"
                if penetration_limit is not None and float(penetration_max) > float(penetration_limit)
                else ""
            )
            print(f"   Penetration Max: {float(penetration_max):.3f} m{fail_suffix}")
        if contact_rate is not None:
            fail_suffix = (
                f" (<= {float(contact_limit):.2f}%)"
                if contact_limit is not None and float(contact_rate) > float(contact_limit)
                else ""
            )
            print(f"   Contact Rate: {float(contact_rate):.2f}%{fail_suffix}")
        if clearance_p05 is not None:
            print(f"   Clearance P05: {float(clearance_p05):.3f} m")
        if fallback_rate is not None:
            fail_suffix = (
                f" (<= {float(fallback_limit):.2f}%)"
                if fallback_limit is not None and float(fallback_rate) > float(fallback_limit)
                else ""
            )
            print(f"   Force Fallback Rate: {float(fallback_rate):.2f}%{fail_suffix}")
    else:
        print("   Chassis-Ground Health: N/A")
    print()

    contract_health = summary.get("curvature_contract_health", {})
    first_fault_chain = summary.get("first_fault_chain", {})
    print("17. CURVATURE CONTRACT HEALTH")
    print("-" * 80)
    if str(contract_health.get("availability", "unavailable")).lower() == "available":
        limits = contract_health.get("limits", {})
        diverged_rate = contract_health.get("curvature_source_diverged_rate")
        map_authority_lost_rate = contract_health.get("curvature_map_authority_lost_rate")
        divergence_p95 = contract_health.get("curvature_source_divergence_p95")
        commit_streak = contract_health.get("curve_intent_commit_streak_max_frames")
        feas_violation = contract_health.get("feasibility_violation_rate")
        backstop_rate = contract_health.get("feasibility_backstop_active_rate")
        consistency = contract_health.get("curvature_contract_consistency_rate")
        map_untrusted = contract_health.get("map_health_untrusted_rate")
        track_mismatch = contract_health.get("track_mismatch_rate")
        complete_contract = contract_health.get("telemetry_completeness_rate_curvature_contract")
        complete_feas = contract_health.get("telemetry_completeness_rate_feasibility")
        source_mode = contract_health.get("primary_source_mode", "unknown")
        print(f"   Primary Source Mode: {source_mode}")
        if map_authority_lost_rate is not None:
            print(
                f"   Map Authority Lost Rate: {float(map_authority_lost_rate):.2f}% "
                f"(<= {float(limits.get('curvature_map_authority_lost_rate_max_pct', 5.0)):.2f}%)"
            )
        if diverged_rate is not None:
            print(
                f"   Curvature Source Diverged Rate: {float(diverged_rate):.2f}% "
                f"(<= {float(limits.get('curvature_source_diverged_rate_max_pct', 100.0)):.2f}%)"
            )
        if divergence_p95 is not None:
            print(f"   Curvature Source Divergence P95: {float(divergence_p95):.5f}")
        if commit_streak is not None:
            print(f"   Curve Intent COMMIT Streak (max): {int(commit_streak)} frames")
        if feas_violation is not None:
            print(
                f"   Feasibility Violation Rate: {float(feas_violation):.2f}% "
                f"(<= {float(limits.get('feasibility_violation_rate_max_pct', 5.0)):.2f}%)"
            )
        if backstop_rate is not None:
            print(f"   Feasibility Backstop Active Rate: {float(backstop_rate):.2f}%")
        if consistency is not None:
            print(
                f"   Contract Consistency Rate: {float(consistency):.2f}% "
                f"(>= {float(limits.get('curvature_contract_consistency_rate_min_pct', 99.0)):.2f}%)"
            )
        if map_untrusted is not None:
            print(
                f"   Map Health Untrusted Rate: {float(map_untrusted):.2f}% "
                f"(<= {float(limits.get('map_health_untrusted_rate_max_pct', 1.0)):.2f}%)"
            )
        if track_mismatch is not None:
            print(
                f"   Track Mismatch Rate: {float(track_mismatch):.2f}% "
                f"(<= {float(limits.get('track_mismatch_rate_max_pct', 0.0)):.2f}%)"
            )
        if complete_contract is not None:
            print(
                f"   Telemetry Completeness (contract): {float(complete_contract):.2f}% "
                f"(>= {float(limits.get('telemetry_completeness_rate_min_pct', 99.0)):.2f}%)"
            )
        if complete_feas is not None:
            print(
                f"   Telemetry Completeness (feasibility): {float(complete_feas):.2f}% "
                f"(>= {float(limits.get('telemetry_completeness_rate_min_pct', 99.0)):.2f}%)"
            )
    else:
        print("   Curvature Contract Health: N/A")
    if isinstance(first_fault_chain, dict):
        print("   First Fault Chain:")
        print(f"     first_divergence_frame: {first_fault_chain.get('first_divergence_frame')}")
        print(f"     first_infeasible_frame: {first_fault_chain.get('first_infeasible_frame')}")
        print(
            "     first_speed_above_feasibility_frame: "
            f"{first_fault_chain.get('first_speed_above_feasibility_frame')}"
        )
        print(f"     first_boundary_breach_frame: {first_fault_chain.get('first_boundary_breach_frame')}")
    # Compute odometer from recording for curve event cross-checking.
    # Uses actual vehicle position deltas (not speed integral) for accuracy.
    _odo_at_frame: dict[int, float] = {}
    _track_total_m: float = 0.0
    try:
        import h5py as _h5
        with _h5.File(str(recording_path), 'r') as _hf:
            _pos = _hf['vehicle/position'][:]
            _deltas = np.linalg.norm(np.diff(_pos[:, [0, 2]], axis=0), axis=1)
            _start_dist = 0.0
            try:
                _cfg_json = _hf['meta/runtime_config_json'][()]
                if isinstance(_cfg_json, bytes):
                    _cfg_json = _cfg_json.decode()
                import json
                _rt_cfg = json.loads(_cfg_json)
                _start_dist = float(_rt_cfg.get('start_distance', 0.0) or 0.0)
            except Exception:
                pass
            _cum_odo = np.cumsum(np.insert(_deltas, 0, 0.0)) + _start_dist
            for i in range(len(_cum_odo)):
                _odo_at_frame[i] = float(_cum_odo[i])
            # Track total length from curve intent diagnostics
            _tw = curve_intent_diag.get("track_windows", {})
            if isinstance(_tw, dict):
                _track_total_m = float(_tw.get("total_length_m", 0.0) or 0.0)
            if _track_total_m <= 0:
                _track_total_m = float(summary.get("track_total_length_m", 0.0) or 0.0)
    except Exception:
        pass

    curve_turn_events = summary.get("curve_turn_events", [])
    if curve_turn_events:
        print("   Curve Turn Events:")
        for event in curve_turn_events:
            pp_min = event.get('pp_lookahead_min_m')
            pp_text = f"{float(pp_min):.2f} m" if pp_min is not None else "N/A"
            curve_start_frame = event.get("curve_start_frame")
            onset_distance = event.get("steering_onset_distance_m")
            onset_ttc = event.get("steering_onset_ttc_s")
            onset_text = (
                f"{float(onset_distance):.2f} m / {float(onset_ttc):.2f} s"
                if onset_distance is not None and onset_ttc is not None
                else "N/A"
            )
            onset_delta_frames = event.get("steering_onset_minus_curve_start_frames")
            onset_delta_text = (
                f"{int(onset_delta_frames):+d} fr"
                if onset_delta_frames is not None
                else "N/A"
            )
            shorten_step_min = event.get("pp_pre_floor_shorten_step_min_m_per_frame")
            rescue_delta_max = event.get("pp_floor_rescue_delta_max_m")
            rescue_delta_mean = event.get("pp_floor_rescue_delta_mean_m")
            entry_severity_p50 = event.get("curve_local_entry_severity_p50")
            entry_on_p50 = event.get("curve_local_entry_on_effective_p50")
            entry_dist_start_p50 = event.get("curve_local_phase_distance_start_effective_p50_m")
            entry_driver_mode = event.get("curve_local_entry_driver_mode")
            owner_mode = event.get("reference_lookahead_owner_mode")
            owner_source = event.get("reference_lookahead_entry_weight_source")
            fallback_rate = event.get("reference_lookahead_fallback_active_rate")
            _entry_fr = event.get('entry_frame', 0)
            _odo_val = _odo_at_frame.get(_entry_fr if isinstance(_entry_fr, int) else 0)
            _odo_text = f"{_odo_val:.0f}m" if _odo_val is not None else "N/A"
            print(
                f"     C{event.get('curve_index', '?')}: entry={_entry_fr}, "
                f"exit={event.get('exit_frame')}, odo={_odo_text}, peak|lat|={float(event.get('peak_lateral_error_m', 0.0)):.3f} m, "
                f"curveStart={curve_start_frame}, ppLdMin={pp_text}, onset={onset_text}, onsetVsStart={onset_delta_text}, "
                f"entrySevP50={(float(entry_severity_p50) if entry_severity_p50 is not None else float('nan')):.2f}, "
                f"entryOnP50={(float(entry_on_p50) if entry_on_p50 is not None else float('nan')):.2f}, "
                f"entryDistStartP50={(float(entry_dist_start_p50) if entry_dist_start_p50 is not None else float('nan')):.2f} m, "
                f"entryDriver={entry_driver_mode or 'N/A'}, "
                f"owner={owner_mode or 'N/A'}, source={owner_source or 'N/A'}, "
                f"fallbackRate={(float(fallback_rate) if fallback_rate is not None else float('nan')):.1f}%, "
                f"preLdStepMin={(float(shorten_step_min) if shorten_step_min is not None else float('nan')):.3f} m/fr, "
                f"floorRescueMax={(float(rescue_delta_max) if rescue_delta_max is not None else float('nan')):.3f} m, "
                f"floorRescueMean={(float(rescue_delta_mean) if rescue_delta_mean is not None else float('nan')):.3f} m, "
                f"lateTurnIn={'YES' if event.get('late_turn_in') else 'NO'}, "
                f"errorPattern={event.get('curve_error_pattern', 'N/A')}, "
                f"sameSign={'{:.0%}'.format(float(event['curve_error_same_sign_rate'])) if event.get('curve_error_same_sign_rate') is not None else 'N/A'}, "
                f"meanErr={'{:+.3f}'.format(float(event['curve_error_mean_m'])) if event.get('curve_error_mean_m') is not None else 'N/A'} m"
            )
    curve_straight_segments = summary.get("curve_straight_segments", [])
    if curve_straight_segments:
        print("   Straight Segment Latch Inspector:")
        for segment in curve_straight_segments:
            print(
                f"     S{segment.get('straight_index', '?')}: frames={segment.get('start_frame')}-{segment.get('end_frame')}, "
                f"far={float(segment.get('far_preview_active_rate', 0.0) or 0.0):.1f}%, "
                f"local={float(segment.get('local_active_rate', 0.0) or 0.0):.1f}%, "
                f"pingpong={int(segment.get('watchdog_pingpong_count', 0) or 0)}, "
                f"gateP50={(segment.get('gate_weight_p50') if segment.get('gate_weight_p50') is not None else float('nan')):.2f}, "
                f"rawP50={(segment.get('local_phase_raw_p50') if segment.get('local_phase_raw_p50') is not None else float('nan')):.2f}"
            )
    print()

    print("18. SAFETY METRICS")
    print("-" * 80)
    print(f"   Out-of-Lane Events: {int(safety.get('out_of_lane_events', 0))}")
    print(f"   Out-of-Lane Time: {safety.get('out_of_lane_time', 0.0):.1f}%")
    if safety.get("out_of_lane_events_full_run") is not None:
        print(
            "   Out-of-Lane Events (Full Run): "
            f"{int(safety.get('out_of_lane_events_full_run', 0))}"
        )
    if safety.get("out_of_lane_time_full_run") is not None:
        print(
            "   Out-of-Lane Time (Full Run): "
            f"{float(safety.get('out_of_lane_time_full_run', 0.0)):.1f}%"
        )
    if safety.get("out_of_lane_event_at_failure_boundary") is not None:
        print(
            "   Out-of-Lane At Failure Boundary: "
            f"{'YES' if safety.get('out_of_lane_event_at_failure_boundary') else 'NO'}"
        )
    print()

    mpc_health = summary.get("mpc_health")
    if mpc_health and mpc_health.get("mpc_frames", 0) > 0:
        print("19. MPC HEALTH")
        print("-" * 80)
        lmpc_n = mpc_health.get('lmpc_frames', mpc_health['mpc_frames'])
        nmpc_n = mpc_health.get('nmpc_frames', 0)
        print(f"   MPC Active Rate: {mpc_health['mpc_rate'] * 100:.1f}% ({mpc_health['mpc_frames']} frames — LMPC: {lmpc_n}, NMPC: {nmpc_n})")
        if mpc_health.get("feasibility_rate") is not None:
            gate = "PASS" if mpc_health.get("feasibility_gate_pass") else "FAIL"
            print(f"   LMPC Feasibility Rate: {mpc_health['feasibility_rate'] * 100:.2f}% [{gate}] (gate: >=99.5%)")
        if mpc_health.get("solve_time_p95_ms") is not None:
            gate = "PASS" if mpc_health.get("solve_time_gate_pass") else "FAIL"
            print(
                f"   LMPC Solve Time P50/P95/Max: "
                f"{mpc_health['solve_time_p50_ms']:.2f} / "
                f"{mpc_health['solve_time_p95_ms']:.2f} / "
                f"{mpc_health['solve_time_max_ms']:.2f} ms [{gate}] (gate: P95<=5ms)"
            )
        if mpc_health.get("fallback_rate") is not None:
            print(f"   LMPC Fallback Rate: {mpc_health['fallback_rate'] * 100:.2f}%")
        if mpc_health.get("budget_exceeded_rate") is not None:
            gate = "PASS" if mpc_health.get("budget_exceeded_gate_pass") else "FAIL"
            print(f"   Lateral Accel Budget Exceeded: {mpc_health['budget_exceeded_rate'] * 100:.3f}% [{gate}] (gate: <1.0%)")
        if mpc_health.get("max_consecutive_failures") is not None:
            print(f"   LMPC Max Consecutive Failures: {mpc_health['max_consecutive_failures']}")
        if nmpc_n > 0:
            if mpc_health.get("nmpc_feasibility_rate") is not None:
                gate = "PASS" if mpc_health.get("nmpc_solve_time_gate_pass") else "FAIL"
                print(f"   NMPC Feasibility Rate: {mpc_health['nmpc_feasibility_rate'] * 100:.2f}%")
            if mpc_health.get("nmpc_solve_time_p95_ms") is not None:
                gate = "PASS" if mpc_health.get("nmpc_solve_time_gate_pass") else "FAIL"
                print(f"   NMPC Solve Time P95: {mpc_health['nmpc_solve_time_p95_ms']:.2f}ms [{gate}] (gate: P95<=20ms)")
            if mpc_health.get("nmpc_fallback_rate") is not None:
                print(f"   NMPC Fallback Rate: {mpc_health['nmpc_fallback_rate'] * 100:.2f}%")
        # Regime transition events (Step 3D)
        try:
            import h5py
            with h5py.File(recording_path, "r") as f:
                if "control/regime" in f:
                    regime = np.array(f["control/regime"][:])
                    speed_arr = np.array(f["vehicle/speed"][:]) if "vehicle/speed" in f else None
                    teleport_arr = (
                        np.array(f["control/teleport_detected"][:]) > 0
                        if "control/teleport_detected" in f else None
                    )
                    transitions = []
                    forced_resets = 0
                    for i in range(1, len(regime)):
                        prev_mpc = regime[i - 1] >= 0.5
                        curr_mpc = regime[i] >= 0.5
                        if prev_mpc != curr_mpc:
                            direction = "PP→MPC" if curr_mpc else "MPC→PP"
                            spd = float(speed_arr[i]) if speed_arr is not None and i < len(speed_arr) else None
                            # Classify MPC→PP: forced reset (teleport guard) or natural downshift
                            cause = ""
                            if not curr_mpc:  # MPC→PP
                                if teleport_arr is not None and i < len(teleport_arr) and teleport_arr[i]:
                                    cause = " (forced reset — teleport guard)"
                                    forced_resets += 1
                                elif teleport_arr is None and spd is not None and spd > 6.0:
                                    # Legacy recording: no teleport field — flag anomalous high-speed drop
                                    cause = " (anomalous — investigate cadence)"
                                    forced_resets += 1
                            transitions.append((i, direction, spd, cause))
                    if transitions:
                        print(f"   Regime Transitions: {len(transitions)}")
                        if forced_resets > 0:
                            print(f"   *** {forced_resets} forced reset(s) detected — "
                                  f"PP ran at high speed due to false teleport guard trigger ***")
                        for fr, direction, spd, cause in transitions[:10]:
                            spd_str = f" @ {spd:.1f} m/s" if spd is not None else ""
                            print(f"     Frame {fr}: {direction}{spd_str}{cause}")
                        if len(transitions) > 10:
                            print(f"     ... and {len(transitions) - 10} more")
        except Exception:
            pass
        print()

    # Track-end detection: flag if recording ended near the track boundary
    # Track-end cosmetic note removed 2026-04-17: the OOL computation in
    # drive_summary_core.py now has GT plausibility filtering (GT_LANE_BOUNDARY_MAX_ABS_M),
    # so mesh-seam artifacts no longer inflate the OOL count and the note's rationale
    # ("emergency stop may be caused by driving off the end of a non-looping track")
    # is obsolete.

    print("20. RECOMMENDATIONS")
    print("-" * 80)
    if recommendations:
        for idx, recommendation in enumerate(recommendations, 1):
            print(f"   {idx}. {recommendation}")
    else:
        print("   ✓ No recommendations")
    print()

    # Section 15: Grade Impact (only for graded recordings)
    grade_metrics = summary.get("grade_metrics")
    if grade_metrics is not None:
        print("21. GRADE IMPACT")
        print("-" * 80)
        print(f"   Max Grade: {grade_metrics['grade_max_pct']:.1f}%"
              f"    Pitch P95: {grade_metrics['pitch_p95_deg']:.1f}\u00b0")
        print(f"   Downhill Speed P95: {grade_metrics['speed_on_downhill_p95']:.1f} m/s"
              f"    Downhill Overspeed: {grade_metrics['overspeed_on_downhill_rate']:.1f}%")
        print(f"   Grade Compensation Active: {grade_metrics['grade_compensation_active_rate']:.1f}%"
              f"    Graded Frames: {grade_metrics['graded_frames']}")
        print()

    # Section 16: ACC Performance (only when ACC was active)
    acc_health = summary.get("acc_health")
    acc_comfort_contract = summary.get("acc_comfort_contract") or {}
    acc_detection_contract = summary.get("acc_detection_contract") or {}
    lead_continuity_contract = summary.get("lead_continuity_contract") or {}
    if acc_health is not None:
        print("22. ACC PERFORMANCE")
        print("-" * 80)
        print(f"   ACC Active: {acc_health['acc_active_pct']:.1f}% of frames")
        print()

        def _gate(val, pass_val):
            return "PASS" if pass_val else "FAIL"

        print("   \u2500\u2500 Longitudinal Safety (\u2192 Safety layer score) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
        coll = acc_health.get("acc_collision_events", 0)
        ttc_viol = acc_health.get("acc_ttc_violation_events", 0)
        nm = acc_health.get("acc_near_miss_events", 0)
        ttc_warn_pct = acc_health.get("acc_ttc_warning_pct", 0.0)
        ttc_p05 = acc_health.get("acc_ttc_p05_s", 999.0)
        min_gap = acc_health.get("acc_min_gap_m", 0.0)
        print(f"   Collision Events:       {coll:<10d} [{_gate(coll, acc_health.get('collision_gate_pass', True))} = 0]"
              f"               Tier 1")
        print(f"   TTC Violations:         {ttc_viol:<10d} [{_gate(ttc_viol, acc_health.get('ttc_violation_gate_pass', True))} = 0]"
              f"               Tier 1")
        print(f"   Near-Miss Events:       {nm:<10d} [{_gate(nm, acc_health.get('near_miss_gate_pass', True))} = 0]"
              f"   gap < 2.0m")
        print(f"   TTC Warning Zone:       {ttc_warn_pct:<9.1f}% [{_gate(ttc_warn_pct, acc_health.get('ttc_warning_gate_pass', True))} < 5%]"
              f"  1.5\u20132.5s")
        print(f"   TTC P05:                {ttc_p05:<9.2f}s [{_gate(ttc_p05, acc_health.get('ttc_p05_gate_pass', True))} \u2265 2.0s]")
        print(f"   Min Gap Observed:       {min_gap:.1f}m          (informational)")
        print()

        print("   \u2500\u2500 Following Convergence (\u2192 instrumentation, not runtime tuning) \u2500\u2500\u2500")
        print(
            f"   Actual / Simple / Dynamic Gap P50:"
            f"{acc_health.get('acc_actual_gap_p50_m', 0.0):>8.2f} /"
            f" {acc_health.get('acc_target_gap_p50_m', 0.0):>6.2f} /"
            f" {acc_health.get('acc_dynamic_gap_p50_m', 0.0):<6.2f}m"
        )
        print(
            f"   Simple / Dynamic / Eq Gap Error P50:"
            f" {acc_health.get('acc_gap_error_p50_m', 0.0):>7.2f} /"
            f" {acc_health.get('acc_dynamic_gap_error_p50_m', 0.0):>7.2f} /"
            f" {acc_health.get('acc_equilibrium_gap_error_p50_m', 0.0):<7.2f}m"
        )
        print(
            f"   Gap > Target (+2/+5/+10m):"
            f" {acc_health.get('acc_gap_above_target_plus_2m_rate', 0.0):.1f}% /"
            f" {acc_health.get('acc_gap_above_target_plus_5m_rate', 0.0):.1f}% /"
            f" {acc_health.get('acc_gap_above_target_plus_10m_rate', 0.0):.1f}%"
        )
        print(
            f"   Following Regime Mode:  {acc_health.get('acc_following_regime_mode', 'unknown')}"
            f"  (track/close/policy/lead/eq/detect/compressed="
            f"{acc_health.get('acc_tracking_rate', 0.0):.1f}%/"
            f"{acc_health.get('acc_closing_rate', 0.0):.1f}%/"
            f"{acc_health.get('acc_policy_limited_tracking_rate', 0.0):.1f}%/"
            f"{acc_health.get('acc_lead_limited_tracking_rate', 0.0):.1f}%/"
            f"{acc_health.get('acc_equilibrium_limited_tracking_rate', 0.0):.1f}%/"
            f"{acc_health.get('acc_detection_limited_following_rate', 0.0):.1f}%/"
            f"{acc_health.get('acc_compressed_rate', 0.0):.1f}%)"
        )
        print(
            f"   Lead Speed / Closure Reserve P50:"
            f" {acc_health.get('acc_lead_speed_estimate_p50_mps', 0.0):.2f} /"
            f" {acc_health.get('acc_closure_reserve_p50_mps', 0.0):.2f} m/s"
        )
        print()

        print("   \u2500\u2500 Following Comfort (\u2192 LongitudinalComfort layer) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
        gap_rmse = acc_health.get("acc_gap_rmse_m", 0.0)
        ttc_min = acc_health.get("acc_ttc_min_s", 999.0)
        jerk_p95 = acc_health.get("acc_jerk_gate_value_mps3", 0.0)
        emrg = acc_health.get("acc_emergency_brake_events", 0)
        print(f"   Gap RMSE:               {gap_rmse:<9.2f}m [{_gate(gap_rmse, acc_health.get('gap_rmse_gate_pass', True))} \u2264 {acc_health.get('acc_gap_rmse_gate_m', 35.0):.0f}m]")
        print(f"   TTC Minimum:            {ttc_min:<9.2f}s [{_gate(ttc_min, acc_health.get('ttc_min_gate_pass', True))} \u2265 2.00s]")
        print(
            f"   Jerk Gate Metric:       {jerk_p95:<9.3f}m/s\u00b3"
            f" [{_gate(jerk_p95, acc_health.get('jerk_gate_pass', True))} \u2264 4.0]"
            f"  ({acc_health.get('acc_jerk_gate_metric_role', 'unknown')})"
        )
        print(
            f"   Jerk Raw / Filtered / Commanded:"
            f" {acc_health.get('acc_jerk_p95_raw_mps3', 0.0):.1f} /"
            f" {acc_health.get('acc_jerk_p95_filtered_mps3', 0.0):.3f} /"
            f" {acc_health.get('acc_commanded_jerk_p95_mps3', 0.0):.3f} m/s\u00b3"
        )
        print(
            f"   Target \u0394P95 / Cmd Accel \u0394P95:"
            f" {acc_health.get('acc_target_speed_delta_p95_mps', 0.0):.4f} m/s /"
            f" {acc_health.get('acc_commanded_accel_delta_p95_mps2', 0.0):.4f} m/s\u00b2"
        )
        if acc_comfort_contract:
            print(
                f"   Artifact Likely / Dominant Hotspot:"
                f" {'YES' if acc_comfort_contract.get('scoring_artifact_likely') else 'NO'} /"
                f" {acc_comfort_contract.get('hotspot_dominant_attribution_mode', 'none')}"
            )
        print(f"   Emergency Brake Events: {emrg}          (informational \u2014 IDM quality)")
        print()

        print("   \u2500\u2500 Sensing \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
        det_rate = acc_health.get("acc_detection_rate", 1.0)
        print(f"   Detection Rate:         {det_rate * 100.0:<9.1f}% [{_gate(det_rate, acc_health.get('detection_gate_pass', True))} \u2265 95%]")
        if acc_detection_contract:
            print(
                f"   Stable Detection / Recent Loss:"
                f" {acc_detection_contract.get('stable_detection_rate_pct', 0.0):.1f}% /"
                f" {acc_detection_contract.get('recent_detection_loss_rate_pct', 0.0):.1f}%"
            )
            print(
                f"   Loss Events / Max No-Detect / Mode:"
                f" {acc_detection_contract.get('detection_loss_event_count', 0)} /"
                f" {acc_detection_contract.get('no_detect_run_length_max', 0)} /"
                f" {acc_detection_contract.get('issue_mode', 'none')}"
            )
        if lead_continuity_contract:
            print(
                f"   Candidate / Missed Candidate / Reject:"
                f" {lead_continuity_contract.get('candidate_present_rate_pct', 0.0):.1f}% /"
                f" {lead_continuity_contract.get('missed_candidate_rate_pct', 0.0):.1f}% /"
                f" {lead_continuity_contract.get('reject_reason_mode', 'none')}"
            )
            print(
                f"   Cone / Same-Lane / Wrong-Lane / Opposite:"
                f" {lead_continuity_contract.get('out_of_cone_rate_pct', 0.0):.1f}% /"
                f" {lead_continuity_contract.get('out_of_cone_same_lane_rate_pct', 0.0):.1f}% /"
                f" {lead_continuity_contract.get('out_of_cone_wrong_lane_rate_pct', 0.0):.1f}% /"
                f" {lead_continuity_contract.get('out_of_cone_opposite_direction_rate_pct', 0.0):.1f}%"
            )
            print(
                f"   Azimuth P50/P95 / HeadingΔ P50/P95:"
                f" {lead_continuity_contract.get('target_azimuth_abs_p50_deg', 0.0):.1f}/"
                f"{lead_continuity_contract.get('target_azimuth_abs_p95_deg', 0.0):.1f} deg /"
                f" {lead_continuity_contract.get('target_heading_delta_abs_p50_deg', 0.0):.1f}/"
                f"{lead_continuity_contract.get('target_heading_delta_abs_p95_deg', 0.0):.1f} deg"
            )
            print(
                f"   Same-Lane Conf P50/P05 / Arc P50:"
                f" {lead_continuity_contract.get('same_lane_confidence_p50', 0.0):.2f}/"
                f"{lead_continuity_contract.get('same_lane_confidence_p05', 0.0):.2f} /"
                f" {lead_continuity_contract.get('target_arc_distance_p50_m', 0.0):.1f} m"
            )
            print(
                f"   Assoc Eligible / Track Active / Source:"
                f" {lead_continuity_contract.get('association_eligible_rate_pct', 0.0):.1f}% /"
                f" {lead_continuity_contract.get('track_active_rate_pct', 0.0):.1f}% /"
                f" {lead_continuity_contract.get('track_source_mode', 'none')}"
            )
            print(
                f"   Raw / Assoc / Hold:"
                f" {lead_continuity_contract.get('raw_detect_rate_pct', 0.0):.1f}% /"
                f" {lead_continuity_contract.get('same_lane_association_rate_pct', 0.0):.1f}% /"
                f" {lead_continuity_contract.get('continuity_hold_rate_pct', 0.0):.1f}%"
            )
            print(
                f"   Hold Reason / Drop / Age P95:"
                f" {lead_continuity_contract.get('continuity_hold_reason_mode', 'none')} /"
                f" {lead_continuity_contract.get('continuity_drop_reason_mode', 'none')} /"
                f" {lead_continuity_contract.get('continuity_hold_age_p95_ms', 0.0):.1f} ms"
            )
        print()

    # Section 23: Trajectory Smoothing & Curve-Segmented Diagnostics
    # Root-cause diagnostic for speed-adaptive smoothing regression.
    # Segments lateral error, smoothing lag, and MPC recovery by straight vs curve.
    try:
        import h5py as _h5
        with _h5.File(recording_path, "r") as f:
            _has_fields = all(k in f for k in [
                "trajectory/diag_raw_ref_x",
                "trajectory/diag_smoothed_ref_x",
                "trajectory/reference_point_curvature",
                "control/lateral_error",
                "control/regime",
            ])
            if _has_fields:
                _raw_x = np.array(f["trajectory/diag_raw_ref_x"][:], dtype=float)
                _smooth_x = np.array(f["trajectory/diag_smoothed_ref_x"][:], dtype=float)
                _curv = np.array(f["trajectory/reference_point_curvature"][:], dtype=float)
                _e_lat = np.array(f["control/lateral_error"][:], dtype=float)
                _regime = np.array(f["control/regime"][:], dtype=float)
                _speed = np.array(f["vehicle/speed"][:], dtype=float) if "vehicle/speed" in f else np.zeros_like(_e_lat)
                _mpc_e_lat = np.array(f["control/mpc_e_lat"][:], dtype=float) if "control/mpc_e_lat" in f else None
                _recovery = np.array(f["control/mpc_recovery_mode_suppressed"][:], dtype=float) if "control/mpc_recovery_mode_suppressed" in f else None
                _smith_raw = np.array(f["control/mpc_smith_raw_e_lat"][:], dtype=float) if "control/mpc_smith_raw_e_lat" in f else None
                _smith_pred = np.array(f["control/mpc_smith_e_lat_predicted"][:], dtype=float) if "control/mpc_smith_e_lat_predicted" in f else None

                _n = min(len(_raw_x), len(_smooth_x), len(_curv), len(_e_lat), len(_regime))
                _raw_x = _raw_x[:_n]; _smooth_x = _smooth_x[:_n]; _curv = _curv[:_n]
                _e_lat = _e_lat[:_n]; _regime = _regime[:_n]; _speed = _speed[:_n]

                _mpc_mask = _regime >= 0.5
                _curve_mask = np.abs(_curv) > 0.0005
                _straight_mpc = (~_curve_mask) & _mpc_mask
                _curve_mpc = _curve_mask & _mpc_mask
                _lag = _smooth_x - _raw_x

                print("23. TRAJECTORY SMOOTHING & CURVE DIAGNOSTICS")
                print("-" * 80)

                # --- Smoothing lag ---
                print("   ── Smoothing Lag (smoothed_ref_x − raw_ref_x) ────────────────")
                for _lbl, _mask in [("All", np.ones(_n, dtype=bool)), ("Straight+MPC", _straight_mpc), ("Curve+MPC", _curve_mpc)]:
                    if _mask.sum() > 0:
                        _al = np.abs(_lag[_mask])
                        print(f"   {_lbl:15s}  P50={np.median(_al):.4f}m  P95={np.percentile(_al,95):.4f}m  Max={np.max(_al):.4f}m")
                print()

                # --- Lag as % of error (smoking-gun metric) ---
                if _curve_mpc.sum() > 10:
                    _abs_lag_c = np.abs(_lag[_curve_mpc])
                    _abs_elat_c = np.abs(_e_lat[_curve_mpc])
                    _safe_elat = np.maximum(_abs_elat_c, 1e-6)
                    _lag_pct = _abs_lag_c / _safe_elat
                    _corr = float(np.corrcoef(_abs_lag_c, _abs_elat_c)[0, 1]) if _curve_mpc.sum() > 10 else 0.0
                    print("   ── Smoothing Lag vs Lateral Error (curve+MPC) ─────────────")
                    print(f"   Lag as % of |e_lat|:  P50={100*np.median(_lag_pct):.1f}%  P95={100*np.percentile(_lag_pct,95):.1f}%")
                    print(f"   Lag-error correlation: {_corr:.3f}  ({'CAUSAL' if abs(_corr) > 0.5 else 'weak' if abs(_corr) > 0.2 else 'none'})")
                    _lag_flag = "YES" if np.percentile(_lag_pct, 95) > 0.5 else "NO"
                    print(f"   Smoothing lag dominates curve error: {_lag_flag}")
                    print()

                # --- Segmented lateral error ---
                print("   ── Lateral Error by Segment ──────────────────────────────────")
                for _lbl, _mask in [("Straight+MPC", _straight_mpc), ("Curve+MPC", _curve_mpc)]:
                    if _mask.sum() > 0:
                        _ae = np.abs(_e_lat[_mask])
                        print(f"   {_lbl:15s}  P50={np.median(_ae):.3f}m  P95={np.percentile(_ae,95):.3f}m  Max={np.max(_ae):.3f}m  ({_mask.sum()} frames)")
                print()

                # --- MPC e_lat by segment (what MPC actually sees) ---
                if _mpc_e_lat is not None:
                    _me = _mpc_e_lat[:_n]
                    print("   ── MPC e_lat by Segment (post Smith+attenuation) ─────────")
                    for _lbl, _mask in [("Straight+MPC", _straight_mpc), ("Curve+MPC", _curve_mpc)]:
                        if _mask.sum() > 0:
                            _ame = np.abs(_me[_mask])
                            print(f"   {_lbl:15s}  P50={np.median(_ame):.3f}m  P95={np.percentile(_ame,95):.3f}m")
                    # Attenuation ratio
                    if _smith_raw is not None and _mpc_mask.sum() > 0:
                        _sr = np.abs(_smith_raw[:_n][_mpc_mask])
                        _me_mpc = np.abs(_me[_mpc_mask])
                        _safe_sr = np.maximum(_sr, 1e-6)
                        _ratio = _me_mpc / _safe_sr
                        print(f"   MPC/raw ratio (P50):  {np.median(_ratio):.3f}  (1.0=no attenuation, <1.0=attenuated)")
                    print()

                # --- Recovery mode ---
                if _recovery is not None:
                    _rec = _recovery[:_n]
                    print("   ── Recovery Mode Suppression ─────────────────────────────")
                    _rec_all = float(_rec[_mpc_mask].mean()) * 100 if _mpc_mask.sum() > 0 else 0
                    print(f"   All MPC:       {_rec_all:.1f}%  ({int(_rec[_mpc_mask].sum())}/{int(_mpc_mask.sum())} frames)")
                    if _curve_mpc.sum() > 0:
                        _rec_curve = float(_rec[_curve_mpc].mean()) * 100
                        _rec_flag = "ELEVATED" if _rec_curve > 10 else "OK"
                        print(f"   Curve+MPC:     {_rec_curve:.1f}%  [{_rec_flag}]  (gate: <=10%)")
                    if _straight_mpc.sum() > 0:
                        _rec_str = float(_rec[_straight_mpc].mean()) * 100
                        print(f"   Straight+MPC:  {_rec_str:.1f}%")
                    print()

                # --- Sign agreement (Smith predictor relevance) ---
                _e_head = np.array(f["control/heading_error"][:_n], dtype=float) if "control/heading_error" in f else None
                if _e_head is not None and _mpc_mask.sum() > 0:
                    _sa = (_e_lat[_mpc_mask] * _e_head[_mpc_mask]) >= 0
                    _sa_pct = float(_sa.mean()) * 100
                    print("   ── Sign Agreement (Smith predictor relevance) ─────────────")
                    print(f"   All MPC sign agree: {_sa_pct:.1f}%  ({'Smith guard inactive' if _sa_pct > 95 else 'Smith guard active'})")
                    print()

                # --- Speed in segments ---
                print("   ── Speed by Segment ─────────────────────────────────────────")
                if _straight_mpc.sum() > 0:
                    print(f"   Straight+MPC:  P50={np.median(_speed[_straight_mpc]):.1f}  Max={np.max(_speed[_straight_mpc]):.1f} m/s")
                if _curve_mpc.sum() > 0:
                    print(f"   Curve+MPC:     P50={np.median(_speed[_curve_mpc]):.1f}  Max={np.max(_speed[_curve_mpc]):.1f} m/s")
                print()

                # --- r_steer_rate scheduling ---
                if "control/mpc_r_steer_rate_effective" in f:
                    _rsr = np.array(f["control/mpc_r_steer_rate_effective"][:_n], dtype=float)
                    _rsr_mpc = _rsr[_mpc_mask]
                    if len(_rsr_mpc) > 0:
                        print("   ── r_steer_rate Scheduling ──────────────────────────────────")
                        print(f"   Effective r_steer_rate:  Mean={np.mean(_rsr_mpc):.3f}  Min={np.min(_rsr_mpc):.3f}  Max={np.max(_rsr_mpc):.3f}")
                        _rsr_diffs = np.abs(np.diff(_rsr_mpc))
                        _rsr_transitions = int(np.sum(_rsr_diffs > 0.1))
                        print(f"   Band transitions: {_rsr_transitions}")
                        if np.max(_rsr_mpc) - np.min(_rsr_mpc) < 0.01:
                            print(f"   NOTE: Scheduling appears INACTIVE (constant value)")
                        print()
                # Dual-mechanism summary: scheduling (rate) + attenuation (gain)
                _has_sched = "control/mpc_r_steer_rate_effective" in f
                _has_atten = False
                if "control/mpc_smith_raw_e_lat" in f and "control/mpc_e_lat" in f:
                    _raw_all = np.abs(np.array(f["control/mpc_smith_raw_e_lat"][:_n], dtype=float))
                    _inp_all = np.abs(np.array(f["control/mpc_e_lat"][:_n], dtype=float))
                    _va = (_raw_all > 0.01) & _mpc_mask
                    if _va.sum() > 10:
                        _atten_ratio = float(np.median(_inp_all[_va] / _raw_all[_va]))
                        _has_atten = _atten_ratio < 0.95
                print(f"   ── Oscillation Damping Status ────────────────────────────")
                print(f"   r_steer_rate scheduling (rate penalty): {'ACTIVE' if _has_sched else 'N/A'}")
                print(f"   e_lat speed attenuation (gain reduction): {'ACTIVE' if _has_atten else 'inactive/minimal'}")
                print()

    except Exception as _exc:
        print("23. TRAJECTORY SMOOTHING & CURVE DIAGNOSTICS")
        print("-" * 80)
        print(f"   (skipped — {_exc})")
        print()

    # ── Section 24: Curvature Distribution ──────────────────────────────────────
    # Shows what κ values exist on this track and how frames distribute across bins.
    # Helps decide regime thresholds without ad-hoc HDF5 inspection.
    try:
        import h5py as _h5
        with _h5.File(recording_path, "r") as f:
            _has_curv = "control/curvature_map_abs" in f or "control/curvature_primary_abs" in f
            if _has_curv:
                # Prefer map curvature (stable, from track geometry); fall back to primary
                _curv_key = "control/curvature_map_abs" if "control/curvature_map_abs" in f else "control/curvature_primary_abs"
                _kappa = np.abs(np.array(f[_curv_key][:], dtype=float))
                _n_k = len(_kappa)

                print("24. CURVATURE DISTRIBUTION")
                print("-" * 80)
                print(f"   Source: {_curv_key.split('/')[-1]}  ({_n_k} frames)")
                print(f"   Range: κ = {np.min(_kappa):.6f} – {np.max(_kappa):.6f}  (R = {1/max(np.max(_kappa), 1e-6):.0f} – {1/max(np.min(_kappa[_kappa > 1e-6]), 1e-6):.0f} m)" if np.any(_kappa > 1e-6) else f"   Range: κ = 0 (straight track)")
                print()

                # Histogram bins — adaptive based on data range
                _bins = [0.0, 0.001, 0.003, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030, 0.050, 1.0]
                _bin_labels = ["0–0.001 (straight)", "0.001–0.003 (gentle)", "0.003–0.005 (mild)",
                               "0.005–0.008 (moderate)", "0.008–0.010 (firm)", "0.010–0.015 (tight)",
                               "0.015–0.020 (R50–67)", "0.020–0.030 (R33–50)", "0.030–0.050 (R20–33)",
                               "0.050+ (R<20)"]
                print("   ── Curvature Histogram ───────────────────────────────────────")
                print(f"   {'Bin':30s}  {'Frames':>7s}  {'%':>6s}  {'Bar'}")
                _max_bar = 40
                for i in range(len(_bins) - 1):
                    _count = int(np.sum((_kappa >= _bins[i]) & (_kappa < _bins[i+1])))
                    _pct = 100.0 * _count / max(_n_k, 1)
                    _bar_len = int(round(_pct / 100.0 * _max_bar))
                    if _count > 0:
                        print(f"   {_bin_labels[i]:30s}  {_count:7d}  {_pct:5.1f}%  {'█' * _bar_len}")

                # Unique values (detect quantized maps)
                _unique = np.unique(np.round(_kappa, 6))
                if len(_unique) <= 10:
                    print(f"\n   Discrete values detected ({len(_unique)}): {', '.join(f'{v:.4f}' for v in _unique)}")
                    print(f"   NOTE: Map curvature is quantized — threshold selection between discrete values is equivalent")
                print()
    except Exception as _exc:
        print("24. CURVATURE DISTRIBUTION")
        print("-" * 80)
        print(f"   (skipped — {_exc})")
        print()

    # ── Section 25: Per-Regime Error Breakdown ──────────────────────────────────
    # Lateral RMSE separately for each control regime (PP, LMPC, NMPC, Stanley).
    # Answers: "which controller is responsible for tracking errors?"
    try:
        import h5py as _h5
        with _h5.File(recording_path, "r") as f:
            _has_regime = all(k in f for k in ["control/regime", "control/lateral_error",
                                                "control/regime_blend_weight"])
            if _has_regime:
                _regime = np.array(f["control/regime"][:], dtype=float)
                _e_lat = np.abs(np.array(f["control/lateral_error"][:], dtype=float))
                _blend = np.array(f["control/regime_blend_weight"][:], dtype=float)
                _speed = np.array(f["vehicle/speed"][:], dtype=float) if "vehicle/speed" in f else np.zeros_like(_e_lat)
                _curv_key = "control/curvature_map_abs" if "control/curvature_map_abs" in f else \
                            "control/curvature_primary_abs" if "control/curvature_primary_abs" in f else None
                _kappa = np.abs(np.array(f[_curv_key][:], dtype=float)) if _curv_key else np.zeros_like(_e_lat)
                _n_r = min(len(_regime), len(_e_lat), len(_blend), len(_speed), len(_kappa))
                _regime = _regime[:_n_r]; _e_lat = _e_lat[:_n_r]; _blend = _blend[:_n_r]
                _speed = _speed[:_n_r]; _kappa = _kappa[:_n_r]

                # Regime masks (settled = blend >= 0.95 to exclude transition frames)
                _settled = _blend >= 0.95
                _transition = ~_settled
                _curve_mask = _kappa > 0.003

                _regime_defs = [
                    ("Stanley",  -1.0, -0.5),   # regime < -0.5
                    ("PP",        0.0,  0.5),   # -0.5 <= regime < 0.5
                    ("LMPC",      1.0,  1.5),   # 0.5 <= regime < 1.5
                    ("NMPC",      2.0,  2.5),   # 1.5 <= regime < 2.5
                ]

                print("25. PER-REGIME ERROR BREAKDOWN")
                print("-" * 80)
                print(f"   {'Regime':10s}  {'Frames':>7s}  {'%':>5s}  {'RMSE':>7s}  {'P50':>7s}  {'P95':>7s}  {'Speed P50':>9s}  {'κ P50':>8s}")
                print(f"   {'─'*10}  {'─'*7}  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*9}  {'─'*8}")

                for _name, _val, _hi in _regime_defs:
                    _mask = _settled & (_regime >= (_val - 0.5)) & (_regime < _hi)
                    _cnt = int(_mask.sum())
                    if _cnt == 0:
                        continue
                    _pct = 100.0 * _cnt / max(_n_r, 1)
                    _rmse = float(np.sqrt(np.mean(_e_lat[_mask] ** 2)))
                    _p50 = float(np.median(_e_lat[_mask]))
                    _p95 = float(np.percentile(_e_lat[_mask], 95))
                    _sp50 = float(np.median(_speed[_mask]))
                    _kp50 = float(np.median(_kappa[_mask]))
                    print(f"   {_name:10s}  {_cnt:7d}  {_pct:4.1f}%  {_rmse:6.4f}m  {_p50:6.4f}m  {_p95:6.4f}m  {_sp50:8.1f}  {_kp50:7.5f}")

                # Transition frames
                _t_cnt = int(_transition.sum())
                if _t_cnt > 0:
                    _t_pct = 100.0 * _t_cnt / max(_n_r, 1)
                    _t_rmse = float(np.sqrt(np.mean(_e_lat[_transition] ** 2)))
                    _t_p50 = float(np.median(_e_lat[_transition]))
                    _t_p95 = float(np.percentile(_e_lat[_transition], 95))
                    _t_sp50 = float(np.median(_speed[_transition]))
                    _t_kp50 = float(np.median(_kappa[_transition]))
                    print(f"   {'Transition':10s}  {_t_cnt:7d}  {_t_pct:4.1f}%  {_t_rmse:6.4f}m  {_t_p50:6.4f}m  {_t_p95:6.4f}m  {_t_sp50:8.1f}  {_t_kp50:7.5f}")
                print()

                # Curve vs straight per regime
                print("   ── Curve vs Straight (per regime, settled only) ─────────────")
                print(f"   {'Regime':10s}  {'Segment':10s}  {'RMSE':>7s}  {'P50':>7s}  {'Frames':>7s}")
                for _name, _val, _hi in _regime_defs:
                    _r_mask = _settled & (_regime >= (_val - 0.5)) & (_regime < _hi)
                    for _seg_name, _seg_mask in [("Straight", ~_curve_mask), ("Curve", _curve_mask)]:
                        _combined = _r_mask & _seg_mask
                        _cnt = int(_combined.sum())
                        if _cnt < 5:
                            continue
                        _rmse = float(np.sqrt(np.mean(_e_lat[_combined] ** 2)))
                        _p50 = float(np.median(_e_lat[_combined]))
                        print(f"   {_name:10s}  {_seg_name:10s}  {_rmse:6.4f}m  {_p50:6.4f}m  {_cnt:7d}")
                print()
    except Exception as _exc:
        print("25. PER-REGIME ERROR BREAKDOWN")
        print("-" * 80)
        print(f"   (skipped — {_exc})")
        print()

    # ── Section 26: Model-Plant Comparison ──────────────────────────────────────
    # Compares MPC's commanded steering vs actual vehicle response.
    # Detects model-plant mismatch (the root cause of MPC under/over-steering).
    try:
        import h5py as _h5
        with _h5.File(recording_path, "r") as f:
            _has_mp = all(k in f for k in ["control/steering", "vehicle/steering_angle_actual",
                                            "control/regime"])
            if _has_mp:
                _steer_cmd = np.array(f["control/steering"][:], dtype=float)
                _steer_actual_raw = np.array(f["vehicle/steering_angle_actual"][:], dtype=float)
                _regime = np.array(f["control/regime"][:], dtype=float)
                _speed = np.array(f["vehicle/speed"][:], dtype=float) if "vehicle/speed" in f else np.zeros_like(_steer_cmd)
                _n_mp = min(len(_steer_cmd), len(_steer_actual_raw), len(_regime), len(_speed))
                _steer_cmd = _steer_cmd[:_n_mp]; _steer_actual_raw = _steer_actual_raw[:_n_mp]
                _regime = _regime[:_n_mp]; _speed = _speed[:_n_mp]

                # Normalize actual steering (stored in degrees) to same scale as command
                # Command is in radians (max_steering), actual is in degrees
                _max_steer_deg = float(np.percentile(np.abs(_steer_actual_raw[_steer_actual_raw != 0]), 99)) if np.any(_steer_actual_raw != 0) else 30.0
                _max_steer_cmd = float(np.percentile(np.abs(_steer_cmd[_steer_cmd != 0]), 99)) if np.any(_steer_cmd != 0) else 0.7

                # MPC heading prediction vs actual
                _mpc_e_heading = np.array(f["control/mpc_e_heading"][:_n_mp], dtype=float) if "control/mpc_e_heading" in f else None
                _gt_heading = np.array(f["control/heading_error"][:_n_mp], dtype=float) if "control/heading_error" in f else None
                _mpc_e_lat = np.array(f["control/mpc_e_lat"][:_n_mp], dtype=float) if "control/mpc_e_lat" in f else None
                _lat_error = np.abs(np.array(f["control/lateral_error"][:_n_mp], dtype=float)) if "control/lateral_error" in f else None

                _mpc_mask = _regime >= 0.5
                _mpc_frames = int(_mpc_mask.sum())

                print("26. MODEL-PLANT COMPARISON")
                print("-" * 80)

                if _mpc_frames < 10:
                    print(f"   MPC active {_mpc_frames} frames — insufficient for comparison")
                else:
                    # Steering command vs actual tracking
                    _cmd_mpc = _steer_cmd[_mpc_mask]
                    _act_mpc = _steer_actual_raw[_mpc_mask]
                    # Convert actual to radians for comparison (Unity uses degrees, negate for sign)
                    _act_rad = np.radians(-_act_mpc)
                    _steer_delta = _cmd_mpc - _act_rad
                    _steer_corr = float(np.corrcoef(_cmd_mpc, _act_rad)[0, 1]) if _mpc_frames > 10 else 0.0

                    print(f"   ── Steering Command vs Actual (MPC frames) ────────────────")
                    print(f"   Command P50/P95:  {np.median(np.abs(_cmd_mpc)):.4f} / {np.percentile(np.abs(_cmd_mpc), 95):.4f} rad")
                    print(f"   Actual P50/P95:   {np.median(np.abs(_act_rad)):.4f} / {np.percentile(np.abs(_act_rad), 95):.4f} rad")
                    print(f"   Delta P50/P95:    {np.median(np.abs(_steer_delta)):.4f} / {np.percentile(np.abs(_steer_delta), 95):.4f} rad")
                    print(f"   Correlation:      {_steer_corr:.4f}")
                    _track_flag = "GOOD" if abs(_steer_corr) > 0.95 else "MISMATCH" if abs(_steer_corr) < 0.8 else "MODERATE"
                    print(f"   Tracking quality: {_track_flag}")
                    print()

                    # MPC e_lat vs scorer's lateral error (reference alignment check)
                    if _mpc_e_lat is not None and _lat_error is not None:
                        _me = np.abs(_mpc_e_lat[_mpc_mask])
                        _le = _lat_error[_mpc_mask]
                        _ref_corr = float(np.corrcoef(_me, _le)[0, 1]) if _mpc_frames > 10 else 0.0
                        _ref_ratio = float(np.median(_me / np.maximum(_le, 1e-6)))
                        _ref_gap = float(np.median(_le - _me))
                        print(f"   ── MPC Reference Alignment ────────────────────────────────")
                        print(f"   MPC e_lat P50:    {np.median(_me):.4f}m")
                        print(f"   Scorer lat P50:   {np.median(_le):.4f}m")
                        print(f"   Median gap:       {_ref_gap:.4f}m  (scorer − MPC)")
                        print(f"   Ratio (MPC/scor): {_ref_ratio:.3f}  (1.0 = aligned)")
                        print(f"   Correlation:      {_ref_corr:.4f}")
                        _align_flag = "ALIGNED" if abs(_ref_gap) < 0.05 and _ref_corr > 0.9 else \
                                      "OFFSET" if abs(_ref_gap) >= 0.05 and _ref_corr > 0.7 else "DIVERGED"
                        print(f"   Reference health: {_align_flag}")
                        print()

                    # Heading prediction accuracy
                    if _mpc_e_heading is not None and _gt_heading is not None:
                        _mh = _mpc_e_heading[_mpc_mask]
                        _gh = _gt_heading[_mpc_mask]
                        _h_corr = float(np.corrcoef(_mh, _gh)[0, 1]) if _mpc_frames > 10 else 0.0
                        _h_rmse = float(np.sqrt(np.mean((_mh - _gh) ** 2)))
                        print(f"   ── Heading Prediction Accuracy ────────────────────────────")
                        print(f"   MPC heading P50:  {np.median(np.abs(_mh)):.4f} rad")
                        print(f"   GT heading P50:   {np.median(np.abs(_gh)):.4f} rad")
                        print(f"   Heading RMSE:     {_h_rmse:.4f} rad")
                        print(f"   Correlation:      {_h_corr:.4f}")
                        print()

                print()
    except Exception as _exc:
        print("26. MODEL-PLANT COMPARISON")
        print("-" * 80)
        print(f"   (skipped — {_exc})")
        print()

    # ── Section 27: Recovery Term Activity ──────────────────────────────────────
    # Activity summary for the lateral-error recovery term (PP pipeline, upstream
    # of rate/jerk limiters). Replaces the deleted orchestrator steering
    # multiplier. Design ref: project_recovery_mode_post_limiter.md.
    try:
        import h5py as _h5
        with _h5.File(recording_path, "r") as f:
            _has_rec = "control/lateral_error_recovery_smoothstep_weight" in f
            print("27. RECOVERY TERM ACTIVITY")
            print("-" * 80)
            if not _has_rec:
                print("   (no recovery-term telemetry in recording — pre-fix run or feature disabled)")
                print()
            else:
                _weight = np.array(f["control/lateral_error_recovery_smoothstep_weight"][:], dtype=float)
                _term = np.array(
                    f["control/lateral_error_recovery_term_applied_rad"][:], dtype=float
                ) if "control/lateral_error_recovery_term_applied_rad" in f else None
                _source_raw = (
                    f["control/lateral_error_recovery_e_lat_source"][:]
                    if "control/lateral_error_recovery_e_lat_source" in f
                    else None
                )
                _shadow = (
                    np.array(f["control/lateral_error_recovery_shadow_mode"][:], dtype=int)
                    if "control/lateral_error_recovery_shadow_mode" in f
                    else None
                )
                _n = len(_weight)
                _active = _weight > 0.01
                _sat = _weight >= 0.999
                _n_active = int(_active.sum())
                _n_sat = int(_sat.sum())
                _active_pct = 100.0 * _n_active / max(1, _n)
                _sat_pct = 100.0 * _n_sat / max(1, _n)

                print(f"   Frames total:         {_n}")
                print(f"   Term active (w>0.01): {_n_active} ({_active_pct:.1f}%)")
                print(f"   Term saturated (w=1): {_n_sat} ({_sat_pct:.1f}%)")

                if _term is not None and _n_active > 0:
                    _term_active = np.abs(_term[_active])
                    print(
                        f"   |term_applied| rad    P50={np.median(_term_active):.4f} "
                        f"P95={np.percentile(_term_active, 95):.4f} "
                        f"max={np.max(_term_active):.4f}"
                    )

                if _source_raw is not None and _n_active > 0:
                    # Fixed-length byte strings in HDF5; strip null and whitespace padding.
                    _src_strs = [
                        bytes(s).rstrip(b"\x00").rstrip().decode("ascii", errors="ignore")
                        for s in _source_raw[:_n]
                    ]
                    _src_active = [s for s, a in zip(_src_strs, _active) if a]
                    _at_car = sum(1 for s in _src_active if s == "at_car")
                    _look = sum(1 for s in _src_active if s == "lookahead")
                    print(f"   e_lat source (active): at_car={_at_car}  lookahead={_look}")
                    if _at_car + _look > 0:
                        _at_car_pct = 100.0 * _at_car / (_at_car + _look)
                        _hint = ""
                        if _at_car_pct > 20.0:
                            _hint = (
                                "  ← at-car dominance: check lookahead/at-car projection on curves "
                                "(see project_proportional_curve_anticipation.md)"
                            )
                        print(f"   at_car dominance:     {_at_car_pct:.1f}%{_hint}")

                if _shadow is not None and _n > 0:
                    _shadow_frames = int((_shadow > 0).sum())
                    if _shadow_frames > 0:
                        print(
                            f"   Shadow mode:          {_shadow_frames}/{_n} frames "
                            f"({100.0 * _shadow_frames / _n:.1f}%) — term computed but NOT applied"
                        )
                    else:
                        print(f"   Shadow mode:          0 — term applied live")
                print()
    except Exception as _exc:
        print("27. RECOVERY TERM ACTIVITY")
        print("-" * 80)
        print(f"   (skipped — {_exc})")
        print()

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive overall drive analysis")
    parser.add_argument("recording", nargs="?", help="Path to recording file")
    parser.add_argument("--latest", action="store_true", help="Analyze latest recording")
    parser.add_argument("--list", action="store_true", help="List available recordings")
    stop_group = parser.add_mutually_exclusive_group()
    stop_group.add_argument("--stop-on-emergency", action="store_true", help="Stop analysis at emergency stop")
    stop_group.add_argument("--no-stop-on-emergency", action="store_true", help="Analyze full run even after emergency stop")
    parser.add_argument(
        "--analyze-to-failure",
        action="store_true",
        help="Truncate analysis at first sustained failure (canonical summary mode)",
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_recordings()
        return
    
    if args.latest:
        recordings_dir = Path("data/recordings")
        if not recordings_dir.exists():
            print("No recordings directory found")
            return
        recordings = sorted(recordings_dir.glob("*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not recordings:
            print("No recordings found")
            return
        recording_path = recordings[0]
    elif args.recording:
        recording_path = Path(args.recording)
    else:
        print("Error: Please provide a recording file or use --latest")
        parser.print_help()
        return
    
    if not recording_path.exists():
        print(f"Error: Recording file not found: {recording_path}")
        return
    
    # Default to failure-truncated analysis unless explicitly disabled.
    analyze_to_failure = True
    if args.analyze_to_failure or args.stop_on_emergency:
        analyze_to_failure = True
    if args.no_stop_on_emergency:
        analyze_to_failure = False

    summary = analyze_recording_summary(recording_path, analyze_to_failure=analyze_to_failure)
    _print_summary_report(recording_path, summary, analyze_to_failure=analyze_to_failure)


if __name__ == "__main__":
    main()
