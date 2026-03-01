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
    print(
        "   Oscillation Amplitude Runaway: "
        f"{'YES' if control_smoothness.get('oscillation_amplitude_runaway') else 'NO'}"
    )
    if curve_intent_diag.get("available"):
        print(
            "   Curve Intent Arm Early Rate: "
            f"{curve_intent_diag.get('arm_early_enough_rate', 0.0):.1f}%"
        )
        print(
            "   Curve Intent Undercall Rate: "
            f"{curve_intent_diag.get('undercall_frame_rate', 0.0):.1f}%"
        )
        print(
            "   Curve Intent Curvature Ratio P50/P95: "
            f"{curve_intent_diag.get('curvature_ratio_p50', 0.0):.2f} / "
            f"{curve_intent_diag.get('curvature_ratio_p95', 0.0):.2f}"
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

    print("10. CHASSIS-GROUND HEALTH")
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
    print("11. CURVATURE CONTRACT HEALTH")
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
    print()

    print("12. SAFETY METRICS")
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

    print("13. RECOMMENDATIONS")
    print("-" * 80)
    if recommendations:
        for idx, recommendation in enumerate(recommendations, 1):
            print(f"   {idx}. {recommendation}")
    else:
        print("   ✓ No recommendations")
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
