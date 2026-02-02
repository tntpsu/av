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
        steering_rate = np.diff(self.data['steering']) / np.diff(self.data['time'])
        steering_jerk = np.diff(steering_rate) / np.diff(self.data['time'][1:]) if len(steering_rate) > 1 else np.array([0.0])
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
            fft_vals = fft(error_centered)
            fft_freqs = fftfreq(len(error_centered), dt)
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
        traj_cfg = self.config.get('trajectory', {})
        curvature_smoothing_enabled = bool(traj_cfg.get('curvature_smoothing_enabled', False))
        curvature_window_m = float(traj_cfg.get('curvature_smoothing_window_m', 12.0))
        curvature_min_speed = float(traj_cfg.get('curvature_smoothing_min_speed', 2.0))
        if self.data['speed'] is not None and len(self.data['speed']) > 1 and len(self.data['time']) > 1:
            dt_series = np.diff(self.data['time'])
            dt_series[dt_series <= 0] = dt
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
                    lat_dt[lat_dt <= 0] = dt
                    lat_jerk = np.diff(lat_accel) / lat_dt
                    if lat_jerk.size > 0:
                        abs_lat_jerk = np.abs(lat_jerk)
                        lateral_jerk_p95 = float(np.percentile(abs_lat_jerk, 95))
                        lateral_jerk_max = float(np.max(abs_lat_jerk))
        if self.data.get('speed_limit') is not None and len(self.data['speed_limit']) > 0:
            speed_limit_zero_rate = float(
                np.sum(self.data['speed_limit'] <= 0.01) / len(self.data['speed_limit']) * 100
            )
        
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
        print()
        print(f"   Speed Limit Missing: {self.metrics.speed_limit_zero_rate:.1f}%")
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


def main():
    parser = argparse.ArgumentParser(description="Comprehensive overall drive analysis")
    parser.add_argument("recording", nargs="?", help="Path to recording file")
    parser.add_argument("--latest", action="store_true", help="Analyze latest recording")
    parser.add_argument("--list", action="store_true", help="List available recordings")
    stop_group = parser.add_mutually_exclusive_group()
    stop_group.add_argument("--stop-on-emergency", action="store_true", help="Stop analysis at emergency stop")
    stop_group.add_argument("--no-stop-on-emergency", action="store_true", help="Analyze full run even after emergency stop")
    
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
    
    stop_on_emergency = True
    if args.no_stop_on_emergency:
        stop_on_emergency = False
    elif args.stop_on_emergency:
        stop_on_emergency = True
    analyzer = DriveAnalyzer(recording_path, stop_on_emergency=stop_on_emergency)
    analyzer.print_report()


if __name__ == "__main__":
    main()

