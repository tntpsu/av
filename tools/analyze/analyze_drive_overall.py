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
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy import stats
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


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
    time_in_lane: float  # % within ±0.5m
    
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
    
    # Safety
    out_of_lane_events: int
    out_of_lane_time: float


class DriveAnalyzer:
    """Comprehensive drive analysis."""
    
    def __init__(self, recording_path: Path):
        """Initialize analyzer."""
        self.recording_path = recording_path
        self.data = {}
        self.metrics = None
        
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
                elif 'vehicle/timestamps' in f:
                    self.data['timestamps'] = np.array(f['vehicle/timestamps'][:])
                    self.data['position'] = np.array(f['vehicle/position'][:]) if 'vehicle/position' in f else None
                    self.data['speed'] = np.array(f['vehicle/speed'][:]) if 'vehicle/speed' in f else None
                else:
                    # Fallback to control timestamps
                    self.data['timestamps'] = np.array(f['control/timestamp'][:])
                
                # Control data
                self.data['steering'] = np.array(f['control/steering'][:])
                self.data['throttle'] = np.array(f['control/throttle'][:]) if 'control/throttle' in f else None
                self.data['brake'] = np.array(f['control/brake'][:]) if 'control/brake' in f else None
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
        n_frames = len(self.data['steering'])
        dt = np.mean(np.diff(self.data['time'])) if len(self.data['time']) > 1 else 0.033
        duration = self.data['time'][-1] if len(self.data['time']) > 0 else n_frames * dt
        
        # 1. PATH TRACKING METRICS
        lateral_error_rmse = np.sqrt(np.mean(self.data['lateral_error']**2)) if self.data['lateral_error'] is not None else 0.0
        lateral_error_mean = np.mean(np.abs(self.data['lateral_error'])) if self.data['lateral_error'] is not None else 0.0
        lateral_error_max = np.max(np.abs(self.data['lateral_error'])) if self.data['lateral_error'] is not None else 0.0
        lateral_error_std = np.std(self.data['lateral_error']) if self.data['lateral_error'] is not None else 0.0
        lateral_error_p50 = np.percentile(np.abs(self.data['lateral_error']), 50) if self.data['lateral_error'] is not None else 0.0
        lateral_error_p95 = np.percentile(np.abs(self.data['lateral_error']), 95) if self.data['lateral_error'] is not None else 0.0
        
        heading_error_rmse = np.sqrt(np.mean(self.data['heading_error']**2)) if self.data['heading_error'] is not None else 0.0
        heading_error_mean = np.mean(np.abs(self.data['heading_error'])) if self.data['heading_error'] is not None else 0.0
        heading_error_max = np.max(np.abs(self.data['heading_error'])) if self.data['heading_error'] is not None else 0.0
        
        # Time in lane (±0.5m)
        time_in_lane = np.sum(np.abs(self.data['lateral_error']) < 0.5) / n_frames * 100 if self.data['lateral_error'] is not None else 0.0
        
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
        control_effort = np.trapz(np.abs(self.data['steering']), self.data['time'])
        
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
        
        # 6. SAFETY METRICS
        # Out of lane events (lateral error > 1.0m, typical lane width is ~3.5m, so 1.0m is significant)
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
        print(f"   Time in Lane (±0.5m): {self.metrics.time_in_lane:.1f}%")
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
        
        # 4. PERCEPTION QUALITY
        print("4. PERCEPTION QUALITY")
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
        
        # 5. TRAJECTORY QUALITY
        print("5. TRAJECTORY QUALITY")
        print("-" * 80)
        print(f"   Trajectory Availability: {self.metrics.trajectory_availability:.1f}%")
        print(f"   Reference Point Accuracy (RMSE): {self.metrics.ref_point_accuracy_rmse:.4f} m")
        print(f"   Trajectory Smoothness: {self.metrics.trajectory_smoothness:.2f} (higher is better)")
        print(f"   Path Curvature Consistency: {self.metrics.path_curvature_consistency:.2f} (higher is better)")
        print()
        
        # 6. SYSTEM HEALTH
        print("6. SYSTEM HEALTH")
        print("-" * 80)
        print(f"   PID Integral Max: {self.metrics.pid_integral_max:.4f}")
        print(f"   PID Reset Frequency: {self.metrics.pid_reset_frequency:.2f} per second")
        print(f"   Error Conflict Rate: {self.metrics.error_conflict_rate:.1f}%")
        print(f"   Stale Command Rate: {self.metrics.stale_command_rate:.1f}%")
        print()
        
        # 7. SAFETY METRICS
        print("7. SAFETY METRICS")
        print("-" * 80)
        print(f"   Out-of-Lane Events: {self.metrics.out_of_lane_events}")
        print(f"   Out-of-Lane Time: {self.metrics.out_of_lane_time:.1f}%")
        print()
        
        # 8. RECOMMENDATIONS
        print("8. RECOMMENDATIONS")
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
        if self.metrics.error_conflict_rate > 20:
            recommendations.append("Reduce error conflicts - check heading/lateral error weighting")
        if self.metrics.pid_integral_max > 0.2:
            recommendations.append("Reduce PID integral accumulation - check integral reset mechanisms")
        
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
    
    analyzer = DriveAnalyzer(recording_path)
    analyzer.print_report()


if __name__ == "__main__":
    main()

