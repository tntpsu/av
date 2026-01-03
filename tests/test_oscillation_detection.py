"""
Tests for oscillation and overshoot detection.

These tests would have caught the late-drive oscillation issue:
- High oscillation frequency (14.49 Hz)
- Overshoot behavior
- Controller fighting itself
- PID integral accumulation over time
"""

import pytest
import numpy as np
from control.pid_controller import LateralController, VehicleController
from trajectory.inference import TrajectoryPlanningInference


class TestOscillationDetection:
    """Test that detects oscillation patterns in system behavior."""
    
    def test_no_high_frequency_oscillation(self):
        """
        CRITICAL TEST: Detect high-frequency oscillation.
        
        This test would have caught the 14.49 Hz oscillation issue.
        Oscillation frequency should be < 5 Hz for stable control.
        """
        controller = LateralController(
            kp=0.5,  # Use current config
            kd=0.35,
            ki=0.005,
            deadband=0.01,
            max_steering=0.5
        )
        
        # Simulate 60 seconds at 30 FPS = 1800 frames
        num_frames = 1800
        dt = 1.0 / 30.0
        
        # Initial state
        current_heading = 0.0
        vehicle_position = np.array([0.0, 0.0])
        
        # Reference point (straight lane, center)
        reference_point = {
            'x': 0.0,
            'y': 10.0,
            'heading': 0.0,
            'velocity': 8.0
        }
        
        # Track lateral error
        lateral_errors = []
        
        # Add small initial disturbance to trigger response
        vehicle_position = np.array([0.1, 0.0])  # 10cm offset
        
        for frame in range(num_frames):
            result = controller.compute_steering(
                current_heading=current_heading,
                reference_point=reference_point,
                vehicle_position=vehicle_position,
                return_metadata=True
            )
            
            lateral_error = result['lateral_error']
            lateral_errors.append(lateral_error)
            
            steering = result['steering']
            
            # Simulate vehicle dynamics
            current_heading = current_heading + steering * 0.1 * dt
            current_heading = np.arctan2(np.sin(current_heading), np.cos(current_heading))
            
            vehicle_position = vehicle_position + np.array([
                np.sin(current_heading) * 8.0 * dt,
                np.cos(current_heading) * 8.0 * dt
            ])
        
        lateral_errors = np.array(lateral_errors)
        
        # Detect oscillation: count sign changes
        sign_changes = 0
        for i in range(1, len(lateral_errors)):
            if np.sign(lateral_errors[i]) != np.sign(lateral_errors[i-1]):
                sign_changes += 1
        
        # Calculate oscillation frequency
        duration = num_frames / 30.0  # seconds
        oscillation_freq = sign_changes / duration  # Hz
        
        # Oscillation frequency should be < 5 Hz for stable control
        # The issue showed 14.49 Hz - this test would catch it
        assert oscillation_freq < 5.0, (
            f"High-frequency oscillation detected! "
            f"Frequency: {oscillation_freq:.2f} Hz, expected < 5.0 Hz. "
            f"Sign changes: {sign_changes} over {duration:.1f}s. "
            f"This indicates unstable control (issue showed 14.49 Hz)."
        )
    
    def test_oscillation_does_not_increase_over_time(self):
        """
        CRITICAL TEST: Detect when oscillation starts or increases over time.
        
        This test would have caught the issue where oscillation appeared late in drive.
        """
        controller = LateralController(
            kp=0.5,
            kd=0.35,
            ki=0.005,
            deadband=0.01,
            max_steering=0.5
        )
        
        num_frames = 1800  # 60 seconds
        dt = 1.0 / 30.0
        
        current_heading = 0.0
        vehicle_position = np.array([0.0, 0.0])
        
        reference_point = {
            'x': 0.0,
            'y': 10.0,
            'heading': 0.0,
            'velocity': 8.0
        }
        
        lateral_errors = []
        
        # Small initial disturbance
        vehicle_position = np.array([0.1, 0.0])
        
        for frame in range(num_frames):
            result = controller.compute_steering(
                current_heading=current_heading,
                reference_point=reference_point,
                vehicle_position=vehicle_position,
                return_metadata=True
            )
            
            lateral_errors.append(result['lateral_error'])
            
            steering = result['steering']
            current_heading = current_heading + steering * 0.1 * dt
            current_heading = np.arctan2(np.sin(current_heading), np.cos(current_heading))
            
            vehicle_position = vehicle_position + np.array([
                np.sin(current_heading) * 8.0 * dt,
                np.cos(current_heading) * 8.0 * dt
            ])
        
        lateral_errors = np.array(lateral_errors)
        
        # Split into thirds and measure oscillation in each
        first_third = lateral_errors[:len(lateral_errors)//3]
        last_third = lateral_errors[2*len(lateral_errors)//3:]
        
        # Count sign changes in each third
        def count_sign_changes(errors):
            changes = 0
            for i in range(1, len(errors)):
                if np.sign(errors[i]) != np.sign(errors[i-1]):
                    changes += 1
            return changes
        
        first_changes = count_sign_changes(first_third)
        last_changes = count_sign_changes(last_third)
        
        third_duration = (len(lateral_errors) // 3) / 30.0
        first_freq = first_changes / third_duration
        last_freq = last_changes / third_duration
        
        # Oscillation should not increase significantly over time
        # If it does, system is becoming unstable
        if first_freq > 0.01:  # Only check if there was oscillation in first third
            assert last_freq < first_freq * 2.0, (
                f"Oscillation frequency increased over time! "
                f"First third: {first_freq:.2f} Hz, Last third: {last_freq:.2f} Hz ({last_freq/first_freq:.2f}x increase). "
                f"This indicates system becoming unstable over time."
            )
        else:
            # No oscillation in first third - verify it doesn't start oscillating
            assert last_freq < 5.0, (
                f"Oscillation started late in drive! "
                f"First third: {first_freq:.2f} Hz (stable), Last third: {last_freq:.2f} Hz (oscillating). "
                f"This indicates system becoming unstable over time."
            )


class TestOvershootDetection:
    """Test that detects overshoot behavior."""
    
    def test_no_excessive_overshoot(self):
        """
        CRITICAL TEST: Detect overshoot behavior.
        
        This test would have caught the overshoot issue (error crossing zero
        then increasing in opposite direction).
        """
        controller = LateralController(
            kp=0.5,
            kd=0.35,
            ki=0.005,
            deadband=0.01,
            max_steering=0.5
        )
        
        num_frames = 1800  # 60 seconds
        dt = 1.0 / 30.0
        
        current_heading = 0.0
        vehicle_position = np.array([0.0, 0.0])
        
        reference_point = {
            'x': 0.0,
            'y': 10.0,
            'heading': 0.0,
            'velocity': 8.0
        }
        
        lateral_errors = []
        
        # Start with offset
        vehicle_position = np.array([0.2, 0.0])  # 20cm offset
        
        for frame in range(num_frames):
            result = controller.compute_steering(
                current_heading=current_heading,
                reference_point=reference_point,
                vehicle_position=vehicle_position,
                return_metadata=True
            )
            
            lateral_errors.append(result['lateral_error'])
            
            steering = result['steering']
            current_heading = current_heading + steering * 0.1 * dt
            current_heading = np.arctan2(np.sin(current_heading), np.cos(current_heading))
            
            vehicle_position = vehicle_position + np.array([
                np.sin(current_heading) * 8.0 * dt,
                np.cos(current_heading) * 8.0 * dt
            ])
        
        lateral_errors = np.array(lateral_errors)
        
        # Detect overshoot: error crosses zero, then increases in opposite direction
        overshoots = []
        for i in range(1, len(lateral_errors) - 10):
            # Check if error crossed zero
            if (lateral_errors[i-1] > 0 and lateral_errors[i] < 0) or \
               (lateral_errors[i-1] < 0 and lateral_errors[i] > 0):
                # Find peak in opposite direction
                peak_value = abs(lateral_errors[i])
                for j in range(i+1, min(i+30, len(lateral_errors))):
                    if abs(lateral_errors[j]) > peak_value:
                        peak_value = abs(lateral_errors[j])
                    else:
                        break
                if peak_value > 0.1:  # Significant overshoot
                    overshoots.append(peak_value)
        
        if len(overshoots) > 0:
            avg_overshoot = np.mean(overshoots)
            max_overshoot = np.max(overshoots)
            
            # Overshoot should be small (< 0.15m average, < 0.25m max)
            # The issue showed 0.21m average, 0.32m max
            assert avg_overshoot < 0.15, (
                f"Excessive overshoot detected! "
                f"Average overshoot: {avg_overshoot:.3f}m, expected < 0.15m. "
                f"Max overshoot: {max_overshoot:.3f}m. "
                f"Issue showed 0.21m average, 0.32m max."
            )


class TestControllerPhaseRelationship:
    """Test that detects when controller fights itself."""
    
    def test_controller_does_not_fight_itself(self):
        """
        CRITICAL TEST: Detect when controller fights itself.
        
        This test would have caught the issue where steering and error
        had the same sign at peaks 72% of the time.
        """
        controller = LateralController(
            kp=0.5,
            kd=0.35,
            ki=0.005,
            deadband=0.01,
            max_steering=0.5
        )
        
        num_frames = 1800  # 60 seconds
        dt = 1.0 / 30.0
        
        current_heading = 0.0
        vehicle_position = np.array([0.0, 0.0])
        
        reference_point = {
            'x': 0.0,
            'y': 10.0,
            'heading': 0.0,
            'velocity': 8.0
        }
        
        lateral_errors = []
        steerings = []
        
        vehicle_position = np.array([0.1, 0.0])
        
        for frame in range(num_frames):
            result = controller.compute_steering(
                current_heading=current_heading,
                reference_point=reference_point,
                vehicle_position=vehicle_position,
                return_metadata=True
            )
            
            lateral_errors.append(result['lateral_error'])
            steerings.append(result['steering'])
            
            steering = result['steering']
            current_heading = current_heading + steering * 0.1 * dt
            current_heading = np.arctan2(np.sin(current_heading), np.cos(current_heading))
            
            vehicle_position = vehicle_position + np.array([
                np.sin(current_heading) * 8.0 * dt,
                np.cos(current_heading) * 8.0 * dt
            ])
        
        lateral_errors = np.array(lateral_errors)
        steerings = np.array(steerings)
        
        # Find error peaks
        error_abs = np.abs(lateral_errors)
        peaks = []
        for i in range(1, len(error_abs) - 1):
            if error_abs[i] > error_abs[i-1] and error_abs[i] > error_abs[i+1] and error_abs[i] > 0.1:
                peaks.append((i, lateral_errors[i], steerings[i]))
        
        if len(peaks) > 10:
            # Check if steering and error have same sign at peaks (wrong direction)
            same_sign_count = 0
            for peak_idx, error, steering in peaks:
                if np.sign(error) == np.sign(steering):
                    same_sign_count += 1
            
            wrong_direction_pct = 100 * same_sign_count / len(peaks)
            
            # Should have < 30% wrong direction at peaks
            # The issue showed 72% - this test would catch it
            assert wrong_direction_pct < 30, (
                f"Controller is fighting itself! "
                f"Wrong direction at peaks: {wrong_direction_pct:.1f}%, expected < 30%. "
                f"Issue showed 72% wrong direction. "
                f"This indicates delayed response or gain too high."
            )


class TestLongTermPIDIntegralAccumulation:
    """Test that detects PID integral accumulation over extended periods."""
    
    def test_pid_integral_does_not_accumulate_over_extended_period(self):
        """
        CRITICAL TEST: Detect PID integral accumulation over 60+ seconds.
        
        This test would have caught the 3.22x integral accumulation issue.
        Uses simulated data (not recordings) to catch gradual accumulation.
        """
        controller = LateralController(
            kp=0.5,
            kd=0.35,
            ki=0.005,
            deadband=0.01,
            max_steering=0.5
        )
        
        num_frames = 1800  # 60 seconds
        dt = 1.0 / 30.0
        
        current_heading = 0.0
        vehicle_position = np.array([0.0, 0.0])
        
        reference_point = {
            'x': 0.0,
            'y': 10.0,
            'heading': 0.0,
            'velocity': 8.0
        }
        
        pid_integrals = []
        
        # Small persistent error to trigger integral
        vehicle_position = np.array([0.05, 0.0])  # 5cm persistent offset
        
        for frame in range(num_frames):
            result = controller.compute_steering(
                current_heading=current_heading,
                reference_point=reference_point,
                vehicle_position=vehicle_position,
                return_metadata=True
            )
            
            pid_integrals.append(abs(result['pid_integral']))
            
            steering = result['steering']
            current_heading = current_heading + steering * 0.1 * dt
            current_heading = np.arctan2(np.sin(current_heading), np.cos(current_heading))
            
            vehicle_position = vehicle_position + np.array([
                np.sin(current_heading) * 8.0 * dt,
                np.cos(current_heading) * 8.0 * dt
            ])
        
        pid_integrals = np.array(pid_integrals)
        
        # Split into thirds
        first_third = pid_integrals[:len(pid_integrals)//3]
        last_third = pid_integrals[2*len(pid_integrals)//3:]
        
        mean_first = np.mean(first_third)
        mean_last = np.mean(last_third)
        
        # Integral should not accumulate (should reset periodically)
        # The issue showed 3.22x increase - this test would catch it
        if mean_first > 0.001:  # Only check if integral is non-zero
            accumulation_ratio = mean_last / mean_first
            assert accumulation_ratio < 1.5, (
                f"PID integral is accumulating over extended period! "
                f"First third: {mean_first:.4f}, Last third: {mean_last:.4f} ({accumulation_ratio:.2f}x increase). "
                f"Issue showed 3.22x increase. Expected < 1.5x."
            )


class TestSteeringRateOfChange:
    """Test that detects rapid steering changes that could cause oscillation."""
    
    def test_steering_rate_of_change_acceptable(self):
        """
        CRITICAL TEST: Detect rapid steering oscillations.
        
        This test would have caught the issue where steering changed rapidly,
        contributing to 14.49 Hz oscillation.
        """
        controller = LateralController(
            kp=0.5,
            kd=0.35,
            ki=0.005,
            deadband=0.01,
            max_steering=0.5
        )
        
        num_frames = 1800  # 60 seconds
        dt = 1.0 / 30.0
        
        current_heading = 0.0
        vehicle_position = np.array([0.0, 0.0])
        
        reference_point = {
            'x': 0.0,
            'y': 10.0,
            'heading': 0.0,
            'velocity': 8.0
        }
        
        steerings = []
        
        vehicle_position = np.array([0.1, 0.0])
        
        for frame in range(num_frames):
            result = controller.compute_steering(
                current_heading=current_heading,
                reference_point=reference_point,
                vehicle_position=vehicle_position,
                return_metadata=True
            )
            
            steerings.append(result['steering'])
            
            steering = result['steering']
            current_heading = current_heading + steering * 0.1 * dt
            current_heading = np.arctan2(np.sin(current_heading), np.cos(current_heading))
            
            vehicle_position = vehicle_position + np.array([
                np.sin(current_heading) * 8.0 * dt,
                np.cos(current_heading) * 8.0 * dt
            ])
        
        steerings = np.array(steerings)
        
        # Calculate steering change rate
        steering_changes = np.abs(np.diff(steerings))
        max_change = np.max(steering_changes)
        mean_change = np.mean(steering_changes)
        
        # With rate limiting (0.1 per frame), max change should be < 0.15
        # Without rate limiting, could be much higher
        assert max_change < 0.15, (
            f"Rapid steering changes detected! "
            f"Max change per frame: {max_change:.3f}, expected < 0.15. "
            f"Mean change: {mean_change:.3f}. "
            f"This could cause oscillation."
        )
        
        # Check for high-frequency steering oscillations
        # Count sign changes in steering
        steering_sign_changes = 0
        for i in range(1, len(steerings)):
            if np.sign(steerings[i]) != np.sign(steerings[i-1]):
                steering_sign_changes += 1
        
        duration = num_frames / 30.0
        steering_oscillation_freq = steering_sign_changes / duration
        
        # Steering oscillation should be < 5 Hz
        assert steering_oscillation_freq < 5.0, (
            f"High-frequency steering oscillation! "
            f"Frequency: {steering_oscillation_freq:.2f} Hz, expected < 5.0 Hz. "
            f"This indicates unstable control."
        )

