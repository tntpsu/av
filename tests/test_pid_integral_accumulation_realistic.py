"""
Realistic PID Integral Accumulation Tests

These tests simulate REAL Unity conditions:
- Vehicle dynamics (position changes with steering)
- Heading changes
- Varying errors (not constant)
- Trajectory smoothing delays

This should catch issues that simple tests miss.
"""

import pytest
import numpy as np
from control.pid_controller import LateralController


class TestPIDIntegralAccumulationRealistic:
    """Test PID integral accumulation with realistic vehicle dynamics."""
    
    def test_integral_does_not_accumulate_with_vehicle_dynamics(self):
        """
        CRITICAL TEST: Verify PID integral does not accumulate with vehicle dynamics.
        
        This test simulates REAL Unity conditions:
        - Vehicle position changes with steering (feedback loop)
        - Heading changes with steering
        - Varying errors (not constant) - KEY DIFFERENCE from simple test
        - Trajectory smoothing delays (alpha=0.80)
        - Continuous noise (like lane detection)
        
        This should catch the 13.19x accumulation issue that simple tests miss.
        """
        # Use current config that failed in Unity
        controller = LateralController(kp=0.4, ki=0.003, kd=0.5, deadband=0.01)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Simulate 6 seconds (180 frames at 30 FPS)
        dt = 0.033
        num_frames = 180
        
        # Initial state (like Unity)
        vehicle_x = 0.1  # Start 10cm offset
        vehicle_y = 0.0
        vehicle_heading = 0.0
        vehicle_speed = 8.0  # m/s
        
        # Reference point (straight lane, center)
        # In Unity, this would come from trajectory planning
        actual_ref_x = 0.0  # Lane center (what trajectory planner wants)
        smoothed_ref_x = 0.0  # Smoothed reference (what controller sees)
        smoothing_alpha = 0.80  # Unity smoothing factor
        reference_heading = 0.0
        
        # Add persistent bias (like trajectory offset in Unity: 0.3258m)
        # Unity has persistent bias that creates accumulation
        persistent_bias = 0.05  # 5cm persistent bias (simulates trajectory offset)
        
        integrals = []
        lateral_errors = []
        
        for i in range(num_frames):
            # Add noise/variation (like lane detection noise in Unity)
            # Unity data shows: 15% of frames have > 0.10m error, max 0.65m
            # 14.1% of frames have > 10cm error changes
            # Need to match this distribution to reproduce the issue
            
            # Base noise (continuous, matches Unity std=0.0922m)
            noise = np.random.normal(0, 0.09)  # 9cm base noise (matches Unity std)
            
            # Large spikes (like lane detection failures or trajectory jumps)
            # Unity: 15% of frames have > 0.10m error
            if np.random.random() < 0.15:  # 15% chance of large spike
                spike = np.random.normal(0, 0.20)  # 20cm spike (can be up to 0.6m with 3-sigma)
                noise += spike
            
            # Very large spikes (occasional, like trajectory failures)
            # Unity: max error change is 0.65m
            if np.random.random() < 0.03:  # 3% chance of very large spike
                large_spike = np.random.normal(0, 0.25)  # 25cm spike (can be up to 0.75m)
                noise += large_spike
            
            actual_ref_x += noise
            actual_ref_x = np.clip(actual_ref_x, -0.7, 0.7)  # Allow up to 0.7m variation (like Unity max 0.65m)
            
            # Add persistent bias (like trajectory offset in Unity)
            # This creates consistent error that accumulates
            actual_ref_x += persistent_bias
            
            # Apply trajectory smoothing delay (like Unity alpha=0.80)
            # This creates lag between actual error and controller seeing it
            smoothed_ref_x = smoothing_alpha * smoothed_ref_x + (1 - smoothing_alpha) * actual_ref_x
            
            # Use smoothed reference (delayed, like Unity)
            reference_x = smoothed_ref_x
            
            # Create reference point (like Unity does)
            reference_point = {
                'x': reference_x - vehicle_x,  # Relative to vehicle
                'y': 10.0,
                'heading': reference_heading - vehicle_heading,  # Relative heading
                'velocity': vehicle_speed
            }
            
            # Compute steering
            result = controller.compute_steering(
                current_heading=vehicle_heading,
                reference_point=reference_point,
                vehicle_position=np.array([vehicle_x, vehicle_y]),
                return_metadata=True
            )
            
            steering = result['steering']
            integrals.append(abs(result['pid_integral']))
            lateral_errors.append(result['lateral_error'])
            
            # Simulate vehicle dynamics (like Unity)
            # Steering affects heading, which affects position
            vehicle_heading += steering * 0.1 * dt  # Simplified steering->heading
            vehicle_heading = np.arctan2(np.sin(vehicle_heading), np.cos(vehicle_heading))
            
            # Vehicle moves forward and laterally
            vehicle_x += np.sin(vehicle_heading) * vehicle_speed * dt
            vehicle_y += np.cos(vehicle_heading) * vehicle_speed * dt
        
        # Split into thirds
        first_third = integrals[:num_frames//3]
        last_third = integrals[2*num_frames//3:]
        
        mean_first = np.mean(first_third)
        mean_last = np.mean(last_third)
        accumulation = mean_last / mean_first if mean_first > 0 else 0
        
        # Should be < 1.5x (same threshold as simple test)
        assert accumulation < 1.5, (
            f"PID integral accumulating with vehicle dynamics! "
            f"First third: {mean_first:.4f}, Last third: {mean_last:.4f} "
            f"({accumulation:.2f}x increase). "
            f"Expected < 1.5x. "
            f"This test should catch the 13.19x accumulation issue."
        )
    
    def test_integral_with_trajectory_smoothing_delay(self):
        """
        Test integral accumulation with trajectory smoothing delays.
        
        In Unity, trajectory smoothing (alpha=0.80) creates delays.
        This can cause integral to accumulate during the delay.
        """
        controller = LateralController(kp=0.4, ki=0.003, kd=0.5, deadband=0.01)
        
        dt = 0.033
        num_frames = 180
        
        # Simulate trajectory smoothing (alpha=0.80)
        # Smoothed reference point lags behind actual reference
        actual_ref_x = 0.0
        smoothed_ref_x = 0.0
        smoothing_alpha = 0.80
        
        vehicle_x = 0.1
        vehicle_heading = 0.0
        
        integrals = []
        
        for i in range(num_frames):
            # Update actual reference (what trajectory planner wants)
            if i % 30 == 0:  # Change every second
                actual_ref_x = 0.05 * np.sin(i * 0.1)  # Varying reference
            
            # Apply smoothing (like Unity does)
            smoothed_ref_x = smoothing_alpha * smoothed_ref_x + (1 - smoothing_alpha) * actual_ref_x
            
            # Use smoothed reference (delayed)
            reference_point = {
                'x': smoothed_ref_x - vehicle_x,
                'y': 10.0,
                'heading': 0.0,
                'velocity': 8.0
            }
            
            result = controller.compute_steering(
                current_heading=vehicle_heading,
                reference_point=reference_point,
                return_metadata=True
            )
            
            integrals.append(abs(result['pid_integral']))
            
            # Update vehicle
            steering = result['steering']
            vehicle_heading += steering * 0.1 * dt
            vehicle_x += np.sin(vehicle_heading) * 8.0 * dt
        
        # Check accumulation
        first_third = integrals[:num_frames//3]
        last_third = integrals[2*num_frames//3:]
        
        mean_first = np.mean(first_third)
        mean_last = np.mean(last_third)
        accumulation = mean_last / mean_first if mean_first > 0 else 0
        
        assert accumulation < 1.5, (
            f"PID integral accumulating with smoothing delays! "
            f"Accumulation: {accumulation:.2f}x. "
            f"This should catch smoothing-related accumulation."
        )

