"""
Tests for PID integral accumulation prevention.

These tests verify that PID integral does not accumulate indefinitely,
which causes persistent steering bias and gradual drift.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from control.pid_controller import LateralController


class TestPIDIntegralAccumulation:
    """Test that PID integral does not accumulate over time."""
    
    def test_integral_does_not_accumulate_over_6_seconds(self):
        """
        CRITICAL TEST: Verify PID integral does not accumulate over 6 seconds.
        
        This test simulates 6 seconds of operation (180 frames at 30 FPS)
        and verifies that integral does not increase significantly.
        
        Current issue: 11.63x increase after 6s - this test should catch it.
        """
        controller = LateralController(kp=0.8, ki=0.01, kd=0.25, deadband=0.0)
        
        # Simulate 6 seconds of operation (180 frames at 30 FPS)
        dt = 0.033  # 30 FPS
        num_frames = 180  # 6 seconds
        
        # Simulate consistent small error (like persistent bias)
        reference_point = {'x': -0.05, 'y': 10.0, 'heading': 0.0, 'velocity': 10.0}
        
        integrals = []
        for i in range(num_frames):
            result = controller.compute_steering(
                current_heading=0.0,
                reference_point=reference_point,
                return_metadata=True
            )
            integrals.append(abs(result['pid_integral']))
        
        # Split into thirds
        first_third = integrals[:num_frames//3]
        last_third = integrals[2*num_frames//3:]
        
        mean_first = np.mean(first_third)
        mean_last = np.mean(last_third)
        
        # Integral should not accumulate (should reset periodically)
        # Allow only 1.5x increase (stricter than real data test)
        assert mean_last < mean_first * 1.5, (
            f"PID integral accumulating over 6 seconds! "
            f"First third: {mean_first:.4f}, Last third: {mean_last:.4f} "
            f"({mean_last/mean_first:.2f}x increase). "
            f"Expected < 1.5x, got {mean_last/mean_first:.2f}x. "
            f"This causes persistent steering bias."
        )
    
    def test_integral_resets_on_small_error(self):
        """Verify integral resets when error is small for extended period."""
        controller = LateralController(kp=0.8, ki=0.01, kd=0.25, deadband=0.0)
        
        # Build up integral with small error
        reference_point = {'x': -0.05, 'y': 10.0, 'heading': 0.0, 'velocity': 10.0}
        for _ in range(20):
            controller.compute_steering(
                current_heading=0.0,
                reference_point=reference_point,
                return_metadata=False
            )
        
        # Check integral before reset
        result_before = controller.compute_steering(
            current_heading=0.0,
            reference_point=reference_point,
            return_metadata=True
        )
        integral_before = abs(result_before['pid_integral'])
        
        # Create small error for extended period (should trigger reset)
        small_ref = {'x': -0.01, 'y': 10.0, 'heading': 0.0, 'velocity': 10.0}
        for _ in range(10):  # More than 6 frames (reset threshold)
            controller.compute_steering(
                current_heading=0.0,
                reference_point=small_ref,
                return_metadata=False
            )
        
        # Check integral after reset
        result_after = controller.compute_steering(
            current_heading=0.0,
            reference_point=small_ref,
            return_metadata=True
        )
        integral_after = abs(result_after['pid_integral'])
        
        # Integral should be reset (reduced)
        assert integral_after < integral_before, (
            f"Integral should reset on small error. "
            f"Before: {integral_before:.4f}, After: {integral_after:.4f}"
        )
    
    def test_integral_resets_on_sign_change(self):
        """Verify integral resets when error changes sign."""
        controller = LateralController(kp=0.8, ki=0.01, kd=0.25, deadband=0.0)
        
        # Build up integral with positive error
        ref_positive = {'x': -0.1, 'y': 10.0, 'heading': 0.0, 'velocity': 10.0}
        for _ in range(10):
            controller.compute_steering(
                current_heading=0.0,
                reference_point=ref_positive,
                return_metadata=False
            )
        
        result_before = controller.compute_steering(
            current_heading=0.0,
            reference_point=ref_positive,
            return_metadata=True
        )
        integral_before = abs(result_before['pid_integral'])
        
        # Change to negative error (sign change should trigger reset)
        ref_negative = {'x': 0.1, 'y': 10.0, 'heading': 0.0, 'velocity': 10.0}
        result_after = controller.compute_steering(
            current_heading=0.0,
            reference_point=ref_negative,
            return_metadata=True
        )
        integral_after = abs(result_after['pid_integral'])
        
        # Integral should be reset (reduced)
        assert integral_after < integral_before, (
            f"Integral should reset on sign change. "
            f"Before: {integral_before:.4f}, After: {integral_after:.4f}"
        )
    
    def test_integral_periodic_reset(self):
        """Verify periodic reset prevents long-term accumulation."""
        controller = LateralController(kp=0.8, ki=0.01, kd=0.25, deadband=0.0)
        
        # Simulate 4 seconds (120 frames) with consistent error
        reference_point = {'x': -0.05, 'y': 10.0, 'heading': 0.0, 'velocity': 10.0}
        
        max_integral = 0.0
        for i in range(120):
            result = controller.compute_steering(
                current_heading=0.0,
                reference_point=reference_point,
                return_metadata=True
            )
            max_integral = max(max_integral, abs(result['pid_integral']))
        
        # Integral should not exceed limit (should reset periodically)
        # With new limit of 0.10 and decay, should stay well below
        assert max_integral < 0.15, (
            f"Integral should not accumulate indefinitely. "
            f"Max: {max_integral:.4f}, Expected < 0.15 (with periodic reset and decay)"
        )
    
    def test_integral_decay_on_large_accumulation(self):
        """Verify integral decay prevents accumulation when integral is large."""
        controller = LateralController(kp=0.8, ki=0.01, kd=0.25, deadband=0.0)
        
        # Force integral to accumulate
        reference_point = {'x': -0.1, 'y': 10.0, 'heading': 0.0, 'velocity': 10.0}
        
        # Build up integral
        for _ in range(30):
            controller.compute_steering(
                current_heading=0.0,
                reference_point=reference_point,
                return_metadata=False
            )
        
        # Check integral
        result = controller.compute_steering(
            current_heading=0.0,
            reference_point=reference_point,
            return_metadata=True
        )
        integral = abs(result['pid_integral'])
        
        # If integral is large (> 0.05), decay should apply
        # Continue for a few more frames to see decay effect
        if integral > 0.05:
            integrals_after = []
            for _ in range(10):
                result = controller.compute_steering(
                    current_heading=0.0,
                    reference_point=reference_point,
                    return_metadata=True
                )
                integrals_after.append(abs(result['pid_integral']))
            
            # Integral should decay (not continue to grow)
            # With 2% decay per frame, should see reduction
            mean_after = np.mean(integrals_after)
            # Allow some growth from error, but decay should limit it
            assert mean_after < integral * 1.2, (
                f"Integral should decay when large. "
                f"Before decay: {integral:.4f}, After: {mean_after:.4f}. "
                f"Expected decay to limit growth."
            )

