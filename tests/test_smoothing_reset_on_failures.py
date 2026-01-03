"""
Tests for reference point smoothing reset during lane failures.

These tests would have caught the bug:
- Reference point smoothing preserving bias during lane failures
"""

import pytest
import numpy as np
from trajectory.inference import TrajectoryPlanningInference
from trajectory.models.trajectory_planner import Trajectory, TrajectoryPoint


class TestSmoothingResetOnLaneFailures:
    """Test that smoothing resets during lane failures."""
    
    def test_smoothing_resets_during_lane_failures(self):
        """
        CRITICAL TEST: Verify smoothing resets when lane detection fails.
        
        Bug: When lane detection failed, trajectory planner correctly generated
        a straight trajectory (x=0.0). However, reference point smoothing (alpha=0.85)
        preserved a significant portion of the previous biased ref_x (e.g., 0.1m right),
        leading to a smoothed ref_x that was still off-center (e.g., 0.2125m right),
        which the controller then followed, causing the veer.
        
        This test would have caught the bug.
        """
        inference = TrajectoryPlanningInference(
            planner_type="rule_based",
            reference_smoothing=0.85,  # High smoothing
            image_width=640.0,
            image_height=480.0
        )
        
        # Simulate previous frame with bias (ref_x = 0.1m right)
        previous_ref_point = {
            'x': 0.1,  # 0.1m right (biased)
            'y': 8.0,
            'heading': 0.0,
            'velocity': 8.0,
            'curvature': 0.0
        }
        inference.last_smoothed_ref_point = previous_ref_point.copy()
        
        # Current frame: Lane detection fails, trajectory planner returns straight trajectory
        # Straight trajectory should have ref_x = 0.0 (centered)
        fallback_trajectory = Trajectory(
            points=[
                TrajectoryPoint(x=0.0, y=0.0, heading=0.0, curvature=0.0, velocity=8.0),
                TrajectoryPoint(x=0.0, y=8.0, heading=0.0, curvature=0.0, velocity=8.0),
            ],
            length=8.0
        )
        
        # Get reference point (should detect fallback and reset smoothing)
        ref_point = inference.get_reference_point(fallback_trajectory, lookahead=8.0)
        
        # When fallback trajectory is detected, smoothing should reset
        # ref_x should be 0.0 (centered), not 0.085 (0.85 * 0.1 + 0.15 * 0.0)
        assert abs(ref_point['x']) < 0.01, (
            f"Smoothing should reset during lane failures. "
            f"ref_x: {ref_point['x']:.4f}m, expected ~0.0m. "
            f"If smoothing preserved bias, ref_x would be ~0.085m"
        )
        
        # Verify last_smoothed_ref_point was reset (or not used)
        # After fallback, last_smoothed_ref_point should be None or reset
        if inference.last_smoothed_ref_point is not None:
            # If it's not None, it should be the centered trajectory (0.0)
            assert abs(inference.last_smoothed_ref_point['x']) < 0.01, (
                f"last_smoothed_ref_point should be reset after fallback. "
                f"x: {inference.last_smoothed_ref_point['x']:.4f}m, expected ~0.0m"
            )
    
    def test_fallback_trajectory_detection(self):
        """
        Test that fallback trajectories are correctly detected.
        
        A fallback trajectory is one where:
        - ref_x = 0.0 (centered)
        - ref_heading = 0.0 (straight)
        - Previous ref_x had bias (not centered)
        """
        inference = TrajectoryPlanningInference(
            planner_type="rule_based",
            reference_smoothing=0.85,
            image_width=640.0,
            image_height=480.0
        )
        
        # Previous frame with bias
        inference.last_smoothed_ref_point = {
            'x': 0.1,
            'y': 8.0,
            'heading': 0.0,
            'velocity': 8.0,
            'curvature': 0.0
        }
        
        # Fallback trajectory (straight, centered)
        fallback_trajectory = Trajectory(
            points=[
                TrajectoryPoint(x=0.0, y=0.0, heading=0.0, curvature=0.0, velocity=8.0),
                TrajectoryPoint(x=0.0, y=8.0, heading=0.0, curvature=0.0, velocity=8.0),
            ],
            length=8.0
        )
        
        # Get reference point
        ref_point = inference.get_reference_point(fallback_trajectory, lookahead=8.0)
        
        # Verify fallback was detected (ref_x should be 0.0, not smoothed)
        # If fallback was NOT detected, ref_x would be ~0.085 (0.85 * 0.1 + 0.15 * 0.0)
        # If fallback WAS detected, ref_x should be ~0.0
        assert abs(ref_point['x']) < 0.05, (
            f"Fallback trajectory should be detected and smoothing reset. "
            f"ref_x: {ref_point['x']:.4f}m, expected < 0.05m. "
            f"If fallback was not detected, ref_x would be ~0.085m"
        )
    
    def test_smoothing_preserves_bias_when_not_fallback(self):
        """
        Test that smoothing works normally when NOT a fallback trajectory.
        
        This ensures we didn't break normal smoothing behavior.
        """
        inference = TrajectoryPlanningInference(
            planner_type="rule_based",
            reference_smoothing=0.85,
            image_width=640.0,
            image_height=480.0
        )
        
        # Previous frame
        inference.last_smoothed_ref_point = {
            'x': 0.1,
            'y': 8.0,
            'heading': 0.0,
            'velocity': 8.0,
            'curvature': 0.0
        }
        
        # Normal trajectory (not a fallback - has slight offset)
        normal_trajectory = Trajectory(
            points=[
                TrajectoryPoint(x=0.05, y=0.0, heading=0.0, curvature=0.0, velocity=8.0),
                TrajectoryPoint(x=0.05, y=8.0, heading=0.0, curvature=0.0, velocity=8.0),
            ],
            length=8.0
        )
        
        # Get reference point
        ref_point = inference.get_reference_point(normal_trajectory, lookahead=8.0)
        
        # Normal smoothing should apply (0.85 * 0.1 + 0.15 * 0.05 = 0.0925)
        # Allow some tolerance
        expected_smoothed = 0.85 * 0.1 + 0.15 * 0.05  # 0.0925
        assert abs(ref_point['x'] - expected_smoothed) < 0.02, (
            f"Normal smoothing should work. "
            f"ref_x: {ref_point['x']:.4f}m, expected ~{expected_smoothed:.4f}m"
        )

