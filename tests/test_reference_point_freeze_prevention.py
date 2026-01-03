"""
Test to prevent reference point freezing that causes steering lock.

This test specifically addresses the issue where:
1. Reference point freezes at a constant value (e.g., 0.338m)
2. Lateral error freezes (same as reference point)
3. Steering locks at a constant value (e.g., 0.443)
4. Car drifts off road

Root cause: Jump detection threshold (0.2m) was too strict for lane_positions,
rejecting valid updates when perception center changed (e.g., 0.338 → -0.056 = 0.394m change).
"""

import pytest
import numpy as np
from trajectory.inference import TrajectoryPlanningInference
from trajectory.models.trajectory_planner import Trajectory, TrajectoryPoint


class TestReferencePointFreezePrevention:
    """Test that reference point updates when perception changes."""
    
    def test_reference_point_updates_with_perception_changes(self):
        """
        CRITICAL TEST: Verify reference point updates when perception center changes.
        
        This prevents the steering lock bug where:
        - Perception center changes (0.338 → -0.056 → -0.040)
        - Reference point freezes at 0.338m
        - Lateral error freezes
        - Steering locks
        """
        inference = TrajectoryPlanningInference(
            planner_type="rule_based",
            reference_smoothing=0.7
        )
        
        # Create a trajectory (not used when using lane_positions)
        trajectory = Trajectory(
            points=[TrajectoryPoint(x=0.0, y=10.0, heading=0.0, velocity=10.0, curvature=0.0)],
            length=10.0
        )
        
        # Simulate the exact scenario from the bug:
        # Frame 1: Perception center at 0.338m
        lane_positions_1 = {
            'left_lane_line_x': -3.238,
            'right_lane_line_x': 3.914
        }
        percept_center_1 = (lane_positions_1['left_lane_line_x'] + lane_positions_1['right_lane_line_x']) / 2.0
        assert abs(percept_center_1 - 0.338) < 0.01, "Perception center should be ~0.338m"
        
        ref_point_1 = inference.get_reference_point(
            trajectory=trajectory,
            lookahead=10.0,
            lane_positions=lane_positions_1,
            use_direct=True,
            timestamp=0.0
        )
        
        assert ref_point_1 is not None, "Reference point should be computed"
        ref_x_1 = ref_point_1['x']
        
        # Frame 2: Perception center changes to -0.056m (0.394m change)
        # This was the change that triggered the bug
        lane_positions_2 = {
            'left_lane_line_x': -4.026,
            'right_lane_line_x': 3.914
        }
        percept_center_2 = (lane_positions_2['left_lane_line_x'] + lane_positions_2['right_lane_line_x']) / 2.0
        assert abs(percept_center_2 - (-0.056)) < 0.01, "Perception center should be ~-0.056m"
        
        # Verify the change is > 0.2m (the old threshold)
        change = abs(percept_center_2 - percept_center_1)
        assert change > 0.2, f"Change should be > 0.2m to test jump detection: {change:.3f}m"
        
        ref_point_2 = inference.get_reference_point(
            trajectory=trajectory,
            lookahead=10.0,
            lane_positions=lane_positions_2,
            use_direct=True,
            timestamp=0.033  # 30 Hz
        )
        
        assert ref_point_2 is not None, "Reference point should be computed"
        ref_x_2 = ref_point_2['x']
        
        # CRITICAL: Reference point should update (not freeze)
        # With minimal_smoothing (alpha_x=0.02), it should update to ~98% of new value
        # So ref_x_2 should be close to percept_center_2, not stuck at ref_x_1
        expected_ref_x_2 = 0.02 * ref_x_1 + 0.98 * percept_center_2  # 98% new, 2% old
        
        # Allow some tolerance for floating point
        assert abs(ref_x_2 - expected_ref_x_2) < 0.1, (
            f"Reference point should update! "
            f"Expected: {expected_ref_x_2:.3f}m (98% of new perception center), "
            f"Got: {ref_x_2:.3f}m, "
            f"Previous: {ref_x_1:.3f}m. "
            f"If frozen, ref_x_2 would be ~{ref_x_1:.3f}m (same as previous)."
        )
        
        # Also verify it's moving in the right direction (toward new perception center)
        if percept_center_2 < percept_center_1:
            # Perception moved left, reference should move left
            assert ref_x_2 < ref_x_1, (
                f"Reference point should move left when perception moves left. "
                f"Previous: {ref_x_1:.3f}m, Current: {ref_x_2:.3f}m"
            )
        else:
            # Perception moved right, reference should move right
            assert ref_x_2 > ref_x_1, (
                f"Reference point should move right when perception moves right. "
                f"Previous: {ref_x_1:.3f}m, Current: {ref_x_2:.3f}m"
            )
    
    def test_reference_point_does_not_freeze_on_large_changes(self):
        """
        Test that reference point accepts large changes when using lane_positions.
        
        When using lane_positions (minimal_smoothing=True), perception is accurate
        and large changes (up to 0.5m) should be accepted, not rejected as jumps.
        """
        inference = TrajectoryPlanningInference(
            planner_type="rule_based",
            reference_smoothing=0.7
        )
        
        trajectory = Trajectory(
            points=[TrajectoryPoint(x=0.0, y=10.0, heading=0.0, velocity=10.0, curvature=0.0)],
            length=10.0
        )
        
        # Frame 1: Start at 0.338m
        lane_positions_1 = {
            'left_lane_line_x': -3.238,
            'right_lane_line_x': 3.914
        }
        ref_point_1 = inference.get_reference_point(
            trajectory=trajectory,
            lookahead=10.0,
            lane_positions=lane_positions_1,
            use_direct=True,
            timestamp=0.0
        )
        ref_x_1 = ref_point_1['x']
        
        # Frame 2: Large change (0.5m) - should be accepted (not rejected as jump)
        # This tests the 0.5m threshold for lane_positions
        lane_positions_2 = {
            'left_lane_line_x': -4.026,
            'right_lane_line_x': 3.914
        }
        percept_center_2 = (lane_positions_2['left_lane_line_x'] + lane_positions_2['right_lane_line_x']) / 2.0
        
        ref_point_2 = inference.get_reference_point(
            trajectory=trajectory,
            lookahead=10.0,
            lane_positions=lane_positions_2,
            use_direct=True,
            timestamp=0.033
        )
        ref_x_2 = ref_point_2['x']
        
        # Verify change is > 0.2m (old threshold) but < 0.5m (new threshold)
        change = abs(percept_center_2 - ref_x_1)
        assert 0.2 < change < 0.5, f"Change should be between 0.2m and 0.5m: {change:.3f}m"
        
        # Reference point should update (not freeze)
        # With minimal_smoothing, it should be close to new perception center
        assert abs(ref_x_2 - percept_center_2) < 0.2, (
            f"Reference point should update toward new perception center. "
            f"Perception center: {percept_center_2:.3f}m, "
            f"Reference point: {ref_x_2:.3f}m, "
            f"Previous: {ref_x_1:.3f}m. "
            f"If frozen, ref_x_2 would be ~{ref_x_1:.3f}m."
        )
    
    def test_reference_point_accepts_jumps_with_lane_positions(self):
        """
        Test that large jumps (>0.2m) are accepted when using lane_positions.
        
        This is the key fix: when using lane_positions (minimal_smoothing=True),
        large changes should be accepted because perception is accurate.
        """
        inference = TrajectoryPlanningInference(
            planner_type="rule_based",
            reference_smoothing=0.7
        )
        
        trajectory = Trajectory(
            points=[TrajectoryPoint(x=0.0, y=10.0, heading=0.0, velocity=10.0, curvature=0.0)],
            length=10.0
        )
        
        # Frame 1: Start at some value
        lane_positions_1 = {
            'left_lane_line_x': -3.0,
            'right_lane_line_x': 4.0
        }
        ref_point_1 = inference.get_reference_point(
            trajectory=trajectory,
            lookahead=10.0,
            lane_positions=lane_positions_1,
            use_direct=True,
            timestamp=0.0
        )
        ref_x_1 = ref_point_1['x']
        
        # Frame 2: Large jump (>0.2m but <0.5m) - should be accepted
        lane_positions_2 = {
            'left_lane_line_x': -4.0,  # Moved left by 1.0m
            'right_lane_line_x': 4.0
        }
        percept_center_2 = (lane_positions_2['left_lane_line_x'] + lane_positions_2['right_lane_line_x']) / 2.0
        
        ref_point_2 = inference.get_reference_point(
            trajectory=trajectory,
            lookahead=10.0,
            lane_positions=lane_positions_2,
            use_direct=True,
            timestamp=0.033
        )
        ref_x_2 = ref_point_2['x']
        
        # Verify the jump is > 0.2m (old threshold that would reject it)
        jump_size = abs(percept_center_2 - ref_x_1)
        assert jump_size > 0.2, f"Jump should be > 0.2m to test acceptance: {jump_size:.3f}m"
        
        # CRITICAL: Reference point should update (not reject the jump)
        # It should move toward the new perception center
        assert abs(ref_x_2 - percept_center_2) < 0.3, (
            f"Large jump should be ACCEPTED when using lane_positions! "
            f"Perception center: {percept_center_2:.3f}m, "
            f"Reference point: {ref_x_2:.3f}m, "
            f"Previous: {ref_x_1:.3f}m, "
            f"Jump size: {jump_size:.3f}m. "
            f"If rejected, ref_x_2 would be ~{ref_x_1:.3f}m (frozen)."
        )

