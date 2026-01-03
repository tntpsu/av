"""
Tests for trajectory planning.
"""

import pytest
import numpy as np
from trajectory.models.trajectory_planner import RuleBasedTrajectoryPlanner, TrajectoryPoint


def test_trajectory_planner():
    """Test trajectory planner."""
    planner = RuleBasedTrajectoryPlanner(lookahead_distance=50.0)
    
    # Create dummy lane coefficients (left and right lanes)
    left_lane = np.array([0.001, 0.0, 1.0])  # Slight curve
    right_lane = np.array([0.001, 0.0, 3.0])  # Parallel lane
    
    trajectory = planner.plan([left_lane, right_lane])
    
    assert trajectory is not None
    assert len(trajectory.points) > 0
    assert trajectory.length > 0


def test_straight_trajectory():
    """Test straight trajectory when no lanes."""
    planner = RuleBasedTrajectoryPlanner()
    
    trajectory = planner.plan([None, None])
    
    assert trajectory is not None
    assert len(trajectory.points) > 0

