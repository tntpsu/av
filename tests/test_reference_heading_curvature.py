import numpy as np

from trajectory.models.trajectory_planner import RuleBasedTrajectoryPlanner


def test_reference_heading_nonzero_on_curved_lanes() -> None:
    """Ensure heading is not forced to 0 when lanes are curved."""
    planner = RuleBasedTrajectoryPlanner(
        image_width=640.0,
        image_height=480.0,
        camera_fov=75.0,
        camera_height=1.2,
    )
    # Curved lanes in image space (small quadratic, noticeable curvature).
    left_coeffs = np.array([0.0005, 0.0, 200.0])
    right_coeffs = np.array([0.0005, 0.0, 440.0])

    ref = planner.compute_reference_point_direct(left_coeffs, right_coeffs, lookahead=8.0)
    assert ref is not None
    assert abs(ref["heading"]) > np.radians(1.0)
