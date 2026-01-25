import numpy as np

from tools.ground_truth_follower import GroundTruthFollower


def test_compute_speed_scale_clamps():
    scale = GroundTruthFollower._compute_speed_scale(curvature=0.5, gain=2.0, min_scale=0.4)
    assert np.isclose(scale, 0.4)


def test_update_vehicle_params_from_state():
    follower = GroundTruthFollower.__new__(GroundTruthFollower)
    follower.wheelbase = 2.5
    follower.max_steer_angle_deg = 30.0
    follower.max_steer_angle_rad = np.deg2rad(follower.max_steer_angle_deg)
    follower.unity_fixed_delta_time = 0.02
    follower.vehicle_params_source = "defaults"

    follower._update_vehicle_params(
        {
            "maxSteerAngle": 35.0,
            "wheelbaseMeters": 2.8,
            "fixedDeltaTime": 0.025,
        }
    )

    assert np.isclose(follower.max_steer_angle_deg, 35.0)
    assert np.isclose(follower.wheelbase, 2.8)
    assert np.isclose(follower.unity_fixed_delta_time, 0.025)
    assert follower.vehicle_params_source == "unity_state"
