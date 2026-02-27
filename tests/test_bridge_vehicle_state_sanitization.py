import math
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bridge import server


def _base_state():
    return {
        "position": {"x": 0.0, "y": 0.5, "z": 0.0},
        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
        "angularVelocity": {"x": 0.0, "y": 0.0, "z": 0.0},
        "speed": 0.0,
        "steeringAngle": 0.0,
        "motorTorque": 0.0,
        "brakeTorque": 0.0,
    }


def test_sanitize_json_compatible_replaces_non_finite_values():
    payload = {
        "speed": 1.0,
        "chassisGroundClearanceM": float("nan"),
        "nested": [1.0, float("inf"), -float("inf")],
    }
    sanitized, replacements = server._sanitize_json_compatible(payload)

    assert replacements == 3
    assert sanitized["speed"] == 1.0
    assert sanitized["chassisGroundClearanceM"] is None
    assert sanitized["nested"] == [1.0, None, None]


def test_vehicle_state_latest_endpoint_sanitizes_nan_without_500():
    previous_state = server.latest_vehicle_state
    try:
        bad_state = _base_state()
        bad_state["chassisGroundClearanceM"] = float("nan")
        server.latest_vehicle_state = bad_state

        body = asyncio.run(server.get_latest_vehicle_state())
        assert body["chassisGroundClearanceM"] is None
    finally:
        server.latest_vehicle_state = previous_state


def test_negative_injection_does_not_block_control_loop_endpoints():
    prev_state = server.latest_vehicle_state
    prev_command = server.latest_control_command
    prev_queue = list(server.vehicle_state_queue)
    prev_events = server.vehicle_state_sanitize_events
    prev_values = server.vehicle_state_sanitize_values
    try:
        injected = _base_state()
        injected["chassisGroundClearanceM"] = float("nan")
        server.latest_vehicle_state = injected
        server.vehicle_state_queue.clear()
        server.vehicle_state_queue.append(injected)
        server.latest_control_command = {"steering": 0.1, "throttle": 0.2, "brake": 0.0}

        for _ in range(5):
            state_resp = asyncio.run(server.get_latest_vehicle_state())
            control_resp = asyncio.run(server.get_control_command())
            assert state_resp["chassisGroundClearanceM"] is None
            assert math.isclose(control_resp["throttle"], 0.2, rel_tol=1e-6)

        assert server.vehicle_state_sanitize_events >= prev_events
        assert server.vehicle_state_sanitize_values >= prev_values
    finally:
        server.latest_vehicle_state = prev_state
        server.latest_control_command = prev_command
        server.vehicle_state_queue.clear()
        server.vehicle_state_queue.extend(prev_queue)
        server.vehicle_state_sanitize_events = prev_events
        server.vehicle_state_sanitize_values = prev_values
