from bridge.client import UnityBridgeClient


def test_build_control_command_includes_randomize_fields():
    client = UnityBridgeClient("http://localhost:8000")
    command = client._build_control_command(
        steering=0.1,
        throttle=0.2,
        brake=0.0,
        ground_truth_mode=True,
        ground_truth_speed=5.0,
        randomize_start=True,
        randomize_request_id=42,
        randomize_seed=7,
    )

    assert command["ground_truth_mode"] is True
    assert command["randomize_start"] is True
    assert command["randomize_request_id"] == 42
    assert command["randomize_seed"] == 7


def test_build_control_command_omits_seed_when_none():
    client = UnityBridgeClient("http://localhost:8000")
    command = client._build_control_command(
        steering=0.0,
        throttle=0.0,
        brake=0.0,
        randomize_start=False,
        randomize_request_id=0,
        randomize_seed=None,
    )

    assert "randomize_seed" not in command
