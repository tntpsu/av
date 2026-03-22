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


def test_async_trajectory_transport_false_skips_background_thread():
    client = UnityBridgeClient(
        "http://localhost:8000",
        async_trajectory_transport=False,
    )
    assert client._trajectory_thread is None
    assert client._trajectory_queue is None
    client.close()


def test_async_trajectory_transport_true_has_trajectory_sender():
    client = UnityBridgeClient(
        "http://localhost:8000",
        async_trajectory_transport=True,
    )
    assert client._trajectory_thread is not None
    assert client._trajectory_queue is not None
    client.close()


def test_bridge_sessions_disable_trust_env_by_default():
    """Avoid macOS System Configuration / _scproxy on every localhost request."""
    client = UnityBridgeClient("http://localhost:8000")
    assert client.session.trust_env is False
    assert client._vehicle_parallel_session.trust_env is False
    client.close()


def test_bridge_sessions_honor_trust_proxy_env():
    client = UnityBridgeClient(
        "http://localhost:8000",
        trust_proxy_env=True,
    )
    assert client.session.trust_env is True
    assert client._vehicle_parallel_session.trust_env is True
    client.close()
