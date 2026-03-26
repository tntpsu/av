import json

import numpy as np

from bridge.client import UnityBridgeClient
from sync_contract import (
    PACKET_CONSUME_POLICY_FRESHEST_WITHIN_BUDGET,
    PACKET_FALLBACK_NONE,
    PACKET_FALLBACK_PAYLOAD_AGE_BUDGET_EXCEEDED,
    PACKET_FALLBACK_PACKET_QUEUE_OVERFLOW,
)


def test_next_sync_packet_bundle_aligns_expected_unity_frame() -> None:
    client = UnityBridgeClient.__new__(UnityBridgeClient)
    client.get_next_sync_packet = lambda: {"unity_frame_count": 42, "packet_id": 7}
    client.get_next_camera_frame_with_metadata = (
        lambda camera_id="front_center": (
            object(),
            1.0,
            100,
            {"unity_frame_count": 42},
        )
    )
    client.get_next_vehicle_state = lambda: {"unityFrameCount": 42}

    bundle = client.get_next_sync_packet_bundle(max_resync_discards=2)

    assert bundle is not None
    assert bundle["fallback_active"] is False
    assert bundle["fallback_reason_code"] == PACKET_FALLBACK_NONE
    assert bundle["discarded_camera_frames"] == 0
    assert bundle["discarded_vehicle_states"] == 0
    assert bundle["frame_data"][3]["unity_frame_count"] == 42
    assert bundle["vehicle_state"]["unityFrameCount"] == 42


def test_next_sync_packet_bundle_discards_stale_components_before_aligning() -> None:
    client = UnityBridgeClient.__new__(UnityBridgeClient)
    client.get_next_sync_packet = lambda: {"unity_frame_count": 50, "packet_id": 8}
    camera_frames = iter(
        [
            (object(), 1.0, 1, {"unity_frame_count": 48}),
            (object(), 1.1, 2, {"unity_frame_count": 50}),
        ]
    )
    vehicle_states = iter(
        [
            {"unityFrameCount": 49},
            {"unityFrameCount": 50},
        ]
    )
    client.get_next_camera_frame_with_metadata = lambda camera_id="front_center": next(camera_frames, None)
    client.get_next_vehicle_state = lambda: next(vehicle_states, None)

    bundle = client.get_next_sync_packet_bundle(max_resync_discards=3)

    assert bundle is not None
    assert bundle["fallback_active"] is False
    assert bundle["discarded_camera_frames"] == 1
    assert bundle["discarded_vehicle_states"] == 1
    assert bundle["frame_data"][3]["unity_frame_count"] == 50
    assert bundle["vehicle_state"]["unityFrameCount"] == 50


def test_next_sync_packet_bundle_flags_queue_overflow_on_component_overshoot() -> None:
    client = UnityBridgeClient.__new__(UnityBridgeClient)
    client.get_next_sync_packet = lambda: {"unity_frame_count": 60, "packet_id": 9}
    client.get_next_camera_frame_with_metadata = (
        lambda camera_id="front_center": (
            object(),
            1.0,
            1,
            {"unity_frame_count": 63},
        )
    )
    client.get_next_vehicle_state = lambda: {"unityFrameCount": 60}

    bundle = client.get_next_sync_packet_bundle(max_resync_discards=1)

    assert bundle is not None
    assert bundle["fallback_active"] is True
    assert bundle["fallback_reason_code"] == PACKET_FALLBACK_PACKET_QUEUE_OVERFLOW


def test_parse_sync_packet_payload_response_round_trips_image_and_metadata() -> None:
    client = UnityBridgeClient.__new__(UnityBridgeClient)
    image = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    metadata = {
        "packet": {
            "packet_id": 17,
            "unity_frame_count": 222,
            "fallback_active": False,
            "fallback_reason_code": "none",
            "payload_queue_depth": 3,
        },
        "vehicle_state": {"unityFrameCount": 222, "speed": 5.0},
        "front_camera": {
            "timestamp": 12.5,
            "frame_id": 19,
            "camera_id": "front_center",
            "unity_frame_count": 222,
            "unity_time": 4.5,
            "shape": list(image.shape),
            "dtype": str(image.dtype),
        },
    }
    boundary = "payload-boundary"
    body = (
        f"--{boundary}\r\nContent-Type: application/json\r\n\r\n".encode("utf-8")
        + json.dumps(metadata).encode("utf-8")
        + f"\r\n--{boundary}\r\nContent-Type: application/octet-stream\r\n\r\n".encode("utf-8")
        + image.tobytes(order="C")
        + f"\r\n--{boundary}--\r\n".encode("utf-8")
    )

    class _Response:
        def __init__(self, content: bytes, content_type: str):
            self.content = content
            self.headers = {"Content-Type": content_type}

    parsed = client._parse_sync_packet_payload_response(
        _Response(body, f'multipart/mixed; boundary="{boundary}"')
    )

    assert np.array_equal(parsed["image"], image)
    assert parsed["frame_id"] == 19
    assert parsed["timestamp"] == 12.5
    assert parsed["packet_meta"]["packet_id"] == 17
    assert parsed["fallback_active"] is False
    assert parsed["fallback_reason_code"] == "none"
    assert parsed["vehicle_state_dict"]["unityFrameCount"] == 222


def test_get_fresh_sync_packet_payload_raw_selects_newest_within_budget() -> None:
    client = UnityBridgeClient.__new__(UnityBridgeClient)
    image = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    metadata = {
        "packet": {
            "packet_id": 17,
            "unity_frame_count": 222,
            "fallback_active": False,
            "fallback_reason_code": "none",
            "payload_queue_depth": 0,
            "consume_policy": PACKET_CONSUME_POLICY_FRESHEST_WITHIN_BUDGET,
            "payload_selected_age_ms": 40.0,
            "payload_selected_fresh": True,
            "payload_warn_age_exceeded": False,
            "payload_stale_drop_count": 2,
            "payload_drained_count": 2,
            "payload_max_drained_age_ms": 420.0,
            "payload_selection_source": "server_selector",
            "payload_server_queue_depth_after_select": 0,
            "payload_server_oldest_age_ms_after_select": float("nan"),
        },
        "vehicle_state": {"unityFrameCount": 222, "speed": 5.0},
        "front_camera": {
            "timestamp": 12.5,
            "frame_id": 19,
            "camera_id": "front_center",
            "unity_frame_count": 222,
            "unity_time": 4.5,
            "shape": list(image.shape),
            "dtype": str(image.dtype),
        },
    }
    boundary = "payload-boundary"
    body = (
        f"--{boundary}\r\nContent-Type: application/json\r\n\r\n".encode("utf-8")
        + json.dumps(metadata).encode("utf-8")
        + f"\r\n--{boundary}\r\nContent-Type: application/octet-stream\r\n\r\n".encode("utf-8")
        + image.tobytes(order="C")
        + f"\r\n--{boundary}--\r\n".encode("utf-8")
    )

    class _Response:
        status_code = 200
        headers = {"Content-Type": f'multipart/mixed; boundary="{boundary}"'}
        content = body

        @staticmethod
        def raise_for_status() -> None:
            return None

    calls = {"count": 0, "params": None}

    class _Session:
        @staticmethod
        def get(url, params=None, timeout=None):
            calls["count"] += 1
            calls["params"] = dict(params or {})
            return _Response()

    client.session = _Session()
    client.base_url = "http://localhost:8000"

    result = client.get_fresh_sync_packet_payload_raw(
        consume_policy=PACKET_CONSUME_POLICY_FRESHEST_WITHIN_BUDGET,
        max_age_ms=250.0,
        warn_age_ms=125.0,
        max_drain_count=8,
    )

    assert calls["count"] == 1
    assert calls["params"]["consume_policy"] == PACKET_CONSUME_POLICY_FRESHEST_WITHIN_BUDGET
    assert result["selection_source"] == "server_selector"
    assert result["payload"]["packet_meta"]["payload_selected_age_ms"] == 40.0
    assert result["selected_fresh"] is True
    assert result["stale_drop_count"] == 2
    assert result["drained_count"] == 2
    assert result["max_drained_age_ms"] == 420.0


def test_get_fresh_sync_packet_payload_raw_flags_stale_timeout_when_only_stale_packets_exist() -> None:
    client = UnityBridgeClient.__new__(UnityBridgeClient)

    class _Response:
        status_code = 404
        headers = {
            "X-AV-Selection-Source": "server_selector",
            "X-AV-Selection-Fallback-Active": "1",
            "X-AV-Selection-Fallback-Reason": PACKET_FALLBACK_PAYLOAD_AGE_BUDGET_EXCEEDED,
            "X-AV-Selected-Payload-Age-Ms": "520.0",
            "X-AV-Selected-Payload-Fresh": "0",
            "X-AV-Selected-Payload-Warn-Age-Exceeded": "1",
            "X-AV-Stale-Drop-Count": "1",
            "X-AV-Drained-Count": "1",
            "X-AV-Max-Drained-Age-Ms": "610.0",
            "X-AV-Server-Queue-Depth-After-Select": "0",
            "X-AV-Server-Oldest-Age-Ms-After-Select": "",
        }

        @staticmethod
        def raise_for_status() -> None:
            raise AssertionError("raise_for_status should not be called for 404 path")

    calls = {"count": 0}

    class _Session:
        @staticmethod
        def get(url, params=None, timeout=None):
            calls["count"] += 1
            return _Response()

    client.session = _Session()
    client.base_url = "http://localhost:8000"

    result = client.get_fresh_sync_packet_payload_raw(
        consume_policy=PACKET_CONSUME_POLICY_FRESHEST_WITHIN_BUDGET,
        max_age_ms=250.0,
        warn_age_ms=125.0,
        max_drain_count=8,
    )

    assert result["payload"] is None
    assert result["fallback_active"] is True
    assert result["fallback_reason_code"] == PACKET_FALLBACK_PAYLOAD_AGE_BUDGET_EXCEEDED
    assert calls["count"] == 1
    assert result["stale_drop_count"] == 1
