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
            "packet_key": "avbridge:17:222",
            "unity_frame_count": 222,
            "fallback_active": False,
            "fallback_reason_code": "none",
            "selection_result": "complete",
            "join_failure_reason_code": "none",
            "join_failure_side_code": "none",
            "selected_failure_contract_reason_code": "none",
            "selected_failure_source_stage_code": "none",
            "source_bundle_active_transport_eligible": True,
            "source_bundle_debug_unbundled_capture": False,
            "camera_capture_contract_reason": "",
            "source_key_present_camera": True,
            "source_key_present_vehicle": True,
            "timeout_event_delta": 0,
            "active_camera_excluded_event_delta": 0,
            "active_camera_excluded_reason_code": "",
            "unbundled_camera_entered_active_path_event_delta": 0,
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
            "packet_key": "avbridge:17:222",
            "unity_frame_count": 222,
            "fallback_active": False,
            "fallback_reason_code": "none",
            "payload_queue_depth": 0,
            "consume_policy": PACKET_CONSUME_POLICY_FRESHEST_WITHIN_BUDGET,
            "selection_result": "complete",
            "payload_selected_age_ms": 40.0,
            "payload_selected_fresh": True,
            "payload_warn_age_exceeded": False,
            "payload_stale_drop_count": 2,
            "payload_drained_count": 2,
            "payload_max_drained_age_ms": 420.0,
            "payload_selection_source": "server_selector",
            "payload_server_queue_depth_after_select": 0,
            "payload_server_oldest_age_ms_after_select": float("nan"),
            "join_failure_reason_code": "none",
            "join_failure_side_code": "none",
            "selected_failure_contract_reason_code": "none",
            "selected_failure_source_stage_code": "none",
            "source_bundle_active_transport_eligible": True,
            "source_bundle_debug_unbundled_capture": False,
            "camera_capture_contract_reason": "",
            "source_key_present_camera": True,
            "source_key_present_vehicle": True,
            "timeout_event_delta": 0,
            "active_camera_excluded_event_delta": 0,
            "active_camera_excluded_reason_code": "",
            "unbundled_camera_entered_active_path_event_delta": 0,
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
    assert result["selection_result"] == "complete"
    assert result["payload"]["packet_meta"]["payload_selected_age_ms"] == 40.0
    assert result["selected_fresh"] is True
    assert result["stale_drop_count"] == 2
    assert result["drained_count"] == 2
    assert result["max_drained_age_ms"] == 420.0
    assert result["selected_failure_contract_reason_code"] == "none"
    assert result["selected_failure_source_stage_code"] == "none"
    assert result["source_bundle_active_transport_eligible"] is True
    assert result["source_bundle_debug_unbundled_capture"] is False
    assert result["camera_capture_contract_reason"] == ""


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
            "X-AV-Selection-Result": "fallback",
            "X-AV-Join-Failure-Reason": "vehicle_component_late",
            "X-AV-Join-Failure-Side": "vehicle",
            "X-AV-Selected-Failure-Contract-Reason": "vehicle_not_sent",
            "X-AV-Selected-Failure-Source-Stage": "source_bundle",
            "X-AV-Source-Bundle-Close-Reason": "open",
            "X-AV-Source-Bundle-Deadline-Ms": "100.0",
            "X-AV-Source-Bundle-Age-Ms": "22.0",
            "X-AV-Source-Bundle-Inflight-Count": "1",
            "X-AV-Source-Bundle-Vehicle-State-Built": "1",
            "X-AV-Source-Bundle-Vehicle-State-Enqueued": "1",
            "X-AV-Source-Bundle-Vehicle-State-Sent": "0",
            "X-AV-Source-Bundle-Camera-Requested": "1",
            "X-AV-Source-Bundle-Camera-Request-Attempted": "1",
            "X-AV-Source-Bundle-Camera-Request-Accepted": "1",
            "X-AV-Source-Bundle-Camera-Request-Rejected-Reason": "",
            "X-AV-Source-Bundle-Camera-Request-Skipped-Reason": "camera_capture_missing",
            "X-AV-Source-Bundle-Camera-Request-Disposition-Code": "skipped_camera_capture_missing",
            "X-AV-Source-Bundle-Camera-Request-Attempt-Age-Ms": "11.0",
            "X-AV-Source-Bundle-Camera-Request-Accept-Age-Ms": "10.0",
            "X-AV-Source-Bundle-Camera-Request-Queue-Depth": "2",
            "X-AV-Source-Bundle-Active-Transport-Eligible": "0",
            "X-AV-Source-Bundle-Debug-Unbundled-Capture": "1",
            "X-AV-Camera-Capture-Contract-Reason": "no_source_bundle_context",
            "X-AV-Source-Bundle-Camera-Sent": "1",
            "X-AV-Source-Bundle-Aborted-Before-Vehicle-Send": "1",
            "X-AV-Source-Bundle-Abort-Reason": "bundle_aborted_camera_request_not_attempted",
            "X-AV-Source-Bundle-Vehicle-Send-Blocked-By-Camera-Request": "1",
            "X-AV-Source-Bundle-Superseded-Before-Send": "0",
            "X-AV-Source-Key-Present-Camera": "1",
            "X-AV-Source-Key-Present-Vehicle": "0",
            "X-AV-Timeout-Event-Delta": "1",
            "X-AV-Join-Failure-Event-Count": "5",
            "X-AV-Active-Camera-Excluded-Event-Delta": "2",
            "X-AV-Active-Camera-Excluded-Reason": "no_source_bundle_context",
            "X-AV-Unbundled-Camera-Entered-Active-Path-Event-Delta": "0",
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
    assert result["selection_result"] == "fallback"
    assert result["join_failure_reason_code"] == "vehicle_component_late"
    assert result["join_failure_side_code"] == "vehicle"
    assert result["selected_failure_contract_reason_code"] == "vehicle_not_sent"
    assert result["selected_failure_source_stage_code"] == "source_bundle"
    assert result["source_bundle_close_reason"] == "open"
    assert result["source_bundle_deadline_ms"] == 100.0
    assert result["source_bundle_age_ms"] == 22.0
    assert result["source_bundle_inflight_count"] == 1
    assert result["source_bundle_vehicle_state_built"] is True
    assert result["source_bundle_vehicle_state_enqueued"] is True
    assert result["source_bundle_vehicle_state_sent"] is False
    assert result["source_bundle_camera_requested"] is True
    assert result["source_bundle_camera_request_attempted"] is True
    assert result["source_bundle_camera_request_accepted"] is True
    assert result["source_bundle_camera_request_rejected_reason"] == ""
    assert result["source_bundle_camera_request_skipped_reason"] == "camera_capture_missing"
    assert (
        result["source_bundle_camera_request_disposition_code"]
        == "skipped_camera_capture_missing"
    )
    assert result["source_bundle_camera_request_attempt_age_ms"] == 11.0
    assert result["source_bundle_camera_request_accept_age_ms"] == 10.0
    assert result["source_bundle_camera_request_queue_depth"] == 2
    assert result["source_bundle_active_transport_eligible"] is False
    assert result["source_bundle_debug_unbundled_capture"] is True
    assert result["camera_capture_contract_reason"] == "no_source_bundle_context"
    assert result["source_bundle_camera_sent"] is True
    assert result["source_bundle_aborted_before_vehicle_send"] is True
    assert (
        result["source_bundle_abort_reason"]
        == "bundle_aborted_camera_request_not_attempted"
    )
    assert result["source_bundle_vehicle_send_blocked_by_camera_request"] is True
    assert result["source_bundle_superseded_before_send"] is False
    assert result["source_key_present_camera"] is True
    assert result["source_key_present_vehicle"] is False
    assert result["timeout_event_delta"] == 1
    assert result["active_camera_excluded_event_delta"] == 2
    assert result["active_camera_excluded_reason_code"] == "no_source_bundle_context"
    assert result["unbundled_camera_entered_active_path_event_delta"] == 0
