"""Tests for shadow synchronized packet assembly on the bridge."""

from __future__ import annotations

import asyncio
import io
import time

import numpy as np
import pytest
from PIL import Image
from requests.structures import CaseInsensitiveDict

import bridge.server as srv
from sync_contract import (
    PACKET_COHERENCE_REASON_COHERENT,
    PACKET_COHERENCE_REASON_FRONT_VEHICLE_FRAME_DELTA_BUDGET_EXCEEDED,
    PACKET_COHERENCE_REASON_FRONT_VEHICLE_TIME_DELTA_BUDGET_EXCEEDED,
    PACKET_CONSUME_POLICY_FRESHEST_WITHIN_BUDGET,
    PACKET_JOIN_FAILURE_PACKET_KEY_MISMATCH,
    PACKET_JOIN_FAILURE_SOURCE_BUNDLE_ABORTED,
    PACKET_JOIN_FAILURE_SIDE_BOTH,
    PACKET_JOIN_FAILURE_VEHICLE_KEY_MISSING,
    PACKET_JOIN_FAILURE_SIDE_VEHICLE,
    PACKET_FALLBACK_PAYLOAD_AGE_BUDGET_EXCEEDED,
    PACKET_SELECTED_FAILURE_CONTRACT_BUNDLE_ABORTED_CAMERA_REQUEST_NOT_ATTEMPTED,
    PACKET_SELECTED_FAILURE_CONTRACT_CAMERA_REQUEST_NOT_ATTEMPTED,
    PACKET_SELECTED_FAILURE_CONTRACT_VEHICLE_NOT_SENT,
    PACKET_SELECTED_FAILURE_STAGE_CAMERA_CAPTURE,
    PACKET_SELECTED_FAILURE_STAGE_SOURCE_BUNDLE,
    PACKET_SELECTION_RESULT_COMPLETE_INCOHERENT,
)


def _tiny_jpeg_bytes() -> bytes:
    img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def _reset_sync_globals():
    srv.latest_camera_frames.clear()
    srv.latest_camera_frame = None
    srv.latest_frame_timestamps_by_id.clear()
    srv.latest_frame_ids_by_id.clear()
    srv.camera_frame_queues.clear()
    srv.camera_drop_count.clear()
    srv.latest_vehicle_state = None
    srv.vehicle_state_queue.clear()
    srv.pending_packets_by_unity_frame.clear()
    srv.pending_packets_by_packet_key.clear()
    srv.assembled_packet_queue.clear()
    srv.assembled_packet_payload_queue.clear()
    srv.latest_sync_packet_data = None
    srv.latest_sync_packet_payload_data = None
    srv.latest_sync_packet_join_failure_reason = "none"
    srv.latest_sync_packet_join_failure_side = "none"
    srv.latest_sync_packet_join_failure_source_key_present_camera = False
    srv.latest_sync_packet_join_failure_source_key_present_vehicle = False
    srv.sync_packet_join_failure_event_count = 0
    srv.sync_packet_drop_count = 0
    srv.sync_packet_payload_drop_count = 0
    srv.sync_packet_orphan_camera_count = 0
    srv.sync_packet_orphan_vehicle_count = 0
    srv.sync_packet_timeout_count = 0
    srv.sync_packet_key_match_count = 0
    srv.sync_packet_unity_fallback_count = 0
    srv.sync_packet_superseded_camera_count = 0
    srv.sync_packet_superseded_vehicle_count = 0
    srv.sync_packet_active_camera_excluded_count = 0
    srv.sync_packet_active_camera_excluded_pending_count = 0
    srv.latest_sync_packet_active_camera_excluded_reason = ""
    srv.sync_packet_unbundled_camera_entered_active_path_count = 0
    srv.sync_packet_unbundled_camera_entered_active_path_pending_count = 0
    srv.latest_sync_packet_join_failure_reason = "none"
    srv.latest_sync_packet_join_failure_side = "none"
    srv.latest_sync_packet_join_failure_source_key_present_camera = False
    srv.latest_sync_packet_join_failure_source_key_present_vehicle = False
    srv.sync_packet_join_failure_event_count = 0
    yield
    srv.pending_packets_by_unity_frame.clear()
    srv.pending_packets_by_packet_key.clear()
    srv.assembled_packet_queue.clear()
    srv.assembled_packet_payload_queue.clear()
    srv.latest_sync_packet_data = None
    srv.latest_sync_packet_payload_data = None


def _vehicle_state(
    unity_frame_count: int,
    unity_time: float,
    *,
    sync_packet_key: str | None = None,
) -> srv.VehicleState:
    return srv.VehicleState(
        position={"x": 0.0, "y": 0.0, "z": 0.0},
        rotation={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        velocity={"x": 0.0, "y": 0.0, "z": 0.0},
        angularVelocity={"x": 0.0, "y": 0.0, "z": 0.0},
        speed=0.0,
        steeringAngle=0.0,
        motorTorque=0.0,
        brakeTorque=0.0,
        unityFrameCount=unity_frame_count,
        unityTime=unity_time,
        syncPacketKey=sync_packet_key,
    )


def test_sync_packet_latest_and_next_round_trip(_reset_sync_globals):
    jpeg = _tiny_jpeg_bytes()

    async def _run() -> None:
        await srv._decode_and_store_camera_frame(
            jpeg,
            "1.25",
            "101",
            "front_center",
            unity_frame_count=123,
            unity_time=4.5,
        )
        await srv.receive_vehicle_state(_vehicle_state(123, 4.5))

        latest = await srv.get_latest_sync_packet()
        queued = await srv.get_next_sync_packet()

        assert latest["complete"] is True
        assert latest["unity_frame_count"] == 123
        assert latest["front_camera"]["frame_id"] == 101
        assert latest["vehicle_state"]["unity_frame_count"] == 123
        assert queued["packet_id"] == latest["packet_id"]

    asyncio.run(_run())


def test_sync_packet_timeout_counts_vehicle_orphan(_reset_sync_globals):
    jpeg = _tiny_jpeg_bytes()

    async def _run() -> None:
        await srv.receive_vehicle_state(_vehicle_state(10, 1.0))
        await srv._decode_and_store_camera_frame(
            jpeg,
            "2.0",
            "202",
            "front_center",
            unity_frame_count=20,
            unity_time=2.0,
            arrival_wall_time=200.0,
        )

        assert srv.sync_packet_timeout_count >= 1
        assert srv.sync_packet_orphan_vehicle_count >= 1

    asyncio.run(_run())


def test_sync_packet_timeout_labels_vehicle_key_missing(_reset_sync_globals):
    async def _run() -> None:
        await srv.receive_vehicle_state(_vehicle_state(10, 1.0, sync_packet_key=None))
        await srv._decode_and_store_camera_frame(
            _tiny_jpeg_bytes(),
            "2.0",
            "202",
            "front_center",
            sync_packet_key="avbridge:11:20",
            unity_frame_count=20,
            unity_time=2.0,
            arrival_wall_time=200.0,
        )
        assert srv.latest_sync_packet_join_failure_reason == PACKET_JOIN_FAILURE_VEHICLE_KEY_MISSING
        assert srv.latest_sync_packet_join_failure_side == PACKET_JOIN_FAILURE_SIDE_VEHICLE
        assert srv.sync_packet_join_failure_event_count >= 1

    asyncio.run(_run())


def test_sync_packet_next_raw_returns_payload_response(_reset_sync_globals):
    jpeg = _tiny_jpeg_bytes()

    async def _run() -> None:
        await srv._decode_and_store_camera_frame(
            jpeg,
            "1.25",
            "101",
            "front_center",
            unity_frame_count=123,
            unity_time=4.5,
        )
        await srv.receive_vehicle_state(_vehicle_state(123, 4.5))

        response = await srv.get_next_sync_packet_raw()

        assert response.headers["content-type"].startswith("multipart/mixed")
        assert response.headers["X-AV-Unity-Frame-Count"] == "123"
        assert int(response.headers["X-AV-Payload-Bytes"]) > 0

    asyncio.run(_run())


def test_sync_packet_packet_key_join_overrides_unity_frame_mismatch(_reset_sync_globals):
    jpeg = _tiny_jpeg_bytes()

    async def _run() -> None:
        packet_key = "front_center:101:123"
        await srv._decode_and_store_camera_frame(
            jpeg,
            "1.25",
            "101",
            "front_center",
            sync_packet_key=packet_key,
            unity_frame_count=123,
            unity_time=4.5,
        )
        await srv.receive_vehicle_state(
            _vehicle_state(
                129,
                4.7,
                sync_packet_key=packet_key,
            )
        )

        latest = await srv.get_latest_sync_packet()

        assert latest["complete"] is True
        assert latest["join_source"] == "packet_key"
        assert latest["join_key_present"] is True
        assert latest["key_match_count"] >= 1
        assert latest["unity_frame_count"] == 123
        assert latest["front_vehicle_frame_delta"] == 123 - 129

    asyncio.run(_run())


def test_sync_packet_snapshot_marks_complete_packet_incoherent_on_large_time_delta(
    _reset_sync_globals,
):
    packet = _payload_packet(packet_id=21, unity_frame_count=123, publish_wall_time=time.time())
    packet["front_camera"]["unity_frame_count"] = 166
    packet["front_camera"]["unity_time"] = 5.2
    packet["front_camera"]["arrival_wall_time"] = time.time()
    packet["vehicle_state"]["unity_frame_count"] = 123
    packet["vehicle_state"]["unity_time"] = 4.5
    packet["vehicle_state"]["arrival_wall_time"] = time.time()
    packet["created_wall_time"] = packet["publish_wall_time"] - 0.75

    snapshot = srv._snapshot_sync_packet(packet, now=time.time())

    assert snapshot["complete"] is True
    assert snapshot["coherence_pass"] is False
    assert snapshot["complete_but_incoherent"] is True
    assert (
        snapshot["coherence_reason_code"]
        == PACKET_COHERENCE_REASON_FRONT_VEHICLE_TIME_DELTA_BUDGET_EXCEEDED
    )
    assert snapshot["front_vehicle_time_delta_budget_exceeded"] is True


def test_sync_packet_snapshot_marks_coherent_packet_with_small_source_delta(
    _reset_sync_globals,
):
    packet = _payload_packet(packet_id=22, unity_frame_count=123, publish_wall_time=time.time())
    packet["front_camera"]["arrival_wall_time"] = time.time()
    packet["front_camera"]["source_packet_context_queue_depth"] = 1
    packet["front_camera"]["source_packet_context_dropped_stale_count"] = 0
    packet["front_camera"]["source_packet_context_missing_count"] = 0
    packet["front_camera"]["source_packet_context_frame_delta"] = 1
    packet["front_camera"]["source_packet_context_time_delta_ms"] = 16.7

    snapshot = srv._snapshot_sync_packet(packet, now=time.time())

    assert snapshot["coherence_pass"] is True
    assert snapshot["coherence_reason_code"] == PACKET_COHERENCE_REASON_COHERENT
    assert snapshot["source_packet_context_queue_depth"] == 1
    assert snapshot["source_packet_context_frame_delta"] == 1
    assert snapshot["source_packet_context_time_delta_ms"] == pytest.approx(16.7, rel=1e-6)


def test_sync_packet_snapshot_classifies_vehicle_not_sent_contract_failure(
    _reset_sync_globals,
):
    packet = {
        "packet_id": 30,
        "front_camera": {
            "source_bundle_camera_requested": True,
            "source_bundle_camera_request_attempted": True,
            "source_bundle_camera_request_accepted": True,
            "source_bundle_vehicle_state_built": True,
            "source_bundle_vehicle_state_enqueued": True,
            "source_bundle_vehicle_state_sent": False,
            "source_bundle_superseded_before_send": False,
            "source_bundle_close_reason": "open",
        },
        "vehicle_state": None,
    }

    snapshot = srv._snapshot_sync_packet(packet, now=time.time())

    assert snapshot["selected_failure_contract_reason_code"] == (
        PACKET_SELECTED_FAILURE_CONTRACT_VEHICLE_NOT_SENT
    )
    assert snapshot["selected_failure_source_stage_code"] == (
        PACKET_SELECTED_FAILURE_STAGE_SOURCE_BUNDLE
    )


def test_sync_packet_snapshot_classifies_camera_request_not_attempted(
    _reset_sync_globals,
):
    packet = {
        "packet_id": 31,
        "front_camera": None,
        "vehicle_state": {
            "source_bundle_camera_requested": False,
            "source_bundle_camera_request_attempted": False,
            "source_bundle_camera_request_accepted": False,
            "source_bundle_camera_sent": False,
            "source_bundle_close_reason": "open",
        },
    }

    snapshot = srv._snapshot_sync_packet(packet, now=time.time())

    assert snapshot["selected_failure_contract_reason_code"] == (
        PACKET_SELECTED_FAILURE_CONTRACT_CAMERA_REQUEST_NOT_ATTEMPTED
    )
    assert snapshot["selected_failure_source_stage_code"] == (
        PACKET_SELECTED_FAILURE_STAGE_SOURCE_BUNDLE
    )


def test_sync_packet_snapshot_classifies_source_bundle_abort_reason(
    _reset_sync_globals,
):
    packet = {
        "packet_id": 31,
        "front_camera": None,
        "vehicle_state": {
            "source_bundle_aborted_before_vehicle_send": True,
            "source_bundle_abort_reason": "bundle_aborted_camera_request_not_attempted",
            "source_bundle_camera_request_attempted": False,
            "source_bundle_camera_request_accepted": False,
            "source_bundle_camera_sent": False,
            "source_bundle_close_reason": "aborted",
        },
    }

    snapshot = srv._snapshot_sync_packet(packet, now=time.time())

    join_failure_reason, join_failure_side, _, _ = srv._classify_join_failure(packet, [packet])
    assert join_failure_reason == PACKET_JOIN_FAILURE_SOURCE_BUNDLE_ABORTED
    assert join_failure_side == "none"
    assert snapshot["selected_failure_contract_reason_code"] == (
        PACKET_SELECTED_FAILURE_CONTRACT_BUNDLE_ABORTED_CAMERA_REQUEST_NOT_ATTEMPTED
    )
    assert snapshot["selected_failure_source_stage_code"] == (
        PACKET_SELECTED_FAILURE_STAGE_SOURCE_BUNDLE
    )
    assert snapshot["source_bundle_aborted_before_vehicle_send"] is True
    assert (
        snapshot["source_bundle_abort_reason"]
        == "bundle_aborted_camera_request_not_attempted"
    )


def test_debug_unbundled_camera_capture_is_excluded_from_active_packet_assembly(
    _reset_sync_globals,
):
    async def _run() -> None:
        await srv._decode_and_store_camera_frame(
            _tiny_jpeg_bytes(),
            "2.0",
            "202",
            "front_center",
            sync_packet_key=None,
            unity_frame_count=20,
            unity_time=2.0,
            source_bundle_active_transport_eligible=False,
            source_bundle_debug_unbundled_capture=True,
            camera_capture_contract_reason="no_source_bundle_context",
            arrival_wall_time=time.time(),
        )

        queue = srv.camera_frame_queues["front_center"]
        assert len(queue) == 1
        assert queue[-1]["source_bundle_debug_unbundled_capture"] is True
        assert srv.latest_sync_packet_data is None
        assert srv.latest_sync_packet_payload_data is None
        assert srv.sync_packet_active_camera_excluded_count == 1
        assert srv.sync_packet_unbundled_camera_entered_active_path_count == 0

        selection_meta = srv._build_selection_meta(
            consume_policy=PACKET_CONSUME_POLICY_FRESHEST_WITHIN_BUDGET,
            selection_source="server_selector",
            selected_age_ms=None,
            warn_age_ms=125.0,
            fresh=False,
            drained_count=0,
            stale_drop_count=0,
            max_drained_age_ms=None,
            fallback_active=True,
            fallback_reason_code=PACKET_FALLBACK_PAYLOAD_AGE_BUDGET_EXCEEDED,
            queue_depth_after_select=0,
            oldest_age_after_select_ms=None,
        )
        assert selection_meta["active_camera_excluded_event_delta"] == 1
        assert selection_meta["active_camera_excluded_reason_code"] == "no_source_bundle_context"
        assert selection_meta["unbundled_camera_entered_active_path_event_delta"] == 0

    asyncio.run(_run())


def test_sync_packet_snapshot_coalesces_source_bundle_fields_from_vehicle_when_front_missing(
    _reset_sync_globals,
):
    packet = {
        "packet_id": 32,
        "publish_wall_time": time.time(),
        "created_wall_time": time.time() - 0.01,
        "front_camera": {
            "unity_frame_count": 200,
            "unity_time": 4.0,
            "arrival_wall_time": time.time(),
        },
        "vehicle_state": {
            "unity_frame_count": 200,
            "unity_time": 4.0,
            "arrival_wall_time": time.time(),
            "source_bundle_camera_requested": True,
            "source_bundle_camera_request_attempted": True,
            "source_bundle_camera_request_accepted": True,
            "source_bundle_camera_request_rejected_reason": "",
            "source_bundle_camera_request_skipped_reason": "",
            "source_bundle_camera_request_disposition_code": "accepted",
            "source_bundle_camera_request_attempt_age_ms": 7.5,
            "source_bundle_camera_request_accept_age_ms": 5.0,
            "source_bundle_camera_request_queue_depth": 2,
            "source_bundle_camera_sent": True,
            "source_bundle_vehicle_state_built": True,
            "source_bundle_vehicle_state_enqueued": True,
            "source_bundle_vehicle_state_sent": True,
            "source_bundle_aborted_before_vehicle_send": False,
            "source_bundle_abort_reason": "",
            "source_bundle_vehicle_send_blocked_by_camera_request": False,
            "source_bundle_close_reason": "open",
            "source_bundle_deadline_ms": 50.0,
            "source_bundle_age_ms": 8.0,
            "source_bundle_inflight_count": 1,
            "source_bundle_superseded_before_send": False,
        },
    }

    snapshot = srv._snapshot_sync_packet(packet, now=time.time())

    assert snapshot["complete"] is True
    assert snapshot["source_bundle_camera_requested"] is True
    assert snapshot["source_bundle_camera_request_attempted"] is True
    assert snapshot["source_bundle_camera_request_accepted"] is True
    assert snapshot["source_bundle_camera_request_skipped_reason"] == ""
    assert snapshot["source_bundle_camera_request_disposition_code"] == "accepted"
    assert snapshot["source_bundle_camera_request_attempt_age_ms"] == pytest.approx(7.5)
    assert snapshot["source_bundle_camera_request_accept_age_ms"] == pytest.approx(5.0)
    assert snapshot["source_bundle_camera_request_queue_depth"] == 2
    assert snapshot["source_bundle_camera_sent"] is True
    assert snapshot["source_bundle_vehicle_state_sent"] is True
    assert snapshot["source_bundle_aborted_before_vehicle_send"] is False
    assert snapshot["source_bundle_abort_reason"] == ""
    assert snapshot["source_bundle_vehicle_send_blocked_by_camera_request"] is False
    assert snapshot["source_bundle_close_reason"] == "open"


def test_sync_packet_payload_response_keeps_snapshot_source_bundle_fields_when_selection_meta_defaults(
    _reset_sync_globals,
):
    packet = _payload_packet(packet_id=33, unity_frame_count=200, publish_wall_time=time.time())
    packet["front_camera"].update(
        {
            "source_bundle_camera_requested": True,
            "source_bundle_camera_request_attempted": True,
            "source_bundle_camera_request_accepted": True,
            "source_bundle_camera_request_queue_depth": 2,
            "source_bundle_active_transport_eligible": True,
            "source_bundle_debug_unbundled_capture": False,
            "camera_capture_contract_reason": "",
            "source_bundle_camera_sent": True,
            "source_bundle_vehicle_state_sent": True,
            "source_bundle_aborted_before_vehicle_send": False,
            "source_bundle_abort_reason": "",
            "source_bundle_vehicle_send_blocked_by_camera_request": False,
        }
    )

    response = srv._sync_packet_payload_response_with_selection(
        packet,
        {
            "selection_result": "complete_coherent",
            "selection_fallback_active": False,
            "selection_fallback_reason_code": "none",
            "source_bundle_camera_requested": None,
            "source_bundle_camera_request_attempted": None,
            "source_bundle_camera_request_accepted": None,
            "source_bundle_camera_request_skipped_reason": "",
            "source_bundle_camera_request_disposition_code": "",
            "source_bundle_camera_request_queue_depth": -1,
            "source_bundle_active_transport_eligible": None,
            "source_bundle_debug_unbundled_capture": None,
            "camera_capture_contract_reason": "",
            "source_bundle_camera_sent": None,
            "source_bundle_aborted_before_vehicle_send": None,
            "source_bundle_abort_reason": "",
            "source_bundle_vehicle_send_blocked_by_camera_request": None,
            "source_bundle_vehicle_state_sent": None,
            "active_camera_excluded_event_delta": 0,
            "active_camera_excluded_reason_code": "",
            "unbundled_camera_entered_active_path_event_delta": 0,
        },
    )

    class _FakeResponse:
        def __init__(self, r):
            self.headers = CaseInsensitiveDict(dict(r.headers))
            self.content = r.body

    from bridge.client import UnityBridgeClient

    parsed = UnityBridgeClient("http://unused")._parse_sync_packet_payload_response(
        _FakeResponse(response)
    )
    packet_meta = parsed["packet_meta"]

    assert packet_meta["source_bundle_camera_requested"] is True
    assert packet_meta["source_bundle_camera_request_attempted"] is True
    assert packet_meta["source_bundle_camera_request_accepted"] is True
    assert packet_meta["source_bundle_camera_request_skipped_reason"] == ""
    assert packet_meta["source_bundle_camera_request_disposition_code"] == ""
    assert packet_meta["source_bundle_camera_request_queue_depth"] == 2
    assert packet_meta["source_bundle_active_transport_eligible"] is True
    assert packet_meta["source_bundle_debug_unbundled_capture"] is False
    assert packet_meta["camera_capture_contract_reason"] == ""
    assert packet_meta["source_bundle_camera_sent"] is True
    assert packet_meta["source_bundle_aborted_before_vehicle_send"] is False
    assert packet_meta["source_bundle_abort_reason"] == ""
    assert packet_meta["source_bundle_vehicle_send_blocked_by_camera_request"] is False
    assert packet_meta["source_bundle_vehicle_state_sent"] is True


def test_get_next_sync_packet_raw_waits_for_late_vehicle_component(_reset_sync_globals):
    jpeg = _tiny_jpeg_bytes()

    async def _run() -> None:
        packet_key = "avbridge:50:200"
        await srv._decode_and_store_camera_frame(
            jpeg,
            "2.0",
            "200",
            "front_center",
            sync_packet_key=packet_key,
            unity_frame_count=200,
            unity_time=4.0,
            source_bundle_close_reason="open",
            source_bundle_camera_requested=True,
            source_bundle_camera_request_attempted=True,
            source_bundle_camera_request_accepted=True,
            source_bundle_camera_sent=True,
            source_bundle_vehicle_state_built=True,
            source_bundle_vehicle_state_enqueued=True,
            source_bundle_vehicle_state_sent=False,
            arrival_wall_time=time.time(),
        )

        async def _send_vehicle() -> None:
            await asyncio.sleep(0.01)
            await srv.receive_vehicle_state(
                _vehicle_state(200, 4.0, sync_packet_key=packet_key)
            )

        sender = asyncio.create_task(_send_vehicle())
        response = await srv.get_next_sync_packet_raw(
            consume_policy=PACKET_CONSUME_POLICY_FRESHEST_WITHIN_BUDGET,
            max_age_ms=250.0,
            warn_age_ms=125.0,
            max_drain_count=8,
            allow_stale_debug=False,
        )
        await sender

        assert response.headers["content-type"].startswith("multipart/mixed")
        assert response.headers["X-AV-Selection-Result"] in {"complete_coherent", "complete"}

    asyncio.run(_run())


def test_get_next_sync_packet_raw_404_uses_pending_partial_contract_reason(_reset_sync_globals):
    jpeg = _tiny_jpeg_bytes()

    async def _run() -> None:
        packet_key = "avbridge:51:201"
        await srv._decode_and_store_camera_frame(
            jpeg,
            "2.0",
            "201",
            "front_center",
            sync_packet_key=packet_key,
            unity_frame_count=201,
            unity_time=4.02,
            source_bundle_close_reason="open",
            source_bundle_camera_requested=True,
            source_bundle_camera_request_attempted=True,
            source_bundle_camera_request_accepted=True,
            source_bundle_camera_sent=True,
            source_bundle_vehicle_state_built=True,
            source_bundle_vehicle_state_enqueued=True,
            source_bundle_vehicle_state_sent=False,
            arrival_wall_time=time.time(),
        )

        with pytest.raises(srv.HTTPException) as exc_info:
            await srv.get_next_sync_packet_raw(
                consume_policy=PACKET_CONSUME_POLICY_FRESHEST_WITHIN_BUDGET,
                max_age_ms=250.0,
                warn_age_ms=125.0,
                max_drain_count=8,
                allow_stale_debug=False,
            )

        exc = exc_info.value
        assert exc.status_code == 404
        assert exc.headers["X-AV-Join-Failure-Reason"] == "vehicle_component_late"
        assert exc.headers["X-AV-Selected-Failure-Contract-Reason"] == PACKET_SELECTED_FAILURE_CONTRACT_VEHICLE_NOT_SENT
        assert exc.headers["X-AV-Selected-Failure-Source-Stage"] == PACKET_SELECTED_FAILURE_STAGE_SOURCE_BUNDLE
        assert exc.headers["X-AV-Source-Bundle-Close-Reason"] == "open"
        assert exc.headers["X-AV-Source-Bundle-Vehicle-State-Sent"] == "0"

    asyncio.run(_run())


def test_sync_packet_mismatched_keys_do_not_join_and_label_mismatch(_reset_sync_globals):
    jpeg = _tiny_jpeg_bytes()

    async def _run() -> None:
        await srv._decode_and_store_camera_frame(
            jpeg,
            "1.25",
            "101",
            "front_center",
            sync_packet_key="avbridge:101:123",
            unity_frame_count=123,
            unity_time=4.5,
        )
        await srv.receive_vehicle_state(
            _vehicle_state(
                123,
                4.5,
                sync_packet_key="avbridge:102:123",
            )
        )
        srv._evict_stale_partial_packets(140, time.time() + 2.0)
        assert srv.latest_sync_packet_join_failure_reason == PACKET_JOIN_FAILURE_PACKET_KEY_MISMATCH
        assert srv.latest_sync_packet_join_failure_side == PACKET_JOIN_FAILURE_SIDE_BOTH

    asyncio.run(_run())


def _payload_packet(packet_id: int, unity_frame_count: int, publish_wall_time: float) -> dict:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    return {
        "packet_id": packet_id,
        "schema_version": 1,
        "unity_frame_count": unity_frame_count,
        "unity_time": float(unity_frame_count) * 0.02,
        "created_wall_time": publish_wall_time - 0.01,
        "publish_wall_time": publish_wall_time,
        "front_camera": {
            "camera_id": "front_center",
            "frame_id": unity_frame_count,
            "timestamp": publish_wall_time,
            "unity_frame_count": unity_frame_count,
            "unity_time": float(unity_frame_count) * 0.02,
            "arrival_wall_time": publish_wall_time,
            "image": image,
        },
        "vehicle_state": {
            "unity_frame_count": unity_frame_count,
            "unity_time": float(unity_frame_count) * 0.02,
            "request_id": packet_id,
            "arrival_wall_time": publish_wall_time,
            "state": {"unityFrameCount": unity_frame_count, "speed": 5.0},
        },
    }


def test_sync_packet_next_raw_selects_freshest_with_server_side_drain(_reset_sync_globals):
    async def _run() -> None:
        now = 100.0
        srv.assembled_packet_payload_queue.append(_payload_packet(1, 10, now - 0.42))
        srv.assembled_packet_payload_queue.append(_payload_packet(2, 11, now - 0.18))
        srv.assembled_packet_payload_queue.append(_payload_packet(3, 12, now - 0.04))

        original_time = srv.time.time
        srv.time.time = lambda: now
        try:
            response = await srv.get_next_sync_packet_raw(
                consume_policy=PACKET_CONSUME_POLICY_FRESHEST_WITHIN_BUDGET,
                max_age_ms=250.0,
                warn_age_ms=125.0,
                max_drain_count=8,
            )
        finally:
            srv.time.time = original_time

        assert response.headers["X-AV-Unity-Frame-Count"] == "12"
        assert response.headers["X-AV-Selection-Source"] == "server_selector"
        assert response.headers["X-AV-Stale-Drop-Count"] == "2"
        assert response.headers["X-AV-Drained-Count"] == "2"
        assert response.headers["X-AV-Server-Queue-Depth-After-Select"] == "0"
        assert len(srv.assembled_packet_payload_queue) == 0

    asyncio.run(_run())


def test_sync_packet_next_raw_returns_selection_headers_on_stale_timeout(_reset_sync_globals):
    async def _run() -> None:
        now = 100.0
        srv.assembled_packet_payload_queue.append(_payload_packet(4, 20, now - 0.61))
        srv.assembled_packet_payload_queue.append(_payload_packet(5, 21, now - 0.52))

        original_time = srv.time.time
        srv.time.time = lambda: now
        try:
            with pytest.raises(srv.HTTPException) as exc_info:
                await srv.get_next_sync_packet_raw(
                    consume_policy=PACKET_CONSUME_POLICY_FRESHEST_WITHIN_BUDGET,
                    max_age_ms=250.0,
                    warn_age_ms=125.0,
                    max_drain_count=8,
                )
        finally:
            srv.time.time = original_time

        exc = exc_info.value
        assert exc.status_code == 404
        assert exc.headers["X-AV-Selection-Fallback-Reason"] == PACKET_FALLBACK_PAYLOAD_AGE_BUDGET_EXCEEDED
        assert exc.headers["X-AV-Selection-Fallback-Active"] == "1"
        assert exc.headers["X-AV-Stale-Drop-Count"] == "1"

    asyncio.run(_run())


def test_sync_packet_next_raw_rejects_complete_but_incoherent_packet(_reset_sync_globals):
    async def _run() -> None:
        now = 100.0
        packet = _payload_packet(6, 30, now - 0.04)
        packet["front_camera"]["unity_frame_count"] = 38
        packet["front_camera"]["unity_time"] = 1.24
        packet["front_camera"]["arrival_wall_time"] = now - 0.04
        packet["vehicle_state"]["unity_frame_count"] = 30
        packet["vehicle_state"]["unity_time"] = 1.20
        packet["vehicle_state"]["arrival_wall_time"] = now - 0.04
        packet["created_wall_time"] = now - 0.65
        srv.assembled_packet_payload_queue.append(packet)

        original_time = srv.time.time
        srv.time.time = lambda: now
        try:
            with pytest.raises(srv.HTTPException) as exc_info:
                await srv.get_next_sync_packet_raw(
                    consume_policy=PACKET_CONSUME_POLICY_FRESHEST_WITHIN_BUDGET,
                    max_age_ms=250.0,
                    warn_age_ms=125.0,
                    max_drain_count=8,
                )
        finally:
            srv.time.time = original_time

        exc = exc_info.value
        assert exc.status_code == 404
        assert (
            exc.headers["X-AV-Selection-Fallback-Reason"]
            == PACKET_COHERENCE_REASON_FRONT_VEHICLE_FRAME_DELTA_BUDGET_EXCEEDED
        )
        assert exc.headers["X-AV-Selection-Result"] == PACKET_SELECTION_RESULT_COMPLETE_INCOHERENT
        assert exc.headers["X-AV-Selection-Fallback-Active"] == "1"

    asyncio.run(_run())
