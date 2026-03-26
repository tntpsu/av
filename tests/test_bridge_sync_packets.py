"""Tests for shadow synchronized packet assembly on the bridge."""

from __future__ import annotations

import asyncio
import io

import numpy as np
import pytest
from PIL import Image

import bridge.server as srv
from sync_contract import (
    PACKET_CONSUME_POLICY_FRESHEST_WITHIN_BUDGET,
    PACKET_FALLBACK_PAYLOAD_AGE_BUDGET_EXCEEDED,
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
    srv.sync_packet_drop_count = 0
    srv.sync_packet_payload_drop_count = 0
    srv.sync_packet_orphan_camera_count = 0
    srv.sync_packet_orphan_vehicle_count = 0
    srv.sync_packet_timeout_count = 0
    srv.sync_packet_key_match_count = 0
    srv.sync_packet_unity_fallback_count = 0
    srv.sync_packet_superseded_camera_count = 0
    srv.sync_packet_superseded_vehicle_count = 0
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
