"""Tests for raw RGB camera transport on the bridge."""

from __future__ import annotations

import asyncio
from collections import deque

import numpy as np
import pytest
from fastapi import HTTPException

import bridge.server as srv
from bridge.client import UnityBridgeClient
from bridge.server import MAX_CAMERA_QUEUE_SIZE, get_latest_camera_frame_raw, get_next_camera_frame_raw


@pytest.fixture
def _reset_front_camera() -> None:
    srv.latest_camera_frames.clear()
    srv.latest_camera_frame = None
    srv.latest_frame_timestamps_by_id.clear()
    srv.latest_frame_ids_by_id.clear()
    srv.camera_frame_queues.pop("front_center", None)


def test_latest_raw_round_trip(_reset_front_camera: None) -> None:
    arr = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
    srv.latest_camera_frames["front_center"] = arr.copy()
    srv.latest_frame_timestamps_by_id["front_center"] = 9.875
    srv.latest_frame_ids_by_id["front_center"] = "42"

    r = asyncio.run(get_latest_camera_frame_raw(camera_id="front_center"))
    assert r.status_code == 200
    assert r.headers.get("X-AV-Shape") == "4,5,3"
    assert r.headers.get("X-AV-Timestamp") == "9.875"
    assert r.headers.get("X-AV-Frame-Id") == "42"
    assert len(r.body) == 60
    out = np.frombuffer(r.body, dtype=np.uint8).reshape(4, 5, 3)
    np.testing.assert_array_equal(out, arr)


def test_latest_raw_404_when_empty(_reset_front_camera: None) -> None:
    with pytest.raises(HTTPException) as exc:
        asyncio.run(get_latest_camera_frame_raw())
    assert exc.value.status_code == 404


def test_next_raw_pops_queue(_reset_front_camera: None) -> None:
    arr = np.zeros((2, 3, 3), dtype=np.uint8, order="C")
    arr[0, 0, 0] = 99
    q = deque(maxlen=MAX_CAMERA_QUEUE_SIZE)
    q.append({"image": arr.copy(), "timestamp": 2.5, "frame_id": "9"})
    srv.camera_frame_queues["front_center"] = q

    r = asyncio.run(get_next_camera_frame_raw(camera_id="front_center"))
    assert r.status_code == 200
    assert r.headers.get("X-AV-Frame-Id") == "9"
    assert r.headers.get("X-AV-Queue-Remaining") == "0"
    out = np.frombuffer(r.body, dtype=np.uint8).reshape(2, 3, 3)
    assert out[0, 0, 0] == 99
    assert len(q) == 0


def test_parse_raw_camera_response() -> None:
    arr = np.ones((2, 2, 3), dtype=np.uint8) * 200
    body = arr.tobytes()
    hdr = {
        "X-AV-Shape": "2,2,3",
        "X-AV-Timestamp": "1.5",
        "X-AV-Frame-Id": "3",
        "X-AV-Queue-Depth": "5",
        "X-AV-Queue-Capacity": "120",
        "X-AV-Drop-Count": "0",
        "X-AV-Decode-In-Flight": "0",
    }

    class _Resp:
        pass

    r = _Resp()
    r.content = body
    r.headers = hdr

    img, ts, fid, meta = UnityBridgeClient._parse_raw_camera_response(r)
    np.testing.assert_array_equal(img, arr)
    assert ts == 1.5
    assert fid == 3
    assert meta["queue_depth"] == 5
    assert meta["decode_in_flight"] is False
