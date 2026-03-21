"""Tests for bridge consumer policy: queue drops, drain_queue, health queue size."""

from __future__ import annotations

import asyncio
import io
from collections import deque

import numpy as np
import pytest
from PIL import Image

import bridge.server as srv


def _tiny_jpeg_bytes() -> bytes:
    img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def _reset_camera_globals():
    srv.latest_camera_frames.clear()
    srv.latest_camera_frame = None
    srv.latest_frame_timestamps_by_id.clear()
    srv.latest_frame_ids_by_id.clear()
    srv.camera_frame_queues.clear()
    srv.camera_drop_count.clear()
    yield
    srv.latest_camera_frames.clear()
    srv.latest_camera_frame = None
    srv.latest_frame_timestamps_by_id.clear()
    srv.latest_frame_ids_by_id.clear()
    srv.camera_frame_queues.clear()
    srv.camera_drop_count.clear()


def test_decode_increments_drop_count_when_queue_full(monkeypatch, _reset_camera_globals):
    monkeypatch.setattr(srv, "MAX_CAMERA_QUEUE_SIZE", 2)
    jpeg = _tiny_jpeg_bytes()

    async def _run() -> None:
        for i in range(4):
            await srv._decode_and_store_camera_frame(jpeg, "1.0", str(100 + i), "front_center")

    asyncio.run(_run())
    assert srv.camera_drop_count.get("front_center", 0) == 2
    q = srv.camera_frame_queues["front_center"]
    assert len(q) == 2


def test_drain_queue_post_clears_fifo(_reset_camera_globals):
    arr = np.ones((2, 2, 3), dtype=np.uint8)
    q = deque(maxlen=10)
    q.append({"image": arr, "timestamp": 1.0, "frame_id": "1", "camera_id": "front_center"})
    srv.camera_frame_queues["front_center"] = q

    body = asyncio.run(srv.drain_camera_queue(camera_id="front_center"))
    assert body["drained"] == 1
    assert body["camera_id"] == "front_center"
    assert len(srv.camera_frame_queues["front_center"]) == 0


def test_health_includes_max_camera_queue_size():
    data = asyncio.run(srv.health_check())
    assert "max_camera_queue_size" in data
    assert int(data["max_camera_queue_size"]) >= 4


def test_parse_stack_yaml_has_ingest_keys():
    from pathlib import Path

    import yaml

    p = Path(__file__).resolve().parents[1] / "config" / "av_stack_config.yaml"
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    stack = d.get("stack") or {}
    assert "bridge_max_camera_queue" in stack
    assert "camera_prefetch" in stack
    assert "clear_bridge_camera_queue_on_start" in stack
