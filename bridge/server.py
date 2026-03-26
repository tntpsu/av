"""
FastAPI server for Unity-Python communication bridge.
Handles camera frames, vehicle state, and control commands.
"""

import asyncio
import base64
import io
import json
import os
import time
import logging
import math
from collections import deque
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from PIL import Image
from starlette.responses import Response

from sync_contract import (
    PACKET_COHERENCE_REASON_COHERENT,
    PACKET_COHERENCE_REASON_COMPONENT_AGE_BUDGET_EXCEEDED,
    PACKET_COHERENCE_REASON_FRONT_VEHICLE_FRAME_DELTA_BUDGET_EXCEEDED,
    PACKET_COHERENCE_REASON_FRONT_VEHICLE_TIME_DELTA_BUDGET_EXCEEDED,
    PACKET_COHERENCE_REASON_JOIN_WAIT_BUDGET_EXCEEDED,
    PACKET_COHERENCE_REASON_MISSING_COMPONENT_TIMESTAMP,
    PACKET_CONSUME_POLICY_FIFO_STRICT,
    PACKET_CONSUME_POLICY_FRESHEST_WITHIN_BUDGET,
    PACKET_CONSUME_POLICY_LATEST_DEBUG,
    PACKET_FALLBACK_NONE,
    PACKET_FALLBACK_PACKET_TIMEOUT,
    PACKET_FALLBACK_PAYLOAD_AGE_BUDGET_EXCEEDED,
    PACKET_FALLBACK_PAYLOAD_DRAIN_CAP,
    PACKET_MODE_PACKET_SHADOW,
    PACKET_JOIN_SOURCE_NONE,
    PACKET_JOIN_SOURCE_PACKET_KEY,
    PACKET_JOIN_SOURCE_UNITY_FRAME_FALLBACK,
    PACKET_JOIN_FAILURE_CAMERA_COMPONENT_LATE,
    PACKET_JOIN_FAILURE_CAMERA_KEY_MISSING,
    PACKET_JOIN_FAILURE_JOIN_WINDOW_EXPIRED,
    PACKET_JOIN_FAILURE_NONE,
    PACKET_JOIN_FAILURE_PACKET_KEY_MISMATCH,
    PACKET_JOIN_FAILURE_SIDE_BOTH,
    PACKET_JOIN_FAILURE_SIDE_CAMERA,
    PACKET_JOIN_FAILURE_SIDE_NONE,
    PACKET_JOIN_FAILURE_SIDE_VEHICLE,
    PACKET_JOIN_FAILURE_SOURCE_BUNDLE_ABORTED,
    PACKET_JOIN_FAILURE_VEHICLE_COMPONENT_LATE,
    PACKET_JOIN_FAILURE_VEHICLE_KEY_MISSING,
    PACKET_SELECTED_FAILURE_CONTRACT_BUNDLE_ABORTED_CAMERA_CAPTURE_MISSING,
    PACKET_SELECTED_FAILURE_CONTRACT_BUNDLE_ABORTED_CAMERA_REQUEST_BUDGET_EXCEEDED,
    PACKET_SELECTED_FAILURE_CONTRACT_BUNDLE_ABORTED_CAMERA_REQUEST_NOT_ATTEMPTED,
    PACKET_SELECTED_FAILURE_CONTRACT_BUNDLE_ABORTED_CAMERA_REQUEST_REJECTED,
    PACKET_SELECTED_FAILURE_CONTRACT_BRIDGE_JOIN_FAILURE,
    PACKET_SELECTED_FAILURE_CONTRACT_CAMERA_NOT_REQUESTED,
    PACKET_SELECTED_FAILURE_CONTRACT_CAMERA_NOT_SENT,
    PACKET_SELECTED_FAILURE_CONTRACT_CAMERA_REQUEST_EXPIRED_BEFORE_CAPTURE,
    PACKET_SELECTED_FAILURE_CONTRACT_CAMERA_REQUEST_NOT_ATTEMPTED,
    PACKET_SELECTED_FAILURE_CONTRACT_CAMERA_REQUEST_REJECTED,
    PACKET_SELECTED_FAILURE_CONTRACT_CAMERA_REQUESTED_NOT_SENT,
    PACKET_SELECTED_FAILURE_CONTRACT_NONE,
    PACKET_SELECTED_FAILURE_CONTRACT_SOURCE_BUNDLE_DEADLINE_EXPIRED,
    PACKET_SELECTED_FAILURE_CONTRACT_VEHICLE_NOT_BUILT,
    PACKET_SELECTED_FAILURE_CONTRACT_VEHICLE_NOT_ENQUEUED,
    PACKET_SELECTED_FAILURE_CONTRACT_VEHICLE_NOT_SENT,
    PACKET_SELECTED_FAILURE_CONTRACT_VEHICLE_SUPERSEDED_BEFORE_SEND,
    PACKET_SELECTED_FAILURE_STAGE_BRIDGE_JOIN,
    PACKET_SELECTED_FAILURE_STAGE_CAMERA_CAPTURE,
    PACKET_SELECTED_FAILURE_STAGE_NONE,
    PACKET_SELECTED_FAILURE_STAGE_SOURCE_BUNDLE,
    PACKET_SELECTION_RESULT_COMPLETE,
    PACKET_SELECTION_RESULT_COMPLETE_COHERENT,
    PACKET_SELECTION_RESULT_COMPLETE_INCOHERENT,
    PACKET_SELECTION_RESULT_FALLBACK,
    PACKET_SELECTION_RESULT_LEGACY,
    PACKET_SELECTION_RESULT_NONE,
    PACKET_SELECTION_SOURCE_LATEST_DEBUG,
    PACKET_SELECTION_SOURCE_NONE,
    PACKET_SELECTION_SOURCE_SERVER,
    SYNC_PACKET_SCHEMA_VERSION,
)

app = FastAPI(title="AV Stack Bridge Server")

# Log slow requests to identify Unity↔Python stalls.
SLOW_REQUEST_SECONDS = 0.05
UNITY_TIME_GAP_SECONDS = 0.2


def _get_bridge_logger() -> logging.Logger:
    log_path = Path(__file__).resolve().parents[1] / "tmp" / "logs" / "av_bridge.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    bridge_logger = logging.getLogger("av_bridge")
    bridge_logger.setLevel(logging.INFO)

    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path)
               for h in bridge_logger.handlers):
        handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        bridge_logger.addHandler(handler)
        bridge_logger.propagate = False

    return bridge_logger


logger = _get_bridge_logger()

# Enable CORS for Unity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
latest_camera_frame: Optional[np.ndarray] = None
latest_frame_timestamp: Optional[float] = None
latest_frame_id: Optional[str] = None
latest_camera_frames: dict[str, np.ndarray] = {}
latest_frame_timestamps_by_id: dict[str, float] = {}
latest_frame_ids_by_id: dict[str, str] = {}
camera_frame_queues: dict[str, deque] = {}


def _read_max_camera_queue_size() -> int:
    """Cap Unity→bridge frame buffer depth (smaller = fresher frames, more deque drops)."""
    raw = os.environ.get("AV_BRIDGE_MAX_CAMERA_QUEUE", "").strip()
    if raw.isdigit():
        v = int(raw)
        return max(4, min(v, 500))
    return 120


MAX_CAMERA_QUEUE_SIZE = _read_max_camera_queue_size()
MAX_SYNC_PACKET_PAYLOAD_QUEUE_SIZE = 64


def _read_nonnegative_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return int(default)
    return max(0, value)


def _read_nonnegative_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(value):
        return float(default)
    return max(0.0, float(value))


SYNC_PACKET_MAX_FRONT_VEHICLE_TIME_DELTA_MS = _read_nonnegative_float_env(
    "AV_SYNC_PACKET_MAX_FRONT_VEHICLE_TIME_DELTA_MS",
    100.0,
)
SYNC_PACKET_MAX_FRONT_VEHICLE_FRAME_DELTA = _read_nonnegative_int_env(
    "AV_SYNC_PACKET_MAX_FRONT_VEHICLE_FRAME_DELTA",
    4,
)
SYNC_PACKET_MAX_JOIN_WAIT_MS = _read_nonnegative_float_env(
    "AV_SYNC_PACKET_MAX_JOIN_WAIT_MS",
    100.0,
)
SYNC_PACKET_MAX_COMPONENT_AGE_MS = _read_nonnegative_float_env(
    "AV_SYNC_PACKET_MAX_COMPONENT_AGE_MS",
    150.0,
)

vehicle_state_queue: deque = deque(maxlen=600)
latest_vehicle_state: Optional[dict] = None
latest_control_command: Optional[dict] = None
latest_trajectory_data: Optional[dict] = None  # Trajectory data for visualization
latest_unity_feedback: Optional[dict] = None  # Unity feedback/status data
shutdown_requested: bool = False  # Track if AV stack is shutting down
play_requested: bool = False  # Track if Unity should start playing
camera_decode_in_flight: dict[str, bool] = {}
camera_drop_count: dict[str, int] = {}
last_camera_arrival_time: dict[str, Optional[float]] = {}
last_camera_frame_id: dict[str, Optional[str]] = {}
last_camera_timestamp: dict[str, Optional[float]] = {}
last_camera_realtime: dict[str, Optional[float]] = {}
last_camera_unscaled: dict[str, Optional[float]] = {}
last_state_arrival_time: Optional[float] = None
last_unity_send_realtime: Optional[float] = None
last_unity_time: Optional[float] = None
last_control_arrival_time: Optional[float] = None
speed_limit_zero_streak: int = 0
vehicle_state_sanitize_events: int = 0
vehicle_state_sanitize_values: int = 0
pending_packets_by_unity_frame: dict[int, dict[str, Any]] = {}
pending_packets_by_packet_key: dict[str, dict[str, Any]] = {}
assembled_packet_queue: deque = deque(maxlen=600)
assembled_packet_payload_queue: deque = deque(maxlen=MAX_SYNC_PACKET_PAYLOAD_QUEUE_SIZE)
latest_sync_packet_data: Optional[dict[str, Any]] = None
latest_sync_packet_payload_data: Optional[dict[str, Any]] = None
sync_packet_drop_count: int = 0
sync_packet_payload_drop_count: int = 0
sync_packet_orphan_camera_count: int = 0
sync_packet_orphan_vehicle_count: int = 0
sync_packet_timeout_count: int = 0
sync_packet_partial_publish_count: int = 0
sync_packet_key_match_count: int = 0
sync_packet_unity_fallback_count: int = 0
sync_packet_superseded_camera_count: int = 0
sync_packet_superseded_vehicle_count: int = 0
sync_packet_id_counter: int = 0
latest_sync_packet_join_failure_reason: str = PACKET_JOIN_FAILURE_NONE
latest_sync_packet_join_failure_side: str = PACKET_JOIN_FAILURE_SIDE_NONE
latest_sync_packet_join_failure_source_key_present_camera: bool = False
latest_sync_packet_join_failure_source_key_present_vehicle: bool = False
sync_packet_join_failure_event_count: int = 0
sync_packet_active_camera_excluded_count: int = 0
sync_packet_active_camera_excluded_pending_count: int = 0
latest_sync_packet_active_camera_excluded_reason: str = ""
sync_packet_unbundled_camera_entered_active_path_count: int = 0
sync_packet_unbundled_camera_entered_active_path_pending_count: int = 0

PARTIAL_PACKET_MAX_AGE_SECONDS = 1.0
PARTIAL_PACKET_MAX_FRAME_LAG = 6
ACTIVE_SELECTION_PARTIAL_WAIT_MS = _read_nonnegative_float_env(
    "AV_SYNC_PACKET_ACTIVE_SELECTION_PARTIAL_WAIT_MS",
    35.0,
)
ACTIVE_SELECTION_PARTIAL_POLL_MS = _read_nonnegative_float_env(
    "AV_SYNC_PACKET_ACTIVE_SELECTION_PARTIAL_POLL_MS",
    5.0,
)


def _sanitize_json_compatible(value: Any) -> tuple[Any, int]:
    """Convert non-finite numeric payload values to JSON-safe nulls."""
    replacements = 0

    def _walk(obj: Any) -> Any:
        nonlocal replacements
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_walk(v) for v in obj]
        if isinstance(obj, tuple):
            return [_walk(v) for v in obj]
        if isinstance(obj, (float, np.floating)):
            fval = float(obj)
            if not math.isfinite(fval):
                replacements += 1
                return None
            return fval
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    return _walk(value), replacements


def _record_vehicle_state_sanitization(replacements: int, context: str) -> None:
    global vehicle_state_sanitize_events, vehicle_state_sanitize_values
    if replacements <= 0:
        return
    vehicle_state_sanitize_events += 1
    vehicle_state_sanitize_values += replacements
    logger.warning(
        "[SANITIZE] context=%s replacements=%d total_events=%d total_values=%d",
        context,
        replacements,
        vehicle_state_sanitize_events,
        vehicle_state_sanitize_values,
    )


def _normalize_camera_id(camera_id: Optional[str]) -> str:
    return camera_id or "front_center"


def _parse_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _parse_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_packet_key(packet_key: Any) -> Optional[str]:
    if packet_key is None:
        return None
    normalized = str(packet_key).strip()
    return normalized or None


def _component_has_packet_key(component: Optional[dict[str, Any]]) -> bool:
    if not isinstance(component, dict):
        return False
    return _normalize_packet_key(component.get("sync_packet_key")) is not None


def _record_excluded_camera_component(component: Optional[dict[str, Any]]) -> None:
    global sync_packet_active_camera_excluded_count
    global sync_packet_active_camera_excluded_pending_count
    global latest_sync_packet_active_camera_excluded_reason

    if not isinstance(component, dict):
        return
    sync_packet_active_camera_excluded_count += 1
    sync_packet_active_camera_excluded_pending_count += 1
    latest_sync_packet_active_camera_excluded_reason = str(
        component.get("camera_capture_contract_reason") or "debug_unbundled_capture"
    )


def _consume_camera_admission_events() -> tuple[int, str, int]:
    global sync_packet_active_camera_excluded_pending_count
    global latest_sync_packet_active_camera_excluded_reason
    global sync_packet_unbundled_camera_entered_active_path_pending_count

    excluded_delta = int(sync_packet_active_camera_excluded_pending_count)
    excluded_reason = str(latest_sync_packet_active_camera_excluded_reason or "")
    unbundled_active_delta = int(sync_packet_unbundled_camera_entered_active_path_pending_count)
    sync_packet_active_camera_excluded_pending_count = 0
    latest_sync_packet_active_camera_excluded_reason = ""
    sync_packet_unbundled_camera_entered_active_path_pending_count = 0
    return excluded_delta, excluded_reason, unbundled_active_delta


def _classify_join_failure(
    packet: dict[str, Any],
    packets_snapshot: list[dict[str, Any]],
) -> tuple[str, str, bool, bool]:
    front = packet.get("front_camera")
    vehicle = packet.get("vehicle_state")
    front_key_present = _component_has_packet_key(front)
    vehicle_key_present = _component_has_packet_key(vehicle)
    packet_unity_frame_count = _parse_optional_int(packet.get("unity_frame_count"))
    packet_key = _normalize_packet_key(packet.get("packet_key"))

    if front is not None and vehicle is None:
        if not front_key_present:
            return (
                PACKET_JOIN_FAILURE_CAMERA_KEY_MISSING,
                PACKET_JOIN_FAILURE_SIDE_CAMERA,
                front_key_present,
                vehicle_key_present,
            )
        if packet_unity_frame_count is not None:
            for other in packets_snapshot:
                if other is packet:
                    continue
                if _parse_optional_int(other.get("unity_frame_count")) != packet_unity_frame_count:
                    continue
                if other.get("vehicle_state") is None:
                    continue
                other_key = _normalize_packet_key(other.get("packet_key"))
                if other_key is not None and packet_key is not None and other_key != packet_key:
                    return (
                        PACKET_JOIN_FAILURE_PACKET_KEY_MISMATCH,
                        PACKET_JOIN_FAILURE_SIDE_BOTH,
                        front_key_present,
                        vehicle_key_present,
                    )
        return (
            PACKET_JOIN_FAILURE_VEHICLE_COMPONENT_LATE,
            PACKET_JOIN_FAILURE_SIDE_VEHICLE,
            front_key_present,
            vehicle_key_present,
        )

    if vehicle is not None and front is None:
        if bool(vehicle.get("source_bundle_aborted_before_vehicle_send", False)):
            return (
                PACKET_JOIN_FAILURE_SOURCE_BUNDLE_ABORTED,
                PACKET_JOIN_FAILURE_SIDE_NONE,
                front_key_present,
                vehicle_key_present,
            )
        if not vehicle_key_present:
            return (
                PACKET_JOIN_FAILURE_VEHICLE_KEY_MISSING,
                PACKET_JOIN_FAILURE_SIDE_VEHICLE,
                front_key_present,
                vehicle_key_present,
            )
        if packet_unity_frame_count is not None:
            for other in packets_snapshot:
                if other is packet:
                    continue
                if _parse_optional_int(other.get("unity_frame_count")) != packet_unity_frame_count:
                    continue
                if other.get("front_camera") is None:
                    continue
                other_key = _normalize_packet_key(other.get("packet_key"))
                if other_key is not None and packet_key is not None and other_key != packet_key:
                    return (
                        PACKET_JOIN_FAILURE_PACKET_KEY_MISMATCH,
                        PACKET_JOIN_FAILURE_SIDE_BOTH,
                        front_key_present,
                        vehicle_key_present,
                    )
        return (
            PACKET_JOIN_FAILURE_CAMERA_COMPONENT_LATE,
            PACKET_JOIN_FAILURE_SIDE_CAMERA,
            front_key_present,
            vehicle_key_present,
        )

    return (
        PACKET_JOIN_FAILURE_JOIN_WINDOW_EXPIRED,
        PACKET_JOIN_FAILURE_SIDE_NONE,
        front_key_present,
        vehicle_key_present,
    )


def _classify_selected_failure_contract(
    packet: dict[str, Any],
    *,
    join_failure_reason_code: str,
    join_failure_side_code: str,
) -> tuple[str, str]:
    front = packet.get("front_camera") or {}
    vehicle = packet.get("vehicle_state") or {}

    def _abort_contract(component: dict[str, Any]) -> Optional[tuple[str, str]]:
        if not bool(component.get("source_bundle_aborted_before_vehicle_send", False)):
            return None
        abort_reason = str(component.get("source_bundle_abort_reason") or "")
        if abort_reason == "bundle_aborted_camera_request_not_attempted":
            return (
                PACKET_SELECTED_FAILURE_CONTRACT_BUNDLE_ABORTED_CAMERA_REQUEST_NOT_ATTEMPTED,
                PACKET_SELECTED_FAILURE_STAGE_SOURCE_BUNDLE,
            )
        if abort_reason == "bundle_aborted_camera_request_rejected":
            return (
                PACKET_SELECTED_FAILURE_CONTRACT_BUNDLE_ABORTED_CAMERA_REQUEST_REJECTED,
                PACKET_SELECTED_FAILURE_STAGE_SOURCE_BUNDLE,
            )
        if abort_reason == "bundle_aborted_camera_request_budget_exceeded":
            return (
                PACKET_SELECTED_FAILURE_CONTRACT_BUNDLE_ABORTED_CAMERA_REQUEST_BUDGET_EXCEEDED,
                PACKET_SELECTED_FAILURE_STAGE_SOURCE_BUNDLE,
            )
        if abort_reason == "bundle_aborted_camera_capture_missing":
            return (
                PACKET_SELECTED_FAILURE_CONTRACT_BUNDLE_ABORTED_CAMERA_CAPTURE_MISSING,
                PACKET_SELECTED_FAILURE_STAGE_SOURCE_BUNDLE,
            )
        return None

    if front and not vehicle:
        abort_contract = _abort_contract(front)
        if abort_contract is not None:
            return abort_contract
        if not bool(front.get("source_bundle_camera_request_attempted", False)):
            return (
                PACKET_SELECTED_FAILURE_CONTRACT_CAMERA_REQUEST_NOT_ATTEMPTED,
                PACKET_SELECTED_FAILURE_STAGE_SOURCE_BUNDLE,
            )
        if bool(front.get("source_bundle_superseded_before_send", False)):
            return (
                PACKET_SELECTED_FAILURE_CONTRACT_VEHICLE_SUPERSEDED_BEFORE_SEND,
                PACKET_SELECTED_FAILURE_STAGE_SOURCE_BUNDLE,
            )
        if not bool(front.get("source_bundle_vehicle_state_built", False)):
            return (
                PACKET_SELECTED_FAILURE_CONTRACT_VEHICLE_NOT_BUILT,
                PACKET_SELECTED_FAILURE_STAGE_SOURCE_BUNDLE,
            )
        if not bool(front.get("source_bundle_vehicle_state_enqueued", False)):
            return (
                PACKET_SELECTED_FAILURE_CONTRACT_VEHICLE_NOT_ENQUEUED,
                PACKET_SELECTED_FAILURE_STAGE_SOURCE_BUNDLE,
            )
        close_reason = str(front.get("source_bundle_close_reason") or "")
        if close_reason == PACKET_SELECTED_FAILURE_CONTRACT_SOURCE_BUNDLE_DEADLINE_EXPIRED:
            return (
                PACKET_SELECTED_FAILURE_CONTRACT_SOURCE_BUNDLE_DEADLINE_EXPIRED,
                PACKET_SELECTED_FAILURE_STAGE_SOURCE_BUNDLE,
            )
        if not bool(front.get("source_bundle_vehicle_state_sent", False)):
            return (
                PACKET_SELECTED_FAILURE_CONTRACT_VEHICLE_NOT_SENT,
                PACKET_SELECTED_FAILURE_STAGE_SOURCE_BUNDLE,
            )

    if vehicle and not front:
        abort_contract = _abort_contract(vehicle)
        if abort_contract is not None:
            return abort_contract
        if not bool(vehicle.get("source_bundle_camera_request_attempted", False)):
            return (
                PACKET_SELECTED_FAILURE_CONTRACT_CAMERA_REQUEST_NOT_ATTEMPTED,
                PACKET_SELECTED_FAILURE_STAGE_SOURCE_BUNDLE,
            )
        if not bool(vehicle.get("source_bundle_camera_request_accepted", False)):
            return (
                PACKET_SELECTED_FAILURE_CONTRACT_CAMERA_REQUEST_REJECTED,
                PACKET_SELECTED_FAILURE_STAGE_CAMERA_CAPTURE,
            )
        close_reason = str(vehicle.get("source_bundle_close_reason") or "")
        if close_reason in (
            PACKET_SELECTED_FAILURE_CONTRACT_SOURCE_BUNDLE_DEADLINE_EXPIRED,
            "camera_request_deadline_expired",
            "camera_capture_deadline_expired",
        ):
            return (
                PACKET_SELECTED_FAILURE_CONTRACT_CAMERA_REQUEST_EXPIRED_BEFORE_CAPTURE,
                PACKET_SELECTED_FAILURE_STAGE_SOURCE_BUNDLE,
            )
        if not bool(vehicle.get("source_bundle_camera_sent", False)):
            return (
                PACKET_SELECTED_FAILURE_CONTRACT_CAMERA_REQUESTED_NOT_SENT,
                PACKET_SELECTED_FAILURE_STAGE_CAMERA_CAPTURE,
            )

    if join_failure_reason_code and join_failure_reason_code != PACKET_JOIN_FAILURE_NONE:
        return (
            PACKET_SELECTED_FAILURE_CONTRACT_BRIDGE_JOIN_FAILURE,
            PACKET_SELECTED_FAILURE_STAGE_BRIDGE_JOIN,
        )

    return (PACKET_SELECTED_FAILURE_CONTRACT_NONE, PACKET_SELECTED_FAILURE_STAGE_NONE)


def _classify_packet_coherence(
    *,
    complete: bool,
    front_vehicle_time_delta_ms: Optional[float],
    front_vehicle_frame_delta: Optional[int],
    join_wait_ms: Optional[float],
    front_age_ms: Optional[float],
    vehicle_age_ms: Optional[float],
) -> tuple[bool, str, bool, bool, bool, bool]:
    if not complete:
        return (
            False,
            PACKET_COHERENCE_REASON_MISSING_COMPONENT_TIMESTAMP,
            False,
            False,
            False,
            False,
        )

    if (
        front_vehicle_time_delta_ms is None
        or front_vehicle_frame_delta is None
        or join_wait_ms is None
        or front_age_ms is None
        or vehicle_age_ms is None
    ):
        return (
            False,
            PACKET_COHERENCE_REASON_MISSING_COMPONENT_TIMESTAMP,
            False,
            False,
            False,
            False,
        )

    time_delta_exceeded = abs(float(front_vehicle_time_delta_ms)) > float(
        SYNC_PACKET_MAX_FRONT_VEHICLE_TIME_DELTA_MS
    )
    frame_delta_exceeded = abs(int(front_vehicle_frame_delta)) > int(
        SYNC_PACKET_MAX_FRONT_VEHICLE_FRAME_DELTA
    )
    join_wait_exceeded = float(join_wait_ms) > float(SYNC_PACKET_MAX_JOIN_WAIT_MS)
    component_age_exceeded = max(float(front_age_ms), float(vehicle_age_ms)) > float(
        SYNC_PACKET_MAX_COMPONENT_AGE_MS
    )

    if time_delta_exceeded:
        return (
            False,
            PACKET_COHERENCE_REASON_FRONT_VEHICLE_TIME_DELTA_BUDGET_EXCEEDED,
            True,
            frame_delta_exceeded,
            join_wait_exceeded,
            component_age_exceeded,
        )
    if frame_delta_exceeded:
        return (
            False,
            PACKET_COHERENCE_REASON_FRONT_VEHICLE_FRAME_DELTA_BUDGET_EXCEEDED,
            time_delta_exceeded,
            True,
            join_wait_exceeded,
            component_age_exceeded,
        )
    if join_wait_exceeded:
        return (
            False,
            PACKET_COHERENCE_REASON_JOIN_WAIT_BUDGET_EXCEEDED,
            time_delta_exceeded,
            frame_delta_exceeded,
            True,
            component_age_exceeded,
        )
    if component_age_exceeded:
        return (
            False,
            PACKET_COHERENCE_REASON_COMPONENT_AGE_BUDGET_EXCEEDED,
            time_delta_exceeded,
            frame_delta_exceeded,
            join_wait_exceeded,
            True,
        )
    return (
        True,
        PACKET_COHERENCE_REASON_COHERENT,
        False,
        False,
        False,
        False,
    )


def _make_partial_packet(
    *,
    unity_frame_count: Optional[int],
    unity_time: Optional[float],
    now: float,
    packet_key: Optional[str],
) -> dict[str, Any]:
    global sync_packet_id_counter
    sync_packet_id_counter += 1
    normalized_packet_key = _normalize_packet_key(packet_key)
    return {
        "packet_id": sync_packet_id_counter,
        "schema_version": SYNC_PACKET_SCHEMA_VERSION,
        "packet_key": normalized_packet_key,
        "unity_frame_count": int(unity_frame_count) if unity_frame_count is not None else None,
        "unity_time": float(unity_time) if unity_time is not None else None,
        "join_source": (
            PACKET_JOIN_SOURCE_PACKET_KEY
            if normalized_packet_key is not None
            else PACKET_JOIN_SOURCE_UNITY_FRAME_FALLBACK
        ),
        "join_key_present": normalized_packet_key is not None,
        "created_wall_time": float(now),
        "superseded_camera_count": 0,
        "superseded_vehicle_count": 0,
        "source_key_present_camera": False,
        "source_key_present_vehicle": False,
        "front_camera": None,
        "vehicle_state": None,
    }


def _register_partial_packet(packet: dict[str, Any]) -> None:
    packet_key = _normalize_packet_key(packet.get("packet_key"))
    unity_frame_count = _parse_optional_int(packet.get("unity_frame_count"))
    if packet_key is not None:
        pending_packets_by_packet_key[packet_key] = packet
    if unity_frame_count is not None and packet_key is None:
        pending_packets_by_unity_frame[unity_frame_count] = packet


def _remove_partial_packet(packet: Optional[dict[str, Any]]) -> None:
    if not isinstance(packet, dict):
        return
    packet_key = _normalize_packet_key(packet.get("packet_key"))
    unity_frame_count = _parse_optional_int(packet.get("unity_frame_count"))
    if packet_key is not None and pending_packets_by_packet_key.get(packet_key) is packet:
        pending_packets_by_packet_key.pop(packet_key, None)
    if unity_frame_count is not None and pending_packets_by_unity_frame.get(unity_frame_count) is packet:
        pending_packets_by_unity_frame.pop(unity_frame_count, None)


def _find_partial_packet(
    *,
    packet_key: Optional[str],
    unity_frame_count: Optional[int],
) -> Optional[dict[str, Any]]:
    normalized_packet_key = _normalize_packet_key(packet_key)
    if normalized_packet_key is not None:
        return pending_packets_by_packet_key.get(normalized_packet_key)
    if unity_frame_count is not None:
        packet = pending_packets_by_unity_frame.get(int(unity_frame_count))
        if packet is not None:
            return packet
    return None


def _evict_stale_partial_packets(current_unity_frame_count: Optional[int], now: float) -> None:
    global sync_packet_orphan_camera_count, sync_packet_orphan_vehicle_count
    global sync_packet_timeout_count
    global latest_sync_packet_join_failure_reason, latest_sync_packet_join_failure_side
    global latest_sync_packet_join_failure_source_key_present_camera
    global latest_sync_packet_join_failure_source_key_present_vehicle
    global sync_packet_join_failure_event_count

    to_remove: list[dict[str, Any]] = []
    unique_packets = {id(packet): packet for packet in pending_packets_by_unity_frame.values()}
    unique_packets.update({id(packet): packet for packet in pending_packets_by_packet_key.values()})
    packets_snapshot = list(unique_packets.values())
    for packet in packets_snapshot:
        too_old_by_time = (float(now) - float(packet.get("created_wall_time", now))) > PARTIAL_PACKET_MAX_AGE_SECONDS
        packet_unity_frame_count = _parse_optional_int(packet.get("unity_frame_count"))
        too_old_by_frame = (
            current_unity_frame_count is not None
            and packet_unity_frame_count is not None
            and str(packet.get("join_source") or "") != PACKET_JOIN_SOURCE_PACKET_KEY
            and int(packet_unity_frame_count) < int(current_unity_frame_count) - PARTIAL_PACKET_MAX_FRAME_LAG
        )
        if not too_old_by_time and not too_old_by_frame:
            continue
        if packet.get("front_camera") is not None and packet.get("vehicle_state") is None:
            sync_packet_orphan_camera_count += 1
        elif packet.get("vehicle_state") is not None and packet.get("front_camera") is None:
            sync_packet_orphan_vehicle_count += 1
        sync_packet_timeout_count += 1
        (
            latest_sync_packet_join_failure_reason,
            latest_sync_packet_join_failure_side,
            latest_sync_packet_join_failure_source_key_present_camera,
            latest_sync_packet_join_failure_source_key_present_vehicle,
        ) = _classify_join_failure(packet, packets_snapshot)
        sync_packet_join_failure_event_count += 1
        to_remove.append(packet)

    for packet in to_remove:
        _remove_partial_packet(packet)


def _pending_partial_packets_snapshot() -> list[dict[str, Any]]:
    unique_packets = {id(packet): packet for packet in pending_packets_by_unity_frame.values()}
    unique_packets.update({id(packet): packet for packet in pending_packets_by_packet_key.values()})
    return [packet for packet in unique_packets.values() if isinstance(packet, dict)]


def _select_pending_partial_candidate(now: float) -> Optional[dict[str, Any]]:
    packets_snapshot = _pending_partial_packets_snapshot()
    if not packets_snapshot:
        return None

    best_packet: Optional[dict[str, Any]] = None
    best_snapshot: Optional[dict[str, Any]] = None
    best_score: Optional[tuple] = None

    for packet in packets_snapshot:
        if packet.get("front_camera") is not None and packet.get("vehicle_state") is not None:
            continue
        join_failure_reason_code, join_failure_side_code, _, _ = _classify_join_failure(
            packet,
            packets_snapshot,
        )
        snapshot = _snapshot_sync_packet(packet, now=now)
        created_wall_time = _parse_optional_float(packet.get("created_wall_time"))
        partial_age_ms = (
            max(0.0, (float(now) - float(created_wall_time)) * 1000.0)
            if created_wall_time is not None
            else float("inf")
        )
        waitable = (
            join_failure_reason_code
            in {
                PACKET_JOIN_FAILURE_VEHICLE_COMPONENT_LATE,
                PACKET_JOIN_FAILURE_CAMERA_COMPONENT_LATE,
            }
            and partial_age_ms <= float(ACTIVE_SELECTION_PARTIAL_WAIT_MS)
        )
        score = (
            1 if waitable else 0,
            1 if _normalize_packet_key(packet.get("packet_key")) is not None else 0,
            1 if join_failure_side_code == PACKET_JOIN_FAILURE_SIDE_VEHICLE else 0,
            -partial_age_ms,
            int(packet.get("packet_id", -1) or -1),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_packet = packet
            best_snapshot = snapshot

    if best_packet is None or best_snapshot is None:
        return None

    join_failure_reason_code, join_failure_side_code, source_key_present_camera, source_key_present_vehicle = _classify_join_failure(
        best_packet,
        packets_snapshot,
    )
    created_wall_time = _parse_optional_float(best_packet.get("created_wall_time"))
    partial_age_ms = (
        max(0.0, (float(now) - float(created_wall_time)) * 1000.0)
        if created_wall_time is not None
        else float("inf")
    )
    waitable = (
        join_failure_reason_code
        in {
            PACKET_JOIN_FAILURE_VEHICLE_COMPONENT_LATE,
            PACKET_JOIN_FAILURE_CAMERA_COMPONENT_LATE,
        }
        and partial_age_ms <= float(ACTIVE_SELECTION_PARTIAL_WAIT_MS)
    )

    return {
        "packet": best_packet,
        "snapshot": best_snapshot,
        "join_failure_reason_code": join_failure_reason_code,
        "join_failure_side_code": join_failure_side_code,
        "source_key_present_camera": source_key_present_camera,
        "source_key_present_vehicle": source_key_present_vehicle,
        "partial_age_ms": partial_age_ms,
        "waitable": waitable,
    }


def _snapshot_sync_packet(packet: dict[str, Any], now: Optional[float] = None) -> dict[str, Any]:
    now_value = float(time.time() if now is None else now)
    front = packet.get("front_camera") or {}
    vehicle = packet.get("vehicle_state") or {}

    def _bundle_value(name: str) -> Any:
        for component in (front, vehicle):
            if not isinstance(component, dict):
                continue
            value = component.get(name)
            if value is None:
                continue
            if isinstance(value, str) and not value:
                continue
            return value
        return None

    front_arrival = _parse_optional_float(front.get("arrival_wall_time"))
    vehicle_arrival = _parse_optional_float(vehicle.get("arrival_wall_time"))
    publish_wall_time = _parse_optional_float(packet.get("publish_wall_time"))
    packet_age_ms = (now_value - publish_wall_time) * 1000.0 if publish_wall_time is not None else None
    front_age_ms = (now_value - front_arrival) * 1000.0 if front_arrival is not None else None
    vehicle_age_ms = (now_value - vehicle_arrival) * 1000.0 if vehicle_arrival is not None else None
    front_frame = _parse_optional_int(front.get("unity_frame_count"))
    vehicle_frame = _parse_optional_int(vehicle.get("unity_frame_count"))
    front_vehicle_frame_delta = None
    if front_frame is not None and vehicle_frame is not None:
        front_vehicle_frame_delta = int(front_frame - vehicle_frame)
    front_time = _parse_optional_float(front.get("unity_time"))
    vehicle_time = _parse_optional_float(vehicle.get("unity_time"))
    front_vehicle_time_delta_ms = None
    if front_time is not None and vehicle_time is not None:
        front_vehicle_time_delta_ms = float(front_time - vehicle_time) * 1000.0
    payload_oldest_age_ms = None
    if len(assembled_packet_payload_queue) > 0:
        oldest_publish_wall_time = _parse_optional_float(assembled_packet_payload_queue[0].get("publish_wall_time"))
        if oldest_publish_wall_time is not None:
            payload_oldest_age_ms = (now_value - oldest_publish_wall_time) * 1000.0
    payload_bytes = 0
    image_array = front.get("image")
    if isinstance(image_array, np.ndarray):
        payload_bytes = int(image_array.nbytes)
    join_wait_ms = None
    created_wall_time = _parse_optional_float(packet.get("created_wall_time"))
    if created_wall_time is not None and publish_wall_time is not None:
        join_wait_ms = max(0.0, (publish_wall_time - created_wall_time) * 1000.0)
    source_key_present_camera = _component_has_packet_key(front)
    source_key_present_vehicle = _component_has_packet_key(vehicle)
    complete = bool(packet.get("front_camera") is not None and packet.get("vehicle_state") is not None)
    (
        coherence_pass,
        coherence_reason_code,
        time_delta_budget_exceeded,
        frame_delta_budget_exceeded,
        join_wait_budget_exceeded,
        component_age_budget_exceeded,
    ) = _classify_packet_coherence(
        complete=complete,
        front_vehicle_time_delta_ms=front_vehicle_time_delta_ms,
        front_vehicle_frame_delta=front_vehicle_frame_delta,
        join_wait_ms=join_wait_ms,
        front_age_ms=front_age_ms,
        vehicle_age_ms=vehicle_age_ms,
    )
    source_context_queue_depth = _parse_optional_int(front.get("source_packet_context_queue_depth"))
    source_context_dropped_stale_count = _parse_optional_int(
        front.get("source_packet_context_dropped_stale_count")
    )
    source_context_missing_count = _parse_optional_int(
        front.get("source_packet_context_missing_count")
    )
    source_context_frame_delta = _parse_optional_int(front.get("source_packet_context_frame_delta"))
    source_context_time_delta_ms = _parse_optional_float(
        front.get("source_packet_context_time_delta_ms")
    )
    selected_failure_contract_reason = PACKET_SELECTED_FAILURE_CONTRACT_NONE
    selected_failure_source_stage = PACKET_SELECTED_FAILURE_STAGE_NONE
    if not complete:
        join_failure_reason_code, join_failure_side_code, _, _ = _classify_join_failure(
            packet,
            [packet],
        )
        (
            selected_failure_contract_reason,
            selected_failure_source_stage,
        ) = _classify_selected_failure_contract(
            packet,
            join_failure_reason_code=join_failure_reason_code,
            join_failure_side_code=join_failure_side_code,
        )
    return {
        "packet_id": int(packet.get("packet_id", -1)),
        "schema_version": int(packet.get("schema_version", SYNC_PACKET_SCHEMA_VERSION)),
        "mode": PACKET_MODE_PACKET_SHADOW,
        "selection_result": PACKET_SELECTION_RESULT_COMPLETE,
        "packet_key": _normalize_packet_key(packet.get("packet_key")),
        "unity_frame_count": _parse_optional_int(packet.get("unity_frame_count")),
        "unity_time": _parse_optional_float(packet.get("unity_time")),
        "join_source": str(packet.get("join_source") or PACKET_JOIN_SOURCE_NONE),
        "join_key_present": bool(packet.get("join_key_present", False)),
        "join_failure_reason_code": PACKET_JOIN_FAILURE_NONE,
        "join_failure_side_code": PACKET_JOIN_FAILURE_SIDE_NONE,
        "selected_failure_contract_reason_code": selected_failure_contract_reason,
        "selected_failure_source_stage_code": selected_failure_source_stage,
        "source_key_present_camera": source_key_present_camera,
        "source_key_present_vehicle": source_key_present_vehicle,
        "coherence_pass": bool(coherence_pass),
        "coherence_reason_code": str(coherence_reason_code or PACKET_COHERENCE_REASON_COHERENT),
        "complete_but_incoherent": bool(complete and not coherence_pass),
        "front_vehicle_time_delta_budget_exceeded": bool(time_delta_budget_exceeded),
        "front_vehicle_frame_delta_budget_exceeded": bool(frame_delta_budget_exceeded),
        "join_wait_budget_exceeded": bool(join_wait_budget_exceeded),
        "component_age_budget_exceeded": bool(component_age_budget_exceeded),
        "source_packet_context_queue_depth": (
            int(source_context_queue_depth) if source_context_queue_depth is not None else None
        ),
        "source_packet_context_dropped_stale_count": (
            int(source_context_dropped_stale_count)
            if source_context_dropped_stale_count is not None
            else None
        ),
        "source_packet_context_missing_count": (
            int(source_context_missing_count) if source_context_missing_count is not None else None
        ),
        "source_packet_context_frame_delta": (
            int(source_context_frame_delta) if source_context_frame_delta is not None else None
        ),
        "source_packet_context_time_delta_ms": (
            float(source_context_time_delta_ms)
            if source_context_time_delta_ms is not None
            else None
        ),
        "source_bundle_close_reason": str(
            _bundle_value("source_bundle_close_reason") or ""
        ),
        "source_bundle_deadline_ms": _parse_optional_float(
            _bundle_value("source_bundle_deadline_ms")
        ),
        "source_bundle_age_ms": _parse_optional_float(
            _bundle_value("source_bundle_age_ms")
        ),
        "source_bundle_inflight_count": _parse_optional_int(
            _bundle_value("source_bundle_inflight_count")
        ),
        "source_bundle_vehicle_state_built": (
            bool(_bundle_value("source_bundle_vehicle_state_built"))
            if _bundle_value("source_bundle_vehicle_state_built") is not None
            else None
        ),
        "source_bundle_vehicle_state_enqueued": (
            bool(_bundle_value("source_bundle_vehicle_state_enqueued"))
            if _bundle_value("source_bundle_vehicle_state_enqueued") is not None
            else None
        ),
        "source_bundle_vehicle_state_sent": (
            bool(_bundle_value("source_bundle_vehicle_state_sent"))
            if _bundle_value("source_bundle_vehicle_state_sent") is not None
            else None
        ),
        "source_bundle_camera_requested": (
            bool(_bundle_value("source_bundle_camera_requested"))
            if _bundle_value("source_bundle_camera_requested") is not None
            else None
        ),
        "source_bundle_camera_request_attempted": (
            bool(_bundle_value("source_bundle_camera_request_attempted"))
            if _bundle_value("source_bundle_camera_request_attempted") is not None
            else None
        ),
        "source_bundle_camera_request_accepted": (
            bool(_bundle_value("source_bundle_camera_request_accepted"))
            if _bundle_value("source_bundle_camera_request_accepted") is not None
            else None
        ),
        "source_bundle_camera_request_rejected_reason": str(
            _bundle_value("source_bundle_camera_request_rejected_reason") or ""
        ),
        "source_bundle_camera_request_skipped_reason": str(
            _bundle_value("source_bundle_camera_request_skipped_reason") or ""
        ),
        "source_bundle_camera_request_disposition_code": str(
            _bundle_value("source_bundle_camera_request_disposition_code") or ""
        ),
        "source_bundle_camera_request_attempt_age_ms": _parse_optional_float(
            _bundle_value("source_bundle_camera_request_attempt_age_ms")
        ),
        "source_bundle_camera_request_accept_age_ms": _parse_optional_float(
            _bundle_value("source_bundle_camera_request_accept_age_ms")
        ),
        "source_bundle_camera_request_queue_depth": _parse_optional_int(
            _bundle_value("source_bundle_camera_request_queue_depth")
        ),
        "source_bundle_active_transport_eligible": (
            bool(_bundle_value("source_bundle_active_transport_eligible"))
            if _bundle_value("source_bundle_active_transport_eligible") is not None
            else None
        ),
        "source_bundle_debug_unbundled_capture": (
            bool(_bundle_value("source_bundle_debug_unbundled_capture"))
            if _bundle_value("source_bundle_debug_unbundled_capture") is not None
            else None
        ),
        "camera_capture_contract_reason": str(
            _bundle_value("camera_capture_contract_reason") or ""
        ),
        "source_bundle_camera_sent": (
            bool(_bundle_value("source_bundle_camera_sent"))
            if _bundle_value("source_bundle_camera_sent") is not None
            else None
        ),
        "source_bundle_aborted_before_vehicle_send": (
            bool(_bundle_value("source_bundle_aborted_before_vehicle_send"))
            if _bundle_value("source_bundle_aborted_before_vehicle_send") is not None
            else None
        ),
        "source_bundle_abort_reason": str(
            _bundle_value("source_bundle_abort_reason") or ""
        ),
        "source_bundle_vehicle_send_blocked_by_camera_request": (
            bool(_bundle_value("source_bundle_vehicle_send_blocked_by_camera_request"))
            if _bundle_value("source_bundle_vehicle_send_blocked_by_camera_request") is not None
            else None
        ),
        "source_bundle_superseded_before_send": (
            bool(_bundle_value("source_bundle_superseded_before_send"))
            if _bundle_value("source_bundle_superseded_before_send") is not None
            else None
        ),
        "active_camera_excluded_event_delta": 0,
        "active_camera_excluded_reason_code": "",
        "unbundled_camera_entered_active_path_event_delta": 0,
        "timeout_event_delta": 0,
        "join_failure_event_count": int(sync_packet_join_failure_event_count),
        "join_wait_ms": join_wait_ms,
        "complete": complete,
        "fallback_active": False,
        "fallback_reason_code": PACKET_FALLBACK_NONE,
        "missing_front": packet.get("front_camera") is None,
        "missing_vehicle": packet.get("vehicle_state") is None,
        "queue_depth": len(assembled_packet_queue),
        "drop_count": int(sync_packet_drop_count),
        "payload_queue_depth": len(assembled_packet_payload_queue),
        "payload_drop_count": int(sync_packet_payload_drop_count),
        "payload_oldest_age_ms": payload_oldest_age_ms,
        "payload_bytes": payload_bytes,
        "payload_fallback_reason_code": PACKET_FALLBACK_NONE,
        "orphan_camera_count": int(sync_packet_orphan_camera_count),
        "orphan_vehicle_count": int(sync_packet_orphan_vehicle_count),
        "timeout_count": int(sync_packet_timeout_count),
        "key_match_count": int(sync_packet_key_match_count),
        "unity_fallback_count": int(sync_packet_unity_fallback_count),
        "superseded_camera_count": int(sync_packet_superseded_camera_count),
        "superseded_vehicle_count": int(sync_packet_superseded_vehicle_count),
        "packet_superseded_camera_count": int(packet.get("superseded_camera_count", 0) or 0),
        "packet_superseded_vehicle_count": int(packet.get("superseded_vehicle_count", 0) or 0),
        "packet_age_ms": packet_age_ms,
        "front_age_ms": front_age_ms,
        "vehicle_age_ms": vehicle_age_ms,
        "front_vehicle_frame_delta": front_vehicle_frame_delta,
        "front_vehicle_time_delta_ms": front_vehicle_time_delta_ms,
        "front_camera": {
            "camera_id": front.get("camera_id"),
            "frame_id": _parse_optional_int(front.get("frame_id")),
            "timestamp": _parse_optional_float(front.get("timestamp")),
            "unity_frame_count": front_frame,
            "unity_time": front_time,
        },
        "vehicle_state": {
            "request_id": vehicle.get("request_id"),
            "unity_frame_count": vehicle_frame,
            "unity_time": vehicle_time,
        },
    }


def _sync_packet_payload_response(packet: dict[str, Any]) -> Response:
    return _sync_packet_payload_response_with_selection(packet, None)


def _payload_queue_oldest_age_ms(now_value: float) -> Optional[float]:
    if len(assembled_packet_payload_queue) <= 0:
        return None
    oldest_publish_wall_time = _parse_optional_float(
        assembled_packet_payload_queue[0].get("publish_wall_time")
    )
    if oldest_publish_wall_time is None:
        return None
    return (now_value - oldest_publish_wall_time) * 1000.0


def _selection_headers(selection_meta: dict[str, Any]) -> dict[str, str]:
    return {
        "X-AV-Selection-Result": _hdr_str(selection_meta.get("selection_result")),
        "X-AV-Selection-Source": _hdr_str(selection_meta.get("payload_selection_source")),
        "X-AV-Selection-Fallback-Active": "1"
        if bool(selection_meta.get("selection_fallback_active", False))
        else "0",
        "X-AV-Selection-Fallback-Reason": _hdr_str(
            selection_meta.get("selection_fallback_reason_code")
        ),
        "X-AV-Selected-Payload-Age-Ms": _hdr_str(
            selection_meta.get("payload_selected_age_ms")
        ),
        "X-AV-Selected-Payload-Fresh": "1"
        if bool(selection_meta.get("payload_selected_fresh", False))
        else "0",
        "X-AV-Selected-Payload-Warn-Age-Exceeded": "1"
        if bool(selection_meta.get("payload_warn_age_exceeded", False))
        else "0",
        "X-AV-Stale-Drop-Count": str(
            int(selection_meta.get("payload_stale_drop_count", 0) or 0)
        ),
        "X-AV-Drained-Count": str(
            int(selection_meta.get("payload_drained_count", 0) or 0)
        ),
        "X-AV-Max-Drained-Age-Ms": _hdr_str(
            selection_meta.get("payload_max_drained_age_ms")
        ),
        "X-AV-Server-Queue-Depth-After-Select": str(
            int(selection_meta.get("payload_server_queue_depth_after_select", 0) or 0)
        ),
        "X-AV-Server-Oldest-Age-Ms-After-Select": _hdr_str(
            selection_meta.get("payload_server_oldest_age_ms_after_select")
        ),
        "X-AV-Join-Failure-Reason": _hdr_str(
            selection_meta.get("join_failure_reason_code")
        ),
        "X-AV-Join-Failure-Side": _hdr_str(
            selection_meta.get("join_failure_side_code")
        ),
        "X-AV-Selected-Failure-Contract-Reason": _hdr_str(
            selection_meta.get("selected_failure_contract_reason_code")
        ),
        "X-AV-Selected-Failure-Source-Stage": _hdr_str(
            selection_meta.get("selected_failure_source_stage_code")
        ),
        "X-AV-Source-Bundle-Close-Reason": _hdr_str(
            selection_meta.get("source_bundle_close_reason")
        ),
        "X-AV-Source-Bundle-Deadline-Ms": _hdr_str(
            selection_meta.get("source_bundle_deadline_ms")
        ),
        "X-AV-Source-Bundle-Age-Ms": _hdr_str(
            selection_meta.get("source_bundle_age_ms")
        ),
        "X-AV-Source-Bundle-Inflight-Count": _hdr_str(
            selection_meta.get("source_bundle_inflight_count")
        ),
        "X-AV-Source-Bundle-Vehicle-State-Built": "1"
        if bool(selection_meta.get("source_bundle_vehicle_state_built", False))
        else (
            ""
            if selection_meta.get("source_bundle_vehicle_state_built") is None
            else "0"
        ),
        "X-AV-Source-Bundle-Vehicle-State-Enqueued": "1"
        if bool(selection_meta.get("source_bundle_vehicle_state_enqueued", False))
        else (
            ""
            if selection_meta.get("source_bundle_vehicle_state_enqueued") is None
            else "0"
        ),
        "X-AV-Source-Bundle-Vehicle-State-Sent": "1"
        if bool(selection_meta.get("source_bundle_vehicle_state_sent", False))
        else (
            ""
            if selection_meta.get("source_bundle_vehicle_state_sent") is None
            else "0"
        ),
        "X-AV-Source-Bundle-Camera-Requested": "1"
        if bool(selection_meta.get("source_bundle_camera_requested", False))
        else (
            ""
            if selection_meta.get("source_bundle_camera_requested") is None
            else "0"
        ),
        "X-AV-Source-Bundle-Camera-Request-Attempted": "1"
        if bool(selection_meta.get("source_bundle_camera_request_attempted", False))
        else (
            ""
            if selection_meta.get("source_bundle_camera_request_attempted") is None
            else "0"
        ),
        "X-AV-Source-Bundle-Camera-Request-Accepted": "1"
        if bool(selection_meta.get("source_bundle_camera_request_accepted", False))
        else (
            ""
            if selection_meta.get("source_bundle_camera_request_accepted") is None
            else "0"
        ),
        "X-AV-Source-Bundle-Camera-Request-Rejected-Reason": _hdr_str(
            selection_meta.get("source_bundle_camera_request_rejected_reason")
        ),
        "X-AV-Source-Bundle-Camera-Request-Skipped-Reason": _hdr_str(
            selection_meta.get("source_bundle_camera_request_skipped_reason")
        ),
        "X-AV-Source-Bundle-Camera-Request-Disposition-Code": _hdr_str(
            selection_meta.get("source_bundle_camera_request_disposition_code")
        ),
        "X-AV-Source-Bundle-Camera-Request-Attempt-Age-Ms": _hdr_str(
            selection_meta.get("source_bundle_camera_request_attempt_age_ms")
        ),
        "X-AV-Source-Bundle-Camera-Request-Accept-Age-Ms": _hdr_str(
            selection_meta.get("source_bundle_camera_request_accept_age_ms")
        ),
        "X-AV-Source-Bundle-Camera-Request-Queue-Depth": _hdr_str(
            selection_meta.get("source_bundle_camera_request_queue_depth")
        ),
        "X-AV-Source-Bundle-Active-Transport-Eligible": "1"
        if bool(selection_meta.get("source_bundle_active_transport_eligible", False))
        else (
            ""
            if selection_meta.get("source_bundle_active_transport_eligible") is None
            else "0"
        ),
        "X-AV-Source-Bundle-Debug-Unbundled-Capture": "1"
        if bool(selection_meta.get("source_bundle_debug_unbundled_capture", False))
        else (
            ""
            if selection_meta.get("source_bundle_debug_unbundled_capture") is None
            else "0"
        ),
        "X-AV-Camera-Capture-Contract-Reason": _hdr_str(
            selection_meta.get("camera_capture_contract_reason")
        ),
        "X-AV-Source-Bundle-Camera-Sent": "1"
        if bool(selection_meta.get("source_bundle_camera_sent", False))
        else (
            ""
            if selection_meta.get("source_bundle_camera_sent") is None
            else "0"
        ),
        "X-AV-Source-Bundle-Aborted-Before-Vehicle-Send": "1"
        if bool(selection_meta.get("source_bundle_aborted_before_vehicle_send", False))
        else (
            ""
            if selection_meta.get("source_bundle_aborted_before_vehicle_send") is None
            else "0"
        ),
        "X-AV-Source-Bundle-Abort-Reason": _hdr_str(
            selection_meta.get("source_bundle_abort_reason")
        ),
        "X-AV-Source-Bundle-Vehicle-Send-Blocked-By-Camera-Request": "1"
        if bool(selection_meta.get("source_bundle_vehicle_send_blocked_by_camera_request", False))
        else (
            ""
            if selection_meta.get("source_bundle_vehicle_send_blocked_by_camera_request") is None
            else "0"
        ),
        "X-AV-Source-Bundle-Superseded-Before-Send": "1"
        if bool(selection_meta.get("source_bundle_superseded_before_send", False))
        else (
            ""
            if selection_meta.get("source_bundle_superseded_before_send") is None
            else "0"
        ),
        "X-AV-Source-Key-Present-Camera": "1"
        if bool(selection_meta.get("source_key_present_camera", False))
        else "0",
        "X-AV-Source-Key-Present-Vehicle": "1"
        if bool(selection_meta.get("source_key_present_vehicle", False))
        else "0",
        "X-AV-Timeout-Event-Delta": str(
            int(selection_meta.get("timeout_event_delta", 0) or 0)
        ),
        "X-AV-Join-Failure-Event-Count": str(
            int(selection_meta.get("join_failure_event_count", 0) or 0)
        ),
        "X-AV-Active-Camera-Excluded-Event-Delta": str(
            int(selection_meta.get("active_camera_excluded_event_delta", 0) or 0)
        ),
        "X-AV-Active-Camera-Excluded-Reason": _hdr_str(
            selection_meta.get("active_camera_excluded_reason_code")
        ),
        "X-AV-Unbundled-Camera-Entered-Active-Path-Event-Delta": str(
            int(
                selection_meta.get(
                    "unbundled_camera_entered_active_path_event_delta", 0
                )
                or 0
            )
        ),
    }


def _selection_404(detail: str, selection_meta: dict[str, Any]) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail=detail,
        headers=_selection_headers(selection_meta),
    )


def _build_selection_meta(
    *,
    consume_policy: str,
    selection_source: str,
    selected_age_ms: Optional[float],
    warn_age_ms: float,
    fresh: bool,
    drained_count: int,
    stale_drop_count: int,
    max_drained_age_ms: Optional[float],
    fallback_active: bool,
    fallback_reason_code: str,
    queue_depth_after_select: int,
    oldest_age_after_select_ms: Optional[float],
    selection_result: str = PACKET_SELECTION_RESULT_NONE,
    join_failure_reason_code: str = PACKET_JOIN_FAILURE_NONE,
    join_failure_side_code: str = PACKET_JOIN_FAILURE_SIDE_NONE,
    selected_failure_contract_reason_code: str = PACKET_SELECTED_FAILURE_CONTRACT_NONE,
    selected_failure_source_stage_code: str = PACKET_SELECTED_FAILURE_STAGE_NONE,
    source_bundle_close_reason: str = "",
    source_bundle_deadline_ms: Optional[float] = None,
    source_bundle_age_ms: Optional[float] = None,
    source_bundle_inflight_count: Optional[int] = None,
    source_bundle_vehicle_state_built: Optional[bool] = None,
    source_bundle_vehicle_state_enqueued: Optional[bool] = None,
    source_bundle_vehicle_state_sent: Optional[bool] = None,
    source_bundle_camera_requested: Optional[bool] = None,
    source_bundle_camera_request_attempted: Optional[bool] = None,
    source_bundle_camera_request_accepted: Optional[bool] = None,
    source_bundle_camera_request_rejected_reason: str = "",
    source_bundle_camera_request_skipped_reason: str = "",
    source_bundle_camera_request_disposition_code: str = "",
    source_bundle_camera_request_attempt_age_ms: Optional[float] = None,
    source_bundle_camera_request_accept_age_ms: Optional[float] = None,
    source_bundle_camera_request_queue_depth: Optional[int] = None,
    source_bundle_active_transport_eligible: Optional[bool] = None,
    source_bundle_debug_unbundled_capture: Optional[bool] = None,
    camera_capture_contract_reason: str = "",
    source_bundle_camera_sent: Optional[bool] = None,
    source_bundle_aborted_before_vehicle_send: Optional[bool] = None,
    source_bundle_abort_reason: str = "",
    source_bundle_vehicle_send_blocked_by_camera_request: Optional[bool] = None,
    source_bundle_superseded_before_send: Optional[bool] = None,
    source_key_present_camera: bool = False,
    source_key_present_vehicle: bool = False,
    timeout_event_delta: int = 0,
    join_failure_event_count: int = 0,
    active_camera_excluded_event_delta: Optional[int] = None,
    active_camera_excluded_reason_code: Optional[str] = None,
    unbundled_camera_entered_active_path_event_delta: Optional[int] = None,
    ) -> dict[str, Any]:
    if active_camera_excluded_event_delta is None:
        (
            active_camera_excluded_event_delta,
            active_camera_excluded_reason_code_value,
            unbundled_camera_entered_active_path_event_delta,
        ) = _consume_camera_admission_events()
    else:
        active_camera_excluded_reason_code_value = str(
            active_camera_excluded_reason_code or ""
        )
        if unbundled_camera_entered_active_path_event_delta is None:
            unbundled_camera_entered_active_path_event_delta = 0
    return {
        "consume_policy": str(consume_policy or ""),
        "selection_result": str(selection_result or PACKET_SELECTION_RESULT_NONE),
        "payload_selection_source": str(selection_source or PACKET_SELECTION_SOURCE_NONE),
        "payload_selected_age_ms": (
            float(selected_age_ms) if selected_age_ms is not None else float("nan")
        ),
        "payload_selected_fresh": bool(fresh),
        "payload_warn_age_exceeded": bool(
            selected_age_ms is not None and math.isfinite(float(selected_age_ms)) and float(selected_age_ms) > float(warn_age_ms)
        ),
        "payload_stale_drop_count": int(stale_drop_count),
        "payload_drained_count": int(drained_count),
        "payload_max_drained_age_ms": (
            float(max_drained_age_ms) if max_drained_age_ms is not None else float("nan")
        ),
        "selection_fallback_active": bool(fallback_active),
        "selection_fallback_reason_code": str(fallback_reason_code or PACKET_FALLBACK_NONE),
        "payload_server_queue_depth_after_select": int(queue_depth_after_select),
        "payload_server_oldest_age_ms_after_select": (
            float(oldest_age_after_select_ms)
            if oldest_age_after_select_ms is not None
            else float("nan")
        ),
        "join_failure_reason_code": str(
            join_failure_reason_code or PACKET_JOIN_FAILURE_NONE
        ),
        "join_failure_side_code": str(
            join_failure_side_code or PACKET_JOIN_FAILURE_SIDE_NONE
        ),
        "selected_failure_contract_reason_code": str(
            selected_failure_contract_reason_code
            or PACKET_SELECTED_FAILURE_CONTRACT_NONE
        ),
        "selected_failure_source_stage_code": str(
            selected_failure_source_stage_code or PACKET_SELECTED_FAILURE_STAGE_NONE
        ),
        "source_bundle_close_reason": str(source_bundle_close_reason or ""),
        "source_bundle_deadline_ms": (
            float(source_bundle_deadline_ms)
            if source_bundle_deadline_ms is not None
            else float("nan")
        ),
        "source_bundle_age_ms": (
            float(source_bundle_age_ms) if source_bundle_age_ms is not None else float("nan")
        ),
        "source_bundle_inflight_count": (
            int(source_bundle_inflight_count)
            if source_bundle_inflight_count is not None
            else -1
        ),
        "source_bundle_vehicle_state_built": source_bundle_vehicle_state_built,
        "source_bundle_vehicle_state_enqueued": source_bundle_vehicle_state_enqueued,
        "source_bundle_vehicle_state_sent": source_bundle_vehicle_state_sent,
        "source_bundle_camera_requested": source_bundle_camera_requested,
        "source_bundle_camera_request_attempted": source_bundle_camera_request_attempted,
        "source_bundle_camera_request_accepted": source_bundle_camera_request_accepted,
        "source_bundle_camera_request_rejected_reason": str(
            source_bundle_camera_request_rejected_reason or ""
        ),
        "source_bundle_camera_request_skipped_reason": str(
            source_bundle_camera_request_skipped_reason or ""
        ),
        "source_bundle_camera_request_disposition_code": str(
            source_bundle_camera_request_disposition_code or ""
        ),
        "source_bundle_camera_request_attempt_age_ms": (
            float(source_bundle_camera_request_attempt_age_ms)
            if source_bundle_camera_request_attempt_age_ms is not None
            else float("nan")
        ),
        "source_bundle_camera_request_accept_age_ms": (
            float(source_bundle_camera_request_accept_age_ms)
            if source_bundle_camera_request_accept_age_ms is not None
            else float("nan")
        ),
        "source_bundle_camera_request_queue_depth": (
            int(source_bundle_camera_request_queue_depth)
            if source_bundle_camera_request_queue_depth is not None
            else -1
        ),
        "source_bundle_active_transport_eligible": source_bundle_active_transport_eligible,
        "source_bundle_debug_unbundled_capture": source_bundle_debug_unbundled_capture,
        "camera_capture_contract_reason": str(camera_capture_contract_reason or ""),
        "source_bundle_camera_sent": source_bundle_camera_sent,
        "source_bundle_aborted_before_vehicle_send": source_bundle_aborted_before_vehicle_send,
        "source_bundle_abort_reason": str(source_bundle_abort_reason or ""),
        "source_bundle_vehicle_send_blocked_by_camera_request": (
            source_bundle_vehicle_send_blocked_by_camera_request
        ),
        "source_bundle_superseded_before_send": source_bundle_superseded_before_send,
        "source_key_present_camera": bool(source_key_present_camera),
        "source_key_present_vehicle": bool(source_key_present_vehicle),
        "timeout_event_delta": int(timeout_event_delta),
        "join_failure_event_count": int(join_failure_event_count),
        "active_camera_excluded_event_delta": int(
            active_camera_excluded_event_delta or 0
        ),
        "active_camera_excluded_reason_code": str(
            active_camera_excluded_reason_code_value or ""
        ),
        "unbundled_camera_entered_active_path_event_delta": int(
            unbundled_camera_entered_active_path_event_delta or 0
        ),
    }


def _overlay_pending_partial_meta(
    selection_meta: dict[str, Any],
    pending_candidate: Optional[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(selection_meta, dict) or not isinstance(pending_candidate, dict):
        return selection_meta
    snapshot = pending_candidate.get("snapshot") or {}
    updated = dict(selection_meta)
    updated["join_failure_reason_code"] = str(
        pending_candidate.get("join_failure_reason_code")
        or snapshot.get("join_failure_reason_code")
        or updated.get("join_failure_reason_code")
        or PACKET_JOIN_FAILURE_NONE
    )
    updated["join_failure_side_code"] = str(
        pending_candidate.get("join_failure_side_code")
        or snapshot.get("join_failure_side_code")
        or updated.get("join_failure_side_code")
        or PACKET_JOIN_FAILURE_SIDE_NONE
    )
    updated["selected_failure_contract_reason_code"] = str(
        snapshot.get("selected_failure_contract_reason_code")
        or updated.get("selected_failure_contract_reason_code")
        or PACKET_SELECTED_FAILURE_CONTRACT_NONE
    )
    updated["selected_failure_source_stage_code"] = str(
        snapshot.get("selected_failure_source_stage_code")
        or updated.get("selected_failure_source_stage_code")
        or PACKET_SELECTED_FAILURE_STAGE_NONE
    )
    updated["source_key_present_camera"] = bool(
        pending_candidate.get(
            "source_key_present_camera",
            updated.get("source_key_present_camera", False),
        )
    )
    updated["source_key_present_vehicle"] = bool(
        pending_candidate.get(
            "source_key_present_vehicle",
            updated.get("source_key_present_vehicle", False),
        )
    )
    for field in (
        "source_bundle_close_reason",
        "source_bundle_deadline_ms",
        "source_bundle_age_ms",
        "source_bundle_inflight_count",
        "source_bundle_vehicle_state_built",
        "source_bundle_vehicle_state_enqueued",
        "source_bundle_vehicle_state_sent",
        "source_bundle_camera_requested",
        "source_bundle_camera_request_attempted",
        "source_bundle_camera_request_accepted",
        "source_bundle_camera_request_rejected_reason",
        "source_bundle_camera_request_skipped_reason",
        "source_bundle_camera_request_disposition_code",
        "source_bundle_camera_request_attempt_age_ms",
        "source_bundle_camera_request_accept_age_ms",
        "source_bundle_camera_request_queue_depth",
        "source_bundle_active_transport_eligible",
        "source_bundle_debug_unbundled_capture",
        "camera_capture_contract_reason",
        "source_bundle_camera_sent",
        "source_bundle_aborted_before_vehicle_send",
        "source_bundle_abort_reason",
        "source_bundle_vehicle_send_blocked_by_camera_request",
        "source_bundle_superseded_before_send",
        "active_camera_excluded_event_delta",
        "active_camera_excluded_reason_code",
        "unbundled_camera_entered_active_path_event_delta",
    ):
        if field in snapshot:
            updated[field] = snapshot.get(field)
    return updated


def _select_sync_packet_payload(
    *,
    consume_policy: str,
    max_age_ms: float,
    warn_age_ms: float,
    max_drain_count: int,
    allow_stale_debug: bool,
) -> tuple[Optional[dict[str, Any]], dict[str, Any]]:
    now_value = float(time.time())
    policy = str(consume_policy or PACKET_CONSUME_POLICY_FIFO_STRICT)

    def _age_ms(packet: Optional[dict[str, Any]]) -> Optional[float]:
        if not isinstance(packet, dict):
            return None
        publish_wall_time = _parse_optional_float(packet.get("publish_wall_time"))
        if publish_wall_time is None:
            return None
        return (now_value - publish_wall_time) * 1000.0

    def _coherence(packet: Optional[dict[str, Any]]) -> tuple[bool, str]:
        if not isinstance(packet, dict):
            return False, PACKET_COHERENCE_REASON_MISSING_COMPONENT_TIMESTAMP
        snapshot = _snapshot_sync_packet(packet, now=now_value)
        return bool(snapshot.get("coherence_pass", False)), str(
            snapshot.get("coherence_reason_code") or PACKET_COHERENCE_REASON_COHERENT
        )

    if policy == PACKET_CONSUME_POLICY_LATEST_DEBUG:
        selected = latest_sync_packet_payload_data
        selected_age_ms = _age_ms(selected)
        coherence_pass, coherence_reason = _coherence(selected)
        fresh = bool(
            selected is not None
            and (
                allow_stale_debug
                or (
                    selected_age_ms is not None
                    and math.isfinite(selected_age_ms)
                    and selected_age_ms <= float(max_age_ms)
                )
            )
        )
        fallback_reason = PACKET_FALLBACK_NONE
        if selected is None:
            fallback_reason = PACKET_FALLBACK_PACKET_TIMEOUT
        elif not fresh:
            fallback_reason = PACKET_FALLBACK_PAYLOAD_AGE_BUDGET_EXCEEDED
        elif not coherence_pass:
            fallback_reason = coherence_reason
        meta = _build_selection_meta(
            consume_policy=policy,
            selection_source=PACKET_SELECTION_SOURCE_LATEST_DEBUG,
            selected_age_ms=selected_age_ms,
            warn_age_ms=warn_age_ms,
            fresh=fresh and coherence_pass,
            drained_count=0,
            stale_drop_count=0,
            max_drained_age_ms=None,
            fallback_active=not (fresh and coherence_pass),
            fallback_reason_code=fallback_reason,
            queue_depth_after_select=len(assembled_packet_payload_queue),
            oldest_age_after_select_ms=_payload_queue_oldest_age_ms(now_value),
            selection_result=(
                PACKET_SELECTION_RESULT_COMPLETE_COHERENT
                if fresh and coherence_pass
                else (
                    PACKET_SELECTION_RESULT_COMPLETE_INCOHERENT
                    if selected is not None and fresh and not coherence_pass
                    else PACKET_SELECTION_RESULT_FALLBACK
                )
            ),
            join_failure_reason_code=latest_sync_packet_join_failure_reason,
            join_failure_side_code=latest_sync_packet_join_failure_side,
            selected_failure_contract_reason_code=PACKET_SELECTED_FAILURE_CONTRACT_BRIDGE_JOIN_FAILURE,
            selected_failure_source_stage_code=PACKET_SELECTED_FAILURE_STAGE_BRIDGE_JOIN,
            source_key_present_camera=latest_sync_packet_join_failure_source_key_present_camera,
            source_key_present_vehicle=latest_sync_packet_join_failure_source_key_present_vehicle,
            timeout_event_delta=1 if selected is None else 0,
            join_failure_event_count=sync_packet_join_failure_event_count,
        )
        return (selected if fresh and coherence_pass else None), meta

    if len(assembled_packet_payload_queue) <= 0:
        meta = _build_selection_meta(
            consume_policy=policy,
            selection_source=PACKET_SELECTION_SOURCE_SERVER,
            selected_age_ms=None,
            warn_age_ms=warn_age_ms,
            fresh=False,
            drained_count=0,
            stale_drop_count=0,
            max_drained_age_ms=None,
            fallback_active=True,
            fallback_reason_code=PACKET_FALLBACK_PACKET_TIMEOUT,
            queue_depth_after_select=0,
            oldest_age_after_select_ms=None,
            selection_result=PACKET_SELECTION_RESULT_FALLBACK,
            join_failure_reason_code=latest_sync_packet_join_failure_reason,
            join_failure_side_code=latest_sync_packet_join_failure_side,
            selected_failure_contract_reason_code=PACKET_SELECTED_FAILURE_CONTRACT_BRIDGE_JOIN_FAILURE,
            selected_failure_source_stage_code=PACKET_SELECTED_FAILURE_STAGE_BRIDGE_JOIN,
            source_key_present_camera=latest_sync_packet_join_failure_source_key_present_camera,
            source_key_present_vehicle=latest_sync_packet_join_failure_source_key_present_vehicle,
            timeout_event_delta=1,
            join_failure_event_count=sync_packet_join_failure_event_count,
        )
        return None, meta

    if policy == PACKET_CONSUME_POLICY_FIFO_STRICT:
        selected = assembled_packet_payload_queue.popleft()
        selected_age_ms = _age_ms(selected)
        coherence_pass, coherence_reason = _coherence(selected)
        fresh = bool(
            selected_age_ms is not None
            and math.isfinite(selected_age_ms)
            and selected_age_ms <= float(max_age_ms)
        )
        meta = _build_selection_meta(
            consume_policy=policy,
            selection_source=PACKET_SELECTION_SOURCE_SERVER,
            selected_age_ms=selected_age_ms,
            warn_age_ms=warn_age_ms,
            fresh=fresh and coherence_pass,
            drained_count=0,
            stale_drop_count=0,
            max_drained_age_ms=None,
            fallback_active=not (fresh and coherence_pass),
            fallback_reason_code=(
                PACKET_FALLBACK_NONE
                if fresh and coherence_pass
                else (
                    coherence_reason
                    if fresh and not coherence_pass
                    else PACKET_FALLBACK_PAYLOAD_AGE_BUDGET_EXCEEDED
                )
            ),
            queue_depth_after_select=len(assembled_packet_payload_queue),
            oldest_age_after_select_ms=_payload_queue_oldest_age_ms(now_value),
            selection_result=(
                PACKET_SELECTION_RESULT_COMPLETE_COHERENT
                if fresh and coherence_pass
                else (
                    PACKET_SELECTION_RESULT_COMPLETE_INCOHERENT
                    if fresh and not coherence_pass
                    else PACKET_SELECTION_RESULT_FALLBACK
                )
            ),
            join_failure_reason_code=(
                PACKET_JOIN_FAILURE_NONE
                if fresh and coherence_pass
                else latest_sync_packet_join_failure_reason
            ),
            join_failure_side_code=(
                PACKET_JOIN_FAILURE_SIDE_NONE
                if fresh and coherence_pass
                else latest_sync_packet_join_failure_side
            ),
            selected_failure_contract_reason_code=(
                PACKET_SELECTED_FAILURE_CONTRACT_NONE
                if fresh and coherence_pass
                else PACKET_SELECTED_FAILURE_CONTRACT_BRIDGE_JOIN_FAILURE
            ),
            selected_failure_source_stage_code=(
                PACKET_SELECTED_FAILURE_STAGE_NONE
                if fresh and coherence_pass
                else PACKET_SELECTED_FAILURE_STAGE_BRIDGE_JOIN
            ),
            source_key_present_camera=(
                _component_has_packet_key(selected.get("front_camera") if isinstance(selected, dict) else None)
                if fresh and coherence_pass
                else latest_sync_packet_join_failure_source_key_present_camera
            ),
            source_key_present_vehicle=(
                _component_has_packet_key(selected.get("vehicle_state") if isinstance(selected, dict) else None)
                if fresh and coherence_pass
                else latest_sync_packet_join_failure_source_key_present_vehicle
            ),
            timeout_event_delta=0 if fresh and coherence_pass else 1,
            join_failure_event_count=sync_packet_join_failure_event_count,
        )
        return (selected if fresh and coherence_pass else None), meta

    newest = assembled_packet_payload_queue[-1]
    older_packets = list(assembled_packet_payload_queue)[:-1]
    drained_count = len(older_packets)
    max_drained_age_ms = None
    if older_packets:
        ages = [_age_ms(packet) for packet in older_packets]
        finite_ages = [float(age) for age in ages if age is not None and math.isfinite(age)]
        if finite_ages:
            max_drained_age_ms = max(finite_ages)
    if max_drain_count >= 0 and drained_count > int(max_drain_count):
        for _ in range(min(int(max_drain_count), len(assembled_packet_payload_queue))):
            assembled_packet_payload_queue.popleft()
        meta = _build_selection_meta(
            consume_policy=policy,
            selection_source=PACKET_SELECTION_SOURCE_SERVER,
            selected_age_ms=_age_ms(newest),
            warn_age_ms=warn_age_ms,
            fresh=False,
            drained_count=min(drained_count, int(max_drain_count)),
            stale_drop_count=min(drained_count, int(max_drain_count)),
            max_drained_age_ms=max_drained_age_ms,
            fallback_active=True,
            fallback_reason_code=PACKET_FALLBACK_PAYLOAD_DRAIN_CAP,
            queue_depth_after_select=len(assembled_packet_payload_queue),
            oldest_age_after_select_ms=_payload_queue_oldest_age_ms(now_value),
            selection_result=PACKET_SELECTION_RESULT_FALLBACK,
            join_failure_reason_code=latest_sync_packet_join_failure_reason,
            join_failure_side_code=latest_sync_packet_join_failure_side,
            selected_failure_contract_reason_code=PACKET_SELECTED_FAILURE_CONTRACT_BRIDGE_JOIN_FAILURE,
            selected_failure_source_stage_code=PACKET_SELECTED_FAILURE_STAGE_BRIDGE_JOIN,
            source_key_present_camera=latest_sync_packet_join_failure_source_key_present_camera,
            source_key_present_vehicle=latest_sync_packet_join_failure_source_key_present_vehicle,
            timeout_event_delta=1,
            join_failure_event_count=sync_packet_join_failure_event_count,
        )
        return None, meta

    assembled_packet_payload_queue.clear()
    selected_age_ms = _age_ms(newest)
    coherence_pass, coherence_reason = _coherence(newest)
    fresh = bool(
        selected_age_ms is not None
        and math.isfinite(selected_age_ms)
        and selected_age_ms <= float(max_age_ms)
    )
    meta = _build_selection_meta(
        consume_policy=policy,
        selection_source=PACKET_SELECTION_SOURCE_SERVER,
        selected_age_ms=selected_age_ms,
        warn_age_ms=warn_age_ms,
        fresh=fresh and coherence_pass,
        drained_count=drained_count,
        stale_drop_count=drained_count,
        max_drained_age_ms=max_drained_age_ms,
        fallback_active=not (fresh and coherence_pass),
        fallback_reason_code=(
            PACKET_FALLBACK_NONE
            if fresh and coherence_pass
            else (
                coherence_reason
                if fresh and not coherence_pass
                else PACKET_FALLBACK_PAYLOAD_AGE_BUDGET_EXCEEDED
            )
        ),
        queue_depth_after_select=len(assembled_packet_payload_queue),
        oldest_age_after_select_ms=_payload_queue_oldest_age_ms(now_value),
        selection_result=(
            PACKET_SELECTION_RESULT_COMPLETE_COHERENT
            if fresh and coherence_pass
            else (
                PACKET_SELECTION_RESULT_COMPLETE_INCOHERENT
                if fresh and not coherence_pass
                else PACKET_SELECTION_RESULT_FALLBACK
            )
        ),
        join_failure_reason_code=(
            PACKET_JOIN_FAILURE_NONE
            if fresh and coherence_pass
            else latest_sync_packet_join_failure_reason
        ),
        join_failure_side_code=(
            PACKET_JOIN_FAILURE_SIDE_NONE
            if fresh and coherence_pass
            else latest_sync_packet_join_failure_side
        ),
        selected_failure_contract_reason_code=(
            PACKET_SELECTED_FAILURE_CONTRACT_NONE
            if fresh and coherence_pass
            else PACKET_SELECTED_FAILURE_CONTRACT_BRIDGE_JOIN_FAILURE
        ),
        selected_failure_source_stage_code=(
            PACKET_SELECTED_FAILURE_STAGE_NONE
            if fresh and coherence_pass
            else PACKET_SELECTED_FAILURE_STAGE_BRIDGE_JOIN
        ),
        source_key_present_camera=(
            _component_has_packet_key(newest.get("front_camera") if isinstance(newest, dict) else None)
            if fresh and coherence_pass
            else latest_sync_packet_join_failure_source_key_present_camera
        ),
        source_key_present_vehicle=(
            _component_has_packet_key(newest.get("vehicle_state") if isinstance(newest, dict) else None)
            if fresh and coherence_pass
            else latest_sync_packet_join_failure_source_key_present_vehicle
        ),
        timeout_event_delta=0 if fresh and coherence_pass else 1,
        join_failure_event_count=sync_packet_join_failure_event_count,
    )
    return (newest if fresh and coherence_pass else None), meta


def _sync_packet_payload_response_with_selection(
    packet: dict[str, Any],
    selection_meta: Optional[dict[str, Any]],
) -> Response:
    now_value = float(time.time())
    snapshot = _snapshot_sync_packet(packet, now=now_value)
    if selection_meta:
        protected_source_keys = {
            "source_bundle_close_reason",
            "source_bundle_deadline_ms",
            "source_bundle_age_ms",
            "source_bundle_inflight_count",
            "source_bundle_vehicle_state_built",
            "source_bundle_vehicle_state_enqueued",
            "source_bundle_vehicle_state_sent",
            "source_bundle_camera_requested",
            "source_bundle_camera_request_attempted",
            "source_bundle_camera_request_accepted",
            "source_bundle_camera_request_rejected_reason",
            "source_bundle_camera_request_skipped_reason",
            "source_bundle_camera_request_disposition_code",
            "source_bundle_camera_request_attempt_age_ms",
            "source_bundle_camera_request_accept_age_ms",
            "source_bundle_camera_request_queue_depth",
            "source_bundle_active_transport_eligible",
            "source_bundle_debug_unbundled_capture",
            "camera_capture_contract_reason",
            "source_bundle_camera_sent",
            "source_bundle_aborted_before_vehicle_send",
            "source_bundle_abort_reason",
            "source_bundle_vehicle_send_blocked_by_camera_request",
            "source_bundle_superseded_before_send",
            "active_camera_excluded_event_delta",
            "active_camera_excluded_reason_code",
            "unbundled_camera_entered_active_path_event_delta",
        }
        for key, value in selection_meta.items():
            if key not in protected_source_keys:
                snapshot[key] = value
                continue
            existing = snapshot.get(key)
            if value is None:
                continue
            if isinstance(value, float) and math.isnan(value):
                continue
            if isinstance(value, int) and value < 0:
                continue
            if isinstance(value, str) and not value and existing not in (None, ""):
                continue
            snapshot[key] = value
        snapshot["fallback_active"] = bool(
            selection_meta.get("selection_fallback_active", False)
        )
        snapshot["fallback_reason_code"] = str(
            selection_meta.get("selection_fallback_reason_code") or PACKET_FALLBACK_NONE
        )
        snapshot["payload_fallback_reason_code"] = str(
            selection_meta.get("selection_fallback_reason_code") or PACKET_FALLBACK_NONE
        )
    front = packet.get("front_camera") or {}
    vehicle = packet.get("vehicle_state") or {}
    image_array = front.get("image")
    if not isinstance(image_array, np.ndarray):
        raise HTTPException(status_code=500, detail="Synchronized payload missing front image bytes")
    vehicle_state = vehicle.get("state")
    if not isinstance(vehicle_state, dict):
        raise HTTPException(status_code=500, detail="Synchronized payload missing vehicle state snapshot")

    image_bytes = image_array.tobytes(order="C")
    metadata = {
        "packet": snapshot,
        "vehicle_state": vehicle_state,
        "front_camera": {
            "camera_id": front.get("camera_id"),
            "timestamp": _parse_optional_float(front.get("timestamp")),
            "frame_id": _parse_optional_int(front.get("frame_id")),
            "unity_frame_count": _parse_optional_int(front.get("unity_frame_count")),
            "unity_time": _parse_optional_float(front.get("unity_time")),
            "shape": list(image_array.shape),
            "dtype": str(image_array.dtype),
        },
    }
    boundary = f"avsync-{int(snapshot.get('packet_id', -1))}-{int(now_value * 1000)}"
    meta_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    body = b"".join(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            b"Content-Type: application/json\r\n",
            b'Content-Disposition: form-data; name="metadata"\r\n\r\n',
            meta_bytes,
            b"\r\n",
            f"--{boundary}\r\n".encode("utf-8"),
            b"Content-Type: application/octet-stream\r\n",
            b'Content-Disposition: form-data; name="image"; filename="front_center.rgb"\r\n\r\n',
            image_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    headers = {
        "X-AV-Packet-Id": str(snapshot.get("packet_id", "")),
        "X-AV-Unity-Frame-Count": _hdr_str(snapshot.get("unity_frame_count")),
        "X-AV-Payload-Bytes": str(len(image_bytes)),
    }
    if selection_meta:
        headers.update(_selection_headers(snapshot))
    return Response(content=body, media_type=f"multipart/mixed; boundary={boundary}", headers=headers)


def _build_payload_packet(packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "packet_id": int(packet.get("packet_id", -1)),
        "schema_version": int(packet.get("schema_version", SYNC_PACKET_SCHEMA_VERSION)),
        "packet_key": _normalize_packet_key(packet.get("packet_key")),
        "unity_frame_count": packet.get("unity_frame_count"),
        "unity_time": packet.get("unity_time"),
        "join_source": packet.get("join_source"),
        "join_key_present": bool(packet.get("join_key_present", False)),
        "created_wall_time": packet.get("created_wall_time"),
        "publish_wall_time": packet.get("publish_wall_time"),
        "superseded_camera_count": int(packet.get("superseded_camera_count", 0) or 0),
        "superseded_vehicle_count": int(packet.get("superseded_vehicle_count", 0) or 0),
        "front_camera": packet.get("front_camera"),
        "vehicle_state": packet.get("vehicle_state"),
    }


def _publish_completed_packet(packet: dict[str, Any], now: float) -> None:
    global latest_sync_packet_data, latest_sync_packet_payload_data
    global sync_packet_drop_count, sync_packet_payload_drop_count
    global sync_packet_key_match_count, sync_packet_unity_fallback_count

    packet["publish_wall_time"] = float(now)
    packet["queue_depth_at_publish"] = len(assembled_packet_queue) + 1
    if str(packet.get("join_source") or "") == PACKET_JOIN_SOURCE_PACKET_KEY:
        sync_packet_key_match_count += 1
    else:
        sync_packet_unity_fallback_count += 1
    if assembled_packet_queue.maxlen is not None and len(assembled_packet_queue) >= assembled_packet_queue.maxlen:
        sync_packet_drop_count += 1
    metadata_packet = _build_payload_packet(packet)
    assembled_packet_queue.append(metadata_packet)
    latest_sync_packet_data = metadata_packet
    if (
        assembled_packet_payload_queue.maxlen is not None
        and len(assembled_packet_payload_queue) >= assembled_packet_payload_queue.maxlen
    ):
        sync_packet_payload_drop_count += 1
    payload_packet = _build_payload_packet(packet)
    assembled_packet_payload_queue.append(payload_packet)
    latest_sync_packet_payload_data = payload_packet


def _attach_camera_component(
    *,
    packet_key: Optional[str],
    unity_frame_count: Optional[int],
    unity_time: Optional[float],
    component: dict[str, Any],
    now: float,
) -> None:
    global sync_packet_superseded_camera_count
    global sync_packet_unbundled_camera_entered_active_path_count
    global sync_packet_unbundled_camera_entered_active_path_pending_count
    normalized_packet_key = _normalize_packet_key(packet_key)
    if unity_frame_count is None and normalized_packet_key is None:
        return
    active_transport_eligible = component.get("source_bundle_active_transport_eligible")
    if active_transport_eligible is False:
        _record_excluded_camera_component(component)
        return
    if component.get("source_bundle_debug_unbundled_capture") is True:
        sync_packet_unbundled_camera_entered_active_path_count += 1
        sync_packet_unbundled_camera_entered_active_path_pending_count += 1
        _record_excluded_camera_component(component)
        return
    _evict_stale_partial_packets(unity_frame_count, now)
    packet = _find_partial_packet(
        packet_key=normalized_packet_key,
        unity_frame_count=unity_frame_count,
    )
    if packet is None:
        packet = _make_partial_packet(
            unity_frame_count=unity_frame_count,
            unity_time=unity_time,
            now=now,
            packet_key=normalized_packet_key,
        )
        _register_partial_packet(packet)
    elif packet.get("front_camera") is not None:
        sync_packet_superseded_camera_count += 1
        packet["superseded_camera_count"] = int(packet.get("superseded_camera_count", 0) or 0) + 1

    if normalized_packet_key is not None and _normalize_packet_key(packet.get("packet_key")) is None:
        packet["packet_key"] = normalized_packet_key
        packet["join_source"] = PACKET_JOIN_SOURCE_PACKET_KEY
        packet["join_key_present"] = True
        _register_partial_packet(packet)
    if packet.get("unity_frame_count") is None and unity_frame_count is not None:
        packet["unity_frame_count"] = int(unity_frame_count)
        _register_partial_packet(packet)
    packet["front_camera"] = component
    packet["source_key_present_camera"] = normalized_packet_key is not None
    if packet.get("unity_time") is None and unity_time is not None:
        packet["unity_time"] = float(unity_time)
    if packet.get("vehicle_state") is not None:
        _publish_completed_packet(packet, now)
        _remove_partial_packet(packet)


def _attach_vehicle_component(
    *,
    packet_key: Optional[str],
    unity_frame_count: Optional[int],
    unity_time: Optional[float],
    component: dict[str, Any],
    now: float,
) -> None:
    global sync_packet_superseded_vehicle_count
    normalized_packet_key = _normalize_packet_key(packet_key)
    if unity_frame_count is None and normalized_packet_key is None:
        return
    _evict_stale_partial_packets(unity_frame_count, now)
    packet = _find_partial_packet(
        packet_key=normalized_packet_key,
        unity_frame_count=unity_frame_count,
    )
    if packet is None:
        packet = _make_partial_packet(
            unity_frame_count=unity_frame_count,
            unity_time=unity_time,
            now=now,
            packet_key=normalized_packet_key,
        )
        _register_partial_packet(packet)
    elif packet.get("vehicle_state") is not None:
        sync_packet_superseded_vehicle_count += 1
        packet["superseded_vehicle_count"] = int(packet.get("superseded_vehicle_count", 0) or 0) + 1

    if normalized_packet_key is not None and _normalize_packet_key(packet.get("packet_key")) is None:
        packet["packet_key"] = normalized_packet_key
        packet["join_source"] = PACKET_JOIN_SOURCE_PACKET_KEY
        packet["join_key_present"] = True
        _register_partial_packet(packet)
    if packet.get("unity_frame_count") is None and unity_frame_count is not None:
        packet["unity_frame_count"] = int(unity_frame_count)
        _register_partial_packet(packet)
    packet["vehicle_state"] = component
    packet["source_key_present_vehicle"] = normalized_packet_key is not None
    if packet.get("unity_time") is None and unity_time is not None:
        packet["unity_time"] = float(unity_time)
    if packet.get("front_camera") is not None:
        _publish_completed_packet(packet, now)
        _remove_partial_packet(packet)


def _hdr_str(value: Optional[Any]) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return str(value)


def _raw_camera_response(
    frame: np.ndarray,
    *,
    camera_key: str,
    timestamp_value: Optional[float],
    frame_id_value: Optional[int],
    unity_frame_count: Optional[int],
    unity_time: Optional[float],
    queue_depth: int,
    queue_capacity: int,
    drop_count: int,
    decode_in_flight: bool,
    latest_age_ms: Optional[float],
    last_arrival_time: Optional[float],
    last_realtime: Optional[float],
    last_unscaled: Optional[float],
    queue_remaining: Optional[int] = None,
) -> Response:
    """Single RGB uint8 payload; metadata only in headers (for Python client)."""
    arr = np.ascontiguousarray(frame, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise HTTPException(status_code=500, detail="Camera frame must be HxWx3 uint8 RGB")
    body = arr.tobytes()
    h, w, c = arr.shape
    headers: dict[str, str] = {
        "X-AV-Shape": f"{h},{w},{c}",
        "X-AV-Camera-Id": camera_key,
        "X-AV-Timestamp": _hdr_str(timestamp_value),
        "X-AV-Frame-Id": _hdr_str(frame_id_value),
        "X-AV-Unity-Frame-Count": _hdr_str(unity_frame_count),
        "X-AV-Unity-Time": _hdr_str(unity_time),
        "X-AV-Queue-Depth": str(int(queue_depth)),
        "X-AV-Queue-Capacity": str(int(queue_capacity)),
        "X-AV-Drop-Count": str(int(drop_count)),
        "X-AV-Decode-In-Flight": "1" if decode_in_flight else "0",
        "X-AV-Latest-Age-Ms": _hdr_str(latest_age_ms),
        "X-AV-Last-Arrival-Time": _hdr_str(last_arrival_time),
        "X-AV-Last-Realtime": _hdr_str(last_realtime),
        "X-AV-Last-Unscaled": _hdr_str(last_unscaled),
    }
    if queue_remaining is not None:
        headers["X-AV-Queue-Remaining"] = str(int(queue_remaining))
    return Response(content=body, media_type="application/octet-stream", headers=headers)


async def _decode_and_store_camera_frame(
    image_data: bytes,
    timestamp: str,
    frame_id: str,
    camera_id: str,
    sync_packet_key: Optional[str] = None,
    source_packet_owner: Optional[str] = None,
    source_packet_owner_update_id: Optional[int] = None,
    source_packet_owner_unity_frame_count: Optional[int] = None,
    source_packet_owner_unity_time: Optional[float] = None,
    source_bundle_close_reason: Optional[str] = None,
    source_bundle_deadline_ms: Optional[float] = None,
    source_bundle_age_ms: Optional[float] = None,
    source_bundle_inflight_count: Optional[int] = None,
    source_bundle_camera_requested: Optional[bool] = None,
    source_bundle_camera_request_attempted: Optional[bool] = None,
    source_bundle_camera_request_accepted: Optional[bool] = None,
    source_bundle_camera_request_rejected_reason: Optional[str] = None,
    source_bundle_camera_request_skipped_reason: Optional[str] = None,
    source_bundle_camera_request_disposition_code: Optional[str] = None,
    source_bundle_camera_request_attempt_age_ms: Optional[float] = None,
    source_bundle_camera_request_accept_age_ms: Optional[float] = None,
    source_bundle_camera_request_queue_depth: Optional[int] = None,
    source_bundle_active_transport_eligible: Optional[bool] = None,
    source_bundle_debug_unbundled_capture: Optional[bool] = None,
    camera_capture_contract_reason: Optional[str] = None,
    source_bundle_camera_sent: Optional[bool] = None,
    source_bundle_aborted_before_vehicle_send: Optional[bool] = None,
    source_bundle_abort_reason: Optional[str] = None,
    source_bundle_vehicle_send_blocked_by_camera_request: Optional[bool] = None,
    source_bundle_vehicle_state_built: Optional[bool] = None,
    source_bundle_vehicle_state_enqueued: Optional[bool] = None,
    source_bundle_vehicle_state_sent: Optional[bool] = None,
    source_bundle_superseded_before_send: Optional[bool] = None,
    source_packet_context_queue_depth: Optional[int] = None,
    source_packet_context_dropped_stale_count: Optional[int] = None,
    source_packet_context_missing_count: Optional[int] = None,
    source_packet_context_frame_delta: Optional[int] = None,
    source_packet_context_time_delta_ms: Optional[float] = None,
    unity_frame_count: Optional[int] = None,
    unity_time: Optional[float] = None,
    realtime_since_startup: Optional[float] = None,
    unscaled_time: Optional[float] = None,
    arrival_wall_time: Optional[float] = None,
) -> None:
    global latest_camera_frame, latest_frame_timestamp, latest_frame_id
    global latest_camera_frames, latest_frame_timestamps_by_id, latest_frame_ids_by_id
    global camera_decode_in_flight, camera_drop_count

    start_time = time.time()
    try:
        img = Image.open(io.BytesIO(image_data))
        img_array = np.array(img)
        stored_arrival_wall_time = float(arrival_wall_time if arrival_wall_time is not None else time.time())
        latest_camera_frames[camera_id] = img_array
        latest_frame_timestamps_by_id[camera_id] = float(timestamp)
        latest_frame_ids_by_id[camera_id] = frame_id
        queue = camera_frame_queues.get(camera_id)
        if queue is None:
            queue = deque(maxlen=MAX_CAMERA_QUEUE_SIZE)
            camera_frame_queues[camera_id] = queue
        else:
            # deque(maxlen=N) drops oldest on overflow; count for HDF5 / PhilViz diagnostics.
            if queue.maxlen is not None and len(queue) >= queue.maxlen:
                camera_drop_count[camera_id] = camera_drop_count.get(camera_id, 0) + 1
        queue.append(
            {
                "image": img_array,
                "timestamp": float(timestamp),
                "frame_id": frame_id,
                "camera_id": camera_id,
                "sync_packet_key": _normalize_packet_key(sync_packet_key),
                "source_packet_owner": str(source_packet_owner or ""),
                "source_packet_owner_update_id": source_packet_owner_update_id,
                "source_packet_owner_unity_frame_count": source_packet_owner_unity_frame_count,
                "source_packet_owner_unity_time": source_packet_owner_unity_time,
                "source_bundle_close_reason": str(source_bundle_close_reason or ""),
                "source_bundle_deadline_ms": source_bundle_deadline_ms,
                "source_bundle_age_ms": source_bundle_age_ms,
                "source_bundle_inflight_count": source_bundle_inflight_count,
                "source_bundle_camera_requested": source_bundle_camera_requested,
                "source_bundle_camera_request_attempted": source_bundle_camera_request_attempted,
                "source_bundle_camera_request_accepted": source_bundle_camera_request_accepted,
                "source_bundle_camera_request_rejected_reason": str(
                    source_bundle_camera_request_rejected_reason or ""
                ),
                "source_bundle_camera_request_skipped_reason": str(
                    source_bundle_camera_request_skipped_reason or ""
                ),
                "source_bundle_camera_request_disposition_code": str(
                    source_bundle_camera_request_disposition_code or ""
                ),
                "source_bundle_camera_request_attempt_age_ms": source_bundle_camera_request_attempt_age_ms,
                "source_bundle_camera_request_accept_age_ms": source_bundle_camera_request_accept_age_ms,
                "source_bundle_camera_request_queue_depth": source_bundle_camera_request_queue_depth,
                "source_bundle_active_transport_eligible": source_bundle_active_transport_eligible,
                "source_bundle_debug_unbundled_capture": source_bundle_debug_unbundled_capture,
                "camera_capture_contract_reason": str(camera_capture_contract_reason or ""),
                "source_bundle_camera_sent": source_bundle_camera_sent,
                "source_bundle_aborted_before_vehicle_send": source_bundle_aborted_before_vehicle_send,
                "source_bundle_abort_reason": str(source_bundle_abort_reason or ""),
                "source_bundle_vehicle_send_blocked_by_camera_request": (
                    source_bundle_vehicle_send_blocked_by_camera_request
                ),
                "source_bundle_vehicle_state_built": source_bundle_vehicle_state_built,
                "source_bundle_vehicle_state_enqueued": source_bundle_vehicle_state_enqueued,
                "source_bundle_vehicle_state_sent": source_bundle_vehicle_state_sent,
                "source_bundle_superseded_before_send": source_bundle_superseded_before_send,
                "source_packet_context_queue_depth": source_packet_context_queue_depth,
                "source_packet_context_dropped_stale_count": source_packet_context_dropped_stale_count,
                "source_packet_context_missing_count": source_packet_context_missing_count,
                "source_packet_context_frame_delta": source_packet_context_frame_delta,
                "source_packet_context_time_delta_ms": source_packet_context_time_delta_ms,
                "unity_frame_count": unity_frame_count,
                "unity_time": unity_time,
                "realtime_since_startup": realtime_since_startup,
                "unscaled_time": unscaled_time,
                "arrival_wall_time": stored_arrival_wall_time,
            }
        )
        if camera_id == "front_center":
            latest_camera_frame = img_array
            latest_frame_timestamp = float(timestamp)
            latest_frame_id = frame_id
        _attach_camera_component(
            packet_key=_normalize_packet_key(sync_packet_key),
            unity_frame_count=unity_frame_count,
            unity_time=unity_time,
            component={
                "camera_id": camera_id,
                "timestamp": float(timestamp),
                "frame_id": frame_id,
                "sync_packet_key": _normalize_packet_key(sync_packet_key),
                "source_packet_owner": str(source_packet_owner or ""),
                "source_packet_owner_update_id": source_packet_owner_update_id,
                "source_packet_owner_unity_frame_count": source_packet_owner_unity_frame_count,
                "source_packet_owner_unity_time": source_packet_owner_unity_time,
                "source_bundle_close_reason": str(source_bundle_close_reason or ""),
                "source_bundle_deadline_ms": source_bundle_deadline_ms,
                "source_bundle_age_ms": source_bundle_age_ms,
                "source_bundle_inflight_count": source_bundle_inflight_count,
                "source_bundle_camera_requested": source_bundle_camera_requested,
                "source_bundle_camera_request_attempted": source_bundle_camera_request_attempted,
                "source_bundle_camera_request_accepted": source_bundle_camera_request_accepted,
                "source_bundle_camera_request_rejected_reason": str(
                    source_bundle_camera_request_rejected_reason or ""
                ),
                "source_bundle_camera_request_skipped_reason": str(
                    source_bundle_camera_request_skipped_reason or ""
                ),
                "source_bundle_camera_request_disposition_code": str(
                    source_bundle_camera_request_disposition_code or ""
                ),
                "source_bundle_camera_request_attempt_age_ms": source_bundle_camera_request_attempt_age_ms,
                "source_bundle_camera_request_accept_age_ms": source_bundle_camera_request_accept_age_ms,
                "source_bundle_camera_request_queue_depth": source_bundle_camera_request_queue_depth,
                "source_bundle_active_transport_eligible": source_bundle_active_transport_eligible,
                "source_bundle_debug_unbundled_capture": source_bundle_debug_unbundled_capture,
                "camera_capture_contract_reason": str(camera_capture_contract_reason or ""),
                "source_bundle_camera_sent": source_bundle_camera_sent,
                "source_bundle_aborted_before_vehicle_send": source_bundle_aborted_before_vehicle_send,
                "source_bundle_abort_reason": str(source_bundle_abort_reason or ""),
                "source_bundle_vehicle_send_blocked_by_camera_request": (
                    source_bundle_vehicle_send_blocked_by_camera_request
                ),
                "source_bundle_vehicle_state_built": source_bundle_vehicle_state_built,
                "source_bundle_vehicle_state_enqueued": source_bundle_vehicle_state_enqueued,
                "source_bundle_vehicle_state_sent": source_bundle_vehicle_state_sent,
                "source_bundle_superseded_before_send": source_bundle_superseded_before_send,
                "source_packet_context_queue_depth": source_packet_context_queue_depth,
                "source_packet_context_dropped_stale_count": source_packet_context_dropped_stale_count,
                "source_packet_context_missing_count": source_packet_context_missing_count,
                "source_packet_context_frame_delta": source_packet_context_frame_delta,
                "source_packet_context_time_delta_ms": source_packet_context_time_delta_ms,
                "unity_frame_count": unity_frame_count,
                "unity_time": unity_time,
                "realtime_since_startup": realtime_since_startup,
                "unscaled_time": unscaled_time,
                "arrival_wall_time": stored_arrival_wall_time,
                "image": img_array,
            },
            now=stored_arrival_wall_time,
        )
    finally:
        duration = time.time() - start_time
        if duration > SLOW_REQUEST_SECONDS:
            logger.warning(
                "[SLOW] /api/camera decode "
                "duration=%.3fs bytes=%d frame_id=%s ts=%s",
                duration,
                len(image_data),
                frame_id,
                timestamp,
            )
        camera_decode_in_flight[camera_id] = False


class VehicleState(BaseModel):
    """Vehicle state from Unity."""
    position: dict  # {"x": float, "y": float, "z": float}
    rotation: dict  # {"x": float, "y": float, "z": float, "w": float}
    velocity: dict  # {"x": float, "y": float, "z": float}
    angularVelocity: dict
    speed: float
    steeringAngle: float
    steeringAngleActual: float = 0.0
    steeringInput: float = 0.0
    desiredSteerAngle: float = 0.0
    motorTorque: float
    brakeTorque: float
    # Vehicle parameters from Unity
    maxSteerAngle: float = 30.0
    wheelbaseMeters: float = 2.5
    fixedDeltaTime: float = 0.02
    unityTime: float = 0.0
    unityFrameCount: int = 0
    unityDeltaTime: float = 0.0
    unitySmoothDeltaTime: float = 0.0
    unityUnscaledDeltaTime: float = 0.0
    unityTimeScale: float = 1.0
    requestId: int = 0
    unitySendRealtime: float = 0.0
    unitySendUtcMs: int = 0
    syncPacketKey: Optional[str] = None
    syncPacketCameraFrameId: Optional[int] = None
    syncPacketCameraUnityFrameCount: Optional[int] = None
    syncPacketCameraTimestamp: Optional[float] = None
    syncPacketSourceBundleCloseReason: Optional[str] = None
    syncPacketSourceBundleDeadlineMs: Optional[float] = None
    syncPacketSourceBundleAgeMs: Optional[float] = None
    syncPacketSourceBundleInflightCount: Optional[int] = None
    syncPacketSourceVehicleStateBuilt: bool = False
    syncPacketSourceVehicleStateEnqueued: bool = False
    syncPacketSourceVehicleStateSent: bool = False
    syncPacketSourceCameraRequested: bool = False
    syncPacketSourceCameraRequestAttempted: bool = False
    syncPacketSourceCameraRequestAccepted: bool = False
    syncPacketSourceCameraRequestRejectedReason: Optional[str] = None
    syncPacketSourceCameraRequestSkippedReason: Optional[str] = None
    syncPacketSourceCameraRequestDispositionCode: Optional[str] = None
    syncPacketSourceCameraRequestAttemptAgeMs: Optional[float] = None
    syncPacketSourceCameraRequestAcceptAgeMs: Optional[float] = None
    syncPacketSourceCameraRequestQueueDepth: Optional[int] = None
    syncPacketSourceCameraSent: bool = False
    syncPacketSourceBundleAbortedBeforeVehicleSend: bool = False
    syncPacketSourceBundleAbortReason: Optional[str] = None
    syncPacketSourceVehicleSendBlockedByCameraRequest: bool = False
    syncPacketSourceSupersededBeforeSend: bool = False
    # Ground truth lane line positions (optional)
    # These represent the painted lane line markings, not the drivable lanes
    groundTruthLeftLaneLineX: float = 0.0  # Left lane line (painted marking) position
    groundTruthRightLaneLineX: float = 0.0  # Right lane line (painted marking) position
    groundTruthLaneCenterX: float = 0.0  # Lane center (midpoint between lane lines)
    # NEW: Path-based steering data
    groundTruthDesiredHeading: float = 0.0  # Desired heading from path (degrees)
    groundTruthPathCurvature: float = 0.0  # Path curvature (1/meters)
    # NEW: Camera calibration - actual screen y pixel where 8m appears (from Unity's WorldToScreenPoint)
    camera8mScreenY: float = -1.0  # -1.0 means not calculated yet
    # NEW: Camera calibration - screen y pixel for ground truth lookahead
    cameraLookaheadScreenY: float = -1.0
    # NEW: Ground truth lookahead distance used for calibration
    groundTruthLookaheadDistance: float = 8.0
    oracleTrajectoryXY: Optional[list[float]] = None
    oracleTrajectoryWorldXYZ: Optional[list[float]] = None
    oracleTrajectoryScreenXY: Optional[list[float]] = None
    oraclePointCount: int = 0
    oracleHorizonMeters: float = 0.0
    oraclePointSpacingMeters: float = 0.0
    oracleSamplesEnabled: bool = False
    rightLaneFiducialsVehicleXY: Optional[list[float]] = None
    rightLaneFiducialsVehicleTrueXY: Optional[list[float]] = None
    rightLaneFiducialsVehicleMonotonicXY: Optional[list[float]] = None
    rightLaneFiducialsWorldXYZ: Optional[list[float]] = None
    rightLaneFiducialsScreenXY: Optional[list[float]] = None
    rightLaneFiducialsPointCount: int = 0
    rightLaneFiducialsHorizonMeters: float = 0.0
    rightLaneFiducialsSpacingMeters: float = 0.0
    rightLaneFiducialsEnabled: bool = False
    # NEW: Camera FOV information - what Unity actually uses
    cameraFieldOfView: float = 0.0  # Unity's Camera.fieldOfView value (always vertical FOV)
    cameraHorizontalFOV: float = 0.0  # Calculated horizontal FOV
    # NEW: Camera position and forward for debugging alignment
    cameraPosX: float = 0.0  # Camera position X (world coords)
    cameraPosY: float = 0.0  # Camera position Y (world coords)
    cameraPosZ: float = 0.0  # Camera position Z (world coords)
    cameraForwardX: float = 0.0  # Camera forward X (normalized)
    cameraForwardY: float = 0.0  # Camera forward Y (normalized)
    cameraForwardZ: float = 0.0  # Camera forward Z (normalized)
    # Top-down camera calibration/projection (for top-down overlay diagnostics)
    topDownCameraPosX: float = 0.0
    topDownCameraPosY: float = 0.0
    topDownCameraPosZ: float = 0.0
    topDownCameraForwardX: float = 0.0
    topDownCameraForwardY: float = 0.0
    topDownCameraForwardZ: float = 0.0
    topDownCameraOrthographicSize: float = 0.0
    topDownCameraFieldOfView: float = 0.0
    # NEW: Debug fields for diagnosing ground truth offset issues
    roadCenterAtCarX: float = 0.0  # Road center X at car's location (world coords)
    roadCenterAtCarY: float = 0.0  # Road center Y at car's location (world coords)
    roadCenterAtCarZ: float = 0.0  # Road center Z at car's location (world coords)
    roadCenterAtLookaheadX: float = 0.0  # Road center X at 8m lookahead (world coords)
    roadCenterAtLookaheadY: float = 0.0  # Road center Y at 8m lookahead (world coords)
    roadCenterAtLookaheadZ: float = 0.0  # Road center Z at 8m lookahead (world coords)
    roadCenterReferenceT: float = 0.0  # Parameter t on road path for reference point
    roadFrameLateralOffset: float = 0.0  # Lateral offset from road center (m, +right)
    roadHeadingDeg: float = 0.0  # Road tangent heading (deg)
    carHeadingDeg: float = 0.0  # Car heading (deg)
    headingDeltaDeg: float = 0.0  # Car - road heading delta (deg, -180..180)
    roadFrameLaneCenterOffset: float = 0.0  # Lookahead road center offset in road frame (m, +right)
    roadFrameLaneCenterError: float = 0.0  # Car offset vs lookahead center (m, +right)
    vehicleFrameLookaheadOffset: float = 0.0  # Lookahead road center offset in vehicle frame (m, +right)
    # GT rotation input telemetry (exact Unity inputs used for target rotation).
    gtRotationDebugValid: bool = False
    gtRotationUsedRoadFrame: bool = False
    gtRotationRejectedRoadFrameHop: bool = False
    gtRotationReferenceHeadingDeg: float = 0.0
    gtRotationRoadFrameHeadingDeg: float = 0.0
    gtRotationInputHeadingDeg: float = 0.0
    gtRotationRoadVsRefDeltaDeg: float = 0.0
    gtRotationAppliedDeltaDeg: float = 0.0
    speedLimit: float = 0.0  # Speed limit at current reference point (m/s)
    speedLimitPreview: float = 0.0  # Speed limit at preview distance ahead (m/s)
    speedLimitPreviewDistance: float = 0.0  # Preview distance used for speed limit (m)
    speedLimitPreviewMinDistance: float = 0.0  # Distance to min limit in preview window (m)
    speedLimitPreviewMid: float = 0.0  # Speed limit at mid preview distance (m/s)
    speedLimitPreviewMidDistance: float = 0.0  # Mid preview distance (m)
    speedLimitPreviewMidMinDistance: float = 0.0  # Distance to min limit in mid window (m)
    speedLimitPreviewLong: float = 0.0  # Speed limit at long preview distance (m/s)
    speedLimitPreviewLongDistance: float = 0.0  # Long preview distance (m)
    speedLimitPreviewLongMinDistance: float = 0.0  # Distance to min limit in long window (m)
    # Grade/pitch/roll telemetry (Step 3)
    pitchRad: float = 0.0       # Vehicle pitch (positive = nose up)
    rollRad: float = 0.0        # Vehicle roll (positive = right lean)
    roadGrade: float = 0.0      # Local road grade (rise/run) from track profile
    chassisGroundMinClearanceM: Optional[float] = None
    chassisGroundEffectiveMinClearanceM: Optional[float] = None
    chassisGroundClearanceM: Optional[float] = None
    chassisGroundPenetrationM: Optional[float] = None
    chassisGroundContact: bool = False
    wheelGroundedCount: Optional[int] = None
    wheelCollidersReady: bool = False
    forceFallbackActive: bool = False

    # Per-wheel diagnostics (4 wheels: FL, FR, RL, RR)
    wheelSidewaysSlip: list[float] = [0.0, 0.0, 0.0, 0.0]
    wheelForwardSlip: list[float] = [0.0, 0.0, 0.0, 0.0]
    wheelContactForce: list[float] = [0.0, 0.0, 0.0, 0.0]
    wheelRpm: list[float] = [0.0, 0.0, 0.0, 0.0]
    wheelSprungMass: list[float] = [0.0, 0.0, 0.0, 0.0]
    wheelContactNormalY: list[float] = [0.0, 0.0, 0.0, 0.0]
    wheelSteerAngleActual: float = 0.0

    # Forward radar — Step 5 ACC (set by Unity AVBridge.ComputeForwardRadar)
    radar_fwd_detected: float = 0.0       # 1.0 = target in range, 0.0 = clear
    radar_fwd_distance_m: float = 0.0     # noisy range to lead vehicle (m)
    radar_fwd_range_rate_mps: float = 0.0 # Doppler range rate (+closing, m/s)
    radar_fwd_snr: float = 0.0            # signal-to-noise proxy (1.0 at 30m)
    lead_collision_detected: bool = False
    lead_collision_override_active: bool = False


class ControlCommand(BaseModel):
    """Control command to Unity."""
    steering: float  # -1.0 to 1.0
    throttle: float  # -1.0 to 1.0
    brake: float     # 0.0 to 1.0
    ground_truth_mode: bool = False  # Enable direct velocity control
    ground_truth_speed: float = 5.0  # Speed for ground truth mode (m/s)
    randomize_start: bool = False  # Randomize car start on oval
    randomize_request_id: int = 0  # Deduplicate randomization requests
    randomize_seed: Optional[int] = None  # Optional seed for repeatability
    emergency_stop: bool = False  # Emergency stop flag from AV stack


class GroundTruthMode(BaseModel):
    """Ground truth mode configuration."""
    enabled: bool
    speed: float = 5.0  # Speed in m/s for ground truth mode


class UnityFeedback(BaseModel):
    """Unity feedback/status data."""
    timestamp: float
    # Control status
    ground_truth_mode_active: bool = False
    control_command_received: bool = False
    actual_steering_applied: Optional[float] = None
    actual_throttle_applied: Optional[float] = None
    actual_brake_applied: Optional[float] = None
    chassis_ground_min_clearance_m: Optional[float] = None
    chassis_ground_effective_min_clearance_m: Optional[float] = None
    chassis_ground_clearance_m: Optional[float] = None
    chassis_ground_penetration_m: Optional[float] = None
    chassis_ground_contact: bool = False
    wheel_grounded_count: Optional[int] = None
    wheel_colliders_ready: bool = False
    force_fallback_active: bool = False
    # Ground truth data status
    ground_truth_data_available: bool = False
    ground_truth_reporter_enabled: bool = False
    path_curvature_calculated: bool = False
    # Errors/warnings
    unity_errors: Optional[str] = None
    unity_warnings: Optional[str] = None
    # Internal state
    car_controller_mode: Optional[str] = None
    av_control_enabled: bool = False
    # Frame info
    unity_frame_count: Optional[int] = None
    unity_time: Optional[float] = None


@app.post("/api/camera")
async def receive_camera_frame(
    image: UploadFile = File(...),
    timestamp: str = Form(...),
    frame_id: str = Form(...),
    camera_id: Optional[str] = Form(None),
    sync_packet_key: Optional[str] = Form(None),
    source_packet_owner: Optional[str] = Form(None),
    source_packet_owner_update_id: Optional[str] = Form(None),
    source_packet_owner_unity_frame_count: Optional[str] = Form(None),
    source_packet_owner_unity_time: Optional[str] = Form(None),
    source_bundle_close_reason: Optional[str] = Form(None),
    source_bundle_deadline_ms: Optional[str] = Form(None),
    source_bundle_age_ms: Optional[str] = Form(None),
    source_bundle_inflight_count: Optional[str] = Form(None),
    source_bundle_camera_requested: Optional[str] = Form(None),
    source_bundle_camera_request_attempted: Optional[str] = Form(None),
    source_bundle_camera_request_accepted: Optional[str] = Form(None),
    source_bundle_camera_request_rejected_reason: Optional[str] = Form(None),
    source_bundle_camera_request_skipped_reason: Optional[str] = Form(None),
    source_bundle_camera_request_disposition_code: Optional[str] = Form(None),
    source_bundle_camera_request_attempt_age_ms: Optional[str] = Form(None),
    source_bundle_camera_request_accept_age_ms: Optional[str] = Form(None),
    source_bundle_camera_request_queue_depth: Optional[str] = Form(None),
    source_bundle_active_transport_eligible: Optional[str] = Form(None),
    source_bundle_debug_unbundled_capture: Optional[str] = Form(None),
    camera_capture_contract_reason: Optional[str] = Form(None),
    source_bundle_camera_sent: Optional[str] = Form(None),
    source_bundle_aborted_before_vehicle_send: Optional[str] = Form(None),
    source_bundle_abort_reason: Optional[str] = Form(None),
    source_bundle_vehicle_send_blocked_by_camera_request: Optional[str] = Form(None),
    source_bundle_vehicle_state_built: Optional[str] = Form(None),
    source_bundle_vehicle_state_enqueued: Optional[str] = Form(None),
    source_bundle_vehicle_state_sent: Optional[str] = Form(None),
    source_bundle_superseded_before_send: Optional[str] = Form(None),
    source_packet_context_queue_depth: Optional[str] = Form(None),
    source_packet_context_dropped_stale_count: Optional[str] = Form(None),
    source_packet_context_missing_count: Optional[str] = Form(None),
    source_packet_context_frame_delta: Optional[str] = Form(None),
    source_packet_context_time_delta_ms: Optional[str] = Form(None),
    unity_frame_count: Optional[str] = Form(None),
    unity_time: Optional[str] = Form(None),
    realtime_since_startup: Optional[str] = Form(None),
    unscaled_time: Optional[str] = Form(None),
    time_scale: Optional[str] = Form(None)
):
    """
    Receive camera frame from Unity.
    
    Args:
        image: JPEG image file
        timestamp: Frame timestamp
        frame_id: Frame ID
    """
    global latest_camera_frame, latest_frame_timestamp
    
    start_time = time.time()
    try:
        # Read image data
        image_data = await image.read()

        camera_key = _normalize_camera_id(camera_id)
        last_arrival = last_camera_arrival_time.get(camera_key)
        last_frame = last_camera_frame_id.get(camera_key)
        last_timestamp = last_camera_timestamp.get(camera_key)
        last_realtime = last_camera_realtime.get(camera_key)
        last_unscaled = last_camera_unscaled.get(camera_key)
        now = time.time()
        if last_arrival is not None:
            gap = now - last_arrival
            if gap > UNITY_TIME_GAP_SECONDS:
                logger.warning(
                    "[ARRIVAL_GAP] /api/camera gap=%.3fs frame_id=%s prev_frame_id=%s camera_id=%s",
                    gap,
                    frame_id,
                    last_frame,
                    camera_key,
                )
                print(
                    "[ARRIVAL_GAP] /api/camera "
                    f"gap={gap:.3f}s frame_id={frame_id} prev_frame_id={last_frame} camera_id={camera_key}"
                )
        last_camera_arrival_time[camera_key] = now
        last_camera_frame_id[camera_key] = frame_id
        try:
            ts_value = float(timestamp)
        except ValueError:
            ts_value = None
        if ts_value is not None:
            if last_timestamp is not None:
                ts_gap = ts_value - last_timestamp
                if ts_gap > UNITY_TIME_GAP_SECONDS:
                    logger.warning(
                        "[CAMERA_TIMESTAMP_GAP] ts_gap=%.3fs frame_id=%s camera_id=%s",
                        ts_gap,
                        frame_id,
                        camera_key,
                    )
                    print(
                        "[CAMERA_TIMESTAMP_GAP] "
                        f"ts_gap={ts_gap:.3f}s frame_id={frame_id} camera_id={camera_key}"
                    )
            last_camera_timestamp[camera_key] = ts_value

        realtime_value = _parse_optional_float(realtime_since_startup)
        unscaled_value = _parse_optional_float(unscaled_time)
        unity_time_value = _parse_optional_float(unity_time)
        unity_frame_count_value = _parse_optional_int(unity_frame_count)
        source_owner_update_id_value = _parse_optional_int(source_packet_owner_update_id)
        source_owner_unity_frame_count_value = _parse_optional_int(
            source_packet_owner_unity_frame_count
        )
        source_owner_unity_time_value = _parse_optional_float(source_packet_owner_unity_time)
        source_bundle_deadline_ms_value = _parse_optional_float(source_bundle_deadline_ms)
        source_bundle_age_ms_value = _parse_optional_float(source_bundle_age_ms)
        source_bundle_inflight_count_value = _parse_optional_int(source_bundle_inflight_count)
        source_bundle_camera_requested_value = _parse_optional_int(source_bundle_camera_requested)
        source_bundle_camera_request_attempted_value = _parse_optional_int(
            source_bundle_camera_request_attempted
        )
        source_bundle_camera_request_accepted_value = _parse_optional_int(
            source_bundle_camera_request_accepted
        )
        source_bundle_aborted_before_vehicle_send_value = _parse_optional_int(
            source_bundle_aborted_before_vehicle_send
        )
        source_bundle_vehicle_send_blocked_by_camera_request_value = _parse_optional_int(
            source_bundle_vehicle_send_blocked_by_camera_request
        )
        source_bundle_camera_request_attempt_age_ms_value = _parse_optional_float(
            source_bundle_camera_request_attempt_age_ms
        )
        source_bundle_camera_request_accept_age_ms_value = _parse_optional_float(
            source_bundle_camera_request_accept_age_ms
        )
        source_bundle_camera_request_queue_depth_value = _parse_optional_int(
            source_bundle_camera_request_queue_depth
        )
        source_bundle_active_transport_eligible_value = _parse_optional_int(
            source_bundle_active_transport_eligible
        )
        source_bundle_debug_unbundled_capture_value = _parse_optional_int(
            source_bundle_debug_unbundled_capture
        )
        source_bundle_camera_sent_value = _parse_optional_int(source_bundle_camera_sent)
        source_bundle_vehicle_state_built_value = _parse_optional_int(source_bundle_vehicle_state_built)
        source_bundle_vehicle_state_enqueued_value = _parse_optional_int(source_bundle_vehicle_state_enqueued)
        source_bundle_vehicle_state_sent_value = _parse_optional_int(source_bundle_vehicle_state_sent)
        source_bundle_superseded_before_send_value = _parse_optional_int(
            source_bundle_superseded_before_send
        )
        source_context_queue_depth_value = _parse_optional_int(source_packet_context_queue_depth)
        source_context_dropped_stale_count_value = _parse_optional_int(
            source_packet_context_dropped_stale_count
        )
        source_context_missing_count_value = _parse_optional_int(
            source_packet_context_missing_count
        )
        source_context_frame_delta_value = _parse_optional_int(source_packet_context_frame_delta)
        source_context_time_delta_ms_value = _parse_optional_float(
            source_packet_context_time_delta_ms
        )
        if realtime_value is not None and last_realtime is not None:
            rt_gap = realtime_value - last_realtime
            if rt_gap > UNITY_TIME_GAP_SECONDS:
                logger.warning(
                    "[CAMERA_REALTIME_GAP] rt_gap=%.3fs frame_id=%s camera_id=%s",
                    rt_gap,
                    frame_id,
                    camera_key,
                )
        if unscaled_value is not None and last_unscaled is not None:
            us_gap = unscaled_value - last_unscaled
            if us_gap > UNITY_TIME_GAP_SECONDS:
                logger.warning(
                    "[CAMERA_UNSCALED_GAP] us_gap=%.3fs frame_id=%s camera_id=%s",
                    us_gap,
                    frame_id,
                    camera_key,
                )
        if ts_value is not None and realtime_value is not None:
            drift = ts_value - realtime_value
            if abs(drift) > 0.2:
                logger.warning(
                    "[CAMERA_TIME_DRIFT] ts=%.3f realtime=%.3f drift=%.3f frame_id=%s scale=%s camera_id=%s",
                    ts_value,
                    realtime_value,
                    drift,
                    frame_id,
                    time_scale or "n/a",
                    camera_key,
                )
        last_camera_realtime[camera_key] = realtime_value or last_realtime
        last_camera_unscaled[camera_key] = unscaled_value or last_unscaled

        try:
            frame_id_int = int(frame_id)
        except ValueError:
            frame_id_int = None
        if frame_id_int is not None and frame_id_int % 100 == 0:
            logger.info(
                "[CAMERA_SAMPLE] frame_id=%s ts=%s realtime=%s unscaled=%s scale=%s arrival=%.3f",
                frame_id,
                f"{ts_value:.3f}" if ts_value is not None else "n/a",
                f"{realtime_value:.3f}" if realtime_value is not None else "n/a",
                f"{unscaled_value:.3f}" if unscaled_value is not None else "n/a",
                time_scale or "n/a",
                now,
            )

        # Decode and store every frame to preserve deterministic capture ordering.
        global camera_decode_in_flight, camera_drop_count
        camera_decode_in_flight[camera_key] = True
        await _decode_and_store_camera_frame(
            image_data,
            timestamp,
            frame_id,
            camera_key,
            sync_packet_key=_normalize_packet_key(sync_packet_key),
            source_packet_owner=source_packet_owner,
            source_packet_owner_update_id=source_owner_update_id_value,
            source_packet_owner_unity_frame_count=source_owner_unity_frame_count_value,
            source_packet_owner_unity_time=source_owner_unity_time_value,
            source_bundle_close_reason=source_bundle_close_reason,
            source_bundle_deadline_ms=source_bundle_deadline_ms_value,
            source_bundle_age_ms=source_bundle_age_ms_value,
            source_bundle_inflight_count=source_bundle_inflight_count_value,
            source_bundle_camera_requested=(
                bool(source_bundle_camera_requested_value)
                if source_bundle_camera_requested_value is not None
                else None
            ),
            source_bundle_camera_request_attempted=(
                bool(source_bundle_camera_request_attempted_value)
                if source_bundle_camera_request_attempted_value is not None
                else None
            ),
            source_bundle_camera_request_accepted=(
                bool(source_bundle_camera_request_accepted_value)
                if source_bundle_camera_request_accepted_value is not None
                else None
            ),
            source_bundle_camera_request_rejected_reason=(
                str(source_bundle_camera_request_rejected_reason or "")
            ),
            source_bundle_camera_request_skipped_reason=(
                str(source_bundle_camera_request_skipped_reason or "")
            ),
            source_bundle_camera_request_disposition_code=(
                str(source_bundle_camera_request_disposition_code or "")
            ),
            source_bundle_camera_request_attempt_age_ms=(
                source_bundle_camera_request_attempt_age_ms_value
            ),
            source_bundle_camera_request_accept_age_ms=(
                source_bundle_camera_request_accept_age_ms_value
            ),
            source_bundle_camera_request_queue_depth=(
                source_bundle_camera_request_queue_depth_value
            ),
            source_bundle_active_transport_eligible=(
                bool(source_bundle_active_transport_eligible_value)
                if source_bundle_active_transport_eligible_value is not None
                else None
            ),
            source_bundle_debug_unbundled_capture=(
                bool(source_bundle_debug_unbundled_capture_value)
                if source_bundle_debug_unbundled_capture_value is not None
                else None
            ),
            camera_capture_contract_reason=(
                str(camera_capture_contract_reason or "")
            ),
            source_bundle_camera_sent=(
                bool(source_bundle_camera_sent_value)
                if source_bundle_camera_sent_value is not None
                else None
            ),
            source_bundle_aborted_before_vehicle_send=(
                bool(source_bundle_aborted_before_vehicle_send_value)
                if source_bundle_aborted_before_vehicle_send_value is not None
                else None
            ),
            source_bundle_abort_reason=(str(source_bundle_abort_reason or "")),
            source_bundle_vehicle_send_blocked_by_camera_request=(
                bool(source_bundle_vehicle_send_blocked_by_camera_request_value)
                if source_bundle_vehicle_send_blocked_by_camera_request_value is not None
                else None
            ),
            source_bundle_vehicle_state_built=(
                bool(source_bundle_vehicle_state_built_value)
                if source_bundle_vehicle_state_built_value is not None
                else None
            ),
            source_bundle_vehicle_state_enqueued=(
                bool(source_bundle_vehicle_state_enqueued_value)
                if source_bundle_vehicle_state_enqueued_value is not None
                else None
            ),
            source_bundle_vehicle_state_sent=(
                bool(source_bundle_vehicle_state_sent_value)
                if source_bundle_vehicle_state_sent_value is not None
                else None
            ),
            source_bundle_superseded_before_send=(
                bool(source_bundle_superseded_before_send_value)
                if source_bundle_superseded_before_send_value is not None
                else None
            ),
            source_packet_context_queue_depth=source_context_queue_depth_value,
            source_packet_context_dropped_stale_count=source_context_dropped_stale_count_value,
            source_packet_context_missing_count=source_context_missing_count_value,
            source_packet_context_frame_delta=source_context_frame_delta_value,
            source_packet_context_time_delta_ms=source_context_time_delta_ms_value,
            unity_frame_count=unity_frame_count_value,
            unity_time=unity_time_value,
            realtime_since_startup=realtime_value,
            unscaled_time=unscaled_value,
            arrival_wall_time=now,
        )
        response = {
            "status": "received",
            "frame_id": frame_id,
            "timestamp": timestamp,
            "camera_id": camera_key,
            "sync_packet_key": _normalize_packet_key(sync_packet_key),
            "unity_frame_count": unity_frame_count_value,
            "unity_time": unity_time_value,
            "dropped": False,
        }
        duration = time.time() - start_time
        if duration > SLOW_REQUEST_SECONDS:
            logger.warning(
                "[SLOW] /api/camera duration=%.3fs bytes=%d frame_id=%s ts=%s camera_id=%s",
                duration,
                len(image_data),
                frame_id,
                timestamp,
                camera_key,
            )
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.post("/api/vehicle/state")
async def receive_vehicle_state(state: VehicleState):
    """
    Receive vehicle state from Unity.
    
    Args:
        state: Vehicle state data
    """
    global latest_vehicle_state, vehicle_state_queue
    start_time = time.time()

    raw_state = state.model_dump()
    latest_vehicle_state, replacements = _sanitize_json_compatible(raw_state)
    _record_vehicle_state_sanitization(replacements, context="receive_vehicle_state")
    vehicle_state_queue.append(latest_vehicle_state)
    _attach_vehicle_component(
        packet_key=_normalize_packet_key(state.syncPacketKey),
        unity_frame_count=_parse_optional_int(state.unityFrameCount),
        unity_time=_parse_optional_float(state.unityTime),
        component={
            "request_id": state.requestId,
            "sync_packet_key": _normalize_packet_key(state.syncPacketKey),
            "sync_packet_camera_frame_id": _parse_optional_int(state.syncPacketCameraFrameId),
            "sync_packet_camera_unity_frame_count": _parse_optional_int(
                state.syncPacketCameraUnityFrameCount
            ),
            "sync_packet_camera_timestamp": _parse_optional_float(state.syncPacketCameraTimestamp),
            "source_bundle_close_reason": str(state.syncPacketSourceBundleCloseReason or ""),
            "source_bundle_deadline_ms": _parse_optional_float(state.syncPacketSourceBundleDeadlineMs),
            "source_bundle_age_ms": _parse_optional_float(state.syncPacketSourceBundleAgeMs),
            "source_bundle_inflight_count": _parse_optional_int(state.syncPacketSourceBundleInflightCount),
            "source_bundle_vehicle_state_built": bool(state.syncPacketSourceVehicleStateBuilt),
            "source_bundle_vehicle_state_enqueued": bool(state.syncPacketSourceVehicleStateEnqueued),
            "source_bundle_vehicle_state_sent": bool(state.syncPacketSourceVehicleStateSent),
            "source_bundle_camera_requested": bool(state.syncPacketSourceCameraRequested),
            "source_bundle_camera_request_attempted": bool(
                state.syncPacketSourceCameraRequestAttempted
            ),
            "source_bundle_camera_request_accepted": bool(
                state.syncPacketSourceCameraRequestAccepted
            ),
            "source_bundle_camera_request_rejected_reason": str(
                state.syncPacketSourceCameraRequestRejectedReason or ""
            ),
            "source_bundle_camera_request_skipped_reason": str(
                state.syncPacketSourceCameraRequestSkippedReason or ""
            ),
            "source_bundle_camera_request_disposition_code": str(
                state.syncPacketSourceCameraRequestDispositionCode or ""
            ),
            "source_bundle_camera_request_attempt_age_ms": _parse_optional_float(
                state.syncPacketSourceCameraRequestAttemptAgeMs
            ),
            "source_bundle_camera_request_accept_age_ms": _parse_optional_float(
                state.syncPacketSourceCameraRequestAcceptAgeMs
            ),
            "source_bundle_camera_request_queue_depth": _parse_optional_int(
                state.syncPacketSourceCameraRequestQueueDepth
            ),
            "source_bundle_camera_sent": bool(state.syncPacketSourceCameraSent),
            "source_bundle_aborted_before_vehicle_send": bool(
                state.syncPacketSourceBundleAbortedBeforeVehicleSend
            ),
            "source_bundle_abort_reason": str(
                state.syncPacketSourceBundleAbortReason or ""
            ),
            "source_bundle_vehicle_send_blocked_by_camera_request": bool(
                state.syncPacketSourceVehicleSendBlockedByCameraRequest
            ),
            "source_bundle_superseded_before_send": bool(state.syncPacketSourceSupersededBeforeSend),
            "unity_frame_count": _parse_optional_int(state.unityFrameCount),
            "unity_time": _parse_optional_float(state.unityTime),
            "arrival_wall_time": time.time(),
            "state": dict(latest_vehicle_state or {}),
        },
        now=time.time(),
    )

    global last_state_arrival_time, last_unity_send_realtime, last_unity_time, speed_limit_zero_streak
    duration = time.time() - start_time
    now = time.time()
    if last_state_arrival_time is not None:
        gap = now - last_state_arrival_time
        if gap > SLOW_REQUEST_SECONDS:
            logger.warning(
                "[ARRIVAL_GAP] /api/vehicle/state gap=%.3fs request_id=%s unity_frame=%s unity_time=%.3f",
                gap,
                state.requestId,
                state.unityFrameCount,
                state.unityTime,
            )
            print(
                "[ARRIVAL_GAP] /api/vehicle/state "
                f"gap={gap:.3f}s request_id={state.requestId} "
                f"unity_frame={state.unityFrameCount} unity_time={state.unityTime:.3f}"
            )
    last_state_arrival_time = now
    if last_unity_time is not None:
        unity_time_gap = state.unityTime - last_unity_time
        if unity_time_gap > UNITY_TIME_GAP_SECONDS:
            logger.warning(
                "[UNITY_TIME_GAP] unity_time_gap=%.3fs request_id=%s unity_frame=%s",
                unity_time_gap,
                state.requestId,
                state.unityFrameCount,
            )
            print(
                "[UNITY_TIME_GAP] "
                f"unity_time_gap={unity_time_gap:.3f}s request_id={state.requestId} "
                f"unity_frame={state.unityFrameCount}"
            )

    if state.unitySendRealtime > 0.0 and last_unity_send_realtime is not None:
        send_gap = state.unitySendRealtime - last_unity_send_realtime
        unity_time_gap = state.unityTime - (last_unity_time or 0.0)
        drift = unity_time_gap - send_gap
        if abs(drift) > 0.2:
            logger.warning(
                "[TIME_DRIFT] unity_time_gap=%.3fs send_gap=%.3fs drift=%.3fs "
                "request_id=%s unity_frame=%s",
                unity_time_gap,
                send_gap,
                drift,
                state.requestId,
                state.unityFrameCount,
            )
            print(
                "[TIME_DRIFT] "
                f"unity_time_gap={unity_time_gap:.3f}s send_gap={send_gap:.3f}s "
                f"drift={drift:.3f}s request_id={state.requestId} "
                f"unity_frame={state.unityFrameCount}"
            )
    if state.unitySendRealtime > 0.0:
        last_unity_send_realtime = state.unitySendRealtime
        last_unity_time = state.unityTime
    if state.unitySendUtcMs:
        now_ms = int(time.time() * 1000)
        arrival_delay_ms = now_ms - int(state.unitySendUtcMs)
        if arrival_delay_ms > 100:
            logger.warning(
                "[ARRIVAL] /api/vehicle/state delay=%dms request_id=%s unity_frame=%s unity_time=%.3f",
                arrival_delay_ms,
                state.requestId,
                state.unityFrameCount,
                state.unityTime,
            )
            print(
                "[ARRIVAL] /api/vehicle/state "
                f"delay={arrival_delay_ms}ms request_id={state.requestId} "
                f"unity_frame={state.unityFrameCount} unity_time={state.unityTime:.3f}"
            )
    if duration > SLOW_REQUEST_SECONDS:
        logger.warning(
            "[SLOW] /api/vehicle/state duration=%.3fs unity_frame=%s unity_time=%.3f",
            duration,
            state.unityFrameCount,
            state.unityTime,
        )

    if state.speedLimit <= 0.01:
        speed_limit_zero_streak += 1
        if speed_limit_zero_streak % 60 == 0:
            logger.warning(
                "[SPEED_LIMIT_MISSING] streak=%d unity_frame=%s unity_time=%.3f",
                speed_limit_zero_streak,
                state.unityFrameCount,
                state.unityTime,
            )
            print(
                "[SPEED_LIMIT_MISSING] "
                f"streak={speed_limit_zero_streak} unity_frame={state.unityFrameCount} "
                f"unity_time={state.unityTime:.3f}"
            )
    else:
        speed_limit_zero_streak = 0

    return {"status": "received"}


@app.get("/api/vehicle/control")
async def get_control_command(request_id: int = 0, unity_send_utc_ms: int = 0):
    """
    Get latest control command for Unity.
    
    Returns:
        Control command (steering, throttle, brake)
    """
    global latest_control_command, last_control_arrival_time
    start_time = time.time()
    now = time.time()
    if last_control_arrival_time is not None:
        gap = now - last_control_arrival_time
        if gap > SLOW_REQUEST_SECONDS:
            logger.warning(
                "[ARRIVAL_GAP] /api/vehicle/control gap=%.3fs request_id=%s",
                gap,
                request_id,
            )
            print(
                "[ARRIVAL_GAP] /api/vehicle/control "
                f"gap={gap:.3f}s request_id={request_id}"
            )
    last_control_arrival_time = now

    if unity_send_utc_ms:
        now_ms = int(time.time() * 1000)
        arrival_delay_ms = now_ms - int(unity_send_utc_ms)
        if arrival_delay_ms > 100:
            logger.warning(
                "[ARRIVAL] /api/vehicle/control delay=%dms request_id=%s",
                arrival_delay_ms,
                request_id,
            )
            print(
                "[ARRIVAL] /api/vehicle/control "
                f"delay={arrival_delay_ms}ms request_id={request_id}"
            )
    
    if latest_control_command is None:
        # Return neutral command if no command available
        response = {
            "steering": 0.0,
            "throttle": 0.0,
            "brake": 0.0,
            "ground_truth_mode": False,
            "ground_truth_speed": 5.0,
            "randomize_start": False,
            "randomize_request_id": 0,
            "randomize_seed": None,
        }
        duration = time.time() - start_time
        if duration > SLOW_REQUEST_SECONDS:
            logger.warning(
                "[SLOW] /api/vehicle/control duration=%.3fs (default)",
                duration,
            )
        return response
    
    # Ensure ground truth fields are present (for backward compatibility)
    result = latest_control_command.copy()
    if "ground_truth_mode" not in result:
        result["ground_truth_mode"] = False
    if "ground_truth_speed" not in result:
        result["ground_truth_speed"] = 5.0
    if "randomize_start" not in result:
        result["randomize_start"] = False
    if "randomize_request_id" not in result:
        result["randomize_request_id"] = 0
    if "randomize_seed" not in result:
        result["randomize_seed"] = None
    
    duration = time.time() - start_time
    if duration > SLOW_REQUEST_SECONDS:
        logger.warning(
            "[SLOW] /api/vehicle/control duration=%.3fs",
            duration,
        )
    return result


@app.post("/api/vehicle/control")
async def set_control_command(command: ControlCommand):
    """
    Set control command (called by AV stack).
    
    Args:
        command: Control command
    """
    global latest_control_command
    
    latest_control_command = command.model_dump()
    
    return {"status": "set"}


@app.get("/api/camera/latest")
async def get_latest_camera_frame(camera_id: str = "front_center"):
    """
    Get latest camera frame (for AV stack processing).
    
    Returns:
        Base64 encoded image
    """
    global latest_camera_frame, latest_camera_frames
    camera_key = _normalize_camera_id(camera_id)
    frame = latest_camera_frames.get(camera_key)
    if frame is None and camera_key == "front_center":
        frame = latest_camera_frame
    if frame is None:
        raise HTTPException(status_code=404, detail="No camera frame available")
    
    # Convert to JPEG
    img = Image.fromarray(frame)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    img_bytes = buffer.getvalue()
    
    # Encode to base64
    img_base64 = base64.b64encode(img_bytes).decode()
    
    frame_id_value = None
    unity_frame_value = None
    unity_time_value = None
    try:
        raw_frame_id = latest_frame_ids_by_id.get(camera_key)
        if raw_frame_id is None and camera_key == "front_center":
            raw_frame_id = latest_frame_id
        frame_id_value = int(raw_frame_id) if raw_frame_id is not None else None
    except (TypeError, ValueError):
        frame_id_value = None
    timestamp_value = latest_frame_timestamps_by_id.get(camera_key)
    if timestamp_value is None and camera_key == "front_center":
        timestamp_value = latest_frame_timestamp
    queue_obj = camera_frame_queues.get(camera_key)
    if queue_obj is not None and len(queue_obj) > 0:
        latest_item = queue_obj[-1]
        unity_frame_value = _parse_optional_int(latest_item.get("unity_frame_count"))
        unity_time_value = _parse_optional_float(latest_item.get("unity_time"))
    queue_depth = len(queue_obj) if queue_obj is not None else 0
    queue_capacity = int(queue_obj.maxlen) if queue_obj is not None and queue_obj.maxlen is not None else MAX_CAMERA_QUEUE_SIZE
    drop_count = int(camera_drop_count.get(camera_key, 0))
    decode_in_flight = bool(camera_decode_in_flight.get(camera_key, False))
    arrival_time = last_camera_arrival_time.get(camera_key)
    now = time.time()
    age_ms = (now - arrival_time) * 1000.0 if arrival_time is not None else None
    last_realtime = last_camera_realtime.get(camera_key)
    last_unscaled = last_camera_unscaled.get(camera_key)
    return {
        "image": img_base64,
        "timestamp": timestamp_value,
        "frame_id": frame_id_value,
        "unity_frame_count": unity_frame_value,
        "unity_time": unity_time_value,
        "camera_id": camera_key,
        "shape": list(frame.shape),
        # Stream freshness/queue diagnostics (instrumentation-only)
        "queue_depth": queue_depth,
        "queue_capacity": queue_capacity,
        "drop_count": drop_count,
        "decode_in_flight": decode_in_flight,
        "latest_age_ms": age_ms,
        "last_arrival_time": arrival_time,
        "last_realtime_since_startup": last_realtime,
        "last_unscaled_time": last_unscaled,
    }


@app.get("/api/camera/latest/raw")
async def get_latest_camera_frame_raw(camera_id: str = "front_center"):
    """
    Latest frame as raw RGB uint8 (HxWx3) octet-stream.
    Metadata in X-AV-* headers — avoids JPEG re-encode for the Python AV stack client.
    """
    global latest_camera_frame, latest_camera_frames
    camera_key = _normalize_camera_id(camera_id)
    frame = latest_camera_frames.get(camera_key)
    if frame is None and camera_key == "front_center":
        frame = latest_camera_frame
    if frame is None:
        raise HTTPException(status_code=404, detail="No camera frame available")

    frame_id_value = None
    unity_frame_value = None
    unity_time_value = None
    try:
        raw_frame_id = latest_frame_ids_by_id.get(camera_key)
        if raw_frame_id is None and camera_key == "front_center":
            raw_frame_id = latest_frame_id
        frame_id_value = int(raw_frame_id) if raw_frame_id is not None else None
    except (TypeError, ValueError):
        frame_id_value = None
    timestamp_value = latest_frame_timestamps_by_id.get(camera_key)
    if timestamp_value is None and camera_key == "front_center":
        timestamp_value = latest_frame_timestamp
    queue_obj = camera_frame_queues.get(camera_key)
    if queue_obj is not None and len(queue_obj) > 0:
        latest_item = queue_obj[-1]
        unity_frame_value = _parse_optional_int(latest_item.get("unity_frame_count"))
        unity_time_value = _parse_optional_float(latest_item.get("unity_time"))
    queue_depth = len(queue_obj) if queue_obj is not None else 0
    queue_capacity = int(queue_obj.maxlen) if queue_obj is not None and queue_obj.maxlen is not None else MAX_CAMERA_QUEUE_SIZE
    drop_count = int(camera_drop_count.get(camera_key, 0))
    decode_in_flight = bool(camera_decode_in_flight.get(camera_key, False))
    arrival_time = last_camera_arrival_time.get(camera_key)
    now = time.time()
    age_ms = (now - arrival_time) * 1000.0 if arrival_time is not None else None
    last_realtime = last_camera_realtime.get(camera_key)
    last_unscaled = last_camera_unscaled.get(camera_key)

    return _raw_camera_response(
        frame,
        camera_key=camera_key,
        timestamp_value=timestamp_value,
        frame_id_value=frame_id_value,
        unity_frame_count=unity_frame_value,
        unity_time=unity_time_value,
        queue_depth=queue_depth,
        queue_capacity=queue_capacity,
        drop_count=drop_count,
        decode_in_flight=decode_in_flight,
        latest_age_ms=age_ms,
        last_arrival_time=arrival_time,
        last_realtime=last_realtime,
        last_unscaled=last_unscaled,
    )


@app.get("/api/camera/next")
async def get_next_camera_frame(camera_id: str = "front_center"):
    """Get next queued camera frame in FIFO order."""
    camera_key = _normalize_camera_id(camera_id)
    queue = camera_frame_queues.get(camera_key)
    if queue is None or len(queue) == 0:
        raise HTTPException(status_code=404, detail="No queued camera frame available")

    item = queue.popleft()
    frame = item["image"]

    img = Image.fromarray(frame)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode()

    frame_id_value = None
    try:
        frame_id_value = int(item.get("frame_id")) if item.get("frame_id") is not None else None
    except (TypeError, ValueError):
        frame_id_value = None

    return {
        "image": img_base64,
        "timestamp": item.get("timestamp"),
        "frame_id": frame_id_value,
        "unity_frame_count": _parse_optional_int(item.get("unity_frame_count")),
        "unity_time": _parse_optional_float(item.get("unity_time")),
        "camera_id": camera_key,
        "shape": list(frame.shape),
        "queue_remaining": len(queue),
    }


@app.post("/api/camera/drain_queue")
async def drain_camera_queue(camera_id: str = "front_center"):
    """Clear queued frames for a camera (latest_* snapshots unchanged). Reduces backlog latency."""
    camera_key = _normalize_camera_id(camera_id)
    q = camera_frame_queues.get(camera_key)
    n = len(q) if q is not None else 0
    if q is not None:
        q.clear()
    return {"drained": n, "camera_id": camera_key}


@app.get("/api/camera/next/raw")
async def get_next_camera_frame_raw(camera_id: str = "front_center"):
    """FIFO camera frame as raw RGB uint8 (see /api/camera/latest/raw)."""
    camera_key = _normalize_camera_id(camera_id)
    queue = camera_frame_queues.get(camera_key)
    if queue is None or len(queue) == 0:
        raise HTTPException(status_code=404, detail="No queued camera frame available")

    item = queue.popleft()
    frame = item["image"]
    remaining = len(queue)
    queue_obj = camera_frame_queues.get(camera_key)
    queue_depth = len(queue_obj) if queue_obj is not None else 0
    queue_capacity = int(queue_obj.maxlen) if queue_obj is not None and queue_obj.maxlen is not None else MAX_CAMERA_QUEUE_SIZE
    drop_count = int(camera_drop_count.get(camera_key, 0))
    decode_in_flight = bool(camera_decode_in_flight.get(camera_key, False))
    arrival_time = last_camera_arrival_time.get(camera_key)
    now = time.time()
    age_ms = (now - arrival_time) * 1000.0 if arrival_time is not None else None
    last_realtime = last_camera_realtime.get(camera_key)
    last_unscaled = last_camera_unscaled.get(camera_key)

    frame_id_value = None
    try:
        frame_id_value = int(item.get("frame_id")) if item.get("frame_id") is not None else None
    except (TypeError, ValueError):
        frame_id_value = None

    return _raw_camera_response(
        frame,
        camera_key=camera_key,
        timestamp_value=float(item.get("timestamp")) if item.get("timestamp") is not None else None,
        frame_id_value=frame_id_value,
        unity_frame_count=_parse_optional_int(item.get("unity_frame_count")),
        unity_time=_parse_optional_float(item.get("unity_time")),
        queue_depth=queue_depth,
        queue_capacity=queue_capacity,
        drop_count=drop_count,
        decode_in_flight=decode_in_flight,
        latest_age_ms=age_ms,
        last_arrival_time=arrival_time,
        last_realtime=last_realtime,
        last_unscaled=last_unscaled,
        queue_remaining=remaining,
    )


@app.get("/api/vehicle/state/latest")
async def get_latest_vehicle_state():
    """
    Get latest vehicle state (for AV stack processing).
    
    Returns:
        Vehicle state data
    """
    global latest_vehicle_state
    
    if latest_vehicle_state is None:
        raise HTTPException(status_code=404, detail="No vehicle state available")

    sanitized_state, replacements = _sanitize_json_compatible(latest_vehicle_state)
    _record_vehicle_state_sanitization(replacements, context="get_latest_vehicle_state")
    return sanitized_state


@app.get("/api/vehicle/state/next")
async def get_next_vehicle_state():
    """Get next queued vehicle state in FIFO order."""
    if len(vehicle_state_queue) == 0:
        raise HTTPException(status_code=404, detail="No queued vehicle state available")
    state = vehicle_state_queue.popleft()
    sanitized_state, replacements = _sanitize_json_compatible(state)
    _record_vehicle_state_sanitization(replacements, context="get_next_vehicle_state")
    return sanitized_state


@app.get("/api/sim/packet/latest")
async def get_latest_sync_packet():
    """Return the most recent complete synchronized front-camera + vehicle-state packet."""
    if latest_sync_packet_data is None:
        raise HTTPException(status_code=404, detail="No synchronized packet available")
    return _snapshot_sync_packet(latest_sync_packet_data)


@app.get("/api/sim/packet/next")
async def get_next_sync_packet():
    """Return the next synchronized packet in FIFO order."""
    if len(assembled_packet_queue) == 0:
        raise HTTPException(status_code=404, detail="No synchronized packet available")
    packet = assembled_packet_queue.popleft()
    return _snapshot_sync_packet(packet)


@app.get("/api/sim/packet/latest/raw")
async def get_latest_sync_packet_raw():
    """Return the most recent synchronized packet with raw image payload and full vehicle state."""
    if latest_sync_packet_payload_data is None:
        raise HTTPException(status_code=404, detail="No synchronized payload packet available")
    return _sync_packet_payload_response(latest_sync_packet_payload_data)


@app.get("/api/sim/packet/next/raw")
async def get_next_sync_packet_raw(
    consume_policy: str = PACKET_CONSUME_POLICY_FIFO_STRICT,
    max_age_ms: float = 250.0,
    warn_age_ms: float = 125.0,
    max_drain_count: int = 128,
    allow_stale_debug: bool = False,
):
    """Return a synchronized payload packet selected according to the requested consume policy."""
    packet, selection_meta = _select_sync_packet_payload(
        consume_policy=consume_policy,
        max_age_ms=max_age_ms,
        warn_age_ms=warn_age_ms,
        max_drain_count=max_drain_count,
        allow_stale_debug=allow_stale_debug,
    )
    if packet is None:
        wait_budget_s = max(0.0, float(ACTIVE_SELECTION_PARTIAL_WAIT_MS) / 1000.0)
        poll_s = max(0.001, float(ACTIVE_SELECTION_PARTIAL_POLL_MS) / 1000.0)
        deadline = time.monotonic() + wait_budget_s
        pending_candidate = _select_pending_partial_candidate(time.time())
        while (
            pending_candidate is not None
            and bool(pending_candidate.get("waitable", False))
            and time.monotonic() < deadline
        ):
            await asyncio.sleep(poll_s)
            packet, selection_meta = _select_sync_packet_payload(
                consume_policy=consume_policy,
                max_age_ms=max_age_ms,
                warn_age_ms=warn_age_ms,
                max_drain_count=max_drain_count,
                allow_stale_debug=allow_stale_debug,
            )
            if packet is not None:
                return _sync_packet_payload_response_with_selection(packet, selection_meta)
            pending_candidate = _select_pending_partial_candidate(time.time())
        if pending_candidate is not None:
            selection_meta = _overlay_pending_partial_meta(selection_meta, pending_candidate)
        raise _selection_404("No synchronized payload packet available", selection_meta)
    return _sync_packet_payload_response_with_selection(packet, selection_meta)


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "has_camera_frame": latest_camera_frames.get("front_center") is not None or latest_camera_frame is not None,
        "has_vehicle_state": latest_vehicle_state is not None,
        "has_sync_packet": latest_sync_packet_data is not None,
        "has_sync_packet_payload": latest_sync_packet_payload_data is not None,
        "vehicle_state_sanitize_events": vehicle_state_sanitize_events,
        "vehicle_state_sanitize_values": vehicle_state_sanitize_values,
        "max_camera_queue_size": MAX_CAMERA_QUEUE_SIZE,
        "sync_packet_queue_depth": len(assembled_packet_queue),
        "sync_packet_drop_count": sync_packet_drop_count,
        "sync_packet_payload_queue_depth": len(assembled_packet_payload_queue),
        "sync_packet_payload_drop_count": sync_packet_payload_drop_count,
        "sync_packet_orphan_camera_count": sync_packet_orphan_camera_count,
        "sync_packet_orphan_vehicle_count": sync_packet_orphan_vehicle_count,
        "sync_packet_timeout_count": sync_packet_timeout_count,
        "sync_packet_key_match_count": sync_packet_key_match_count,
        "sync_packet_unity_fallback_count": sync_packet_unity_fallback_count,
        "sync_packet_superseded_camera_count": sync_packet_superseded_camera_count,
        "sync_packet_superseded_vehicle_count": sync_packet_superseded_vehicle_count,
    }


@app.post("/api/shutdown")
async def shutdown_signal():
    """
    Signal that AV stack is shutting down.
    Unity can poll this to detect when to exit play mode.
    """
    global shutdown_requested
    shutdown_requested = True
    return {
        "status": "shutdown",
        "message": "AV stack is shutting down",
        "timestamp": time.time()
    }


@app.get("/api/shutdown")
async def check_shutdown():
    """
    Check if AV stack is shutting down.
    Unity polls this to detect when to exit play mode.
    Returns shutdown status only if shutdown was requested.
    """
    global shutdown_requested
    if shutdown_requested:
        return {
            "status": "shutdown",
            "message": "AV stack is shutting down",
            "timestamp": time.time()
        }
    else:
        return {
            "status": "running",
            "message": "AV stack is running",
            "timestamp": time.time()
        }


@app.post("/api/unity/play")
async def request_play():
    """
    Request Unity to start playing (enter play mode).
    Unity can poll this to detect when to enter play mode.
    """
    global play_requested
    play_requested = True
    return {
        "status": "play_requested",
        "message": "Unity play mode requested",
        "timestamp": time.time()
    }


@app.get("/api/unity/play")
async def check_play_request():
    """
    Check if Unity should start playing.
    Unity polls this to detect when to enter play mode.
    Returns play status only if play was requested.
    """
    global play_requested
    if play_requested:
        # Reset flag after reading (one-time request)
        play_requested = False
        return {
            "status": "play",
            "message": "Unity should enter play mode",
            "timestamp": time.time()
        }
    else:
        return {
            "status": "no_action",
            "message": "No play request",
            "timestamp": time.time()
        }


@app.post("/api/trajectory")
async def set_trajectory_data(trajectory: dict):
    """
    Set trajectory data (called by AV stack for visualization).
    
    Args:
        trajectory: Trajectory data with points, reference_point, lateral_error
    """
    global latest_trajectory_data
    latest_trajectory_data = trajectory
    return {"status": "received"}


@app.get("/api/trajectory")
async def get_trajectory_data():
    """
    Get latest trajectory data for Unity visualization.
    
    Returns:
        Trajectory data with points, reference_point, lateral_error
    """
    global latest_trajectory_data
    
    if latest_trajectory_data is None:
        # Return empty trajectory if no data available
        return {
            "trajectory_points": [],
            "reference_point": [0.0, 0.0, 0.0, 0.0],
            "lateral_error": 0.0,
            "timestamp": time.time()
        }
    
    return latest_trajectory_data


@app.post("/api/groundtruth/mode")
async def set_ground_truth_mode(mode: GroundTruthMode):
    """
    Enable or disable ground truth mode in Unity CarController.
    
    Ground truth mode uses direct velocity control for precise path following,
    bypassing physics forces. This is ideal for collecting ground truth data.
    
    Args:
        mode: Ground truth mode configuration (enabled, speed)
    """
    # This will be handled by Unity's AVBridge when it polls for control commands
    # We'll store it in the control command so Unity can read it
    global latest_control_command
    
    # Store ground truth mode in control command metadata
    if latest_control_command is None:
        latest_control_command = {
            "steering": 0.0,
            "throttle": 0.0,
            "brake": 0.0
        }
    
    latest_control_command["ground_truth_mode"] = mode.enabled
    latest_control_command["ground_truth_speed"] = mode.speed
    
    return {
        "status": "set",
        "ground_truth_mode": mode.enabled,
        "speed": mode.speed
    }


@app.post("/api/unity/feedback")
async def receive_unity_feedback(feedback: UnityFeedback):
    """
    Receive Unity feedback/status data.
    
    Args:
        feedback: Unity feedback data
    """
    global latest_unity_feedback
    
    latest_unity_feedback = feedback.model_dump()
    
    return {"status": "received"}


@app.get("/api/unity/feedback/latest")
async def get_latest_unity_feedback():
    """Get latest Unity feedback/status payload."""
    if latest_unity_feedback is None:
        raise HTTPException(status_code=404, detail="No Unity feedback available")
    return latest_unity_feedback


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the bridge server."""
    print(f"Starting AV Stack Bridge Server on {host}:{port}")
    print(f"Camera queue maxlen per stream: {MAX_CAMERA_QUEUE_SIZE} (AV_BRIDGE_MAX_CAMERA_QUEUE)")
    print("Endpoints:")
    print("  POST /api/camera - Receive camera frame from Unity")
    print("  POST /api/vehicle/state - Receive vehicle state from Unity")
    print("  GET  /api/vehicle/control - Get control command for Unity")
    print("  POST /api/vehicle/control - Set control command")
    print("  GET  /api/camera/latest - Get latest camera frame")
    print("  GET  /api/camera/latest/raw - Latest frame raw RGB (octet-stream, X-AV-* headers)")
    print("  GET  /api/camera/next/raw - Next queued frame raw RGB")
    print("  POST /api/camera/drain_queue - Clear FIFO queue (latest snapshot unchanged)")
    print("  GET  /api/vehicle/state/latest - Get latest vehicle state")
    print("  GET  /api/health - Health check")
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
