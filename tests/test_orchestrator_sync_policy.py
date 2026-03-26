import numpy as np

from av_stack.orchestrator import AVStack
from av_stack.speed_helpers import _classify_teleport_discontinuity
from sync_contract import (
    PACKET_CONSUME_POLICY_FRESHEST_WITHIN_BUDGET,
    PACKET_FALLBACK_PACKET_LEGACY_PATH,
    PACKET_MODE_LATEST_PARALLEL,
    PACKET_MODE_PACKET_FIFO_ACTIVE,
    PACKET_MODE_PACKET_SHADOW,
)


def test_apply_sync_packet_shadow_populates_vehicle_state_fields() -> None:
    stack = AVStack.__new__(AVStack)
    stack.sync_packet_mode = PACKET_MODE_PACKET_SHADOW
    vehicle_state_dict = {"unityFrameCount": 130}
    stack._apply_sync_packet_shadow(
        vehicle_state_dict,
        {
            "schema_version": 1,
            "packet_id": 44,
            "unity_frame_count": 128,
            "complete": True,
            "fallback_active": False,
            "fallback_reason_code": "none",
            "queue_depth": 2,
            "drop_count": 3,
            "orphan_camera_count": 4,
            "orphan_vehicle_count": 5,
            "timeout_count": 6,
            "packet_age_ms": 8.0,
            "front_age_ms": 3.0,
            "vehicle_age_ms": 4.0,
            "front_vehicle_frame_delta": 0,
            "front_vehicle_time_delta_ms": 1.5,
            "missing_front": False,
            "missing_vehicle": False,
        },
    )

    assert vehicle_state_dict["sync_packet_mode"] == PACKET_MODE_PACKET_SHADOW
    assert vehicle_state_dict["sync_packet_schema_version"] == 1
    assert vehicle_state_dict["sync_packet_id"] == 44
    assert vehicle_state_dict["sync_packet_skipped_unity_frames"] == 2
    assert vehicle_state_dict["sync_packet_fallback_reason_code"] == "none"


def test_apply_sync_packet_shadow_marks_legacy_fallback_without_packet() -> None:
    stack = AVStack.__new__(AVStack)
    stack.sync_packet_mode = PACKET_MODE_LATEST_PARALLEL
    vehicle_state_dict = {"unityFrameCount": 10}

    stack._apply_sync_packet_shadow(vehicle_state_dict, None)

    assert vehicle_state_dict["sync_packet_fallback_active"] is True
    assert (
        vehicle_state_dict["sync_packet_fallback_reason_code"]
        == PACKET_FALLBACK_PACKET_LEGACY_PATH
    )


def test_compute_sync_packet_continuity_uses_last_consumed_unity_frame() -> None:
    stack = AVStack.__new__(AVStack)
    stack.last_consumed_sync_packet_id = 7
    stack.last_consumed_sync_packet_unity_frame = 100

    skipped = stack._compute_sync_packet_continuity({"packet_id": 8, "unity_frame_count": 104})

    assert skipped == 3
    assert stack.last_consumed_sync_packet_id == 8
    assert stack.last_consumed_sync_packet_unity_frame == 104


def test_apply_sync_packet_shadow_prefers_packet_continuity_skip_count_in_active_mode() -> None:
    stack = AVStack.__new__(AVStack)
    stack.sync_packet_mode = PACKET_MODE_PACKET_FIFO_ACTIVE
    vehicle_state_dict = {"unityFrameCount": 130}

    stack._apply_sync_packet_shadow(
        vehicle_state_dict,
        {
            "schema_version": 1,
            "packet_id": 55,
            "unity_frame_count": 130,
            "complete": True,
            "continuity_skipped_unity_frames": 6,
        },
    )

    assert vehicle_state_dict["sync_packet_mode"] == PACKET_MODE_PACKET_FIFO_ACTIVE
    assert vehicle_state_dict["sync_packet_skipped_unity_frames"] == 6


def test_capture_active_sync_packet_inputs_uses_payload_endpoint() -> None:
    stack = AVStack.__new__(AVStack)
    stack.sync_packet_mode = PACKET_MODE_PACKET_FIFO_ACTIVE
    stack.last_consumed_sync_packet_id = None
    stack.last_consumed_sync_packet_unity_frame = None
    stack._sync_packet_consume_policy = PACKET_CONSUME_POLICY_FRESHEST_WITHIN_BUDGET
    stack._sync_packet_payload_max_age_ms = 250.0
    stack._sync_packet_payload_warn_age_ms = 125.0
    stack._sync_packet_payload_max_drain_count = 128
    stack._sync_packet_payload_allow_stale_debug = False

    class _Bridge:
        @staticmethod
        def get_fresh_sync_packet_payload_raw(**kwargs):
            assert kwargs["consume_policy"] == PACKET_CONSUME_POLICY_FRESHEST_WITHIN_BUDGET
            return {
                "payload": {
                    "image": "image-bytes",
                    "timestamp": 5.0,
                    "frame_id": 91,
                    "camera_meta": {"unity_frame_count": 77, "camera_id": "front_center"},
                    "vehicle_state_dict": {"unityFrameCount": 77, "speed": 8.0},
                    "packet_meta": {
                    "schema_version": 1,
                    "packet_id": 12,
                    "unity_frame_count": 77,
                    "complete": True,
                    "join_source": "packet_key",
                    "join_key_present": True,
                    "join_wait_ms": 18.0,
                    "key_match_count": 9,
                    "unity_fallback_count": 1,
                    "superseded_camera_count": 2,
                    "superseded_vehicle_count": 3,
                    "packet_superseded_camera_count": 0,
                    "packet_superseded_vehicle_count": 1,
                    "payload_queue_depth": 4,
                    "payload_drop_count": 2,
                },
            },
                "consume_policy": PACKET_CONSUME_POLICY_FRESHEST_WITHIN_BUDGET,
                "selected_age_ms": 42.0,
                "selected_fresh": True,
                "warn_age_exceeded": False,
                "stale_drop_count": 3,
                "drained_count": 3,
                "max_drained_age_ms": 211.0,
                "selection_source": "server_selector",
                "server_queue_depth_after_select": 0,
                "server_oldest_age_ms_after_select": float("nan"),
            }

    stack.bridge = _Bridge()

    captured = stack._capture_active_sync_packet_inputs()

    assert captured is not None
    assert captured["frame_data"] == (
        "image-bytes",
        5.0,
        91,
        {"unity_frame_count": 77, "camera_id": "front_center"},
    )
    assert captured["vehicle_state_dict"]["unityFrameCount"] == 77
    assert captured["packet"]["packet_id"] == 12
    assert captured["packet"]["continuity_skipped_unity_frames"] == 0
    assert captured["packet"]["payload_selected_age_ms"] == 42.0
    assert captured["packet"]["payload_stale_drop_count"] == 3
    assert captured["packet"]["payload_selection_source"] == "server_selector"
    assert captured["packet"]["payload_server_queue_depth_after_select"] == 0
    assert captured["packet"]["join_source"] == "packet_key"
    assert captured["packet"]["join_wait_ms"] == 18.0


def test_classify_teleport_discontinuity_suppresses_continuity_backed_motion() -> None:
    result = _classify_teleport_discontinuity(
        np.array([0.0, 0.0], dtype=float),
        np.array([2.1, 0.0], dtype=float),
        base_distance_threshold_m=2.0,
        speed_mps=15.0,
        control_dt_s=0.08,
        unity_dt_s=0.15,
        skipped_unity_frames=8,
        fallback_active=False,
        payload_selected_fresh=True,
        payload_selected_age_ms=50.0,
        expected_motion_ratio_threshold=1.8,
        expected_motion_margin_m=0.75,
        hard_override_distance_m=8.0,
        hard_override_ratio_threshold=4.0,
        continuity_skip_frames_threshold=3,
        continuity_warn_age_ms=125.0,
    )

    assert result["teleport_detected"] is False
    assert result["guard_suppressed"] is True
    assert result["continuity_suspect"] is True


def test_classify_teleport_discontinuity_hard_override_still_triggers() -> None:
    result = _classify_teleport_discontinuity(
        np.array([0.0, 0.0], dtype=float),
        np.array([12.0, 0.0], dtype=float),
        base_distance_threshold_m=2.0,
        speed_mps=15.0,
        control_dt_s=0.08,
        unity_dt_s=0.15,
        skipped_unity_frames=8,
        fallback_active=True,
        payload_selected_fresh=False,
        payload_selected_age_ms=300.0,
        expected_motion_ratio_threshold=1.8,
        expected_motion_margin_m=0.75,
        hard_override_distance_m=8.0,
        hard_override_ratio_threshold=4.0,
        continuity_skip_frames_threshold=3,
        continuity_warn_age_ms=125.0,
    )

    assert result["teleport_detected"] is True
    assert result["guard_suppressed"] is False
