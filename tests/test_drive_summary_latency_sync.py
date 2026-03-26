from pathlib import Path

import h5py
import numpy as np

from tools.drive_summary_core import analyze_recording_summary


def _write_latency_recording(
    path: Path,
    *,
    include_e2e: bool,
    traj_offset_s: float,
    control_offset_s: float,
    irregular_indices: tuple[int, ...] = (),
    irregular_gap_s: float = 0.22,
    include_transport: bool = False,
) -> None:
    n = 121
    t = np.linspace(0.0, 4.0, n, dtype=np.float64)
    for idx in irregular_indices:
        if 0 < idx < n:
            t[idx:] += float(irregular_gap_s)
    zeros = np.zeros(n, dtype=np.float32)

    with h5py.File(path, "w") as f:
        f.create_dataset("vehicle/timestamps", data=t)
        f.create_dataset("vehicle/speed", data=np.full(n, 5.0, dtype=np.float32))
        f.create_dataset("camera/timestamps", data=t)
        f.create_dataset("trajectory/timestamps", data=t + traj_offset_s)
        f.create_dataset("control/timestamps", data=t + control_offset_s)
        f.create_dataset("control/steering", data=zeros)
        f.create_dataset("control/lateral_error", data=zeros)
        f.create_dataset("control/heading_error", data=zeros)
        f.create_dataset("control/total_error", data=zeros)
        f.create_dataset("control/pid_integral", data=zeros)
        f.create_dataset("control/emergency_stop", data=np.zeros(n, dtype=np.int8))
        f.create_dataset("ground_truth/left_lane_line_x", data=np.full(n, -3.5, dtype=np.float32))
        f.create_dataset("ground_truth/right_lane_line_x", data=np.full(n, 3.5, dtype=np.float32))

        if include_e2e:
            f.create_dataset("control/e2e_latency_ms", data=np.linspace(24.0, 42.0, n, dtype=np.float32))
            mode_dtype = h5py.string_dtype(encoding="utf-8", length=64)
            f.create_dataset(
                "control/e2e_latency_mode",
                data=np.array(["input_ready_to_command_sent"] * n, dtype=mode_dtype),
            )
        if include_transport:
            dtype = h5py.string_dtype(encoding="utf-8", length=64)
            f.create_dataset(
                "vehicle/sync_packet_mode",
                data=np.array(["packet_shadow"] * n, dtype=dtype),
            )
            f.create_dataset("vehicle/sync_packet_schema_version", data=np.ones(n, dtype=np.int16))
            f.create_dataset("vehicle/sync_packet_complete", data=np.ones(n, dtype=np.int8))
            f.create_dataset("vehicle/sync_packet_fallback_active", data=np.zeros(n, dtype=np.int8))
            f.create_dataset(
                "vehicle/sync_packet_fallback_reason_code",
                data=np.array(["none"] * n, dtype=dtype),
            )
            f.create_dataset(
                "vehicle/sync_packet_consume_policy",
                data=np.array(["freshest_within_budget"] * n, dtype=dtype),
            )
            f.create_dataset("vehicle/sync_packet_queue_depth", data=np.full(n, 2, dtype=np.int16))
            f.create_dataset("vehicle/sync_packet_payload_queue_depth", data=np.full(n, 1, dtype=np.int16))
            f.create_dataset("vehicle/sync_packet_skipped_unity_frames", data=np.zeros(n, dtype=np.int16))
            f.create_dataset("vehicle/sync_packet_age_ms", data=np.full(n, 8.0, dtype=np.float32))
            f.create_dataset("vehicle/sync_packet_payload_oldest_age_ms", data=np.full(n, 12.0, dtype=np.float32))
            f.create_dataset("vehicle/sync_packet_payload_bytes", data=np.full(n, 921600, dtype=np.int32))
            f.create_dataset("vehicle/sync_packet_payload_fallback_reason_code", data=np.array(["none"] * n, dtype=dtype))
            f.create_dataset("vehicle/sync_packet_payload_selected_age_ms", data=np.full(n, 32.0, dtype=np.float32))
            f.create_dataset("vehicle/sync_packet_payload_selected_fresh", data=np.ones(n, dtype=np.int8))
            f.create_dataset("vehicle/sync_packet_payload_warn_age_exceeded", data=np.zeros(n, dtype=np.int8))
            f.create_dataset("vehicle/sync_packet_payload_stale_drop_count", data=np.full(n, 3, dtype=np.int16))
            f.create_dataset("vehicle/sync_packet_payload_drained_count", data=np.full(n, 3, dtype=np.int16))
            f.create_dataset("vehicle/sync_packet_payload_max_drained_age_ms", data=np.full(n, 64.0, dtype=np.float32))
            f.create_dataset("vehicle/sync_packet_payload_selection_source", data=np.array(["server_selector"] * n, dtype=dtype))
            f.create_dataset("vehicle/sync_packet_payload_selection_fallback_active", data=np.zeros(n, dtype=np.int8))
            f.create_dataset("vehicle/sync_packet_payload_selection_fallback_reason_code", data=np.array(["none"] * n, dtype=dtype))
            f.create_dataset("vehicle/sync_packet_payload_server_queue_depth_after_select", data=np.zeros(n, dtype=np.int16))
            f.create_dataset("vehicle/sync_packet_payload_server_oldest_age_ms_after_select", data=np.zeros(n, dtype=np.float32))
            f.create_dataset("vehicle/sync_front_age_ms", data=np.full(n, 6.0, dtype=np.float32))
            f.create_dataset("vehicle/sync_vehicle_age_ms", data=np.full(n, 5.0, dtype=np.float32))
            f.create_dataset("vehicle/sync_front_vehicle_frame_delta", data=np.zeros(n, dtype=np.float32))
            f.create_dataset("vehicle/sync_front_vehicle_time_delta_ms", data=np.full(n, 2.0, dtype=np.float32))
            f.create_dataset("vehicle/sync_packet_missing_front", data=np.zeros(n, dtype=np.int8))
            f.create_dataset("vehicle/sync_packet_missing_vehicle", data=np.zeros(n, dtype=np.int8))
            f.create_dataset("control/target_speed_final", data=np.full(n, 15.0, dtype=np.float32))
            f.create_dataset("control/target_speed_post_limits", data=np.full(n, 15.0, dtype=np.float32))
            f.create_dataset("control/target_speed_raw", data=np.full(n, 15.0, dtype=np.float32))
            f.create_dataset("trajectory/reference_point_velocity", data=np.full(n, 15.0, dtype=np.float32))
            f.create_dataset("control/reference_velocity_effective", data=np.full(n, 15.0, dtype=np.float32))
            f.create_dataset("control/post_jump_cooldown_active", data=np.zeros(n, dtype=np.int8))
            f.create_dataset(
                "control/post_jump_reason_code",
                data=np.array(["none"] * n, dtype=dtype),
            )
            f.create_dataset("control/teleport_detected", data=np.zeros(n, dtype=np.int8))
            f.create_dataset("control/teleport_motion_ratio", data=np.ones(n, dtype=np.float32))
            f.create_dataset("control/teleport_guard_suppressed", data=np.zeros(n, dtype=np.int8))
            f.create_dataset("control/teleport_continuity_suspect", data=np.zeros(n, dtype=np.int8))
            f.create_dataset(
                "control/teleport_guard_reason_code",
                data=np.array(["none"] * n, dtype=dtype),
            )
            f.create_dataset("control/teleport_dynamic_threshold_m", data=np.full(n, 2.0, dtype=np.float32))


def test_latency_sync_available_and_passing(tmp_path: Path) -> None:
    recording = tmp_path / "latency_sync_pass.h5"
    _write_latency_recording(
        recording,
        include_e2e=True,
        traj_offset_s=0.005,
        control_offset_s=0.004,
    )

    summary = analyze_recording_summary(recording)
    latency_sync = summary["latency_sync"]
    assert latency_sync["e2e"]["availability"] == "available"
    assert latency_sync["e2e"]["pass"] is True
    assert latency_sync["sync_alignment"]["availability"] == "available"
    assert latency_sync["sync_alignment"]["pass"] is True
    assert latency_sync["cadence"]["availability"] == "available"
    assert latency_sync["cadence"]["pass"] is True
    assert latency_sync["cadence"]["tuning_valid"] is True
    assert latency_sync["cadence"]["status"] == "good"
    assert latency_sync["overall"]["status"] == "good"


def test_latency_sync_legacy_e2e_unavailable(tmp_path: Path) -> None:
    recording = tmp_path / "latency_sync_legacy.h5"
    _write_latency_recording(
        recording,
        include_e2e=False,
        traj_offset_s=0.006,
        control_offset_s=0.006,
    )

    summary = analyze_recording_summary(recording)
    latency_sync = summary["latency_sync"]
    assert latency_sync["e2e"]["availability"] == "unavailable"
    assert latency_sync["e2e"]["pass"] is None
    assert latency_sync["sync_alignment"]["availability"] == "available"
    assert latency_sync["cadence"]["availability"] == "available"


def test_latency_sync_detects_alignment_failure(tmp_path: Path) -> None:
    recording = tmp_path / "latency_sync_fail.h5"
    _write_latency_recording(
        recording,
        include_e2e=True,
        traj_offset_s=0.080,
        control_offset_s=0.090,
    )

    summary = analyze_recording_summary(recording)
    latency_sync = summary["latency_sync"]
    assert latency_sync["sync_alignment"]["pass"] is False
    assert latency_sync["overall"]["status"] == "poor"
    assert latency_sync["overall"]["issues"]


def test_latency_sync_detects_cadence_failure(tmp_path: Path) -> None:
    recording = tmp_path / "latency_sync_cadence_fail.h5"
    _write_latency_recording(
        recording,
        include_e2e=True,
        traj_offset_s=0.004,
        control_offset_s=0.004,
        irregular_indices=(6, 12, 18, 24, 30),
        irregular_gap_s=0.22,
    )

    summary = analyze_recording_summary(recording)
    latency_sync = summary["latency_sync"]
    cadence = latency_sync["cadence"]
    assert cadence["availability"] == "available"
    assert cadence["pass"] is False
    assert cadence["tuning_valid"] is False
    assert cadence["status"] == "poor"
    assert "irregular_rate_max" in cadence["limits"]
    assert "severe_irregular_rate_max" in cadence["limits"]
    assert latency_sync["overall"]["status"] == "poor"


def test_transport_contract_summary_available(tmp_path: Path) -> None:
    recording = tmp_path / "transport_contract.h5"
    _write_latency_recording(
        recording,
        include_e2e=True,
        traj_offset_s=0.004,
        control_offset_s=0.004,
        include_transport=True,
    )

    summary = analyze_recording_summary(recording)
    transport = summary["transport_contract"]
    speed_intent = summary["speed_intent"]
    run_intent = summary["run_intent"]

    assert transport["availability"] == "available"
    assert transport["packet_mode"] == "packet_shadow"
    assert transport["consume_policy"] == "freshest_within_budget"
    assert transport["packet_completeness_rate"] == 100.0
    assert transport["fallback_active_rate"] == 0.0
    assert transport["packet_queue_depth"]["p95"] == 2.0
    assert transport["payload_queue_depth"]["p95"] == 1.0
    assert transport["payload_selected_age_ms"]["p50"] == 32.0
    assert transport["payload_selected_fresh_rate"] == 100.0
    assert transport["payload_selection_source_mode"] == "server_selector"
    assert transport["payload_server_queue_depth_after_select"]["p95"] == 0.0
    assert transport["teleport_guard_suppressed_rate"] == 0.0
    assert transport["teleport_guard_reason_mode"] == "none"
    assert speed_intent["availability"] == "available"
    assert speed_intent["effective_reference_speed_mps"]["p50"] == 15.0
    assert run_intent["availability"] == "available"
    assert run_intent["mode"] == "free_flow"
