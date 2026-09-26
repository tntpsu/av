"""Recorder: runtime config snapshot + silent-dropout / ramp HDF5 wiring."""

import json
from pathlib import Path

import h5py
import pytest
import yaml

from data.formats.data_format import ControlCommand, RecordingFrame
from data.recorder import DataRecorder


def test_runtime_config_snapshot_written_once(tmp_path: Path) -> None:
    base = Path(__file__).resolve().parents[1] / "config" / "av_stack_config.yaml"
    with open(base) as fh:
        cfg = yaml.safe_load(fh) or {}
    rec = DataRecorder(str(tmp_path), recording_name="cfg_snap_test")
    rec.write_runtime_config_snapshot(cfg, str(base))
    rec.write_runtime_config_snapshot(cfg, str(base))  # idempotent

    with h5py.File(rec.output_file, "r") as f:
        assert "meta" in f
        raw = f["meta/runtime_config_json"][0]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        snap = json.loads(raw)
        assert "resolved_mpc_params" in snap
        assert snap["resolved_mpc_params"]["elat_ramp_rate_m_per_frame"] >= 0.0
        assert snap["config_overlay_path"] is not None
        # 2026-09-26: longitudinal / ACC / profiler / governor scalars are in the
        # snapshot so the sweep playbook's runtime probe can see them.
        lon = snap["control"]["longitudinal"]
        assert "speed_drag_gain" in lon and "max_accel" in lon
        assert "radar_range_offset_m" in snap["acc"] and "cutout_requires_no_lead" in snap["acc"]
        assert "a_lat_tracking_budget_g" in snap["trajectory"]["velocity_profiler"]
        assert "curve_cap_peak_lat_accel_g" in snap["trajectory"]["speed_governor"]
        # scalar filter: no lists/dicts leak into these sections
        for sec in (lon, snap["acc"], snap["trajectory"]["velocity_profiler"], snap["trajectory"]["speed_governor"]):
            assert all(isinstance(v, (bool, int, float, str, type(None))) for v in sec.values())


def test_diag_silent_and_ramp_datasets_flush(tmp_path: Path) -> None:
    rec = DataRecorder(str(tmp_path), recording_name="diag_flush_test")
    ts = 0.0
    cc = ControlCommand(
        timestamp=ts,
        steering=0.0,
        throttle=0.0,
        brake=0.0,
        diag_silent_elat_dropout_active=True,
        mpc_elat_ramp_active=True,
    )
    frame = RecordingFrame(
        timestamp=ts,
        frame_id=0,
        control_command=cc,
    )
    rec.record_frame(frame)
    rec.close()

    with h5py.File(rec.output_file, "r") as f:
        assert f["control/diag_silent_elat_dropout_active"].shape[0] == 1
        assert f["control/mpc_elat_ramp_active"].shape[0] == 1
        assert int(f["control/diag_silent_elat_dropout_active"][0]) == 1
        assert int(f["control/mpc_elat_ramp_active"][0]) == 1
