from __future__ import annotations

import json
import sys
import importlib.util
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "analyze" / "run_gate_and_triage.py"
spec = importlib.util.spec_from_file_location("run_gate_and_triage", MODULE_PATH)
assert spec and spec.loader
gate_triage = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate_triage
spec.loader.exec_module(gate_triage)


def _write_minimal_recording(path: Path, emergency_stop: bool = False, out_of_lane: bool = False) -> None:
    n = 20
    t = np.linspace(0.0, 1.9, n)
    left = np.full(n, -3.5, dtype=np.float32)
    right = np.full(n, 3.5, dtype=np.float32)
    if out_of_lane:
        right[:] = -0.2

    e_stop = np.zeros(n, dtype=np.int8)
    if emergency_stop:
        e_stop[5:] = 1

    with h5py.File(path, "w") as f:
        f.create_dataset("control/timestamps", data=t)
        f.create_dataset("vehicle/timestamps", data=t)
        f.create_dataset("vehicle/unity_time", data=t)
        f.create_dataset("vehicle/speed", data=np.full(n, 4.0, dtype=np.float32))
        f.create_dataset("control/steering", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/lateral_error", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/heading_error", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/total_error", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/pid_integral", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/emergency_stop", data=e_stop)
        f.create_dataset("ground_truth/left_lane_line_x", data=left)
        f.create_dataset("ground_truth/right_lane_line_x", data=right)
        metadata = {
            "recording_provenance": {
                "track_id": "s_loop",
                "start_t": 0.0,
                "git_sha_short": "unit",
                "policy_profile": "seg_default",
            }
        }
        f.attrs["metadata"] = json.dumps(metadata)


def test_extract_run_metrics_flags_low_fps_as_invalid(tmp_path: Path) -> None:
    rec = tmp_path / "low_fps.h5"
    n = 20
    t = np.linspace(0.0, 10.0, n)  # ~2 FPS
    with h5py.File(rec, "w") as f:
        f.create_dataset("control/timestamps", data=t)
        f.create_dataset("vehicle/timestamps", data=t)
        f.create_dataset("vehicle/unity_time", data=t)
        f.create_dataset("vehicle/speed", data=np.full(n, 4.0, dtype=np.float32))
        f.create_dataset("control/steering", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/lateral_error", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/heading_error", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/total_error", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/pid_integral", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/emergency_stop", data=np.zeros(n, dtype=np.int8))
        f.create_dataset("ground_truth/left_lane_line_x", data=np.full(n, -3.5, dtype=np.float32))
        f.create_dataset("ground_truth/right_lane_line_x", data=np.full(n, 3.5, dtype=np.float32))

    _, metrics = gate_triage._extract_run_metrics(
        rec,
        analyze_to_failure=True,
        min_control_fps=15.0,
        max_unity_time_gap_s=0.25,
    )
    assert metrics["run_validity_pass"] is False
    assert metrics["control_fps"] < 15.0
    assert any("low_control_fps" in reason for reason in metrics["run_validity_reasons"])


def test_evaluate_gate_fails_when_any_run_invalid_sampling() -> None:
    row = gate_triage.RunArtifacts(
        recording_path=Path("/tmp/fake.h5"),
        track_id="s_loop",
        recording_id="fake.h5",
        summary={},
        metrics={
            "emergency_stop_count": 0,
            "out_of_lane_events_full_run": 0,
            "oscillation_runaway": False,
            "run_validity_pass": False,
        },
        stage6={},
        trigger_reasons=[],
    )
    baseline = gate_triage.RunArtifacts(
        recording_path=Path("/tmp/base.h5"),
        track_id="s_loop",
        recording_id="base.h5",
        summary={},
        metrics={
            "lateral_p95": 0.5,
            "first_300_lateral_p99": 0.5,
            "last_300_lateral_p99": 0.5,
            "steering_jerk_max": 1.0,
            "run_validity_pass": True,
        },
        stage6={},
        trigger_reasons=[],
    )
    result = gate_triage._evaluate_gate(
        rows=[row],
        baseline_rows=[baseline],
        required_track_ids=["s_loop"],
        expected_runs_per_track=1,
        regression_budget={
            "highway_lateral_p95_max_delta_m": 0.02,
            "sloop_first_300_p99_min_improvement_m": 0.03,
            "sloop_lateral_p95_max_delta_m": 0.0,
            "cap_tracking_frames_above_cap_1p0mps_rate_max_pct": 2.0,
            "cap_tracking_error_p95_max_mps": 0.5,
            "cap_tracking_recovery_frames_p95_max": 30.0,
            "cap_tracking_hard_ceiling_applied_rate_max_pct": 0.5,
        },
        diagnostics_preflight={"available": True},
        mode="quick",
    )
    assert result["checks"]["run_validity_pass"] is False
    assert result["pass_fail"] is False


def test_matrix_hash_is_deterministic() -> None:
    payload = {"mode": "quick", "recording_ids": ["a.h5", "b.h5"], "runs": 2}
    h1 = gate_triage.compute_matrix_hash(payload)
    h2 = gate_triage.compute_matrix_hash(payload)
    assert h1 == h2


def test_matrix_hash_changes_when_av_start_t_changes() -> None:
    p0 = {"mode": "quick", "recording_ids": ["a.h5"], "av_start_t": 0.0}
    p1 = {"mode": "quick", "recording_ids": ["a.h5"], "av_start_t": 1.0}
    assert gate_triage.compute_matrix_hash(p0) != gate_triage.compute_matrix_hash(p1)


def test_validate_generated_recording_provenance_rejects_mismatch(tmp_path: Path) -> None:
    rec = tmp_path / "bad_provenance.h5"
    _write_minimal_recording(rec)
    with h5py.File(rec, "a") as f:
        meta = json.loads(f.attrs["metadata"])
        meta["recording_provenance"]["track_id"] = "wrong_track"
        f.attrs["metadata"] = json.dumps(meta)
    try:
        gate_triage._validate_generated_recording_provenance(
            rec,
            expected_track_id="s_loop",
            expected_start_t=0.0,
        )
    except RuntimeError as exc:
        assert "track mismatch" in str(exc)
    else:
        raise AssertionError("expected provenance validation failure")


def test_execute_gate_runs_sets_av_start_t_env(monkeypatch, tmp_path: Path) -> None:
    track = tmp_path / "s_loop.yml"
    track.write_text("name: s_loop\n")
    rec = tmp_path / "recording.h5"
    _write_minimal_recording(rec)
    calls = {"subprocess": None, "validated": False}
    latest_seq = [None, rec]

    def fake_latest():
        if latest_seq:
            return latest_seq.pop(0)
        return rec

    def fake_run(cmd, cwd, env, capture_output, text, check):
        calls["subprocess"] = {"cmd": cmd, "cwd": cwd, "env": env}
        class R:
            returncode = 0
            stderr = ""
        return R()

    def fake_validate(recording_path, *, expected_track_id, expected_start_t):
        calls["validated"] = True
        assert recording_path == rec
        assert expected_track_id == "s_loop"
        assert abs(expected_start_t - 1.25) < 1e-9

    monkeypatch.setattr(gate_triage, "_latest_recording", fake_latest)
    monkeypatch.setattr(gate_triage.subprocess, "run", fake_run)
    monkeypatch.setattr(gate_triage, "_validate_generated_recording_provenance", fake_validate)

    out = gate_triage._execute_gate_runs(
        tracks=[track],
        runs_per_track=1,
        duration_s=5,
        av_start_t=1.25,
    )
    assert out == [(rec, "s_loop")]
    assert calls["subprocess"] is not None
    assert calls["subprocess"]["env"]["AV_START_T"] == "1.25"
    assert calls["validated"] is True


def test_validate_contract_required_fields_enforced() -> None:
    payload = {
        "schema_version": "v1",
        "git_sha": "abc",
        "config_hash": "cfg",
        "matrix_hash": "mx",
        "recording_ids": ["r1.h5"],
        "pass_fail": True,
        "regression_budget": {},
    }
    gate_triage.validate_contract_required_fields(payload)


def test_run_gate_and_triage_generates_contract_artifacts(tmp_path: Path, monkeypatch) -> None:
    rec = tmp_path / "failing.h5"
    _write_minimal_recording(rec, emergency_stop=True, out_of_lane=True)

    reports_dir = tmp_path / "reports" / "gates"
    monkeypatch.setattr(gate_triage, "REPORTS_DIR", reports_dir)

    base = tmp_path / "baseline.h5"
    _write_minimal_recording(base, emergency_stop=False, out_of_lane=False)

    argv = [
        "run_gate_and_triage.py",
        "--label",
        "unit",
        "--recordings",
        str(rec),
        "--baseline-recordings",
        str(base),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    rc = gate_triage.main()
    assert rc == 0

    bundles = sorted(reports_dir.glob("*_unit"))
    assert bundles, "expected a bundle directory"
    bundle = bundles[-1]

    gate_report = json.loads((bundle / "gate_report.json").read_text())
    decision = json.loads((bundle / "decision.json").read_text())
    triage_packets = sorted((bundle / "triage_packets").glob("*.json"))

    gate_triage.validate_contract_required_fields(gate_report)
    gate_triage.validate_contract_required_fields(decision)
    assert triage_packets, "expected a triage packet for failing run"
    triage = json.loads(triage_packets[0].read_text())
    gate_triage.validate_contract_required_fields(triage)

    schemas = {
        "gate": json.loads((REPO_ROOT / "docs" / "analysis" / "contracts" / "gate_report.schema.json").read_text()),
        "triage": json.loads((REPO_ROOT / "docs" / "analysis" / "contracts" / "triage_packet.schema.json").read_text()),
        "decision": json.loads((REPO_ROOT / "docs" / "analysis" / "contracts" / "decision.schema.json").read_text()),
    }
    for required_key in schemas["gate"].get("required", []):
        assert required_key in gate_report
    for required_key in schemas["triage"].get("required", []):
        assert required_key in triage
    for required_key in schemas["decision"].get("required", []):
        assert required_key in decision

    assert gate_report["recording_ids"] == [rec.name]
    assert decision["pass_fail"] is False
    baseline_report = json.loads((bundle / "baseline" / "baseline_report.json").read_text())
    gate_triage.validate_contract_required_fields(baseline_report)
    assert baseline_report["recording_ids"] == [base.name]
    assert "provenance" in baseline_report["runs"][0]
