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
        },
    )
    assert result["checks"]["run_validity_pass"] is False
    assert result["pass_fail"] is False


def test_matrix_hash_is_deterministic() -> None:
    payload = {"mode": "quick", "recording_ids": ["a.h5", "b.h5"], "runs": 2}
    h1 = gate_triage.compute_matrix_hash(payload)
    h2 = gate_triage.compute_matrix_hash(payload)
    assert h1 == h2


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

    argv = [
        "run_gate_and_triage.py",
        "--label",
        "unit",
        "--recordings",
        str(rec),
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
