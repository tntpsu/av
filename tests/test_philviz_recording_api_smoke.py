"""
Smoke tests for PhilViz Flask recording analysis routes (cadence, oscillation, grade-lateral).

Builds a minimal synthetic HDF5 in tmp_path (no large fixture required).
Run: pytest tests/test_philviz_recording_api_smoke.py -v
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

SMOKE_NAME = "philviz_smoke_minimal.h5"


def _write_minimal_recording(path: Path, n: int = 80) -> None:
    """Enough structure for cadence_breakdown, oscillation_attribution, grade_lateral."""
    t = np.linspace(0.0, float(n - 1) / 20.0, n, dtype=np.float64)
    sent = t + 0.0005 * np.arange(n, dtype=np.float64)
    inp = sent - 0.0001
    steer = np.zeros(n, dtype=np.float32)
    lat = np.full(n, 0.05, dtype=np.float32)
    grade = np.zeros(n, dtype=np.float32)
    grade[: n // 3] = -0.05

    with h5py.File(path, "w") as f:
        f.create_dataset("vehicle/timestamps", data=t.astype(np.float64))
        f.create_dataset("vehicle/speed", data=np.full(n, 8.0, dtype=np.float32))
        f.create_dataset("vehicle/road_grade", data=grade)
        f.create_dataset("control/timestamps", data=t.astype(np.float64))
        f.create_dataset("control/e2e_control_sent_mono_s", data=sent.astype(np.float64))
        f.create_dataset("control/e2e_inputs_ready_mono_s", data=inp.astype(np.float64))
        f.create_dataset("control/e2e_latency_ms", data=np.full(n, 5.0, dtype=np.float32))
        f.create_dataset("control/steering", data=steer)
        f.create_dataset("control/lateral_error", data=lat)
        f.create_dataset("control/heading_error", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/total_error", data=lat)
        f.create_dataset("control/total_error_scaled", data=lat)


@pytest.fixture()
def philviz_client(monkeypatch, tmp_path: Path):
    _write_minimal_recording(tmp_path / SMOKE_NAME)
    import tools.debug_visualizer.server as srv

    monkeypatch.setattr(srv, "RECORDINGS_DIR", tmp_path)
    return srv.app.test_client()


def test_cadence_breakdown_json(philviz_client) -> None:
    r = philviz_client.get(f"/api/recording/{SMOKE_NAME}/cadence-breakdown")
    assert r.status_code == 200, r.get_data(as_text=True)
    data = r.get_json()
    assert data is not None
    assert data.get("schema_version") == "cadence_breakdown_v1"
    assert "error" not in data


def test_oscillation_attribution_json(philviz_client) -> None:
    r = philviz_client.get(
        f"/api/recording/{SMOKE_NAME}/oscillation-attribution?pre_failure_only=false"
    )
    assert r.status_code == 200, r.get_data(as_text=True)
    data = r.get_json()
    assert data is not None
    assert data.get("schema_version") == "oscillation_attribution_v1"


def test_grade_lateral_json(philviz_client) -> None:
    r = philviz_client.get(
        f"/api/recording/{SMOKE_NAME}/grade-lateral?pre_failure_only=false"
    )
    assert r.status_code == 200, r.get_data(as_text=True)
    data = r.get_json()
    assert data is not None
    assert data.get("schema_version") == "grade_lateral_v1"
    assert "error" not in data


def test_grade_lateral_flat_focus_json(philviz_client) -> None:
    r = philviz_client.get(f"/api/recording/{SMOKE_NAME}/grade-lateral-flat-focus")
    assert r.status_code == 200, r.get_data(as_text=True)
    data = r.get_json()
    assert data is not None
    assert "error" not in data
    assert "flat_high_lateral_ranges" in data
