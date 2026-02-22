from pathlib import Path
import importlib.util

import h5py
import numpy as np

from tools.drive_summary_core import analyze_recording_summary as analyze_core
from tools.debug_visualizer.backend.summary_analyzer import (
    analyze_recording_summary as analyze_philviz_adapter,
)


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


project_root = Path(__file__).resolve().parents[1]
analyzer_module = _load_module(
    "analyze_drive_overall",
    project_root / "tools/analyze/analyze_drive_overall.py",
)
analyze_cli_adapter = analyzer_module.analyze_recording_summary


def _write_recording(path: Path) -> None:
    n = 16
    t = np.linspace(0.0, 1.5, n)
    speed = np.linspace(0.0, 4.0, n, dtype=np.float32)
    ref_speed = np.full(n, 3.0, dtype=np.float32)

    with h5py.File(path, "w") as f:
        f.create_dataset("vehicle/timestamps", data=t)
        f.create_dataset("vehicle/speed", data=speed)
        f.create_dataset("control/steering", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/lateral_error", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/heading_error", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/total_error", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/pid_integral", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("control/emergency_stop", data=np.zeros(n, dtype=np.int8))
        f.create_dataset("trajectory/reference_point_velocity", data=ref_speed)
        f.create_dataset("ground_truth/left_lane_line_x", data=np.full(n, -3.5, dtype=np.float32))
        f.create_dataset("ground_truth/right_lane_line_x", data=np.full(n, 3.5, dtype=np.float32))


def test_summary_parity_between_core_and_adapters(tmp_path: Path) -> None:
    recording = tmp_path / "parity_recording.h5"
    _write_recording(recording)

    core_summary = analyze_core(recording, analyze_to_failure=False)
    cli_summary = analyze_cli_adapter(recording, analyze_to_failure=False)
    philviz_summary = analyze_philviz_adapter(recording, analyze_to_failure=False)

    assert core_summary == cli_summary
    assert core_summary == philviz_summary
