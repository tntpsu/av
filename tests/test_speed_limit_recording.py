import importlib.util
from pathlib import Path

import numpy as np


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_record_frame_accepts_speed_limit():
    project_root = Path(__file__).resolve().parents[1]
    av_module = _load_module("av_stack", project_root / "av_stack.py")

    class DummyRecorder:
        def __init__(self):
            self.frames = []

        def record_frame(self, frame):
            self.frames.append(frame)

    av_stack = av_module.AVStack.__new__(av_module.AVStack)
    av_stack.frame_count = 0
    av_stack.recorder = DummyRecorder()
    av_stack.bridge = None

    image = np.zeros((2, 2, 3), dtype=np.uint8)
    control_command = {"steering": 0.0, "throttle": 0.0, "brake": 0.0}

    av_stack._record_frame(
        image,
        0.0,
        {},
        None,
        None,
        control_command,
        speed_limit=12.3,
    )

    assert len(av_stack.recorder.frames) == 1
    assert av_stack.recorder.frames[0].vehicle_state.speed_limit == 12.3
