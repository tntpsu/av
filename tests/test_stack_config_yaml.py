"""Sanity checks for stack.* keys in default av_stack_config.yaml."""

from __future__ import annotations

from pathlib import Path

from av_stack.config import load_config


def test_default_config_has_stack_and_perception_device_keys() -> None:
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "av_stack_config.yaml"
    cfg = load_config(str(cfg_path))
    stack = cfg.get("stack", {})
    assert "target_loop_hz" in stack
    assert float(stack["target_loop_hz"]) > 0
    assert "use_raw_camera_transport" in stack
    assert "bridge_max_camera_queue" in stack
    assert "camera_prefetch" in stack
    assert "clear_bridge_camera_queue_on_start" in stack
    perc = cfg.get("perception", {})
    assert "prefer_mps" in perc
    assert "use_gpu" in perc
