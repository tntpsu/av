"""Steering excitation for plant identification (tools/sysid_injection.py)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from tools.sysid_injection import InjectionConfig  # noqa: E402


def test_disabled_by_default_and_in_base_config():
    import yaml
    assert InjectionConfig().enabled is False
    cfg = yaml.safe_load((REPO_ROOT / "config" / "av_stack_config.yaml").read_text())
    assert cfg["experiment"]["steering_injection"]["enabled"] is False
    assert InjectionConfig.from_config(cfg).value(30.0, 10.0) == 0.0


def test_window_amplitude_and_determinism():
    c = InjectionConfig(enabled=True, amplitude=0.1, start_s=12.0, duration_s=50.0)
    t = np.arange(0, 80, 0.05)
    r = c.series(t, np.full_like(t, 10.0))
    assert np.all(r[t < 12.0] == 0.0) and np.all(r[t > 62.0] == 0.0)
    assert 0.09 < np.max(np.abs(r)) <= 0.1 + 1e-9
    assert np.allclose(r, c.series(t, np.full_like(t, 10.0)))
    assert c.value(30.0, 10.0) == pytest.approx(r[int(30.0 / 0.05)], abs=1e-9)
    assert c.value(30.0, 1.0) == 0.0                       # speed gate


def test_tones_are_present_and_only_tones():
    c = InjectionConfig(enabled=True, amplitude=0.1, start_s=0.0, duration_s=200.0)
    t = np.arange(0, 200, 0.05); r = c.series(t)
    f = np.fft.rfftfreq(len(t), 0.05); P = np.abs(np.fft.rfft(r * np.hanning(len(t))))
    tone_power = [P[np.argmin(np.abs(f - fk))] for fk in c.tones_hz()]
    assert min(tone_power) > 20 * np.median(P)
    assert P[np.argmin(np.abs(f - 2.5))] < 0.05 * max(tone_power)     # no energy off the tone grid


def test_overlays_set_speed_and_enable():
    import yaml
    for v in (8, 10, 12, 14):
        o = yaml.safe_load((REPO_ROOT / "config" / f"exp_sysid_v{v:02d}.yaml").read_text())
        assert o["experiment"]["steering_injection"]["enabled"] is True
        assert o["trajectory"]["target_speed"] == float(v)
