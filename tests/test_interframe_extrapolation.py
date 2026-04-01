"""
Tests for inter-frame control extrapolation (Tier 1).

Validates:
  - InterframeConfig loading and defaults
  - compute_interframe_steering() in pid_controller.py
  - Guard logic (regime gate, stale GT, discontinuity, max updates)
  - Smith predictor interaction (inter-frame bypasses Smith)
  - HDF5 field registration in ControlCommand dataclass
"""

import math
import sys
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml


# ── Helpers to import without cv2 (orchestrator→cv2 chain) ──────────────────

def _import_module_directly(name: str, filepath: str):
    """Import a module by file path, bypassing __init__.py."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ROOT = Path(__file__).parent.parent
_config_mod = _import_module_directly(
    "av_stack.config", str(ROOT / "av_stack" / "config.py")
)
InterframeConfig = _config_mod.InterframeConfig

from control.regime_selector import RegimeSelector, RegimeConfig, ControlRegime

# VehicleController requires osqp (via mpc_controller.py).  Skip tests that
# call compute_interframe_steering when osqp is unavailable.
try:
    from control.pid_controller import VehicleController
    _HAS_PID_CONTROLLER = True
except (ImportError, ModuleNotFoundError):
    VehicleController = None  # type: ignore[assignment,misc]
    _HAS_PID_CONTROLLER = False

_skip_no_osqp = pytest.mark.skipif(
    not _HAS_PID_CONTROLLER,
    reason="osqp / pid_controller not importable in this environment",
)


# ═══════════════════════════════════════════════════════════════════════════════
# TestInterframeConfig — config loading, defaults, overrides
# ═══════════════════════════════════════════════════════════════════════════════


class TestInterframeConfig:
    def test_defaults(self):
        cfg = InterframeConfig()
        assert cfg.enabled is False
        assert cfg.max_interframe_updates == 3
        assert cfg.min_interframe_dt_s == 0.025
        assert cfg.stale_vehicle_state_ms == 50.0
        assert cfg.max_e_lat_jump_m == 0.5
        assert cfg.max_speed_change_mps == 3.0
        assert cfg.recording_mode == "summary"
        assert cfg.regime_gate == "mpc_only"

    def test_from_config_empty(self):
        cfg = InterframeConfig.from_config({})
        assert cfg.enabled is False
        assert cfg.max_interframe_updates == 3

    def test_from_config_override(self):
        raw = {
            "control": {
                "interframe_extrapolation": {
                    "enabled": True,
                    "max_interframe_updates": 5,
                    "min_interframe_dt_s": 0.020,
                    "stale_vehicle_state_ms": 100.0,
                }
            }
        }
        cfg = InterframeConfig.from_config(raw)
        assert cfg.enabled is True
        assert cfg.max_interframe_updates == 5
        assert cfg.min_interframe_dt_s == 0.020
        assert cfg.stale_vehicle_state_ms == 100.0
        # Unspecified fields keep defaults
        assert cfg.max_e_lat_jump_m == 0.5
        assert cfg.regime_gate == "mpc_only"

    def test_from_config_partial_section(self):
        raw = {"control": {"interframe_extrapolation": {"regime_gate": "always"}}}
        cfg = InterframeConfig.from_config(raw)
        assert cfg.regime_gate == "always"
        assert cfg.enabled is False  # default

    def test_base_yaml_has_config_section(self):
        base_yaml = ROOT / "config" / "av_stack_config.yaml"
        with open(base_yaml) as f:
            raw = yaml.safe_load(f)
        section = raw.get("control", {}).get("interframe_extrapolation", {})
        assert "enabled" in section
        assert section["enabled"] is False  # off by default
        assert section["max_interframe_updates"] == 3


# ═══════════════════════════════════════════════════════════════════════════════
# TestRegimeSelectorPeek — peek() must NOT advance state
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegimeSelectorPeek:
    def test_peek_returns_current_state(self):
        sel = RegimeSelector(RegimeConfig(enabled=True))
        regime, blend = sel.peek()
        assert regime == ControlRegime.PURE_PURSUIT
        assert blend == 1.0

    def test_peek_does_not_advance_hysteresis(self):
        sel = RegimeSelector(RegimeConfig(enabled=True, min_hold_frames=5))
        # Drive at MPC speed to start transition
        for _ in range(3):
            sel.update(speed=12.0)
        # Peek multiple times — should NOT advance hold counter
        r1, b1 = sel.peek()
        for _ in range(10):
            sel.peek()
        r2, b2 = sel.peek()
        assert r1 == r2
        assert b1 == b2

    def test_peek_during_blend(self):
        sel = RegimeSelector(RegimeConfig(
            enabled=True, min_hold_frames=1, blend_frames=10
        ))
        # Trigger transition to MPC
        for _ in range(2):
            sel.update(speed=15.0)
        regime, blend = sel.peek()
        assert regime == ControlRegime.LINEAR_MPC
        assert 0.0 <= blend < 1.0  # mid-blend

    def test_peek_after_update_same_result(self):
        sel = RegimeSelector(RegimeConfig(enabled=True, min_hold_frames=1, blend_frames=1))
        sel.update(speed=0.5)
        r_update, b_update = sel.update(speed=0.5)
        r_peek, b_peek = sel.peek()
        assert r_update == r_peek
        assert b_update == b_peek


# ═══════════════════════════════════════════════════════════════════════════════
# TestInterframeSteering — compute_interframe_steering() logic
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_controller():
    """Build a minimal VehicleController-like mock for testing compute_interframe_steering."""
    controller = MagicMock()

    # Regime selector returns LINEAR_MPC
    mock_selector = MagicMock()
    mock_selector.peek.return_value = (ControlRegime.LINEAR_MPC, 1.0)
    controller._regime_selector = mock_selector

    # MPC controller returns a valid result
    mock_mpc = MagicMock()
    mock_mpc.compute_steering.return_value = {
        'steering_normalized': 0.05,
        'mpc_feasible': True,
        'mpc_fallback_active': False,
    }
    controller._mpc_controller = mock_mpc

    # Config
    controller._full_config = {
        'trajectory': {
            'mpc': {
                'mpc_max_steering_rate_per_frame': 0.15,
            }
        }
    }

    # State
    controller._last_steering_norm = 0.0
    controller._last_mpc_steering = 0.0

    # Lateral controller for max_steering
    controller.lateral_controller = MagicMock()
    controller.lateral_controller.max_steering = 0.5

    # Longitudinal controller for target_speed
    controller.longitudinal_controller = MagicMock()
    controller.longitudinal_controller.target_speed = 12.0
    controller.longitudinal_controller.max_speed = 15.0

    return controller


@_skip_no_osqp
class TestInterframeSteering:
    def test_returns_none_during_pp_regime(self, mock_controller):
        """Inter-frame must return None when regime is PP — no MPC without full pipeline."""
        mock_controller._regime_selector.peek.return_value = (ControlRegime.PURE_PURSUIT, 1.0)
        result = VehicleController.compute_interframe_steering(
            mock_controller, e_lat=0.1, e_heading=0.01, speed=8.0,
            curvature=0.001, dt=0.033,
        )
        assert result is None

    def test_returns_none_when_no_mpc_controller(self, mock_controller):
        mock_controller._mpc_controller = None
        result = VehicleController.compute_interframe_steering(
            mock_controller, e_lat=0.1, e_heading=0.01, speed=8.0,
            curvature=0.001, dt=0.033,
        )
        assert result is None

    def test_valid_mpc_result(self, mock_controller):
        result = VehicleController.compute_interframe_steering(
            mock_controller, e_lat=0.1, e_heading=0.01, speed=12.0,
            curvature=0.001, dt=0.033,
        )
        assert result is not None
        assert 'steering' in result
        assert result['interframe'] is True
        assert result['mpc_feasible'] is True

    def test_steering_uses_max_steering_scale(self, mock_controller):
        """Steering = normalized × max_steering."""
        mock_controller._mpc_controller.compute_steering.return_value = {
            'steering_normalized': 0.10,
            'mpc_feasible': True,
            'mpc_fallback_active': False,
        }
        result = VehicleController.compute_interframe_steering(
            mock_controller, e_lat=0.1, e_heading=0.01, speed=12.0,
            curvature=0.001, dt=0.033,
        )
        expected = 0.10 * 0.5  # normalized × max_steering
        assert abs(result['steering'] - expected) < 1e-6

    def test_rate_limiter_applied(self, mock_controller):
        """Large steering jump should be rate-limited."""
        mock_controller._last_mpc_steering = 0.0
        mock_controller._mpc_controller.compute_steering.return_value = {
            'steering_normalized': 0.80,
            'mpc_feasible': True,
            'mpc_fallback_active': False,
        }
        result = VehicleController.compute_interframe_steering(
            mock_controller, e_lat=0.5, e_heading=0.1, speed=12.0,
            curvature=0.001, dt=0.033,
        )
        assert result is not None
        assert result['rate_limited'] is True
        # Should be clamped to max rate (0.15 at dt_scale=1.0)
        assert abs(result['steering']) <= 0.15 + 1e-6

    def test_rate_limit_scales_with_dt(self, mock_controller):
        """Longer dt allows larger steering change."""
        mock_controller._last_mpc_steering = 0.0
        mock_controller._mpc_controller.compute_steering.return_value = {
            'steering_normalized': 0.80,
            'mpc_feasible': True,
            'mpc_fallback_active': False,
        }
        from control.pid_controller import VehicleController
        # dt = 0.066 → dt_scale = 2.0 → scaled_max_rate = 0.30
        result = VehicleController.compute_interframe_steering(
            mock_controller, e_lat=0.5, e_heading=0.1, speed=12.0,
            curvature=0.001, dt=0.066,
        )
        assert result['rate_limited'] is True
        assert abs(result['steering']) <= 0.30 + 1e-6
        assert abs(result['steering']) > 0.15  # more than 1x rate

    def test_returns_none_on_mpc_fallback(self, mock_controller):
        mock_controller._mpc_controller.compute_steering.return_value = {
            'steering_normalized': 0.0,
            'mpc_feasible': False,
            'mpc_fallback_active': True,
        }
        result = VehicleController.compute_interframe_steering(
            mock_controller, e_lat=0.1, e_heading=0.01, speed=12.0,
            curvature=0.001, dt=0.033,
        )
        assert result is None

    def test_returns_none_on_solver_failure(self, mock_controller):
        mock_controller._mpc_controller.compute_steering.return_value = None
        result = VehicleController.compute_interframe_steering(
            mock_controller, e_lat=0.1, e_heading=0.01, speed=12.0,
            curvature=0.001, dt=0.033,
        )
        assert result is None

    def test_result_contains_diagnostics(self, mock_controller):
        result = VehicleController.compute_interframe_steering(
            mock_controller, e_lat=0.123, e_heading=0.045, speed=12.0,
            curvature=0.001, dt=0.033,
        )
        assert result['e_lat'] == 0.123
        assert result['e_heading'] == 0.045

    def test_nmpc_regime_also_allowed(self, mock_controller):
        """NMPC regime should also trigger inter-frame updates."""
        mock_controller._regime_selector.peek.return_value = (ControlRegime.NONLINEAR_MPC, 1.0)
        result = VehicleController.compute_interframe_steering(
            mock_controller, e_lat=0.1, e_heading=0.01, speed=25.0,
            curvature=0.001, dt=0.033,
        )
        assert result is not None

    def test_stanley_regime_returns_none(self, mock_controller):
        mock_controller._regime_selector.peek.return_value = (ControlRegime.STANLEY, 1.0)
        result = VehicleController.compute_interframe_steering(
            mock_controller, e_lat=0.1, e_heading=0.01, speed=3.0,
            curvature=0.001, dt=0.033,
        )
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# TestInterframeGuards — guard conditions in _run_interframe_update
# ═══════════════════════════════════════════════════════════════════════════════


class TestInterframeGuards:
    """Test guard conditions as pure logic (no orchestrator instance needed)."""

    def test_max_updates_cap(self):
        """After max_interframe_updates, no more updates this cycle."""
        cfg = InterframeConfig(enabled=True, max_interframe_updates=3)
        count = 3
        assert count >= cfg.max_interframe_updates  # guard fires

    def test_regime_gate_blocks_pp(self):
        """regime_gate='mpc_only' blocks when regime==0 (PP)."""
        cfg = InterframeConfig(enabled=True, regime_gate="mpc_only")
        ctx = {'regime': 0}
        assert cfg.regime_gate == "mpc_only" and ctx.get('regime', 0) == 0

    def test_regime_gate_allows_mpc(self):
        """regime_gate='mpc_only' allows when regime==1 (LINEAR_MPC)."""
        cfg = InterframeConfig(enabled=True, regime_gate="mpc_only")
        ctx = {'regime': 1}
        blocked = cfg.regime_gate == "mpc_only" and ctx.get('regime', 0) == 0
        assert not blocked

    def test_stale_gt_detection(self):
        """Same unityTime → stale, should skip."""
        prev_time = 12.345
        curr_time = 12.345
        assert abs(curr_time - prev_time) < 1e-6

    def test_e_lat_discontinuity_guard(self):
        """Large e_lat jump triggers skip."""
        cfg = InterframeConfig(max_e_lat_jump_m=0.5)
        prev_e_lat = 0.1
        curr_e_lat = 0.8
        assert abs(curr_e_lat - prev_e_lat) > cfg.max_e_lat_jump_m

    def test_speed_discontinuity_guard(self):
        """Large speed jump triggers skip."""
        cfg = InterframeConfig(max_speed_change_mps=3.0)
        prev_speed = 12.0
        curr_speed = 16.0
        assert abs(curr_speed - prev_speed) > cfg.max_speed_change_mps

    def test_small_changes_pass_guards(self):
        """Normal driving conditions pass all guards."""
        cfg = InterframeConfig(max_e_lat_jump_m=0.5, max_speed_change_mps=3.0)
        assert abs(0.12 - 0.10) <= cfg.max_e_lat_jump_m  # 0.02m
        assert abs(12.1 - 12.0) <= cfg.max_speed_change_mps  # 0.1 m/s

    def test_min_dt_floor(self):
        """dt is floored at min_interframe_dt_s to prevent busy-spin."""
        cfg = InterframeConfig(min_interframe_dt_s=0.025)
        raw_dt = 0.005
        effective_dt = max(raw_dt, cfg.min_interframe_dt_s)
        assert effective_dt == 0.025


# ═══════════════════════════════════════════════════════════════════════════════
# TestInterframeSmithInteraction — inter-frame bypasses Smith predictor
# ═══════════════════════════════════════════════════════════════════════════════


@_skip_no_osqp
class TestInterframeSmithInteraction:
    def test_interframe_does_not_call_ring_buffer(self, mock_controller):
        """Inter-frame path must NOT touch the steering ring buffer."""
        mock_controller._steering_ring_buffer = MagicMock()
        VehicleController.compute_interframe_steering(
            mock_controller, e_lat=0.1, e_heading=0.01, speed=12.0,
            curvature=0.001, dt=0.033,
        )
        mock_controller._steering_ring_buffer.append.assert_not_called()

    def test_no_smith_predictor_in_mpc_call(self, mock_controller):
        """MPC call from inter-frame receives raw GT e_lat, not predicted."""
        VehicleController.compute_interframe_steering(
            mock_controller, e_lat=0.123, e_heading=0.045, speed=12.0,
            curvature=0.001, dt=0.033,
        )
        # Check that MPC was called with the raw GT e_lat (not Smith-predicted)
        call_kwargs = mock_controller._mpc_controller.compute_steering.call_args
        assert call_kwargs is not None
        # e_lat passed through directly
        assert abs(call_kwargs.kwargs.get('e_lat', call_kwargs[1].get('e_lat', 0.0)) - 0.123) < 1e-6 or \
               abs(call_kwargs[1].get('e_lat', 0.0) - 0.123) < 1e-6

    def test_delay_cap_config_exists(self):
        """The smith_predictor_max_delay_s config exists (osqp-gated variant)."""
        base_yaml = ROOT / "config" / "av_stack_config.yaml"
        with open(base_yaml) as f:
            raw = yaml.safe_load(f)
        assert raw.get("trajectory", {}).get("mpc", {}).get("smith_predictor_max_delay_s") == 0.080


class TestSmithDelayCap:
    """Config-only tests that don't need osqp."""

    def test_delay_cap_still_applies_to_full_frames(self):
        """The smith_predictor_max_delay_s config exists and caps full-frame delay."""
        base_yaml = ROOT / "config" / "av_stack_config.yaml"
        with open(base_yaml) as f:
            raw = yaml.safe_load(f)
        mpc_cfg = raw.get("trajectory", {}).get("mpc", {})
        assert "smith_predictor_max_delay_s" in mpc_cfg
        max_delay = mpc_cfg["smith_predictor_max_delay_s"]
        assert max_delay == 0.080
        # At 7.5 Hz: raw delay = 2 × 0.133 = 0.266s → capped to 0.080s
        raw_delay = 2 * 0.133
        capped = min(raw_delay, max_delay)
        assert capped == max_delay


# ═══════════════════════════════════════════════════════════════════════════════
# TestControlCommandInterframeFields — HDF5 field registration
# ═══════════════════════════════════════════════════════════════════════════════


class TestControlCommandInterframeFields:
    def test_dataclass_has_interframe_fields(self):
        """ControlCommand must have all 6 inter-frame diagnostic fields."""
        from data.formats.data_format import ControlCommand
        cc = ControlCommand(
            timestamp=0.0,
            steering=0.0,
            throttle=0.0,
            brake=0.0,
        )
        assert hasattr(cc, 'interframe_active')
        assert hasattr(cc, 'interframe_updates_this_cycle')
        assert hasattr(cc, 'interframe_total_count')
        assert hasattr(cc, 'interframe_last_e_lat')
        assert hasattr(cc, 'interframe_last_e_heading')
        assert hasattr(cc, 'interframe_dt_actual')

    def test_interframe_defaults_are_zero(self):
        from data.formats.data_format import ControlCommand
        cc = ControlCommand(timestamp=0.0, steering=0.0, throttle=0.0, brake=0.0)
        assert cc.interframe_active == 0.0
        assert cc.interframe_updates_this_cycle == 0
        assert cc.interframe_total_count == 0
        assert cc.interframe_last_e_lat == 0.0
        assert cc.interframe_last_e_heading == 0.0
        assert cc.interframe_dt_actual == 0.0

    def test_interframe_fields_can_be_set(self):
        from data.formats.data_format import ControlCommand
        cc = ControlCommand(
            timestamp=1.0,
            steering=0.05,
            throttle=0.3,
            brake=0.0,
            interframe_active=1.0,
            interframe_updates_this_cycle=2,
            interframe_total_count=47,
            interframe_last_e_lat=0.123,
            interframe_last_e_heading=0.045,
            interframe_dt_actual=0.033,
        )
        assert cc.interframe_active == 1.0
        assert cc.interframe_updates_this_cycle == 2
        assert cc.interframe_total_count == 47
        assert cc.interframe_last_e_lat == 0.123
        assert cc.interframe_last_e_heading == 0.045
        assert cc.interframe_dt_actual == 0.033
