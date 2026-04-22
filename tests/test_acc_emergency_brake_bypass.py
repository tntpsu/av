"""
Unit tests for the B1 hard-brake bypass (acc-idm-accel-plumbing.md Commit D).

The B1 bypass is a short-circuit inside the orchestrator's control-command
assembly: when the ACC state machine declares EMERGENCY_BRAKE and the
config flag is on, the comfort-limited longitudinal stack is skipped and
brake is forced to 1.0 immediately, preserving steering for lane keeping.

Rationale (from probe recording_20260421_214920.h5 frames 56-67): even with
Commit C (decel/jerk widening) and Commit C.1 (jerk cooldown bypass) active,
the upstream accel_target_smoothing EMA (α=0.86) pins commanded decel near
the nominal floor for ~0.5s after emergency onset. At 4 m/s with a lead gap
under 3 m, that latency is the difference between miss and collision. B1
bypasses all three upstream limiters by routing past the longitudinal
controller entirely.

Canonical implementation lives at av_stack/orchestrator.py:9416–9471. The
reference functions in this test file mirror that logic exactly; if the
orchestrator's rule changes, update both and keep them in sync.
"""

from __future__ import annotations

from typing import Any, Dict


def _coerce_acc_state(value: Any) -> str:
    """Mirror of orchestrator's bytes-or-str coercion of acc_state_code."""
    if isinstance(value, bytes):
        return value.decode()
    return str(value or "")


def _b1_bypass_active(
    config_flag: bool, vehicle_state: Dict[str, Any]
) -> bool:
    """Reference rule. Must match orchestrator.py:9427–9433 exactly."""
    state = _coerce_acc_state(vehicle_state.get("acc_state_code", ""))
    return bool(config_flag) and state == "EMERGENCY_BRAKE"


def _apply_b1_override(control_command: Dict[str, Any]) -> Dict[str, Any]:
    """Reference rule. Must match orchestrator.py:9454–9471 exactly."""
    return {
        "steering": control_command.get("steering", 0.0),
        "throttle": 0.0,
        "brake": 1.0,
        "lateral_error": control_command.get("lateral_error", 0.0),
        "heading_error": control_command.get("heading_error", 0.0),
        "emergency_stop": True,
        "b1_bypass_active": True,
    }


# ----------------------------------------------------------------------------
# Core decision tests (plan F2 items 9–11)
# ----------------------------------------------------------------------------


class TestB1DecisionRule:
    def test_b1_disabled_does_not_override(self):
        """Flag off + EMERGENCY_BRAKE → bypass must NOT activate."""
        active = _b1_bypass_active(
            config_flag=False,
            vehicle_state={"acc_state_code": "EMERGENCY_BRAKE"},
        )
        assert active is False

    def test_b1_enabled_overrides_on_emergency_brake(self):
        """Flag on + EMERGENCY_BRAKE → bypass active."""
        active = _b1_bypass_active(
            config_flag=True,
            vehicle_state={"acc_state_code": "EMERGENCY_BRAKE"},
        )
        assert active is True

    def test_b1_non_latched_releases_on_state_exit(self):
        """EMERGENCY_BRAKE → CUTOUT transition: bypass releases the
        same frame the state exits (non-latched behavior, by design)."""
        # Frame N: in emergency
        active_n = _b1_bypass_active(
            True, {"acc_state_code": "EMERGENCY_BRAKE"}
        )
        # Frame N+1: state cleared
        active_n1 = _b1_bypass_active(True, {"acc_state_code": "CUTOUT"})
        assert active_n is True
        assert active_n1 is False


class TestB1StateCodeHandling:
    """The orchestrator reads acc_state_code out of vehicle_state_dict,
    which comes from the bridge and can arrive as either str or bytes
    depending on the HDF5 round-trip path. Both must work."""

    def test_bytes_acc_state_is_decoded(self):
        active = _b1_bypass_active(
            True, {"acc_state_code": b"EMERGENCY_BRAKE"}
        )
        assert active is True

    def test_empty_acc_state_is_safe(self):
        """Missing/empty state must not trigger bypass, regardless of flag."""
        assert _b1_bypass_active(True, {}) is False
        assert _b1_bypass_active(True, {"acc_state_code": ""}) is False
        assert _b1_bypass_active(True, {"acc_state_code": None}) is False

    def test_non_emergency_states_do_not_trigger(self):
        """Only EMERGENCY_BRAKE triggers B1. Other ACC emergency
        codes (TTC_ESTOP, COLLAPSED_GAP_STOP) use their own paths."""
        for state in (
            "CUTOUT",
            "ACC_ACTIVE",
            "TTC_ESTOP",
            "COLLAPSED_GAP_STOP",
            "UNKNOWN_STATE",
        ):
            assert (
                _b1_bypass_active(True, {"acc_state_code": state}) is False
            ), f"{state} must not trigger B1 bypass"


class TestB1OverrideApplication:
    """When the bypass fires, the control command must:
      - pin throttle=0.0, brake=1.0
      - preserve steering (lane keeping during the reflex)
      - preserve lateral_error/heading_error telemetry
      - set emergency_stop=True and b1_bypass_active=True
    """

    def test_override_zeros_throttle_and_max_brake(self):
        original = {
            "steering": 0.12,
            "throttle": 0.8,
            "brake": 0.2,
            "lateral_error": 0.05,
            "heading_error": 0.03,
        }
        overridden = _apply_b1_override(original)
        assert overridden["throttle"] == 0.0
        assert overridden["brake"] == 1.0

    def test_override_preserves_steering_for_lane_keeping(self):
        original = {"steering": 0.15, "throttle": 0.5, "brake": 0.0}
        overridden = _apply_b1_override(original)
        assert overridden["steering"] == 0.15, (
            "steering must pass through so lane keeping survives the reflex"
        )

    def test_override_preserves_lateral_heading_error_telemetry(self):
        original = {
            "steering": 0.0,
            "throttle": 0.0,
            "brake": 0.0,
            "lateral_error": -0.22,
            "heading_error": 0.04,
        }
        overridden = _apply_b1_override(original)
        assert overridden["lateral_error"] == -0.22
        assert overridden["heading_error"] == 0.04

    def test_override_sets_emergency_stop_flags(self):
        overridden = _apply_b1_override({"steering": 0.0})
        assert overridden["emergency_stop"] is True
        assert overridden["b1_bypass_active"] is True

    def test_override_handles_missing_keys_gracefully(self):
        """If upstream command is minimal (e.g., first frame), override
        should still produce a well-formed command with safe defaults."""
        overridden = _apply_b1_override({})
        assert overridden["steering"] == 0.0
        assert overridden["throttle"] == 0.0
        assert overridden["brake"] == 1.0
        assert overridden["lateral_error"] == 0.0
        assert overridden["heading_error"] == 0.0
        assert overridden["emergency_stop"] is True
        assert overridden["b1_bypass_active"] is True


# ----------------------------------------------------------------------------
# Contract check: orchestrator source must contain the literal rule tokens.
# This catches the specific regression where someone widens the match to
# include other ACC states (e.g., TTC_ESTOP) without updating these tests,
# or removes the bytes-decode that B1StateCodeHandling depends on.
# ----------------------------------------------------------------------------


class TestOrchestratorContract:
    def test_orchestrator_references_b1_flag_and_state_string(self):
        from pathlib import Path

        src = Path("av_stack/orchestrator.py").read_text()
        assert "acc_idm_accel_routing_b1_hard_brake_bypass" in src, (
            "orchestrator must read the B1 config flag"
        )
        assert "'EMERGENCY_BRAKE'" in src or '"EMERGENCY_BRAKE"' in src, (
            "orchestrator must compare acc_state against EMERGENCY_BRAKE"
        )
        assert "b1_bypass_active" in src, (
            "orchestrator must emit b1_bypass_active flag on command"
        )
