# Robust Full-Stack Roadmap (Unified, Layered, and Gated)

**Last Updated:** 2026-02-17  
**Current Focus:** Layer 2, Stage 1 (First-Turn Entry Reliability) + Phase 2 completion gates  
**Change-Control Rule:** If scope, stage, phase status, or promotion gates change, update this roadmap in the same PR/commit before considering work complete.

## Scope
Single source of truth for the long-term AV roadmap and current incremental execution plan.
This combines:
- the original phase roadmap (Phase 0+),
- the robust full-stack trajectory design intent,
- and the current lane-keeping-first execution ladder.

## Status Snapshot (Crossed-out = completed)
- ~~Phase 0: Dynamic-horizon readiness instrumentation and baseline gates~~
- ~~Phase 1: Dynamic effective horizon diagnostics-only policy~~
- ~~Phase 1b: Telemetry/triage tuning pass~~
- Phase 2: Curvature conditioning + comfort stabilization + trajectory integration (**in progress**)
- **Post-Phase-2 checkpoint:** architecture escalation decision (**pending hard stop**)
- Phase 3+: behavior/actors/multi-lane/urban/robustness (**pending**)

## Global Promotion Gates (Apply at Every Stage)
- **Safety:** no uncontained failures (track safety envelope and emergency-stop reason).
- **Determinism:** repeated runs within defined variance on key metrics.
- **Comfort:** steering/longitudinal jerk and rate envelopes stay within limits.
- **Observability:** required telemetry present in HDF5 + PhilViz.
- **Attribution:** for regressions, isolate layer cause (perception/trajectory/control) before promoting.

## Layer 1: Foundation Phases (Completed)

### ~~Phase 0: Readiness + Instrumentation~~
**Delivered**
- ~~Baseline telemetry and readiness checks for dynamic horizon policy.~~
- ~~Initial gate structure and diagnostics surfacing.~~

### ~~Phase 1: Dynamic Horizon (Diagnostics Only)~~
**Delivered**
- ~~Per-frame dynamic effective horizon diagnostics (speed/curvature/confidence scales).~~
- ~~Limiter-code explainability and visualizer exposure.~~

### ~~Phase 1b: Triage/Telemetry Tuning~~
**Delivered**
- ~~Expanded limiter/suppression diagnostics and waterfall observability.~~
- ~~Improved replay/visualization reliability and projection diagnostics.~~

## Layer 2: Lane-Keeping Execution Ladder (Current Working Order)
This is the practical de-risk sequence for implementation and testing.

### Stage 1: First-Turn Entry Reliability (Current Focus)
**Goal:** clear first turn without centerline crossing, out-of-bounds, or comfort breach.

**Gate to pass Stage 1**
- No centerline cross in first-turn window.
- No out-of-bounds in first-turn window.
- Comfort metrics remain within envelope.

### Stage 2: Turn Exit Stability
**Goal:** maintain centering and comfort while unwinding out of turn.

### Stage 3: Turn-to-Turn Handoff
**Goal:** stable transition across entry -> exit -> next entry without oscillatory handoff.

### Stage 4: Longer-Horizon Stability
**Goal:** no drift accumulation over longer runtime.

### Stage 5: Turn-Radius Coverage Expansion
**Goal:** robust lane-keeping across wider turn-radius set.

### Stage 6: Grade and Banking Expansion
**Goal:** add elevation effects after flat-turn robustness is achieved.
- 6a: straight grades.
- 6b: banked turns.

### Stage 7: Actor/Lead-Vehicle Introduction (After Lane-Keeping Robustness)
**Goal:** introduce forward-following behavior only after lane-keeping matrix is stable.

**Entry criteria for Stage 7**
- Lane-keeping pass rate meets target across scenario matrix (for example, >90% repeated runs).
- No comfort regression vs baseline gates.
- Failure modes are rare and attributable.

## Layer 3: Capability Expansion Phases
These are the long-term product capabilities layered on top of the ladder above.

### Phase 2: Curvature Conditioning + Comfort Stabilization (In Progress)
**Goal:** reduce curvature noise and stabilize comfort without sacrificing centering.

**Key features**
- Curvature smoothing/preview window (10-20 m path window).
- Use smoothed curvature for:
  - lateral jerk calculation,
  - curvature-based speed limits,
  - steering-rate scaling.
- Lateral-accel cap integrated into speed planner (`v^2 * curvature`).
- Dynamic effective horizon behavior integration (not diagnostics-only).

**Primary gates**
- Curvature sweep: r20/r30/r40/r60, 40s each.
- r20/r30 centered >= 70%.
- lateral_rmse <= 0.40 m.
- accel_p95 <= 25 m/s^2.
- jerk_p95 <= 400 m/s^3.
- lat_jerk_p95 <= 5.0.

**Post-Phase-2 hard stop (required)**
- Decide whether to escalate to stronger architecture changes (for example MPC / richer behavior stack) before continuing.

### Phase 3: Behavior Layer + Lead Vehicle Integration
**Goal:** realistic car-following with jerk-limited deceleration.

**Key features**
- Behavior layer with IDM/ACC-style desired speed.
- Lead-vehicle model and synthetic lead profiles in tests.
- Speed planner consumes behavior-level desired speed.

**Test gates**
- Lead-vehicle scenarios:
  - constant lead speed (5-10 m/s),
  - lead decel event (2-3 m/s^2),
  - stop-and-go.
- No collision or emergency stop.
- Min time-gap >= 1.5s.
- jerk_p95 <= 400 m/s^3.
- Speed RMSE <= 2.0 m/s vs desired.

### Phase 4: Multi-Lane Behavior (Highway)
**Goal:** lane-change/merge/pass behavior without destabilizing base control.

**Key features**
- Lane-change planner + safety checks.
- Multi-lane map support in YAML (lane count/width).
- Gap-acceptance decisioning.

**Test gates**
- Lane-change success >= 95%.
- No collision.
- Lateral jerk p95 <= 6.0.
- Time-in-target-lane >= 95%.

### Phase 5: Urban Features (Intersections + Signals)
**Goal:** handle stop lines, traffic lights, and turn policies.

**Key features**
- Signal state ingestion.
- Stop-line handling with smooth braking.
- Protected/unprotected turn policy.

**Test gates**
- Stop-line compliance >= 98%.
- Red-light violations == 0.
- jerk_p95 <= 450 m/s^3.

### Phase 6: Robustness + Regression
**Goal:** maintain performance under perturbation and prevent regressions.

**Key features**
- Noise/latency perturbation.
- Wind/grade variability.
- Full regression suite on prior scenarios.

**Test gates**
- Stress: noise + latency + grade.
- <=10% degradation in core metrics.
- No new failure modes.

## Artifacts and Ownership (Rolling)
- Phase reports and gate results stored under `tmp/analysis/`.
- Visualizer panels/waterfalls updated with each promoted diagnostic.
- Scenario and evaluation scripts under `tools/analyze/`.
- This roadmap must be updated whenever a phase/stage changes status.
