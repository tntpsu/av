# Roadmap: Phase 2 and Beyond (with Test Gates)

## Scope
This roadmap starts after Phase 1 lane‑keeping + longitudinal fidelity is complete.
It defines phased feature work and the test gates required to advance.

## Common Test Gates (All Phases)
- **No emergency stops:** `emergency_stop_frame == 0` across all runs.
- **Determinism:** repeated runs within ±10% on key metrics.
- **Recording completeness:** required telemetry present in HDF5 and visualizer.

## Phase 2: Curvature Conditioning + Comfort Stabilization
**Goal:** reduce curvature noise and stabilize comfort without sacrificing centering.

**Key features**
- Curvature smoothing/preview window (10–20 m path window).
- Use smoothed curvature for:
  - lateral jerk calculation
  - curvature‑based speed limits
  - steering rate scaling
- Lateral accel cap integrated into speed planner (v^2 * curvature).

**Test gates**
- **Curvature sweep:** r20/r30/r40/r60, 40s each.
- **Metrics:**
  - r20/r30 centered >= 70%
  - lateral_rmse <= 0.40m
  - accel_p95 <= 25 m/s^2
  - jerk_p95 <= 400 m/s^3
  - lat_jerk_p95 <= 5.0

**Artifacts**
- Updated `curve_sweep.py` report with curvature‑smoothed metrics.
- Visualizer panel showing smoothed curvature and comfort metrics.

## Phase 3: Behavior Layer + Lead Vehicle Integration
**Goal:** enable realistic car‑following with jerk‑limited deceleration.

**Key features**
- Behavior layer with IDM/ACC‑style desired speed.
- Lead‑vehicle model and synthetic lead profiles in tests.
- Speed planner consumes behavior‑level desired speed.

**Test gates**
- **Lead‑vehicle scenarios:**
  - Constant lead speed (5–10 m/s)
  - Decel event (lead slows 2–3 m/s^2)
  - Stop‑and‑go (low‑speed oscillations)
- **Metrics:**
  - No collision or emergency stop
  - Min time‑gap >= 1.5s
  - Jerk_p95 <= 400 m/s^3
  - Speed RMSE <= 2.0 m/s vs desired

**Artifacts**
- `tools/analyze/lead_vehicle_eval.py`
- Visualizer overlay of lead distance, time gap, desired vs planned speed

## Phase 4: Multi‑Lane Behavior (Highway)
**Goal:** lane change, merge, pass, and re‑center without destabilizing control.

**Key features**
- Lane change planner with safety checks.
- Multi‑lane map support in YAML (lane count, lane widths).
- Lateral decisioning with gap acceptance.

**Test gates**
- **Scenario suite:** merge, pass slower vehicle, lane change to avoid slow lead.
- **Metrics:**
  - Lane change success rate >= 95%
  - No collision
  - Lateral jerk p95 <= 6.0
  - Time‑in‑target‑lane >= 95%

**Artifacts**
- `tools/analyze/lane_change_eval.py`
- Visualizer timeline: lane change start/commit/complete

## Phase 5: Urban Features (Intersections + Signals)
**Goal:** handle traffic lights, stop lines, and unprotected turns.

**Key features**
- Traffic light state ingestion (map or simulated perception).
- Stop line handling with smooth braking.
- Protected/unprotected turn policies.

**Test gates**
- **Urban scenarios:** stop, go, yellow decision, unprotected turn.
- **Metrics:**
  - Stop line compliance >= 98%
  - Red‑light violations == 0
  - Jerk_p95 <= 450 m/s^3

**Artifacts**
- `tools/analyze/intersection_eval.py`
- Visualizer: signal state + stop line overlay

## Phase 6: Robustness + Regression
**Goal:** ensure stability under sensor noise and environment changes.

**Key features**
- Noise injection (perception perturbations, latency).
- Wind/grade variability.
- Regression suite on all prior scenarios.

**Test gates**
- **Stress tests:** noise + latency + grade.
- **Metrics:**
  - <= 10% degradation in core metrics
  - No emergent failure modes

**Artifacts**
- `tools/analyze/robustness_eval.py`
- CI gate for regression suite
