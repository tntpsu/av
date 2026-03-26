# Highway Mild-Curve Lateral Plan

## Goal

Fix the highway lateral oscillation/swerving on gentle map-backed arcs in a way that is:

1. robust on `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h2_steady.yml`
2. diagnosable in PhilViz/CLI without manual reverse-engineering
3. transferable to other mild-curvature highway tracks, not just this one recording

This plan is based on:
- `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_183057.h5`

## Current Root Cause

The current run is clean enough to trust transport and perception:

- transport completeness/fallback: `100.0% / 0.0%`
- coherence pass: `100.0%`
- lane detection: `100.0%`
- stale perception: `0.0%`

The highway failure is now a lateral/reference contract problem:

1. the map/reference says a mild arc exists at about `kappa ~= 0.002`
2. highway curve recognition remains inactive:
   - `curve_intent_state = STRAIGHT`
   - `curve_local_state = STRAIGHT`
3. lookahead remains long:
   - `reference_lookahead_target` on high-error frames: about `14.4 m`
   - `pp_lookahead_distance` on high-error frames: about `9.4 m`
4. MPC is feasible and not clipped/rate-limited
5. steering remains too soft while the controller chases a far-ahead point
6. lateral error grows into an oscillatory chase

Important nuance:

- `control/lateral_error` gets large
- but `vehicle/road_frame_lane_center_offset` stays small, about `4-5 cm` p95 on high-error frames

That means the car is not physically near lane departure. The stack is mostly wrong about the reference geometry it should be tracking through the mild arc.

## Why Current Tooling Is Not Enough

Today the tools expose symptoms, but not the actual contract failure.

### What exists today

Generic signals exist in:
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/backend/triage_engine.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/backend/issue_detector.py`
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`

Examples:
- `ctrl_lateral_oscillation`
- `mpc_steering_oscillation`
- `curve late turn-in (lookahead floor rescue)`
- analyzer warnings like:
  - oscillation amplitude growth
  - curve intent late arm

### What is missing

Nothing today directly says:

1. mild highway arc is present in the reference
2. curve recognizer stayed `STRAIGHT`
3. lookahead stayed in straight/high-speed mode
4. actual road-frame lane offset stayed small
5. therefore the root cause is under-activated mild-curve reference shaping, not perception or transport

That missing attribution is the first thing this plan fixes.

## Success Criteria

Functional:

1. highway mild arcs with `reference_point_curvature ~= 0.002` are recognized as curves early enough
2. lookahead contracts for that regime before large oscillatory error appears
3. lateral RMSE/P95 materially improve on the same scenario
4. no regression in transport, safety, or ACC ownership

Attribution:

1. PhilViz and CLI can directly identify mild-curve under-activation
2. issue detector and triage surface a dedicated root-cause event, not just generic oscillation
3. actual lane offset vs controller lateral error is visible side-by-side

## Phase H1: Add Explicit Attribution First

### Why

If the next runtime change works, we need to know why. If it fails, we need to know whether it failed because:

1. mild-curve recognition still did not arm
2. recognition armed but lookahead stayed long
3. lookahead shortened but actual lane offset still oscillated

### Files

- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/server.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/backend/issue_detector.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/backend/triage_engine.py`

### New first-class metrics

Add per-frame/summary metrics for:

1. `mild_curve_present`
   - based on map/reference curvature band, not GT-only
   - target band should cover highway `~0.002`

2. `mild_curve_recognition_inactive`
   - true when:
     - mild curve present
     - `curve_intent_state == STRAIGHT`
     - `curve_local_state == STRAIGHT`

3. `reference_geometry_mismatch`
   - true when:
     - `|control/lateral_error|` high
     - `|vehicle/road_frame_lane_center_offset|` small

4. `mild_curve_long_lookahead`
   - true when:
     - mild curve present
     - lookahead remains in straight/high-speed band

5. `mild_curve_underactivated_tracking_window`
   - composite diagnostic:
     - mild curve present
     - recognizer inactive
     - long lookahead
     - high controller lateral error
     - small actual lane offset

### New PhilViz cards

Add:

1. `Highway Curve Contract`
   - mild-curve present rate
   - recognition-active rate
   - under-activation rate
   - long-lookahead-on-mild-curve rate

2. `Reference Geometry Mismatch`
   - `control/lateral_error` P50/P95
   - `vehicle/road_frame_lane_center_offset` P50/P95
   - mismatch rate

3. `Mild Curve Steering Response`
   - `reference_point_curvature`
   - `mpc_kappa_ref`
   - `pp_lookahead_distance`
   - steering command

### New timeline rows

Add rows for:

1. `mild_curve_present`
2. `mild_curve_recognition_inactive`
3. `reference_geometry_mismatch`
4. `mild_curve_long_lookahead`
5. `road_frame_lane_center_offset`
6. `control/lateral_error`

### New issue detector item

Add a dedicated issue:

- `highway_mild_curve_underactivation`

Condition:

1. mild curve present
2. `curve_intent_state` inactive
3. `curve_local_state` inactive
4. long lookahead retained
5. `|control/lateral_error|` exceeds threshold
6. `|vehicle/road_frame_lane_center_offset|` remains small

This should be a direct root-cause issue, not a derived summary.

### New triage pattern

Add to:
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/backend/triage_engine.py`

Pattern:
- `highway_mild_curve_underactivation`

Triage output should say plainly:

> Mild map-backed highway arc is present, but curve recognition stayed STRAIGHT and lookahead remained in straight mode. Large controller lateral error with small actual lane offset indicates a reference geometry mismatch, not transport or perception failure.

Config levers should point to:
- `/Users/philiptullai/Documents/Coding/av/config/av_stack_config.yaml`
- `/Users/philiptullai/Documents/Coding/av/config/mpc_highway.yaml`
- `/Users/philiptullai/Documents/Coding/av/config/acc_highway.yaml`

## Phase H2: Short-Lived Hypothesis Validation in Highway Overlays

### Why

This phase is not the final architecture.

It exists to answer one question quickly and cleanly:

> If the stack recognizes the mild highway arc earlier and shortens lookahead for that regime, does the swerving/oscillation collapse on the clean transport run?

The overlay changes in this phase are diagnostic scaffolding. They are allowed only as a temporary proof step before the behavior is lifted into dynamic core logic.

The current base thresholds are tuned above the actual highway arc in this run.

Observed in the recording:
- `reference_point_curvature ~= 0.002`

Current base thresholds still include:
- `curve_phase_preview_enter_threshold: 0.01`
- `curve_phase_preview_exit_threshold: 0.008`
- `curve_phase_preview_curvature_min: 0.0025`
- `curve_phase_path_curvature_min: 0.003`
- `curve_local_phase_path_curvature_min: 0.003`
- `curve_local_entry_severity_curvature_min: 0.003`
- `reference_lookahead_entry_preview_curvature_min: 0.003`

That is too high for this regime.

### Files

- `/Users/philiptullai/Documents/Coding/av/config/mpc_highway.yaml`
- `/Users/philiptullai/Documents/Coding/av/config/acc_highway.yaml`

### Required changes

Override highway-specific mild-curve thresholds so they cover the observed arc band.

Targets to override:

1. `curve_phase_preview_enter_threshold`
2. `curve_phase_preview_exit_threshold`
3. `curve_phase_preview_curvature_min`
4. `curve_phase_path_curvature_min`
5. `curve_local_phase_path_curvature_min`
6. `curve_local_entry_severity_curvature_min`
7. `reference_lookahead_entry_preview_curvature_min`
8. `local_curve_reference_curvature_min`

Also add temporary highway-overlay shaping for:

1. `reference_lookahead_speed_table_entry`
2. if needed, `reference_lookahead_speed_table`
3. if needed, mild-curve-specific owner scaling in the highway overlay

The key is not globally shortening lookahead. It is:

1. shortening only for the gentle arc regime
2. only when curve recognition is actually active
3. using the overlay only to validate the diagnosis on the existing run/scenario

### Acceptance gate for H2

On the rerun:

1. mild-curve-present frames should no longer have near-zero recognizer activation
2. `mild_curve_recognition_inactive` should materially drop
3. startup and transport remain clean

### Exit rule for H2

Do not stop at a passing overlay-tuned highway run.

If H2 fixes the swerving, the next phase is mandatory:
- move the logic into dynamic core behavior
- remove reliance on track-specific mild-curve thresholds as the real solution

## Phase H3: Lift the Fix Into Dynamic Core Logic

### Why

The final stack should not depend on `highway`-specific magic numbers for mild arcs.

The real fix should be transferable across tracks and should derive behavior from:

1. current speed
2. map-backed curvature ahead
3. distance/time to next curve
4. curve persistence / expected heading change
5. reference confidence and coherence

That means the overlay values from H2 are only allowed to serve as a proof step.

### Files

- `/Users/philiptullai/Documents/Coding/av/config/av_stack_config.yaml`
- `/Users/philiptullai/Documents/Coding/av/trajectory/utils.py`
- `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/server.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`

### Required changes

Replace fixed mild-curve thresholds/lookahead behavior with dynamic regime logic.

#### H3.1 Dynamic mild-curve activation

Do not rely on fixed values like:

- `curve_phase_preview_curvature_min`
- `curve_phase_path_curvature_min`
- `curve_local_phase_path_curvature_min`
- `curve_local_entry_severity_curvature_min`

as the final owner of behavior.

Instead derive activation from a dynamic signal such as:

1. map/reference curvature magnitude
2. expected heading change over preview horizon
3. lateral displacement significance over preview horizon
4. distance/time to next curve
5. speed

Practical direction:

- compute a mild-curve relevance score, not just `kappa > fixed_min`
- allow long-radius arcs at high speed to activate even when raw `kappa` is only around `0.002`

#### H3.2 Dynamic lookahead shaping

Replace the final reliance on fixed highway overlay tables with a dynamic function:

- `L = f(speed, curvature_severity, distance_to_curve, time_to_curve, curve_progress)`

Expected behavior:

1. straight road: long lookahead
2. mild highway arc at speed: moderate contraction
3. tighter curve: stronger contraction
4. exit: smooth unwind

#### H3.3 Dynamic local severity

Entry severity should be based on:

1. curvature persistence over distance
2. previewed heading change
3. speed-scaled relevance

not only raw curvature threshold crossing.

#### H3.4 Keep the tooling aligned with the dynamic logic

The debug tools must show:

1. why the dynamic activator fired
2. what dynamic severity bucket/range applied
3. how that changed lookahead

so the new logic is directly attributable.

### Acceptance gate for H3

The dynamic lift is complete only if:

1. the overlay values from H2 are no longer required for the behavior
2. mild-curve recognition still works on the highway scenario
3. the same logic is defensible on other mild-curvature tracks
4. PhilViz/CLI can explain activation from dynamic inputs, not hidden per-track numbers

## Phase H4: Revalidate and Reclassify the Swerving

### Live validation order

1. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h2_steady.yml`
2. `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h8_curve_catchup.yml`

Reference recordings:
- `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_183057.h5`
- `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_094053.h5`

### What to measure

1. `control/lateral_error` RMSE/P95
2. `vehicle/road_frame_lane_center_offset` RMSE/P95
3. `mild_curve_underactivated_tracking_window` rate
4. `curve_intent_state` / `curve_local_state` activation on mild curves
5. `pp_lookahead_distance` on mild curves
6. oscillation zero-cross interval and RMS growth
7. whether the behavior still holds after removing temporary H2 overlay scaffolding

### Pass condition

We should consider the root issue fixed only if:

1. lateral error drops materially
2. road-frame actual lane offset stays low
3. mild-curve under-activation issue disappears or becomes rare
4. transport remains at current clean state
5. the behavior still holds once the solution is represented as dynamic core logic

## Tests

### Analyzer/debug tests

1. Add drive-summary contract tests for:
   - `mild_curve_present`
   - `reference_geometry_mismatch`
   - `mild_curve_underactivated_tracking_window`
   - dynamic activation reason / bucket summaries

2. Add issue detector tests for:
   - triggering `highway_mild_curve_underactivation`
   - not triggering it when actual lane offset is also large

3. Add triage tests ensuring the new pattern emits:
   - root cause
   - config levers
   - distinguishing rationale vs generic oscillation
   - explicit dynamic activator/long-lookahead explanation after H3

4. Add PhilViz/CLI parity tests for the new rates and modes.

### Runtime tests

1. Unit tests for dynamic mild-curve activation:
   - low-speed straight should stay inactive
   - high-speed mild arc at `kappa ~= 0.002` should activate
   - tighter curve should activate more strongly

2. Unit tests for dynamic lookahead shaping:
   - increasing speed on same mild arc should reduce lookahead relevance threshold
   - increasing curvature persistence should shorten lookahead
   - exit/unwind should lengthen lookahead smoothly

3. Contract tests for debug surfaces:
   - dynamic activation reason is recorded
   - dynamic lookahead regime is recorded

### Live analysis checks

After the first rerun, explicitly inspect:

1. frames where `|control/lateral_error| > 0.7`
2. overlap with:
   - `mild_curve_present`
   - `curve_intent_state != STRAIGHT`
   - `curve_local_state != STRAIGHT`
   - `vehicle/road_frame_lane_center_offset`

If the next run still shows:
- small actual lane offset
- large controller lateral error

then keep working on reference geometry/recognition.

If the next run instead shows:
- actual lane offset growing with the error

then the problem has transitioned into a true control authority/tuning problem and can be treated separately.

## Recommended execution order

1. Implement H1 first: attribution and triage
2. Implement H2: temporary overlay-based hypothesis validation
3. If H2 confirms the diagnosis, implement H3 immediately: dynamic core logic lift
4. Re-run the same scenario with the dynamic path
5. Only after that decide whether the remaining swerving is still a reference problem or now a true controller tuning problem

## Answer to the current tooling question

No. The current tools do not make this root cause obvious enough.

Today they tell you:

1. oscillation is growing
2. curve intent arms late
3. lookahead is long/oscillatory in generic terms

They do not directly tell you:

1. the arc is mild but real
2. the recognizer stayed inactive
3. the controller error is large while the actual lane offset stays small
4. therefore the failure is a mild-curve reference/activation bug, not perception or transport

That is why this plan includes the issue/triage/debug additions as required work, not optional cleanup.
