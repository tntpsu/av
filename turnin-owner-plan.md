# Clean Turn-In Ownership Plan

## Goal

Fix late curve turn-in without adding more overlapping policies.

The target architecture is:

1. Map/preview owns curve awareness and speed preparation.
2. Local scheduler owns only local curve activation state.
3. Reference generation owns steering-onset timing.
4. Controller tracks the generated reference and does not invent another turn policy.

This plan is intentionally designed to remove fighting layers, not add another one.

## Current Execution Status

Phase A and Phase B have both now been executed.

Current status:

1. Phase A succeeded in cleaning up scheduler trust:
   - startup/pre-distance arm removed
   - `commit_without_ready = 0`
   - `commit_without_distance_ready = 0`
   - no re-entry-without-gate
   - no watchdog ping-pong
2. Phase B has now been implemented and validated on:
   - `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260308_005118.h5`
3. The Phase B owner handoff is real:
   - `turn_in_owner.owner_mode = phase_active`
   - `turn_in_owner.entry_weight_source = curve_local_phase`
   - `turn_in_owner.fallback_active_rate = 0.0%`
4. First owner-path shortening pass has now been executed on:
   - `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260308_013250.h5`
5. This improved the drive materially:
   - overall score `95.4`
   - lateral RMSE `0.298 m`
   - lateral P95 `0.437 m`
   - C1 peak lateral `0.669 m`
   - C3 peak lateral `0.634 m`
   - PP floor rescue reduced on C1/C3
6. But the analyzer still does not call it fully resolved:
   - C1 onset vs curve start `+21 frames`
   - C3 onset vs curve start `+18 frames`
   - local curve active on straights still `32.0%`
   - cadence still not tuning-valid
7. Decision:
   - keep the owner-path shortening direction
   - do not promote highway yet
   - next work, if needed, should be incremental refinement of active-owner entry timing / straight latch behavior, not new policy layers

## Workstream C1 Completed (2026-03-09)

Root cause: `curve_local_phase_time_start_s: 2.0` and `curve_local_phase_time_start_tight_s: 2.5` introduced in this session were too wide. On s_loop, the lane polynomial sees curves within ~9m everywhere due to tight geometry, so `time_to_curve ≈ 2.0s` throughout S2 — below the 2.5s threshold, opening the time gate for the entire straight. Fix: reverted both to `1.2` (same as code default used by Phase B).

2-run result after fix:
- S2 latch: 10.8% / 11.2% (from 45.6%)
- Overall latch: 16.7% / 15.6% (from 36.7%)
- C1 peak: 0.572m / 0.612m ≤ 0.67m ✓
- C3 peak: 0.578m / 0.566m ≤ 0.65m ✓
- No e-stops, no out-of-lane ✓
- Score: 95.4 / 95.3
- Highway non-regression: 98.4/100 ✓

All Workstream C1 acceptance gates pass.

---

## Current Post-Validation Position (2026-03-08)

Two repeat `s_loop` validations now show that the active-owner shortening fix is real, not one-off sampling noise:

1. `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260308_013250.h5`
   - score `95.4`
   - lateral RMSE `0.298 m`
   - lateral P95 `0.437 m`
   - C1 peak `0.669 m`
   - C3 peak `0.634 m`
2. `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260308_014146.h5`
   - score `95.1`
   - lateral RMSE `0.299 m`
   - lateral P95 `0.476 m`
   - C1 peak `0.648 m`
   - C3 peak `0.653 m`
3. Compared with the older post-persistence reference `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260307_173138.h5`, both repeats remain materially better on the two problem turns:
   - C1 `0.815 m -> 0.669 / 0.648 m`
   - C3 `0.683 m -> 0.634 / 0.653 m`
4. This means the owner-path shortening direction should be kept.
5. The remaining `s_loop` gaps are now narrower and better defined:
   - `curve_local_active_on_straights` remains high at `32.0% / 33.2%`
   - PP floor rescue is still too large on all four turns, especially C1/C3
   - analyzer still flags late turn-in
   - cadence remains invalid, so these runs are not promotion-valid

The current next phase is therefore:

1. Keep the architecture exactly as-is:
   - map/preview awareness stays
   - `phase_active` remains the owner on `s_loop`
   - PP floor remains a backstop only
2. Do **not** add another lateral policy.
3. Clean up only two residual behaviors:
   - local-state latch on straights
   - oversized PP floor rescue at curve entry

## Latest Cleanup Execution (2026-03-08)

I executed the first narrow cleanup pass with three changes:

1. `curve_local_time_ready` now requires finite `time_to_curve_s` before it can go true.
2. The analyzer/straight-segment summary now excludes `curve_local_in_curve_now` frames from the "active on straights" metric so the metric matches true straight behavior.
3. `/Users/philiptullai/Documents/Coding/av/config/mpc_sloop.yaml` now makes the PP floor a `COMMIT` backstop instead of an `ENTRY` backstop, with a slightly looser PP shorten-slew limit:
   - `pp_curve_local_floor_state_min: COMMIT`
   - `pp_curve_local_shorten_slew_m_per_frame: 0.10`

Validation:

1. Focused tests passed: `136 passed`
2. New compiled-player `s_loop` run:
   - `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260308_105436.h5`

Result:

1. The drive stayed in-family with the prior good repeats:
   - score `95.2`
   - lateral RMSE `0.303 m`
   - lateral P95 `0.467 m`
   - C1 peak `0.674 m`
   - C3 peak `0.681 m`
2. No e-stop, no out-of-lane, no runaway.
3. The reporting fix materially changed the straight-latch story:
   - `Curve Local Active On Straights: 15.9%`
   - straight inspector:
     - `S1: 22.5%`
     - `S2: 11.9%`
4. The cleanup pass did **not** materially reduce the remaining turn-entry rescue problem:
   - C1 `floorRescueMax = 1.736 m`
   - C3 `floorRescueMax = 1.793 m`
5. Turn-in is still late by analyzer metric:
   - onset-vs-start p50 `19 frames`
6. Cadence is still invalid, so the run is not tuning-valid.

Decision after this cleanup pass:

1. Keep the metric fix.
2. Keep the `curve_local_time_ready` semantic fix.
3. The `COMMIT`-only floor change is acceptable to keep because it did not destabilize the run.
4. But it is not sufficient to resolve the remaining C1/C3 rescue problem.
5. The next `s_loop` step should target the owner-path contraction shape itself, not another ownership or scheduler rewrite.

## Current Next Phase: S-Loop Cleanup Without New Layers

### Objective

Reduce residual straight-latch and floor-rescue behavior while preserving the improved C1/C3 tracking already achieved.

### Non-Goals

1. Do not revert `phase_active` ownership on `s_loop`.
2. Do not introduce another curve-entry policy layer.
3. Do not tune highway in the same change set.

### Workstream C1: Reduce Local-State Latch On Straights

Primary files:

1. `/Users/philiptullai/Documents/Coding/av/trajectory/utils.py`
2. `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
3. `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py`
4. `/Users/philiptullai/Documents/Coding/av/data/recorder.py`
5. `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
6. `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/server.py`
7. `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`

Implementation direction:

1. Keep `arm_ready` as a diagnostic/pre-entry readiness signal.
2. Tighten what is allowed to count as active local steering state on straights.
3. Make `ENTRY` and `COMMIT` activation require either:
   - distance-ready, or
   - in-curve, or
   - explicit path-sustain after local relevance was already earned
4. Do not let far/time readiness alone keep active local ownership alive across the straight once the local gate has decayed.
5. Keep the readiness telemetry, but make analyzer and PhilViz distinguish:
   - `ready_on_straight`
   - `active_on_straight`
   so the operator can see whether the system is merely preparing or actually owning turn-in too early.

Expected effect:

1. Preserve early enough arm for C1/C3.
2. Lower `curve_local_active_on_straights` without reintroducing startup arm or re-entry ping-pong.

### Workstream C2: Reduce PP Floor Rescue Magnitude

Primary files:

1. `/Users/philiptullai/Documents/Coding/av/trajectory/utils.py`
2. `/Users/philiptullai/Documents/Coding/av/control/pid_controller.py`
3. `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
4. `/Users/philiptullai/Documents/Coding/av/config/mpc_sloop.yaml`
5. `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
6. `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/server.py`
7. `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`

Implementation direction:

1. Keep the current owner-path shortening logic.
2. Start contraction a little earlier inside the owner path, not with a new guard.
3. Smooth the pre-floor contraction so the floor rescues less magnitude, rather than shortening more abruptly.
4. Treat the PP floor as a backstop only:
   - it should remain available
   - it should intervene less
5. Preserve the current C1/C3 gains while reducing rescue on:
   - C1
   - C2
   - C3
   - C4

Expected effect:

1. Lower `floorRescueMax` and `floorRescueMean`.
2. Keep or improve current C1/C3 peak lateral error.
3. Avoid reintroducing the earlier broad-guard failure mode.

### Workstream C3: PhilViz / Analyzer Triage Requirements

Required surfacing:

1. `curve_local_active_on_straights` by straight segment
2. `curve_local_ready_on_straights` by straight segment
3. `pp_curve_local_floor_active`
4. `floorRescueMax` / `floorRescueMean` per turn
5. owner-path pre-floor contraction rate per turn

Required analyzer checks:

1. distinguish `ready-but-not-active` from `active-too-early`
2. flag turns where floor rescue exceeds the configured budget
3. flag turns where active-on-straight fell, but C1/C3 peak lateral regressed

### Tests Required

1. Unit tests in `/Users/philiptullai/Documents/Coding/av/tests/test_reference_lookahead.py`
   - active local state cannot be sustained by far/time readiness alone on long straights
   - path-sustain still works after true local relevance is earned
   - owner-path contraction stays monotonic and gap-free
2. Passthrough / recorder tests:
   - `/Users/philiptullai/Documents/Coding/av/tests/test_control.py`
   - `/Users/philiptullai/Documents/Coding/av/tests/test_control_command_hdf5.py`
3. Analyzer contract tests:
   - `/Users/philiptullai/Documents/Coding/av/tests/test_drive_summary_contract.py`
4. PhilViz/analyzer parity:
   - both surfaces must report the same active/ready/floor-rescue values on the same recording

### Acceptance Gates

The cleanup phase is only successful if all of the following hold on two compiled-player `s_loop` runs:

1. no e-stop
2. no out-of-lane
3. C1 peak lateral does not regress beyond the current repeat band (`~0.65-0.67 m`)
4. C3 peak lateral does not regress beyond the current repeat band (`~0.63-0.65 m`)
5. `curve_local_active_on_straights` drops materially below the current `32-33%`
6. PP floor rescue max decreases on C1 and C3
7. startup arm, re-entry-without-gate, and watchdog ping-pong stay at `0`

## Current Problem

The current `s_loop` path is still mixing ownership:

1. Far preview is active almost all the time on `s_loop`.
2. Local scheduler now has meaningful telemetry, but it is not yet the clean owner of local activation.
3. `/Users/philiptullai/Documents/Coding/av/config/mpc_sloop.yaml` is still `curve_scheduler_mode: phase_shadow`, so steering-onset ownership is not actually handed to the local scheduler yet.
4. Planner-side shorten guard and PP local floor are both still shaping the same onset region.
5. Result: the system "knows" the curve exists, but steering still starts after curve start on C1/C3.

That is why the stack feels like it is reacting late even though trajectory/map curvature is present.

## Design Principles

1. One owner per question.
2. No duplicated lateral policies at adjacent layers.
3. Keep telemetry, remove ambiguity.
4. Promote only after trust gates pass.
5. Backstops may remain temporarily, but they must not shape nominal behavior.

## Target Ownership Model

### Layer 1: Map / Preview

Owner: trajectory/map preview path

Responsibilities:

1. Compute preview curvature and curve severity.
2. Compute distance/time to next curve.
3. Feed longitudinal speed planning and diagnostics.
4. Feed local scheduler inputs.

Must not:

1. Directly force local steering activation.
2. Directly own PP/local turn-in timing.

Primary code:

1. `/Users/philiptullai/Documents/Coding/av/trajectory/utils.py`
2. `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`

### Layer 2: Local Scheduler

Owner: `compute_curve_phase_scheduler(...)`

Responsibilities:

1. Decide `STRAIGHT -> ENTRY -> COMMIT -> REARM -> STRAIGHT`.
2. Use only local-relevance evidence for local activation.
3. Allow sustain only once the curve is genuinely local.
4. Expose trust telemetry explaining why it is active.

Must not:

1. Directly tune lookahead magnitude.
2. Directly apply steering floor behavior.
3. Reuse far-preview awareness as implicit local activation.

Primary code:

1. `/Users/philiptullai/Documents/Coding/av/trajectory/utils.py`

### Layer 3: Reference Generation

Owner: `compute_reference_lookahead(...)` and its call site

Responsibilities:

1. Convert local scheduler state into steering-onset behavior.
2. Own the nominal local lookahead contraction through entry.
3. Be the single source of turn-in timing once promoted.

Must not:

1. Fight the scheduler with a second curve-entry policy.
2. Require a PP floor rescue to create normal turn-in.

Primary code:

1. `/Users/philiptullai/Documents/Coding/av/trajectory/utils.py`
2. `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`

### Layer 4: Controller

Owner: low-level steering tracking

Responsibilities:

1. Track the reference cleanly.
2. Apply physical rate/jerk limits.
3. Report whether it could follow the commanded reference.

Must not:

1. Add another curve-entry policy.
2. Recompute onset timing from far-preview or local heuristics.

Primary code:

1. `/Users/philiptullai/Documents/Coding/av/control/pid_controller.py`

## What We Keep

Keep these:

1. New local scheduler telemetry:
   - `curve_local_arm_ready`
   - `curve_local_time_ready`
   - `curve_local_in_curve_now`
   - `curve_local_arm_phase_raw`
   - `curve_local_sustain_phase_raw`
   - `curve_local_path_sustain_active`
2. Existing analyzer/PhilViz surfacing for curve-local diagnostics.
3. PP local floor temporarily, but only as a safety backstop.
4. Map-based preview curvature and speed cap path.

## What We Stop Treating As Owners

These may remain temporarily, but they must stop being treated as nominal turn-in owners:

1. `reference_lookahead_entry_shorten_*`
2. `pp_curve_local_lookahead_floor_*`
3. any path-only pre-entry local arm behavior
4. any far-preview-driven local commit behavior

The rule is simple:

1. scheduler decides local activation
2. reference generator decides lookahead under that activation
3. controller tracks

## Implementation Plan

### Phase A: Make Local Scheduler Trustworthy

Objective:

Make the scheduler a clean local-activation state machine before it is allowed to own lookahead.

Code changes:

1. `/Users/philiptullai/Documents/Coding/av/trajectory/utils.py`
   - tighten `COMMIT` qualification so `COMMIT` cannot occur before real local readiness
   - keep `ENTRY` permissive enough to prepare, but require stronger evidence for `COMMIT`
   - split arm criteria from commit criteria explicitly
   - ensure path-based sustain is allowed only after local curve entry, not as pre-entry arm
2. `/Users/philiptullai/Documents/Coding/av/config/av_stack_config.yaml`
3. `/Users/philiptullai/Documents/Coding/av/config/mpc_sloop.yaml`
4. `/Users/philiptullai/Documents/Coding/av/config/mpc_highway.yaml`
5. `/Users/philiptullai/Documents/Coding/av/config/highway_15mps.yaml`
   - add separate knobs for:
     - local entry threshold
     - local commit threshold
     - local commit minimum distance/time readiness
     - optional commit hold frames

Add telemetry:

1. `curve_local_commit_ready`
2. `curve_local_commit_driver`
3. `curve_local_commit_without_ready_count`
4. `curve_local_commit_ready_at_entry`

Files:

1. `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py`
2. `/Users/philiptullai/Documents/Coding/av/data/recorder.py`
3. `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
4. `/Users/philiptullai/Documents/Coding/av/control/pid_controller.py`
5. `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
6. `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
7. `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/server.py`
8. `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`

Acceptance to leave Phase A:

1. `curve_local_arm_without_ready_count = 0`
2. `curve_local_commit_without_distance_ready_count = 0`
3. `curve_local_commit_without_ready_count = 0`
4. `curve_local_reentry_without_gate_count = 0`
5. `curve_local_watchdog_pingpong_count = 0`
6. `curve_local_active_on_straights` materially lower than current `~31%`

If Phase A does not pass:

1. do not touch lookahead ownership
2. do not retune floor/guard first

### Phase B: Hand Steering-Onset Ownership To Reference Generation

Objective:

Once the scheduler is trusted, make reference generation the single nominal owner of turn-in timing.

Code changes:

1. `/Users/philiptullai/Documents/Coding/av/config/mpc_sloop.yaml`
   - switch `curve_scheduler_mode: phase_shadow -> phase_active`
2. `/Users/philiptullai/Documents/Coding/av/trajectory/utils.py`
   - under `phase_active`, use `curve_local_phase` as the entry-weight owner for lookahead contraction
   - keep binary/far-preview path as fallback only, not nominal owner
3. `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
   - ensure diagnostics clearly report:
     - active owner mode
     - active entry-weight source
     - whether fallback path was used

Add telemetry:

1. `reference_lookahead_owner_mode`
2. `reference_lookahead_entry_weight_source`
3. `reference_lookahead_fallback_active`

Acceptance to leave Phase B:

1. steering onset before curve start on C1 and C3
2. lower C1/C3 peak lateral error versus `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260307_213812.h5`
3. lower PP floor rescue on C1/C3
4. no regression in out-of-lane or e-stop behavior

### Phase C: Demote Temporary Backstops

Objective:

After `phase_active` proves it owns nominal turn-in, reduce overlapping heuristics.

Code changes:

1. `/Users/philiptullai/Documents/Coding/av/config/mpc_sloop.yaml`
2. `/Users/philiptullai/Documents/Coding/av/config/av_stack_config.yaml`
   - reduce or disable:
     - `reference_lookahead_entry_shorten_*`
     - `pp_curve_local_lookahead_floor_*`
   - keep only the minimum backstop needed for safety
3. `/Users/philiptullai/Documents/Coding/av/trajectory/utils.py`
   - if needed, mark guard/floor engagement as "backstop only" in diagnostics

Acceptance to leave Phase C:

1. PP floor rescue is rare and small
2. entry-shorten guard is rare and small
3. trajectory score improves or stays stable
4. no new oscillation/runaway behavior

### Phase D: Generalize Across Curve Radii

Objective:

Make the solution robust for both gentle and tight curves without adding separate policies per track.

Implementation:

1. keep one scheduler
2. keep one owner path
3. allow only severity-scaled parameters inside that path

Allowed severity scaling:

1. local entry horizon
2. local commit horizon
3. local reference lookahead target

Not allowed:

1. separate turn-type policies
2. multiple competing local entry mechanisms
3. curve-specific ad hoc overrides

Files:

1. `/Users/philiptullai/Documents/Coding/av/trajectory/utils.py`
2. `/Users/philiptullai/Documents/Coding/av/config/*.yaml`

## PhilViz / Analyzer Requirements

These are required so the next debugging loop is not ambiguous.

### Summary Card

Add a `Turn-In Owner` summary card showing:

1. scheduler mode
2. active reference-lookahead owner
3. fallback active rate
4. commit-without-ready count
5. arm-without-ready count
6. median steering onset vs curve start

### Per-Turn Table

For each curve, show:

1. local `ENTRY` frame and distance
2. local `COMMIT` frame and distance
3. steering onset frame and distance
4. onset minus curve-start frames
5. PP floor rescue max/mean
6. owner mode
7. fallback active

### Timeline Rows

Add explicit rows for:

1. `curve_local_commit_ready`
2. `reference_lookahead_owner_mode`
3. `reference_lookahead_entry_weight_source`
4. `reference_lookahead_fallback_active`

### Analyzer / PhilViz Parity Requirement

This is mandatory.

The CLI analyzer and PhilViz must tell the same ownership story for the same recording.

Required implementation:

1. `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
   - compute the canonical owner/fallback/commit-ready summary values
2. `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
   - print those canonical values directly from the summary contract
3. `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/server.py`
   - expose the same canonical fields from the analyzer contract or summary payload
4. `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`
   - render the same values without recomputing different logic in the UI

Rule:

1. owner mode
2. fallback active
3. commit-ready counts
4. steering-onset timing

must come from one canonical analysis contract, not separate UI-only logic.

## Tests

### Unit Tests

Update:

1. `/Users/philiptullai/Documents/Coding/av/tests/test_reference_lookahead.py`
2. `/Users/philiptullai/Documents/Coding/av/tests/test_control.py`
3. `/Users/philiptullai/Documents/Coding/av/tests/test_control_command_hdf5.py`
4. `/Users/philiptullai/Documents/Coding/av/tests/test_drive_summary_contract.py`

Add coverage for:

1. `ENTRY` can arm before `COMMIT`
2. `COMMIT` requires commit-ready evidence
3. path-only sustain does not create pre-entry arm
4. `phase_active` uses local phase as owner
5. fallback is labeled when triggered

Add parity / compatibility coverage for:

1. older recordings without the new owner fields still analyze cleanly
2. PhilViz summary fields match analyzer contract fields
3. owner/fallback/commit-ready values are not recomputed differently in UI code

### Live Validation

Order:

1. `s_loop` first
2. highway second

Compiled-player command:

```bash
source /Users/philiptullai/Documents/Coding/av/venv/bin/activate && \
./start_av_stack.sh --force --build-unity-player --skip-unity-build-if-clean --run-unity-player \
  --track-yaml /Users/philiptullai/Documents/Coding/av/tracks/s_loop.yml \
  --config /Users/philiptullai/Documents/Coding/av/config/mpc_sloop.yaml \
  --duration 65
```

Then highway:

```bash
source /Users/philiptullai/Documents/Coding/av/venv/bin/activate && \
./start_av_stack.sh --force --build-unity-player --skip-unity-build-if-clean --run-unity-player \
  --track-yaml /Users/philiptullai/Documents/Coding/av/tracks/highway_65.yml \
  --config /Users/philiptullai/Documents/Coding/av/config/mpc_highway.yaml \
  --duration 65
```

### End-Of-Plan Revalidation

The plan is not complete until live revalidation is finished.

Required closeout sequence:

1. rerun `s_loop` with compiled player after Phase A
2. rerun `s_loop` again after Phase B
3. compare against:
   - `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260307_213812.h5`
   - `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260307_173138.h5`
4. only if `s_loop` passes, run highway non-regression
5. analyze both runs with:
   - `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
6. verify PhilViz shows the same owner/turn-in story as the CLI analyzer

The plan only exits green if:

1. tests pass
2. `s_loop` live validation passes
3. highway non-regression passes
4. analyzer and PhilViz agree on the result

## Acceptance Criteria

### S-Loop

1. no e-stop
2. no out-of-lane
3. steering onset before curve start on C1 and C3
4. C1 and C3 peak lateral error improved versus `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260307_213812.h5`
5. PP floor rescue reduced on C1 and C3
6. `curve_local_commit_without_distance_ready_count = 0`

### Highway

1. no e-stop
2. no out-of-lane
3. no new oscillation runaway
4. no material regression in lateral P95 versus current highway baseline

## Highway Impact Containment

This work can affect highway because the scheduler/reference code is shared.

To keep the risk controlled:

1. Phase A code may ship behind shared logic, but promotion decisions are made on `s_loop` first.
2. Phase B ownership promotion must start in:
   - `/Users/philiptullai/Documents/Coding/av/config/mpc_sloop.yaml`
   only.
3. Do not switch:
   - `/Users/philiptullai/Documents/Coding/av/config/mpc_highway.yaml`
   to `phase_active` until `s_loop` has already passed.
4. Highway is a regression gate during `s_loop` development, not a simultaneous tuning target.
5. If highway regresses during Phase A shared-code changes, stop promotion and fix the shared regression before continuing.

## What Not To Do

1. Do not add another curve-entry heuristic layer.
2. Do not tune steering gains first.
3. Do not tune speed caps first.
4. Do not promote `phase_active` before scheduler trust gates pass.
5. Do not rely on PP floor rescue as the normal turn-in mechanism.

## Recommended Execution Order

1. Phase A only
2. Validate on `s_loop`
3. If Phase A passes, Phase B on `s_loop` only
4. Validate and compare against recent references
5. If Phase B passes, Phase C simplification
6. Then highway non-regression
7. Confirm PhilViz/analyzer parity on the final `s_loop` and highway runs

## Expected End State

At the end of this plan:

1. far preview still exists, but only for awareness and speed planning
2. local scheduler is trusted and explainable
3. reference generation is the single nominal owner of steering onset
4. controller only tracks
5. temporary heuristics are demoted to true backstops

That is the cleanest path to fix turn-in without fighting layers.
