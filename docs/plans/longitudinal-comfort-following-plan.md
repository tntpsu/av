# Longitudinal Comfort And Following Plan

## Goal

Fix longitudinal reporting and gap-convergence logic without reintroducing unsafe ACC behavior.

This plan is driven by two reference runs:

- clean baseline:
  - `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_230150.h5`
- failed catch-up experiment:
  - `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_234730.h5`

The clean baseline is safe and high quality overall. The failed experiment proves that a blind positive-acceleration catch-up assist is the wrong fix.

## Executive Conclusion

The current `over_conservative_trailing` classification is not comparing the baseline run against the actual IDM equilibrium the controller is using.

Today we report convergence against:

- `vehicle/acc_target_gap_m = s0 + v*T`

But the controller behavior is governed by IDM:

- `a_idm = a_max * [1 - (v/v0)^4 - (s*/gap)^2]`
- `s* = s0 + max(0, v*T + v*Δv/(2*sqrt(a*b)))`

That means:

1. `acc_target_gap_m` is a useful diagnostic.
2. It is not the steady-state equilibrium gap when `v` is close to `v0`.
3. So the current reporting overstates “failure to converge.”

On the clean baseline, late steady-state frames show:

- actual gap p50:
  - `27.55 m`
- simple target gap `s0+vT` p50:
  - `20.11 m`
- actual minus simple target:
  - `+7.43 m`
- ego speed p50:
  - `12.04 m/s`
- final target speed p50:
  - `12.41 m/s`
- closing rate p50:
  - `0.02 m/s`

With those values, the IDM interaction term is still large enough that the controller is already near a deceleration equilibrium. So the clean run is not evidence that “the controller should obviously close 7 m more.”

The failed experiment confirms this:

- adding blind catch-up acceleration did not improve convergence
- it reduced robustness
- it produced:
  - detection dropouts
  - emergency brake events
  - collapsed-gap stop
  - off-road failure

So the next step is not another runtime catch-up tweak. It is a contract and attribution cleanup first, then an explicitly bounded acquisition-mode design if policy still needs to change.

## Current Evidence

### Clean baseline

Reference:
- `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_230150.h5`

What is true:

1. Safety is good.
- collisions: `0`
- near misses: `0`
- TTC p05: `26.58 s`
- minimum gap: `27.3 m`

2. ACC owns the longitudinal target cleanly.
- `vehicle/acc_target_speed_mps` and `control/target_speed_final` match in steady ACC_ACTIVE operation

3. Late steady-state closure is extremely small.
- late closing rate p50:
  - `0.02 m/s`
- late `acc_target_speed - ego_speed` p50:
  - `0.36 m/s`
- late `acc_target_speed - estimated_lead_speed` p50:
  - `0.38 m/s`

4. The currently reported “too conservative” gap is against the wrong semantic target.
- late actual gap minus `s0+vT` p50:
  - `+7.43 m`
- but this is not the same as “7.43 m above the IDM equilibrium gap”

### Failed catch-up experiment

Reference:
- `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_234730.h5`

What is true:

1. The additive catch-up assist was unsafe.
- score: `59.0`
- first `gap <= 10m`: frame `570`
- lead-collision override starts: frame `954`
- ACC states degraded into:
  - `DETECTION_LOSS`
  - `EMERGENCY_BRAKE`
  - `COLLAPSED_GAP_STOP`

2. It did not even solve the reported convergence problem.
- actual / target gap p50 got worse:
  - `41.59 / 21.63 m`
- `gap > target + 10m` got worse:
  - `65.3%`

3. Detection robustness degraded materially.
- radar detection rate:
  - `81.3%`
- max no-detect run:
  - `26` frames

So a blind additive acceleration bonus is ruled out.

## Root Cause

### RC1. Convergence reporting is using the wrong target semantic

Current reporting in:
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`

uses:
- `acc_target_gap_m = s0 + v*T`

as if it were the authoritative tracking target.

That is not correct for IDM steady-state behavior when:

1. speed is already close to free-flow target
2. interaction term is still non-trivial
3. closure reserve to the lead is small

This makes `over_conservative_trailing` too eager and can push the team toward unsafe fixes.

### RC2. ACC currently has no explicit, bounded acquisition mode

The controller has:

1. emergency / cutout / loss states
2. one generic `ACC_ACTIVE` state for all following

It does not distinguish:

1. far-gap acquisition
2. settled following

That means any attempt to “close faster” by modifying `ACC_ACTIVE` risks changing steady-state safety behavior and interacts badly with radar dropouts.

### RC3. We are missing the telemetry needed to tell whether we are:

1. tracking IDM equilibrium correctly
2. lead-speed limited
3. policy limited
4. detection limited

Right now we only have enough telemetry to argue about `s0+vT`, which is not sufficient.

## Required Design Direction

### Principle 1

Do not tune ACC convergence against `s0+vT` alone.

### Principle 2

Do not add positive acceleration directly into the generic `ACC_ACTIVE` IDM output.

### Principle 3

If we want faster long-gap closure, add a distinct and bounded acquisition policy with:

1. explicit entry/exit rules
2. explicit closure cap
3. explicit stability gates
4. explicit diagnostics

## Workstreams

### L4. ACC Gap Contract Correction

Goal:
- make reporting and debugging compare actual gap to the right longitudinal semantics

Files:
- `/Users/philiptullai/Documents/Coding/av/control/acc_controller.py`
- `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
- `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py`
- `/Users/philiptullai/Documents/Coding/av/data/recorder.py`
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`

Implementation:
1. Export first-class IDM telemetry:
- `acc_idm_dynamic_gap_m`
- `acc_idm_equilibrium_gap_m`
- `acc_idm_accel_mps2`
- `acc_lead_speed_estimate_mps`
- `acc_closure_reserve_mps`
- `acc_convergence_mode`

2. Redefine convergence reporting around three contracts:
- simple target gap:
  - `s0 + v*T`
- dynamic gap:
  - `s*`
- equilibrium-limited regime:
  - actual gap relative to the implied IDM steady-state envelope

3. Change summary classifications:
- `tracking_simple_gap`
- `tracking_dynamic_gap`
- `lead_limited_tracking`
- `policy_limited_tracking`
- `detection_limited_following`
- `compressed`

4. Stop promoting `over_conservative_trailing` as a top issue unless:
- detection is stable
- no recent loss/cutout
- actual gap stays materially above the corrected tracking envelope
- and the condition persists after the startup acquisition window

Acceptance:
- the clean baseline no longer gets a misleading top-level convergence failure
- the summary explains whether the gap is large because of policy, lead speed, or IDM equilibrium

### L5. Detection-Stability Contract For Following

Goal:
- make it obvious when radar instability invalidates convergence analysis

Files:
- `/Users/philiptullai/Documents/Coding/av/control/radar_sensor.py`
- `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
- `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py`
- `/Users/philiptullai/Documents/Coding/av/data/recorder.py`
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`

Implementation:
1. Record detection stability metrics:
- `acc_detection_stable_frames`
- `acc_recent_detection_loss`
- `acc_detection_loss_event_delta`
- `acc_no_detect_run_length`

2. Gate convergence attribution:
- if detection health is poor, classify as `detection_limited_following`
- do not recommend policy tuning from those windows

3. Add a dedicated summary block:
- `ACC Detection Contract`

Acceptance:
- the failed catch-up run is labeled detection-limited / invalid for convergence tuning

### L6. Explicit Far-Acquisition Mode Design

Goal:
- if policy change is still needed after L4-L5, add a separate acquisition mode rather than mutating generic IDM following

Files:
- `/Users/philiptullai/Documents/Coding/av/control/acc_controller.py`
- `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
- tests and tooling files above

Design:
1. Add explicit acquisition state or submode:
- `ACC_ACQUIRE`

2. Enter only when all are true:
- lead detected and stable for N frames
- no recent detection loss
- no recent emergency/collapsed-gap stop
- gap above corrected acquisition threshold
- TTC healthy with wide margin
- closure reserve below minimum desired reserve

3. Acquisition policy should be:
- lead-speed-referenced
- closure-capped
- bounded by free-flow target
- not an additive raw acceleration bonus

4. Example output policy:
- `target_speed = min(free_flow_target, lead_speed_estimate + closure_cap_mps)`
- where `closure_cap_mps` is modest and state-dependent

5. Exit when:
- gap reaches acquisition exit envelope
- detection becomes unstable
- closing rate exceeds cap
- any safety guard approaches warning band

Telemetry:
- `acc_acquire_active`
- `acc_acquire_reason`
- `acc_acquire_closure_cap_mps`
- `acc_acquire_exit_reason`

Acceptance:
- any new acquisition mode must preserve:
  - no collisions
  - no near misses
  - no lead-collision override
  - no detection-health regression
- and materially improve only the corrected convergence metric

### L7. Recommendation Cleanup

Goal:
- keep top-level recommendations aligned with the corrected contract

Files:
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`

Implementation:
1. Replace the current unconditional “ACC following too conservative” recommendation trigger.
2. Use corrected regime attribution from L4-L5.
3. Prefer specific wording:
- `lead-limited following`
- `policy-limited acquisition`
- `detection-limited following`

Acceptance:
- clean safe runs are not pushed toward unsafe “close faster” tuning unless the corrected contract proves policy under-closing

## Validation Order

1. Implement L4 only.
- no runtime ACC behavior change

2. Re-run:
- `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h2_steady.yml`

3. Confirm whether the clean baseline still shows a real convergence problem after corrected semantics.

4. Implement L5.
- make failed / unstable runs explicitly non-actionable for convergence tuning

5. Only if a real policy-limited gap remains, implement L6.

6. Then update L7 recommendation wording.

## Required Output After L4-L5

For any highway ACC run, the tools should answer directly:

1. Is the run safe behind the lead?
2. Is the large gap simply `s0+vT` vs IDM equilibrium semantics?
3. Is the car lead-speed limited?
4. Is the car policy limited?
5. Is the run invalid for convergence tuning because detection stability is bad?

## Immediate Recommendation

Do not make another runtime ACC convergence change before L4 and L5 are in place.

The current evidence supports:

1. reporting contract bug
2. missing acquisition-mode architecture
3. unsafe outcome from blind catch-up assist

It does not support another direct tuning pass inside the existing `ACC_ACTIVE` state.

### L8. Curve Catch-Up Lead-Continuity Contract

Goal:
- make ACC robust to same-lane lead tracking through curves without making opposite-lane or wrong-lane vehicles capturable

Reference failure:
- `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260327_095121.h5`

Observed failure chain:
1. H8 is still a real catch-up scenario:
   - lead starts `200 m` ahead at `10 m/s`
   - ego target is `15 m/s`
2. Radar detects the lead and gap closes correctly down to roughly `61 m`, `47 m`, `29 m`, `20 m`, `15 m`, then `~8 m`
3. During curve-entry / curved-road geometry, the forward radar repeatedly drops to zero for `10-28` frame runs
4. ACC exits to `FREE_FLOW` / `DETECTION_LOSS`
5. longitudinal ownership falls back to the speed governor
6. the lead reappears only briefly in `EMERGENCY_BRAKE`, then disappears again
7. first `lead_collision_override_active` appears at frame `715`

Primary root cause hypothesis:
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/AVBridge.cs` uses a strict `±5°` forward-cone check
- on `R500` curves, same-lane lead azimuth can exceed that even when the lead is still the correct ACC target
- producer-side radar continuity is therefore too brittle for curve-following

Files:
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/AVBridge.cs`
- `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/LeadVehicle.cs`
- `/Users/philiptullai/Documents/Coding/av/control/radar_sensor.py`
- `/Users/philiptullai/Documents/Coding/av/control/acc_controller.py`
- `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
- `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py`
- `/Users/philiptullai/Documents/Coding/av/data/recorder.py`
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/server.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`

Implementation:

#### L8.1 Producer-side radar-continuity attribution

Record why a lead was not returned on each frame, not just `detected=0`.

Add fields:
- `radar_fwd_candidate_present`
- `radar_fwd_reject_reason`
- `radar_fwd_target_azimuth_deg`
- `radar_fwd_target_heading_delta_deg`
- `radar_fwd_target_same_lane_confidence`
- `radar_fwd_target_lane_offset_m`
- `radar_fwd_target_arc_distance_m`

Required reject reasons:
- `out_of_range`
- `out_of_cone`
- `wrong_lane`
- `opposite_direction`
- `occluded_or_invalid`
- `no_candidate`

Acceptance:
- H8 can tell us whether the dropout is really `out_of_cone`
- we no longer have to infer producer failure from zeroed radar fields

#### L8.2 Same-lane target association contract

Do not let ACC depend on a pure forward-cone detector.

Producer-side target selection must prefer:
1. same-lane lead candidate
2. positive arc-distance ahead
3. similar heading / travel direction
4. bounded lateral offset relative to ego lane center

Reject:
1. opposite-lane traffic
2. oncoming traffic
3. adjacent-lane wrong target
4. geometrically visible but non-followable objects

Important design rule:
- do not fix H8 by blindly widening `radarHalfAngleDeg`
- widen coverage only if it is paired with same-lane / same-direction gating

Acceptance:
- same-lane curved lead remains capturable
- opposite-lane/oncoming candidate is explicitly rejected with `wrong_lane` or `opposite_direction`

#### L8.3 Short-horizon lead continuity track

ACC should not drop straight to free-flow on every brief producer miss if continuity confidence is still high.

Add a small lead-continuity layer between raw radar return and ACC consumption:
- hold last accepted lead track for a short bounded horizon
- decay confidence with time and geometry error
- use predicted arc-distance / lead-speed estimate during the hold
- force exit immediately if same-lane confidence collapses

This is not multi-object fusion.
It is a bounded continuity bridge for the one authoritative same-lane lead.

Suggested output fields:
- `acc_lead_track_active`
- `acc_lead_track_source`
- `acc_lead_track_age_ms`
- `acc_lead_track_confidence`
- `acc_lead_track_hold_reason`
- `acc_lead_track_drop_reason`

Acceptance:
- brief H8 producer misses do not hand ownership back to free-flow immediately
- lead continuity does not persist through wrong-lane / opposite-direction evidence

#### L8.4 Consumer-state changes

Update ACC state semantics so temporary continuity holds are distinct from true no-lead loss.

Add or derive:
- `LEAD_TRACK_HOLD`
- `detection_limited_following`
- `association_limited_following`

Behavior:
1. raw no-detect with strong continuity confidence:
   - hold bounded lead track
   - do not immediately ramp to free-flow
2. raw no-detect with poor same-lane confidence or stale track:
   - enter `DETECTION_LOSS` / `FREE_FLOW` as today

Acceptance:
- H8 no longer oscillates between `ACC_ACTIVE` and `DETECTION_LOSS` on short curve misses
- true lead loss still exits quickly

#### L8.5 Tooling and triage

Add direct issue types:
- `lead_continuity_curve_dropout`
- `wrong_lane_target_rejected`
- `opposite_direction_target_rejected`
- `lead_track_hold_exhausted`

PhilViz/analyzer cards:
- `Lead Continuity Contract`
- `Lead Association Contract`

They should answer directly:
1. Did we have a real same-lane candidate?
2. Why was it rejected?
3. Did continuity hold bridge the miss?
4. Did ACC fall back because the lead was truly gone, or because producer association failed?

#### L8.6 Validation order

1. Reproduce H8 on current baseline:
- `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h8_curve_catchup.yml`

2. Add attribution only and confirm dominant reject reason.

3. Implement same-lane association without continuity hold.

4. Re-run H8.

5. Implement bounded continuity hold only if H8 still fails due to short producer misses.

6. Re-run H8 and then a same-lane steady baseline:
- `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h2_steady.yml`

7. Add explicit wrong-target tests before declaring success.

#### L8.7 Required new scenarios

Add scenarios or validation cases for:
1. same-lane curve catch-up:
   - existing `H8`
2. opposite-lane oncoming pass on curve
3. opposite-lane oncoming pass on straight
4. wrong-lane same-direction vehicle if the track setup supports it

Promotion gate:
- fixing H8 must not make opposite-lane/oncoming vehicles capturable by ACC

### L9. ACC Robustness Scope

The right target is not “similar to a top company because we widened radar coverage.”

The right target is:
1. explicit same-lane lead association
2. short-horizon track continuity
3. direct rejection of wrong-lane / opposite-direction candidates
4. attribution that makes failures obvious

This is directionally similar to how production systems are structured:
- object association first
- continuity / tracking second
- controller consumption third

What we have today is still much simpler than a top-tier production stack.
Top programs usually rely on:
1. multi-object tracking
2. lane association against map / path geometry
3. same-direction gating
4. continuity through temporary sensor misses
5. multi-sensor fusion, not a single narrow forward-cone producer

So the answer is:
- yes, the direction in L8 is aligned with a serious architecture
- no, the current `±5° cone -> zero return -> free-flow` behavior is not production-grade
