Run a targeted diagnostic sequence for the latest (or specified) AV stack recording. Determines the primary issue category and runs the minimal correct tool sequence — no more guessing which of 103 tools to use.

The user's request is: $ARGUMENTS

## Step 1 — Identify recording

If $ARGUMENTS contains a file path (ends in `.h5`), use that file.
Otherwise use `--latest`.

## Step 2 — Run primary analysis AND triage engine

Run BOTH the analysis AND the PhilViz triage engine. The triage engine has 20+ pattern detectors (floor_rescue, heading_oscillation, regime_blend, etc.) that catch known issues automatically — don't rediscover them manually.

```bash
# Primary analysis (metrics + key issues)
python3 tools/analyze/analyze_drive_overall.py --latest

# Triage engine (pattern detection — catches known issues instantly)
python3 tools/analyze/run_gate_and_triage.py --latest
```

**Read the triage output FIRST.** If the triage engine identifies a known pattern (e.g., `traj_curve_late_turn_in`, `pp_floor_formula_curvature_lag`, `heading_gate_oscillation`), use that as the primary diagnosis — don't repeat the investigation manually.

Also run the signal chain blame trace for lateral/trajectory issues:
```bash
python3 tools/analyze/trace_curve_entry.py --latest
```

## Step 2.5 — Odometer sanity check for curve events

If the analysis reports curve events (C1, C2, etc.), cross-check them against the actual track position using the odometer (`odo=` field in the curve event output). Compare the `odo` value to the track YAML segment layout:

1. Read `tracks/<track>.yml` to get segment start/end distances
2. For each curve event, check: does the `odo` value fall within an actual arc segment?
3. If a curve event's `odo` falls on a straight segment, it's a **false positive** — flag it and do not treat it as a real curve failure

Also check the TRACK-END NOTE in the analysis output. If present, the emergency stop may be from driving off a non-looping track, not a control failure.

This prevents wasting time diagnosing false curve detections on straights (e.g., hairpin_15 "C1"/"C2" were on the 80m straight, not real curves).

## Step 3 — Classify the primary issue

Parse the output to identify the issue category. Use **first match** from this priority list:

| Priority | Category | Trigger condition |
|----------|----------|-------------------|
| 1 | `safety_issue` | Emergency stops > 0, OR OOL time > 0% |
| 2 | `acc_issue` | ACC section present AND (Gap RMSE > 0.5m OR TTC < 2.0s OR collision > 0 OR detection rate < 95%) |
| 3 | `mpc_issue` | MPC fallback rate > 0.5% OR LMPC feasibility < 99.5% |
| 4 | `lateral_error_issue` | Lateral RMSE > 0.40m OR OOL events > 0 |
| 5 | `comfort_issue` | Jerk P95 > 6.0 OR Accel P95 > 3.0 |
| 6 | `perception_issue` | Perception detection rate < 80% OR stale rate > 5% |
| 7 | `trajectory_issue` | Trajectory layer score < 80 |
| 8 | `control_issue` | Control layer score < 80 |
| 9 | `unknown` | Overall score < 90 but no specific layer clearly red |

## Step 3.5 — Blame Disambiguation

When the blame output names a signal (e.g., "Ld_target SHORT", "entry_severity LOW", "FF negligible"), disambiguate the failure mode before accepting it as the root cause. A blame label describes WHAT is wrong, not WHY or WHERE to fix it.

For each blame label, ask:

| Blame label | Possible meanings | How to disambiguate |
|-------------|-------------------|---------------------|
| "Ld_target SHORT" | (a) Ld contracted too late (onset timing) | Check onset frame vs curve start frame |
| | (b) Ld contracted to wrong value | Compare Ld at curve vs physics ideal sqrt(8R×e) |
| | (c) Controller ignores Ld (e.g., MPC cost trade-off) | Check if MPC pre-steers despite Ld being correct |
| "entry_severity LOW" | (a) Curve too gentle to trigger | Check κ vs severity thresholds |
| | (b) Severity scaling is correct but insufficient | Check if higher severity would actually help |
| "FF negligible" | (a) FF floor blocks activation | Check κ vs ff_curvature_min |
| | (b) FF gain too low | Check FF contribution vs geometric demand |
| | (c) FF is correct but controller overrides it | Check total steer vs FF alone |

```
BLAME DISAMBIGUATION
============================================================
  Blame: <label from trace_curve_entry>
  Meaning (a): <description> — <evidence for/against>
  Meaning (b): <description> — <evidence for/against>
  
  Root cause: <which meaning, with evidence>
  Actionable subsystem: <what to fix — may differ from the blamed signal>
============================================================
```

**Critical:** "Ld_target SHORT" does NOT always mean "fix Ld." If the controller's cost function accepts entry error as optimal (MPC q_lat trade-off), fixing Ld won't help — the controller will still not pre-steer. The actionable subsystem may be the controller weights, not the lookahead chain.

## Step 4 — Run the targeted tools

### If `safety_issue`:
```bash
python3 tools/analyze/build_failure_packet.py <recording>
python3 tools/analyze/run_gate_and_triage.py <recording>
```

### If `acc_issue`:
```bash
python3 tools/analyze/acc_pipeline_analysis.py --latest
```

**Then reproduce it WITHOUT Unity before reading the recording any further.**
`tests/acc_closedloop_harness.py` closes the loop radar → ACCController →
owner-resolver → LongitudinalController → safety clip → point-mass plant using the
production controllers built from the real merged YAML. A scenario runs in
< 1 s and the trace uses HDF5 field names, so it is comparable column-for-column
with the recording. It is calibrated against a known-good (H5 PASS) and a
known-bad (G2 TTC 1.58 vs sweep 1.54–1.67).

```bash
pytest tests/test_acc_closedloop.py -v                    # reproducers + mechanism tests
python3 tests/acc_closedloop_harness.py                   # print the G2 trace
```

For a new scenario: add a `run_<scenario>()` shortcut (lead profile, grade,
initial state from the recording's onset frame), confirm the harness reproduces
the sweep's TTC/e-stop within ~10 %, THEN run counterfactuals by passing
`acc=build_acc_controller(cfg, <param>=…)` / `longitudinal=build_longitudinal_controller(cfg, <param>=…)`.
A knob that removes the e-stop across the `PLANT_GRID` is a cause; one that
doesn't is a negative control — record both in `docs/agent/tasks.md`.

#### ACC blame disambiguation

Like the lateral table in Step 3.5: the symptom names WHAT, not WHERE to fix.

| Symptom | Possible meanings | How to disambiguate (harness counterfactual) |
|---|---|---|
| TTC < 2.0 s approaching a stopped/slow lead | (a) CUTOUT hand-off — ego < `acc.cutout_speed_mps`, owner-resolver hands the governor's free-flow target to the controller with a lead 5–10 m ahead | `cutout_throttle_frames() > 0`; `cutout_speed_mps=0` removes the e-stop → (a). **G2 mechanism, 2026-09-22.** Fixed by `acc.cutout_requires_no_lead: true` (kill-switch). If it recurs, check that flag first. |
| | (b) Jerk-cooldown pin — `accel_cmd_raw` sits at ≈ −0.43 while IDM demands −4…−10; `jerk_cooldown_scale` (0.4) applied every frame the measured-jerk cap re-arms it | `jerk_cooldown_frames=0` removes the e-stop → (b). Only visible where the bypass at pid_controller.py:5114 is NOT taken: flat ground (\|gravity\| < 0.1) and pre-emergency states. **Masked on grades.** Fixed by `control.longitudinal.acc_jerk_cooldown_bypass_states: [ACC_ACTIVE, CUTOUT]`. |
| | (c) Actuator latency (the 2026-04-20 explanation) | Vary `PointMassPlant(actuator_lag_frames=0…3)`. If the e-stop appears at lag 0 too, latency is not the cause. |
| | (d) Grade feed-forward propulsive bias (the night-27 explanation) | `grade_ff_gain=0.0` — if the e-stop persists, grade FF is not the cause. It did on G2. |
| COLLAPSED_GAP_STOP / "near-miss" at a reported gap of 2–5 m, or `radar_fwd_track_source == collision_override` | Radar frame — `radar_fwd_distance_m` is centre-to-centre (AVBridge.cs:3048); bumpers touch at a reported ~4.43 m. `s0`, EB floor, collapsed-stop and near-miss gates are all inside the lead body. **The scorer's `distance < 0` collision count is always 0; read `vehicle/lead_collision_detected` instead.** | Fixed 2026-09-22 by `acc.radar_range_offset_m: 4.43` (ForwardRadarSensor) + scorer counting `lead_collision_detected` as collisions. If it recurs: check the flag is non-zero and equals `scoring_registry.ACC_RADAR_RANGE_OFFSET_M`; harness `_legacy_cfg` reproduces the contact. |
| `acc_request_estop` / EMERGENCY_BRAKE frames inflated | (a) Real repeated near-misses | Check `acc_state_code` transitions, not frame counts |
| | (b) Standstill EB latch — EMA range-rate never reaches exactly 0, `> 0.0` test stays true with gap < 3 m; B1 bypass tags every entry `emergency_stop=True` (124 phantom e-stops on a clean stop) | Fixed 2026-09-22: `acc.emergency_brake_min_closing_mps: 0.1`, `acc.emergency_brake_abs_gap_m: 1.5` (< s0), and the scorer no longer counts EMERGENCY_BRAKE frames as e-stops. If it recurs, check those two keys. |
| Detection < 95 % | (a) Startup variance (lead at ~149 m when radar first locks) | Worst frames clustered in the first seconds → re-run before diagnosing. See `feedback_acc_stale_detection_artifact` |
| | (b) Heading-delta classifier rejecting a valid lead (`radar_fwd_reject_reason` = opposite_direction on arcs/grades) | Lives in `unity/…/AVBridge.cs:2721–3095`, not Python. See `project_acc_radar_heading_delta_cluster` |

**Do not tune `idm_comfortable_decel_mps2` for any of these** — the harness shows
2.5→4.0 changes nothing; IDM already demands −7 to −10 and the demand is not
being delivered (`project_acc_brake_authority_findings`).

Also note `orchestrator.py:9864` steps ACC with a hardcoded `dt = 1/30` while
frames arrive at ~1/13 s: IDM target-speed integration runs at 43 % of design
rate. Not the G2 cause, but it changes every ACC time constant.

Then suggest: `/trace brake_onset` and `/trace speed_drop` to inspect the signal chain.

### If `mpc_issue`:
```bash
python3 tools/analyze/mpc_pipeline_analysis.py --latest
```
Also check sections 24–26 from the primary analysis:
- **§24 Curvature Distribution:** shows whether the track has enough low-κ sections for LMPC, and whether the map is quantized (affecting regime threshold selection)
- **§25 Per-Regime Error Breakdown:** shows RMSE per controller (PP vs LMPC vs NMPC) and per segment (curve vs straight) — identifies which controller is responsible for tracking errors
- **§26 Model-Plant Comparison:** shows steering command vs actual, MPC reference alignment health, and heading prediction accuracy — detects model-plant mismatch

Run the physics-based regime boundary analysis to check if the regime selector config matches physics:
```bash
python3 tools/analyze/analyze_regime_boundaries.py <track_name> --decompose
```
This computes expected tracking error for PP, LMPC, and NMPC from first principles (chord error, model-plant mismatch, cost trade-offs) and shows the optimal controller at each (κ, v) operating point. If the suggested thresholds differ from config, the regime selector may be routing to the wrong controller.

Then suggest: `/trace regime_transition` to inspect PP↔MPC switch conditions.

### If `lateral_error_issue`:
First, check **§25 Per-Regime Error Breakdown** from the primary analysis. If a specific regime (PP, LMPC, NMPC) dominates the error, focus on that controller. If errors appear during transitions, check blend_frames and min_hold_frames config.

Then check if the regime selector is routing to the right controller for this track's operating points:
```bash
python3 tools/analyze/analyze_regime_boundaries.py <track_name> --decompose
```
Compare the "Winner" column against what the regime selector actually chose (from §25). If physics says NMPC but config routes to LMPC or PP, the regime config needs updating.

Then run the signal chain blame trace to identify the lookahead bottleneck:
```bash
python3 tools/analyze/trace_curve_entry.py <recording>
```
Use the BLAME output to determine which mechanism in the lookahead chain is the constraint before running deeper diagnostics.

```bash
python3 tools/analyze/build_failure_packet.py <recording>
python3 tools/analyze/analyze_phase_to_failure.py <recording>
python3 tools/analyze/counterfactual_layer_swap.py <recording>
```

### If `comfort_issue` (and NOT lateral_error_issue):
```bash
python3 tools/analyze/analyze_oscillation_root_cause.py <recording>
```
Then suggest: `/trace speed_drop` or `/trace brake_onset`.

### If `perception_issue`:
```bash
python3 tools/analyze/analyze_perception_questions.py <recording>
```

### If `trajectory_issue` or `control_issue`:
```bash
python3 tools/analyze/analyze_phase_to_failure.py <recording>
python3 tools/analyze/counterfactual_layer_swap.py <recording>
```

### If `unknown`:
```bash
python3 tools/analyze/run_gate_and_triage.py <recording>
python3 tools/analyze/counterfactual_layer_swap.py <recording>
```
And suggest opening PhilViz: `python3 tools/debug_visualizer/server.py` → Triage + Blame tabs.

## Step 5 — Industry Context Check

Before proposing a fix, briefly state how top AV companies handle this class of problem:

```
INDUSTRY CONTEXT:
  Problem class: <e.g., steering jerk on tight curves, late curve turn-in, oscillation>
  Standard approach: <what Waymo/Aurora/Cruise/Comma do — e.g., unified output
    rate/jerk limiter, gain-scheduled MPC, curvature-proportional feedforward>
  Why it applies: <1 sentence on why the standard approach fits our system>
  Anti-pattern to avoid: <common ad-hoc fix that top companies don't use —
    e.g., per-track overlays, binary gates on continuous signals, post-hoc clamps
    without controller awareness>
```

This check prevents reinventing solutions that have known-good industry patterns, and flags ad-hoc fixes that will create technical debt.

## Step 6 — Fix Level Triage

Before recommending any fix, classify it and check for robustness:

```
DESIGN SMELL CHECK (mandatory — check BEFORE assigning fix level):
  □ Binary gate on continuous signal?     → proportional weight
  □ Proxy stacking (≥2 params ≈ 1 qty)?  → unify into physics formula
  □ Frame-rate dependent formula (dt²)?   → convert to physical units
  □ Static lookup table for physics qty?  → replace with first-principles formula
  □ Post-hoc clamp creating discontinuity? → make planner aware of constraint
  □ Wrapper forwarding gap?                → verify config reaches inner controller via recording

  Smells detected: <count>
  → If ≥1: minimum ARCHITECTURE level. Run /plan-feature.

FIX LEVEL TRIAGE:
  Level: TUNING / CONFIG / CODE PATCH / ARCHITECTURE
    - TUNING: adjusting existing numeric params within their intended range
    - CONFIG: adding/changing config knobs (new params, new overlay keys)
    - CODE PATCH: changing priority/ordering/logic within existing architecture
    - ARCHITECTURE: changing how components interact, what signals drive decisions
  Proxy stacking: yes/no
    (Are multiple params approximating the same physical quantity?
     e.g., speed threshold + curvature guard + per-track overrides all
     approximating "can MPC handle this curve?")
  Overlay workarounds: <count> tracks with overlays working around this subsystem
    (If ≥2 overlays patch the same subsystem → architecture review needed)
  Cross-track: would this fix eliminate per-track workarounds? yes/no
  If ARCHITECTURE: describe what the unified signal/abstraction should be
```

**Rules:**
- NEVER recommend per-track overlay changes as the primary fix for systemic issues
- If ≥2 track overlays work around the same subsystem, flag it as an architecture smell
- Always prefer the fix level that eliminates the most per-track workarounds
- If ANY design smell detected, minimum level is ARCHITECTURE — do not propose tuning or code patch
- Label any per-track suggestion as "BAND-AID — not recommended" if a robust alternative exists

## Step 7 — Synthesize and output

If the primary issue is lateral error or late turn-in, include the PP Lookahead Signal Chain diagram from docs/agent/architecture.md in your diagnostic output to show which mechanism is the bottleneck.

Present a structured summary:

```
DIAGNOSTIC SUMMARY
==================
Recording: <file>
Overall Score: <score>/100

PRIMARY ISSUE: <category>
  Symptom: <what the metric shows>
  Root cause hint: <what the tool output suggests>
  Pareto rank: <if this deduction appears in /pareto, show rank and cross-track Σ pts>

FIX LEVEL TRIAGE:
  Level: <level>
  Proxy stacking: <yes/no — detail if yes>
  Overlay workarounds: <count> (<list subsystem>)
  Cross-track robust: <yes/no>

RECOMMENDED NEXT STEPS:
  1. <specific action — e.g. "/trace brake_onset to see reference_velocity at brake frames">
  2. <second action>
  3. <third action if needed>

A/B TEST NEEDED: yes/no (yes if a config change is being considered)
  If yes, specify regression tracks: /e2e <track1>, /e2e <track2>, ...
```

Gate thresholds: `tools/scoring_registry.py`
Tool decision tree: `.claude/docs/tool_selection_guide.md`
Field reference: `.claude/docs/hdf5_field_reference.md`
