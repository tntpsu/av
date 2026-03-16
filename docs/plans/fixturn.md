# Fix Turn Entry / Exit Hunting Plan

**Status:** Proposed implementation plan for the low-speed PP turn-hunting issue seen on short-straight, map-backed tracks.

## 1) Problem Statement

Latest evidence from `data/recordings/recording_20260306_191155.h5`:

1. No safety failure occurred, but visible turn-entry/exit hunting is present on the main turns.
2. The run is **PP-only**, not MPC:
   - `mpc_frames = 0`
   - `pp_frames = 1142`
3. The stack stays effectively in curve mode for almost the whole lap:
   - `curve_phase_preview_upcoming = 100%`
   - `curve_intent_state ~= COMMIT` for `1132 / 1142` frames
   - `curve_intent_commit_streak_max_frames = 592`
4. PP lookahead shrinks aggressively at real curve entry:
   - turn 1: `~6.6 m -> 3.0 m`
   - turn 3: `~6.45 m -> 3.0 m`
5. Peak lateral error happens at entry, not because of speed infeasibility:
   - turn 1 peak `|lat_error| ~= 0.78 m`
   - turn 3 peak `|lat_error| ~= 0.63 m`
   - `overspeed_into_curve_rate = 0%`
   - `turn_infeasible_rate_when_curve_cap_active = 0%`

Interpretation:

1. The core issue is **not** missing map preview and **not** insufficient speed reduction.
2. The core issue is that a **far-preview signal** is being used too directly to drive **local PP lookahead shrink / local curve state**.
3. On short-straight loops, “another curve exists ahead” can legitimately be true for most of the lap. That is acceptable for preview and speed planning, but it is not acceptable as the sole trigger for collapsing the local lateral tracking horizon.

## 2) Design Goal

Separate the stack into two distinct horizons:

1. **Far preview / corridor awareness**
   - Purpose: map-backed knowledge that a curve is ahead somewhere soon.
   - Allowed to stay active across short straights.
   - Used for speed preview, map diagnostics, and anticipation telemetry.
2. **Local steering entry phase**
   - Purpose: determine when PP should start shortening local lookahead and behaving like it is entering the current curve.
   - Must only become active when the curve is locally relevant.
   - Must not remain effectively latched because a later curve is visible far ahead.

Target behavior:

1. It is acceptable for a far-preview signal to stay high around the whole `s_loop`.
2. It is **not** acceptable for PP local lookahead to collapse to the hard minimum on every near-straight segment.
3. The local PP horizon should stay farther out through entry and contract smoothly.

## 3) Constraints

1. Do not regress the promoted hybrid architecture:
   - PP remains the low-speed controller.
   - MPC remains the highway/high-speed controller.
2. Do not retune speed governor first. Current data does not show a speed-feasibility failure.
3. Keep behavior changes narrow:
   - Primary scope: PP local steering phase and local lookahead contraction.
   - Secondary scope: telemetry/analyzer clarity.
4. Highway must be used as non-regression coverage because trajectory/orchestrator code is shared.

## 4) Implementation Plan

### Phase A — Split Far Preview From Local Steering Phase

**Primary files**

1. `trajectory/utils.py`
2. `av_stack/orchestrator.py`
3. `control/pid_controller.py`

**Current code path**

1. `av_stack/orchestrator.py` computes `curve_phase_diag = compute_curve_phase_scheduler(...)`.
2. `curve_intent` and `curve_intent_state` are derived directly from that scheduler.
3. `compute_reference_lookahead(...)` uses the resulting phase to blend into the entry lookahead table.
4. `control/pid_controller.py` consumes the resulting reference point and PP lookahead behavior.

**Change**

Introduce two explicitly separate signals:

1. `curve_preview_far_upcoming`
   - Derived from long-horizon map / preview curvature.
   - This is the “there is another curve ahead” signal.
   - It is allowed to stay high across short straights.
2. `curve_local_phase` (0..1) + `curve_local_state` (`STRAIGHT/ENTRY/COMMIT/REARM`)
   - Derived from local relevance only.
   - This is the signal that is allowed to shorten PP lookahead.

**Concrete code changes**

1. In `trajectory/utils.py`:
   - Add a helper such as `_compute_local_curve_relevance_weight(...)`.
   - Inputs should include:
     - `distance_to_curve_start_m`
     - `time_to_curve_s`
     - `path_curvature_abs`
     - optional `curve_at_car` / in-curve flag when available
   - Modify `compute_curve_phase_scheduler(...)` so the preview term used for the **local** scheduler is distance/time gated rather than raw-preview-only.
   - Keep raw preview evidence available in diagnostics so far-preview observability is not lost.
2. In `av_stack/orchestrator.py`:
   - Extend the scheduler call to pass `distance_to_curve_start_m`.
   - Carry both:
     - `curve_preview_far_upcoming`
     - `curve_local_phase`
     - `curve_local_state`
   - Keep the current map-backed far preview fields as telemetry.
   - Use `curve_local_phase` for local lookahead blending instead of letting always-on far preview dominate.
3. In `control/pid_controller.py`:
   - Treat far preview as informational / anticipatory.
   - Use local state for PP-local curve behavior.
   - Do not let a permanently high far-preview flag imply “local COMMIT”.

**Key rule**

If a future curve is visible but the vehicle is still on a short straight, the stack may keep:

1. `curve_preview_far_upcoming = true`

but should not force:

1. `curve_local_state = COMMIT`
2. PP local lookahead contraction to the floor

### Phase B — Add a PP-Specific Local Lookahead Floor and Gentler Contraction

**Primary files**

1. `control/pid_controller.py`
2. `config/av_stack_config.yaml`
3. `config/mpc_sloop.yaml`
4. `config/mpc_highway.yaml`

**Reason**

Even after the semantic split, PP can still get into trouble if the local horizon collapses too hard at entry. The latest run shows `~6.5 m -> 3.0 m` on entry. That is too aggressive for the observed low-speed PP behavior.

**Change**

Add a PP-only local lookahead floor and a dedicated shorten-slew for local curve entry.

**Concrete implementation**

1. Add PP-only config keys in the lateral section:
   - `pp_curve_local_lookahead_floor_enabled`
   - `pp_curve_local_lookahead_floor_speed_table`
   - `pp_curve_local_shorten_slew_m_per_frame`
   - `pp_curve_local_floor_state_min`
2. In `control/pid_controller.py` PP path:
   - When `control_mode == "pure_pursuit"` and `curve_local_state in {"ENTRY", "COMMIT"}`:
     - compute an effective PP lookahead floor from speed
     - clamp the local PP lookahead distance to that floor
     - limit how quickly PP lookahead can shorten
3. Keep this PP-specific:
   - do not apply the PP floor logic to MPC
   - do not change the speed governor’s long-horizon map preview behavior in the same patch

**Initial tuning target**

Use an initial PP local floor around:

1. `~4.5-5.0 m` at `~6.0-6.5 m/s`

This is a starting point for the first validation pass, not a final promoted number.

### Phase C — Clarify Telemetry, Recorder Contracts, and Analyzer Semantics

**Primary files**

1. `data/formats/data_format.py`
2. `data/recorder.py`
3. `tools/drive_summary_core.py`
4. `tools/analyze/analyze_drive_overall.py`
5. `tools/debug_visualizer/backend/summary_analyzer.py`
6. `tools/debug_visualizer/server.py`
7. `tools/debug_visualizer/visualizer.js`
8. `tests/test_control_command_hdf5.py`
9. `tests/test_drive_summary_parity.py`

**Reason**

Right now the telemetry story blurs far preview and local steering phase, which makes the diagnosis ambiguous.

**New telemetry to add**

1. `control/curve_preview_far_upcoming`
2. `control/curve_local_phase`
3. `control/curve_local_phase_raw`
4. `control/curve_local_state`
5. `control/curve_local_phase_source`
6. `control/curve_local_distance_ready`
7. `control/curve_local_distance_horizon_m`
8. `control/curve_local_rearm_cooldown_active`
9. `control/curve_local_force_straight_active`
10. `control/curve_local_commit_streak_frames`
11. `control/pp_curve_local_floor_active`
12. `control/pp_curve_local_floor_m`
13. `control/pp_curve_local_lookahead_pre_floor`
14. `control/pp_curve_local_lookahead_post_floor`
15. `control/pp_curve_local_shorten_slew_active`
16. `control/pp_curve_local_shorten_delta_m`
17. `control/reference_lookahead_local_gate_weight`

**Backward-compatibility rule**

1. Keep legacy `curve_intent_*` / `curve_phase_*` fields in recordings for now.
2. Add the new `curve_local_*` names as the canonical interface.
3. In `tools/drive_summary_core.py`, prefer `curve_local_*` if present and fall back to legacy fields when loading older recordings.
4. In `tools/debug_visualizer/backend/summary_analyzer.py`, keep adapter parity by importing the canonical summary from `tools.drive_summary_core` only.

**Concrete code changes**

1. In `data/formats/data_format.py`:
   - extend `ControlCommand` with the new `curve_local_*` and `pp_curve_local_*` fields
   - keep types aligned with existing control telemetry patterns (`Optional[float]`, `Optional[bool]`, `Optional[str]`)
2. In `data/recorder.py`:
   - extend `_create_datasets()` with the new `control/...` datasets
   - extend `_write_control_commands()` to persist those fields
   - use the existing string-dataset conventions already used for `curve_phase_state`, `curve_intent_state`, and `path_curvature_source_used`
3. In `tests/test_control_command_hdf5.py`:
   - extend the current curve-transition contract test to assert the new datasets are written
   - add one test that writes only legacy `curve_intent_*` data and confirms older recordings still load
4. In `tools/drive_summary_core.py`:
   - add a small loader helper that resolves canonical fields from recording data:
     - first try `curve_local_*`
     - then fall back to legacy `curve_intent_*`
     - then fall back to `curve_phase_*` if needed
   - compute summary metrics from the canonical loaded series, not from ad hoc field lookups
5. In `tools/analyze/analyze_drive_overall.py`:
   - print the new local-vs-far metrics in the trajectory/path-tracking summary block
   - print limits next to each new metric so the CLI output is self-explanatory
6. In `tests/test_drive_summary_parity.py`:
   - extend the synthetic recording writer to include at least one of the new fields
   - keep the parity assertion unchanged so CLI, core summary, and PhilViz adapter remain exact matches

**Analyzer metrics to add**

1. `curve_preview_far_active_straight_rate`
2. `curve_local_active_straight_rate`
3. `curve_local_commit_streak_max_frames`
4. `pp_entry_lookahead_min_m`
5. `pp_entry_lookahead_shorten_rate_min_mps`
6. `pp_turn_entry_peak_lateral_error` by curve event when track windows are available

**PhilViz additions**

Add one compact diagnostics row/card showing:

1. Far preview active?
2. Local steering phase/state?
3. Local lookahead floor active?
4. PP lookahead before floor vs after floor

This will let you answer:

1. “Are we previewing curves all the time?”
2. “Is local steering phase also active all the time?”
3. “Did the PP local floor save us from collapsing to 3 m?”

### Phase D — Add Invariants and Semantic Failure Classification

**Primary files**

1. `tools/drive_summary_core.py`
2. `tools/debug_visualizer/backend/issue_detector.py`
3. `tools/debug_visualizer/backend/diagnostics.py`
4. `tests/test_drive_summary_contract.py`
5. `tests/test_drive_summary_parity.py`

**Reason**

Without explicit invariants, the plan still relies too much on visual interpretation. The analyzer should be able to say exactly which semantic contract broke first.

**Invariants to implement**

1. **Far-preview-only is allowed**
   - `curve_preview_far_upcoming = true` with `curve_local_state = STRAIGHT` is valid and should not be flagged by itself.
2. **Local COMMIT must be locally relevant**
   - if `curve_local_state in {ENTRY, COMMIT}` and `curve_local_distance_ready = false`, count a semantic violation
3. **Local COMMIT cannot stay latched through straight segments**
   - if local state remains `COMMIT` while path curvature stays below the configured exit threshold for more than `N` frames, count a latch violation
4. **PP floor is a hard contract when active**
   - if `pp_curve_local_floor_active = true`, then `pp_curve_local_lookahead_post_floor >= pp_curve_local_floor_m - epsilon`
5. **Lookahead collapse must be locally explained**
   - if `reference_lookahead_active` shortens faster than the configured PP shorten slew while `curve_local_distance_ready = false`, count a lookahead-collapse violation

**Concrete code changes**

1. In `tools/drive_summary_core.py`:
   - add a helper such as `_build_curve_local_contract_summary(data, config, n_frames)`
   - compute:
     - `curve_local_commit_without_distance_ready_count`
     - `curve_local_commit_without_distance_ready_rate`
     - `curve_local_latched_straight_count`
     - `curve_local_latched_straight_rate`
     - `pp_curve_local_floor_breach_count`
     - `pp_curve_lookahead_collapse_violation_count`
     - `curve_local_contract_available`
   - return a new top-level summary block:
     - `curve_local_contract`
2. In `tools/debug_visualizer/backend/issue_detector.py`:
   - map those contract violations into explicit issue records:
     - `trajectory_curve_local_commit_too_early`
     - `trajectory_curve_local_latched`
     - `trajectory_curve_lookahead_collapse`
     - `trajectory_curve_local_floor_breach`
   - use the earliest violating frame as the issue frame
   - include concise evidence fields in the issue payload:
     - local state
     - distance-ready flag
     - lookahead pre/post values
     - path curvature
3. In `tools/debug_visualizer/backend/diagnostics.py`:
   - surface these issues in the trajectory diagnosis text so “root cause” points to a semantic violation instead of generic “trajectory issue”
4. In `tests/test_drive_summary_contract.py`:
   - assert that the new `curve_local_contract` block exists
   - assert old recordings without new fields return `curve_local_contract_available = false` instead of crashing
5. In `tests/test_drive_summary_parity.py`:
   - ensure the new summary block is preserved identically across:
     - `tools.drive_summary_core`
     - `tools/analyze/analyze_drive_overall.py`
     - PhilViz adapter

### Phase E — Add Turn-Event Segmentation for Per-Turn Triage

**Primary files**

1. `tools/drive_summary_core.py`
2. `tools/analyze/analyze_drive_overall.py`
3. `tools/debug_visualizer/server.py`
4. `tools/debug_visualizer/visualizer.js`
5. `tests/test_drive_summary_contract.py`

**Reason**

“Turn 1 looked bad” is not sufficient. We need event-level metrics so the analyzer can distinguish entry overshoot, apex instability, and exit hunting.

**Implementation approach**

1. Reuse the existing track-window logic currently implemented in `tools/debug_visualizer/server.py::_load_track_curve_windows`.
2. Move that logic into a shared helper module that both the server and `tools/drive_summary_core.py` can import.
3. Use recording provenance `track_id` first. If that is missing, allow analyzer callers to pass an explicit track path as an override in a follow-up patch if needed.

**Turn-event builder**

In `tools/drive_summary_core.py`, add a helper such as `_build_curve_turn_events(data, curve_windows, n_frames)` that emits one record per curve:

1. `curve_index`
2. `entry_start_frame`
3. `entry_end_frame`
4. `apex_frame`
5. `exit_end_frame`
6. `peak_lateral_error_frame`
7. `peak_lateral_error_m`
8. `peak_heading_error_rad`
9. `entry_speed_mps`
10. `entry_lookahead_pre_floor_min_m`
11. `entry_lookahead_post_floor_min_m`
12. `entry_lookahead_shorten_rate_min_mps`
13. `lateral_zero_crossings_entry`
14. `local_state_active_ratio`
15. `local_commit_without_distance_ready_count`

**Summary exposure**

1. Add a new summary block:
   - `curve_turn_events`
2. Add rollups:
   - `worst_turn_entry_peak_lateral_error_m`
   - `worst_turn_entry_lookahead_min_m`
   - `worst_turn_local_contract_violation_count`
3. In `tools/analyze/analyze_drive_overall.py`, print the worst two turn events when available.

### Phase F — Surface the New Triage Story in PhilViz

**Primary files**

1. `tools/debug_visualizer/server.py`
2. `tools/debug_visualizer/visualizer.js`
3. `tools/debug_visualizer/backend/issue_detector.py`

**Reason**

The summary tab should immediately answer:

1. Are we merely previewing a future curve?
2. Did local steering arm too early or stay latched?
3. Did PP lookahead collapse too hard?
4. Which exact turn event was worst?

**Concrete UI/API work**

1. In `tools/debug_visualizer/server.py`:
   - include the new per-frame fields in the frame JSON payload under the `control` block
   - expose the new summary blocks returned by `tools.drive_summary_core`
   - expose turn-event arrays without extra re-derivation in the server
2. In `tools/debug_visualizer/visualizer.js`:
   - add a summary card named something like `Curve Local-State Health`
   - show:
     - `curve_preview_far_active_straight_rate`
     - `curve_local_active_straight_rate`
     - `curve_local_commit_streak_max_frames`
     - `pp_entry_lookahead_min_m`
     - `pp_entry_lookahead_shorten_rate_min_mps`
   - show limits next to each number in the same style as the newer latency/sync cards
   - make the failing metric value red, not just the label
3. Add a per-turn table or compact list in PhilViz summary/diagnostics:
   - one row per curve event
   - sort by worst `peak_lateral_error_m`
   - click row to jump to `peak_lateral_error_frame`
4. Add timeline/diagnostic fields in the frame inspector:
   - `curve_preview_far_upcoming`
   - `curve_local_state`
   - `curve_local_distance_ready`
   - `pp_curve_local_floor_active`
   - `pp_curve_local_lookahead_pre_floor`
   - `pp_curve_local_lookahead_post_floor`
5. Add issues-tab filters for the new issue ids created in Phase D.

### Phase G — Add Replay-Based A/B Triage Before Live Promotion

**Primary files**

1. `tools/reprocess_recording.py`
2. `tools/drive_summary_core.py`
3. `tools/analyze/analyze_drive_overall.py`
4. `tests/test_drive_summary_parity.py`

**Reason**

We need a deterministic way to compare old vs new local-phase behavior on the same input stream before trusting live Unity runs.

**Implementation approach**

1. Extend `tools/reprocess_recording.py` so it can be used as a repeatable offline comparison tool for this fix:
   - accept an explicit config path
   - preserve source recording provenance in the new output metadata
   - label outputs as baseline vs candidate
2. Run baseline and candidate replay on:
   - `data/recordings/recording_20260306_191155.h5`
3. Analyze both outputs with `tools/analyze/analyze_drive_overall.py`
4. Save an A/B report bundle with:
   - summary JSON for both runs
   - `curve_local_contract` block for both runs
   - `curve_turn_events` for both runs
   - a small markdown decision note listing what improved and what regressed

**Why this matters**

1. It removes Unity cadence/physics noise from the first pass.
2. It lets us verify semantic improvement:
   - far preview can stay high
   - local state should de-latch on straights
   - PP lookahead should not collapse prematurely

## 5) Test Plan

### Unit tests — trajectory / scheduler

**File:** `tests/test_reference_lookahead.py`

Add tests for:

1. Far preview can remain high while local curve phase stays low when distance/time to the active curve is not yet local.
2. Local curve phase arms as distance/time enters the configured local window.
3. Local curve phase can rearm for the next curve without forcing full straight-only semantics on very short straights.
4. `compute_reference_lookahead(...)` uses local phase for entry blending rather than raw far-preview-only evidence.
5. Local entry blending does not collapse lookahead to the hard minimum when the PP floor is enabled.
6. Lookahead contraction obeys the new PP shorten-slew limit.
7. Local-phase contract terms expose enough telemetry to explain why the phase did or did not arm.

### Unit tests — controller behavior

**File:** `tests/test_control.py`

Add tests for:

1. `curve_preview_far_upcoming` may stay true while `curve_local_state` is `STRAIGHT` or low-phase on a short straight.
2. PP local floor activates in `ENTRY/COMMIT` and prevents `pp_lookahead_distance` from dropping below configured floor.
3. Single-owner mode no longer forces effectively permanent local `COMMIT` on a short-straight loop input sequence.
4. Local lookahead contraction is smooth across a synthetic entry sequence.
5. PP floor logic is inactive when `control_mode != pure_pursuit`.
6. Local COMMIT is rejected when distance-ready is false.
7. Local latch watchdog / force-straight behavior is reflected in the new `curve_local_*` fields.

### Recorder / telemetry contract tests

**Files**

1. `tests/test_control_command_hdf5.py`
2. `tests/test_drive_summary_contract.py`
3. `tests/test_drive_summary_parity.py`

Add tests for:

1. New HDF5 fields are recorded correctly.
2. Analyzer contract includes the new local/far metrics when available.
3. Old recordings without the new fields still load without crash.
4. Canonical summary parity holds across CLI and PhilViz adapters once the new blocks are added.
5. `curve_local_contract` summary block contains explicit availability and limit fields.

### Analyzer / issue-detector tests

**Files**

1. `tests/test_drive_summary_contract.py`
2. `tests/test_drive_summary_parity.py`
3. new targeted test file for issue detector semantic violations if needed

Add tests for:

1. A synthetic recording with far preview high but local state straight does **not** trigger an issue.
2. A synthetic recording with `curve_local_state = COMMIT` while `curve_local_distance_ready = false` does trigger `trajectory_curve_local_commit_too_early`.
3. A synthetic recording with rapid lookahead collapse triggers `trajectory_curve_lookahead_collapse`.
4. Turn-event segmentation produces deterministic event frames and entry metrics on a fixed synthetic sequence.

### Optional analyzer parity tests

**Files**

1. `tests/test_drive_summary_latency_sync.py`
2. new targeted tests beside `tools/drive_summary_core.py` metrics if needed

Add tests for:

1. Local/far preview rates are computed deterministically from fixed synthetic inputs.
2. Turn-entry lookahead minima are computed correctly when curve windows are present.
3. Turn-event rollups and `curve_local_contract` limits serialize identically through all summary adapters.

## 6) Live Validation Plan

### Stage 0 — Replay A/B before Unity

1. Reprocess `data/recordings/recording_20260306_191155.h5` with baseline and candidate configs.
2. Confirm:
   - `curve_preview_far_active_straight_rate` may stay high
   - `curve_local_active_straight_rate` drops materially
   - `curve_local_commit_without_distance_ready_count = 0`
   - `pp_curve_lookahead_collapse_violation_count = 0`
3. Do not promote to live validation if the replay A/B still shows local latch or unexplained lookahead collapse.

### Stage 1 — Reproduce and validate on PP low-speed track first

Use the current promoted hybrid config for the low-speed track:

1. `./start_av_stack.sh --force --build-unity-player --skip-unity-build-if-clean --config config/mpc_sloop.yaml --track-yaml tracks/s_loop.yml --duration 60`

Run:

1. Baseline `3x` with current code/config
2. Candidate `3x` with the local-phase + PP-floor changes

Analyze each with:

1. `python tools/analyze/analyze_drive_overall.py <recording> --analyze-to-failure`

Record:

1. overall score
2. lateral RMSE / P95 / max
3. out-of-lane events
4. oscillation runaway
5. `curve_preview_far_active_straight_rate`
6. `curve_local_active_straight_rate`
7. `curve_local_commit_streak_max_frames`
8. `curve_local_commit_without_distance_ready_count`
9. `pp_curve_lookahead_collapse_violation_count`
10. `pp_entry_lookahead_min_m`
11. `pp_entry_lookahead_shorten_rate_min_mps`
12. turn-1 and turn-3 entry peak lateral error
13. `tuning_valid`

Hard gate:

1. Use runs for tuning decisions only when `tuning_valid = true`.

### Stage 2 — Highway non-regression

Use the promoted highway config:

1. `./start_av_stack.sh --force --build-unity-player --skip-unity-build-if-clean --config config/mpc_highway.yaml --track-yaml tracks/highway_65.yml --duration 120`

Run:

1. Baseline `2x`
2. Candidate `2x`

Required checks:

1. `mpc_frames` remains nonzero and dominant on highway
2. no e-stop
3. no out-of-lane
4. no material regression in highway lateral RMSE / score
5. PP-local curve telemetry may be present, but it must not materially change highway MPC dominance

### Stage 3 — Compare against the latest reproduce case

Replay-check against:

1. `data/recordings/recording_20260306_191155.h5`

Goal:

1. Validate that the new analyzer metrics clearly explain the prior issue:
   - far preview high
   - local steering phase should have been lower / more local
   - PP local floor would have prevented collapse to `3.0 m`
2. Validate that PhilViz shows the same story on the same recording without needing manual frame-by-frame interpretation

## 7) Acceptance Criteria

### Functional acceptance

1. PP low-speed turn-entry hunting is reduced on `s_loop`-style runs.
2. No out-of-lane and no e-stop.
3. No oscillation runaway.
4. Turn-entry peak lateral error improves materially versus the current reproduce case.
5. No new severe slowdowns or comfort regressions are introduced on the low-speed validation runs.

### Semantic / observability acceptance

1. Far preview and local steering phase are independently visible in telemetry.
2. Analyzer can distinguish:
   - “curve is visible ahead”
   - “local steering should already be in curve mode”
3. PhilViz makes this distinction legible in one place.
4. `curve_local_contract` exists in summary output with explicit limit values.
5. Issue detector can emit at least:
   - early local commit
   - local latch on straight
   - unexplained lookahead collapse
   - floor breach

### Invariant acceptance

1. `curve_local_commit_without_distance_ready_count = 0` on promoted runs.
2. `pp_curve_local_floor_breach_count = 0` on promoted runs.
3. `pp_curve_lookahead_collapse_violation_count = 0` on promoted runs.
4. `curve_local_active_straight_rate` is materially below the current reproduce case and no longer explains the lap as “always in curve mode.”

### Non-regression acceptance

1. Highway hybrid run still behaves as highway hybrid:
   - MPC remains active on highway
   - no safety regression
2. Existing curve-intent / reference-lookahead / HDF5 tests stay green.
3. Summary parity tests stay green.

## 8) Rollback / Guardrails

1. Keep all new behavior behind config flags.
2. Land the semantic split before promoting any new PP floor values.
3. Land telemetry + analyzer contract changes even if behavior changes are held back; observability work should not be coupled to promotion.
4. If highway regresses, disable only the PP local-floor and local-phase entry behavior while keeping telemetry additions.
5. If new canonical `curve_local_*` fields are absent, analyzer and PhilViz must fall back to legacy `curve_intent_*` fields.
6. Do not change speed governor curve-cap tuning in the same patch set.

## 9) Recommended Execution Order

1. Add canonical `curve_local_*` telemetry fields and recorder support.
2. Add local/far signal split in code.
3. Add unit tests for the split.
4. Add `curve_local_contract` summary block and parity tests.
5. Add issue-detector semantic violations.
6. Add turn-event segmentation and CLI reporting.
7. Add PP local floor + shorten-slew logic.
8. Add PhilViz summary/timeline surfacing.
9. Run replay A/B on `recording_20260306_191155.h5`.
10. Run `3x` s-loop validation using only `tuning_valid=true` runs for tuning judgment.
11. Run `2x` highway non-regression.
12. Promote only if the low-speed improvement is real, the semantic contracts are clean, and highway remains clean.
