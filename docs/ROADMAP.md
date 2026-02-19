# Robust Full-Stack Roadmap (Unified, Layered, and Gated)

**Last Updated:** 2026-02-18  
**Current Focus:** Layer 2, Stage 1 (First-Turn Entry Reliability) authority-stack consistency under comfort-aware dynamic policy  
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

**Canonical validation policy (active)**
- Stage 1 promotion/tuning gates use canonical start only: `start_t=0.0`.
- Non-canonical starts are exploratory and are not used for Stage 1 promotion decisions unless explicitly requested.

**Current micro-steps (tracked and gated)**
- `S1-M1` (done): add bounded low-visibility fallback controls in perception path:
  - `perception.low_visibility_fallback_blend_alpha`
  - `perception.low_visibility_fallback_max_consecutive_frames`
  - `perception.low_visibility_fallback_center_shift_cap_m`
- `S1-M2` (done): run A/B gate for fallback TTL (`max_consecutive_frames=1000` vs `8`, 2 pairs, 45s, `s_loop`):
  - result: median first centerline cross improved from `185.5` -> `201.0` frames with bounded TTL.
  - interpretation: bounded hold helps delay first failure in this scenario.
- `S1-M3` (done, low-confidence): run focused A/B for blend alpha (`1.0` vs `0.35`, 1 pair, 45s):
  - result: stale percentage improved (`31.6%` -> `11.1%`) and authority gap reduced (`0.450` -> `0.418`) with `0.35`, but centerline-cross changed (`199` -> `191`) in opposite direction.
  - interpretation: blend needs more repeats before promotion decision.
- `S1-M4` (next): run 3-pair A/B for blend alpha and center-shift cap to de-noise variance and lock defaults.
- `S1-M5` (next): if `S1-M4` passes, add PhilViz row for fallback mode state (`active`, `ttl_exceeded`, `blend_alpha`, `center_shift_cap`) for direct attribution.
- `S1-M6` (done): add control attribution telemetry for pre-failure triage:
  - per-frame `control/steering_authority_gap`
  - per-frame `control/steering_transfer_ratio`
  - per-frame `control/steering_first_limiter_stage_code` (`0=none,1=rate,2=jerk,3=hard_clip,4=smoothing`)
  - curve-entry summary now includes `first_limiter_hit_frame` and `first_limiter_hit_stage`.
- `S1-M7` (done): run control A/B for curve-entry assist toggle (`curve_entry_assist_enabled=0` vs `1`, 2 pairs, 45s, `s_loop`):
  - result: median first centerline cross improved from `189.5` -> `201.5` frames with assist ON.
  - result: median time-to-failure improved from `10.058s` -> `10.350s`.
  - interpretation: assist ON helps delay first failure; root classification remains `steering-authority-limited`.
- `S1-M8` (done): run 3-pair bounded sweep for assist strength (`rate_boost`, `jerk_boost`) with watchdog/rearm unchanged:
  - `rate_boost`: `1.12` vs `1.25` -> `1.25` regressed (median cross `199` -> `189`, median TTF `10.116s` -> `9.883s`).
  - `jerk_boost`: `1.08` vs `1.20` -> mixed/inconclusive (`cross` improved `185` -> `192`, but TTF regressed `10.183s` -> `9.633s` and transfer ratio worsened).
  - interpretation: do not promote stronger boosts yet; keep conservative assist strengths.
- `S1-M9` (done): run 3-pair confirmation at conservative assist settings (`enabled=true`, `rate_boost=1.12`, `jerk_boost=1.08`) vs assist OFF:
  - centerline cross median improved: `174` -> `192` (assist ON better).
  - first-failure median improved: `174` -> `192` (assist ON better).
  - time-to-failure median regressed: `11.133s` -> `10.500s` (assist ON worse in this sample).
  - interpretation: directional gain on crossing exists, but promotion is blocked due to mixed TTF outcome; keep assist default OFF until one more confirmation pass.
- `S1-M10` (done): run final deterministic 3-pair confirmation at fixed canonical start (`start_t=0.0`) with explicit comfort fields in batch output:
  - centerline cross median: `174` -> `174` (no gain with assist ON).
  - first-failure median: `174` -> `174` (no gain with assist ON).
  - time-to-failure median: `11.117s` -> `11.417s` (slight gain with assist ON).
  - classification remained `steering-authority-limited` in all runs.
  - comfort fields are now emitted by batch harness but currently unavailable (`n/a`) from summary source for these runs.
  - decision: hold default as assist OFF until either crossing gains appear at fixed start or control architecture checkpoint is taken.
- `S1-M11` (done): implement and evaluate entry-phase limiter schedule + handoff (`curve_entry_schedule_frames: 0` vs `20`, fixed `start_t=0.0`, 3 pairs, 45s):
  - centerline cross median: `189` -> `193` (small gain), but one run regressed (`193` -> `187`).
  - first-failure median: `189` -> `193` (small gain), with the same regression run.
  - time-to-failure median: `10.133s` -> `9.751s` (regression).
  - classification remained `steering-authority-limited` in all runs.
  - decision: do not promote schedule as default; keep it OFF and take the architecture checkpoint for stronger control-policy shaping.
- `S1-M12` (done): architecture-checkpoint experiment with phase-structured commit mode (`curve_commit_mode_enabled: false` vs `true`, fixed `start_t=0.0`, 3 pairs, fail-fast emergency stop `0.5s`):
  - centerline cross median: `203` -> `205` (slight gain).
  - first-failure median: `203` -> `205` (slight gain).
  - time-to-failure median: `10.100s` -> `9.883s` (regression).
  - classification remained `steering-authority-limited` in all runs.
  - decision: do not promote commit mode as default; retain OFF and continue architecture checkpoint toward stronger phase scheduling + speed/trajectory co-shaping.
- `S1-M13` (done): add bounded speed/authority coupling (temporary curve-mode speed cap while commit mode is active) + fail-fast A/B validation (`curve_commit_mode_enabled: false` vs `true`, fixed `start_t=0.0`, 3 pairs, `emergency_stop_end_run_after_seconds=0.5`):
  - activation verified in treatment recordings (`curve_commit_mode_active` and `curve_mode_speed_cap_clamped` > 0 frames).
  - centerline cross median: `197` -> `208` (improved).
  - first-failure median: `197` -> `209` (improved).
  - time-to-failure median: `9.817s` -> `10.083s` (improved).
  - caveat: one treatment trial switched to `mixed-or-unclear` classification (no centerline cross before OOB frame).
  - decision: keep default OFF for now; run one more fixed-start confirmation pass before any promotion.
- `S1-M14` (done): fixed-start confirmation pass for S1-M13 under fail-fast settings (`curve_commit_mode_enabled: false` vs `true`, fixed `start_t=0.0`, 3 pairs, `emergency_stop_end_run_after_seconds=0.5`):
  - centerline cross median: `185` -> `206` (improved).
  - first-failure median: `185` -> `206` (improved).
  - time-to-failure median: `9.967s` -> `9.900s` (slight regression).
  - classification: treatment produced `2x steering-authority-limited`, `1x mixed-or-unclear`.
  - decision: hold defaults OFF; require one additional confirmation at a second fixed start before promotion.
- `S1-M15` (done): second fixed-start confirmation attempt at `start_t=0.20` (same fail-fast A/B protocol):
  - regime shifted away from steering-limited to `perception-limited` / `mixed-or-unclear`.
  - baseline mostly had no centerline-cross signal before failure; treatment mostly became perception-limited.
  - decision: this start is not a valid control-promotion gate for S1-M13.
- `S1-M16` (done): replacement second-start confirmation at `start_t=0.10`:
  - both arms classified primarily as `perception-limited` (with one `mixed-or-unclear` treatment run).
  - no centerline-cross metric available in either arm (`n/a`) for promotion criteria.
  - treatment had lower median first-failure frame/time than baseline and higher stale percentage.
  - decision: do not promote coupled commit+speed-cap policy yet; keep defaults OFF and return to upstream perception stabilization for non-canonical starts before reattempting multi-start control promotion.
- `S1-M17` (done): lock Stage 1 gating to canonical start only (`start_t=0.0`) per user decision:
  - canonical start is now the only source for Stage 1 promote/hold decisions.
  - non-canonical starts are explicitly exploratory unless user requests otherwise.
- `S1-M18` (done): rerun canonical gate after policy lock (`curve_commit_mode_enabled: false` vs `true`, fixed `start_t=0.0`, 3 pairs, fail-fast `0.5s`):
  - centerline cross median: `201` -> `197.5` (worse).
  - first-failure median: `201` -> `211` (better).
  - time-to-failure median: `10.117s` -> `10.999s` (better).
  - treatment classification shifted (`2x mixed-or-unclear`, `1x steering-authority-limited`) and stale percentage increased (`15.8%` -> `30.8%`).
  - decision: hold default OFF; not promotable due mixed signal and centerline-cross regression on canonical gate.
- `S1-M19` (done): define phase-based curve authority envelope + scoring contracts from existing telemetry:
  - added phase envelope config: `tools/analyze/curve_authority_envelope.yaml`.
  - `tools/analyze/analyze_phase_to_failure.py` now emits `curve_entry`, `curve_commit`, `curve_steady` authority metrics and pass/fail checks.
- `S1-M20` (done): diagnostics hardening for unambiguous failure triage packets:
  - added `tools/analyze/build_failure_packet.py` to emit `packet.json` + frame-window `window.csv`.
  - packet includes `first_limiter_hit_frame` and per-stage limiter deltas around failure.
- `S1-M21` (done): reusable canonical sweep harness:
  - added `tools/analyze/sweep_curve_gated_policies.py` (fixed `start_t=0.0`, fail-fast emergency stop override, ranked outputs).
  - outputs are written to `data/reports/sweeps/s_loop_curve_sweep_<timestamp>/`.
- `S1-M22` (done): first-pass + validation sweep execution on `s_loop`:
  - first-pass (1 repeat) briefly favored `entry_commit_aggressive` due one non-failure run.
  - validation (2 repeats) ranked `entry_schedule_soft` best with median first-failure `232.5` vs baseline `189.0`.
  - report artifacts: `data/reports/sweeps/s_loop_curve_sweep_20260218_003008/` and `data/reports/sweeps/s_loop_curve_sweep_20260218_003219/`.
- `S1-M23` (done): promote best validated candidate to baseline config:
  - promoted `entry_schedule_soft` parameters:
    - `curve_entry_schedule_enabled: true`
    - `curve_entry_schedule_frames: 24`
    - `curve_entry_schedule_min_rate: 0.22`
    - `curve_entry_schedule_min_jerk: 0.16`
    - `curve_entry_schedule_min_hold_frames: 8`
  - `curve_commit_mode_enabled` remains `false` pending further evidence.
- `S1-M24` (done): calibrate phase-envelope targets against canonical sweep telemetry and re-score:
  - updated envelope targets in `tools/analyze/curve_authority_envelope.yaml` to avoid all-zero pass states.
  - calibrated validation ranking remains `entry_schedule_soft` best.
- `S1-M25` (done): execute literal Phase 4 completion from the `s_loop` tuning plan:
  - Stage A trajectory-locked screen was run for baseline vs promoted candidate (`phase4_stageA_*` artifacts).
  - Stage A result was non-discriminative (no clear reject signal), so Stage B was used as decider.
  - Stage B canonical A/B (`start_t=0.0`, repeats=2) favored promoted policy by first-failure/time-to-failure medians.
- `S1-M26` (done): execute Phase 5 post-promotion robustness mini-set:
  - ran additional fixed-start A/B checks at `start_t=0.05` and `start_t=0.10`.
  - outcomes were perception-limited on non-canonical starts; canonical promotion remains valid and baseline unchanged.
  - consolidated artifacts under `data/reports/sweeps/s_loop_phase45_20260218/`.
- `S1-M27` (done): tighten crossing attribution + comfort-policy reshaping for dynamic authority:
  - centerline-cross detection now treats first intrusion as failure onset (`persist=1`, near-zero threshold guard) across analyzers and PhilViz issue detection.
  - dynamic comfort policy now uses `g_lat` as hard guardrail and `g_jerk` as soft damped penalty (with smoothing + floor) to avoid one-frame jerk spikes zeroing authority.
  - added telemetry for smoothed jerk and split comfort gates (`accel_gate`, `jerk_penalty`) in HDF5 + PhilViz.
- `S1-M28` (done): align downstream limiter stack with comfort-aware dynamic authority:
  - added comfort-aware dynamic hard-clip lift (`dynamic_curve_hard_clip_boost_*`) so clip ceiling can expand when `g_lat` headroom exists.
  - exposed hard-clip boost/cap/effective-limit telemetry in HDF5 + PhilViz.
  - added control test coverage for reduced hard-clip loss when dynamic clip lift is enabled.
- `S1-M29` (in progress): validate clip-lift behavior on canonical 25s sanity runs:
  - latest validation recording: `data/recordings/recording_20260218_233236.h5`.
  - first intrusion moved to frame `204` and no offroad emergency stop occurred in this run.
  - hard-clip lift engaged (`dynamic_curve_hard_clip_boost` nonzero) with effective clip ceiling rising above base cap.
- `S1-M30` (next): tune hard-clip lift gain/cap with canonical A/B so authority gain improves first-intrusion timing without comfort regressions.

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
