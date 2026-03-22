# Robust Full-Stack Roadmap (Unified, Layered, and Gated)

**Last Updated:** 2026-03-20
**Current Focus:** Layer 2, Stage 2 (Robustness & Speed Expansion) — oscillation attribution (S2-M6) + Step 3.5 (2DOF FF + curve-contamination clamp) implemented, pending live E2E validation
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
- ~~Phase 2: Curvature conditioning + comfort stabilization + trajectory integration~~ **(done via S1-M32–S1-M39)**
- ~~Post-Phase-2 checkpoint: architecture escalation~~ **(done — Pure Pursuit, S1-M33)**
- ~~Layer 2 Stage 1: First-Turn Entry Reliability~~ **(done — S1-M39, scores 95–96/100)**
- ~~Layer 2 Stage 1 tooling: Stack Isolation~~ **(done — S1-M40)**
- **Layer 2 Stage 2: Robustness, Infrastructure & Speed Expansion** ← **current**
  - S2-M1 ✓, S2-M2 ✓, S2-M3 ✓, S2-M4 → Phase 2 MPC (2.1-2.7b all ✓)
  - Phase 2 MPC: PP (<10 m/s) + LMPC (10-20 m/s) with curvature preview, highway median 98.1
- Phase 3+: behavior/actors/multi-lane/urban/robustness (**see Capability Roadmap below**)

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

### Stage 1: First-Turn Entry Reliability (Historical Complete Stage)
**Goal:** clear first turn without centerline crossing, out-of-bounds, or comfort breach.

This stage is complete and retained for traceability. The active program focus is the Stage 2 / `S2-M5` work called out at the top of this document.

**Canonical validation policy (active)**
- Stage 1 promotion/tuning gates use canonical start only: `start_t=0.0`.
- Non-canonical starts are exploratory and are not used for Stage 1 promotion decisions unless explicitly requested.

**Historical micro-steps (tracked and gated)**
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
- `S1-M29` (done): validate clip-lift behavior on canonical 25s sanity runs:
  - latest validation recording: `data/recordings/recording_20260218_233236.h5`.
  - first intrusion moved to frame `204` and no offroad emergency stop occurred in this run.
  - hard-clip lift engaged (`dynamic_curve_hard_clip_boost` nonzero) with effective clip ceiling rising above base cap.
- `S1-M30` (done): tune authority transfer under canonical A/B and lock trajectory-side cap gain:
  - executed canonical repeated gate package (`repeats=3`, fixed `start_t=0.0`) with dynamic horizon A/B.
  - result: treatment improved authority transfer and failure timing (`authority_gap_mean` ~`0.316 -> 0.006`, `transfer_ratio_mean` ~`0.462 -> 0.988`, `time_to_failure_s` ~`10.27 -> 24.45` median), with higher stale usage dominated by `left_lane_low_visibility`.
  - trajectory sweep (`far_band_contribution_cap_gain: 0.35 -> 0.45`) improved authority and reduced stale in single-pair canonical pass; promoted `0.45` into config for next gate cycle.
- `S1-M31` (done): add Stage-1 promotion evidence package + summary/compare triage upgrades:
  - baseline freeze and analyzer consistency report: `data/reports/baseline_freeze_report.json`.
  - mandatory E2E promotion suite report: `data/reports/e2e_promotion_suite_report.json` (explicitly separate from unit/integration verification).
  - PhilViz now includes layered summary scoring (hybrid-cap), waterfall section navigation, failure banner priority, compare tab MVP, and recording provenance surfacing.
- `S1-M32` (done): S-loop turn clearance — signal chain + speed + control:
  - **A1:** curvature guard prevents heading zeroing on curve approach (trajectory_planner.py).
  - **A2:** heading estimated from lane-center history when lane_coeffs unavailable.
  - **A3:** curvature-aware smoothing alpha reduction and rate-limit relaxation.
  - **A4:** curvature_preview at 1.5× lookahead for early feedforward priming.
  - **B1:** curvature-based speed cap enabled at 8.0 m/s default.
  - **B2:** curvature_preview triggers deceleration before curve entry (0.25g comfort cap).
  - **C1:** control curvature_smoothing_alpha reduced 0.7 → 0.5.
  - **C2:** curvature_preview blended into feedforward for 30% early priming.
  - **C3:** straight→curve integral decay (flush straight-line bias in 5 frames).
  - **V1–V5:** PhilViz auto-triage: signal chain waterfall, suppression attribution, speed-curvature overlay, compare tab auto-populate, Signal Integrity score layer.
  - **D1:** baseline signal chain diagnostic (`analyze_signal_chain.py`).
  - new tests in `tests/test_signal_chain_fixes.py` covering A1–A4.
  - validation status: D2/D3/D4 require live Unity runs for A/B confirmation.
- `S1-M33` (done): Pure Pursuit control — architecture escalation from PID to geometric steering:
  - **Rationale:** 8 rounds of PID tuning proved oscillation is architectural (error→overcorrect→overshoot cycle), not a tuning gap. PID baseline: F655, mean|gcx|=0.60m, 22% hard clip, 19 sign changes.
  - **Implementation:** `pure_pursuit` control mode in `LateralController` computes `steering = atan(2 * wheelbase * sin(alpha) / ld)` directly from reference point geometry. Bypasses PID-specific feedforward, curve entry scheduling, dynamic authority, commit mode, feedback gain scheduling, and deadband.
  - **Safety nets:** Reference point jump clamping (0.5m/frame), stale perception hold with 0.98/frame decay, small PID feedback (0.15 gain) for disturbance rejection.
  - **Telemetry:** pp_alpha, pp_lookahead_distance, pp_geometric_steering, pp_feedback_steering, pp_ref_jump_clamped, pp_stale_hold_active in HDF5 + PhilViz v63.
  - **E2E validation (2 runs, canonical start, s-loop 40s):** Initial runs showed improvement but PID-era post-processing pipeline (jerk limit, EMA smoothing) defeated PP steering — see S1-M35 for fix.
  - **Architecture escalation checkpoint addressed:** PP implementation IS the architecture escalation; PID oscillation was unfixable via tuning, geometric controller resolved it. Pipeline bypass (S1-M35) was required to deliver PP's output cleanly.
  - **Rollback:** `control_mode: pid` (one line config change); all PID machinery preserved.

- `S1-M34` (done): Speed control architecture — comfort governor consolidation:
  - **Problem:** 12 serial speed suppression layers compounding to limit speed (mean 4.35 m/s, target 12 m/s). Each layer independently "safe" but compounding effect was severe.
  - **Phase 0 bug fixes:** (a) Speed planner `max_accel`/`max_jerk` override bug (planner used longitudinal's limits instead of its own), (b) `launch_speed_floor` pathological acceleration (uncapped recalculated accel), (c) Unity GT lookahead override (fixed value replacing dynamic scaling).
  - **Architecture:** Extracted `control/speed_governor.py` module — standalone `SpeedGovernor` class with 4 clean concerns: (1) physics-based comfort governor `v = sqrt(max_lat_accel_g * 9.81 / |curvature|)`, (2) curve preview anticipatory deceleration, (3) perception horizon guardrail (decoupled from control lookahead), (4) jerk-limited speed planner.
  - **Removed:** 8 redundant layers (curvature bins, `_apply_speed_limits`, `speed_limit_preview` cascade, `steering_speed_guard`, `straight_target_hold`, `target_speed_blend`, `straight_speed_smoothing`, `curve_mode_speed_cap`, `perception_health_status` percentage reductions).
  - **Config:** Single global `comfort_governor_max_lat_accel_g: 0.25` threshold.
  - **PP lookahead fix:** `reference_lookahead_scale_min: 1.0` (lookahead no longer shrinks with speed).
  - **Stale hold:** Speed capped at current when perception stale for 3+ consecutive frames.
  - **Longitudinal tuning:** `startup_ramp_seconds: 2.5` (from 4.0), `straight_throttle_cap: 0.45` (from 0.3), `throttle_rate_limit: 0.06` (from 0.04).
  - **Telemetry:** `speed_governor_comfort_speed`, `speed_governor_preview_speed`, `speed_governor_horizon_speed` in HDF5 + PhilViz v64.
  - **Tests:** 33 unit tests (7 Phase 0 bugfix + 19 SpeedGovernor + 7 existing planner), all passing.
  - **Validation:** E2E Unity validation pending.

- `S1-M35` (done): PP steering pipeline bypass — eliminate phase-lag oscillation:
  - **Problem:** S1-M33 PP implementation kept PID-era steering post-processing pipeline active (jerk limiting at 0.12/s², EMA smoothing alpha=0.9, adaptive rate computation, sign flip override). This introduced 0.3-0.5s phase lag that defeated PP's geometric steering: jerk limiter active 85% of turn frames, steering transfer ratio as low as 0.34, and re-introduced the oscillation PP was meant to fix.
  - **Fix:** Bypass jerk limiting, EMA smoothing, sign flip override, adaptive rate computation, and straight-away adaptive tuning when `control_mode == "pure_pursuit"`. PP path uses only: (1) simple fixed rate limit (`pp_max_steering_rate: 0.4`), (2) hard clip (`max_steering`). Turn feasibility governor and lateral accel estimation still computed for speed governor and telemetry.
  - **Config:** `pp_max_steering_rate: 0.4`, `pp_feedback_gain: 0.0` (pure geometric for initial validation). `reference_lookahead_scale_min: 0.45`, `reference_lookahead_tight_scale: 0.55` (shorter lookahead for stronger PP authority in curves).
  - **Telemetry:** `pp_pipeline_bypass_active` flag in HDF5 + PhilViz v65. Waterfall bypass note in diagnostics.
  - **Tests:** 66 unit tests passing (3 new PP pipeline tests + 10 existing PP + 53 PID/longitudinal). Zero PID regressions.
  - **E2E validation (s-loop, canonical start):** Failure at F571 (vs F295 pre-bypass). Oscillation eliminated (0.01 sign-change rate). Jerk limited: 0%, EMA smoothing: 0%, transfer ratio ~1.0. Remaining failure is upstream reference point loss in second S-turn transition (trajectory/perception issue, not control).

- `S1-M36` (done): Trajectory ref tracking + speed governor tuning — clear s-loop:
  - **Problem:** S1-M35 showed PP control was correct but trajectory `ref_x` lagged GT center by up to 0.86m during S-turn transitions. Root cause: `ref_x_rate_limit: 0.12` too tight for rapid lane center movement (0.25+ m/frame). Curvature-based scaling didn't activate because curvature at the reference point is near-zero during transitions. Also, speed 10+ m/s was too fast for turn radius.
  - **Fix 1 (ref tracking):** `ref_x_rate_limit: 0.12→0.22`, `ref_x_rate_limit_turn_scale_max: 2.5→4.0`, `ref_x_rate_limit_precurve_scale_max: 5.0→6.0`, `ref_x_rate_limit_curvature_min: 0.003→0.001`, `ref_x_rate_limit_curvature_scale_max: 3.0→5.0`, `lane_position_smoothing_alpha: 0.35→0.20`. Added drift-rate-based dynamic rate limit relaxation in trajectory module.
  - **Fix 2 (speed):** `curve_mode_speed_cap_mps: 8.0→7.5`.
  - **Fix 3 (steering authority):** `max_steering: 0.6→0.7`.
  - **E2E validation (s-loop, canonical start):** Full 40s run, 733 frames, **0 out-of-lane events**. Lag max: 0.37m (vs 0.86m), rate limit active: 0% (vs 5%), speed in curves: 7.2 mean / 9.3 max (vs 10+ m/s). Closest margin to boundary: 0.46m left, 0.55m right.
  - **Remaining:** Lateral error RMSE 0.55m, P95 0.97m — tuning opportunity. Stale rate 20.7% (all `left_lane_low_visibility`, 0 sign violations).

- `S1-M37` (done): Dynamic per-radius speed control + PP tuning:
  - **Problem:** Speed control used 6 redundant limiting mechanisms; only track speed limit (8.0 m/s) was active. Comfort governor (0.25g) was too permissive for measured curvatures. Every curve got the same fixed speed. Lateral RMSE was 0.55m (target < 0.40m). Pipeline measured ~40% of actual curvature.
  - **Fix 1 (calibrated comfort governor):** `comfort_governor_max_lat_accel_g: 0.25→0.20`, added `curvature_calibration_scale: 2.5` to compensate for pipeline under-measurement. Formula: `v = sqrt(a_lat / (|k| * scale))`. Keeps physics meaningful (0.20g = real passenger comfort), scale absorbs pipeline distortion.
  - **Fix 2 (curvature history):** Track last 5 curvature values, use `max(history)` for comfort speed to prevent speed spikes during S-turn transitions where curvature momentarily drops to zero.
  - **Fix 3 (cleanup):** Removed ad-hoc curvature cap block (duplicate of comfort governor), removed unused `curve_mode_speed_cap_mps` and associated dead variables/telemetry. Raised curvature bins to safety-only (3.0 m/s²).
  - **Fix 4 (lateral error):** `pp_feedback_gain: 0.0→0.10` for steady-state drift correction.
  - **Tests:** 124 tests passing (9 new: calibration scale, curvature history, per-radius speed).
  - **E2E validation (3x s-loop, canonical start):** All 3 runs: 100% in lane, 0 out-of-lane events. Lateral RMSE: 0.320m / 0.329m / 0.325m (from 0.55m baseline, target < 0.40m — achieved). Centeredness: 90.8% / 93.3% / 90.7%. Score: 78.1 / 78.0 / 78.1.

- `S1-M38` (done): PP steering jerk reduction via jerk-limited rate ramp:
  - **Problem:** Steering jerk was the largest score penalty (-20 points). 61-73% of jerk events occurred at hard PP rate limit clip transitions (rate jumping from "following demand" to "clamped at 0.4" in one frame). Max jerk: 93-123/s² across 3 baseline runs.
  - **Fix (jerk-limited rate ramp):** Added `pp_max_steering_jerk: 30.0` (per s²) parameter. Instead of hard-clipping the rate, the rate itself ramps with bounded jerk: `max_rate_delta = pp_max_steering_jerk * dt²`. Also added max_steering ceiling anticipation to prevent jerk from hard clip at max_steering boundary. State tracked via `_pp_last_steering_rate` (post-clip, post-ceiling, decayed on stale perception).
  - **Telemetry:** New HDF5 fields `control/pp_steering_jerk_limited` (flag) and `control/pp_effective_steering_rate` (rate tracking).
  - **Tests:** 126 tests passing (2 new: jerk bounded, rate ramp convergence). All existing PP tests pass.
  - **E2E validation (3x s-loop, canonical start):** All 3 runs: 100% in lane, 0 out-of-lane events. Jerk (proper dt): max 47.0-64.6/s² (from 93-123), P95 21.0-24.8/s² (target <30), events >30: 10-15 (from 31-33). Lateral RMSE: 0.338-0.348m (no regression). Speed: mean 6.0-6.2, max 9.6-9.9 m/s.

- `S1-M39` (done): Longitudinal comfort tuning:
  - **Problem:** accel_p95 and jerk_p95 exceed Phase 1 gates (accel_p95 ≤ 3.0 m/s², jerk_p95 ≤ 6.0 m/s³). Baseline: accel_p95 3.24 m/s², jerk_p95 89.43 m/s³ (post S1-M38).
  - **Fix (config alignment + slew):** (1) Align speed planner limits with longitudinal controller (max_accel, max_jerk, max_decel). (2) Curve transition: curve_preview_max_decel 1.8→1.2, slew rates down. (3) Startup/low-speed: startup_accel_limit 0.7→0.5, low_speed_accel_limit 0.8→0.6. (4) Rate limits: throttle_rate_limit, brake_rate_limit, throttle_smoothing_alpha tightened.
  - **PhilViz v69:** Comfort gates aligned to 3.0 m/s² / 6.0 m/s³; LongitudinalComfort limit labels updated; Accel Cap Active / Jerk Cap Active rows in Control section.
  - **Metric fix:** Gap-filtered `steering_jerk_max` (1.5× median dt two-sided filter) eliminates Unity frame-drop artifacts. Raw preserved as `steering_jerk_max_raw`. Score recalibrated: jerk penalty zero at/below cap (18.0).
  - **Tests:** 126 passing (config alignment + signal chain).
  - **E2E validation (2026-02-22):** All comfort gates pass on both tracks.
    - s_loop (60s): score 95.6/100, lateral RMSE 0.203m, accel P95 1.32 m/s², cmd jerk P95 1.44 m/s³, steer jerk 18.4 (at cap), 0 e-stops.
    - highway_65 (60s): score 96.2/100, lateral RMSE 0.044m, accel P95 1.08 m/s², cmd jerk P95 1.44 m/s³, steer jerk 18.0 (at cap), 0 e-stops.

- `S1-M40` (done): Stack isolation tooling:
  - **Goal:** Per-layer replay enabling isolated regression testing (attribution of errors to perception, trajectory, or control).
  - **Stage 1 (T-020):** `replay_perception_locked.py` — monkeypatches `perception.detect` to return locked 4-tuple from HDF5; trajectory + control run live. Reconstruction of `lane_coeffs` from vlen float32 HDF5 using degree-2 polynomial + `num_lanes_detected` + `left/right_lane_line_x` disambiguation. Falls back to live perception if lock data missing.
  - **Stage 2 (pre-existing):** `replay_trajectory_locked.py` — locks trajectory reference point output.
  - **Stage 3 (pre-existing):** `replay_control_locked.py` — locks control commands.
  - **Stage 4 (pre-existing):** Deterministic latency injection via `--lock-latency-ms` in trajectory-locked harness.
  - **Stage 5 (pre-existing + extended):** `counterfactual_layer_swap.py` — now includes 4-run perception self/cross matrix and 3-layer scorecard (`upstream-perception-dominant`, `upstream-trajectory-dominant`, `downstream-control-dominant`, etc.).
  - **Stage 6 (T-025):** PhilViz Chain tab — per-frame causal chain view (Perception → Trajectory → Control) with stale-propagation banner, replay-mode locked-layer context, and `perception_lock_source_recording` badge key.
  - **Cadence triage:** `tools/analyze/cadence_breakdown.py` (CLI + JSON), PhilViz **Cadence** tab (`/api/recording/.../cadence-breakdown`), gate bundle `cadence_breakdown/*.json`. Attributes wait_input vs pipeline, queue depth, frame-id skips; `UNITY_DT_SPIKE` uses `vehicle/unity_delta_time` (not `stream_front_unity_dt_ms`, which is stream lag). Recordings also embed **`control/perf_*_ms`** layer timings (perception / planning / control / recorder ingest / wait_input); see `docs/plans/perf_layer_timings_impl.md`. CLI/API default **`target_hz`** matches **`stack.target_loop_hz`** in `config/av_stack_config.yaml` when not overridden.
  - **Perf sequencing:** While **`QUEUE_BACKLOG`** is present (front queue pinned), prioritize **consumer / ingest** (bridge loop, async ingest, Unity/Python FPS alignment, Plan A transport) over **Plan B** mask/recorder micro-opts; see `docs/plans/perf_pareto_plan_b_postprocess_recorder_arch.md` (Priority note). **Shipped:** `stack.camera_prefetch`, `stack.clear_bridge_camera_queue_on_start`, `stack.bridge_max_camera_queue` → `AV_BRIDGE_MAX_CAMERA_QUEUE`, bridge **drop counter** + **drain_queue** (see `docs/plans/perf_pareto_plan_a_bridge_gpu_topdown.md`).

**Gate to pass Stage 1** ✓ *Passed via S1-M39 validation (2026-02-22)*
- No centerline cross in first-turn window. ✓
- No out-of-bounds in first-turn window. ✓
- Comfort metrics remain within envelope. ✓

### Stage 2: Robustness, Infrastructure & Speed Expansion ← **CURRENT STAGE**
**Goal:** before any capability expansion (actors, multi-lane), harden the stack's testability,
document its parameter surface, and validate it at higher speeds and across more track geometries.
Stages 2–4 as originally scoped (turn exit, turn-to-turn handoff, longer-horizon stability) were
implicitly achieved through S1-M36–S1-M39: the stack now runs 60s full-course passes on two
tracks at 95+ score with zero lane violations. The meaningful next barrier is not turn-shape
coverage — it is the ability to iterate safely and quickly as the system grows in scope.

- **Trajectory / signal integrity (2026-03-21):** Heading-zero gate releases when the **curve scheduler** reports far-preview phase, **ENTRY/COMMIT**, or **time-to-curve** inside a configurable window (not only `preview_curvature_abs`); YAML tweaks for **earlier** curve anticipation (`curve_phase_preview_curvature_min`, `curve_preview_far_on`, `curve_local_phase_reentry_gate_min`). Validates with `tests/test_steering_jerk_and_heading_gate_fixes.py`.

**Current micro-steps**
- `S2-M1` (next): Automated comfort-gate regression — HDF5-replay-based pytest that checks all
  S1-M39 gates without Unity running. Foundation for safe iteration on all future milestones.
  - Entry criterion: ≥1 recorded run per track on disk; gate checks accel P95, jerk P95,
    lateral RMSE, e-stops using the existing replay isolation harnesses.
  - Promotion gate: comfort gates pass on replay without live Unity in CI.
- `S2-M2` (next): Config parameter documentation — `CONFIG_GUIDE.md` covering all 100+ YAML
  parameters with ranges, defaults, and tuning effects. Derived from M37–M39 tuning history.
  - Entry criterion: S2-M1 done (document params once we can verify effects via replay).
  - Promotion gate: every param in `av_stack_config.yaml` has a documented range and effect.
- `S2-M3` (next): `av_stack.py` module decomposition — extract lane gating logic
  (`clamp_lane_center_and_width`, `apply_lane_ema_gating`, `blend_lane_pair_with_previous`)
  into its own `perception/lane_gating.py` module. 5,420-line orchestrator is the primary
  source of cascading-bug risk for all future milestones.
  - Entry criterion: S2-M1 done (automated gate confirms no regression after refactor).
  - Promotion gate: all 126+ tests pass; live run scores within 2 points of S1-M39 baseline.
- `S2-M4` (future): Higher-speed validation — push target speed from 12 m/s → 15 m/s on
  highway_65. Validate comfort governor scaling, PP lookahead adaptation, accel/jerk gates.
  - Entry criterion: S2-M1 + S2-M2 done (need automated gates + documented params before
    pushing speed, to make regressions observable).
  - Promotion gate: all S1-M39 comfort gates pass at 15 m/s on highway_65 (3× runs).
- `S2-M5` ✅ **COMPLETE (2026-03-15)**: Track coverage expansion — 3 additional tracks validated:
  mixed_radius (94.1), sweeping_highway (93.6), hairpin_15 (91.6 with Stanley k=3.0).
  All meet promotion gate: RMSE ≤ 0.40m, 0 e-stops, all comfort gates pass (3× runs each).
  Curvature-adjusted scoring deployed (floor=3×|κ|). Per-track gate overrides removed.
- `S2-M6` ✅ **COMPLETE (2026-03-20)**: Oscillation attribution — shared library `tools/oscillation_attribution.py` (lead–lag on perception→ref→error→steer, subtype, `recommended_fixes` incl. hill-highway co-tuning guidance). **Auto-tag:** `attrs['metadata']` (`recording_provenance.track_id`, `recording_name`, notes) + filename match `hill_highway` (see `tracks/hill_highway.yml`). **PhilViz:** `GET .../oscillation-attribution` + Chain tab (default auto). **CLI:** `python tools/oscillation_attribution.py <recording.h5>`; overrides `--force-hill-highway` / `--no-hill-highway`. **Tests:** `tests/test_oscillation_attribution.py`. Complements `BlameTracer`; pre-failure window aligns with `drive_summary` `failure_frame`.
  - **Tuning (2026-03-21):** `av_stack_config.yaml` `control.lateral` — PP oscillation mitigation (`pp_feedback_gain` 0.075, feedback_gain 1.05–1.25, `base_error_smoothing_alpha` 0.78, `steering_smoothing_alpha` 0.92, `grade_steering_damping_gain` 6.5, `curve_mode_speed_cap_mps` 7.0). **Regression:** `tests/test_oscillation_regression_metrics.py`; optional env `AV_HILL_HIGHWAY_OSCILLATION_REGRESSION_H5` for post-fix HDF5.

**Gate to pass Stage 2**
- Automated comfort-gate CI passing without Unity (S2-M1).
- All 100+ config params documented (S2-M2).
- `av_stack.py` decomposed into ≥3 modules with no score regression (S2-M3).
- Stack validated at 15 m/s on highway_65 (S2-M4).
- ≥1 additional track validated beyond s_loop/highway_65 (S2-M5).

### Stage 3: Turn-to-Turn Handoff *(Implicitly achieved via S1-M36–S1-M39)*
**Original goal:** stable transition across entry → exit → next entry without oscillatory handoff.
**Status:** Validated on 60s full-course s_loop runs (multiple S-turn sequences, 95+ score, 0
violations). No explicit milestone needed; promoted as a gate to Stage 2 was satisfied during
Stage 1 progression.

### Stage 4: Longer-Horizon Stability *(Implicitly achieved via S1-M36–S1-M39)*
**Original goal:** no drift accumulation over longer runtime.
**Status:** 60s runs on both tracks stable with monotonically low lateral RMSE and 0 e-stops.
Oscillation slope confirmed negative (stable) on both tracks. Promoted as satisfied.

### Stage 5: Turn-Radius Coverage Expansion
**Goal:** robust lane-keeping across wider turn-radius set.
**Progress:** highway_65 (large-radius, high-speed) validated at 96.2/100. Tight-radius stress
test (r < 15m dedicated track) is mapped as S2-M5 above.

### Stage 6: Grade and Banking Expansion
**Goal:** add elevation effects after flat-turn robustness is achieved.
- 6a: straight grades.
- 6b: banked turns.
**Entry criterion:** Stage 2 complete.

### Stage 7: Actor/Lead-Vehicle Introduction
**Goal:** introduce forward-following behavior only after lane-keeping matrix is stable.

**Entry criteria for Stage 7**
- Lane-keeping pass rate meets target across scenario matrix (>90% repeated runs).
- No comfort regression vs baseline gates.
- Failure modes are rare and attributable.
- Automated regression CI active (S2-M1) — required to safely coexist lateral + longitudinal
  control loops during ACC/car-following development.

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

**Post-Phase-2 hard stop (ADDRESSED via S1-M33 + S1-M35)**
- Architecture escalation decision: **Pure Pursuit implemented** (S1-M33) with **pipeline bypass** (S1-M35). PID oscillation was unfixable via tuning; geometric Pure Pursuit controller with direct-delivery pipeline eliminates oscillation by design. Current gating issue: reference point stability during S-turn transitions (trajectory/perception, not control). Rollback to PID available via `control_mode: pid`.

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

---

## Capability Roadmap — Next 10 Steps (2026-03-04)

Ordered sequence of major capability additions. Each step builds on the previous.
Steps 1-4 harden the existing foundation. Steps 5-7 add core driving capabilities.
Steps 8-10 add intelligence and robustness.

**Current state:** PP (s-loop, <10 m/s) + Stanley (hairpin, <6 m/s) + Linear MPC (highway, 10-20 m/s)
with map-based curvature preview. Five tracks validated. No actors. Single lane.
GT lane center serves as our HD map signal (same role as pre-built HD maps in real AV stacks).
Curvature-adjusted scoring active (floor=3×|κ| per frame). 749 tests, 32/32 comfort gates green.

### Step 1 — Track Coverage Expansion (S2-M5) ✅ COMPLETE (2026-03-15)

**Goal:** Prove the control architecture generalizes beyond s_loop and highway_65.

**Validated tracks (all pass promotion gate: RMSE ≤ 0.40m, 0 e-stops, 3× runs):**
- mixed_radius: 94.1/100 (PP↔MPC hybrid, R40-R200)
- sweeping_highway: 93.6/100 (MPC-dominant, R300 corners)
- hairpin_15: 91.6/100 (Stanley k=3.0, R15/R20 back-to-back arcs)

**Key outcomes:**
- Curvature-adjusted scoring deployed — unified formula, no per-track gate overrides
- Stanley low-speed regime (Phase 2.9) — reactive controller for tight turns below MPC threshold
- Curvature feedforward tested and rejected (no RMSE gain on consecutive tight arcs)

**Gate:** Met — all 5 tracks validated with comfort gates green
**Unlocks:** Confidence that control works on diverse geometry, not just tuned tracks

### Step 2 — Automated A/B CI Regression (T-033)

**Goal:** Prevent parameter regressions automatically on every config or code change.

**What to build:**
- CI pipeline that runs comfort gate tests on committed recordings
- Automated A/B comparison: any config change must not regress scores by > 2 points
- Gate report as CI artifact (pass/fail + metric deltas)

**Entry:** Step 1 done (need multiple track recordings in the gate set)
**Gate:** CI runs green on all track recordings; blocks merge on regression
**Unlocks:** Safe iteration speed for all future steps — every change is automatically validated

### Step 2 — Automated A/B CI Regression (T-033) ✅ COMPLETE (2026-03-16)

**Goal:** Automatically detect and block scoring regressions on every push/PR.

**What was built:**
- `tests/test_scoring_regression.py` — 25 tests: re-scores 5 golden recordings against frozen baselines
- `tests/fixtures/scoring_baselines.json` — frozen per-track metrics (score, adj_rmse, accel, jerk)
- `tools/ci/check_config_regression.py` — config diff detector (scoring-critical param classification)
- CI workflow: scoring regression step (blocking) + config change report (informational, PR-only)
- Gate bundle regression deltas (conftest.py) → PhilViz Gates tab color-coded delta table

**Gate:** CI runs green on all track recordings; blocks merge on regression — ✅
**Unlocks:** Safe iteration speed for all future steps — every change is automatically validated

### Step 3 — Grade and Banking

**Goal:** Add elevation effects to the dynamics model and simulation.

**What to build:**
- Pitch/roll in Unity vehicle state output
- Gravity component in MPC dynamics (longitudinal: `a_eff = a - g·sin(pitch)`)
- Lateral load transfer on banked turns (effective gravity changes)
- Banked and graded track profiles
- Validate comfort gates hold on graded tracks

**Entry:** Step 2 done ✅ (CI catches regressions from dynamics changes)
**Gate:** All comfort gates pass on graded + banked tracks (3x runs each)
**Unlocks:** More realistic simulation, validates MPC dynamics model is extensible

### Step 3.5 — 2DOF Feedforward Alignment (LMPC steady-state fix)

**Goal:** Eliminate systematic outside-curve bias in the linear MPC without per-track tuning.

**Root cause:** The existing `r_steer_rate * (δ[k+1] − δ[k])²` penalty fights holding a
constant-curvature steering angle, causing late turn-in and e_lat P50 = 0.405m on R100 curves.

**What was built:**
- `MPCSolver._feedforward_delta_norm(κ, L, δ_max)` — bicycle-model FF angle, normalized
- `ff_alignment_enabled` param in `MPCParams` (default: `True`)
- Per-solve linear `q` correction: `q[δ_k] += r_sr·Δff_k`, `q[δ_{k+1}] -= r_sr·Δff_k`
  — shifts rate cost to penalize Δ(δ − δ_ff)² instead of Δδ²; P matrix unchanged
- 4 new tests in `TestFFAlignment` (formula, straight no-op, disabled regression, convergence)

**Key property:** On straight roads (κ=0) the correction is exactly zero — no behavior change
on highway_65. On constant-κ segments the correction eliminates the rate penalty for holding
curve steer so the QP converges to δ_ff without fighting itself.

**Additional fix — Dynamic Lane-Midpoint Curve-Contamination Clamp (2026-03-20):**
Root-cause triage (trajectory layer isolation) identified that `coord_conversion_distance` in
`orchestrator.py` was using the full PP lookahead even when it extended past upcoming curve starts.
On sloop (short straights, κ ≈ 0.01), this biased `center_x` by ~14 cm before the turn began,
explaining persistent lateral offset independent of control/perception tuning.

- **Formula:** max_d_into_curve = √(2 · threshold / κ); clamp only fires on straight approach
- **New function:** `compute_lane_midpoint_clamp()` in `av_stack/lane_gating.py` (pure, tested)
- **Config params:** `lane_midpoint_max_curve_contamination_m: 0.05` (default),
  `lane_midpoint_clamp_min_distance_m: 3.0` in `config/av_stack_config.yaml`
- **Diagnostics:** `lane_midpoint_clamp_active`, `_dist_m`, `_kappa_preview` recorded per frame
- **Tests:** 9 tests in `tests/test_lane_midpoint_clamp.py` — all pass; full suite 962/962

**Entry:** Step 3 done ✅ (grade compensation validated on hill_highway)
**Gate:** sloop/oval adj_rmse ≤ 0.25m; `diag_raw_ref_x` straight-segment mean within ±0.03m of zero. **Pending live E2E validation.**
**Unlocks:** Reduces steady-state curve error before deciding whether NMPC is required

### Step 4 — NMPC + Full Hierarchical Hybrid (plan.md §2.7-2.8)

**Goal:** Extend control to >20 m/s and mixed-speed routes (highway ramp → highway → off-ramp).

**What to build:**
- `control/nmpc_controller.py` — CasADi + IPOPT nonlinear MPC
  - Full nonlinear bicycle dynamics (no small-angle approximation)
  - `sin()`/`tan()` in dynamics, automatic Jacobian/Hessian via CasADi
  - Handles `|e_heading| > 0.25 rad` where LMPC linearization breaks
- Extend `RegimeSelector` for 3-regime operation (PP → LMPC → NMPC)
- Mid-curve transition suppression
- Warm-start handoff between LMPC and NMPC
- Speed regime: PP (<10 m/s) → blend → LMPC (10-20 m/s) → blend → NMPC (>20 m/s)

**Entry:** Step 3.5 done (2DOF FF alignment validated — prove LMPC steady-state error is resolved
before adding NMPC complexity; if e_lat still exceeds gate after FF alignment, re-evaluate.)
**Gate:** Mixed-speed route (ramp + highway + local) completes with 0 e-stops, score ≥ 90
**Unlocks:** Full speed envelope coverage, enables highway on/off-ramp scenarios

### Step 5 — Lead Vehicle Following / ACC

**Goal:** First actor interaction — react to a vehicle ahead in the same lane.

**What to build:**
- IDM (Intelligent Driver Model) or time-gap ACC for desired speed computation
- Lead vehicle detection (start with GT position from Unity, extend to perception later)
- Speed planner consumes lead-vehicle desired speed as additional constraint
- Scenarios: constant lead speed, lead deceleration, stop-and-go

**Entry:** Step 2 done (CI prevents lateral regressions while adding longitudinal behavior)
**Gate:** No collision, min time-gap ≥ 1.5s, jerk P95 ≤ 6.0 m/s³, speed RMSE ≤ 2.0 m/s vs desired
**Unlocks:** Longitudinal planning that reacts to the world, not just the road

### Step 6 — Multi-Lane Perception + Map

**Goal:** See and represent multiple lanes (still stay in ego lane).

**What to build:**
- Extend segmentation model to detect lane boundaries (not just center)
- Lane-level map representation (lane count, lane ID, adjacency graph)
- Per-frame output: ego lane geometry + neighbor lane geometry
- Adjacent lane occupancy awareness (which lanes have vehicles)

**Entry:** Step 5 done (actor awareness infrastructure exists)
**Gate:** Lane detection accuracy ≥ 95% on multi-lane tracks, no regression on single-lane tracks
**Unlocks:** Foundation for lane change decisions

### Step 7 — Lane Change Planning + Execution

**Goal:** First lateral decision-making — change lanes to pass or merge.

**What to build:**
- Gap acceptance logic (is there safe space in the target lane?)
- Lateral trajectory generation (quintic polynomial or spline path)
- MPC tracks lane-change trajectory (this is where MPC excels over PP)
- Safety validation: collision-free trajectory check before committing
- Scenarios: free lane change, merge with traffic, abort on closing gap

**Entry:** Steps 5 + 6 done (need actor awareness + multi-lane perception)
**Gate:** Lane change success ≥ 95%, no collision, lateral jerk P95 ≤ 6.0, time-in-target-lane ≥ 95%
**Unlocks:** Highway driving that isn't stuck behind slow vehicles

### Step 8 — Prediction (Other Vehicle Trajectories)

**Goal:** Anticipate what other vehicles will do over the next 2-5 seconds.

**What to build:**
- Constant-velocity prediction baseline (trivial but useful)
- Lane-following prediction (vehicles follow their detected lane)
- Interaction-aware prediction (vehicles react to ego behavior)
- Eventually: learned prediction models if sufficient data collected

**Entry:** Step 7 done (lane changes need prediction for safe gap acceptance)
**Gate:** Prediction horizon ≥ 3s, ADE ≤ 1.0m at 2s, lane change safety improves with prediction
**Unlocks:** Proactive rather than reactive behavior, safer multi-agent interactions

### Step 9 — Intersection Handling + Global Route Planning

**Goal:** Navigate stop lines, traffic signals, and turn policies. Introduce a global planner
that decides *which* path to take at junctions — the first step toward cross-country navigation.

**What to build:**
- Traffic signal state ingestion from Unity
- Stop-line detection and smooth braking approach
- Protected turns first (green arrow = go)
- Unprotected turns (yield to oncoming traffic)
- Turn trajectory planning through intersections
- **Global planner:** road network graph (nodes = intersections, edges = road segments);
  A* or Dijkstra to compute a route from A to B
- **Route profile handoff:** global planner emits an ordered polyline of road segments
  (curvature, speed limits, junction type at each meter) that replaces the current static
  track YAML. The local planner (PP/MPC) follows whatever profile it receives — it does not
  need to know "left or right", only the geometry ahead.

**Architectural note (2026-03-10):**
The current system treats a track YAML as a fixed closed loop. Phase 3 config work
(auto-derive thresholds from geometry) builds the right abstraction — but the *input*
is still a static file. In a map-based world, the route profile is dynamically assembled
by the global planner from an HD map graph on each trip. The local planner code changes
minimally; the intelligence of which road segment to follow lives entirely in the global
planner layer. Intersection left/right decisions are navigation decisions, not control
decisions.

**Entry:** Steps 5 + 8 done (need actor awareness + prediction for yield decisions)
**Gate:** Stop-line compliance ≥ 98%, red-light violations = 0, jerk P95 ≤ 6.0 m/s³,
route followed correctly through ≥ 3 junction types
**Unlocks:** Urban driving capability — stack goes from "highway lane-keeper" to "city navigator"

### Step 10 — End-to-End Robustness + Regression

**Goal:** Prove everything works together under stress and perturbation.

**What to build:**
- Noise injection on perception (Gaussian blur, dropout, latency spikes)
- Wind/grade perturbation on dynamics
- Combined scenario matrices (lane change during lead-vehicle braking on a banked curve)
- Full regression suite covering all prior scenarios
- Performance budgets per component (perception < 5ms, control < 2ms, etc.)

**Entry:** Steps 1-9 done
**Gate:** ≤ 10% degradation in core metrics under stress, no new failure modes, full suite green
**Unlocks:** Confidence the system is robust, not just demo-able

---

## Artifacts and Ownership (Rolling)
- Phase reports and gate results stored under `tmp/analysis/`.
- Visualizer panels/waterfalls updated with each promoted diagnostic.
- Scenario and evaluation scripts under `tools/analyze/`.
- This roadmap must be updated whenever a phase/stage changes status.
