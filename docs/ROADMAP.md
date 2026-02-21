# Robust Full-Stack Roadmap (Unified, Layered, and Gated)

**Last Updated:** 2026-02-20  
**Current Focus:** Layer 2, Stage 1 (First-Turn Entry Reliability) signal-chain clearance + curvature-based speed + control refinement  
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

## Artifacts and Ownership (Rolling)
- Phase reports and gate results stored under `tmp/analysis/`.
- Visualizer panels/waterfalls updated with each promoted diagnostic.
- Scenario and evaluation scripts under `tools/analyze/`.
- This roadmap must be updated whenever a phase/stage changes status.
