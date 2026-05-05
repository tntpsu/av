# AV Stack — Agent Memory: Tasks

**Last updated:** 2026-05-05

---

## Queued — pick up next

### T-ACC-TRIAGE-B — Overnight triage stage for ACC sweep (Proposal B)

**Context:** Proposal A (ACC composite score, landed 2026-05-05) added a continuous
0–100 score alongside the existing PASS/FAIL gates so the sweep email shows trend
signal. Proposal B is the second half: have the nightly agent **diagnose** each
FAIL/WARN scenario with `/diagnose` and append a hypothesis + recommended next
step to the email. Read-only — no code changes, no PRs.

**Why deferred:** A first lets us see how the *shape* of the new scores changes
night-over-night before deciding what triage hypotheses are worth generating.
A run, gather 2–3 nights of score data, then build B with that data informing
the triage clustering rules.

**Plan summary** (full design in conversation 2026-05-05 + memory
`project_acc_triage_stage_b_deferred.md`):
1. New Step 4 in `tools/nightly/acc-sweep/PROMPT.md`: for each FAIL/WARN, run
   `/diagnose <recording>`, write a 5-line triage block per failure to the report.
2. **Cluster identical signatures.** If H2/H6/G1/G2 all fail with `gt_right_offroad`,
   diagnose ONCE on the worst, emit one cluster block — don't burn 4 diagnoses.
3. **Suppress known-issue scenarios.** A1's "target_speed mismatch" is already
   documented; agent should emit "(known issue: see project_xxx)" instead of
   re-diagnosing.
4. Cap at **6 `/diagnose` runs/night** (~$3 added cost; total ~$6 stays under
   $10 budget cap).
5. `compose_subject()` extends with `triage=N_clusters_diagnosed,M_known`.
6. **No code/config changes from the agent.** Just hypothesis + suggested
   next manual step.

**Activate when:** A has been running ≥3 nights AND we have ≥1 score-driven
finding the gates missed (e.g., a scenario that PASSed verdict but Behavior<70).

### Current Focus

### T-ACC-IDM-PLUMBING — ACC emergency brake authority (2026-04-21, Commits A/B/C/C.1/D landed; G2 validated; H5 harness deferred)

Restored ACC actuator authority in EMERGENCY_BRAKE / TTC_ESTOP / COLLAPSED_GAP_STOP states.
IDM's -12 m/s² saturation was being clipped by three stacked comfort limiters; each commit peeled one off.

- ✅ A (1a5826d): scoring-registry exemption infrastructure
- ✅ B (f718818): scoring-consumer wiring
- ✅ C (f79bbea): controller `effective_max_decel`/`dynamic_max_jerk` widening
- ✅ C.1 (805b305): jerk cooldown `× 0.4` bypass in emergency
- ✅ D (390f057): orchestrator B1 short-circuit — `EMERGENCY_BRAKE → brake=1.0`, 12 unit tests
- ✅ G2 E2E validation: **885 → 2 e-stops** on `hill_g2_stop_on_grade.yml` (99.8% reduction).
  EMERGENCY_BRAKE 4/4 @ brake=1.0 (D path), COLLAPSED_GAP_STOP 2748/2778 @ brake=1.0 (C+C.1 path).
  Weighted 91.3, capped to 79.0 by yellow-Safety (2 residual e-stops) + yellow-SignalIntegrity (heading suppression).
- ⏸️ G1 (H5): **DEFERRED** — track harness failure prevented lead engagement (see `project_h5_harness_deferred.md` memory tripwire; escalate on third hit).
- 🟡 Phase H (docs): in progress this session.
- 🟢 Plan closed as "mechanism proven." Residuals tracked as separate tasks below.

Recording: `recording_20260421_230013.h5`. Plan: `.claude/plans/acc-idm-accel-plumbing.md`.

#### Follow-ups from T-ACC-IDM-PLUMBING (separately scoped)

- **T-HEAD-SUPPR — SignalIntegrity "heading suppression on curves" -25 deduction.**
  Present on G2 (score 79 capped by SignalIntegrity=75). Pre-exists this plan.
  Start: `/diagnose data/recordings/recording_20260421_230013.h5` focused on
  heading-suppression rate. If tunable → `/iterate`; if architectural → `/plan-feature`.
  **Executing this session.**
- **T-ACC-STATE-LATE — 2 residual G2 e-stops: ACC state machine transitions late.**
  Frame-level inspection of `recording_20260421_230013.h5` showed: (a) at onset-1,
  ACC_ACTIVE held for 8 frames while IDM demanded -5 m/s² but the controller
  applied +throttle; (b) at onset-2, 10+ frames of CUTOUT with a closing lead
  before state flipped to EMERGENCY_BRAKE. B1 fires correctly the frame the state
  finally transitions — the bug is upstream in the ACC decision layer, not the
  actuator chain. Possibly the same routing gap flagged in
  `project_acc_brake_authority_findings.md` for H5. Start: `/diagnose` + `/plan-feature`.
- **T-ACC-UNIFY — Post-Phase-G refactor: unify all three ACC emergency states under D's orchestrator short-circuit.**
  G2 evidence: EMERGENCY_BRAKE (D path) hits brake=1.0 on 100% of frames; COLLAPSED_GAP_STOP (C+C.1 path)
  on 98.9%; TTC_ESTOP only 69% (16 frames total, but notable gap). Current architecture has
  two paradigms — orchestrator short-circuit vs in-controller widening — which is a design
  smell. Refactor: move TTC_ESTOP + COLLAPSED_GAP_STOP routing into the orchestrator's
  `_apply_acc_emergency_override()`, delete emergency-aware branches from `pid_controller.py`
  (net -60 LOC, -3 config flags). Defer until G2 + H5 fully validated — refactor risk > benefit today.
- **T-B1-HDF5-PLUMBING — Wire `control/b1_bypass_active` into HDF5 via the 6-location pattern.**
  The flag is set on `control_command` in D but isn't recorded; proof of B1 firing on G2 was
  inferred from brake=1.0 during EMERGENCY_BRAKE rather than observed directly. Small; maybe 30 min.
- **T-H5-HARNESS — Diagnose H5 harness deterministic failure.**
  Two-hit tripwire set in `project_h5_harness_deferred.md`. Symptoms: curve-cap=100% /
  target=2.09 m/s / bridge payload age 34s on what should be straight highway stop-go.
  Escalate to full diagnostic session on third hit OR if same symptom appears on any other track.

### T-FRENET-MPC — Frenet-frame MPC reference (2026-04-18, Phases A–F landed, Phase G gated on user)

Replace the MPC scalar `e_lat(0)` lookahead-projection input with Frenet `d`
(linearized: `d = gt_cross_track_lookahead + Ld·sin(gt_heading_error_rad)`,
PLUS-sign verified in Phase 0 against at-car reference on 2152 H2 frames).
Intended to fix H2 ACC regression (Trajectory 98.1→79.0 from commit 7e3caf0
hidden `Ld × q_lat` heading-leak gain). Shadow-mode-first rollout.

- ✅ Phase 0: offline Phase-0 verification (`tools/analyze/analyze_frenet_shadow.py` confirmed PLUS formula)
- ✅ Phase A: 3 config keys under `trajectory.mpc:`, default `frenet_linearized` + `shadow_mode=true`
- ✅ Phase B: helpers `_compute_frenet_d_linearized` + `_select_mpc_e_lat_reference` in `control/pid_controller.py`, LMPC+NMPC call sites replaced, inactive-branch defaults safe
- ✅ Phase C: HDF5 6-location pattern — 3 new fields (`mpc_e_lat_frenet_linearized_m`, `mpc_e_lat_shadow_delta_m`, `mpc_e_lat_frenet_shadow_mode`)
- ✅ Phase E: diagnostics — 2 issue detectors, 2 triage patterns + 4 metrics, 2 layer-health flags, §N pipeline analysis section
- ✅ Phase F: 14 unit tests + 2 integration tests (end-to-end MPC + LMPC/NMPC parity). Critical sign-pin test locks PLUS formula. All 44 MPC tests + 32 comfort gate tests green.
- ⏸️ Phase G: **BLOCKED on user confirmation.** G1 shadow E2E on H2 → G2 active → G3 5-track regression sweep → G4 ACC re-sweep (H5/H6/A1/G1/G2).
- ⏸️ Phase H: `docs/agent/architecture.md` §5 Control update (controller-side — after G2 lands); `CONFIG_GUIDE.md` docs for new keys.

Plan: `.claude/plans/greedy-swimming-naur.md`. Memory: `project_frenet_mpc_reference.md`.

---

**MPC QP reformulations tested (2DOF + preview weighting) — both disabled, RMSE ceiling confirmed at 0.111m from kinematic model-plant mismatch. Next: system identification or frequency-dependent damping.**

### Pre-computed Velocity Profile — DONE ✅ (2026-04-10)
Forward+backward kinematic velocity profile from track geometry. Additive speed ceiling alongside existing preview/curve_cap chain. `a_lat_tracking_budget_g: 0.05` calibrated from closed-loop data. 19 unit tests, diagnostic tooling across 3 tools, ACC dynamic recomputation ready. Next: improve lateral controller → raise tracking budget → profile becomes binding constraint → faster curve speeds.

**Follow-ups (deferred):**
- Remove ~29 deprecated curve_preview/curve_cap/anticipation params (only after profile is binding)
- Eliminate per-track speed overlays (hairpin target_speed, mixed curve_cap_peak_lat_accel_g)
- Validate ACC scenarios with profile active

### Inter-Frame Control Extrapolation — DONE ✅ (2026-04-01)
Lightweight MPC updates between camera frames using fresh GT vehicle state. Effective rate: 17.7 Hz (camera 11.3 Hz + ~1 interframe/cycle). Smith delay cap at 80ms. Key bug fixed: `_run_interframe_update()` had `time.sleep(self.frame_interval)` on every exit — main loop already rate-limits at 25ms, so internal sleeps were throttling. 35 tests in `test_interframe_extrapolation.py`. Enabled in `mpc_highway.yaml`.

### LMPC Oscillation Fix — DONE ✅ (2026-03-31)
Root cause: e_lat ramp limiter (0.05 m/frame) created limit cycle. Ramp disabled, r_steer_rate scheduling kept, e_lat attenuation removed. 28 tests passing. All diagnostic tools updated.

**Step 4 NMPC — COMPLETE ✅ (2026-03-23). Best: H-3 97.5/100, 0 e-stops. Step 5 ACC plan ready.**

**Phase completion status:**
- ✅ Phase A: `tracks/autobahn_30.yml` + `config/mpc_autobahn.yaml` + scoring_registry NMPC constants
- ✅ Phase F: `config/av_stack_config.yaml` — `trajectory.nmpc` section, `control.regime.lmpc_max_speed_mps`
- ✅ Phase B: `control/nmpc_controller.py` — NMPCParams, NMPCSolver (scipy SLSQP + analytical adjoint gradient), NMPCController. **Adjoint bug fixed: A_k[1,2] was (v/L)·tan(δ)·dt, corrected to (1/L)·tan(δ)·dt.**
- ✅ Phase C: `control/pid_controller.py` — NONLINEAR_MPC dispatch block, LMPC fallback path, nmpc_* metadata
- ✅ Phase D: `data/recorder.py` + `data/formats/data_format.py` + `av_stack/orchestrator.py` — 7 nmpc_* HDF5 fields (all 6 registration locations)
- ✅ Phase E1-E5: Tools (analyze_drive_overall, mpc_pipeline backend, triage_engine, issue_detector, layer_health) + drive_summary_core.py NMPC metrics
- ✅ Phase G1+G2: Unit tests (test_nmpc_controller.py 29 tests, test_regime_selector_nmpc.py 17 tests) — 46/46 passing
- ✅ nmpc_q_lat speed-adaptive scaling: added ("trajectory.nmpc","nmpc_q_lat") to `_MPC_WEIGHT_AUTO_DERIVE_PARAMS` with base=0.7,v_ref=15.0,κ_ref=0.002. All 145 tests green.
- ✅ **Phase H-2 PASSED (2026-03-23):** 0 e-stops, 91.2/100 (recording_20260323_095252.h5). NMPC 269/714 frames, 100% feasibility, P95=12.34ms.

**Phase H-3 PASSED (2026-03-23): 97.5/100, 0 e-stops, oscillation NOT runaway.**
Recording: `recording_20260323_111239.h5`

Changes applied:
- `control/regime_selector.py`: added `lmpc_nmpc_blend_frames` (default 30) and `lmpc_nmpc_min_hold_frames` (default 40) — independent of PP↔LMPC blend/hold
- `config/mpc_autobahn_nmpc_test.yaml`: target_speed 15→25 m/s, removed explicit nmpc_q_lat=0.7 (auto-derive now ~1.62 at 25 m/s), lmpc_nmpc_blend_frames=30, lmpc_nmpc_min_hold_frames=40
- `tests/test_regime_selector_nmpc.py`: added 2 tests for new params (18/18 passing)

Results vs H-2:
| Metric | H-2 (15 m/s) | H-3 (25 m/s) |
|---|---|---|
| Score | 91.2/100 | **97.5/100** |
| Lateral RMSE | 0.212m | **0.071m** |
| Lateral P95 | 0.554m | **0.163m** |
| Oscillation runaway | YES | **NO** |
| RMS growth | 0.004→0.501m | 0.006→0.095m |
| Max Δsteer @ transition | 0.172 | **0.078** (55% ↓) |
| Control score | 80/100 | **100/100** |
| Centeredness | 92.7% | **100%** |

**Speed finding**: The `target_speed: 15.0` test overlay was the bottleneck — NOT the Unity motor.
Car now reaches 25 m/s (11.7% overspeed rate at target=25). Speed RMSE 5.3 m/s = car averages ~20 m/s during acceleration; peaks above 25 m/s. `maxMotorTorque=1500f` is sufficient. T-079 CLOSED.

**Open items:**
- 10 regime transitions still occur (MPC→PP at 12-14 m/s periodically). Not yet identified. Low priority given 97.5 score.
- SignalIntegrity -18.8 (heading suppression on curves) — only remaining deduction. Pre-existing.

**Phase 2.8 VALIDATED on highway (2026-03-12).** MPC pipeline fixes (2.8.1–2.8.4) complete:
- 2.8.1: Recovery mode suppression (skip ×1.2/×1.5 when MPC active)
- 2.8.2: Steering feedback fix (`update_last_steering_norm` after all post-hoc modifications)
- 2.8.3: MPC-aware rate limiter (`mpc_max_steering_rate_per_frame: 0.15 rad/frame`)
- 2.8.4: Configurable delay compensation (`mpc_delay_frames: 2`, linear extrapolation)
- Highway validation: median 97.4/100, 0 e-stops (5 runs)

**Phase 2.8.6 mixed_radius — VALIDATED ✅ (2026-03-13):** median **93.85/100** (4 MPC-active runs: 93.9/93.8/93.9/93.8), 0 e-stops.
- Root cause of prior 79/100: base `q_lat=10.0` caused MPC hunting on R150/R200 (κ=0.007). Fix: `mpc_q_lat=0.5 + mpc_r_steer_rate=2.0` (matches highway-validated settings) → e_lat P95 0.97→0.43m.
- Score cap explained: 79 was Trajectory yellow cap (P95>0.47m triggers hard 79 cap in scoring). Not cadence.
- Golden recording registered: `recording_20260313_122635.h5`. All comfort gate tests passing (32/32).
- Note: ~2/6 runs are PP-only if circuit starts near tight corners. Score gate: 90 for MPC-active runs.

**Workstream C1 COMPLETE (2026-03-09):** Straight latch 36.7% → 15.6%. Fixed to 1.2s. C1/C3 peaks 0.572-0.612m. Highway non-regression 98.4/100.

**Workstream C2 DEFERRED:** PP floor rescue (~1.7m) is structural arc-chord geometry on R40. Low ROI. Only revisit if mixed_radius/sweeping_highway exhibit actual gate failures.

---

## Completed Since Last Update

| ID | Task | Date |
|---|---|---|
| — | Bridge regression fix: CameraCapture/AVBridge targetFPS 13→15; bridge/client.py persistent executor | 2026-03-22 |
| — | Teleport observability: `control/teleport_detected` + `control/teleport_jump_m` in HDF5 (5 locations) | 2026-03-22 |
| — | Tool updates: forced_pp_transition_count metric, priority-0 "STOP PP GAIN TUNING" recommendation, flat_oscillation_not_grade + forced_pp_regime_reset triage patterns | 2026-03-22 |
| — | Clean baseline established: hill_highway 94.9/100, forced_pp_transition_count=0 (recording_20260322_165640.h5) | 2026-03-22 |
| — | Codex A/B results invalidated: grade_steering_damping_gain/pp_feedback_gain/pp_max_steering_rate A/B tests all ran against artifact-corrupted baseline (2-4 false teleports each) — all revert to prior defaults | 2026-03-22 |
| T-078 | Late turn-in root cause fully investigated: MPC cost function structural trade-off (q_lat=1.60, r_steer_rate=2.0 → breakeven q_lat=11.5 for full fix). kappa_ref preview fix tried → oscillation runaway → reverted. PP floor rescue BENIGN (floor lowering caused regression). **DEFERRED.** | 2026-03-22 |
| 3.5 | 2DOF FF alignment live validated: ff_alignment=True vs False A/B on hill_highway. +0.4 pts, jerk cap hit without (9.2→18.0), adj_rmse 0.297→0.309m, oscillation runaway without. Feature confirmed beneficial, stays default=True. ROADMAP adj_rmse gate (0.25m) not met — gate pre-dates Step 4 MPC-as-primary and should be updated. | 2026-03-22 |
| Step 5 A-D | NMPC Phases A-D: autobahn_30 track, mpc_autobahn.yaml overlay, nmpc_controller.py (SLSQP + adjoint gradient), pid_controller.py dispatch, HDF5 7-field registration. 1334 tests passing. | 2026-03-22 |
| Step 5 E+G | NMPC Phases E (tools) + G (unit tests): issue_detector/layer_health/triage_engine/mpc_pipeline/analyze_drive_overall/drive_summary_core NMPC updates; 46 new unit tests (29 controller + 17 regime selector). Adjoint bug fixed: A_k[1,2] corrected from (v/L) to (1/L). | 2026-03-22 |
| Step 5 H | NMPC Phase H E2E: autobahn_30 PARTIAL PASS. NMPC activated (317/1075 frames, 29.5%), P95=14.66ms, 100% feasibility, 0% LMPC fallback — solver mechanically correct. BUT 2 e-stops from oscillation runaway: nmpc_q_lat=2.0 tuned for 21 m/s, over-aggressive at 12-13 m/s. SmithΔ 0.485m at e-stop frames. Adjoint fix (1/L vs v/L) confirmed by solver correctness. | 2026-03-22 |
| Step 5 nmpc_q_lat | Speed-adaptive nmpc_q_lat: added to `_MPC_WEIGHT_AUTO_DERIVE_PARAMS` (base=0.7,v_ref=15,κ_ref=0.002). Test YAML: 2.0→0.7. Base config: 2.0→0.7. Auto-derive: at 25m/s autobahn→1.94, hill_highway 12m/s κ=0.010→2.24. All 145 tests green. Phase H-2 re-run needed to confirm 0 e-stops. | 2026-03-22 |
| Step 5 H-2 | Phase H-2 PASSED: 0 e-stops ✅, 91.2/100 ✅ (target ≥85). NMPC 37.7% frames, 100% feasibility, P95=12.34ms. Residual: oscillation amplitude runaway (RMS 0.004→0.501m), 11 regime transitions (Δsteer=0.172 at handoff), NLP cost std/median=1.3 (warm-start diverging). Root cause: chatter at 9 m/s NMPC threshold. Recording: recording_20260323_095252.h5. | 2026-03-23 |
| Step 4 H-3 | Phase H-3 PASSED: 97.5/100, 0 e-stops. regime_selector.py: lmpc_nmpc_blend_frames=30/min_hold_frames=40. Test overlay: target_speed 15→25 m/s, nmpc_q_lat auto-derive (~1.62 at 25 m/s). Oscillation NOT runaway (0.006→0.095m). Lateral RMSE 0.212→0.071m. Max Δsteer at transition 0.172→0.078 (55% ↓). Control 80→100/100. Speed ceiling confirmed NOT Unity motor limit — was test overlay target_speed=15. Car reaches 25+ m/s. 18/18 regime selector tests passing. Recording: recording_20260323_111239.h5. | 2026-03-23 |
| Step 4 H-4 | Phase H-4 FAILED (79.0/100) → inference.py REVERTED. Attempted: add map_preview_κ≤0.001 to heading gate ON guard to fix SignalIntegrity -18.8. Failure mode: `_map_preview_curvature_abs` permanently ≥0.00167 on R600 autobahn → gate never re-arms → jitter propagates → RMSE 0.071→0.301m. Correct fix: pass `current_path_kappa` into gate function (deferred — does not block Step 5). Autobahn baseline H-3 97.5/100 stands. | 2026-03-23 |
| Step 5 plan | `docs/plans/step5_acc_plan.md` — 5-phase Lead Vehicle Following/ACC plan. IDM longitudinal law, safety layer, PhilViz ACC tab, 5 E2E scenarios. ROADMAP updated. | 2026-03-23 |
| Debug gaps | 4 tooling gaps fixed: issue_detector κ threshold 0.003→0.0005, triage_engine heading_gate_stuck_on_curve pattern, diagnostics.py heading_suppression_rate + map_preview_curvature per-frame, PhilViz exposes control/curvature_preview_abs. | 2026-03-23 |
| ACC Phase D | Full ACC toolset: scoring_registry (14 constants), drive_summary_core (_build_acc_health_summary + ACC penalties), issue_detector (6 issue types), triage_engine (4 patterns), layer_health (_score_acc), acc_pipeline_analysis.py CLI, PhilViz ACC tab (acc_pipeline.py backend), analyze_drive_overall Section 16. 19 tests. 0 scoring regressions. | 2026-03-23 |
| ACC Phase A | Python data pipeline: VehicleState +8 fields, DataRecorder 5 HDF5 locations × 8 fields (vehicle/radar_fwd_* + vehicle/acc_*), orchestrator _pf_run_acc_sensor + init + VehicleState injection, config acc: block (enabled=false). test_lead_vehicle_data_pipeline.py extended to 23 tests. 0 regressions. 1403 tests total. | 2026-03-23 |
| ACC Phase B | IDM longitudinal controller: acc_controller.py (ACCState enum, ACCParams + from_config(), ACCOutput, ACCController — 6-state machine + IDM + bumpless transfer). Orchestrator: ACCController init, _pf_run_acc_sensor stores RadarReading, new _pf_apply_acc_override (between governor and trajectory planner). Config: 5 new acc keys. test_acc_controller.py 15 tests all passing. 1525 tests total, 0 regressions. | 2026-03-24 |
| ACC Phase C | Safety layer wiring: emergency_stop_latched set in _pf_apply_acc_override when TTC_ESTOP fires (reason='acc_ttc_violation', same latch mechanism as OOL events). test_acc_safety.py 16 tests — TTC guard boundaries (< threshold), EMERGENCY_BRAKE conditions (gap factor, range_rate sign, priority), detection-loss hysteresis (5 frames, 3 re-arm). 1538 tests total, 0 regressions. | 2026-03-24 |
| MPC 2DOF FF | 2DOF feedforward/feedback decomposition in mpc_controller.py. QP variable ε=δ−δ_ff, curvature cancels (99.9%). Marginal 2% RMSE improvement — MPC already finds FF in 2 frames. Disabled (`mpc_ff_decomposition_enabled: false`). 22 tests. Diagnostic: diagnose_ff_decomp.py. | 2026-04-13 |
| MPC preview q_lat | Preview-weighted stage cost shaping: q_lat ramps from step0_scale to 1.0 across horizon. At scale=0.7: halves jerk but no RMSE gain, does not shift stability boundary. Disabled (`mpc_q_lat_preview_step0_scale: 1.0`). 15 tests. | 2026-04-13 |
| RMSE ceiling | Confirmed RMSE 0.111m floor at q_lat=2.0 is kinematic model-plant mismatch. Neither QP reformulation breaks it. Stability boundary (q_lat≈2.5) is whole-horizon, not step-0. Next: sysid or frequency-dependent damping. | 2026-04-13 |

---

### T-NMPC-COLD-START-DETERMINISM — NMPC sign-determinism fix across BLAS (2026-05-02)

NMPC SLSQP cold-start at `kappa=0` with non-zero `e_lat` was producing different signs on Mac Accelerate vs Linux OpenBLAS, blocking CI since 2026-04-12. Root cause: `x0` was set to `_feedforward_delta_norm(kappa=0, ...) = 0`, putting SLSQP at the saddle point of a near-symmetric cost surface. BLAS arithmetic noise tipped the basin choice.

**Fix:** Added a sign-preserving lateral-error seed to cold-start `x0[0]` in `control/nmpc_controller.py` (after the existing curvature feedforward). Magnitude is bounded to `cold_start_e_lat_seed_max_frac × max_steer_rad` (default 10%) so it doesn't dominate SLSQP optimization.

- ✅ Phase 0: `tools/analyze/validate_nmpc_cold_start.py` confirmed root cause; gain sweep showed gain≥0.3 produces deterministic sign across `e_lat ∈ [-1, 1]`
- ✅ Phase A: 2 new fields in `NMPCParams` (`cold_start_e_lat_seed_gain=0.5`, `cold_start_e_lat_seed_max_frac=0.10`) with `from_config` mapping
- ✅ Phase B: cold-start path in `SLSQPNMPCSolver.solve()` adds `x0[0] += clip(-e_lat * gain, ±seed_max)` after curvature feedforward
- ✅ Phase F: 6 new unit tests in `TestColdStartSeed` class — applied/zero/clipped/disabled/no-warm-start coverage. All 91 controller tests + xdist regression suite green
- ⏸️ Phase G2 (E2E hairpin_15 validation): deferred to next session — change only affects first-frame cold-start; warm-start chain takes over from frame 2; production behavior change is negligible
- ⏸️ Phase G1 (golden recording revalidation): deferred — no scoring code changed, expected delta = 0
- ⏸️ Phase H (memory file capturing the solver-cold-start-determinism pattern): deferred

**Why fields were added at code level:** This is a CODE PATCH, not config tuning. The two fields exist as config knobs only to enable disable/tune via YAML if a future regression appears (rollback in 30 sec without code revert).

**Production impact:** Minimal. NMPC is gated `kappa ≥ 0.015` in production, so the cold-start saddle case (straight road + non-zero e_lat) is rarely hit. When it is, warm-start dominates from frame 2.

## Backlog (immediate)

| ID | Task | Rationale |
|---|---|---|
| T-031 | ~~Activate and test MPC controller~~ | ✅ Done — Phase 2 MPC (2.1-2.7b) |
| T-033 | ~~Automated A/B regression in CI~~ | ✅ Done (2026-03-16) — `test_scoring_regression.py` (25 tests), `check_config_regression.py`, CI workflow, PhilViz regression deltas |
| T-070 | ~~Delete 2 remaining permanently-skipped obsolete tests~~ | ✅ Done (2026-03-16) — deleted `test_control_lateral_error_straight_road` and `TestHeadingAccumulation` stub |
| T-071 | ~~Fix 3 recording-dependent skips → pin to golden recordings~~ | ✅ Done (2026-03-16) — `test_camera_height_assumption` → xfail, `test_actual_lane_width_from_recorded_data` → golden fixture (3-5m range), `test_heading_fix_with_real_unity_data` → golden fixture |
| T-072 | ~~Fix 2 test logic issues~~ | ✅ Done (2026-03-16) — `test_reference_point_format` uses `get_reference_point()`, `test_pre_deployment_30_seconds` inverted skip removed |
| T-073 | ~~Centralized scoring registry~~ | ✅ Done (2026-03-16) — `tools/scoring_registry.py` (single source of truth for 25+ thresholds). Consumers: conftest.py, drive_summary_core.py, layer_health.py, issue_detector.py. 27 guard tests. 842 tests passing. |
| T-074 | ~~Unity shader pre-warming~~ | ✅ Already functional — `ShaderPrewarmer.cs` uses `@RuntimeInitializeOnLoadMethod(BeforeSceneLoad)` which auto-runs without scene attachment. CameraCapture has startup grace period. |
| T-075 | ~~Arc-distance curve phase scheduler — highway config~~ | ✅ Config done (2026-03-16) — highway overlay uses `reference_lookahead_entry_track_name: highway_65`, legacy time gates removed. Scoring regression passes (10/10). Live Unity A/B deferred. |
| T-076 | ~~Config Phase 3: auto-derive curvature thresholds~~ | ✅ Done (2026-03-16) — 4th-root scaling from track κ_max, 22 derivable params, highway overlay 90→65 lines, 37 new tests, 811 total passing |
| T-077 | ~~Cadence investigation — mixed_radius severe_rate~~ | ✅ Resolved differently (2026-03-13). Root cause was NOT cadence — it was MPC hunting (`q_lat=10.0` on R150/R200). Fix: `q_lat=0.5 + r_steer_rate=2.0`. Score 79→94. Cadence has zero weight in scoring. Instrumentation plan (`cadence-performance-plan.md`) deferred as nice-to-have diagnostics. |
| ~~T-JERK-MEAS~~ | ~~Post-limiter recovery-mode steering multiplier — unified lateral-error term via /plan-feature~~ | ✅ **Resolved (2026-04-18)** via `.claude/plans/greedy-swimming-naur.md` — PP recovery term (continuous smoothstep, upstream of rate/jerk limiters) replaces orchestrator post-limiter 1.2×/1.5× multiplier. Implementation at `control/pid_controller.py:2618`; orchestrator multiplier deleted at `av_stack/orchestrator.py:9420-9443` (throttle reductions + e-stop retained). Hairpin_15: **80 → 100 Control, 95.7 → 98.7 overall, jerk max 30.08 → 18.0**. Regression sweep (highway_65, s_loop, mixed_radius, hill_highway): all within frozen tolerances. Sub-question (signal disagreement) resolved via `max(|lookahead|, |gt_at_car|)` with HDF5 telemetry for shadow validation; §27 analyzer + `recovery_signal_disagreement` triage pattern catch this regime going forward. See `project_pp_recovery_term.md` for full evidence chain; `project_recovery_mode_post_limiter.md` superseded. Tests: 16 new unit+integration tests, 57/57 scoring+comfort-gate tests PASS. |

## Forward Roadmap (see `docs/ROADMAP.md §Capability Roadmap`)

| Step | Capability | Status |
|---|---|---|
| — | PP turn-entry C1 fix (time_start 2.5→1.2s) | **✅ Done (2026-03-09)** |
| — | PP turn-entry C2 (floor rescue ~1.7m structural) | **DEFERRED** — see current_state.md |
| — | Phase 2.8 MPC pipeline integration (2.8.1–2.8.4) | **✅ Done (2026-03-12)** |
| — | Phase 2.8.6 mixed_radius validation | **✅ Done (2026-03-13)** |
| — | mpc_pipeline_analysis.py + PhilViz MPC Pipeline tab | **✅ Done (2026-03-13)** |
| — | Phase 2.9 Stanley low-speed regime + hairpin validation | **✅ Done (2026-03-15)** |
| — | Curvature-adjusted scoring + unified gates | **✅ Done (2026-03-15)** |
| — | Track coverage expansion (S2-M5) — 5 tracks validated | **✅ Done (2026-03-15)** |
| 1 | Automated A/B CI regression (T-033) | **✅ Done (2026-03-16)** |
| 2 | Config Phase 3: auto-derive curvature (T-076) | **✅ Done (2026-03-16)** |
| 3 | Grade and banking | **✅ Done (2026-03-17)** — hill_highway 89.6/100 E2E validated |
| 4 | MPC as primary lateral controller + q_lat auto-derive | **✅ Done (2026-03-17)** — curvature guard, 954 tests |
| T-078 | Lookahead contraction smoothing at curve entry (late turn-in) | **DEFERRED (2026-03-22)** — root cause: MPC cost function trade-off (needs q_lat=11.5 → hunting risk). PP floor rescue is benign. Baseline 94.9/100 accepted. |
| 3.5 | 2DOF FF alignment (`ff_alignment_enabled`) | **✅ Validated (2026-03-22)** — A/B: +0.4 pts, jerk 9.2→18.0 without it (cap hit), adj_rmse 0.297→0.309m without. Stays enabled (default=True). |
| 4 | NMPC + full hierarchical hybrid | **✅ COMPLETE (2026-03-23)** — H-3 97.5/100, 0 e-stops, 25 m/s |
| 4.5 | MPC QP reformulations (2DOF + preview weighting) | **Tested, DISABLED (2026-04-13)** — RMSE 0.111m ceiling is kinematic model-plant mismatch. Next: sysid dynamic model or frequency-dependent damping. |
| 5 | Lead vehicle following / ACC | **Plan updated (2026-03-23)** — `docs/plans/step5_acc_plan.md`. Simulated radar (SphereCast + Doppler), NOT GT. `RadarSensor` ABC → reusable for Step 7 BSM. Full toolset: scoring_registry, issue_detector, triage_engine, layer_health, PhilViz ACC tab, CLI tool. ~52 new tests. Begin Phase A. |
| 6 | Multi-lane perception + map | Pending |
| 7 | Lane change planning + execution | Pending |
| 8 | Prediction (other vehicle trajectories) | Pending |
| 9 | Intersection handling | Pending |
| 10 | End-to-end robustness + regression | Pending |
| T-PHILVIZ-MOBILE | PhilViz mobile-friendly CSS — responsive breakpoints, touch-optimized controls, large tap targets, no hover-only interactions. User now accesses via VPN from mobile (2026-05-02). | **Queued** |
| T-PHILVIZ-PICKER | Track / scenario multi-select picker widget — lists `tracks/*.yml` + `tracks/scenarios/*.yml`, supports 1-or-many selection. Standalone reusable widget that other pages consume. | **Queued** (depends on MOBILE) |
| T-PHILVIZ-SKILLS | Skills runner page — web UI to trigger slash commands (`/sweep`, `/e2e`, `/acc-sweep`, `/diagnose`, etc.). Open design questions: SSE vs polling vs websockets for output streaming; cancel UX for long-running skills (some are 15+ min); auth model (currently VPN-only perimeter, no per-user auth). | **Queued** (depends on PICKER) |

---

## Nightly Automation Roadmap

Active nightly jobs (launchd, see `tools/nightly/`):

| Job | Schedule | Purpose | Status |
|---|---|---|---|
| fix-tests | Daily 2:00 AM | Run pytest, autofix STALE_BASELINE/IMPORT/OBSOLETE_TEST, open PR for fixes, report REAL_BREAK / FLAKY | ✅ Active (2026-05-01) |
| sweep (lateral) | Daily 3:00 AM | Cross-track regression sweep via `/sweep`, email digest of score deltas vs baselines. **Lateral base tracks only** | ✅ Active (2026-05-01) |
| acc-sweep | Daily 4:00 AM | ACC scenario sweep via `/acc-sweep`, verifies each `tracks/scenarios/*.yml` against its `Expected:` header gate criteria. Per-scenario PASS/FAIL/WARN/SKIPPED/AMBIGUOUS | ✅ Active (2026-05-02) |
| process-health | Sunday 5:00 AM | Weekly Pareto digest via `/process-health` from `data/reports/improvement_log.json` (moved from 4am to make room for acc-sweep) | ✅ Active (2026-05-01, rescheduled 2026-05-02) |

Deferred (advisory list — add when there's appetite for more inbox volume):

| Idea | Cadence | Effort | Why |
|---|---|---|---|
| Stale-branch sweep | Weekly Sun | S | Close merged `nightly/*` branches, surface untouched-14d branches in digest |
| Storage hygiene digest | Weekly Sun | S | List recordings + logs eligible for archival/deletion (proposes only in V1, never auto-deletes; golden recordings are sacred) |
| Dependency staleness | Weekly | S | `pip list --outdated` digest with CVE flags; no auto-upgrade |
| Memory hygiene | Monthly | S | Surface SUPERSEDED entries and 60+ day untouched memories from `~/.claude/projects/-Users-philtullai-av/memory/`; ask which to prune |
| Performance metric trend dashboard | Daily snapshot, weekly digest | M | Snapshot per-track scores to a CSV/SQLite; weekly trend email shows e.g. "lateral RMSE drifted +0.02m over 14d" before crossing the gate |
| Config drift digest | Weekly Fri | S | Wrap `tools/ci/check_config_regression.py --base origin/main~7d` in a digest of "what config changed this week and which scoring tests it affects" |
| `recording_provenance.scenario_id` HDF5 field | One-shot refactor | M | Add scenario filename to recording metadata at recorder.py write time; downstream consumers (scoreboard, sweep, acc-sweep) can disambiguate ACC runs from lateral runs on the same base track without filename heuristics |
| `/scores --scenarios` flag | One-shot tooling | S | Filter scoreboard.py output to ACC scenarios only; default behavior preserved. Pairs with above |
| `/pareto` and `/iterate-pareto` ACC awareness | One-shot tooling | M | Add scenario dimension to cross-track Pareto buckets; today they bucket by base track only |
| Per-scenario ACC baselines in `tests/fixtures/` | One-shot data | S | Establish frozen `expected_ttc_min`, `expected_collisions=0`, etc. per scenario so `/acc-sweep` can detect regressions, not just gate violations |

Explicitly NOT on the roadmap (anti-pattern for unattended cron):

- `/iterate` or `/iterate-pareto` overnight — needs human-in-the-loop classification per `feedback_iterate_fix_level.md` and `feedback_tuning_vs_architecture.md`. Only run interactively.
- Auto-merge of nightly fix-tests PRs — even mechanical fixes deserve a human glance.
- Code formatters/linters — single-developer research repo, on-demand `/simplify` is enough.

---

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Curvature measurement errors cascade to speed + steering | High | T-032: better perception; T-022 control sensitivity analysis (tooling exists) |
| 100+ config params, no automated regression | High | T-033: CI A/B batch testing |
| Perception stale-frame fallback breaks silently | Medium | Add explicit logging + stale-frame rate metric in analyzer |
| Segmentation model untrained / mislabeled | Critical | T-032: retrain with multi-track GT data. Pipeline ready: `./tools/segmentation/train_pipeline.sh`. See `training/TRAINING_GUIDE.md` |
| Temporal sync issues under CPU load | Medium | Latency injection tooling exists (`--lock-latency-ms`) |
| Runtime track-end stop bypassed on loop:true tracks | Medium | T-080: Unity doesn't loop some track meshes even when YAML says `loop: true`. Odometer stop (orchestrator:3551) only fires when `_track_loop=False`. Need runtime auto-detection (e.g., GT boundary corruption or odometer > total_length) to override the YAML flag and stop gracefully before OOL. |
| RMSE 0.111m floor from kinematic model-plant mismatch | Medium | Kinematic bicycle model limits MPC tracking accuracy. QP reformulations (2DOF, preview weighting) cannot break through. Next: system identification of dynamic plant model (`tools/analyze/sysid_dynamic_model.py`) or frequency-dependent damping. |

---

## Completed Tasks

| ID | Task | Milestone | Date |
|---|---|---|---|
| — | Pure Pursuit mode implementation | S1-M33 | Pre-2026 |
| — | PID pipeline bypass in PP mode | S1-M35 | Pre-2026 |
| — | Trajectory ref tracking + speed tuning | S1-M36 | Pre-2026 |
| — | Dynamic per-radius speed control | S1-M37 | Feb 2026 |
| — | Steering jerk reduction (PP mode) | S1-M38 | Feb 2026 |
| T-001 | Validate S1-M39 comfort gates | S1-M39 | 2026-02-22 |
| T-002 | Commit M39 changes (debug_visualizer, metric fixes) | S1-M39 | 2026-02-22 |
| T-003 | Integrate highway_65 track into test workflow | S1-M39 | 2026-02-22 |
| T-010 | 3× s_loop baseline — all comfort gates pass | S1-M39 | 2026-02-22 |
| — | Longitudinal comfort tuning | S1-M39 | 2026-02-22 |
| — | Gap-filtered steering_jerk_max metric + score recalibration | S1-M39 | 2026-02-22 |
| T-020 | Perception-locked replay (`replay_perception_locked.py` + counterfactual_layer_swap) | S1-M40 | 2026-02-23 |
| T-025 | PhilViz causal Chain tab + replay badge awareness | S1-M40 | 2026-02-23 |
| T-040 | Stage 2 milestone definition (S2-M1–S2-M5) in ROADMAP.md | S1-M40 | 2026-02-23 |
| T-012a–f | PhilViz Phases 3–5: layer_health, blame_tracer, triage_engine + all frontend tabs | PhilViz | 2026-02-23 |
| S2-M1 | Automated comfort-gate regression (19 tests, 2-tier, golden recordings) | S2-M1 | 2026-02-23 |
| S2-M2 | Config parameter documentation (CONFIG_GUIDE.md — 328 params) | S2-M2 | 2026-02-23 |
| T-050 | PhilViz cross-tab metric consistency + longitudinal comfort sub-panel | PhilViz | 2026-02-24 |
| T-051 | PhilViz Compare tab: pinned 1v1 table + localStorage persistence | PhilViz | 2026-02-24 |
| T-052 | Gates tab auto-bundle: conftest hooks write schema-v1 bundle on green run | PhilViz | 2026-02-24 |
| S2-M3 | av_stack.py → av_stack/ package (lane_gating, config, speed_helpers, orchestrator) + _process_frame → 13 _pf_* sub-methods | S2-M3 | 2026-02-24 |
| T-060 | Test suite cleanup: 29 pre-existing failures → 0 failed, 506 passed, 9 skipped | Post-S2-M3 | 2026-02-24 |
| — | Phase 2.1–2.7b MPC (solver, regime selector, highway true-state, curvature preview) | Phase 2 | 2026-03-09 |
| — | Phase 2.8 MPC pipeline fixes (recovery suppression, steering feedback, rate limiter, delay compensation) | Phase 2.8 | 2026-03-12 |
| — | Phase 2.9 Stanley low-speed regime + curvature-adjusted scoring | Phase 2.9 | 2026-03-15 |
| — | S2-M5 Track coverage expansion (5 tracks validated, unified scoring) | S2-M5 | 2026-03-15 |
| T-033 | Automated A/B CI regression (scoring_regression.py, check_config_regression.py) | Step 1 | 2026-03-16 |
| T-076 | Config Phase 3: auto-derive curvature thresholds (4th-root scaling, 22 params) | Step 2 | 2026-03-16 |
| T-073 | Centralized scoring registry (scoring_registry.py, 25+ thresholds, 27 guard tests) | Step 2 | 2026-03-16 |
| — | Step 3: Grade & Banking — full-stack grade compensation (Unity C# → Python → MPC/PP → scoring → PhilViz), hill_highway track, 40 new tests | Step 3 | 2026-03-16 |
| — | Step 3D E2E: hill_highway 89.6/100 (Trajectory gate cleared via q_lat auto-derive) | Step 3D | 2026-03-17 |
| — | Step 4: MPC as primary — curvature guard (κ>0.020→PP), mpc_min_speed base=3.0, per-track overlays, regime mask fixes, 9 registry constants, 17 new tests | Step 4 | 2026-03-17 |
| — | MPC q_lat auto-derive: `_derive_mpc_weights()` with 0.75-power formula, κ_mpc_active clamping, 8 new tests | Step 4 | 2026-03-17 |
| — | Step 5 Phase D: ACC scoring toolset (scoring_registry 14 constants, issue_detector 6 ACC types, triage 4 patterns, layer_health, PhilViz ACC tab, CLI) | Step 5 | 2026-03-24 |
| — | Step 5 Phase A: Python data pipeline (RadarSensor ABC, ForwardRadarSensor, VehicleState +8 fields, HDF5 recorder, orchestrator stub, acc: config block) | Step 5 | 2026-03-24 |
| — | Step 5 Phase B: IDM controller (ACCState, ACCParams.from_config, ACCOutput, ACCController 6-state machine, orchestrator wire-up with bumpless transfer) | Step 5 | 2026-03-24 |
| — | Step 5 Phase C: Safety layer (emergency_stop_latched wired from _pf_apply_acc_override, TTC strict-< boundary, detection-loss hysteresis) | Step 5 | 2026-03-24 |
| — | Step 5 Phase E: Unity integration (SpeedProfiler.cs, TrackWaypointFollower.cs, LeadVehicle.cs, TrackConfig.cs LeadVehicleConfig, TrackLoader.cs parser bugfix, CarController.cs +4 radar fields, AVBridge.cs SpawnLeadVehicle + ComputeForwardRadar, RoadGenerator.cs public ActiveTrackConfig, 3 acc_*.yaml overlays, 11 scenario YAMLs in tracks/scenarios/) | Step 5 | 2026-03-24 |
| — | MPC 2DOF FF decomposition: QP variable ε=δ−δ_ff, 99.9% curvature cancellation, marginal gain, disabled. 22 tests + diagnostic tool. | Step 4.5 | 2026-04-13 |
| — | MPC preview-weighted q_lat: stage cost shaping (step0_scale ramp), halves jerk at 0.7 but no RMSE gain, disabled. 15 tests. | Step 4.5 | 2026-04-13 |
| — | RMSE ceiling analysis: confirmed 0.111m floor from kinematic model-plant mismatch, stability boundary is whole-horizon not step-0. | Step 4.5 | 2026-04-13 |
