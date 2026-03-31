# AV Stack — Agent Memory: Tasks

**Last updated:** 2026-03-31

---

## Current Focus

**Step 5 ACC in progress. LMPC oscillation fix COMPLETE ✅ (2026-03-31). Highway 97.5, Autobahn 96.4.**

### LMPC Oscillation Fix — DONE ✅ (2026-03-31)
Dual-mechanism: r_steer_rate speed scheduling (rate penalty) + e_lat speed attenuation (gain reduction), both curvature-gated (kappa_off=0.005). 28 tests passing. All diagnostic tools updated.

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

---

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
| 5 | Lead vehicle following / ACC | **Plan updated (2026-03-23)** — `docs/plans/step5_acc_plan.md`. Simulated radar (SphereCast + Doppler), NOT GT. `RadarSensor` ABC → reusable for Step 7 BSM. Full toolset: scoring_registry, issue_detector, triage_engine, layer_health, PhilViz ACC tab, CLI tool. ~52 new tests. Begin Phase A. |
| 6 | Multi-lane perception + map | Pending |
| 7 | Lane change planning + execution | Pending |
| 8 | Prediction (other vehicle trajectories) | Pending |
| 9 | Intersection handling | Pending |
| 10 | End-to-end robustness + regression | Pending |

---

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Curvature measurement errors cascade to speed + steering | High | T-032: better perception; T-022 control sensitivity analysis (tooling exists) |
| 100+ config params, no automated regression | High | T-033: CI A/B batch testing |
| Perception stale-frame fallback breaks silently | Medium | Add explicit logging + stale-frame rate metric in analyzer |
| CV fallback accuracy degrades on curves | Medium | T-032: train segmentation model |
| Temporal sync issues under CPU load | Medium | Latency injection tooling exists (`--lock-latency-ms`) |

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
