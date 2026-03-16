# AV Stack — Agent Memory: Tasks

**Last updated:** 2026-03-16 (T-073, T-074)

---

## Current Focus

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
| 3 | Grade and banking | Pending |
| 4 | NMPC + full hierarchical hybrid (plan.md §2.7-2.8) | Pending |
| 5 | Lead vehicle following / ACC | Pending |
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
