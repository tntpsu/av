# AV Stack — Agent Memory: Tasks

**Last updated:** 2026-03-04

---

## Current Focus

**Phase 2.7b validated.** PP (<10 m/s) + LMPC (10-20 m/s) with map-based curvature preview.
Highway median 98.1, s-loop 97.1, 0 e-stops.

**Phase 2.8 planned** — MPC pipeline integration for mixed-speed routes. See `plan.md §2.8`.
MPC low-speed experiment failed (RMSE 0.63m vs PP 0.21m). Root causes: orchestrator recovery mode interference, missing rate limiter, under-compensated delay. `mpc_sloop.yaml` reverted to `pp_max_speed_mps: 10.0`.

**Workstream C1 COMPLETE (2026-03-09):** Straight latch 36.7% → 15.6%. Root cause was `curve_local_phase_time_start_tight_s: 2.5` (too wide for s_loop lane-poly geometry). Fixed to 1.2s. S2 latch 10.8-11.2%, C1/C3 peaks 0.572-0.612m / 0.566-0.578m. Highway non-regression 98.4/100. All C1 acceptance gates pass.

**Remaining open:** Workstream C2 (PP floor rescue ~1.7m structural) and Phase 2.8 planning.

---

## Backlog (immediate)

| ID | Task | Rationale |
|---|---|---|
| T-031 | ~~Activate and test MPC controller~~ | ✅ Done — Phase 2 MPC (2.1-2.7b) |
| T-033 | Automated A/B regression in CI | Prevent parameter regressions on every config change (Roadmap Step 2) |

## Forward Roadmap (see `docs/ROADMAP.md §Capability Roadmap`)

| Step | Capability | Status |
|---|---|---|
| — | PP turn-entry C1 fix (time_start 2.5→1.2s) | **✅ Done (2026-03-09)** |
| — | PP turn-entry C2 (floor rescue ~1.7m structural) | Next |
| — | Phase 2.8 MPC pipeline integration (plan.md §2.8) | Planned |
| 1 | Track coverage expansion (S2-M5) | Next |
| 2 | Automated A/B CI regression (T-033) | Pending |
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
