# AV Stack — Agent Memory: Tasks

**Last updated:** 2026-02-24

---

## Current Focus

**S2-M4 — Higher-speed validation (12 → 15 m/s on `highway_65`)**
Entry criteria met (S2-M1 + S2-M2 + S2-M3 all done). See `docs/ROADMAP.md §Stage 2` for full entry/promotion gates.

First step: A/B batch at 12 vs 15 m/s to identify which comfort gates break first.

---

## Backlog (non-milestone robustness work)

| ID | Task | Rationale |
|---|---|---|
| T-031 | Activate and test MPC controller | `control/mpc_controller.py` exists but is unused; potential performance gain over PP |
| T-032 | Train and validate segmentation model on s_loop data | CV fallback is de facto perception; ML would improve curve accuracy |
| T-033 | Automated A/B regression in CI | Prevent parameter regressions on every config change |

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
