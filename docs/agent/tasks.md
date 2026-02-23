# AV Stack — Agent Memory: Tasks

**Last updated:** 2026-02-22

---

## Active Tasks

| ID | Task | Priority | Status |
|---|---|---|---|
| T-020 | Implement perception layer stack isolation (replay with locked perception output) | P0 | Next |
| T-012 | Consolidate PhilViz per CONSOLIDATION_PLAN.md | P1 | Pending |
| T-011 | Tag S1-M39 completion in ROADMAP.md | P1 | Pending |

---

## Suggested Next Tasks (S1-M40 — Stack Isolation)

Listed in recommended order of execution:

### Stack Isolation (TODO Stage 1–6)

| ID | Task | Rationale |
|---|---|---|
| T-020 | Implement perception layer stack isolation (replay with locked perception output) | Enables attribution of tracking errors to perception vs. downstream |
| T-021 | Build trajectory-locked replay harness | Isolate control behavior from trajectory quality |
| T-022 | Build control-locked sensitivity analysis | Understand how perception noise propagates through control |
| T-023 | Deterministic latency injection profiles | Test control robustness under simulated latency |
| T-024 | Cross-layer counterfactual analysis (complete) | Root cause analysis tooling |
| T-025 | PhilViz upstream/downstream signal clarity | Make causal chain visible in visualizer |

### Robustness & Technical Debt

| ID | Task | Rationale |
|---|---|---|
| T-030 | Refactor av_stack.py into smaller modules | 5,420 lines is too large; lane gating logic should be in its own module |
| T-031 | Activate and test MPC controller | mpc_controller.py exists but is unused; potential performance gain |
| T-032 | Train and validate segmentation model on s_loop data | CV fallback works but ML perception would improve curve accuracy |
| T-033 | Automated regression: run A/B batch on every config change in CI | Prevent parameter regressions |
| T-034 | Document all 100+ config parameters with ranges and effects | Reduce trial-and-error tuning time |

### Stage 2 Prep

| ID | Task | Rationale |
|---|---|---|
| T-040 | Define Stage 2 milestones in ROADMAP.md (multi-lane, lane change) | Required before starting Stage 2 work |
| T-041 | Evaluate highway_65.yml track for Stage 2 testing | Track now integrated; assess suitability for higher-speed scenarios |

---

## Identified Improvements

### High Impact

1. **Stack isolation tooling (T-020–T-025)**
   - Current: Counterfactual replay exists but stack isolation is incomplete
   - Target: Full per-layer replay enabling isolated regression testing
   - Impact: Reduces debugging time; makes cross-layer attribution accurate

2. **Decompose `av_stack.py`**
   - Current: 5,420-line monolith orchestrating everything
   - Target: Extract lane gating module, perception coordinator, timing manager
   - Impact: Dramatically reduces bug surface and review complexity

3. **Automate comfort gate validation**
   - Current: Manual 3× run + analyze script
   - Target: `pytest`-integrated CI check using recorded HDF5 replay
   - Impact: Every config change gets instant regression feedback

4. **Train segmentation model**
   - Current: CV fallback is de facto perception path
   - Target: Trained CNN on s_loop data with GT lane labels
   - Impact: Better curve lane detection, reduced EMA gating dependency

5. **Config parameter documentation**
   - Current: 100+ YAML params with minimal inline documentation
   - Target: `CONFIG_GUIDE.md` updated with ranges, defaults, effects for every parameter
   - Impact: Reduces time-to-tune for each milestone by 30–50%

---

## Risks to Address

| Risk | Severity | Mitigation |
|---|---|---|
| `av_stack.py` complexity causes cascading bugs | High | T-030: decompose into modules |
| Curvature measurement errors cascade to speed + steering | High | T-022: control sensitivity analysis; T-032: better perception |
| 100+ config params, no automated regression | High | T-033: CI A/B batch testing |
| Perception stale-frame fallback breaks silently | Medium | Add explicit logging + metric for stale-frame rate in analyzer |
| Temporal sync issues under CPU load | Medium | T-023: latency injection testing |
| CV fallback accuracy degrades on curves | Medium | T-032: train segmentation model |

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
| — | A/B batch: metric extraction fix + new columns | S1-M39 | 2026-02-22 |
