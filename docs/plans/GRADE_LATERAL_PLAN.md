# Grade-aware lateral — implementation plan (observability first)

**Status:** Draft execution plan  
**Last updated:** 2026-03-22  
**Roadmap anchor:** `docs/ROADMAP.md` — Stage 6 (Grade and Banking), observability track **G6-L0–G6-L3**

**Problem statement:** Road grade changes longitudinal load, pitch, and lateral–longitudinal coupling. The stack already has **`vehicle/road_grade`** and **`control.lateral.grade_steering_damping_gain`**, but **attribution** (PhilViz + CLI + tests) is thin compared to flat-road lateral tuning. This plan **instruments and visualizes first**, then tunes policy with proof.

**Principles (align with project rules):**
- **One analytics module** consumed by CLI, `drive_summary` (optional keys), and PhilViz API — no duplicate math.
- **Pre-failure metrics** default where failure exists (`executive_summary.failure_frame`).
- **PhilViz + CLI parity** for every new diagnostic.

---

## Existing building blocks (do not fork)

| Asset | Role |
|-------|------|
| `vehicle/road_grade` | Per-frame grade (rise/run) in HDF5 |
| `control.lateral.grade_steering_damping_gain` | Smoothing alpha reduction under grade (`pid_controller`) |
| `tools/debug_visualizer/backend/triage_engine.py` | Grade vs pitch cross-checks |
| `tools/oscillation_attribution.py` | Pattern for HDF5 → JSON + recommendations |
| `tools/drive_summary_core.py` | Session metrics; extend with optional `grade_lateral` block |
| PhilViz Chain / Diagnostics tabs | Host new panels or reuse chart overlay |

---

## Milestones

### G6-L0 — Telemetry contract & “effective grade lateral” signals

**Goal:** Every recording intended for grade analysis has **explicit, documented** signals.

**Deliverables:**
1. **Schema** (`data/formats/data_format.py` + recorder):  
   - Confirm / add per-frame: `road_grade`, and if available `pitch_rad` (Unity).  
   - **Optional (recommended):** `control/grade_steering_smoothing_alpha_effective` or `control/lateral_grade_damping_factor` (0–1) — whatever the controller actually applies after `grade_steering_damping_gain`.  
   - Document units and sign convention in `docs/CONFIG_GUIDE.md` (grade: rise/run; downhill negative if that’s Unity’s convention — match `triage_engine`).

2. **Orchestrator / lateral:** Single function that computes **display factor** (for logging only if no new HDF5 field yet): e.g. `effective_steering_smoothing_alpha` given grade + config.

**Exit criteria:** New or existing fields written every frame on graded tracks; **no** behavior change required for L0 (instrumentation-only PR acceptable).

**Tests:** HDF5 contract test — minimal synthetic file with `road_grade` populated; loader sees fields.

---

### G6-L1 — Analytics module (`grade_lateral_analysis.py`)

**Goal:** Pure Python **recording → JSON** report: metrics **split by grade bin**, pre-failure optional.

**Suggested API:**

```python
def analyze_grade_lateral(
    recording_path: Path,
    *,
    pre_failure_only: bool = True,
    grade_bins: tuple = ("down", "flat", "up"),  # thresholds e.g. ±0.02
) -> dict:
    ...
```

**Metrics (v1):**
- Frame counts and % time per bin.
- Mean / p95 `|lateral_error|`, steering jerk proxy, `|d(lateral_error)/dt|` per bin.
- **Oscillation:** `oscillation_zero_crossing_rate_hz` and RMS slope **computed on segments restricted to each bin** (or full run with bin-masked frames — define one rule and stick to it).
- **Correlation summary:** e.g. “chatter proxy rate downhill vs not” (scalar).

**Exit criteria:** JSON schema version `grade_lateral_v1`; matches output from CLI and Flask.

**Tests:**  
- **Synthetic HDF5:** known grade pattern (step downhill mid-run) + synthetic lateral error; assert bin counts and conditional means.  
- **No Unity** required.

---

### G6-L2 — CLI + `drive_summary` / gate hooks

**Goal:** Same numbers in **terminal** and **batch reports**.

**Deliverables:**
1. **CLI:** `python tools/analyze_grade_lateral.py <recording.h5> [--json] [--pre-failure-only]`  
2. **Optional:** `analyze_recording_summary(..., include_grade_lateral=True)` or separate call merged in `executive_summary` / `system_health` — avoid slowing every summary if expensive; default **off** or **lazy**.

3. **`run_gate_and_triage` (optional):** New optional column `grade_lateral_downhill_chatter_rate` fail if > threshold (behind flag).

**Exit criteria:** Documented in `docs/OSCILLATION_ATTRIBUTION.md` sibling section or `docs/GRADE_LATERAL.md`.

**Tests:** Contract test on synthetic; snapshot of JSON keys.

---

### G6-L3 — PhilViz integration

**Goal:** Visual **direct answer** for “is lateral worse on grade?”

**Deliverables:**
1. **API:** `GET /api/recording/<file>/grade-lateral?pre_failure_only=true` → same JSON as CLI.
2. **UI (minimal):**  
   - **Diagnostics** or **Chain** sub-panel: table of bins × metrics + one **sparkline** or Chart.js series: `road_grade` vs frame index with vertical band shading (down / flat / up).  
   - Optional: link from **Oscillation Attribution** panel (“Open grade breakdown”).

3. **README:** `tools/debug_visualizer/README.md` bullet.

**Exit criteria:** Load `hill_highway` recording → panel shows non-empty bin stats; no console errors.

**Tests:** Optional `tests/test_philviz_grade_endpoint.py` mocking filesystem — or manual checklist only if Flask test harness is heavy (prefer lightweight route test).

---

### G6-L4 — Policy tuning (after L0–L3)

**Goal:** Change **`grade_steering_damping_gain`**, schedule, or PP feedback by grade **only with** L1 metrics proving improvement.

**Deliverables:** A/B protocol doc; before/after HDF5 pair; update `config/av_stack_config.yaml` with comments referencing `GRADE_LATERAL_PLAN.md`.

**Exit criteria:** Pre-failure oscillation / chatter metrics **improve in downhill bin** without regressing flat/uphill or overall score gate.

---

## Risk & scope control

- **Scope creep:** Do **not** implement full vehicle dynamics (load transfer model) in L0–L3 — stay **observability + attribution**.  
- **Banking (6b):** Out of scope for this plan until Stage 6b explicitly scheduled; document “grade only” vs “bank” in metrics when `roll` appears.

---

## Done checklist (program level)

- [x] L0: Telemetry contract documented + tests  
- [x] L1: `grade_lateral_v1` JSON + synthetic tests  
- [x] L2: CLI + optional drive_summary merge  
- [x] L3: PhilViz API + panel  
- [ ] L4: Policy change gated on L1 metrics — see `docs/GRADE_LATERAL_TUNING.md` (A/B protocol; no default change until metrics prove improvement)  
- [x] `docs/ROADMAP.md` markers updated when phases complete  

---

## Open questions

1. **Unity:** Is `pitch_rad` always reliable vs `road_grade`? (Triage already warns when they disagree — reuse.)  
2. **Bin thresholds:** Start with ±0.02 (match ad-hoc analysis); make YAML-configurable later.  
3. **MPC vs PP:** Regime splits — add `regime_blend` or `mpc_active` breakdown in v2 if needed.

---

## References

- `tracks/hill_highway.yml` — graded oval  
- `docs/OSCILLATION_ATTRIBUTION.md` — failure attribution policy  
- `tools/trim_recording_h5.py` — smaller fixtures for CI  
