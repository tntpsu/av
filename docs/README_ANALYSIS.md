# README and TODO Analysis

**Date:** 2026-02-21  
**Purpose:** Accuracy check, layer capabilities, TODO alignment, and deduplication audit.

## Drive Summary Source Of Truth (Implemented)

- Canonical metric computation: `tools/drive_summary_core.py`
- PhilViz adapter: `tools/debug_visualizer/backend/summary_analyzer.py`
- CLI adapter: `tools/analyze/analyze_drive_overall.py`
- Contract: `docs/analysis/DRIVE_SUMMARY_CONTRACT.md`

---

## 1. README Accuracy Issues

### Control Layer — Outdated

| README Says | Actual State |
|-------------|--------------|
| "PID controller with feedforward (path curvature) + feedback (error correction)" | **Default is Pure Pursuit** (`control_mode: pure_pursuit` in config). PID and Stanley are available but PP is primary. |
| "control/pid_controller.py - Feedforward + feedback PID controller" | Same file hosts **LateralController** with modes: `pure_pursuit`, `pid`, `stanley`. PP bypasses PID-era jerk/smoothing/sign-flip. |

**Recommendation:** Update to state Pure Pursuit as default, with PID/Stanley as alternatives.

### Roadmap Section (Lines 374–391) — Inconsistent

| Item | Issue |
|------|-------|
| `[ ] ML-based perception models` | **Inaccurate.** Segmentation model is already ML-based (`--segmentation-checkpoint`). |
| Roadmap location | README has a mini-roadmap; **docs/ROADMAP.md** is the single source of truth per .cursorrules. README should defer to it. |

### Minor Inaccuracies

- **Bridge:** README says `bridge/server.py`; actual bridge lives under `bridge/` (correct).
- **Recording format:** README lists `vehicle_state/`; some recordings use `vehicle/` as well. Consider "vehicle state (position, speed, heading...)" as a generic description.
- **CONFIG_GUIDE:** References `CONFIG_GUIDE.md` at project root — correct.

---

## 2. Layer Capabilities / Methods — Gaps

README gives high-level descriptions but not a concrete "methods at each layer" section. **docs/ARCHITECTURE.md** has more detail but is also slightly behind on control.

### Perception

**README:** "Segmentation default with CV fallback (color masks, edge detection, Hough lines) and polynomial fitting"

**Missing:**
- Temporal filtering (EMA, jump detection)
- Low-visibility fallback (`low_visibility_fallback_*`)
- RANSAC polynomial fitting
- Lane center gate and lane-side change clamp

**Where it lives:** `perception/inference.py`, `perception/models/lane_detection.py`

### Trajectory

**README:** "Rule-based path planning with reference point smoothing and bias correction"

**Missing:**
- Dynamic lookahead (speed/curvature/confidence)
- Jerk-limited speed planner
- Curvature-based speed cap (`v_max = sqrt(a_lat/|κ|)`)
- Heading zero gate (straight-road classification)
- Multi-lookahead heading blend (optional)
- Vehicle-frame lookahead offset from Unity
- Reference-point smoothing with curvature-aware alpha and rate limiting

**Where it lives:** `trajectory/inference.py`, `trajectory/models/trajectory_planner.py`, `trajectory/utils.py`, `control/speed_governor.py`

### Control

**README:** "PID controller with feedforward + feedback"

**Missing:**
- **Pure Pursuit** as default lateral mode
- PP parameters: `pp_feedback_gain`, `pp_max_steering_jerk`, `pp_ref_jump_clamp`, `pp_stale_decay`, `pp_max_steering_rate`
- Curve feedforward scheduling and curvature preview blend
- Steering rate/jerk limits (PP bypass path)
- Longitudinal: jerk-limited speed tracking, accel/jerk caps

**Where it lives:** `control/pid_controller.py` (LateralController + longitudinal), `control/vehicle_model.py`

**Recommendation:** Add a short "Methods at each layer" subsection (or a pointer to ARCHITECTURE.md) that lists:
- Perception: segmentation + CV fallback, temporal filtering, RANSAC fitting
- Trajectory: rule-based planner, dynamic lookahead, jerk-limited speed planner
- Control: Pure Pursuit (default), PID/Stanley alternatives, curve feedforward, rate/jerk limits

---

## 3. TODO Accuracy and Duplication

### Three Overlapping Trackers

| Document | Role | Overlap |
|----------|------|---------|
| **README Roadmap** (374–391) | High-level milestones | Duplicates docs/ROADMAP.md |
| **docs/TODO.md** | Near-term, phase-by-phase tasks | Overlaps ROADMAP Stage 1 |
| **docs/ROADMAP.md** | Single source of truth (per .cursorrules) | Canonical |

### docs/TODO.md

**Strengths:**
- Detailed S1-M33, S1-M35, S1-M36, S1-M37, S1-M38, S1-M39 with clear completion markers
- Stack Isolation TODOs (Stage 0–6) are useful
- Promotion policy is explicit

**Problems:**
- **"Current Focus: S-Loop Turn Clearance (S1-M32)"** — S1-M32 items are marked completed; ROADMAP shows current focus as Stage 1, S1-M4 next
- **"Phase 1: Finish Lane-Keeping + Longitudinal Fidelity"** — Overlaps ROADMAP Layer 2 / Stage 1
- **"Recent Accomplishments (2026-02-10)"** — Perception/Control tuning is partially superseded by Pure Pursuit migration
- **Phase 2/3 work items** — Some are already implemented (e.g. curvature preview, speed cap)

### Duplicative Content

1. **README Roadmap vs docs/ROADMAP.md**  
   - README’s mini-roadmap repeats completed work (Unity setup, bridge, recorder, etc.) and future work (ML perception, MPC, etc.).  
   - **Action:** Replace README roadmap with a short summary and link to docs/ROADMAP.md.

2. ** docs/TODO.md vs docs/ROADMAP.md**  
   - Both describe Stage 1 and phase completion.  
   - **Action:** Treat ROADMAP as source of truth; TODO becomes a "near-term checklist" that links to ROADMAP stages and avoids redefining them.

3. **CONFIG_GUIDE.md**  
   - Example shows `control_mode: pid`; actual config uses `pure_pursuit`.  
   - **Action:** Update CONFIG_GUIDE to reflect current default and PP parameters.

---

## 4. Recommendations

### README

1. **Control:** Change "PID controller" to "Pure Pursuit (default) with PID/Stanley alternatives" and note feedforward + feedback.
2. **Roadmap:** Replace the detailed list with a brief status and link: *"See [docs/ROADMAP.md](docs/ROADMAP.md) for current stages and milestones."*
3. **ML perception:** Remove or rephrase "ML-based perception models" so it does not conflict with the existing segmentation model.
4. **Layer methods:** Add a subsection (or link to ARCHITECTURE) summarizing methods per layer (see Section 2 above).

### docs/TODO.md

1. **Current focus:** Align with docs/ROADMAP.md; remove or update "Current Focus: S-Loop Turn Clearance (S1-M32)".
2. **Recent Accomplishments:** Prune or mark as historical; keep only what still applies after PP migration.
3. **Phase 2/3:** Cross-check with ROADMAP and actual code; mark completed items as done.
4. **Single source of truth:** Add a short note at the top: *"For stage/phase status and promotion gates, see [ROADMAP.md](ROADMAP.md). This file tracks near-term checklists and Stack Isolation TODOs."*

### CONFIG_GUIDE.md

1. **Lateral control:** Update examples to use `control_mode: pure_pursuit` and document key PP parameters.

### docs/ARCHITECTURE.md

1. **Control:** Update the Control section to state Pure Pursuit as default and note PID/Stanley as alternative modes.

---

## 5. Document Hierarchy (Clarification)

| Document | Purpose | Keep Updated |
|----------|---------|--------------|
| **README.md** | Overview, quick start, high-level architecture | Yes |
| **docs/ROADMAP.md** | Stages, phases, promotion gates | **Single source of truth** |
| **docs/TODO.md** | Near-term tasks, Stack Isolation checklist | Yes; must align with ROADMAP |
| **docs/ARCHITECTURE.md** | Layer design, interfaces, signal chain | Yes |
| **CONFIG_GUIDE.md** | Config parameters, tuning | Yes |
