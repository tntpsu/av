# Grade–lateral policy tuning (G6-L4)

**Roadmap:** `docs/ROADMAP.md` — G6-L4  
**Plan:** `docs/plans/GRADE_LATERAL_PLAN.md` (L4 exit criteria)

G6-L0–L3 deliver **observability** (HDF5 telemetry, `grade_lateral_v1`, PhilViz). **G6-L4** is the phase where you **change** lateral behavior for grade — only with measured proof.

## Preconditions (do not skip)

1. **Baseline metrics** on the same track (e.g. `hill_highway`) using the same run profile:
   - `python tools/analyze_grade_lateral.py <before>.h5 --json`
   - Note `bins.down` vs `bins.flat` — `steering_zero_crossing_rate_hz_detrended`, `lateral_error_abs_p95_m`, etc.
   - Optional: `python tools/oscillation_attribution.py <before>.h5 --json` for oscillation subtype context.
2. **Pre-failure window** for attribution: use default `pre_failure_only` / `executive_summary.failure_frame` per project rules.
3. **Comfort / summary gates** unchanged or explicitly compared: `analyze_recording_summary` overall score and gate list.

## Primary tuning knob (config)

- **`control.lateral.grade_steering_damping_gain`** (`config/av_stack_config.yaml`)  
  Increases grade-proportional reduction of lateral error smoothing α (more smoothing on steeper grade).  
  **Increase** if downhill chatter is high; **decrease** if the car feels sluggish / under-reactive on grade.

**A/B log (hill_highway E2E):**

| Change | `recording_*.h5` | Result |
|--------|-------------------|--------|
| **6.5 → 7.5** | `20260322_123744` (before) vs `20260322_125614` (after) | **Fail promotion:** overall score ↓ (~95.2 → ~94.4), lateral p95 ↑, oscillation ZC ↑, steer jerk at cap; downhill \|lat\| improved slightly but **flat** lateral and oscillation **worsened**. **Reverted to 6.5.** |
| **0.075 → 0.070** (`pp_feedback_gain`) | `20260322_123744` (0.075) vs `20260322_131928` (0.070) | **Fail promotion:** overall score ↓ (~95.2 → ~94.4), **flat** `lateral_error_abs_p95_m` ↑ (~0.477 → ~0.511); downhill \|lat\| p95 improved slightly but downhill detrended ZC ↑; **reverted to 0.075.** |

Related: **`base_error_smoothing_alpha`** — global lateral blend; changing it affects all road grades, not only grade bins.

---

## How to robustly resolve (after failed grade-only bump)

**Observation:** Worst **flat** spans had **tiny** `|trajectory_ref_x − perception_lane_center|` — the problem is **not** “planner disagrees with lane center” in the recording. Oscillation attribution points to **control-loop** character with **perception** as the labeled primary layer (lead–lag + subtype). **Grade damping alone** is the wrong lever if regressions show up on **flat** and in **global oscillation**.

**Recommended sequence (one knob per A/B, same track + duration):**

1. **Flat / PP lateral (primary for this symptom)**  
   - **`pp_feedback_gain` 0.070** on `hill_highway` **did not promote** (see A/B table) — do **not** ship that step as default. Prefer next: small **`pp_max_steering_rate`** reduction, or a **micro-step** (e.g. 0.075 → 0.072) with the same gates — **re-run** `grade_lateral_v1` + `inspect_flat_focus_layers.py` + `executive_summary`.  
   - Goal: lower **flat-bin** \|lat\| p95 and **oscillation_zero_crossing_rate_hz** without new failures.

2. **Perception stability (if ref−perc stays small but error is large)**  
   - Temporal / jump logic on **`perception/inference`** (or lane smoothing) — **only** with evidence: high jump rate, or PhilViz frame review at **flat_focus** ranges.

3. **Regime / blend**  
   - If MPC or blend is active, add **regime-stratified** metrics (v2) or inspect **`regime_blend_weight`** / limiter activity at bad frames — avoid tuning grade damping until blend is clean.

4. **Grade damping (retry later)**  
   - Only after (1) stabilizes **flat** and overall score; try **small** steps (e.g. 6.5 → 6.8), **not** 6.5 → 7.5, and **gate** on **all** bins + score.

5. **Always**  
   - Pre-failure windows only for attribution; keep **PhilViz** flat-focus + `inspect_flat_focus_layers` **pair** for each candidate.

## A/B protocol (minimal)

| Step | Action |
|------|--------|
| A1 | Record `before.h5` on target track (same speed profile / duration as feasible). |
| A2 | Export `grade_lateral_v1` + (optional) oscillation JSON for `before.h5`. |
| B1 | Change **one** parameter (e.g. `grade_steering_damping_gain` ±10–20%). |
| B2 | Record `after.h5` under same conditions. |
| B3 | Export same JSON for `after.h5`. |
| C1 | **Pass** if: downhill-bin chatter / \|lateral\| metrics improve **and** flat/up bins and overall `executive_summary` score do not regress beyond agreed tolerance. |
| C2 | **Fail** if: improvement is only post-failure or ambiguous — fix instrumentation first (PhilViz / CLI). |

## Artifacts to keep

- Pair of HDF5 files (`before` / `after`) and the two `grade_lateral_v1` JSON blobs (or full CLI stdout).
- Short note: config diff (which key, old → new).

## Exit criteria (from plan)

Pre-failure oscillation / chatter metrics **improve in the downhill bin** without regressing flat/uphill or the overall score gate.

**No default YAML change** should be merged for G6-L4 until the above A/B is documented for that promotion.
