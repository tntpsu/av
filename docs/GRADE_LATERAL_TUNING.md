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

**Log:** 2026-03-22 — baseline nudged **6.5 → 7.5** pending next E2E `grade_lateral_v1` + gate comparison (G6-L4).

Related: **`base_error_smoothing_alpha`** — global lateral blend; changing it affects all road grades, not only grade bins.

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
