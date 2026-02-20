# Stage 1 Tuning Sweeps - 2026-02-20

All sweeps used canonical `start_t=0.0` on `tracks/s_loop.yml` via `tools/analyze/run_ab_batch.py`.

## Perception Sweep

- Param: `perception.low_visibility_fallback_max_consecutive_frames`
- A/B: `8` vs `6` (`repeats=1`, `duration=30s`)
- Result:
  - stale percentage unchanged (`29.51%` vs `29.51%`)
  - B regressed failure timing (`23.58s` -> `19.15s`)
- Decision: keep `8` (no promotion).

## Trajectory Sweep

- Param: `trajectory.far_band_contribution_cap_gain`
- A/B: `0.35` vs `0.45` (`repeats=1`, `duration=30s`)
- Result:
  - authority gap improved (`0.0420` -> `0.0106`)
  - transfer ratio improved (`0.7743` -> `0.9812`)
  - stale percentage improved (`30.00%` -> `23.08%`)
  - B had no failure in run window.
- Decision: promote `far_band_contribution_cap_gain=0.45`.

## Control Sweep

- Param: `control.lateral.steering_jerk_limit`
- A/B: `0.12` vs `0.10` (`repeats=1`, `duration=30s`)
- Result:
  - B regressed (`A: no failure`, `B: failure @ 19.35s`)
  - authority and stale also worsened with B.
- Decision: keep `0.12` (no promotion).

## Gate Evidence

- Canonical repeated A/B package (`repeats=3`) executed through:
  - `tools/analyze/run_e2e_promotion_suite.py --run-ab-repeat3`
- Report:
  - `data/reports/e2e_promotion_suite_report.json`
- Baseline freeze/consistency:
  - `data/reports/baseline_freeze_report.json`
