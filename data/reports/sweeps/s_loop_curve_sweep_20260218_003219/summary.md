# Curve-Gated Policy Sweep

Output dir: `/Users/philiptullai/Documents/Coding/av/data/reports/sweeps/s_loop_curve_sweep_20260218_003219`

## Ranking

1. `entry_schedule_soft` score=32.50 fail_med=232.5 emergency_pct=100.0 env_pass_pct=0.0 jerk_p95_med=0.000
2. `entry_commit_aggressive` score=-8.50 fail_med=191.5 emergency_pct=100.0 env_pass_pct=0.0 jerk_p95_med=0.000
3. `entry_commit_balanced` score=-9.00 fail_med=191.0 emergency_pct=100.0 env_pass_pct=0.0 jerk_p95_med=0.000
4. `baseline_off` score=-11.00 fail_med=189.0 emergency_pct=100.0 env_pass_pct=0.0 jerk_p95_med=0.000

## Post-run calibration note

- Envelope pass percentages above are from pre-calibration thresholds.
- After calibrated envelope re-scoring (see `RUN_MANIFEST.md`), `entry_schedule_soft` remains rank-1 and shows `50.0%` phase pass rate.
