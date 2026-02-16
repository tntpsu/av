# Phase 0 Readiness Gate: Dynamic Effective Horizon

This document defines the execution contract before implementing Phase 1 (`dynamic_effective_horizon_m` diagnostics-only).

## 1) Baseline Freeze

- Baseline config source: `config/av_stack_config.yaml`
- Baseline metrics source: `tmp/analysis/trajectory_step_baseline_metrics.json`
- Baseline recording reference: `data/recordings/recording_20260215_180158.h5`

Pinned trajectory settings for Phase 0 baseline:

- `trajectory.trajectory_eval_y_min_ratio: 0.10`
- `trajectory.centerline_midpoint_mode: coeff_average`
- `trajectory.projection_model: legacy`
- `trajectory.multi_lookahead_curve_heading_smoothing_alpha: 0.75`
- `trajectory.x_clip_limit_m: 15.0`
- `trajectory.reference_lookahead: 9.0`
- `trajectory.dynamic_reference_lookahead: true`
- `trajectory.lookahead_distance: 20.0`

## 2) Scenario Matrix (Must Pass as a Set)

- Tight curves: `tracks/s_loop.yml` (radius 40m arcs), 45s
- Moderate curves: `tracks/oval.yml` (radius 50m arcs), 45s
- Sustained curve: `tracks/circle.yml` (continuous right arc), 45s
- Straights + curve entries: `tracks/s_loop.yml` straights and straight-to-arc transitions, 45s

## 3) Promotion Contract

Required improvements:

- `8-12m` mean lateral error: candidate <= baseline * 0.95
- `12-20m` mean lateral error: candidate <= baseline * 0.95

No-regression requirements:

- `0-8m` mean lateral error: candidate <= baseline + 0.02m
- Underturn rate: candidate <= baseline + 0.02
- Turn-authority ratio10 mean: candidate >= baseline - 0.05

Temporal stability requirements:

- Horizon-switch event rate <= 0.10 (once added in Phase 1 diagnostics)
- `diag_smoothing_jump_reject_rate` <= baseline + 0.02
- `diag_ref_x_rate_limit_active_rate` <= baseline + 0.03

Rollback triggers:

- Gate fails in 2 or more scenario rows
- Any scenario has near-field (`0-8m`) regression > 0.03m
- Any scenario has underturn-rate regression > 0.03
- New diagnostics do not localize change by layer and in-layer location

## 4) A/B Execution Contract

- Run length: 45s
- One behavior change per runset
- Required analyzers:
  - `tools/analyze/analyze_trajectory_layer_localization.py`
  - `tools/analyze/analyze_trajectory_turn_authority_root_cause.py`

## 5) Phase 2 Hard Checkpoint Reminder

After Phase 2 completes, stop and decide:

- Continue incremental path (Phase 3) only if improvements are stable and diagnostically explainable.
- Escalate to stronger architecture changes if improvements are inconsistent or regressions persist.
- Roll back immediately if safety/stability gates fail.

## 6) Machine-Readable Companion

All details above are mirrored in:

- `tmp/analysis/phase0_readiness_package.json`
