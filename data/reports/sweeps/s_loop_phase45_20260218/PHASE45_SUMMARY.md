# s_loop Phase 4 + Phase 5 Completion Summary

## Scope completed
- Phase 4 Stage A (trajectory-locked screen)
- Phase 4 Stage B (canonical full-stack validation)
- Phase 5 (post-promotion robustness mini-set)

## Phase 4 Stage A: trajectory-locked screen

Runs:
- baseline candidate:
  - `phase4_stageA_baseline_off_summary.json`
  - `phase4_stageA_baseline_off_envelope.txt`
- promoted candidate:
  - `phase4_stageA_entry_schedule_soft_summary.json`
  - `phase4_stageA_entry_schedule_soft_envelope.txt`
- hard-case baseline/promoted pair:
  - `phase4_stageA_hard_baseline_off_*`
  - `phase4_stageA_hard_entry_schedule_soft_*`

Result:
- Stage A was non-discriminative for this setup (both candidates effectively tied on first-failure and envelope lines for replayed tracks).
- No candidate was rejected at Stage A; proceeded to Stage B.

## Phase 4 Stage B: canonical full-stack validation (`start_t=0.0`)

Artifact:
- `phase4_stageB_canonical_start0.txt`

A/B summary (`control.lateral.curve_entry_schedule_enabled`: A=`false`, B=`true`, repeats=2):
- A median first-failure frame: `186.0`
- B median first-failure frame: `223.5`
- A median time-to-failure: `9.892s`
- B median time-to-failure: `11.550s`
- B improved authority metrics (`authority_gap_mean`, `transfer_ratio_mean`) vs A.
- B had one `mixed-or-unclear` classification; one run remained `steering-authority-limited`.

Decision:
- Keep promoted policy (`entry_schedule_soft`) as baseline winner on canonical Stage-1 gate.

## Phase 5: post-promotion robustness mini-set

Artifacts:
- `phase5_robustness_start005.txt`
- `phase5_robustness_start010.txt`

Fixed-start mini-set:
- `start_t=0.05`: baseline outperformed promoted run in this single pair, both `perception-limited`.
- `start_t=0.10`: both A/B survived run (no first failure), both `perception-limited`.

Interpretation:
- Non-canonical robustness remains dominated by perception-limited regime, not cleanly control-authority-limited.
- This does not invalidate canonical Stage-1 promotion; it indicates upstream perception stabilization remains the limiting factor on alternate starts.

## Final status
- Phase 4: completed
- Phase 5: completed (mini robustness set executed and documented)
- Baseline remains:
  - `curve_entry_schedule_enabled: true`
  - `curve_entry_schedule_frames: 24`
  - `curve_entry_schedule_min_rate: 0.22`
  - `curve_entry_schedule_min_jerk: 0.16`
  - `curve_entry_schedule_min_hold_frames: 8`
