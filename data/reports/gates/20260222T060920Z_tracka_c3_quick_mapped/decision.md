# Gate Decision

- decision: `reject`
- pass_fail: `False`
- generated_at_utc: `2026-02-22T06:09:20Z`

## Gate Checks
- hard_safety_pass: `True`
- run_count_gate_pass: `True`
- highway_non_regression: `True`
- sloop_first_300_improvement: `False`
- sloop_lateral_p95_non_regression: `True`

## Next Actions
- Improve trajectory/reference quality first (curve-entry continuity and authority).
- Run one isolated A/B pair and apply the strongest single attribution lever.

## Failed/Regressed Runs
- `recording_20260222_005653.h5` (s_loop): sloop_first_300_no_improvement, sloop_lateral_p95_regression
- `recording_20260222_005833.h5` (s_loop): sloop_first_300_no_improvement
