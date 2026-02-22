# Gate Decision

- decision: `reject`
- pass_fail: `False`
- generated_at_utc: `2026-02-22T17:32:24Z`

## Gate Checks
- run_validity_pass: `True`
- hard_safety_pass: `False`
- run_count_gate_pass: `True`
- highway_non_regression: `None`
- sloop_first_300_improvement: `False`
- sloop_lateral_p95_non_regression: `False`

## Next Actions
- Run one isolated A/B pair and apply the strongest single attribution lever.
- Improve trajectory/reference quality first (curve-entry continuity and authority).

## Failed/Regressed Runs
- `recording_20260222_122858.h5` (s_loop): emergency_stop, out_of_lane, sloop_first_300_no_improvement, sloop_lateral_p95_regression
- `recording_20260222_123040.h5` (s_loop): emergency_stop, out_of_lane
