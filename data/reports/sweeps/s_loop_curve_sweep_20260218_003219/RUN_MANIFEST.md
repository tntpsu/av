# s_loop Curve Tuning Run Manifest

## Canonical sweep settings
- track: `tracks/s_loop.yml`
- start policy: fixed `start_t=0.0`
- run duration: `30s`
- fail-fast: `safety.emergency_stop_end_run_after_seconds=0.5`
- repeats: `2` per candidate

## Envelope calibration (post-run)
- envelope file: `tools/analyze/curve_authority_envelope.yaml`
- calibrated to separate clearly failing runs from best observed run without forcing all-zero pass outcomes.

## Candidate ranking (validated, calibrated envelope)
1. `entry_schedule_soft`
   - median first-failure frame: `232.5`
   - envelope phase pass rate: `50.0%`
2. `entry_commit_aggressive`
   - median first-failure frame: `191.5`
   - envelope phase pass rate: `0.0%`
3. `entry_commit_balanced`
   - median first-failure frame: `191.0`
   - envelope phase pass rate: `0.0%`
4. `baseline_off`
   - median first-failure frame: `189.0`
   - envelope phase pass rate: `0.0%`

## Promotion decision
- promoted candidate: `entry_schedule_soft`
- baseline config changes applied in `config/av_stack_config.yaml`:
  - `curve_entry_schedule_enabled: true`
  - `curve_entry_schedule_frames: 24`
  - `curve_entry_schedule_min_rate: 0.22`
  - `curve_entry_schedule_min_jerk: 0.16`
  - `curve_entry_schedule_min_hold_frames: 8`
  - `curve_commit_mode_enabled: false` (unchanged)

## Failure-packet artifacts
- best candidate representative:
  - `failure_packet_entry_schedule_soft/packet.json`
  - `failure_packet_entry_schedule_soft/window.csv`
- baseline representative:
  - `failure_packet_baseline_off/packet.json`
  - `failure_packet_baseline_off/window.csv`
