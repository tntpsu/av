# Drive Summary Contract

## Canonical Source

- Metric computation source of truth: `tools/drive_summary_core.py`
- PhilViz adapter: `tools/debug_visualizer/backend/summary_analyzer.py`
- CLI adapter: `tools/analyze/analyze_drive_overall.py`

Both adapters must call the canonical module and must not implement independent metric formulas.

## Schema Versioning

- Top-level field: `summary_schema_version`
- Current version: `v1`
- Any breaking response change requires a version bump and compatibility notes.

## Required Top-Level Keys (`v1`)

- `summary_schema_version`
- `executive_summary`
- `path_tracking`
- `layer_scores`
- `layer_score_breakdown`
- `control_mode`
- `control_smoothness`
- `speed_control`
- `comfort`
- `control_stability`
- `perception_quality`
- `trajectory_quality`
- `turn_bias`
- `alignment_summary`
- `latency_sync`
- `system_health`
- `safety`
- `recommendations`
- `config`
- `time_series`

## Behavioral Rules

- `analyze_to_failure=false`: analyze entire run.
- `analyze_to_failure=true`: truncate arrays at first sustained failure frame.
- Out-of-lane detection prefers ground-truth boundaries when present, then falls back to center/error signals.
- Sustained out-of-lane events require consecutive-frame thresholds (single-frame spikes are ignored).
- Missing datasets must degrade gracefully to `0.0`, `None`, or empty collections per field semantics (no hard crash).

## Added Diagnostic Fields (`v1`)

- `control_smoothness.oscillation_zero_crossing_rate_hz`
- `control_smoothness.oscillation_rms_growth_slope_mps`
- `control_smoothness.oscillation_rms_window_start_m`
- `control_smoothness.oscillation_rms_window_end_m`
- `control_smoothness.oscillation_rms_window_p95_m`
- `control_smoothness.oscillation_rms_windows_count`
- `control_smoothness.oscillation_amplitude_runaway`
- `safety.out_of_lane_events_full_run`
- `safety.out_of_lane_time_full_run`
- `safety.out_of_lane_event_at_failure_boundary`
- `latency_sync.e2e`
- `latency_sync.sync_alignment`
- `latency_sync.overall`
- `comfort.metric_roles`
- `comfort.hotspot_attribution`

## Comfort Metric Semantics

- `comfort.commanded_jerk_p95`: command-domain gate metric derived from throttle/brake command derivatives.
- `comfort.acceleration_p95_filtered`: gate metric for longitudinal acceleration comfort.
- `comfort.jerk_p95_filtered`: measured outcome-domain diagnostic (filtered speed derivative).
- `comfort.jerk_p95`: measured outcome-domain raw diagnostic (unfiltered speed derivative).
- `comfort.hotspot_attribution`: top-N longitudinal hotspots with root-cause attribution labels.

## Adapter Responsibilities

- CLI adapter formats canonical summary for terminal output.
- PhilViz adapter returns canonical summary as JSON.
- Neither adapter may re-derive metrics independently.
