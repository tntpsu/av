# Oscillation attribution

**Purpose:** After a run (especially **grade / hill highway** segments), answer *what layer or mechanism* is driving steering–error oscillation and *what to tune first*.

## Outputs

- **Subtype:** `limiter_dominant`, `perception_jitter`, `reference_step`, `regime_blend`, `control_error_loop`, or `unclassified`.
- **Primary layer:** `perception` / `trajectory` / `control` (heuristic + subtype override).
- **Lead–lag (frames):** perception→reference, reference→error, error→steering (first-difference cross-correlation).
- **recommended_fixes:** prioritized list with `area` + `action` (optional **hill highway** co-tuning bullet when metadata or heuristics say so).

**`hill_highway` object in JSON:** `co_tuning_guidance`, `source` (`metadata_track` | `curve_fraction_heuristic` | `force_on` | `force_off` | `none`), `metadata_match`, `override`.

**Auto-tagging:** If `recording_provenance.track_id` is `hill_highway` (or the filename / notes contain `hill_highway`), the co-tuning guidance is enabled without flags. Matches `tracks/hill_highway.yml` (`name: hill_highway`).

Pre-failure analysis uses the same **`failure_frame`** as `drive_summary_core` (`executive_summary.failure_frame`).

## CLI

```bash
python tools/oscillation_attribution.py data/recordings/<run>.h5
# Full recording (not truncated at first failure):
python tools/oscillation_attribution.py data/recordings/<run>.h5 --full-run --json
# Force co-tuning bullet on/off (overrides metadata):
python tools/oscillation_attribution.py data/recordings/<run>.h5 --force-hill-highway
python tools/oscillation_attribution.py data/recordings/<run>.h5 --no-hill-highway
```

## PhilViz

1. Start the debug visualizer server (default recordings under `data/recordings/`).
2. Load a recording → **Chain** tab → **Oscillation Attribution** panel (same JSON as API).

API:

`GET /api/recording/<path>/oscillation-attribution?pre_failure_only=true`

Optional override: `hill_highway=auto` (default) | `true` | `false`

## Hill highway (what to fix first)

When **`hill_highway.co_tuning_guidance`** is true (from **metadata** or **curve heuristic**), the tool adds stack-level guidance to co-tune:

- **Curve / speed / feasibility** caps vs demand,
- **Steering rate/jerk / limiter** headroom,
- **Trajectory** curvature contract vs **control** authority.

If **subtype** is `limiter_dominant`, prioritize **control** limits and curve-entry assist; if `perception_jitter`, prioritize **perception** smoothing and stale handling; if `reference_step`, prioritize **trajectory** reference smoothing and jump clamps.

## Verification after config tuning (hill_highway)

1. Re-run Unity on **hill_highway** with updated `config/av_stack_config.yaml` and save a **new** HDF5.
2. **CI / local gate:** set `AV_HILL_HIGHWAY_OSCILLATION_REGRESSION_H5` to that file and run:
   ```bash
   AV_HILL_HIGHWAY_OSCILLATION_REGRESSION_H5=data/recordings/<new>.h5 pytest tests/test_oscillation_regression_metrics.py::test_post_fix_recording_no_oscillation_runaway -v
   ```
   Expect `oscillation_amplitude_runaway == False`.
3. **Synthetic regression** (always on): `tests/test_oscillation_regression_metrics.py` ensures constant-amplitude oscillation on curved tracks does **not** false-trigger runaway, and growing envelopes still do.

## Relation to Blame Trace

| Feature | Blame Trace (PhilViz) | Oscillation attribution |
|--------|------------------------|-------------------------|
| Basis | Layer **health score** drops | Limiter metrics, ref jumps, perc–ref residual, lead–lag |
| Question | Which layer degraded first? | What mechanism drives oscillation / what to tune? |

Use both: **Blame Trace** for onset ordering; **Oscillation Attribution** for oscillation-specific fixes.
