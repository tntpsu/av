Show the latest scores for each track, including layer breakdowns and comfort gates.

The user's request is: $ARGUMENTS

## Run the scoreboard script

```bash
python3 tools/analyze/scoreboard.py $ARGUMENTS
```

Arguments:
- No args → latest recording per track
- Track name (e.g. "s_loop") → that track's latest only
- Number (e.g. "3") → 3 most recent recordings
- Path to `.h5` file → that specific recording
- `--json` → machine-readable output

The script handles all extraction, formatting, and Pareto summary.

## If the user needs more detail

- `/pareto` — full cross-track analysis with root cause clustering
- `/diagnose <track>` — deep root cause analysis for a specific track
- `/e2e <track>` — re-run and re-score a track end-to-end

## References
- Scoreboard script: `tools/analyze/scoreboard.py`
- Scoring logic: `tools/drive_summary_core.py`
- Thresholds: `tools/scoring_registry.py`
- Track baselines: `tests/fixtures/scoring_baselines.json`
