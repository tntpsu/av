Show the latest scores for each track, including layer breakdowns and comfort gates.

The user's request is: $ARGUMENTS

## Step 1 — Find recordings

Find all `.h5` recordings in `data/recordings/`, sorted newest-first.

Parse $ARGUMENTS:
- If a `.h5` path is given → show scores for only that recording
- If a track name is given (e.g. "s_loop", "highway_65") → show only that track's latest
- If "latest" or blank → show the latest recording per track (one per track_id)
- If a number N is given (e.g. "3") → show the N most recent recordings regardless of track

## Step 2 — Extract scores

Run a Python script using `tools/drive_summary_core.py::analyze_recording_summary()` to extract scores from each recording.

```bash
python3 -c "
import sys, json
sys.path.insert(0, '.')
from pathlib import Path
from tools.drive_summary_core import analyze_recording_summary

recordings_dir = Path('data/recordings')
recordings = sorted(recordings_dir.glob('*.h5'), key=lambda p: p.stat().st_mtime, reverse=True)

# Collect latest per track (or all, depending on args)
seen_tracks = set()
results = []
for rec in recordings[:50]:  # scan up to 50 recent files
    try:
        summary = analyze_recording_summary(rec, analyze_to_failure=True)
        track_id = (summary.get('run_intent') or {}).get('track_id') or 'unknown'
        if track_id in seen_tracks:
            continue
        seen_tracks.add(track_id)
        results.append({
            'file': rec.name,
            'track_id': track_id,
            'overall_score': summary.get('executive_summary', {}).get('overall_score'),
            'layer_scores': summary.get('layer_scores', {}),
            'layer_breakdown': summary.get('layer_score_breakdown', {}),
            'comfort': summary.get('comfort', {}),
            'key_issues': summary.get('executive_summary', {}).get('key_issues', []),
            'cap_reason': summary.get('executive_summary', {}).get('score_breakdown', {}).get('cap_reason', 'none'),
            'failure_detected': summary.get('executive_summary', {}).get('failure_detected', False),
        })
    except Exception as e:
        pass

print(json.dumps(results, indent=2, default=str))
"
```

Adjust the Python script based on $ARGUMENTS (filter by track, limit count, etc.).

## Step 3 — Present the scoreboard

Format the output as a clear table. Use this structure:

```
TRACK SCOREBOARD
════════════════════════════════════════════════════════════════════

Track: <track_id>  │  Recording: <filename>
Overall: <score>/100  <PASS if ≥95 else NEEDS WORK>  <cap reason if any>

  Layer Scores:
    Safety              <score>  <✓ if ≥95, ✗ if <95>
    Trajectory          <score>  <✓ if ≥95, ✗ if <95>
    Control             <score>  <✓ if ≥95, ✗ if <95>
    Perception          <score>  <✓ if ≥95, ✗ if <95>
    LongitudinalComfort <score>  <✓ if ≥95, ✗ if <95>
    SignalIntegrity     <score>  <✓ if ≥95, ✗ if <95>

  Curve Tracking (if Trajectory < 95):
    C1: onset=+<N>fr, peak|lat|=<X>m, Ld_min=<X>m, floor_rescue=<X>m
    BLAME: <from trace_curve_entry.py if available>

  Top Deductions (if any layer <95, show the largest deductions for that layer):
    <layer>: <deduction_name> -<value>pts

  Key Issues: <list if any>
────────────────────────────────────────────────────────────────────
(repeat for each track)

SUMMARY: <N>/<total> tracks passing (all layers ≥ 95)
```

**Layer pass threshold: 95** (per project standard in `tools/scoring_registry.py`).

Show tracks sorted by overall score ascending (worst first) so problems are immediately visible.

If no recordings are found, tell the user and suggest running the stack:
```
./start_av_stack.sh --unity-auto-play --duration 60
```

## Step 4 — Append Pareto summary

After the scoreboard, append a condensed Pareto showing the top 3 highest-ROI fixes.
Use the same scoring data already extracted in Step 2 — aggregate all deductions across tracks,
sort by total points lost, and show:

```
TOP 3 HIGHEST-ROI FIXES (from /pareto)
════════════════════════════════════════
1. <deduction_name> — <Σ pts> across <N> tracks → <suggested fix>
2. <deduction_name> — <Σ pts> across <N> tracks → <suggested fix>
3. <deduction_name> — <Σ pts> across <N> tracks → <suggested fix>

Run /pareto for full cross-track analysis with root cause clustering.
```

## References
- Scoring logic: `tools/drive_summary_core.py`
- Thresholds: `tools/scoring_registry.py`
- Track baselines: `tests/fixtures/scoring_baselines.json`
- Pareto analysis: `/pareto` skill
- Process health: `/process-health` skill
