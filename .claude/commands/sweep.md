Run e2e on all tracks (or a subset) and compare against baselines to detect regressions.

The user's request is: $ARGUMENTS

## Step 1 — Parse arguments

- No args → run all 6 tracks: s_loop, highway_65, hairpin_15, sweeping_highway, mixed_radius, hill_highway
- Track names → run only those (e.g., "s_loop mixed_radius")
- "--quick" → skip builds, just analyze latest recordings per track
- "--dry-run" → show what would run without executing

## Step 2 — Read baselines

```bash
cat tests/fixtures/scoring_baselines.json
```

Extract per-track baseline scores.

## Step 3 — Run e2e on each track

For each track, run:
```bash
./start_av_stack.sh --build-unity-player --skip-unity-build-if-clean --run-unity-player --duration 60 --track-yaml tracks/<track>.yml
```

**Guards:**
- s_loop is non-looping — use `--duration 60`
- After 6 consecutive runs, force rebuild (omit `--skip-unity-build-if-clean`)
- If recording < 500KB → Unity health issue, rebuild and retry (don't count)

After each run, analyze:
```bash
python3 tools/analyze/analyze_drive_overall.py <recording>
```

## Step 4 — Compare against baselines

For each track, extract:
- Overall score
- Layer scores (Safety, Trajectory, Control, Perception, LongitudinalComfort, SignalIntegrity)
- E-stop count
- Key issues

Compare against baselines from `tests/fixtures/scoring_baselines.json`.

## Step 5 — Gate evaluation

**PASS criteria (ALL must hold):**
- No track dropped > 2 pts overall vs baseline
- No new e-stops (that weren't in baseline)
- No layer dropped below 60 on any track

**Flag criteria (warnings, not failures):**
- Any track dropped > 1 pt overall
- Any layer dropped below 95 that was previously ≥ 95

## Step 6 — Output

```
CROSS-TRACK SWEEP
═══════════════════════════════════════════════════════════════
Track            Baseline    Now     Delta    Gate
s_loop           <base>      <now>   <delta>  PASS/FAIL/FLAG
highway_65       <base>      <now>   <delta>  PASS/FAIL/FLAG
mixed_radius     <base>      <now>   <delta>  PASS/FAIL/FLAG
sweeping_highway <base>      <now>   <delta>  PASS/FAIL/FLAG
hairpin_15       <base>      <now>   <delta>  PASS/FAIL/FLAG
hill_highway     <base>      <now>   <delta>  PASS/FAIL/FLAG
═══════════════════════════════════════════════════════════════
GATE: PASS / FAIL

Regressions (> 2 pts): <list or "none">
New e-stops: <list or "none">
Flags (> 1 pt): <list or "none">

If FAIL:
  → /diagnose <regressed_track> to investigate
  → Revert changes before promoting

If PASS:
  → Safe to commit and push
  → Run /log-fix to record any fixes applied
```

## References
- Baselines: `tests/fixtures/scoring_baselines.json`
- Track YAMLs: `tracks/*.yml`
- Analysis: `tools/analyze/analyze_drive_overall.py`
- Gate thresholds: `tools/scoring_registry.py`
