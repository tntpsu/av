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

Run tracks in this order with rebuilds between batches:

```bash
# Batch 1 (full rebuild): tracks 1-3
./start_av_stack.sh --duration 60 --track-yaml tracks/<track1>.yml
./start_av_stack.sh --skip-unity-build-if-clean --duration 60 --track-yaml tracks/<track2>.yml
./start_av_stack.sh --skip-unity-build-if-clean --duration 60 --track-yaml tracks/<track3>.yml

# Batch 2 (rebuild): tracks 4-6
./start_av_stack.sh --duration 60 --track-yaml tracks/<track4>.yml
./start_av_stack.sh --skip-unity-build-if-clean --duration 60 --track-yaml tracks/<track5>.yml
./start_av_stack.sh --skip-unity-build-if-clean --duration 60 --track-yaml tracks/<track6>.yml
```

After each run, verify recording > 500KB and > 200 frames before continuing.

**Guards:**
- s_loop is non-looping — use `--duration 60`
- **Unity health protocol (CRITICAL):**
  - Force a full player rebuild before the FIRST run (no `--skip-unity-build-if-clean` on first track)
  - After every 3 tracks, force another rebuild to prevent segfaults
  - If recording < 500KB or < 200 frames → Unity health issue → rebuild and retry
  - If 2 consecutive failures on same track → skip that track, note as "UNITY HEALTH FAIL"
  - Unity segfaults after ~36 consecutive launch/kill cycles — a full sweep of 6 tracks
    with retries could hit this limit. If you see exit code 139, stop and rebuild.

After the FIRST recording, verify config reached runtime:
```bash
python3 -c "import h5py,json; f=h5py.File('<recording>','r'); lat=json.loads(f['meta/runtime_config_json'][0]).get('control',{}).get('lateral',{}); print('formula_enabled:', lat.get('pp_curve_local_floor_formula_enabled')); print('profile_enabled:', lat.get('pp_steering_profile_enabled'))"
```
If either shows `None` or `False` when expected `True` → config wiring issue. Stop and fix before continuing.

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
