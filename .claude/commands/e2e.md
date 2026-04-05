End-to-end: build, run, analyze, and validate a scenario in one shot. Outputs a structured verdict and tells you exactly what to feed into `/diagnose` if anything fails.

The user's request is: $ARGUMENTS

## Step 1 — Resolve scenario (same as /run)

Parse $ARGUMENTS using the alias table in `.claude/docs/scenario_registry.md`. Accept fuzzy names:

| If user says... | Map to alias |
|----------------|--------------|
| "h2", "steady", "highway steady" | `h2_steady` |
| "h3", "brake", "hard brake" | `h3_brake` |
| "h4", "accel away" | `h4_accel` |
| "h5", "stop go" | `h5_stopgo` |
| "h6", "close gap" | `h6_close` |
| "h7", "catchup" | `h7_catchup` |
| "h8", "curve catchup" | `h8_curve` |
| "a1", "autobahn steady" | `a1_steady` |
| "a2", "autobahn brake" | `a2_brake` |
| "g1", "grade following", "hill following" | `g1_grade` |
| "g2", "stop on grade" | `g2_stop` |
| "highway", "hwy", "65" | `highway` |
| "sloop", "s loop", "s_loop" | `s_loop` |
| "mixed", "mixed radius" | `mixed` |
| "hill", "hill highway" | `hill` |
| "autobahn", "high speed" | `autobahn` |
| "hairpin" | `hairpin` |

**Config override:** If user says "with <config>" or "using <config>", use that config overlay instead of the registry default.

If alias not found, list 3 closest and ask.

## Step 2 — Build and run

Look up the scenario in `.claude/docs/scenario_registry.md` for config overlay, track YAML, and duration.

Run:
```bash
./start_av_stack.sh --build-unity-player --skip-unity-build-if-clean \
  --config config/<overlay>.yaml \
  --track-yaml tracks/<track>.yml \
  --duration <seconds>
```

Special cases:
- `hill` track with no explicit config override: **omit `--config`** (auto-derived q_lat)
- If user provides a config override, use it: `--config config/<user_config>.yaml`
- **NEVER omit `--track-yaml`** — car falls through road without it

Wait for the run to complete.

**Crash prevention (Unity stability):**
Unity degrades after ~36 consecutive launch/kill cycles. To prevent segfaults:
- If this is part of a multi-track validation sweep (e.g., regression testing 5+ tracks), **force a Unity player rebuild every 6 runs** by omitting `--skip-unity-build-if-clean` on every 6th invocation.
- If the run fails with a segfault (exit code 139 or "Segmentation fault" in output), **force rebuild and retry once**:
  ```bash
  ./start_av_stack.sh --build-unity-player \
    --config config/<overlay>.yaml \
    --track-yaml tracks/<track>.yml \
    --duration <seconds>
  ```
- If a recording is produced but is < 500KB, treat it as a failed run (empty stub from crashed Unity). Delete it and retry with forced rebuild.
- When running A/B batches (`run_ab_batch.py` with ≥5 runs), note that the batch script handles its own launches — warn the user if total runs across all tracks in the session exceeds 30.

If the run fails (non-zero exit) after retry, report the error and stop.

## Step 3 — Analyze

Run ALL of these in sequence:
```bash
python3 tools/analyze/analyze_drive_overall.py --latest
```

If the scenario is an ACC scenario (h2-h8, a1-a2, g1-g2), also run:
```bash
python3 tools/analyze/acc_pipeline_analysis.py --latest
```

If MPC is expected to be active (any track with MPC config, or hill_highway with dynamic model), also run:
```bash
python3 tools/analyze/mpc_pipeline_analysis.py --latest
```

## Step 4 — Extract metrics and evaluate gates

From the analysis output, extract and evaluate against gates from `tools/scoring_registry.py`:

### Standard comfort gates:
- Accel P95 <= 3.0 m/s^2
- Jerk P95 <= 6.0 m/s^3
- Lateral RMSE <= 0.40 m
- Centered frames >= 70%
- Emergency stops = 0

### ACC gates (if ACC scenario):
- Collisions = 0
- TTC min >= 2.0s
- Gap RMSE <= 35m
- Radar detection >= 95%

### Scoring gates:
- Overall score (note vs baseline if available in `tests/fixtures/scoring_baselines.json`)
- Every layer score >= `LAYER_SCORE_MIN_PASS` from `tools/scoring_registry.py` (currently 95)
- Read the current value from scoring_registry.py — do not hardcode

## Step 5 — Output structured report

```
E2E REPORT — <scenario alias>
============================================================
Recording: <filename>
Overall Score: <score>/100

COMFORT GATES:
  [PASS/FAIL] Accel P95:     <value> m/s^2  (gate: 3.0)
  [PASS/FAIL] Jerk P95:      <value> m/s^3  (gate: 6.0)
  [PASS/FAIL] Lateral RMSE:  <value> m      (gate: 0.40)
  [PASS/FAIL] Centered:      <value>%       (gate: 70%)
  [PASS/FAIL] E-stops:       <count>        (gate: 0)

ACC GATES (if applicable):
  [PASS/FAIL] Collisions:    <count>        (gate: 0)
  [PASS/FAIL] TTC min:       <value>s       (gate: 2.0)
  [PASS/FAIL] Gap RMSE:      <value>m       (gate: 35)
  [PASS/FAIL] Radar det.:    <value>%       (gate: 95%)

LAYER SCORES:
  Perception:  <score>  [OK/WARN/RED]
  Trajectory:  <score>  [OK/WARN/RED]
  Control:     <score>  [OK/WARN/RED]

MPC/DYNAMIC MODEL (if applicable):
  EKF source:       <imu/derived>
  EKF updates:      <count>
  C_f / C_r:        <values>
  Innovation P95:   <value> rad/s

VERDICT: PASS / FAIL
============================================================
```

## Step 6 — Overlay Dependency Audit (on FAIL)

If VERDICT is FAIL and the scenario uses a config overlay, check whether the overlay contains parameters that work around a base-config or code limitation:

1. Read the config overlay used for this scenario
2. For each non-geometry parameter in the overlay (anything other than track YAML, duration, speed limits derived from track geometry), check if ≥2 other overlays in `config/mpc_*.yaml` override the same parameter or subsystem
3. If ≥2 overlays work around the same subsystem, flag it as an architecture smell

```
OVERLAY AUDIT:
  Overlay: config/<overlay>.yaml
  Workaround params: <list of params that work around base config/code limitations>
  Same subsystem overrides in other overlays: <count> (<list>)
  Architecture smell: yes/no
  → If yes: <subsystem> has <N> track-specific workarounds — architecture review needed before tuning
```

If architecture smell is detected, recommend `/plan-feature` for an architecture fix rather than further tuning.

## Step 7 — Diagnose recommendation

If VERDICT is FAIL, classify the primary failure using the same priority table as `/diagnose`:

| Priority | Category | Trigger |
|----------|----------|---------|
| 1 | safety | E-stops > 0 or OOL > 0% |
| 2 | acc | ACC gate failures |
| 3 | mpc | MPC fallback > 0.5% |
| 4 | lateral | Lateral RMSE > 0.40 |
| 5 | comfort | Jerk or accel gate fail |
| 6 | perception | Detection < 80% |
| 7 | trajectory | Trajectory layer < LAYER_SCORE_MIN_PASS |
| 8 | control | Control layer < LAYER_SCORE_MIN_PASS |

Then output:

```
NEXT STEP:
  /diagnose                          <-- run this for full root cause analysis
  Primary issue category: <category>
  Key metric to investigate: <metric_name> = <value> (gate: <threshold>)
```

If VERDICT is PASS:

```
ALL GATES PASSED - no /diagnose needed.
Promotion checklist:
  [ ] Run regression on other tracks: /e2e highway, /e2e sloop
  [ ] Update baseline if score improved: tests/fixtures/scoring_baselines.json
  [ ] Update docs: docs/agent/current_state.md
```

---

Gate thresholds: `tools/scoring_registry.py`
Scenario registry: `.claude/docs/scenario_registry.md`
