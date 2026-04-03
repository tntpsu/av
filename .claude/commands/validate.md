Run acceptance gate checks for a recording or suite, and return a structured PASS/FAIL verdict.

The user's request is: $ARGUMENTS

## Step 1 — Determine what to validate

Parse $ARGUMENTS:
- If a `.h5` path is given → validate that specific recording
- If "latest" or no recording given → use latest recording (`ls -t data/recordings/*.h5 | head -1`)
- If `--suite <name>` → batch-validate all scenarios in that suite (see `.claude/docs/scenario_registry.md`)
- If `--gate comfort` → only run comfort gates
- If `--gate acc` → only run ACC gates
- If `--gate scoring` → only run scoring regression
- Default (no flag) → run all applicable gates

## Step 2 — Run the tests

### Always run Tier 1 comfort gate tests (fast, ~0.3s, no Unity):
```bash
pytest tests/test_comfort_gate_replay.py -v -k "Synthetic or Boundary"
```

### If `full` or `scoring` gate set:
```bash
pytest tests/test_scoring_regression.py -v
```

### If a live recording is available:
```bash
python3 tools/analyze/run_gate_and_triage.py <recording>
python3 tools/analyze/analyze_drive_overall.py <recording>
```

### If the recording is an ACC scenario (check: `vehicle/acc_active` field exists and has non-zero values):
```bash
python3 tools/analyze/acc_pipeline_analysis.py --file <recording>
```

## Step 3 — Extract gate values

From the `analyze_drive_overall.py` output, extract:
- Overall score
- Accel P95 (m/s²)
- Jerk P95 (m/s³)
- Lateral RMSE (m)
- Centered frames (%)
- Emergency stops (count)
- ACC: TTC min (s), Gap RMSE (m), Collision count, Radar detection rate (%)

## Step 4 — Evaluate against gates

### Standard comfort gates (from `tools/scoring_registry.py`):
- Accel P95 ≤ 3.0 m/s²
- Jerk P95 ≤ 6.0 m/s³
- Lateral P95 ≤ 0.40 m
- Centered frames ≥ 70%
- Emergency stops = 0

### ACC promotion gates:
- Collisions = 0
- TTC min ≥ 2.0s
- Gap RMSE ≤ 0.5m
- Jerk P95 ≤ 4.0 m/s³
- Radar detection rate ≥ 95%
- Lateral regression vs baseline ≤ 0.5 pts

### Scoring regression gates:
- Overall score within ±tolerance of frozen baseline
- Every layer score ≥ 60
- Trajectory layer ≥ 80

## Step 5 — Output validation report

```
VALIDATION REPORT — <recording or suite>
==========================================
OVERALL: PASS / FAIL

Comfort Gates:
  [PASS/FAIL] Accel P95:     <value> m/s²  / 3.0
  [PASS/FAIL] Jerk P95:      <value> m/s³  / 6.0
  [PASS/FAIL] Lateral P95:   <value> m     / 0.40
  [PASS/FAIL] Centered:      <value>%      / 70%
  [PASS/FAIL] E-stops:       <count>       / 0

ACC Gates (if applicable):
  [PASS/FAIL] Collisions:    <count>       / 0
  [PASS/FAIL] TTC min:       <value>s      / 2.0
  [PASS/FAIL] Gap RMSE:      <value>m      / 0.5
  [PASS/FAIL] Jerk P95 ACC:  <value> m/s³  / 4.0
  [PASS/FAIL] Radar det.:    <value>%      / 95%

Scoring:
  [PASS/FAIL] Overall score: <value>       / baseline ± tolerance
  [PASS/FAIL] Trajectory:    <value>       / ≥ 80

NEXT STEPS (if FAIL):
  - /diagnose <recording>   → identify root cause
  - /trace <event_type>     → inspect signal chain
  - /run --suite <suite>    → re-run after fix
```

Gate thresholds: `tools/scoring_registry.py`
Scenario registry: `.claude/docs/scenario_registry.md`
