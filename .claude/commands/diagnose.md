Run a targeted diagnostic sequence for the latest (or specified) AV stack recording. Determines the primary issue category and runs the minimal correct tool sequence — no more guessing which of 103 tools to use.

The user's request is: $ARGUMENTS

## Step 1 — Identify recording

If $ARGUMENTS contains a file path (ends in `.h5`), use that file.
Otherwise use `--latest`.

## Step 2 — Always run the primary analysis first

```bash
python3 tools/analyze/analyze_drive_overall.py --latest
```
(or `python3 tools/analyze/analyze_drive_overall.py <path>` if a file was specified)

## Step 3 — Classify the primary issue

Parse the output to identify the issue category. Use **first match** from this priority list:

| Priority | Category | Trigger condition |
|----------|----------|-------------------|
| 1 | `safety_issue` | Emergency stops > 0, OR OOL time > 0% |
| 2 | `acc_issue` | ACC section present AND (Gap RMSE > 0.5m OR TTC < 2.0s OR collision > 0 OR detection rate < 95%) |
| 3 | `mpc_issue` | MPC fallback rate > 0.5% OR LMPC feasibility < 99.5% |
| 4 | `lateral_error_issue` | Lateral RMSE > 0.40m OR OOL events > 0 |
| 5 | `comfort_issue` | Jerk P95 > 6.0 OR Accel P95 > 3.0 |
| 6 | `perception_issue` | Perception detection rate < 80% OR stale rate > 5% |
| 7 | `trajectory_issue` | Trajectory layer score < 80 |
| 8 | `control_issue` | Control layer score < 80 |
| 9 | `unknown` | Overall score < 90 but no specific layer clearly red |

## Step 4 — Run the targeted tools

### If `safety_issue`:
```bash
python3 tools/analyze/build_failure_packet.py <recording>
python3 tools/analyze/run_gate_and_triage.py <recording>
```

### If `acc_issue`:
```bash
python3 tools/analyze/acc_pipeline_analysis.py --latest
```
Then suggest: `/trace brake_onset` and `/trace speed_drop` to inspect the signal chain.

### If `mpc_issue`:
```bash
python3 tools/analyze/mpc_pipeline_analysis.py --latest
```
Then suggest: `/trace regime_transition` to inspect PP↔MPC switch conditions.

### If `lateral_error_issue`:
```bash
python3 tools/analyze/build_failure_packet.py <recording>
python3 tools/analyze/analyze_phase_to_failure.py <recording>
python3 tools/analyze/counterfactual_layer_swap.py <recording>
```

### If `comfort_issue` (and NOT lateral_error_issue):
```bash
python3 tools/analyze/analyze_oscillation_root_cause.py <recording>
```
Then suggest: `/trace speed_drop` or `/trace brake_onset`.

### If `perception_issue`:
```bash
python3 tools/analyze/analyze_perception_questions.py <recording>
```

### If `trajectory_issue` or `control_issue`:
```bash
python3 tools/analyze/analyze_phase_to_failure.py <recording>
python3 tools/analyze/counterfactual_layer_swap.py <recording>
```

### If `unknown`:
```bash
python3 tools/analyze/run_gate_and_triage.py <recording>
python3 tools/analyze/counterfactual_layer_swap.py <recording>
```
And suggest opening PhilViz: `python3 tools/debug_visualizer/server.py` → Triage + Blame tabs.

## Step 5 — Synthesize and output

Present a structured summary:

```
DIAGNOSTIC SUMMARY
==================
Recording: <file>
Overall Score: <score>/100

PRIMARY ISSUE: <category>
  Symptom: <what the metric shows>
  Root cause hint: <what the tool output suggests>

RECOMMENDED NEXT STEPS:
  1. <specific action — e.g. "/trace brake_onset to see reference_velocity at brake frames">
  2. <second action>
  3. <third action if needed>

A/B TEST NEEDED: yes/no (yes if a config change is being considered)
```

Gate thresholds: `tools/scoring_registry.py`
Tool decision tree: `.claude/docs/tool_selection_guide.md`
Field reference: `.claude/docs/hdf5_field_reference.md`
