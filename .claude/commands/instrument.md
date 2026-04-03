Add or remove named debug logging probes in `control/pid_controller.py` or `av_stack/orchestrator.py`. Probes are zero-cost when disabled — guarded by a config flag.

The user's request is: $ARGUMENTS

## Step 1 — Parse the action

From $ARGUMENTS, determine:
- **ACTION**: `add` or `remove` or `list`
- **PROBE NAME**: from the standard list below, or "custom"
- **Extra details** (for custom): file, function, signals

## Standard Named Probes

| Name | File | What it captures |
|------|------|-----------------|
| `brake_onset` | `pid_controller.py` | `reference_velocity`, `current_speed`, `raw_speed_error`, `throttle`, `brake`, `accel_cmd` — at the overspeed brake trigger |
| `mpc_elat` | `pid_controller.py` | `mpc_e_lat`, `mpc_smith_raw_e_lat`, `mpc_smith_e_lat_predicted`, `lateral_error`, `speed` |
| `regime_trans` | `orchestrator.py` | `regime`, `regime_blend_weight`, `mpc_fallback_active`, `speed`, `lateral_error` |
| `curve_entry` | `orchestrator.py` | `curve_local_phase`, `curve_local_state`, `curve_anticipation_active`, `distance_to_next_curve_m`, `speed` |
| `acc_override` | `orchestrator.py` | `acc_state_code`, `acc_gap_error_m`, `acc_target_speed_mps`, `adjusted_target_speed` before and after override |

Full templates: `.claude/docs/probe_pattern_reference.md`

---

## If ACTION = list

Check for existing probes:
```bash
grep -rn "PROBE:" control/pid_controller.py av_stack/orchestrator.py
```
Report which probes are currently inserted.

---

## If ACTION = add

1. **Look up the standard template** from `.claude/docs/probe_pattern_reference.md` for the requested probe name.

2. **Find the insertion point**. Read the relevant function to locate the exact line:
   - `brake_onset` → after `raw_speed_error < -self.overspeed_brake_threshold` check in `LongitudinalController.compute_control()`
   - `mpc_elat` → inside the MPC path in `LateralController` or MPCController
   - `regime_trans` → inside `_pf_compute_steering()` in orchestrator.py, after regime selection
   - `curve_entry` → inside speed governor or curve intent code in orchestrator.py
   - `acc_override` → inside `_pf_apply_acc_override()` in orchestrator.py

3. **Show the exact code block** that will be inserted. Do NOT insert yet — show for review:
   ```
   --- INSERT after line <N> in <file> ---
   # PROBE:<name> inserted by /instrument on <date>
   if getattr(self, '_probe_<name>_enabled', False):
       ...
   # END PROBE:<name>
   ----------------------------------------
   ```

4. **Show the config change** to enable it:
   ```yaml
   # Add to config/av_stack_config.yaml under stack: (or any overlay YAML)
   stack:
     probe_<name>_enabled: true
   ```
   And the orchestrator init line to read it (show the exact line and where it goes).

5. **Show the grep command** to find output in logs:
   ```bash
   ./start_av_stack.sh [scenario flags] 2>&1 | grep "PROBE:<name>" | head -20
   ```

6. **Ask for confirmation** before modifying any file.

7. **After confirmation** — make the edit, then run:
   ```bash
   pytest tests/ -v -x --tb=short
   ```
   to verify 0 regressions.

---

## If ACTION = remove

1. Find the probe:
   ```bash
   grep -n "PROBE:<name>" control/pid_controller.py av_stack/orchestrator.py
   ```

2. Show the exact lines to delete (PROBE:<name> block inclusive).

3. Show the config key to remove: `probe_<name>_enabled`

4. After confirming, make the deletion and run:
   ```bash
   pytest tests/ -v -x --tb=short
   ```

---

## For custom probes

If the probe name is not in the standard list, ask:
1. Which file? (`pid_controller.py` or `orchestrator.py`)
2. Which function/method?
3. After which line or variable assignment?
4. Which signals to capture?

Then generate a custom probe following the template in `.claude/docs/probe_pattern_reference.md`.

---

## Critical reminders

- **Never insert probes without user confirmation**
- **Probes are INACTIVE by default** until config flag is set to `true`
- **Never place probes in tight loops without frame-modulo guard** (`% 30 == 0`)
- **Always run `pytest tests/ -v -x`** after any code change
- **Always remove probes before committing** (or keep them with the config flag = false)

Probe reference: `.claude/docs/probe_pattern_reference.md`
