Generate a self-contained Python script that traces HDF5 signals around a specific event type in an AV stack recording, then run it immediately.

The user's request is: $ARGUMENTS

## Step 1 — Parse the request

Extract:
- **EVENT TYPE**: one of `brake_onset`, `speed_drop`, `regime_transition`, `acc_brake`, `ool_event`, `mpc_infeasible`, `curve_entry`, `steering_clip`, `estop`
  - If unclear, infer from context: "brake" → `brake_onset`, "speed" → `speed_drop`, "MPC" → `mpc_infeasible`, "OOL" or "lane" → `ool_event`
- **RECORDING**: file path, or "latest" (default). Find latest with: `ls -t data/recordings/*.h5 | head -1`
- **WINDOW**: frames before/after event (default: 10 before, 20 after — keep compact)
- **EXTRA SIGNALS**: any additional fields the user mentioned

## Step 2 — Select default signals by event type

Use `.claude/docs/hdf5_field_reference.md` for field paths.

**brake_onset** (trigger: `brake[i]>0.05` and `brake[i-1]<=0.05`)
Fields: `vehicle/speed`, `control/brake`, `control/brake_before_limits`, `control/throttle`, `control/longitudinal_accel_cmd_raw`, `control/target_speed_final`, `control/target_speed_post_limits`, `vehicle/acc_active`, `vehicle/acc_gap_error_m`, `vehicle/acc_ttc_s`

**speed_drop** (trigger: `diff(speed) < -1.0`)
Fields: `vehicle/speed`, `control/target_speed_final`, `control/target_speed_post_limits`, `control/brake`, `control/throttle`, `vehicle/acc_active`, `control/emergency_stop`

**regime_transition** (trigger: `regime` crosses 0.5 threshold)
Fields: `control/regime`, `control/regime_blend_weight`, `vehicle/speed`, `control/lateral_error`, `control/mpc_fallback_active`, `control/mpc_feasible`, `control/mpc_solve_time_ms`

**acc_brake** (trigger: `brake_before_limits >= 0.18` and `acc_active`)
Fields: `vehicle/acc_ttc_s`, `vehicle/acc_gap_error_m`, `vehicle/radar_fwd_distance_m`, `vehicle/radar_fwd_detected`, `control/brake`, `control/brake_before_limits`, `vehicle/speed`, `vehicle/acc_active`

**ool_event** (trigger: `|lateral_error| > 0.5` starts)
Fields: `control/lateral_error`, `control/steering`, `control/steering_hard_clip_active`, `perception/confidence`, `perception/using_stale_data`, `control/curve_local_phase`, `control/regime`

**mpc_infeasible** (trigger: `mpc_feasible == 0`)
Fields: `control/mpc_feasible`, `control/mpc_solve_time_ms`, `control/mpc_consecutive_failures`, `control/mpc_fallback_active`, `vehicle/speed`, `control/lateral_error`, `control/regime`

**curve_entry** (trigger: `curve_local_phase` rises above 0.1)
Fields: `control/curve_local_phase`, `control/curve_local_state`, `control/curve_anticipation_active`, `vehicle/speed`, `control/target_speed_final`, `control/lateral_error`, `control/curve_local_distance_ready`, `control/curve_local_arm_ready`

**steering_clip** (trigger: `steering_hard_clip_active == 1`)
Fields: `control/steering_hard_clip_active`, `control/steering_before_limits`, `control/steering`, `control/steering_hard_clip_delta`, `control/lateral_error`, `control/regime`, `vehicle/speed`

**estop** (trigger: `emergency_stop == 1`)
Fields: `control/emergency_stop`, `control/lateral_error`, `vehicle/speed`, `vehicle/acc_ttc_s`, `vehicle/acc_gap_error_m`, `control/brake`, `control/throttle`

## Step 3 — Generate and immediately run the trace script

Generate and execute this script using the Bash tool:

```python
import h5py
import numpy as np

RECORDING = "<resolved_path>"
WINDOW_BEFORE = <window_before>
WINDOW_AFTER = <window_after>

with h5py.File(RECORDING, "r") as f:
    n = len(f["control/brake"][:])

    # --- Event detection ---
    <event_detection_code>

    # --- Load signals ---
    signals = {}
    for path in [<field_list>]:
        try:
            signals[path] = f[path][:]
        except KeyError:
            signals[path] = None

event_frames = np.where(event_mask)[0] if len(event_mask.shape) > 0 else []

print(f"Recording: {RECORDING}")
print(f"Event type: <event_type>  |  {len(event_frames)} event(s) found")
if len(event_frames) == 0:
    print("No events found.")
else:
    print(f"Event frames: {event_frames[:10].tolist()}")
    for evt in event_frames[:3]:
        s = max(0, evt - WINDOW_BEFORE)
        e = min(n, evt + WINDOW_AFTER + 1)
        print(f"\n{'='*72}")
        print(f"EVENT @ frame {evt}  |  window [{s}:{e}]")
        print(f"{'='*72}")
        short_names = [p.split("/")[-1][:13] for p in signals]
        print("frame".ljust(7) + "".join(n.ljust(15) for n in short_names))
        for fi in range(s, e):
            marker = " <<<" if fi == evt else ""
            row = str(fi).ljust(7)
            for path, arr in signals.items():
                if arr is None or fi >= len(arr):
                    row += "N/A".ljust(15)
                else:
                    v = arr[fi]
                    try:
                        row += f"{float(v):.4f}".ljust(15)
                    except (ValueError, TypeError):
                        row += str(v)[:13].ljust(15)
            print(row + marker)
```

Run this script with `python3 /tmp/trace_<event_type>.py` (write it to /tmp first, then execute).

## Step 4 — Interpret the output

After the script runs, look at the signal values in the window around each event and call out:
- What changed just before the event (1–3 frames prior)
- Whether the event matches a known pattern (from `.claude/docs/tool_selection_guide.md`)
- Whether `/diagnose` or `pipeline-tracer` should be run next

Field reference: `.claude/docs/hdf5_field_reference.md`
