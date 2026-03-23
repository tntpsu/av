# Step 5 — Lead Vehicle Following / ACC (Adaptive Cruise Control)

**Created:** 2026-03-23
**Entry gate:** Step 4 NMPC ✅ complete (97.5/100 autobahn_30 at 25 m/s)
**Status:** Planning

---

## Goal

First actor-awareness feature: detect a vehicle ahead, maintain a safe following gap at target speed, and decelerate/reaccelerate smoothly. Operates entirely within the existing lane-keeping envelope — no lane changes, no overtaking.

---

## Promotion Gates

| Gate | Threshold |
|---|---|
| No collision in any scenario | 0 contact events |
| Minimum TTC across all runs | ≥ 2.0 s |
| Gap RMSE in steady following | ≤ 0.5 m |
| Jerk P95 during following | ≤ 4.0 m/s³ |
| Free-flow lateral score regression | ≤ 0.5 pts vs Phase H-3 baseline |
| Detection-loss fallback latency | ≤ 5 frames to revert to target speed |

---

## Architecture Overview

```
Unity C# (LeadVehicle.cs)
  leadVehicleDistance_m ──────► AVBridge.cs
  leadVehicleRelativeSpeed_mps ►        │
  leadVehicleDetected           ►        │ GT JSON
  leadVehicleTTC_s              ►        ▼
                                av_stack/orchestrator.py
                                         │
                            ┌────────────▼────────────┐
                            │    ACCController         │
                            │  (acc_controller.py)     │
                            │  IDM longitudinal law    │
                            │  + safety layer          │
                            └────────────┬────────────┘
                                         │ target_speed override
                                         ▼
                            Existing longitudinal chain
                            (PID + jerk limiter, unchanged)
                                         │
                                         ▼
                                   HDF5 Recorder
                            (4 new ACC fields, 6 locations)
```

**Key design decision:** ACC outputs a *target speed* to the existing longitudinal PID chain. This preserves all existing comfort gates (jerk limiting, accel clamping) without modification.

---

## Phase A — Lead Vehicle Data Pipeline

**Goal:** Establish the data contract between Unity and Python for lead vehicle state.

### Unity Side (`AVBridge.cs`)

Add to the GT JSON payload sent each frame:
```csharp
leadVehicleDetected: bool,        // true if vehicle within detection_range_m
leadVehicleDistance_m: float,     // bumper-to-bumper gap (m), 0 if not detected
leadVehicleRelativeSpeed_mps: float, // ego_speed - lead_speed (+ = closing)
leadVehicleTTC_s: float,          // distance / max(relative_speed, 0.1) (s)
```

Also create `LeadVehicle.cs` — a kinematic vehicle following a speed profile (constant, decel, accel, stop). Spawned at configurable distance ahead of ego start position via track YAML.

### Python Side

**`data/formats/data_format.py`** — add to `ControlCommandData`:
```python
lead_vehicle_detected: Optional[bool] = None
lead_vehicle_distance_m: Optional[float] = None
lead_vehicle_relative_speed_mps: Optional[float] = None
lead_vehicle_ttc_s: Optional[float] = None
```

**`data/recorder.py`** — 6-location HDF5 registration for each field under `vehicle/` group:
1. `create_dataset` (shape=(0,), maxshape)
2. List init (`lead_vehicle_*_list = []`)
3. Per-frame append
4. Resize loop
5. Write
6. `ControlCommandData` construction in `orchestrator.py`

**Tests:** `tests/test_lead_vehicle_data_pipeline.py`
- Field presence in HDF5 after mock run
- Range validation (distance ≥ 0, TTC ≥ 0)
- Null safety when lead vehicle absent

---

## Phase B — IDM Longitudinal Controller

**Goal:** Replace constant `target_speed` with gap-keeping speed when ACC is active.

### Control Law — Intelligent Driver Model (IDM)

```
a_IDM = a_max × [1 - (v/v0)^4 - (s*(v,Δv) / gap)^2]

s*(v,Δv) = s0 + max(0, v·T + v·Δv / (2·√(a_max·b)))
```

Parameters:
| Symbol | Meaning | Default |
|---|---|---|
| `s0` | Minimum gap | 2.0 m |
| `T` | Time headway | 1.5 s |
| `v0` | Target (free-flow) speed | from `target_speed` config |
| `a_max` | Max acceleration | 2.0 m/s² |
| `b` | Comfortable deceleration | 2.5 m/s² |

IDM output is an acceleration. Convert to target speed: `v_target = v + a_IDM × dt`.

### New File: `control/acc_controller.py`

```python
class ACCParams:
    enabled: bool = False
    min_gap_m: float = 2.0
    time_headway_s: float = 1.5
    max_accel_mps2: float = 2.0
    comfortable_decel_mps2: float = 2.5
    detection_range_m: float = 60.0
    cutout_speed_mps: float = 5.0      # below this, revert to fixed-speed mode

class ACCController:
    def compute_target_speed(
        self,
        ego_speed: float,
        free_flow_target: float,
        lead_detected: bool,
        gap_m: float,
        relative_speed_mps: float,
    ) -> float:
        """Returns target speed for longitudinal PID chain."""
```

### Config additions (`config/av_stack_config.yaml`):
```yaml
acc:
  enabled: false           # disabled by default — no regression on existing tracks
  min_gap_m: 2.0
  time_headway_s: 1.5
  max_accel_mps2: 2.0
  comfortable_decel_mps2: 2.5
  detection_range_m: 60.0
  cutout_speed_mps: 5.0
```

**Tests:** `tests/test_acc_controller.py` (15+ tests)
- IDM formula correctness (free flow, approach, following, stopped lead)
- Edge cases: gap = 0, lead faster than ego, relative speed negative
- Cutout at low speed
- No regression on free-flow (lead absent → returns `free_flow_target` unchanged)

---

## Phase C — Safety Layer

**Goal:** Hard limits on top of IDM output — belt and suspenders above the comfort gates.

### Emergency Brake
If `gap_m < 1.5 × v` AND `relative_speed_mps > 0` (closing): override with `a_max_brake = -4.0 m/s²` directly (bypasses IDM, feeds target speed chain).

### TTC Guard
If `TTC < 1.5s`: escalate to e-stop via existing `emergency_stop` mechanism (same path as OOL events). Log reason `acc_ttc_violation`.

### Detection-Loss Fallback
If `lead_vehicle_detected = False` for > `fallback_frames` (default 5) consecutive frames: revert ACC to free-flow (`target_speed` from config). Hysteretic re-engagement: require `lead_detected = True` for 3 frames before re-arming ACC.

**Tests:** `tests/test_acc_safety.py` (10+ tests)
- TTC < 1.5s triggers e-stop
- Emergency brake override fires at correct gap threshold
- Fallback after 5 detection-loss frames
- Hysteretic re-engagement (not re-arms after single detection frame)

---

## Phase D — Scoring + Observability

**Goal:** Make ACC quality visible in scores and PhilViz.

### New HDF5 Fields (`control/` group)
```
control/acc_active              float32   1=ACC mode, 0=free-flow
control/acc_target_gap_m        float32   desired gap (s0 + v×T)
control/acc_gap_error_m         float32   actual_gap - target_gap
control/acc_ttc_s               float32   current TTC
```

### Scoring (`tools/drive_summary_core.py`)

New ACC scoring section (only populated when `acc_active` frames > 10%):
```python
acc_gap_rmse_m       = rms(acc_gap_error[acc_mask])   # gate: ≤ 0.5m
acc_ttc_min_s        = min(acc_ttc[acc_mask])          # gate: ≥ 2.0s
acc_jerk_p95_mps3    = percentile(|jerk[acc_mask]|, 95) # gate: ≤ 4.0
acc_collision_events = sum(gap < 0.5)                  # gate: 0
```

### PhilViz — new ACC tab (`tools/debug_visualizer/backend/acc_pipeline.py`)
- Gap vs target gap (dual-axis with TTC)
- IDM acceleration output vs actual acceleration
- ACC active / inactive state indicator
- TTC timeline with 1.5s and 2.0s threshold lines

**Tests:** `tests/test_acc_scoring.py` (8+ tests)
- Metric computation against synthetic HDF5
- Gate pass/fail boundaries
- No-op when `acc_active` frames < 10%

---

## Phase E — Unity Integration + E2E Validation

**Goal:** Real end-to-end with scripted lead vehicle.

### Unity `LeadVehicle.cs`
Kinematic vehicle (no physics simulation — avoids Unity WheelCollider complexity):
```csharp
// Speed profiles selectable via track YAML:
// "constant_20"  → 20 m/s throughout
// "decel_hard"   → 20 → 5 m/s over 3s at 5s mark
// "stop_go"      → alternating 0/20 m/s every 10s
// "accel_away"   → 10 → 25 m/s (ego must not hunt)
```

Start position: `lead_vehicle_start_distance_m` ahead of ego (default 30m), configurable in track YAML.

### Validation Track: **highway_65**
Flat, long straights, predictable geometry → minimal lateral interference with ACC longitudinal logic. R800 curves still exercised with ACC active (ACC should not fight with curve decel).

### 5 Validation Scenarios

| Scenario | Lead profile | Expected behavior |
|---|---|---|
| 1. Free-flow | No lead vehicle | ACC inactive, lane-keeping identical to Phase H-3 |
| 2. Constant following | 20 m/s constant | Gap closes from 80m → target (35m), holds ± 0.5m |
| 3. Lead braking hard | 20→5 m/s in 3s | IDM decelerates smoothly, TTC never < 2.0s |
| 4. Lead accel away | 10→25 m/s | Gap opens, IDM lets ego accelerate back to target speed |
| 5. Stop-and-go | 0/20 alternating | Smooth stop + restart, no jerk spikes, no collision |

### Per-scenario Gates
- 0 collision events (gap never < 0 m)
- TTC minimum ≥ 2.0s
- Jerk P95 ≤ 4.0 m/s³
- Lateral RMSE ≤ 0.15m (no worse than Phase H-3 free-flow)
- Gap RMSE ≤ 0.5m in steady following (scenario 2 only)

---

## File Manifest

| File | Type | Notes |
|---|---|---|
| `unity/.../LeadVehicle.cs` | New | Kinematic scripted actor |
| `unity/.../AVBridge.cs` | Modified | +4 GT lead-vehicle fields |
| `control/acc_controller.py` | New | IDM + safety layer |
| `config/av_stack_config.yaml` | Modified | `acc:` block |
| `data/formats/data_format.py` | Modified | +4 `ControlCommandData` fields |
| `data/recorder.py` | Modified | 6-location HDF5 registration |
| `av_stack/orchestrator.py` | Modified | ACC dispatch in longitudinal path |
| `tools/drive_summary_core.py` | Modified | ACC scoring section |
| `tools/debug_visualizer/backend/acc_pipeline.py` | New | PhilViz ACC tab backend |
| `tools/debug_visualizer/server.py` | Modified | Register ACC tab route |
| `tests/test_lead_vehicle_data_pipeline.py` | New | ~10 tests |
| `tests/test_acc_controller.py` | New | ~15 tests |
| `tests/test_acc_safety.py` | New | ~10 tests |
| `tests/test_acc_scoring.py` | New | ~8 tests |

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Unity GT bridge lag (same 1-2 frame delay as all GT fields) | ACC uses same delay compensation as steering — `lead_vehicle_distance` extrapolated 2 frames |
| IDM instability at very close gaps (gap → 0, denominator → ∞) | Clamp IDM acceleration to `[-a_max_brake, a_max]`; TTC guard is backstop at 1.5s |
| ACC longitudinal jerk fights existing jerk limiter | ACC outputs a *target speed* not acceleration — the existing limiter chain is the sole authority |
| Lateral-longitudinal coupling (decel on curve shifts weight, affects steering) | Test scenarios 3+5 on straight segments first; add highway_65 curve-entry decel scenario only after straight validation |
| Unity kinematic lead vehicle teleports if speed profile steps are too abrupt | Ramp speed changes over ≥ 0.5s in `LeadVehicle.cs` |

---

## Revision History

| Date | Change |
|---|---|
| 2026-03-23 | Initial plan created. Step 4 NMPC confirmed complete (entry gate met). |
