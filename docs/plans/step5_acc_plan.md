# Step 5 — Lead Vehicle Following / ACC (Adaptive Cruise Control)

**Created:** 2026-03-23
**Updated:** 2026-03-23 — simulated radar, full toolset, Step 7 forward-look, multi-track E2E (12 scenarios, 3 tracks), track YAML lead_vehicle schema, TrackWaypointFollower
**Entry gate:** Step 4 NMPC ✅ complete (97.5/100 autobahn_30 at 25 m/s)
**Status:** Planning

---

## Goal

First actor-awareness feature: detect a vehicle ahead using a simulated forward radar,
maintain a safe following gap at target speed, and decelerate/reaccelerate smoothly.
Operates entirely within the existing lane-keeping envelope — no lane changes, no overtaking.

---

## Sensing Architecture — Simulated Radar (NOT Ground Truth)

The Unity GT bridge (distance, TTC pre-computed) is NOT used for following behavior.
Instead, Unity simulates the physical sensing process:

```
Unity Physics.SphereCast (forward cone, 60m, ±5°)
  ├─ hit.distance + Gaussian(σ=0.15m) ──────────► radar_fwd_distance_m
  ├─ Δdistance/dt NOT used for range rate (too noisy)
  ├─ Doppler model: Dot(ego.velocity - lead.velocity, forward)
  │    + Gaussian(σ=0.05 m/s)          ──────────► radar_fwd_range_rate_mps
  ├─ hit == null OR range > 60m         ──────────► radar_fwd_detected = false
  └─ signal-to-noise proxy (inverse range²) ──────► radar_fwd_snr (0–1)
```

**Why Doppler, not differentiated range:**
At 30 FPS (dt=0.033s), differentiating range with σ=0.15m gives σ_rate ≈ 6.4 m/s RMS —
completely unusable. Real 77 GHz FMCW radar measures Doppler from frequency shift
directly (σ~0.05–0.1 m/s). The Unity velocity difference model is the physics equivalent.

**Python ACCSensor filter layer** (EMA before IDM sees the signal):
```python
gap_filtered    = α_gap  × gap_raw  + (1−α_gap)  × gap_prev    # α=0.30
rate_filtered   = α_rate × rate_raw + (1−α_rate) × rate_prev   # α=0.20
```
Lag: ~2–3 frames (70–100 ms). At 20 m/s closing, unobserved gap ≈ 1.4–2.0 m —
safe given IDM time headway T=1.5 s already builds in 30 m of buffer.

### Step 7 Forward-Look — Radar Abstraction for Blind Spot Monitor

`control/radar_sensor.py` establishes a base class used by both Step 5 and Step 7:

```python
class RadarSensor(ABC):
    """Base interface — swappable in Step 7 without touching ACCController."""
    @abstractmethod
    def read_frame(self, raw: dict) -> RadarReading: ...

class ForwardRadarSensor(RadarSensor):
    """Step 5: long-range forward radar (60m, ±5° FOV). SphereCast + Doppler."""
    ...

# Step 7 adds:
# class RearCornerRadarSensor(RadarSensor):
#     """Short-range ±60° FOV, 30m — blind spot / rear cross-traffic."""
#     ...
```

Production ACC radar spec this models: 77 GHz FMCW, 150–250 m range, ±15° FOV.
Blind spot radar (Step 7): same technology, 24–77 GHz, 30–50 m range, ±60° FOV.

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
| Radar detection rate (lead present) | ≥ 95% of frames |

---

## Full Architecture

```
Unity Physics.SphereCast ──────────────────────────────────────────────────┐
Unity Doppler (velocity diff)                                               │
Unity LeadVehicle.cs (kinematic actor)                                      │
Unity AVBridge.cs (+4 radar_fwd_ fields in GT JSON)                          │
                                                                             │ HTTP
                                                                             ▼
                                                         av_stack/orchestrator.py
                                                                   │
                                                    _pf_run_acc_sensor()
                                                                   │
                                          control/radar_sensor.py (ACCSensor)
                                          ForwardRadarSensor.read_frame()
                                          EMA filter: gap α=0.30, rate α=0.20
                                                                   │ RadarReading
                                          control/acc_controller.py
                                          ACCController.compute_target_speed()
                                          IDM law + safety layer
                                                                   │ target_speed override
                                                                   ▼
                                               Existing longitudinal PID chain
                                               (rate limit + jerk limiter, unchanged)
                                                                   │
                                                             HDF5 Recorder
                                                  (8 new fields: 4 radar_ + 4 acc_)
```

---

## ACC Authority, State Machine & Control Integration

### ACC State Machine

`ACCController` is always in exactly one of five states. Priority ordering (highest first):

```
┌─────────────────────────────────────────────────────────────────┐
│  PRIORITY  STATE               CONDITION (evaluated in order)   │
│  ────────  ──────────────────  ────────────────────────────────  │
│  1 (HIGH)  TTC_ESTOP           TTC < ACC_TTC_CRITICAL_S (1.5s)  │
│  2         EMERGENCY_BRAKE     gap < 1.5×v AND closing          │
│  3         CUTOUT              ego_speed < cutout_speed_mps      │
│  4         DETECTION_LOSS      detected=False × ≥ fallback_frames│
│  5         ACC_ACTIVE          detected=True, all above clear    │
│  6 (LOW)   FREE_FLOW           default / ACC disabled            │
└─────────────────────────────────────────────────────────────────┘
```

**Transition rules:**
- **TTC_ESTOP** fires `emergency_stop()` (same path as OOL). Overrides ALL other states. Cannot re-arm within a run — Unity scene reset required.
- **EMERGENCY_BRAKE** overrides IDM but does NOT trigger e-stop. Feeds `a = −4.0 m/s²` clamped through the speed chain. TTC guard still re-evaluates every frame in this state.
- **CUTOUT** (`v < cutout_speed_mps`) returns `free_flow_target` via disengage ramp. Does NOT clear detection-loss counter — hysteresis state preserved across cutout.
- **DETECTION_LOSS** returns `free_flow_target` via disengage ramp (not a step — see Bumpless Transfer below).
- **Re-arm** from DETECTION_LOSS: `detected=True` for ≥ `reengage_frames` consecutive frames AND `ego_speed ≥ cutout_speed_mps`. On re-arm, enters ACC_ACTIVE via soft-engage ramp.
- TTC guard evaluates raw (unfiltered) TTC in ALL states including DETECTION_LOSS and CUTOUT.

---

### Bumpless Transfer — Disengage and Engage Ramps

**Problem without ramp:** When ACC disengages (detection loss, cutout), `target_speed` steps from the ACC-commanded value (e.g., 15 m/s following a slow lead) back to `free_flow_target` (e.g., 25 m/s) in one frame — a ~300 m/s² command before the jerk limiter catches it. This would fail the `ACC_JERK_P95_GATE_MPS3 = 4.0 m/s³` gate.

**Disengage ramp:**
```python
# On transition OUT of ACC_ACTIVE → FREE_FLOW or DETECTION_LOSS:
v_target = min(v_target_prev + params.disengage_ramp_mps2 × dt, free_flow_target)
# Ramps at 2.0 m/s² until free_flow_target is reached (~5s for a 10 m/s gap)
```

**Soft-engage (first ACC_ACTIVE frame after engage or re-arm):**
```python
# Seed IDM output at current ego speed, not at free_flow_target:
self._v_target_prev = ego_speed
# IDM then computes a_IDM from this seed — no discontinuous step
```
Without seeding, a lead detected at 25m when ego is at 25 m/s causes an immediate large IDM deceleration on frame 1.

**EMA cold-start:** On the first frame where `detected=True` after a period of no detection, initialize EMA state to the raw reading rather than the stale filtered value:
```python
if first_detection_after_loss:
    self._gap_filtered    = gap_raw      # not 0 from last dropout
    self._rate_filtered   = rate_raw
```
Without this, EMA ramps from 0 → actual gap over ~3 frames, causing IDM to see a falsely small gap and command unnecessary hard braking on re-engagement.

**New `ACCParams` field:**
```python
disengage_ramp_mps2: float = 2.0   # ramp rate on disengage (matches max_accel_mps2)
```

---

### Grade Feedforward + ACC — Why They Don't Fight

Grade FF and ACC operate in **different domains** and are not architecturally in conflict:

```
ACC output:  v_target  (speed setpoint)
Grade FF:    ΔThrottle (additive PID feedforward bias, torque domain)

Longitudinal PID:
  error = v_target − ego_speed
  u     = Kp×error + Ki×∫error + Kd×d(error)/dt + grade_ff_bias
```

When ACC commands a lower `v_target` for a stopped lead, the PID computes a large negative speed error → braking torque. Grade FF adds a small positive bias (`≈ g × sin(grade) ≈ 0.85 m/s²` at 5%). Net deceleration = `IDM_decel − grade_ff_bias`. At the steepest hill_highway grade: 2.5 − 0.85 = **1.65 m/s²** — still decelerating safely.

**G2 config gate:** `comfortable_decel_mps2 ≥ 2 × grade_ff_bias` ensures net decel > 0 on any grade in the ODD. 2.5 > 1.7 ✅. Document this constraint in a comment in `config/acc_hill_highway.yaml` — no code change required.

---

## Phase A — Lead Vehicle Data Pipeline

**Goal:** Establish the Unity → Python radar data contract and track YAML vehicle schema.
No controller logic yet. Lead vehicle must work on any track.

### Track YAML — Lead Vehicle Block (universal, any track)

Add an optional `lead_vehicle:` top-level block to any track YAML.
Absence = no lead vehicle spawned. Works with all 9 existing tracks.
Python reads this block at startup to initialize `ForwardRadarSensor` and arm ACC.

```yaml
# Example — add to any track YAML (e.g. highway_65.yml, autobahn_30.yml)
lead_vehicle:
  enabled: true
  start_distance_m: 30.0     # arc-distance ahead of ego spawn on centerline
  respect_speed_limits: true    # default true — caps profile speed at per-segment speed_limit
  speed_profile:
    type: constant            # constant | slower | decel | stop_go | accel_away
    speed_mps: 20.0           # base speed (all profiles)
    # decel extras:
    decel_to_mps: 5.0         # brake target
    decel_duration_s: 3.0     # ramp duration (min 0.5s — prevents teleport)
    decel_at_s: 5.0           # time after run start to begin decel
    # stop_go extras:
    stop_duration_s: 5.0      # time held at 0 m/s
    go_duration_s: 10.0       # time held at speed_mps
    # accel_away extras:
    accel_to_mps: 25.0        # accelerate to this speed
    accel_duration_s: 4.0     # ramp duration
```

**Speed profile types:**

| Type | Description | Key scenario |
|---|---|---|
| `constant` | Fixed speed throughout | Steady-following, free-flow baseline |
| `slower` | Constant speed < ego target — ego closes gap | Catch-up on straight vs curve |
| `decel` | Starts at speed, ramps down at configurable time | Hard-brake test |
| `stop_go` | Alternates 0 ↔ speed_mps on timer | Stop-and-go comfort |
| `accel_away` | Starts slow, accelerates ahead of ego | Ego must not hunt |

**`respect_speed_limits` (default: `true`):**
When true, `SpeedProfiler.cs` queries the current waypoint segment's `speed_limit_mph` and applies
`min(profile_speed, segment_limit)` — the lead vehicle naturally slows for tight curves just as
the ego's speed governor does. Set to `false` only for precise test scenarios (e.g. H3 hard-brake)
where the profile must execute exactly regardless of geometry.

**`start_distance_m` controls WHERE the catch-up happens:**
Gap closes at `(ego_target_speed − lead_speed)` m/s. Pick `start_distance_m` so the
catch-up arc position lands in the desired track geometry (straight vs curve).

Example on highway_65 (550m initial straight, first R500 curve at 550m):
```
start_distance_m: 30m  + lead 15 m/s, ego 25 m/s → closes in ~3s → catch-up at ~75m → STRAIGHT
start_distance_m: 430m + same                       → closes in ~43s → catch-up at ~475m → CURVE ENTRY
```

Example on autobahn_30 (600m straight, R600 semicircle starts at 600m):
```
start_distance_m: 30m  → catch-up at ~75m  → STRAIGHT (NMPC active)
start_distance_m: 560m → catch-up at ~605m → CURVE ENTRY (LMPC↔NMPC boundary)
```

### Unity Side

**`unity/.../TrackWaypointFollower.cs`** (new — standalone component):

Pre-bakes the track YAML centerline into Unity world-space waypoints at `Start()`.
Attached to `LeadVehicle` so it stays on-road on any track geometry including curves.
Without this, the lead vehicle would drive straight off the road on any arc segment.

```csharp
public class TrackWaypointFollower : MonoBehaviour {
    // Populated by TrackBuilder at scene startup from the loaded track YAML
    public List<Vector3> Waypoints;        // 1m-spaced centerline points, world space
    public float LookAheadDistance = 5f;   // scales with speed at runtime

    // Returns heading correction (deg/s) toward next waypoint
    public float ComputeSteeringCorrection(Vector3 position, Vector3 forward): ...
    // Returns current arc-distance (used to compute spawn position)
    public float ArcDistanceFromStart(Vector3 position): ...
}
```

`TrackBuilder.cs` (existing, modified): after generating road mesh, populates
`TrackWaypointFollower.Waypoints` with 1m-spaced centerline waypoints from the
same segment geometry used for the Python-side `TrackProfile`. This guarantees
the lead vehicle follows the exact same path curvature used for speed planning.

**`unity/.../LeadVehicle.cs`** (new — first-class Unity scene object):

Required components:
- `Rigidbody (isKinematic = true)` — participates in physics for SphereCast hit detection
- `BoxCollider` (approx 4.5m × 1.8m × 1.5m) — **the SphereCast must hit this to sense the vehicle**
- `MeshRenderer` + simple car mesh — visible in PhilViz camera frame inspector
- `TrackWaypointFollower` — keeps vehicle on centerline through curves

```csharp
void FixedUpdate() {
    float speed    = _profiler.ComputeSpeed(Time.time);         // speed profile
    float steerDeg = _follower.ComputeSteeringCorrection(...);  // track following
    transform.position += transform.forward * speed * Time.fixedDeltaTime;
    transform.Rotate(0, steerDeg * Time.fixedDeltaTime, 0);
}
```

`TrackBuilder.cs` spawns the prefab at `lead_vehicle.start_distance_m` arc-distance
along the waypoint path at scene `Start()`. Heading set from waypoint tangent.
If `lead_vehicle.enabled = false` (or block absent), no prefab instantiated.

**Unity layer setup for lead vehicle:**
`LeadVehicle` prefab must be placed on a dedicated Unity layer (e.g., `LeadVehicle`, layer 8).
The ego camera's **Culling Mask** must EXCLUDE this layer — the lead vehicle mesh is invisible
to the segmentation model's camera. The `Physics.SphereCast` must INCLUDE the layer.
Without this, the car body appears in ego's camera frame and the segmentation model may
misclassify its pixels as road surface or obstacles, corrupting lane detection at close range.

**`unity/.../AVBridge.cs`** (modified) — add to GT JSON payload:
```csharp
radar_fwd_detected:       bool,   // SphereCast hit within cone — false if no lead vehicle spawned
radar_fwd_distance_m:     float,  // hit.distance + Gaussian(σ=0.15f)
radar_fwd_range_rate_mps: float,  // Dot(egoVel-leadVel, forward) + Gaussian(σ=0.05f)
radar_fwd_snr:            float,  // 1.0 / (1.0 + distance²/900)
```

SphereCast fires from ego front bumper, forward direction. Hits `LeadVehicle.BoxCollider`.

```csharp
const float RADAR_MAX_RANGE_M    = 60.0f;
const float RADAR_HALF_ANGLE_DEG = 5.0f;
const float RADAR_NOISE_RANGE_M  = 0.15f;
const float RADAR_NOISE_RATE_MPS = 0.05f;
```

### Enable Procedure — Two Independent Switches

| Switch | File | Controls |
|---|---|---|
| `lead_vehicle.enabled: true` | Track YAML or scenario YAML | Unity spawns the physical vehicle + BoxCollider |
| `acc.enabled: true` | Config overlay (`acc_*.yaml`) | Python activates `ForwardRadarSensor` + `ACCController` |

Both are required for a live ACC run. They are **decoupled by design** — spawning a lead vehicle
with `acc.enabled: false` lets you validate radar signal quality in HDF5 before enabling
the controller. Recommended first run:

```bash
# Simplest enable — modify highway_65.yml directly, use acc_highway.yaml overlay
./start_av_stack.sh \
  --config config/acc_highway.yaml \
  --track-yaml tracks/highway_65.yml \   # after adding lead_vehicle: block
  --duration 60
```

Recommended default `lead_vehicle:` block for first enable:
```yaml
lead_vehicle:
  enabled: true
  start_distance_m: 40.0          # 40m ahead — ~5.7s at 7 m/s closing rate
  respect_speed_limits: true
  speed_profile:
    type: constant
    speed_mps: 18.0               # slightly slower than 25 m/s ego target
```
`start_distance_m: 40.0` gives the controller time to observe, engage, and settle before
the first curve. `start_distance_m: 10.0` immediately stresses the safety layer from frame 1.

### Python Side

**`control/radar_sensor.py`** (new):
```python
@dataclass
class RadarReading:
    detected: bool
    gap_m: float           # EMA-filtered
    range_rate_mps: float  # EMA-filtered (+ = closing)
    snr: float
    gap_raw: float         # pre-filter (for HDF5)
    range_rate_raw: float

class RadarSensor(ABC):
    @abstractmethod
    def read_frame(self, raw: dict) -> RadarReading: ...

class ForwardRadarSensor(RadarSensor):
    """60m forward radar with EMA filter. Step 5 implementation."""
    def __init__(self, gap_alpha=0.30, rate_alpha=0.20): ...
    def read_frame(self, raw: dict) -> RadarReading: ...
```

**`data/formats/data_format.py`** — add to `ControlCommandData`:
```python
# Radar sensor (raw — for diagnostics)
radar_fwd_detected: Optional[bool] = None
radar_fwd_distance_m: Optional[float] = None
radar_fwd_range_rate_mps: Optional[float] = None
radar_fwd_snr: Optional[float] = None
# ACC controller output
acc_active: Optional[float] = None       # 1.0=ACC mode, 0.0=free-flow
acc_target_gap_m: Optional[float] = None # desired gap (s0 + v×T)
acc_gap_error_m: Optional[float] = None  # filtered_gap - target_gap
acc_ttc_s: Optional[float] = None        # filtered_gap / max(range_rate, 0.1)
```

**`data/recorder.py`** — 6-location HDF5 registration for ALL 8 fields under `vehicle/` group:
1. `create_dataset` (shape=(0,), maxshape=(None,))
2. List init (`radar_fwd_distance_m_list = []` etc.)
3. Per-frame append
4. Resize loop
5. Write
6. `ControlCommandData` construction in `orchestrator.py`

### Tests

**`tests/test_lead_vehicle_data_pipeline.py`** (~12 tests):
- All 8 HDF5 fields present and correct dtype after mock run
- `radar_fwd_distance_m` ≥ 0, `radar_fwd_snr` in [0, 1]
- `acc_ttc_s` ≥ 0 when detected
- All fields write zeros / sentinels when lead absent (not NaN)
- EMA filter convergence: step input → output tracks within 3 frames at α=0.30
- `ForwardRadarSensor.read_frame()` with detected=False → `RadarReading.detected=False`, gap=0

---

## Phase B — IDM Longitudinal Controller

**Goal:** Replace constant `target_speed` with gap-keeping speed when ACC is active.

### Control Law — Intelligent Driver Model (IDM)

```
a_IDM = a_max × [1 - (v/v0)^4 - (s*(v,Δv) / gap)^2]

s*(v,Δv) = s0 + max(0, v·T + v·Δv / (2·√(a_max·b)))
```

Output is an acceleration, converted to target speed: `v_target = clamp(v + a_IDM × dt, 0, v0)`.
The existing longitudinal PID + jerk limiter chain is the sole authority on actual acceleration.

### New File: `control/acc_controller.py`

```python
@dataclass
class ACCParams:
    enabled: bool = False
    min_gap_m: float = 2.0               # s0
    time_headway_s: float = 1.5          # T
    max_accel_mps2: float = 2.0          # a_max
    comfortable_decel_mps2: float = 2.5  # b
    detection_range_m: float = 60.0
    cutout_speed_mps: float = 5.0        # revert to fixed-speed below this
    fallback_frames: int = 5             # detection-loss frames before free-flow
    reengage_frames: int = 3             # frames of detection before re-arm
    disengage_ramp_mps2: float = 2.0    # ramp rate on disengage (matches max_accel_mps2)

class ACCController:
    """
    State machine: TTC_ESTOP > EMERGENCY_BRAKE > CUTOUT > DETECTION_LOSS > ACC_ACTIVE > FREE_FLOW
    See ## ACC Authority, State Machine & Control Integration for full transition rules.
    Bumpless transfer: disengage ramp at disengage_ramp_mps2; soft-engage seeds from ego_speed.
    EMA cold-start: re-initializes filtered state to raw reading on first detection after dropout.
    """
    def compute_target_speed(
        self,
        ego_speed: float,
        free_flow_target: float,
        reading: RadarReading,
        dt: float,
    ) -> tuple[float, float, float, float]:
        """Returns (target_speed, acc_active, target_gap_m, gap_error_m)."""
```

### Config additions (`config/av_stack_config.yaml`)

```yaml
acc:
  enabled: false                  # disabled by default — zero regression risk
  min_gap_m: 2.0
  time_headway_s: 1.5
  max_accel_mps2: 2.0
  comfortable_decel_mps2: 2.5
  detection_range_m: 60.0
  cutout_speed_mps: 5.0
  fallback_frames: 5
  reengage_frames: 3
  disengage_ramp_mps2: 2.0    # ramp on detection-loss/cutout disengage; matches max_accel_mps2
```

### Tests

**`tests/test_acc_controller.py`** (~18 tests):
- IDM formula: free-flow (no lead) → returns `free_flow_target`
- IDM formula: closing at 5 m/s gap=20m → expected deceleration within tolerance
- IDM formula: stopped lead (gap=5m, v_lead=0) → v_target < v_ego
- IDM formula: lead faster than ego → v_target ≈ v0 (not constrained)
- IDM edge case: gap → 0.1m → a clamped to -a_max_brake
- IDM edge case: relative speed negative (lead pulling away) → no emergency
- Cutout: ego speed < 5 m/s → returns free_flow_target regardless
- EMA filter integration: noisy gap input → smooth target speed output
- Detection-loss hysteresis: 4 frames → still ACC; 5 frames → free-flow
- Re-engage hysteresis: 2 frames detection after loss → still free-flow; 3 frames → ACC
- `acc_active` flag correct in all state transitions
- `acc_target_gap_m` = s0 + v×T at steady following speed

---

## Phase C — Safety Layer

**Goal:** Hard limits above IDM output — belt and suspenders above the comfort gates.

Priority ordering is defined in `## ACC Authority, State Machine & Control Integration`.
This phase implements all three hard-limit mechanisms; their relative priority is enforced
by the state machine evaluation order in `compute_target_speed()`.

### Three mechanisms in `ACCController`

**1. Emergency brake override**
If `gap_m < 1.5 × ego_speed` AND `range_rate_mps > 0` (closing):
Override with `a = -4.0 m/s²` → feeds target speed chain.
Reason logged: `acc_emergency_brake`.

**2. TTC guard**
If filtered `TTC < 1.5 s`: escalate to e-stop via existing `emergency_stop` mechanism
(same code path as OOL events). Reason logged: `acc_ttc_violation`.

**3. Detection-loss fallback**
`radar_fwd_detected = False` for > `fallback_frames` (5) consecutive frames →
revert ACC to free-flow. Re-arm requires `detected = True` for `reengage_frames` (3) consecutive
frames. Log: `acc_detection_loss` / `acc_detection_reengaged`.

### Tests

**`tests/test_acc_safety.py`** (~12 tests):
- TTC < 1.5 s → e-stop triggered with reason `acc_ttc_violation`
- TTC = 1.5 s exactly → e-stop triggered (boundary inclusive)
- TTC = 1.6 s → no e-stop
- Emergency brake fires at gap < 1.5×v when closing
- Emergency brake does NOT fire when gap < 1.5×v but lead is faster (range_rate ≤ 0)
- Detection-loss fallback after exactly 5 frames
- Detection-loss fallback does NOT trigger at 4 frames
- Hysteretic re-engage: 2 frames → still free-flow; 3 frames → re-armed
- Emergency brake output clamped: target speed ≥ 0 (no negative speed command)

---

## Phase D — Scoring, Toolset, and Observability

**Goal:** Every ACC failure mode must be attributable via the full toolset.
Pattern: follow the MPC pipeline precedent exactly.

### D1 — Scoring Registry (`tools/scoring_registry.py`)

Add 14 new ACC constants — single source of truth for all downstream consumers.
Grouped into three tiers matching the safety gradient:

```python
# ── ACC — Hard safety gates (Tier 1) ────────────────────────────────────────
ACC_COLLISION_GATE: int = 0                  # events — zero tolerance; score → 0
ACC_TTC_CRITICAL_S: float = 1.5             # s  — e-stop trigger (same path as OOL)
ACC_NEAR_MISS_GAP_M: float = 2.0            # m  — gap < s0; inside minimum gap (orange)

# ── ACC — Graduated safety thresholds (Tier 2) ───────────────────────────────
ACC_TTC_MIN_GATE_S: float = 2.0             # s  — promotion gate (must stay above)
ACC_TTC_WARNING_S: float = 2.5              # s  — start of warning zone (2.0–2.5s)
ACC_TTC_COMFORTABLE_S: float = 4.0          # s  — comfortable headway (green)
ACC_DESIRED_GAP_FRACTION_WARN: float = 0.5  # —  — < 50% desired gap = IDM struggling
ACC_NEAR_MISS_PENALTY_PTS: float = 15.0     # pts — per near-miss event (Safety layer)
ACC_TTC_WARNING_PENALTY_PER_PCT: float = 0.3 # pts — per % of ACC frames in warning zone

# ── ACC — Comfort and quality (Tier 3) ───────────────────────────────────────
ACC_GAP_RMSE_GATE_M: float = 0.50           # m  — steady-following gap RMSE gate
ACC_JERK_P95_GATE_MPS3: float = 4.0         # m/s³ — following jerk (tighter than 6.0 free-flow)
ACC_DETECTION_RATE_GATE: float = 0.95       # —  — min detection rate when lead present
ACC_EMERGENCY_BRAKE_GAP_FACTOR: float = 1.5 # —  — gap < factor×speed → emergency brake
ACC_MIN_ACTIVE_FRAME_RATE: float = 0.10     # —  — min fraction for ACC section to activate
```

### D2 — Drive Summary Core (`tools/drive_summary_core.py`)

#### Longitudinal Safety Metrics

New metrics computed when `acc_active_pct > ACC_MIN_ACTIVE_FRAME_RATE`:

```python
# Tier 1 — Hard safety
acc_collision_events      = sum(radar_fwd_distance_m < 0.0)          # gate: 0
acc_ttc_violation_events  = sum(acc_ttc_s < ACC_TTC_CRITICAL_S)  # gate: 0 (each fires e-stop)

# Tier 2 — Graduated safety
acc_near_miss_events      = count_sustained(gap < ACC_NEAR_MISS_GAP_M, min_frames=3)
acc_ttc_warning_pct       = mean(acc_ttc_s[acc_mask] < ACC_TTC_WARNING_S) × 100
acc_min_gap_m             = min(radar_fwd_distance_m[acc_mask])       # informational, worst case
acc_ttc_p05_s             = percentile(acc_ttc_s[acc_mask], 5)   # P05 more robust than min

# Tier 3 — Comfort + quality
acc_gap_rmse_m            = rms(acc_gap_error_m[acc_mask])        # gate: ≤ ACC_GAP_RMSE_GATE_M
acc_ttc_min_s             = min(acc_ttc_s[acc_mask])              # gate: ≥ ACC_TTC_MIN_GATE_S
acc_jerk_p95_mps3         = percentile(|jerk[acc_mask]|, 95)      # gate: ≤ ACC_JERK_P95_GATE_MPS3
acc_detection_rate        = mean(radar_fwd_detected[lead_frames])     # gate: ≥ ACC_DETECTION_RATE_GATE
acc_active_pct            = mean(acc_active) × 100                # informational
acc_emergency_brake_events = sum(diff(acc_emergency_brake) > 0)   # informational (IDM quality)
```

**`acc_ttc_p05_s`** (5th percentile) is more useful than the minimum — a single bad radar
frame can make `min` misleading. P05 captures sustained close-following behavior.

#### Rollup into Scoring Layers

ACC safety penalties feed directly into the **existing Safety layer** (weight 29.4%),
alongside OOL and emergency stop penalties — NOT into Control or a separate section:

```python
# Safety layer deductions — add to existing list:
"Longitudinal Safety": {
    "deductions": [
        {"name": "Collision Events",
         "value": acc_collision_penalty,       # per event: 100 pts → hard zero
         "limit": "0 events"},
        {"name": "Near-Miss Events",
         "value": acc_near_miss_penalty,       # ACC_NEAR_MISS_PENALTY_PTS × count, capped 30
         "limit": "0 events"},
        {"name": "TTC Warning Zone",
         "value": acc_ttc_warning_penalty,     # ACC_TTC_WARNING_PENALTY_PER_PCT × warning_pct
         "limit": "< 5% of ACC frames"},
    ]
}
```

ACC jerk P95 > `ACC_JERK_P95_GATE_MPS3` (4.0 m/s³) feeds into the **LongitudinalComfort
layer** alongside existing Acceleration P95 and Jerk P95 deductions.

Hard zero rule: `acc_collision_events > 0` → `score = 0` (same path as catastrophic OOL).
TTC violations already trigger e-stop → existing Safety layer `safety_event_penalty` fires.

**Run comfort gate and scoring regression tests** after touching `drive_summary_core.py`.

**Roll-back contract:** `acc.enabled: false` in config disables all ACC logic, but the 8 new
HDF5 fields are always written (sentinel zeros). Every consumer that reads ACC fields must
guard on `acc_active_pct < ACC_MIN_ACTIVE_FRAME_RATE` before computing any ACC metric or
rendering any ACC section. Failure to add this guard means roll-back requires code changes,
not just config. Required in: `drive_summary_core.py`, `acc_pipeline_analysis.py`,
`acc_pipeline.py` backend, `layer_health._score_acc()`, `issue_detector` ACC modes,
`analyze_drive_overall.py` Section 15. This is a **Phase D acceptance criterion.**

### D3 — Issue Detector (`tools/debug_visualizer/backend/issue_detector.py`)

Add 6 per-frame ACC issue modes. Import from registry: `ACC_TTC_CRITICAL_S`,
`ACC_TTC_WARNING_S`, `ACC_NEAR_MISS_GAP_M`, `ACC_JERK_P95_GATE_MPS3`.

| Issue key | Condition | Severity | Safety tier |
|---|---|---|---|
| `acc_collision` | `radar_fwd_distance_m ≤ 0.0` | **critical** | Tier 1 — hard zero |
| `acc_ttc_critical` | `acc_ttc_s < ACC_TTC_CRITICAL_S (1.5s)` AND `acc_active=1` | **critical** | Tier 1 — e-stop |
| `acc_near_miss` | `radar_fwd_distance_m < ACC_NEAR_MISS_GAP_M (2.0m)` AND gap > 0 | **warning** | Tier 2 — inside min gap |
| `acc_ttc_warning_zone` | `ACC_TTC_CRITICAL_S ≤ acc_ttc_s < ACC_TTC_WARNING_S (2.5s)` | **warning** | Tier 2 — close headway |
| `acc_detection_loss_sustained` | `radar_fwd_detected=0` for ≥ 10 consecutive frames | warning | Sensing |
| `acc_jerk_spike` | `|jerk|` in ACC-active frame > `ACC_JERK_P95_GATE_MPS3 × 2 (8.0)` | warning | Comfort |

Each issue includes: frame_idx, timestamp, relevant signal values (gap, TTC, speed, acc_active), HDF5 field references.

### D4 — Triage Engine (`tools/debug_visualizer/backend/triage_engine.py`)

Add 4 ACC pattern signatures to the pattern library:

| Pattern | Signals | Root cause | Config lever |
|---|---|---|---|
| `acc_following_too_close` | `acc_ttc_warning_pct > 5%` OR `acc_near_miss_events > 0` | `time_headway_s` too small for track speed; or radar range-rate noise causing IDM under-estimation of closing speed | `time_headway_s` ↑ (1.5→2.0); verify Doppler σ |
| `acc_gap_tracking_poor` | `acc_gap_rmse_m > ACC_GAP_RMSE_GATE_M` AND `acc_active_pct > 10` | IDM oscillating around desired gap; `comfortable_decel_mps2` too low | `time_headway_s` ↑; `max_accel_mps2` ↓ |
| `acc_detection_dropout` | `acc_detection_rate < ACC_DETECTION_RATE_GATE` | SphereCast FOV too narrow; lead vehicle occlusion at curve | `detection_range_m` ↑; check `RADAR_HALF_ANGLE_DEG` |
| `acc_idm_hunting` | `acc_gap_error` sign changes > 20/min in ACC frames | IDM underdamped; `comfortable_decel_mps2` / `max_accel_mps2` asymmetry | `time_headway_s` ↑; `comfortable_decel_mps2` ↑ |

`acc_following_too_close` is the highest-priority pattern — it signals potential safety gate
failure before a collision occurs and should surface as a priority-0 recommendation in PhilViz.

Each pattern returns: hypothesis, root cause, fix hint, affected config keys.

### D5 — Layer Health (`tools/debug_visualizer/backend/layer_health.py`)

Add `_score_acc()` method alongside existing `_score_perception`, `_score_trajectory`, `_score_control`:

```python
def _score_acc(self, f: h5py.File, i: int) -> float:
    """Per-frame ACC health: 1.0=healthy, 0.0=critical. Only scored when acc_active=1."""
    # TTC margin (50%):  0.0 if TTC < critical (1.5s), 1.0 if TTC > comfortable (4.0s), linear between
    # Gap margin (30%):  0.0 if gap < near_miss (2.0m), 1.0 if gap ≥ desired_gap, linear between
    # Detection (20%):   1.0 if detected, 0.5 if not (penalty but not zero — could be valid dropout)
```

New `acc_health` key in the per-frame output dict. Surfaced in the PhilViz Layers tab.
Color coding mirrors the safety gradient: green (≥ 0.8), yellow (0.5–0.8), red (< 0.5).
Import from registry: `ACC_TTC_CRITICAL_S`, `ACC_TTC_COMFORTABLE_S`, `ACC_TTC_MIN_GATE_S`,
`ACC_NEAR_MISS_GAP_M`, `ACC_GAP_RMSE_GATE_M`.

### D6 — analyze_drive_overall.py

Add Section 15 "ACC Performance" (prints only when `acc_active_pct > ACC_MIN_ACTIVE_FRAME_RATE`):

```
=== Section 15: ACC Performance ===
ACC Active:             42.3% of frames

── Longitudinal Safety (→ Safety layer score) ──────────────────
Collision Events:       0          [PASS = 0]           Tier 1
TTC Violations:         0          [PASS = 0]           Tier 1
Near-Miss Events:       0          [PASS = 0]   gap < 2.0m
TTC Warning Zone:       1.2%       [PASS < 5%]  1.5–2.5s
TTC P05:                2.94s      [PASS ≥ 2.0s]
Min Gap Observed:       8.3m       (informational)

── Following Comfort (→ LongitudinalComfort layer) ─────────────
Gap RMSE:               0.31m      [PASS ≤ 0.50m]
TTC Minimum:            2.84s      [PASS ≥ 2.00s]
Jerk P95 (ACC):         2.1 m/s³   [PASS ≤ 4.0]
Emergency Brake Events: 0          (informational — IDM quality)

── Sensing ─────────────────────────────────────────────────────
Detection Rate:         98.2%      [PASS ≥ 95%]
```

### D7 — ACC Pipeline Analysis CLI (`tools/analyze/acc_pipeline_analysis.py`)

Standalone diagnostic tool (mirrors `mpc_pipeline_analysis.py`):

```bash
python tools/analyze/acc_pipeline_analysis.py --latest
python tools/analyze/acc_pipeline_analysis.py --file data/recordings/foo.h5
```

Four analysis cards:

| Card | Content |
|---|---|
| 1. Radar Health | Detection rate, SNR distribution, dropout events, false-negative estimate |
| 2. IDM State Quality | gap error timeline, range rate accuracy, s* vs actual gap |
| 3. Safety Layer Status | TTC distribution (with 1.5/2.0s lines), emergency brake events, fallback transitions |
| 4. Severe Frame Inspector | Top-N frames by `|acc_gap_error_m|` with context signals |

### D8 — PhilViz ACC Tab (`tools/debug_visualizer/backend/acc_pipeline.py`)

New backend module (mirrors `mpc_pipeline.py` — 4-card structure). Returns `None` for
recordings with `acc_active_pct < ACC_MIN_ACTIVE_FRAME_RATE` (safe for all existing tracks).

**Card 1 — Radar Health:**
- Detection rate (% frames)
- SNR histogram
- Dropout event timeline (frames where detected=False)
- Raw vs filtered gap overlay (shows EMA smoothing effect)

**Card 2 — Gap Tracking:**
- Actual gap vs target gap (dual time-series)
- Gap error with ±ACC_GAP_RMSE_GATE_M threshold lines
- IDM acceleration output vs actual acceleration

**Card 3 — Safety Timeline:**
- TTC time-series with 1.5 s (e-stop) and 2.0 s (gate) threshold lines
- Emergency brake events (red markers)
- ACC active/inactive state band

**Card 4 — Worst Frame Inspector:**
- Top-5 frames by |gap_error| or lowest TTC
- Per-frame context: speed, gap, range_rate, TTC, acc_active, jerk

**`tools/debug_visualizer/server.py`** — register `/api/acc_pipeline` route (same pattern as MPC).

**PhilViz restart rule applies:** kill + restart after editing `server.py`, `backend/acc_pipeline.py`, or `drive_summary_core.py`.

### Tests

**`tests/test_acc_scoring.py`** (~10 tests):
- ACC section populates when `acc_active_pct > 10%`
- ACC section absent (no keys) when `acc_active_pct < 10%` (zero regression risk)
- `acc_gap_rmse_m` computed correctly against synthetic HDF5
- `acc_ttc_min_s` = min across ACC-active frames only
- Collision event: `gap < 0.5` → `acc_collision_events ≥ 1`
- Gate pass/fail at boundary values (0.50m, 2.0s, 4.0 m/s³)
- Registry constants imported correctly in `drive_summary_core.py`

---

## Phase E — Unity Integration + E2E Validation

**Goal:** Multi-track E2E validation covering steady following, safety layer stress,
catch-up on straight vs curve, grade interaction, and high-speed NMPC+ACC coexistence.

### Track overlay configs (3 required)

**`config/acc_highway.yaml`** — inherits `mpc_highway.yaml`:
```yaml
acc:
  enabled: true
  time_headway_s: 1.5
  detection_range_m: 60.0
```

**`config/acc_autobahn.yaml`** — inherits `mpc_autobahn.yaml`:
```yaml
acc:
  enabled: true
  time_headway_s: 1.5
  detection_range_m: 60.0
```

**`config/acc_hill_highway.yaml`** — base config (no MPC overlay needed):
```yaml
acc:
  enabled: true
  time_headway_s: 2.0    # larger headway — grade changes add braking distance
  detection_range_m: 60.0
```

### Full Scenario Matrix (12 scenarios, 3 tracks)

Each scenario uses a dedicated track YAML with a `lead_vehicle:` block configured
for that scenario. Track YAMLs are NOT modified in-place — scenario configs use
`lead_vehicle.start_distance_m` and `speed_profile` to control geometry and timing.

#### Track 1: highway_65 — Foundation (8 scenarios)

All scenarios use `config/acc_highway.yaml`. ego target: 25 m/s.

| # | Name | start_dist | Profile | What it tests |
|---|---|---|---|---|
| H1 | **Free-flow baseline** | — (no lead) | — | ACC inactive; lateral regression ≤ 0.5 pts vs H-3 |
| H2 | **Steady following** | 80m | constant 20 m/s | Gap closes to ~37m, holds ±0.5m; Gap RMSE ≤ 0.5m |
| H3 | **Hard brake (straight)** | 30m | decel 20→5 m/s at 5s | IDM decelerates; TTC min ≥ 2.0s; no e-stop |
| H4 | **Lead accelerates away** | 30m | accel_away 10→25 m/s | Ego re-accelerates smoothly; gap error oscillation < 3m |
| H5 | **Stop-and-go** | 20m | stop_go 0↔20 m/s, 10s | Smooth stop + restart; jerk P95 ≤ 4.0; 0 collisions |
| H6 | **Close-gap start** | 10m | constant 5 m/s | Safety layer fires immediately; TTC guard holds; 0 collision |
| H7 | **Straight catch-up** | 30m | slower 15 m/s | Gap closes on 550m straight; steady follow after |
| H8 | **Curve catch-up** | 430m | slower 15 m/s | Gap closes entering R500 curve; ACC + curve decel coexist |

**H8 geometry detail (highway_65):**
- Lead at 430m arc-distance, lead speed 15 m/s, ego 25 m/s
- Ego catches up in ~43s × (25+15)/2 m/s ≈ at ~475m arc = R500 curve entry
- ACC must not fight curve decel: curve speed cap ~24 m/s → ACC inactive (cutout at 5 m/s)
- Real test: gap closes smoothly without abrupt mode-switch jerk at curve entry

#### Track 2: autobahn_30 — High-Speed + NMPC Coexistence (2 scenarios)

ego target: 25 m/s. NMPC active above 21 m/s. Both ego and lead in NMPC regime.

| # | Name | start_dist | Profile | What it tests |
|---|---|---|---|---|
| A1 | **High-speed steady following** | 40m | constant 20 m/s | ACC + NMPC active simultaneously; no regime interference |
| A2 | **Hard brake at NMPC speed** | 40m | decel 20→10 m/s at 5s | ACC decel while NMPC→LMPC downshift occurs; no jerk spike |

**A2 interaction detail:**
Lead brakes → ACC commands decel → ego speed drops → LMPC→NMPC downshift at 21 m/s.
The regime transition must not cause a steering spike that destabilizes ACC tracking.
Key metric: Δsteering at transition ≤ 0.10 (vs 0.078 in H-3 free-flow).

#### Track 3: hill_highway — Grade + ACC Interaction (2 scenarios)

ego target: 12 m/s. Grades: +5% uphill, -5% downhill. `time_headway_s: 2.0` (larger buffer).

| # | Name | start_dist | Profile | What it tests |
|---|---|---|---|---|
| G1 | **Grade following** | 25m | constant 10 m/s | ACC active across uphill + downhill; grade FF doesn't fight ACC |
| G2 | **Lead stops on uphill** | 15m | decel 10→0 m/s at 10s | Ego must stop on grade; IDM + grade FF together ≥ 0 accel |

**G2 interaction detail:**
Lead stops on +5% grade. Ego IDM commands decel. Grade FF simultaneously commands more throttle
(to maintain speed against gravity). These two must not cancel into a flat/slow decel that
fails the TTC gate. `comfortable_decel_mps2: 2.5` must dominate over grade FF at +5%.

### Per-scenario promotion gates (all 12 must pass)

| Gate | Threshold | Applies to |
|---|---|---|
| Collision events | 0 | All |
| TTC minimum | ≥ 2.0 s | All lead-present scenarios |
| Jerk P95 (ACC active frames) | ≤ 4.0 m/s³ | All |
| Lateral RMSE | ≤ 0.15 m | All |
| Gap RMSE (steady following) | ≤ 0.50 m | H2, H7, A1, G1 |
| Radar detection rate | ≥ 95% (lead-present) | All lead-present |
| Free-flow lateral regression | ≤ 0.5 pts vs H-3 | H1 |
| Δsteering at NMPC transition | ≤ 0.10 | A2 |

### Scenario track YAML files

Each scenario gets its own YAML file that overlays the `lead_vehicle:` block onto the
base track geometry. Base track geometry is never modified.

```
tracks/scenarios/
  highway_h2_steady.yml          # inherits highway_65 segments, adds lead_vehicle:
  highway_h3_hard_brake.yml
  highway_h4_accel_away.yml
  highway_h5_stop_go.yml
  highway_h6_close_gap.yml
  highway_h7_straight_catchup.yml
  highway_h8_curve_catchup.yml
  autobahn_a1_steady.yml
  autobahn_a2_hard_brake.yml
  hill_g1_grade_following.yml
  hill_g2_stop_on_grade.yml
```

Scenario YAMLs use a `base:` key to inherit track geometry:
```yaml
base: tracks/highway_65.yml    # all segments, road_width, loop, etc.
lead_vehicle:
  enabled: true
  start_distance_m: 430.0
  speed_profile:
    type: slower
    speed_mps: 15.0
```

`TrackProfile` loader (Python) and `TrackBuilder.cs` (Unity) both handle `base:` inheritance.

---

## File Manifest

| File | Type | Notes |
|---|---|---|
| `unity/.../TrackWaypointFollower.cs` | New | Centerline waypoint path for lead vehicle — any track |
| `unity/.../LeadVehicle.cs` | New | Kinematic actor: 5 speed profiles, BoxCollider, MeshRenderer |
| `unity/.../TrackBuilder.cs` | Modified | Populate waypoints + spawn lead vehicle from track YAML |
| `unity/.../AVBridge.cs` | Modified | +4 `radar_` fields (SphereCast + Doppler) |
| `control/radar_sensor.py` | New | `RadarSensor` ABC + `ForwardRadarSensor` — Step 7 base |
| `control/acc_controller.py` | New | IDM + safety layer (`ACCParams`, `ACCController`) |
| `config/av_stack_config.yaml` | Modified | `acc:` block (enabled=false by default) |
| `config/acc_highway.yaml` | New | ACC highway overlay |
| `config/acc_autobahn.yaml` | New | ACC autobahn overlay |
| `config/acc_hill_highway.yaml` | New | ACC hill_highway overlay (time_headway_s=2.0) |
| `tracks/scenarios/*.yml` | New | 11 scenario YAMLs (base + lead_vehicle block) |
| `data/formats/data_format.py` | Modified | +8 fields to `ControlCommandData` |
| `data/recorder.py` | Modified | 6-location HDF5 registration × 8 fields |
| `av_stack/orchestrator.py` | Modified | `_pf_run_acc_sensor()` + ACC dispatch in longitudinal path |
| `tools/scoring_registry.py` | Modified | 8 new ACC constants |
| `tools/drive_summary_core.py` | Modified | Section 15 ACC scoring |
| `tools/analyze/acc_pipeline_analysis.py` | New | 4-card CLI diagnostic (mirrors mpc_pipeline_analysis.py) |
| `tools/analyze/analyze_drive_overall.py` | Modified | Section 15 printout |
| `tools/debug_visualizer/backend/acc_pipeline.py` | New | PhilViz ACC tab backend |
| `tools/debug_visualizer/backend/issue_detector.py` | Modified | 4 new ACC failure modes |
| `tools/debug_visualizer/backend/triage_engine.py` | Modified | 3 new ACC patterns |
| `tools/debug_visualizer/backend/layer_health.py` | Modified | `_score_acc()` + `acc_health` key |
| `tools/debug_visualizer/server.py` | Modified | Register `/api/acc_pipeline` route |
| `tests/test_lead_vehicle_data_pipeline.py` | New | ~12 tests — L1 unit |
| `tests/test_acc_controller.py` | New | ~18 tests — L1 unit |
| `tests/test_acc_safety.py` | New | ~12 tests — L1 unit |
| `tests/test_acc_scoring.py` | New | ~10 tests — L1 unit |
| `tests/test_acc_replay.py` | New | ~15 tests — L2 replay-based (make_acc_recording) |
| `tests/test_acc_properties.py` | New | ~10 tests — L4 IDM invariants + radar fuzz |
| `tests/conftest.py` | Modified | Add `make_acc_recording()` builder |
| `tests/fixtures/golden_recordings.json` | Modified | ACC H1/H2 golden recordings (after Phase E) |
| `tests/fixtures/scoring_baselines.json` | Modified | ACC metric baselines (after Phase E) |

**Total new tests: ~67** (unit/integration — E2E scenarios are live runs, not pytest)

---

## Diagnostic Coverage Checklist (Mandatory)

Per the diagnostic coverage rule, every new failure mode must appear in all three locations:

| Failure mode | issue_detector | triage_engine | layer_health | Score impact |
|---|---|---|---|---|
| Collision (gap ≤ 0m) | ✓ `acc_collision` | — (hard zero, no tuning) | ✓ `acc_health` = 0.0 | **Score → 0** |
| TTC critical (< 1.5s) | ✓ `acc_ttc_critical` | — (→ e-stop, no tuning) | ✓ `acc_health` = 0.0 | Safety layer e-stop penalty |
| Near-miss (gap < 2.0m) | ✓ `acc_near_miss` | ✓ `acc_following_too_close` | ✓ `acc_health` penalty | Safety layer −15 pts/event |
| TTC warning zone (1.5–2.5s) | ✓ `acc_ttc_warning_zone` | ✓ `acc_following_too_close` | ✓ `acc_health` penalty | Safety layer −0.3 pts/% |
| Sustained detection loss | ✓ `acc_detection_loss_sustained` | ✓ `acc_detection_dropout` | ✓ `acc_health` × 0.5 | Informational only |
| ACC jerk spike | ✓ `acc_jerk_spike` | ✓ `acc_idm_hunting` | ✓ `acc_health` penalty | LongitudinalComfort layer |

---

## Testing

### Overview

Testing is split into four layers. All pytest tests must pass before each phase promotion gate.
E2E scenarios (Phase E) are live runs — not pytest — and use the per-scenario gates in that section.

```
Layer 1 — Unit tests (no Unity, no HDF5 files)
  ├─ test_lead_vehicle_data_pipeline.py  (~12 tests)  Phase A
  ├─ test_acc_controller.py              (~18 tests)  Phase B
  ├─ test_acc_safety.py                  (~12 tests)  Phase C
  └─ test_acc_scoring.py                 (~10 tests)  Phase D

Layer 2 — Replay-based synthetic tests (Tier 1, no Unity)
  └─ test_acc_replay.py  (~15 tests)
     Driven by make_acc_recording() in conftest.py
     Tests drive_summary_core.py ACC section without live runs

Layer 3 — Regression guard (Tier 2, golden recordings)
  ├─ test_comfort_gate_replay.py (existing — re-run after Phase D, comfort gates unchanged)
  └─ test_scoring_regression.py  (existing — new ACC baselines registered after Phase E)

Layer 4 — Invariant / property / fuzz (no Unity, mathematical guarantees)
  └─ test_acc_properties.py  (~10 tests)
```

---

### Layer 1 — Unit Tests

Already covered per-phase in the checklist (A12, B8, C4, D10). Key emphases:

**`test_acc_controller.py`** — IDM formula correctness is the most important unit coverage:
- `a_IDM` matches analytical formula for 5 (v, gap, Δv) combinations
- Free-flow: `gap → ∞` → `a_IDM ≈ a_max(1 − (v/v0)^4)` — accelerates to v0
- Desired gap: `s*(v, Δv) = s0 + vT + vΔv/(2√(ab))` matches 3 reference cases
- Cutout: `v < cutout_speed_mps` → returns `v_free` immediately
- Detection-loss hysteresis: exactly 5 frames → free-flow; exactly 3 frames → re-arm
- Edge: `gap → 0` → `a_IDM → −∞` clipped to `−a_max`; does not blow up

**`test_acc_safety.py`** — safety contract boundaries:
- TTC guard fires at 1.49s, does NOT fire at 1.51s
- Emergency brake fires at `gap < 1.5 × ego_speed AND range_rate > 0`; does NOT fire when lead is faster (range_rate < 0)
- Both guards are stateless — can be triggered on frame 1 without warm-up

---

### Layer 2 — Replay-Based Synthetic Tests

**`tests/conftest.py` — `make_acc_recording()`:**

Extends `make_nominal_recording()` to include all 8 ACC/radar HDF5 fields.
Returns a path to a synthetic HDF5 that `drive_summary_core.py` can score.

```python
def make_acc_recording(
    tmp_path,
    n_frames=300,
    acc_active_start=0,      # frame index ACC engages
    gap_m=None,              # callable (frame) → gap, or float constant
    range_rate_mps=0.0,      # m/s (positive = closing)
    n_near_miss_frames=0,    # frames where gap < 2.0m
    n_ttc_warn_frames=0,     # frames where TTC in 1.5–2.5s window
    collision=False,         # injects gap ≤ 0 event
) -> Path
```

Key: the builder must write sentinel values (gap=0, detected=0, active=0) for frames where
ACC is inactive, so `drive_summary_core.py` ignores them when computing gap RMSE.
The EMA-filtered fields (`radar_fwd_distance_m`) and raw fields both need to be populated
to avoid `nan` in the metric computation.

**`tests/test_acc_replay.py`** (~15 tests):
- Nominal steady-following → gap RMSE ≤ 0.5m; Safety layer unchanged (no deductions)
- 3 near-miss events → Safety layer deduction = `3 × ACC_NEAR_MISS_PENALTY_PTS`
- TTC warning zone 20% of active frames → Safety deduction matches `20 × ACC_TTC_WARNING_PENALTY_PER_PCT`
- Collision event → overall score forced to 0.0 regardless of other layers
- ACC inactive all frames → score identical to free-flow baseline (no ACC fields affect score)
- 0 active frames but fields present → `acc_active_frame_rate < ACC_MIN_ACTIVE_FRAME_RATE` → informational note, no penalty
- Detection rate 90% (< 95% gate) → logged as issue but does NOT affect Safety score
- Stop-and-go jerk → `acc_jerk_p95 > ACC_JERK_P95_GATE_MPS3` → LongitudinalComfort layer deduction

---

### Layer 3 — Regression Guard

**After Phase D** — run immediately (touches `drive_summary_core.py`):
```bash
pytest tests/test_comfort_gate_replay.py -v -k "Synthetic or Boundary"   # Tier 1 ~0.3s
pytest tests/test_scoring_regression.py -v                                 # Tier 2 ~0.5s
```
These test existing non-ACC golden recordings. They should be **unchanged** — ACC scoring
adds new metrics but must not disturb the existing Safety, LongitudinalComfort, or Trajectory
layer deductions when `acc_active = 0` for all frames.

**After Phase E (live E2E runs)** — register ACC golden recordings:
1. Pick 2 recordings: one steady-following run (H2) and one free-flow run (H1)
2. Add filenames to `tests/fixtures/golden_recordings.json`
3. Add expected scores to `BASELINE_SCORES` in `tests/conftest.py`
4. Add per-metric baselines (gap_rmse, acc_ttc_p05, detection_rate) to `tests/fixtures/scoring_baselines.json`
5. Re-run full suite: `pytest tests/ -v` — must be green

ACC scoring baselines should be locked with **strict tolerance**: gap RMSE ±0.10m,
acc_ttc_p05 ±0.3s. This is tighter than the ±0.03m RMSE tolerance used for lateral metrics,
because ACC gap error depends on IDM params that rarely change.

---

### Layer 4 — Invariant / Property / Fuzz Tests

**`tests/test_acc_properties.py`** (~10 tests):

These test mathematical guarantees that go beyond formula correctness:

| Test | Invariant |
|---|---|
| `test_idm_gap_stays_positive_under_hard_decel` | With `a_max=2.5, b=1.5, s0=2.0`, starting at gap=3.0m, IDM a_IDM ≤ 0 — never accelerates into vehicle |
| `test_idm_free_flow_converges_to_v0` | Simulate 30s at gap=200m: speed reaches within 0.2 m/s of v0=25 m/s |
| `test_idm_no_oscillation_at_equilibrium` | At steady-state gap, `|a_IDM| < 0.05 m/s²` — system is at rest |
| `test_radar_sensor_output_always_valid` | 500 random (detected, distance, rate) inputs → gap ≥ 0, snr ∈ [0,1], no NaN/Inf |
| `test_ema_filter_bounded_by_input` | EMA output always between two consecutive raw inputs — no overshoot |
| `test_safety_guard_fires_before_contact` | Parametrize 20 (gap, rate) pairs; TTC guard always fires ≥ 1 frame before gap ≤ 0 at the given approach rate |
| `test_acc_scoring_no_op_when_inactive` | Zero active frames → Safety deduction = 0; LongitudinalComfort deduction = 0 |
| `test_acc_scoring_collision_overrides_all` | Even with all other layers = 100: collision field → total = 0.0 |

**Radar sensor fuzz** (part of `test_acc_properties.py`):
```python
@pytest.mark.parametrize("seed", range(20))
def test_radar_sensor_fuzz_no_nan(seed):
    rng = np.random.default_rng(seed)
    sensor = ForwardRadarSensor()
    for _ in range(100):
        raw = {
            "radar_fwd_detected": bool(rng.integers(0, 2)),
            "radar_fwd_distance_m": float(rng.uniform(0, 80)),   # includes > 60m edge
            "radar_fwd_range_rate_mps": float(rng.normal(0, 5)), # large noise
            "radar_fwd_snr": float(rng.uniform(0, 1.5)),         # includes out-of-range
        }
        r = sensor.read_frame(raw)
        assert not np.isnan(r.gap_m)
        assert not np.isnan(r.range_rate_mps)
        assert r.snr >= 0.0
```

---

### Test Count Summary

| File | Tests | Tier | When to run |
|---|---|---|---|
| `test_lead_vehicle_data_pipeline.py` | ~12 | L1 Unit | After Phase A |
| `test_acc_controller.py` | ~18 | L1 Unit | After Phase B |
| `test_acc_safety.py` | ~12 | L1 Unit | After Phase C |
| `test_acc_scoring.py` | ~10 | L1 Unit | After Phase D |
| `test_acc_replay.py` | ~15 | L2 Replay | After Phase D |
| `test_acc_properties.py` | ~10 | L4 Property | After Phase B (IDM) |
| `test_comfort_gate_replay.py` (existing) | 32 | L3 Regression | After Phase D |
| `test_scoring_regression.py` (existing) | 25 | L3 Regression | After Phase D; ACC baselines after Phase E |
| **Total new tests** | **~67** | | |

**CI timing estimate:**
- L1 unit tests (67 tests, pure Python): < 1s
- L2 replay tests (15 tests, h5py synthetic writes): < 0.5s
- L4 property tests: IDM stability simulation = 900 steps × 20 seeds = 18,000 IDM calls ≈ 20ms; radar fuzz = 2,000 calls ≈ 2ms. Total L4 < 0.1s.
- **Full new test suite: < 2s.** No CI time budget risk.

**Bumpless transfer test coverage** (add to `test_acc_controller.py`):
- Disengage: run 5 ACC-active frames at 15 m/s following, then 5 frames `detected=False` → verify `v_target` does NOT jump to 25 m/s on frame 6; verify ramp at ≤ 2.0 m/s² per frame
- Soft-engage: first ACC frame with gap=25m, ego=25 m/s → verify `v_target` on frame 1 is within `a_max×dt` of 25 m/s (not a large jump down)
- EMA cold-start: detected=False for 10 frames, then detected=True gap=40m → verify EMA output frame 11 = 40m (not a partial ramp from 0)

**Cut-in — explicitly out of scope:**
A vehicle appearing suddenly at close range from an adjacent lane is outside the ODD (single lane, no merging actors). If it occurs (e.g., Unity scene misconfiguration), the TTC guard will fire an e-stop correctly — this is the right behavior, not a failure. Document in `docs/ODD.md` (P6).

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Differentiated range rate unusable (σ≈6 m/s) | Use Doppler model (velocity diff) in Unity, σ≈0.05 m/s |
| EMA lag at high closing speed | T=1.5s headway provides 30m buffer; 2-frame lag ≈ 1.4m |
| IDM hunting from residual noise | Existing jerk limiter absorbs oscillation; `time_headway_s` is primary stability knob |
| ACC longitudinal jerk fights existing jerk limiter | ACC outputs target speed, not acceleration — limiter chain is sole authority |
| Unity kinematic lead vehicle teleports on abrupt profile steps | Ramp all speed changes over ≥ 0.5s in `LeadVehicle.cs` |
| ACC active during curve entry, fighting curve decel | `cutout_speed_mps=5.0` — ACC disengages at low speed; curve decel governs peak |
| False detection (empty road, SNR artifact) | `radar_fwd_detected` requires SphereCast hit — no hit, no detection |
| Jerk spike on ACC disengage (step target speed) | Disengage ramp at `disengage_ramp_mps2=2.0 m/s²` — target ramps gradually to free_flow_target |
| Jerk spike on ACC engage (IDM sees small gap immediately) | Soft-engage: seeds `v_target_prev = ego_speed` on first ACC frame |
| EMA reports falsely small gap after detection dropout | Cold-start fix: re-initialize EMA to raw reading on first detection after loss |
| Lead vehicle mesh corrupts lane detection | LeadVehicle on dedicated Unity layer excluded from ego camera culling mask |
| E2E safety gate passes by noise luck (single run) | ≥3 runs required for safety-gated scenarios (H3, H5, H6, A2, G2) |
| A2 NMPC transition failure attributed to ACC incorrectly | A2 prerequisite: resolve NMPC 12–14 m/s transitions; or isolate via regime log analysis |
| Consumer breaks when ACC inactive (sentinel zeros in HDF5) | Roll-back contract: `acc_active_pct < ACC_MIN_ACTIVE_FRAME_RATE` guard in all consumers |
| Step 7 radar field naming conflict in HDF5 schema | `radar_fwd_*` prefix for forward radar; Step 7 uses `radar_rear_left_*`, `radar_rear_right_*` |

---

## Implementation Checklist

Ordered by dependency. Each phase ends with a validation step before the next phase begins.
Check off tasks as completed. E2E scenarios (Phase E) are live runs — not pytest.

### Phase A — Lead Vehicle Data Pipeline

**Unity C#**
- [ ] **A1** `TrackWaypointFollower.cs` — 1m-spaced centerline waypoints from track YAML segments; `ComputeSteeringCorrection()`, `ArcDistanceFromStart()`
- [ ] **A2** `SpeedProfiler.cs` — 5 profile types (`constant`, `slower`, `decel`, `stop_go`, `accel_away`); `respect_speed_limits` flag applies `min(profile_speed, segment_limit)`; all speed ramps ≥ 0.5s
- [ ] **A3** `LeadVehicle.cs` — `Rigidbody` (isKinematic=true) + `BoxCollider` + `MeshRenderer` + `TrackWaypointFollower` + `SpeedProfiler`; kinematic move in `FixedUpdate`; assign prefab to `LeadVehicle` Unity layer (layer 8)
- [ ] **A4** `TrackBuilder.cs` — read `lead_vehicle:` block from YAML; populate `TrackWaypointFollower.Waypoints`; spawn `LeadVehicle` prefab at `start_distance_m` arc position with correct heading; no-op if block absent or `enabled: false`
- [ ] **A5** `AVBridge.cs` — add 4 `radar_fwd_*` fields; `Physics.SphereCast` from ego front bumper (LayerMask INCLUDES `LeadVehicle` layer); Gaussian noise (σ=0.15m range, σ=0.05 m/s rate); Doppler model; SNR proxy; `radar_fwd_detected=false` when no lead spawned
- [ ] **A5a** Ego camera `Culling Mask` — EXCLUDE `LeadVehicle` layer; confirm segmentation model camera does not render lead vehicle mesh; `SphereCast` LayerMask is separate from render mask and must still include it

**Python**
- [ ] **A6** `TrackProfile` loader — parse `lead_vehicle:` block; parse `base:` key for scenario YAML inheritance; expose `lead_vehicle_config` to orchestrator
- [ ] **A7** `control/radar_sensor.py` — `RadarReading` dataclass; `RadarSensor` ABC; `ForwardRadarSensor` with EMA filter (α_gap=0.30, α_rate=0.20); **EMA cold-start**: re-initialize filtered state to raw reading on first detection after any gap/dropout (not from stale 0); returns sentinel `RadarReading(detected=False)` when lead absent
- [ ] **A8** `data/formats/data_format.py` — +8 fields to `ControlCommandData` (4 `radar_fwd_*` + 4 `acc_*`)
- [ ] **A9** `data/recorder.py` — 6-location HDF5 registration × 8 fields (create_dataset, list init, per-frame append, resize, write, ControlCommandData construction)
- [ ] **A10** `av_stack/orchestrator.py` — add `_pf_run_acc_sensor()` sub-method; init `ForwardRadarSensor` at startup if `lead_vehicle_config.enabled`; populate `ControlCommandData` radar fields each frame

**Scenario YAMLs**
- [ ] **A11** Create `tracks/scenarios/` directory + 11 scenario YAMLs (each with `base:` + `lead_vehicle:` block per Phase E scenario matrix)

**Tests**
- [ ] **A12** `tests/test_lead_vehicle_data_pipeline.py` (~12 tests) — all 8 HDF5 fields present; range ≥ 0; EMA convergence; sentinel when no lead; `base:` inheritance parsing

**Phase A Validation (live run)**
- [ ] Run `highway_65.yml` with `lead_vehicle.enabled=true`, `acc.enabled=false`; verify 8 HDF5 fields in recording; verify no NaN/zero-only fields; verify PhilViz frame inspector shows lead vehicle mesh

---

### Phase T — Test Infrastructure

Must be completed before Phase B. Adds the replay-based and property test scaffolding used by Phases B–D.

**Replay framework**
- [ ] **T1** `tests/conftest.py` — `make_acc_recording()` builder: writes a synthetic HDF5 with all 8 ACC/radar fields; parameters: `n_frames`, `acc_active_start`, `gap_m` (callable or constant), `range_rate_mps`, `n_near_miss_frames`, `n_ttc_warn_frames`, `collision` (bool); sentinel zeros for inactive frames; fields in `vehicle/` group matching Phase A schema
- [ ] **T2** `tests/test_acc_replay.py` (~15 tests) — replay-based drive_summary_core.py tests: nominal following → Safety deductions = 0; near-miss events → Safety deduction = `n × ACC_NEAR_MISS_PENALTY_PTS`; TTC warning zone 20% → Safety deduction matches registry formula; collision → score = 0.0 override; ACC inactive all frames → score identical to free-flow; detection rate < gate → informational only, no penalty; jerk P95 > gate → LongitudinalComfort deduction
- [ ] **T3** `tests/test_acc_properties.py` (~13 tests) — IDM invariants (gap never goes negative from IDM alone, free-flow convergence, equilibrium rest); radar fuzz (20 random seeds × 100 frames → no NaN/Inf, snr ≥ 0); EMA bounded by input; **EMA cold-start** (re-initialized to raw on first detection); safety guard fires before contact; collision override; **bumpless disengage** (v_target ramp ≤ 2.0 m/s²/frame); **soft-engage** (v_target frame-1 ≤ ego_speed + a_max×dt)

**Tests**
- [ ] Run `pytest tests/test_acc_replay.py tests/test_acc_properties.py -v` — all green before Phase B

---

### Phase B — IDM Longitudinal Controller

- [ ] **B1** `control/acc_controller.py` — `ACCParams` dataclass (all 9 config fields including `disengage_ramp_mps2`)
- [ ] **B2** `control/acc_controller.py` — `ACCController.compute_target_speed()`: full state machine (priority order: TTC_ESTOP > EMERGENCY_BRAKE > CUTOUT > DETECTION_LOSS > ACC_ACTIVE > FREE_FLOW); IDM law; `s*(v,Δv)` formula; `a_IDM → v_target`; **disengage ramp** on state exit to FREE_FLOW; **soft-engage seed** (`v_target_prev = ego_speed`) on first ACC_ACTIVE frame; `acc_active` flag; `acc_target_gap_m`; `acc_gap_error_m`; `acc_ttc_s` outputs
- [ ] **B3** `config/av_stack_config.yaml` — `acc:` block (`enabled: false` by default)
- [ ] **B4** `config/acc_highway.yaml` — ACC highway overlay (`acc.enabled: true`, inherits `mpc_highway.yaml`)
- [ ] **B5** `config/acc_autobahn.yaml` — ACC autobahn overlay
- [ ] **B6** `config/acc_hill_highway.yaml` — ACC hill highway overlay (`time_headway_s: 2.0`)
- [ ] **B7** `av_stack/orchestrator.py` — instantiate `ACCController` at startup if `acc.enabled`; call after `_pf_run_acc_sensor()`; replace `target_speed` with ACC output when `acc_active=1`; populate `acc_*` HDF5 fields

**Tests**
- [ ] **B8** `tests/test_acc_controller.py` (~18 tests) — IDM formula correctness; free-flow pass-through; cutout; hysteresis; edge cases (gap→0, lead faster than ego); `acc_target_gap_m` = s0 + v×T

**Phase B Validation (live run)**
- [ ] Run scenario H2 (steady following); verify gap closes from 80m to ~37m and holds; verify `acc_active=1` in HDF5; verify `acc_gap_error_m` < 1m in steady state

---

### Phase C — Safety Layer

- [ ] **C1** `ACCController` — emergency brake: `gap_m < 1.5 × ego_speed AND range_rate > 0` → `a = -4.0 m/s²`; log reason `acc_emergency_brake`
- [ ] **C2** `ACCController` — TTC guard: filtered `TTC < 1.5s` → e-stop via existing `emergency_stop` mechanism; log reason `acc_ttc_violation`
- [ ] **C3** `ACCController` — detection-loss fallback: ≥ 5 consecutive frames `detected=False` → free-flow; re-arm after 3 consecutive detected frames; log `acc_detection_loss` / `acc_detection_reengaged`

**Tests**
- [ ] **C4** `tests/test_acc_safety.py` (~12 tests) — TTC < 1.5s → e-stop; boundary at exactly 1.5s; emergency brake fires at correct condition; does NOT fire when lead is faster; detection-loss at exactly 5 frames; hysteretic re-arm

**Phase C Validation (live run)**
- [ ] Run scenario H6 (close-gap start, 10m, lead 5 m/s); verify 0 collisions; verify safety layer engages (emergency brake or TTC guard visible in HDF5); TTC min ≥ 2.0s

---

### Phase D — Scoring, Toolset, and Observability

- [ ] **D1** `tools/scoring_registry.py` — 14 new ACC constants in 3 tiers: Tier 1 hard safety (`ACC_COLLISION_GATE`, `ACC_TTC_CRITICAL_S`, `ACC_NEAR_MISS_GAP_M`); Tier 2 graduated (`ACC_TTC_MIN_GATE_S`, `ACC_TTC_WARNING_S`, `ACC_TTC_COMFORTABLE_S`, `ACC_DESIRED_GAP_FRACTION_WARN`, `ACC_NEAR_MISS_PENALTY_PTS`, `ACC_TTC_WARNING_PENALTY_PER_PCT`); Tier 3 comfort/quality (`ACC_GAP_RMSE_GATE_M`, `ACC_JERK_P95_GATE_MPS3`, `ACC_DETECTION_RATE_GATE`, `ACC_EMERGENCY_BRAKE_GAP_FACTOR`, `ACC_MIN_ACTIVE_FRAME_RATE`)
- [ ] **D2** `tools/drive_summary_core.py` — longitudinal safety metrics (Tier 1: collisions, TTC violations; Tier 2: near-miss events, TTC warning %, TTC P05, min gap; Tier 3: gap RMSE, jerk P95, detection rate, emergency brake count). Wire Tier 1+2 penalties into **Safety layer** deductions. Wire Tier 3 jerk into **LongitudinalComfort** layer. Collision → hard score zero override.
- [ ] **D3** `tools/debug_visualizer/backend/issue_detector.py` — 6 ACC issue modes: `acc_collision` (critical), `acc_ttc_critical` (critical), `acc_near_miss` (warning, gap < 2.0m), `acc_ttc_warning_zone` (warning, TTC 1.5–2.5s), `acc_detection_loss_sustained` (warning), `acc_jerk_spike` (warning)
- [ ] **D4** `tools/debug_visualizer/backend/triage_engine.py` — 4 patterns: `acc_following_too_close` (priority-0, near-miss + TTC warning), `acc_gap_tracking_poor`, `acc_detection_dropout`, `acc_idm_hunting`
- [ ] **D5** `tools/debug_visualizer/backend/layer_health.py` — `_score_acc()`: TTC margin (50%), gap margin (30%), detection quality (20%); green/yellow/red thresholds from registry; `acc_health` key in per-frame output
- [ ] **D6** `tools/analyze/analyze_drive_overall.py` — Section 15 three-section printout (Longitudinal Safety / Following Comfort / Sensing); only prints when ACC active ≥ 10% of frames
- [ ] **D7** `tools/analyze/acc_pipeline_analysis.py` — 4-card CLI tool (Radar Health, IDM State Quality, Safety Layer Status, Severe Frame Inspector)
- [ ] **D8** `tools/debug_visualizer/backend/acc_pipeline.py` — PhilViz ACC tab backend; 4 cards; returns `None` for non-ACC recordings
- [ ] **D9** `tools/debug_visualizer/server.py` — register `/api/acc_pipeline` route

**Tests + Regression**
- [ ] **D10** `tests/test_acc_scoring.py` (~10 tests) — metric computation; gate boundaries; no-op when inactive; registry constants imported correctly
- [ ] **D11** `pytest tests/test_comfort_gate_replay.py -v -k "Synthetic or Boundary"` — touches `drive_summary_core.py`
- [ ] **D12** `pytest tests/test_scoring_regression.py -v` — touches `drive_summary_core.py`

**Phase D Validation**
- [ ] Run Phase B validation recording through `acc_pipeline_analysis.py --latest`; verify all 4 cards render
- [ ] Restart PhilViz; verify ACC tab loads; verify existing non-ACC recordings show no ACC tab (returns `None`)

---

### Phase E — E2E Validation (12 scenarios, live runs)

Run each scenario; record results. All gates must pass before promotion.

**Run requirements — reproducibility under radar noise:**
- **Safety-gated scenarios** (H3, H5, H6, A2, G2) require **≥ 3 runs each.** All 3 must pass
  the safety gates (`TTC min ≥ 2.0s`, `0 collisions`). A single run can pass or fail by
  radar noise luck. Use `run_ab_batch.py` with `--runs 3` and inspect the worst-case run.
- **Quality-gated scenarios** (H2, H4, H7, H8, A1, G1) require **1 run** — gap RMSE and
  jerk P95 are averaged metrics, less sensitive to single-frame noise.
- **H1 (free-flow)** — 1 run. No lead vehicle → no noise variance.

**A2 prerequisite:** Scenario A2 (NMPC regime transition under ACC decel) requires that the
NMPC 12–14 m/s transition issue (10 unexplained MPC→PP switches, documented in
`current_state.md`) is **resolved or explicitly bounded** before A2 is run. An ACC decel from
25 → 10 m/s will pass through this range, and a spurious MPC→PP transition during braking
would show up as a `Δsteering` gate failure for reasons unrelated to ACC. Options:
1. Fix the root cause (pass `current_path_kappa` not preview into `_update_heading_zero_gate`) — preferred.
2. Gate A2 separately: run A2 with `--duration 120` and check regime log; if any MPC→PP
   transition occurs outside the ACC decel window (frames where `acc_active=0`), mark as NMPC issue, not ACC.

**highway_65 — 8 scenarios** (`config/acc_highway.yaml`)
- [ ] **H1** Free-flow — no lead vehicle; lateral regression ≤ 0.5 pts vs H-3 baseline
- [ ] **H2** Steady following — constant 20 m/s lead; Gap RMSE ≤ 0.5m
- [ ] **H3** Hard brake — lead 20→5 m/s at 5s; TTC min ≥ 2.0s; 0 e-stops
- [ ] **H4** Accel away — lead 10→25 m/s; gap error oscillation < 3m; no hunting
- [ ] **H5** Stop-and-go — 0↔20 m/s, 10s cycle; jerk P95 ≤ 4.0; 0 collisions
- [ ] **H6** Close-gap start — lead 5 m/s, start 10m ahead; 0 collisions; TTC min ≥ 2.0s
- [ ] **H7** Straight catch-up — lead 15 m/s, start 30m ahead; steady follow after catch-up
- [ ] **H8** Curve catch-up — lead 15 m/s, start 430m ahead; ACC+curve decel coexist; no jerk spike at curve entry

**autobahn_30 — 2 scenarios** (`config/acc_autobahn.yaml`)
- [ ] **A1** High-speed steady — lead 20 m/s, ego 25 m/s; NMPC+ACC simultaneously active; no regime interference; Gap RMSE ≤ 0.5m
- [ ] **A2** Hard brake at NMPC speed — lead 20→10 m/s; LMPC→NMPC downshift under ACC decel; Δsteering at transition ≤ 0.10; TTC min ≥ 2.0s

**hill_highway — 2 scenarios** (`config/acc_hill_highway.yaml`)
- [ ] **G1** Grade following — lead 10 m/s constant; ACC active across +5% uphill and -5% downhill; grade FF and ACC not in conflict; Gap RMSE ≤ 0.5m
- [ ] **G2** Stop on grade — lead 10→0 m/s at 10s on uphill; ego stops safely; `comfortable_decel_mps2` dominates grade FF; 0 collisions; TTC min ≥ 2.0s

---

### Promotion and Documentation

- [ ] **P1** All 12 E2E scenarios pass all gates (table above)
- [ ] **P2** `docs/ROADMAP.md` — Step 5 ACC ✅ with date and score
- [ ] **P3** `docs/agent/current_state.md` — Step 5 complete, Step 6 next
- [ ] **P4** `docs/agent/tasks.md` — Step 5 row marked complete
- [ ] **P5** `docs/agent/architecture.md` — ACC + radar architecture section; `_pf_run_acc_sensor()` in orchestrator decomposition
- [ ] **P6** `docs/ODD.md` — update capabilities (ACC, lead vehicle following) and sensor assumptions (forward radar simulated)
- [ ] **P7** `docs/agent/MEMORY.md` — update project snapshot; add ACC pitfalls/lessons to memory files

---

## Recovery Phase F — Longitudinal Owner Simplification and Lead-Contact Attribution

This phase is required because the current highway ACC path shows a split authority bug:

- ACC reduces `target_speed_post_limits`
- but `target_speed_final`, planner target speed, `trajectory/reference_point_velocity`, and `reference_velocity_effective` stay at free-flow

This phase also closes the attribution gap around the Unity lead-collision override so the next failure can be identified directly from HDF5, CLI analysis, and PhilViz.

### Architectural goal

Reduce the longitudinal path to one canonical owner chain:

1. free-flow target
2. governor target
3. ACC target
4. safety target
5. final longitudinal target

Only one stage should resolve the final target. Every downstream speed field must be derived from that result instead of carrying partially-overlapping ownership.

### Phase F1 — Canonical longitudinal decision object

- [ ] **F1.1** `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py` — introduce one per-frame longitudinal decision structure or dict returned by a dedicated resolver such as `_pf_resolve_longitudinal_target(...)`
- [ ] **F1.2** `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py` — move final speed ownership out of `_pf_run_speed_governor()` and `_pf_apply_acc_override()` mutation patterns; `gov['adjusted_target_speed']` should stop being the hidden shared mutable authority
- [ ] **F1.3** `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py` — final decision must explicitly contain:
  - `free_flow_target_mps`
  - `governor_target_mps`
  - `acc_target_mps`
  - `safety_target_mps`
  - `final_target_mps`
  - `final_owner_code`
  - `final_owner_reason`
  - `acc_request_estop`
- [ ] **F1.4** `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py` — after final decision resolves, push the same `final_target_mps` into:
  - planner target speed
  - reference-point velocity
  - `target_speed_final`
  - `reference_velocity_effective`
- [ ] **F1.5** `/Users/philiptullai/Documents/Coding/av/trajectory/models/trajectory_planner.py` — stop relying on stale planner-internal `self.target_speed` as the hidden owner for runtime longitudinal authority; planner velocity generation must consume the resolved final target supplied for the current frame

### Phase F2 — ACC state and collapsed-gap safety

- [ ] **F2.1** `/Users/philiptullai/Documents/Coding/av/control/acc_controller.py` — export ACC state explicitly per frame:
  - `FREE_FLOW`
  - `ACC_ACTIVE`
  - `EMERGENCY_BRAKE`
  - `TTC_ESTOP`
  - `CUTOUT`
  - `DETECTION_LOSS`
- [ ] **F2.2** `/Users/philiptullai/Documents/Coding/av/control/acc_controller.py` — add a collapsed-gap safety branch that does not depend on positive range rate; a detected lead with gap already below the emergency floor must still request a stop or emergency brake
- [ ] **F2.3** `/Users/philiptullai/Documents/Coding/av/control/acc_controller.py` — record whether the stop came from:
  - TTC critical
  - emergency brake by closing rate
  - collapsed-gap override
  - detection loss fallback
- [ ] **F2.4** `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/AVBridge.cs` — surface a dedicated lead-collision / collision-override flag in vehicle state instead of forcing analysts to infer it indirectly from `radar_fwd_distance_m = 0.1`
- [ ] **F2.5** `/Users/philiptullai/Documents/Coding/av/unity/AVSimulation/Assets/Scripts/LeadVehicle.cs` — keep the existing collision behavior, but publish an explicit telemetry bit or reason field when collision/trigger overlap is active

### Phase F3 — Recorder contract additions

- [ ] **F3.1** `/Users/philiptullai/Documents/Coding/av/data/formats/data_format.py` — add:
  - `acc_state_code`
  - `acc_target_speed_mps`
  - `acc_request_estop`
  - `acc_safety_mode_code`
  - `lead_collision_override_active`
  - `lead_collision_detected`
  - `planner_target_speed_applied_mps`
  - `final_longitudinal_target_mps`
  - `final_longitudinal_owner_code`
  - `final_longitudinal_owner_reason`
  - `reference_velocity_source_code`
- [ ] **F3.2** `/Users/philiptullai/Documents/Coding/av/data/recorder.py` — persist those fields end-to-end into HDF5

### Phase F4 — Debug tool and attribution upgrades

- [ ] **F4.1** `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py` — add a `longitudinal_owner` summary block with:
  - owner-mode histogram
  - ACC target vs final target disagreement rate
  - planner target vs final target disagreement rate
  - collapsed-gap override rate
  - lead-collision override active rate
- [ ] **F4.2** `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py` — print a dedicated authority waterfall section for ACC recordings:
  - free-flow
  - governor
  - ACC
  - safety
  - final target
  - reference velocity
- [ ] **F4.3** `/Users/philiptullai/Documents/Coding/av/tools/analyze/acc_pipeline_analysis.py` — add cards for:
  - ACC state distribution
  - final owner distribution
  - ACC-to-final target mismatch events
  - lead-collision override windows
- [ ] **F4.4** `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/backend/acc_pipeline.py` — expose the same waterfall and state fields to PhilViz
- [ ] **F4.5** `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/server.py` — add frame inspector fields for:
  - `acc_state_code`
  - `acc_target_speed_mps`
  - `planner_target_speed_applied_mps`
  - `final_longitudinal_target_mps`
  - `final_longitudinal_owner_code`
  - `reference_velocity_source_code`
  - `lead_collision_override_active`
- [ ] **F4.6** `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js` — add a `Longitudinal Owner` card and timeline bands for:
  - ACC state
  - final owner
  - lead-collision override
  - target mismatch episodes
- [ ] **F4.7** `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/backend/issue_detector.py` — add new issue modes:
  - `acc_owner_mismatch`
  - `acc_collapsed_gap_no_stop`
  - `lead_collision_override_active`
- [ ] **F4.8** `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/backend/triage_engine.py` — add pattern signatures:
  - `acc_target_not_applied`
  - `lead_contact_following_invalid`

### Phase F5 — Tests

- [ ] **F5.1** `/Users/philiptullai/Documents/Coding/av/tests/test_acc_controller.py` — add state-code and collapsed-gap tests:
  - collapsed gap with zero range-rate still requests stop
  - state code output matches the branch taken
- [ ] **F5.2** `/Users/philiptullai/Documents/Coding/av/tests/test_acc_replay.py` — add replay assertions for target ownership:
  - when ACC target drops, `final_longitudinal_target_mps` and `reference_velocity_effective` drop with it
  - no mismatch between final target and planner-applied target in active ACC frames
- [ ] **F5.3** `/Users/philiptullai/Documents/Coding/av/tests/test_control_command_hdf5.py` — verify new ownership and collision-override datasets exist
- [ ] **F5.4** `/Users/philiptullai/Documents/Coding/av/tests/test_drive_summary_contract.py` — verify summary fields load and legacy recordings degrade cleanly
- [ ] **F5.5** `/Users/philiptullai/Documents/Coding/av/tests/test_visualizer_transport_parity.py` or equivalent PhilViz parity test — ensure CLI and PhilViz agree on longitudinal owner and lead-collision attribution

### Phase F6 — Live validation

- [ ] Re-run `/Users/philiptullai/Documents/Coding/av/tracks/scenarios/highway_h2_steady.yml` with `/Users/philiptullai/Documents/Coding/av/config/acc_highway.yaml`
- [ ] Verify on the newest recording:
  - no sustained `0.1 m` lead-contact state without a commanded stop
  - `acc_target_speed_mps`, `final_longitudinal_target_mps`, `trajectory/reference_point_velocity`, and `reference_velocity_effective` agree within tolerance during ACC-active frames
  - PhilViz and CLI both attribute the final owner directly
- [ ] Only after this passes should late-drive swerving be treated as a standalone lateral problem

### Acceptance gates for Recovery Phase F

1. No frame may show `acc_active=1`, `acc_gap_error_m << 0`, and `reference_velocity_effective` still pinned at free-flow.
2. A detected lead at collapsed gap must trigger either:
   - emergency stop
   - emergency brake
   - explicit invalid-run/contact labeling
3. Debug tooling must expose the final longitudinal owner without manual inference.
4. The latest highway ACC run must no longer produce a near-miss driven by owner mismatch.

---

## Revision History

| Date | Change |
|---|---|
| 2026-03-23 | Initial plan created. Step 4 NMPC confirmed complete (entry gate met). |
| 2026-03-23 | **Full rewrite:** Unity GT sensing replaced with simulated radar (SphereCast + Doppler). Added `RadarSensor` abstraction for Step 7 reuse. Full toolset: `scoring_registry`, `issue_detector`, `triage_engine`, `layer_health`, `acc_pipeline_analysis.py` CLI, PhilViz ACC tab, `analyze_drive_overall` Section 15. ~52 new unit tests. |
| 2026-03-23 | **Phase A + E expansion:** Track YAML `lead_vehicle:` block schema (universal, any track). `TrackWaypointFollower.cs` — lead vehicle stays on-road through curves. `tracks/scenarios/` directory for 11 scenario YAMLs with `base:` inheritance. Phase E expanded to 12 scenarios across 3 tracks (highway_65 × 8, autobahn_30 × 2, hill_highway × 2). Specific tests for curve catch-up (H8), NMPC regime transition under decel (A2), grade+ACC stop interaction (G2). |
| 2026-03-23 | **Longitudinal safety + scoring rollup:** 14 scoring registry constants (3 tiers). Near-miss (gap < 2.0m) and TTC warning zone (1.5–2.5s) added as Tier 2 graduated safety. Safety metrics wire into existing Safety layer score (29.4% weight), not isolated Section 15. Collision → hard score zero. issue_detector expanded to 6 modes; triage_engine to 4 patterns (acc_following_too_close priority-0). layer_health `_score_acc()` uses TTC/gap/detection gradient. Section 15 report restructured into 3 sub-sections. Checklist D1–D6 updated. |
| 2026-03-23 | **Testing strategy section added.** 4-layer test coverage: L1 unit (52 tests, per-phase), L2 replay synthetic (`make_acc_recording()` + `test_acc_replay.py`, 15 tests), L3 regression guard (existing `test_comfort_gate_replay` + `test_scoring_regression` + golden recording registration procedure after Phase E), L4 invariant/property/fuzz (`test_acc_properties.py`, 10 tests — IDM gap-positive invariant, EMA bounded, safety fires before contact, collision override). Phase T checklist added between Phase A and Phase B. Total new tests raised to ~67. |
| 2026-03-23 | **Lead vehicle clarifications:** `SpeedProfiler.cs` as separate component. `respect_speed_limits: true` default — caps profile speed at per-segment limit via `min(profile_speed, segment_limit)`. Two-step enable documented: `lead_vehicle.enabled` (track YAML, spawns Unity object) and `acc.enabled` (config overlay, arms Python controller) are independent. Default first-enable config: `start_distance_m: 40m`, `speed_mps: 18m/s`, `respect_speed_limits: true`. Implementation checklist added. |
| 2026-03-23 | **Chief AV/release engineering review — 12 gaps addressed.** P0: (1) Bumpless transfer added — disengage ramp `disengage_ramp_mps2=2.0 m/s²` + soft-engage seed + EMA cold-start re-init; (2) ACC state machine fully documented with 6-state priority table and all transition rules; (3) Grade FF + ACC interaction proven architecturally non-conflicting — documented with math; (4) ≥3 runs required for safety-gated scenarios H3/H5/H6/A2/G2. P1: A2 NMPC prerequisite gate added; roll-back contract (`acc_active_pct` guard in every consumer) made explicit acceptance criterion; lead vehicle Unity layer exclusion from ego camera culling mask; CI timing estimated (< 2s for all new tests). P2: `radar_fwd_*` field naming applied throughout (Step 7 compatible); cut-in explicitly OOD; CPU budget < 10μs noted. Total risks table expanded from 7 → 17 rows. |
| 2026-03-26 | **Recovery Phase F added.** Documents the current highway ACC owner-mismatch failure, the collapsed-gap safety gap, and a file-by-file implementation checklist to simplify the longitudinal authority chain and add direct lead-contact attribution to HDF5, CLI analysis, and PhilViz. |
