# AV Stack — Configuration Guide

**Last updated:** 2026-02-23
**Config file:** `config/av_stack_config.yaml`
**Total parameters:** 328 across 7 sections

---

## How to Use This Guide

Three tiers of documentation depth:

| Tier | What | How documented |
|---|---|---|
| **1 — Primary levers** | ~30 params that directly govern comfort gates and primary behavior | Full entry: description, range, gate impact, tuning notes |
| **2 — Subsystem tuning** | ~80 active params touched for specific behavior refinement | One-liner with safe range |
| **3 — Advanced / inactive** | ~220 remaining; includes all bypassed PID/Stanley params | Listed in Appendix with defaults and tier flag |

**Start here:** If you're about to tune a comfort metric, read the Quick Map (§2) first.
**Changing a param?** Follow the Safe Change Checklist (§3) before committing.
**Searching?** The Full Parameter Index (Appendix) has every param — grep-friendly.

---

## § 2 — Comfort Gate → Parameter Quick Map

The six S1-M39 comfort gates and the parameters that control them:

| Gate | Threshold | Primary Params | Section |
|---|---|---|---|
| `accel_p95_filtered` | ≤ 3.0 m/s² | `max_accel`, `throttle_rate_limit`, `throttle_smoothing_alpha`, `startup_accel_limit` | §5 |
| `commanded_jerk_p95` | ≤ 6.0 m/s³ | `max_jerk_min/max`, `jerk_cooldown_frames`, `jerk_cooldown_scale`, `throttle_rate_limit` | §5 |
| `lateral_error_p95` | ≤ 0.40 m | `reference_smoothing`, `lane_smoothing_alpha`, `pp_feedback_gain`, `ref_x_rate_limit` | §4, §6 |
| `time_in_lane_centered` | ≥ 70 % | `reference_lookahead`, `dynamic_reference_lookahead`, `reference_smoothing` | §6 |
| `out_of_lane_events` | = 0 | `emergency_stop_error`, `max_lateral_error`, `curve_speed_preview_*`, `steering_speed_guard_*` | §7, §4 |
| `steering_jerk_max` | ≤ 20.0 | `pp_max_steering_jerk`, `pp_max_steering_rate` | §4 |

**Target speed** (`control.longitudinal.target_speed`) is a root parameter — changes cascade to all six gates.
It must be kept in sync across **three** config locations (see §5).

---

## § 3 — Safe Change Checklist

Before changing any Tier 1 or Tier 2 parameter:

```
[ ] Read the parameter's entry in this guide — understand the effect and safe range
[ ] If it affects a comfort gate: run the comfort gate tests first to establish baseline
    pytest tests/test_comfort_gate_replay.py -v -k "Synthetic or Boundary"
[ ] Change only ONE parameter at a time (or a tightly coupled pair)
[ ] A/B test with ≥ 5 runs:
    python tools/analyze/run_ab_batch.py --config-a config/baseline.yaml \
                                          --config-b config/candidate.yaml --runs 5
[ ] Re-run comfort gate tests with golden recordings:
    pytest tests/test_comfort_gate_replay.py -v
[ ] If golden recording scores shift > ±2 points: record 3 fresh validation runs and
    update tests/fixtures/golden_recordings.json + tests/conftest.py BASELINE_SCORES
```

**Do not change `av_stack.py` logic** to match a parameter change — config is the tuning surface.
**Do not change parameter values without A/B testing** — the 100+ param surface has unexpected interactions.

---

## § 4 — Active Lateral Control (Pure Pursuit)

Pure Pursuit is the active lateral steering mode (`control.lateral.control_mode = pure_pursuit`).
All PID and Stanley params in this section are **bypassed** and documented in §9.

### Tier 1 — Primary Levers

#### `control.lateral.control_mode`
**Default:** `pure_pursuit` | **Options:** `pure_pursuit` / `pid` / `stanley`

Active steering algorithm selector. Changing this requires re-tuning the corresponding gains.
`pure_pursuit` is the only validated mode as of S1-M39. `pid` and `stanley` are preserved
but untested at current speeds.

---

#### `control.lateral.pp_feedback_gain`
**Default:** `0.10` | **Units:** rad/m | **Range:** [0.05, 0.20]

Pure Pursuit lateral error correction gain. Scales how strongly detected lateral offset
drives steering adjustment on top of the geometric PP command.
- Too high (>0.15): oscillation on straights, excessive steering jerk on curves
- Too low (<0.07): under-responds to lateral drift; centered% drops
- **Comfort gate impact:** `lateral_error_p95`, `time_in_lane_centered`

---

#### `control.lateral.pp_max_steering_rate`
**Default:** `0.4` | **Units:** rad/frame | **Range:** [0.20, 0.60]

Maximum per-frame change in steering command. Primary rate limiter in PP mode — the
PID-era adaptive rate/jerk/smoothing system is bypassed. Acts as a hard ceiling on
how quickly the vehicle can steer.
- Lower (0.25–0.30): smoother steering; may miss tight curve entry at 12 m/s
- Higher (0.50+): faster response but can produce visible steering snaps
- **Comfort gate impact:** `steering_jerk_max` (indirectly)

---

#### `control.lateral.pp_max_steering_jerk`
**Default:** `18.0` | **Units:** norm/s² | **Range:** [10.0, 22.0]
**Comfort gate: `steering_jerk_max` ≤ 20.0**

Steering jerk cap. At the current value of 18.0, the vehicle operates near but below the
20.0 gate. The gap-filtered metric at S1-M39 was 18.0–18.4 (within cap).
- Do not raise above 22.0 without re-validating the gap-filter metric
- Lowering to 14–15 reduces steering aggressiveness noticeably in curves

---

#### `control.lateral.pp_stale_decay`
**Default:** `0.98` | **Units:** fraction/frame | **Range:** [0.90, 0.99]

Per-frame multiplicative decay applied to steering output when perception is stale (no
valid lane detection). At 0.98: ~50% decay in 35 frames (~1.75s at 20 Hz).
- 0.95: steering decays to ~50% in ~14 frames — more conservative
- 0.99: ~70 frames to 50% — more persistent hold
- **Affects:** behavior during perception failures; no direct gate impact under normal operation

---

#### `control.lateral.pp_min_lookahead`
**Default:** `0.5 m` | **Units:** metres | **Range:** [0.3, 2.0]

Minimum effective lookahead distance. Prevents PP geometric singularity when the reference
point is at or near the vehicle. Rarely needs adjustment.

---

#### `control.lateral.pp_ref_jump_clamp`
**Default:** `0.5 m` | **Units:** metres/frame | **Range:** [0.3, 1.0]

Maximum reference point lateral jump per frame. Second safety net behind `ref_x_rate_limit`
in the trajectory layer. Prevents discrete reference hops from causing steering spikes.

---

### Tier 2 — Subsystem Tuning

**Curvature smoothing:**

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `curvature_smoothing_alpha` | 0.5 | [0.3, 0.7] | EMA on incoming curvature; reduced at C1 since trajectory smooths upstream |
| `curvature_transition_threshold` | 0.01 | — | Change threshold triggering transition smoothing |
| `curvature_transition_alpha` | 0.3 | [0.15, 0.5] | Blend alpha during curvature transitions |
| `curvature_stale_hold_seconds` | 0.3 s | [0.1, 0.5] | How long to hold last curvature when signal stale |

**Curve feedforward system** (adds curvature-based steering offset on top of PP geometric command):

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `curve_feedforward_gain` | 1.0 | [0.8, 1.3] | Baseline curvature → steering feedforward multiplier |
| `curve_feedforward_threshold` | 0.012 | — | Minimum curvature to activate feedforward |
| `curve_feedforward_curvature_clamp` | 0.04 | — | Clamps curvature input to feedforward calculation |
| `curve_feedforward_bins[0].gain_scale` | 1.0 | — | Scale for mild curves (curvature 0.0–0.02) |
| `curve_feedforward_bins[1].gain_scale` | 1.3 | [1.1, 1.5] | Scale for sharp curves (curvature 0.02+) |

**Dynamic curve authority** (`dynamic_curve_authority_enabled = true`):
Boosts steering rate/jerk limits during curves when vehicle is falling behind the turn:

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `dynamic_curve_rate_boost_gain` | 0.6 | [0.3, 1.0] | Gain for rate boost when deficit detected |
| `dynamic_curve_rate_boost_max` | 0.34 | [0.20, 0.50] | Maximum rate boost (added to base rate limit) |
| `dynamic_curve_jerk_boost_gain` | 2.0 | [1.0, 3.5] | Jerk boost gain |
| `dynamic_curve_jerk_boost_max_factor` | 3.8 | — | Max jerk boost multiplier |
| `dynamic_curve_hard_clip_boost_max` | 0.2 | — | Max hard clip relaxation during curve deficit |
| `dynamic_curve_entry_governor_gain` | 0.8 | [0.5, 1.0] | Scales entry boost by curve anticipation score |
| `dynamic_curve_entry_governor_max_scale` | 1.3 | — | Maximum entry governor scale factor |
| `dynamic_curve_entry_governor_stale_floor_scale` | 1.15 | — | Minimum scale when perception is stale |
| `dynamic_curve_comfort_lat_accel_comfort_max_g` | 0.18 g | — | Lateral accel comfort ceiling for boost gating |
| `dynamic_curve_speed_low_mps` | 4.0 m/s | — | Low speed bound for speed-scaled boost |
| `dynamic_curve_speed_high_mps` | 10.0 m/s | — | High speed bound for speed-scaled boost |
| `dynamic_curve_speed_boost_max_scale` | 1.4 | — | Max boost scale at low speed |

**Curve intent / upcoming detection:**

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `curve_upcoming_enter_threshold` | 0.012 | — | Curvature above which "curve upcoming" is set |
| `curve_upcoming_exit_threshold` | 0.009 | — | Hysteresis exit threshold |
| `curve_upcoming_on_frames` | 2 | — | Frames above threshold before "upcoming" asserted |
| `road_curve_enter_threshold` | 0.011 | — | Threshold for "vehicle currently in curve" |
| `road_curve_exit_threshold` | 0.009 | — | Hysteresis exit for "in curve" |
| `straight_curvature_threshold` (lateral) | 0.01 | — | Below this, road is considered straight |
| `curve_intent_single_owner_mode` | true | — | Only one system drives curve intent at a time |

**Speed-based steering adjustment:**

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `curve_mode_speed_cap_mps` | 7.0 m/s | [6.0, 10.0] | Speed cap in active curve mode; reduced from 8.0 at S-turn transitions |
| `speed_gain_min/max` | 1.0–1.5 | — | Curvature-dependent steering gain range |

**Curve unwind policy** (`curve_unwind_policy_enabled = true`) — softens steering limits after peak:

| Parameter | Default | Effect |
|---|---|---|
| `curve_unwind_frames` | 12 | Duration of unwind phase (frames) |
| `curve_unwind_rate_scale_end` | 0.8 | Rate limit multiplier at end of unwind |
| `curve_unwind_jerk_scale_end` | 0.7 | Jerk limit multiplier at end of unwind |
| `curve_unwind_integral_decay` | 0.85 | Integral decay factor during unwind |

**Turn feasibility governor** (telemetry-only, `turn_feasibility_governor_enabled = true`):

| Parameter | Default | Effect |
|---|---|---|
| `turn_feasibility_curvature_min` | 0.002 | Min curvature for feasibility reporting |
| `turn_feasibility_guardband_g` | 0.015 g | Safety margin on lateral accel limit |

---

## § 5 — Longitudinal Speed & Comfort

### Tier 1 — Primary Levers

#### `control.longitudinal.target_speed`
**Default:** `12.0` | **Units:** m/s
⚠️ **Must be kept in sync across three locations:**
1. `control.longitudinal.target_speed` ← this param
2. `control.longitudinal.max_speed` ← hard ceiling
3. `safety.max_speed` ← emergency stop trigger
4. `trajectory.target_speed` ← speed planner reference

Changing this to 15 m/s requires updating all four, plus re-validating comfort gates
(accel/jerk gates are more easily tripped at higher speed).

---

#### `control.longitudinal.max_accel`
**Default:** `1.2` | **Units:** m/s² | **Range:** [0.8, 2.0]
**Comfort gate: `accel_p95_filtered` ≤ 3.0 m/s²**

Hard acceleration ceiling. The filtered accel gate measures EMA-smoothed acceleration so
peak values above 1.2 can still pass the gate if they're brief. The S1-M39 validated value
is 1.08–1.32 m/s². Raising above 1.5 risks gate violations on aggressive throttle demands.

---

#### `control.longitudinal.max_decel`
**Default:** `2.4` | **Units:** m/s² | **Range:** [1.5, 3.5]

Maximum braking deceleration. Affects comfort on overspeed events and curve approach.
Raising above 3.0 can produce passenger-noticeable braking. Keep ≤ `trajectory.speed_planner.max_decel`.

---

#### `control.longitudinal.max_jerk_min` / `max_jerk_max`
**Defaults:** `1.0` / `4.0` | **Units:** m/s³ | **Range:** [0.5, 3.0] / [2.0, 8.0]
**Comfort gate: `commanded_jerk_p95` ≤ 6.0 m/s³**

Adaptive jerk envelope. The actual per-frame jerk cap is linearly interpolated by speed
error between these two bounds:
- Speed error < `jerk_error_min` (0.5 m/s): cap = `max_jerk_min` = 1.0 m/s³
- Speed error > `jerk_error_max` (3.0 m/s): cap = `max_jerk_max` = 4.0 m/s³

S1-M39 tightened this from [2.0, 6.0] to [1.0, 4.0] — S1-M39 validated jerk P95 = 1.44 m/s³.
The gate allows 6.0, so there is significant headroom.

---

#### `control.longitudinal.jerk_cooldown_frames`
**Default:** `8` | **Units:** frames (at 20 Hz = 0.4s) | **Range:** [4, 16]
**Key S1-M39 comfort parameter.**

After the jerk cap is triggered, the system holds a reduced limit for this many frames.
Prevents oscillation where repeated cap triggers amplify jerk. At 8 frames (0.4s), the
vehicle has time to recover before the next aggressive accel/decel request is honoured.
- Too low (< 4): jerk cap oscillation visible in recordings
- Too high (> 16): sluggish throttle response after any speed change

---

#### `control.longitudinal.jerk_cooldown_scale`
**Default:** `0.4` | **Range:** [0.2, 0.6]

Multiplier applied to `max_jerk_min` during cooldown. Effective jerk cap during cooldown
= `max_jerk_min × jerk_cooldown_scale` = 1.0 × 0.4 = 0.4 m/s³. This ensures the
recovery phase is smooth. Raising above 0.6 reduces the comfort benefit of cooldown.

---

#### `control.longitudinal.throttle_rate_limit`
**Default:** `0.04` | **Units:** throttle fraction/frame | **Range:** [0.02, 0.08]
**Key S1-M39 comfort parameter. Comfort gate: `commanded_jerk_p95`**

Maximum throttle command change per frame. At 20 Hz, 0.04/frame = 0.8 throttle units/second.
This is the most direct lever for smoothing throttle inputs.
- Reduce to 0.02: very smooth, but noticeably slow to accelerate from low speed
- Raise to 0.08: faster response but throttle spikes become visible in jerk metrics

---

#### `control.longitudinal.brake_rate_limit`
**Default:** `0.05` | **Units:** brake fraction/frame | **Range:** [0.03, 0.10]

Maximum brake command change per frame. Slightly higher than `throttle_rate_limit` since
braking authority needs to act more quickly for comfort and safety. Affects brake smoothness
but also emergency responsiveness.

---

#### `control.longitudinal.throttle_smoothing_alpha`
**Default:** `0.7` | **Range:** [0.5, 0.85]

EMA applied to final throttle output before sending to Unity. At 0.7, ~3-frame half-life
(≈ 0.15s). S1-M39 raised from 0.6 to 0.7. Higher = smoother but slower throttle response.
**Comfort gate impact:** `accel_p95_filtered` (less spiking), `commanded_jerk_p95`.

---

#### `control.longitudinal.startup_accel_limit`
**Default:** `0.5` | **Units:** m/s² | **Range:** [0.3, 0.8]

Soft acceleration limit during the startup ramp phase (`startup_ramp_seconds` = 2.5s,
until vehicle reaches `startup_speed_threshold` = 2.0 m/s). Prevents an initial
lurch from zero. S1-M39 reduced from a higher value for smoother launches.

---

#### `control.longitudinal.target_speed_slew_rate_up`
**Default:** `1.5` | **Units:** m/s² | **Range:** [0.5, 3.0]

Maximum rate at which the internal target speed can increase per second. Governs
how aggressively the controller commands acceleration during a speed step change.

#### `control.longitudinal.target_speed_slew_rate_down`
**Default:** `1.8` | **Units:** m/s² | **Range:** [1.0, 3.0]

Maximum rate at which the internal target speed can decrease. S1-M39 slowed this
from a higher value — gentler target ramp-down reduces commanded jerk during deceleration.

---

### Tier 2 — Subsystem Tuning

**Speed error and mode control:**

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `speed_error_deadband` | 0.05 m/s | — | Speed errors smaller than this produce no correction |
| `speed_error_gain_under` | 0.12 | — | P gain when below target speed |
| `speed_error_gain_over` | 0.08 | — | P gain when above target (asymmetric: gentler overspeed response) |
| `mode_switch_min_time` | 0.4 s | [0.2, 0.8] | Minimum dwell before switching accel/decel mode (prevents flapping) |
| `overspeed_brake_threshold` | 0.4 m/s | — | Overspeed above this adds progressive brake |
| `overspeed_brake_max` | 0.18 | — | Maximum progressive brake command |
| `accel_mode_threshold` | 0.15 | — | Throttle level above which accel mode is asserted |

**Startup and launch:**

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `startup_ramp_seconds` | 2.5 s | [1.5, 4.0] | Duration of soft startup phase |
| `startup_speed_threshold` | 2.0 m/s | — | Speed above which startup ramp disengages |
| `startup_throttle_cap` | 0.15 | — | Hard throttle cap during startup |
| `low_speed_accel_limit` | 0.6 m/s² | — | S1-M39: accel limit below `low_speed_speed_threshold` |
| `launch_throttle_cap_min/max` | 0.15–0.55 | — | Throttle cap range during active launch phase |
| `launch_throttle_ramp_seconds` | 2.5 s | — | Launch throttle ramp duration |

**Coast and mode transitions:**

| Parameter | Default | Effect |
|---|---|---|
| `coast_hold_seconds` | 0.7 s | Duration of zero-command coast phase during mode transitions |
| `coast_throttle_kp` | 0.1 | P gain for maintaining coasting speed |
| `coast_throttle_max` | 0.12 | Max throttle during coast hold |
| `straight_throttle_cap` | 0.45 | Throttle ceiling on straights (allows more on curves) |

**Accel tracking inner loop** (`accel_tracking_enabled = true`):

| Parameter | Default | Effect |
|---|---|---|
| `accel_tracking_error_scale` | 0.35 | Scale for inner accel error correction |
| `accel_pid_kp/ki/kd` | 0.5/0.02/0.02 | Inner accel PID gains |
| `continuous_accel_control` | true | Maintains continuous accel profile |
| `accel_target_smoothing_alpha` | 0.86 | EMA on accel target |

**Output shaping:**

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `throttle_curve_gamma` | 1.4 | [1.0, 2.0] | Power curve shaping (>1 = more responsive near zero) |
| `speed_drag_gain` | 0.035 | — | Aerodynamic drag feedforward coefficient |
| `speed_smoothing` | 0.3 | [0.1, 0.5] | EMA on measured vehicle speed estimate |
| `min_throttle_hold` | 0.16 | — | Minimum throttle to maintain when above `min_throttle_hold_speed` |

---

## § 6 — Trajectory Reference Planning

### Tier 1 — Primary Levers

#### `trajectory.reference_lookahead`
**Default:** `9.0` | **Units:** metres | **Range:** [5.0, 15.0]

Baseline Pure Pursuit lookahead distance. This is the fallback value when
`dynamic_reference_lookahead` is true and speed interpolation is unavailable.
In practice, the speed table governs lookahead during normal operation.
- Shorter lookahead: tighter response, more sensitive to perception noise, better curve tracking
- Longer lookahead: smoother, less oscillation on straights, slower curve entry response

---

#### `trajectory.dynamic_reference_lookahead`
**Default:** `true`

Enables the speed-indexed lookahead table (`reference_lookahead_speed_table`). Leave `true`
for Pure Pursuit mode — the static `reference_lookahead` value is intended as a fallback only.

---

#### `trajectory.reference_lookahead_speed_table`

Speed → lookahead distance lookup, interpolated linearly. Current tuned values:

| Speed (m/s) | Lookahead (m) |
|---|---|
| 0.0 | 7.0 |
| 4.47 | 9.0 |
| 6.71 | 10.0 |
| 8.94 | 11.0 |
| 11.18 | 14.0 |
| 13.41 | 15.0 |

At cruise (12 m/s): lookahead ≈ 14.4 m. Increasing the high-speed entries smooths
straight-line tracking; decreasing them sharpens curve response but may cause oscillation.
**Validate any table change with A/B batch before promoting.**

---

#### `trajectory.reference_lookahead_speed_table_entry`

Shorter-lookahead table used during curve approach phase (when track geometry is available).
Protects turn entries by giving tighter response when the vehicle is approaching a curve.
Active when `reference_lookahead_entry_source = "track"`.

---

#### `trajectory.reference_smoothing`
**Default:** `0.75` | **Range:** [0.50, 0.90]
**Comfort gate: `lateral_error_p95`, `time_in_lane_centered`**

EMA coefficient on the reference point (target x position for PP). Higher = smoother
trajectory but introduces lag when the lane position changes.
- 0.75: ~4-frame half-life at 20 Hz ≈ 0.2s lag
- 0.90: ~7-frame half-life — noticeably smoothed, potentially slow on sharp turns
- 0.50: very responsive but passes through more perception noise

---

#### `trajectory.lane_smoothing_alpha`
**Default:** `0.8` | **Range:** [0.50, 0.90]

EMA coefficient applied to lane polynomial coefficients before reference computation.
Higher = less noisy lane estimate, but slower to track real lane changes.
Works in tandem with `reference_smoothing` — both affect lateral tracking quality.

---

#### `trajectory.ref_x_rate_limit`
**Default:** `0.22` | **Units:** m/frame | **Range:** [0.10, 0.40]

Maximum lateral movement of the reference point per frame. Prevents fast perception
updates from causing steering spikes. At 20 Hz, 0.22 m/frame = 4.4 m/s lateral reference
velocity — fast enough for real lane changes but suppresses frame-to-frame jitter.

---

#### `trajectory.target_speed`
**Default:** `12.0` | **Units:** m/s
**Keep in sync with `control.longitudinal.target_speed` and `safety.max_speed`.**

Reference speed for the trajectory speed planner. The planner uses this when computing
speed profiles. Mismatch with the longitudinal controller's target causes conflicting
commands and oscillation.

---

#### `trajectory.curve_context_curvature_gain`
**Default:** `6.0` | **Range:** [3.0, 10.0]

Scales the curvature preview signal into the curve anticipation score. Higher = more
sensitive to upcoming curves (earlier warning to control). Raising above 8.0 can cause
the vehicle to slow pre-emptively on mild bends.

---

### Tier 2 — Subsystem Tuning

**Speed governor** (`trajectory.speed_governor.enabled = true`):

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `speed_governor.comfort_governor_max_lat_accel_g` | 0.20 g | [0.15, 0.28] | Max lateral accel ceiling for speed governance |
| `speed_governor.curvature_calibration_scale` | 2.5 | — | Scales estimated curvature for lateral accel computation |
| `speed_governor.curve_preview_max_decel_mps2` | 1.2 m/s² | [0.8, 2.0] | S1-M39: max decel for proactive curve speed reduction |
| `speed_governor.curve_preview_min_curvature` | 0.002 | — | Min curvature to activate curve preview governor |

**Curve speed preview** (`curve_speed_preview_enabled = true`):

| Parameter | Default | Effect |
|---|---|---|
| `curve_speed_preview_lookahead_scale` | 1.6 | How far ahead to look for speed reduction (× reference_lookahead) |
| `curve_speed_preview_distance` | 10.0 m | Fixed preview distance |
| `curve_speed_preview_decel` | 1.8 m/s² | Max decel for preview-driven speed reduction |
| `curve_speed_preview_min_curvature` | 0.002 | Min curvature to activate preview speed reduction |

**Steering speed guard** (`steering_speed_guard_enabled = true`):

| Parameter | Default | Effect |
|---|---|---|
| `steering_speed_guard_threshold` | 0.55 | Steering magnitude above which speed is guarded |
| `steering_speed_guard_scale` | 0.7 | Speed scaling factor when steering guard active |
| `steering_speed_guard_min_speed` | 3.0 m/s | Min speed below which guard does not reduce speed further |

**Reference point slew limiting:**

| Parameter | Default | Effect |
|---|---|---|
| `reference_lookahead_slew_rate_m_per_frame` | 0.25 m | Max lookahead change per frame |
| `reference_lookahead_slew_rate_shorten_m_per_frame` | 0.35 m | Faster shortening (entering curve) |
| `reference_lookahead_slew_rate_lengthen_m_per_frame` | 0.20 m | Slower lengthening (exiting curve) |

**Tight curve handling:**

| Parameter | Default | Effect |
|---|---|---|
| `reference_lookahead_tight_curvature_threshold` | 0.01 | Curvature above which tight-curve scale applies |
| `reference_lookahead_tight_scale` | 0.35 | Lookahead multiplier for tight curves |
| `reference_lookahead_tight_blend_band` | 0.006 | Blend band around tight threshold |

**Multi-lookahead blend** (`multi_lookahead_enabled = true`):

| Parameter | Default | Effect |
|---|---|---|
| `multi_lookahead_far_scale` | 1.4 | Far point = reference_lookahead × 1.4 |
| `multi_lookahead_blend_alpha` | 0.25 | Weight of far point in blend |
| `multi_lookahead_curvature_threshold` | 0.01 | Min curvature to activate far-point blending |

**Curve phase scheduler** (`curve_scheduler_mode = "phase_shadow"` — telemetry only):

| Parameter | Default | Effect |
|---|---|---|
| `curve_scheduler_mode` | `phase_shadow` | `phase_shadow`: compute phase, no actuation; `phase_active`: drives entry blend |
| `curve_phase_on/off` | 0.45/0.30 | Score thresholds for phase entry/exit |
| `curve_phase_ema_alpha` | 0.45 | EMA on phase score |
| `curve_phase_commit_on` | 0.55 | Score threshold for "committed to curve entry" |

**Curve anticipation** (`curve_anticipation_shadow_only = true` — telemetry only):

| Parameter | Default | Effect |
|---|---|---|
| `curve_anticipation_enabled` | true | Currently shadow-only; no actuation until `shadow_only = false` |
| `curve_anticipation_score_on/off` | 0.65/0.50 | Anticipation activation thresholds |
| `curve_anticipation_curvature_min/max` | 0.0025/0.015 | Curvature range for anticipation signal |

**Reference rate limiting on turns:**

| Parameter | Default | Effect |
|---|---|---|
| `ref_sign_flip_threshold` | 0.12 m | Lateral error above which reference sign flip is considered |
| `ref_sign_flip_disable_on_turn` | true | Suppresses sign flip during turns (heading error active) |
| `ref_x_rate_limit_curvature_scale_max` | 5.0 | Max rate limit relaxation on curves |

**Speed limit preview and stability:**

| Parameter | Default | Effect |
|---|---|---|
| `speed_limit_preview_distance` | 12.0 m | Lookahead for speed limit enforcement |
| `speed_limit_preview_decel` | 2.5 m/s² | Max decel for speed limit compliance |
| `speed_limit_preview_stability_frames` | 10 | Frames of stable reading before limit applies |

---

## § 7 — Safety Limits

All parameters in this section are Tier 1. Changes here affect the OOL event gate and
emergency stop behavior — validate carefully.

| Parameter | Default | Units | Description |
|---|---|---|---|
| `safety.max_speed` | 12.0 | m/s | Hard speed ceiling. **Keep in sync with target speeds.** |
| `safety.max_lateral_error` | 2.0 | m | Soft OOL threshold → triggers recovery mode |
| `safety.recovery_lateral_error` | 1.0 | m | Hysteresis: below this, recovery mode deactivates |
| `safety.emergency_stop_error` | 4.0 | m | Hard OOL threshold → immediate emergency stop |
| `safety.emergency_brake_threshold` | 1.5 | m/s | Speed excess above max triggering emergency brake |
| `safety.speed_prevention_threshold` | 0.9 | fraction | Throttle fraction above which overspeed prevention activates |
| `safety.speed_prevention_brake_amount` | 0.2 | fraction | Brake command applied by overspeed prevention |
| `safety.lane_width` | 7.0 | m | Nominal lane width assumption for boundary geometry |
| `safety.car_width` | 1.9 | m | Vehicle width used in OOL boundary calculation |
| `safety.allowed_outside_lane` | 1.0 | m | Additional margin beyond lane boundary before OOL counted |
| `safety.emergency_stop_use_gt_lane_boundaries` | true | — | Use ground truth lane boundaries for OOL (more accurate) |
| `safety.emergency_stop_gt_lane_boundary_margin` | 0.05 | m | Safety margin applied to GT boundaries |
| `safety.emergency_stop_release_speed` | 0.2 | m/s | Speed below which emergency stop self-releases |
| `safety.emergency_stop_end_run_after_seconds` | 3.0 | s | Run ends this many seconds after emergency stop |

**`out_of_lane_events` gate note:** Events are counted as ≥ 10 consecutive frames with the
vehicle outside lane boundaries. `emergency_stop_error` governs when an e-stop triggers
but OOL events are measured from the GT lane boundaries + margin, independent of e-stop.

---

## § 8 — Perception & Lane Gating

### Tier 1 — Primary Levers

#### `perception.lane_center_ema_alpha`
**Default:** `0.15` | **Range:** [0.05, 0.30]

EMA coefficient on the lane center estimate. At 0.15, a new measurement has 15% weight;
the previous estimate retains 85%. This is the primary "trust" knob for lane detection.
- Too low (< 0.08): very sluggish — real lane changes take many frames to register
- Too high (> 0.25): passes through per-frame noise more directly into the reference

---

#### `perception.lane_center_gate_m`
**Default:** `0.3` | **Units:** metres

Frame-to-frame change gate. Lane center updates larger than this value are rejected as
outliers and the previous estimate is held. Works with `lane_center_ema_alpha` as a
two-stage filter: gate rejects outliers, EMA smooths accepted updates.

---

#### `perception.low_visibility_fallback_enabled`
**Default:** `true`

Enables the stale-frame fallback path. When confidence drops below threshold or zero
lanes are detected for more than one frame, the system blends toward the last known
lane estimate. Disabling this produces immediate straight-line steering on detection failure.

---

#### `perception.low_visibility_fallback_max_consecutive_frames`
**Default:** `8` | **Units:** frames (= 0.4s at 20 Hz) | **Range:** [4, 20]

Maximum number of consecutive stale frames before fallback disengages. After this TTL,
the system holds a pure straight-line estimate. Too short → perception jitter causes frequent
fallback exits; too long → stale detection holds too long on genuine visibility loss.

---

#### `perception.model_fallback_confidence_hard_threshold`
**Default:** `0.1`

Confidence score below which CV fallback is forced regardless of other conditions.
The segmentation model returns confidence [0, 1] per frame. Below 0.1, no reliable
detection — CV SimpleLaneDetector is substituted.

---

### Tier 2 — Subsystem Tuning

**Lane gating (EMA + gate system):**

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `lane_width_ema_alpha` | 0.20 | [0.10, 0.35] | EMA on lane width estimate |
| `lane_width_gate_m` | 0.6 | — | Rejects lane width changes larger than this |
| `lane_center_ab_beta` | 0.12 | — | A/B filter beta weight for center estimate |
| `lane_width_ab_beta` | 0.08 | — | A/B filter beta weight for width estimate |
| `lane_line_change_clamp_m` | 0.25 | — | Per-frame clamp on individual lane line position changes |
| `lane_center_change_clamp_m` | 0.4 | — | Per-frame clamp on lane center |
| `lane_width_change_clamp_m` | 0.5 | — | Per-frame clamp on lane width |

**Model/detection quality:**

| Parameter | Default | Effect |
|---|---|---|
| `model_fallback_zero_lane_confidence_threshold` | 0.6 | Below this → "zero lanes detected" fallback path |
| `low_visibility_fallback_blend_alpha` | 0.35 | Blend weight of last-known vs. straight-line in fallback |
| `low_visibility_fallback_center_shift_cap_m` | 0.25 m | Maximum center shift allowed per frame in fallback |

**RANSAC polynomial fitting:**

| Parameter | Default | Effect |
|---|---|---|
| `segmentation_ransac_enabled` | true | RANSAC outlier rejection in polynomial fitting |
| `segmentation_ransac_residual_px` | 10.0 px | Inlier threshold |
| `segmentation_ransac_min_inliers` | 8 | Minimum inliers for valid fit |
| `segmentation_ransac_max_trials` | 100 | Max RANSAC iterations |
| `segmentation_fit_min_row_ratio` | 0.45 | Top crop — min pixel row for fitting (fraction of height) |
| `segmentation_fit_max_row_ratio` | 0.75 | Bottom crop — max pixel row for fitting |

**Lane side sign enforcement:**

| Parameter | Default | Effect |
|---|---|---|
| `lane_side_sign_enforce` | true | Enforces left lane x < 0, right lane x > 0 |
| `lane_side_sign_min_abs_m` | 0.25 m | Minimum absolute x before sign enforcement applies |
| `lane_side_sign_skip_curve_phase` | true | Disables enforcement during curves (sign can legitimately swap) |

---

## § 9 — Inactive / Legacy Parameters

These parameters exist in the YAML but are **not active** in the current configuration.

### Lateral PID mode (bypassed — `control_mode = pure_pursuit`)

The following params only apply when `control.lateral.control_mode = "pid"`:

| Parameter | Default | Purpose when active |
|---|---|---|
| `control.lateral.kp` | 1.1 | Lateral PID proportional gain |
| `control.lateral.ki` | 0.004 | Lateral PID integral gain |
| `control.lateral.kd` | 0.18 | Lateral PID derivative gain |
| `control.lateral.integral_limit` | 0.1 | PID integral windup limit |
| `control.lateral.deadband` | 0.05 | Lateral error deadband for PID |
| `control.lateral.heading_weight` | 0.5 | Composite error: heading contribution |
| `control.lateral.lateral_weight` | 0.5 | Composite error: crosstrack contribution |
| `control.lateral.feedback_gain_min/max` | 1.1–1.4 | Adaptive PID feedback gain range |
| `control.lateral.base_error_smoothing_alpha` | 0.7 | Error EMA in PID mode |
| `control.lateral.heading_error_smoothing_alpha` | 0.45 | Heading error EMA in PID mode |
| `control.lateral.error_clip` | 1.57 | PID input clamp (≈ π/2 rad) |

### Stanley mode (bypassed — `control_mode = pure_pursuit`)

Only apply when `control.lateral.control_mode = "stanley"`:

| Parameter | Default | Purpose when active |
|---|---|---|
| `control.lateral.stanley_k` | 1.5 | Stanley crosstrack gain |
| `control.lateral.stanley_soft_speed` | 2.0 m/s | Stanley soft speed floor |
| `control.lateral.stanley_heading_weight` | 1.0 | Heading vs. crosstrack balance in Stanley |

### Disabled experimental modes

| Parameter | Default | Notes |
|---|---|---|
| `control.lateral.curve_entry_assist_enabled` | false | Experimental entry assist; not validated |
| `control.lateral.curve_entry_schedule_enabled` | false | Experimental scheduled entry; not validated |
| `control.lateral.curve_commit_mode_enabled` | false | Experimental commit mode; not validated |
| `trajectory.curve_anticipation_shadow_only` | true | Anticipation is telemetry-only; no actuation |
| `trajectory.curve_anticipation_single_owner_mode` | false | Would give anticipation exclusive control |

### Legacy `lateral_control` top-level section

The entire `lateral_control:` section is a legacy compatibility wrapper. Its values are
overridden by `control.lateral.*` in all active code paths. Do not tune these.

| Parameter | Default |
|---|---|
| `lateral_control.heading_weight` | 0.5 |
| `lateral_control.lateral_weight` | 0.5 |
| `lateral_control.steering_smoothing_alpha` | 0.7 |
| `lateral_control.steering_rate_scale_min` | 0.8 |
| `lateral_control.steering_jerk_limit` | 0.0 |

---

## § 10 — Logging

| Parameter | Default | Effect |
|---|---|---|
| `logging.frame_log_interval` | 30 frames | How often to emit a per-frame log line (~1.5s intervals at 20 Hz) |
| `logging.log_lateral_error` | true | Include lateral error in frame log |
| `logging.log_reference_point` | true | Include reference point x in frame log |

---

## Appendix — Full Parameter Index

All 328 parameters. Format: `Parameter | Default | Unit | Tier`

Tier key: `1` = primary lever, `2` = subsystem tuning, `3` = advanced/inactive

### control.lateral

| Parameter | Default | Unit | Tier |
|---|---|---|---|
| `control.lateral.kp` | 1.1 | — | 3 (PID, inactive) |
| `control.lateral.ki` | 0.004 | — | 3 (PID, inactive) |
| `control.lateral.kd` | 0.18 | — | 3 (PID, inactive) |
| `control.lateral.max_steering` | 0.7 | rad | 2 |
| `control.lateral.deadband` | 0.05 | — | 3 (PID, inactive) |
| `control.lateral.heading_weight` | 0.5 | — | 3 (PID, inactive) |
| `control.lateral.lateral_weight` | 0.5 | — | 3 (PID, inactive) |
| `control.lateral.error_clip` | 1.57 | rad | 3 (PID, inactive) |
| `control.lateral.integral_limit` | 0.1 | — | 3 (PID, inactive) |
| `control.lateral.steering_smoothing_alpha` | 0.92 | — | 2 |
| `control.lateral.control_mode` | pure_pursuit | — | 1 |
| `control.lateral.stanley_k` | 1.5 | — | 3 (Stanley, inactive) |
| `control.lateral.stanley_soft_speed` | 2.0 | m/s | 3 (Stanley, inactive) |
| `control.lateral.stanley_heading_weight` | 1.0 | — | 3 (Stanley, inactive) |
| `control.lateral.pp_feedback_gain` | 0.075 | rad/m | 1 |
| `control.lateral.pp_max_steering_jerk` | 18.0 | norm/s² | 1 |
| `control.lateral.pp_min_lookahead` | 0.5 | m | 1 |
| `control.lateral.pp_ref_jump_clamp` | 0.5 | m | 1 |
| `control.lateral.pp_stale_decay` | 0.98 | /frame | 1 |
| `control.lateral.pp_max_steering_rate` | 0.4 | rad/frame | 1 |
| `control.lateral.feedback_gain_min` | 1.05 | — | 3 (PID, inactive) |
| `control.lateral.feedback_gain_max` | 1.25 | — | 3 (PID, inactive) |
| `control.lateral.feedback_gain_curvature_min` | 0.003 | — | 3 (PID, inactive) |
| `control.lateral.feedback_gain_curvature_max` | 0.03 | — | 3 (PID, inactive) |
| `control.lateral.base_error_smoothing_alpha` | 0.78 | — | 3 (PID, inactive) |
| `control.lateral.heading_error_smoothing_alpha` | 0.45 | — | 3 (PID, inactive) |
| `control.lateral.straight_window_frames` | 60 | frames | 3 |
| `control.lateral.straight_oscillation_high` | 0.2 | — | 3 |
| `control.lateral.straight_oscillation_low` | 0.05 | — | 3 |
| `control.lateral.curve_feedforward_gain` | 1.0 | — | 2 |
| `control.lateral.curve_feedforward_threshold` | 0.012 | — | 2 |
| `control.lateral.curve_feedforward_gain_min` | 1.0 | — | 2 |
| `control.lateral.curve_feedforward_gain_max` | 1.15 | — | 2 |
| `control.lateral.curve_feedforward_curvature_min` | 0.001 | — | 2 |
| `control.lateral.curve_feedforward_curvature_max` | 0.015 | — | 2 |
| `control.lateral.curve_feedforward_curvature_clamp` | 0.04 | — | 2 |
| `control.lateral.curvature_scale_factor` | 8.5 | — | 3 |
| `control.lateral.curvature_scale_threshold` | 0.0005 | — | 3 |
| `control.lateral.curvature_stale_hold_seconds` | 0.3 | s | 2 |
| `control.lateral.curvature_stale_hold_min_abs` | 0.0005 | — | 3 |
| `control.lateral.curvature_smoothing_alpha` | 0.5 | — | 2 |
| `control.lateral.curvature_transition_threshold` | 0.01 | — | 2 |
| `control.lateral.curvature_transition_alpha` | 0.3 | — | 2 |
| `control.lateral.curve_feedforward_bins[0].gain_scale` | 1.0 | — | 2 |
| `control.lateral.curve_feedforward_bins[1].gain_scale` | 1.3 | — | 2 |
| `control.lateral.straight_curvature_threshold` | 0.01 | — | 2 |
| `control.lateral.curve_upcoming_enter_threshold` | 0.012 | — | 2 |
| `control.lateral.curve_upcoming_exit_threshold` | 0.009 | — | 2 |
| `control.lateral.curve_upcoming_on_frames` | 2 | frames | 2 |
| `control.lateral.curve_upcoming_off_frames` | 2 | frames | 2 |
| `control.lateral.curve_phase_use_distance_track` | false | — | 3 |
| `control.lateral.curve_phase_track_name` | s_loop | — | 3 |
| `control.lateral.curve_phase_use_preview_curvature` | true | — | 2 |
| `control.lateral.curve_phase_preview_enter_threshold` | 0.01 | — | 2 |
| `control.lateral.curve_phase_preview_exit_threshold` | 0.008 | — | 2 |
| `control.lateral.curve_phase_preview_on_frames` | 1 | frames | 3 |
| `control.lateral.curve_phase_preview_off_frames` | 2 | frames | 3 |
| `control.lateral.curve_at_car_distance_min_m` | 0.0 | m | 3 |
| `control.lateral.curve_intent_single_owner_mode` | true | — | 2 |
| `control.lateral.road_curve_enter_threshold` | 0.011 | — | 2 |
| `control.lateral.road_curve_exit_threshold` | 0.009 | — | 2 |
| `control.lateral.road_straight_hold_invalid_frames` | 6 | frames | 3 |
| `control.lateral.steering_rate_curvature_min` | 0.003 | — | 3 |
| `control.lateral.steering_rate_curvature_max` | 0.015 | — | 3 |
| `control.lateral.steering_rate_scale_min` | 0.55 | — | 3 |
| `control.lateral.curve_rate_floor_moderate_error` | 0.2 | — | 2 |
| `control.lateral.curve_rate_floor_large_error` | 0.24 | — | 2 |
| `control.lateral.straight_sign_flip_error_threshold` | 0.03 | — | 3 |
| `control.lateral.straight_sign_flip_rate` | 0.5 | — | 3 |
| `control.lateral.straight_sign_flip_frames` | 6 | frames | 3 |
| `control.lateral.steering_jerk_limit` | 0.12 | — | 3 (PID mode only) |
| `control.lateral.steering_jerk_curve_scale_max` | 1.0 | — | 3 |
| `control.lateral.steering_jerk_curve_min` | 0.003 | — | 3 |
| `control.lateral.steering_jerk_curve_max` | 0.015 | — | 3 |
| `control.lateral.curve_entry_assist_enabled` | false | — | 3 (disabled) |
| `control.lateral.curve_entry_assist_error_min` | 0.3 | m | 3 (disabled) |
| `control.lateral.curve_entry_assist_heading_error_min` | 0.1 | rad | 3 (disabled) |
| `control.lateral.curve_entry_assist_curvature_min` | 0.01 | — | 3 (disabled) |
| `control.lateral.curve_entry_assist_rate_boost` | 1.12 | — | 3 (disabled) |
| `control.lateral.curve_entry_assist_jerk_boost` | 1.08 | — | 3 (disabled) |
| `control.lateral.curve_entry_assist_max_frames` | 18 | frames | 3 (disabled) |
| `control.lateral.curve_entry_assist_watchdog_rate_delta_max` | 0.22 | — | 3 (disabled) |
| `control.lateral.curve_entry_assist_rearm_frames` | 20 | frames | 3 (disabled) |
| `control.lateral.curve_entry_schedule_enabled` | false | — | 3 (disabled) |
| `control.lateral.curve_entry_schedule_frames` | 24 | frames | 3 (disabled) |
| `control.lateral.curve_entry_schedule_min_rate` | 0.22 | — | 3 (disabled) |
| `control.lateral.curve_entry_schedule_min_jerk` | 0.16 | — | 3 (disabled) |
| `control.lateral.curve_entry_schedule_min_hold_frames` | 12 | frames | 3 (disabled) |
| `control.lateral.curve_entry_schedule_min_curve_progress_ratio` | 0.2 | — | 3 (disabled) |
| `control.lateral.curve_entry_schedule_fallback_only_when_dynamic` | true | — | 3 (disabled) |
| `control.lateral.curve_entry_schedule_fallback_deficit_frames` | 6 | frames | 3 (disabled) |
| `control.lateral.curve_entry_schedule_fallback_rate_deficit_min` | 0.03 | — | 3 (disabled) |
| `control.lateral.curve_entry_schedule_fallback_rearm_cooldown_frames` | 18 | frames | 3 (disabled) |
| `control.lateral.curve_entry_schedule_handoff_transfer_ratio` | 0.68 | — | 3 (disabled) |
| `control.lateral.curve_entry_schedule_handoff_error_fall` | 0.03 | m | 3 (disabled) |
| `control.lateral.dynamic_curve_authority_enabled` | true | — | 2 |
| `control.lateral.dynamic_curve_rate_deficit_deadband` | 0.01 | — | 2 |
| `control.lateral.dynamic_curve_rate_boost_gain` | 0.6 | — | 2 |
| `control.lateral.dynamic_curve_rate_boost_max` | 0.34 | — | 2 |
| `control.lateral.dynamic_curve_jerk_boost_gain` | 2.0 | — | 2 |
| `control.lateral.dynamic_curve_jerk_boost_max_factor` | 3.8 | — | 2 |
| `control.lateral.dynamic_curve_hard_clip_boost_gain` | 1.0 | — | 2 |
| `control.lateral.dynamic_curve_hard_clip_boost_max` | 0.2 | — | 2 |
| `control.lateral.dynamic_curve_entry_governor_enabled` | true | — | 2 |
| `control.lateral.dynamic_curve_entry_governor_gain` | 0.8 | — | 2 |
| `control.lateral.dynamic_curve_entry_governor_max_scale` | 1.3 | — | 2 |
| `control.lateral.dynamic_curve_entry_governor_stale_floor_scale` | 1.15 | — | 2 |
| `control.lateral.dynamic_curve_entry_governor_exclusive_mode` | true | — | 2 |
| `control.lateral.dynamic_curve_entry_governor_anticipatory_enabled` | true | — | 2 |
| `control.lateral.dynamic_curve_entry_governor_upcoming_phase_weight` | 0.75 | — | 2 |
| `control.lateral.dynamic_curve_authority_precurve_enabled` | true | — | 2 |
| `control.lateral.dynamic_curve_authority_precurve_scale` | 1.0 | — | 2 |
| `control.lateral.dynamic_curve_single_owner_mode` | true | — | 2 |
| `control.lateral.dynamic_curve_single_owner_min_rate` | 0.22 | — | 2 |
| `control.lateral.dynamic_curve_single_owner_min_jerk` | 0.6 | — | 2 |
| `control.lateral.dynamic_curve_comfort_lat_accel_comfort_max_g` | 0.18 | g | 2 |
| `control.lateral.dynamic_curve_comfort_lat_accel_peak_max_g` | 0.25 | g | 2 |
| `control.lateral.dynamic_curve_comfort_lat_jerk_comfort_max_gps` | 0.30 | g/s | 2 |
| `control.lateral.dynamic_curve_lat_jerk_smoothing_alpha` | 0.25 | — | 2 |
| `control.lateral.dynamic_curve_lat_jerk_soft_start_ratio` | 0.60 | — | 2 |
| `control.lateral.dynamic_curve_lat_jerk_soft_floor_scale` | 0.35 | — | 2 |
| `control.lateral.dynamic_curve_speed_low_mps` | 4.0 | m/s | 2 |
| `control.lateral.dynamic_curve_speed_high_mps` | 10.0 | m/s | 2 |
| `control.lateral.dynamic_curve_speed_boost_max_scale` | 1.4 | — | 2 |
| `control.lateral.turn_feasibility_governor_enabled` | true | — | 2 |
| `control.lateral.turn_feasibility_curvature_min` | 0.002 | — | 2 |
| `control.lateral.turn_feasibility_guardband_g` | 0.015 | g | 2 |
| `control.lateral.turn_feasibility_use_peak_bound` | true | — | 2 |
| `control.lateral.curve_unwind_policy_enabled` | true | — | 2 |
| `control.lateral.curve_unwind_frames` | 12 | frames | 2 |
| `control.lateral.curve_unwind_rate_scale_start` | 1.0 | — | 2 |
| `control.lateral.curve_unwind_rate_scale_end` | 0.8 | — | 2 |
| `control.lateral.curve_unwind_jerk_scale_start` | 1.0 | — | 2 |
| `control.lateral.curve_unwind_jerk_scale_end` | 0.7 | — | 2 |
| `control.lateral.curve_unwind_integral_decay` | 0.85 | — | 2 |
| `control.lateral.curve_commit_mode_enabled` | false | — | 3 (disabled) |
| `control.lateral.curve_commit_mode_max_frames` | 24 | frames | 3 (disabled) |
| `control.lateral.curve_commit_mode_min_rate` | 0.22 | — | 3 (disabled) |
| `control.lateral.curve_commit_mode_min_jerk` | 4.0 | — | 3 (disabled) |
| `control.lateral.curve_commit_mode_transfer_ratio_target` | 0.72 | — | 3 (disabled) |
| `control.lateral.curve_commit_mode_error_fall` | 0.03 | m | 3 (disabled) |
| `control.lateral.curve_commit_mode_exit_consecutive_frames` | 4 | frames | 3 (disabled) |
| `control.lateral.curve_commit_mode_retrigger_on_dynamic_deficit` | true | — | 3 (disabled) |
| `control.lateral.curve_commit_mode_dynamic_deficit_frames` | 8 | frames | 3 (disabled) |
| `control.lateral.curve_commit_mode_dynamic_deficit_min` | 0.03 | — | 3 (disabled) |
| `control.lateral.curve_commit_mode_retrigger_cooldown_frames` | 12 | frames | 3 (disabled) |
| `control.lateral.curve_mode_speed_cap_enabled` | true | — | 2 |
| `control.lateral.curve_mode_speed_cap_mps` | 7.0 | m/s | 2 |
| `control.lateral.curve_mode_speed_cap_min_ratio` | 0.55 | — | 2 |
| `control.lateral.speed_gain_min_speed` | 4.0 | m/s | 2 |
| `control.lateral.speed_gain_max_speed` | 10.0 | m/s | 2 |
| `control.lateral.speed_gain_min` | 1.0 | — | 2 |
| `control.lateral.speed_gain_max` | 1.5 | — | 2 |
| `control.lateral.speed_gain_curvature_min` | 0.002 | — | 2 |
| `control.lateral.speed_gain_curvature_max` | 0.008 | — | 2 |

### control.longitudinal

| Parameter | Default | Unit | Tier |
|---|---|---|---|
| `control.longitudinal.kp` | 0.25 | — | 2 |
| `control.longitudinal.ki` | 0.02 | — | 2 |
| `control.longitudinal.kd` | 0.01 | — | 2 |
| `control.longitudinal.max_accel` | 1.2 | m/s² | 1 |
| `control.longitudinal.max_decel` | 2.4 | m/s² | 1 |
| `control.longitudinal.max_jerk` | 0.7 | m/s³ | 2 |
| `control.longitudinal.max_jerk_min` | 1.0 | m/s³ | 1 |
| `control.longitudinal.max_jerk_max` | 4.0 | m/s³ | 1 |
| `control.longitudinal.jerk_error_min` | 0.5 | m/s | 2 |
| `control.longitudinal.jerk_error_max` | 3.0 | m/s | 2 |
| `control.longitudinal.accel_feedforward_gain` | 1.0 | — | 2 |
| `control.longitudinal.decel_feedforward_gain` | 1.0 | — | 2 |
| `control.longitudinal.speed_for_jerk_alpha` | 0.8 | — | 3 |
| `control.longitudinal.jerk_cooldown_frames` | 8 | frames | 1 |
| `control.longitudinal.jerk_cooldown_scale` | 0.4 | — | 1 |
| `control.longitudinal.speed_error_to_accel_gain` | 0.1 | — | 2 |
| `control.longitudinal.speed_error_deadband` | 0.05 | m/s | 2 |
| `control.longitudinal.speed_error_gain_under` | 0.12 | — | 2 |
| `control.longitudinal.speed_error_gain_over` | 0.08 | — | 2 |
| `control.longitudinal.overspeed_accel_zero_threshold` | 0.2 | m/s | 2 |
| `control.longitudinal.overspeed_brake_threshold` | 0.4 | m/s | 2 |
| `control.longitudinal.overspeed_brake_min` | 0.03 | — | 2 |
| `control.longitudinal.overspeed_brake_max` | 0.18 | — | 2 |
| `control.longitudinal.overspeed_brake_gain` | 0.25 | — | 2 |
| `control.longitudinal.accel_mode_threshold` | 0.15 | — | 2 |
| `control.longitudinal.decel_mode_threshold` | 0.1 | — | 2 |
| `control.longitudinal.speed_error_accel_threshold` | 0.2 | m/s | 2 |
| `control.longitudinal.speed_error_brake_threshold` | -0.6 | m/s | 2 |
| `control.longitudinal.mode_switch_min_time` | 0.4 | s | 2 |
| `control.longitudinal.coast_hold_seconds` | 0.7 | s | 2 |
| `control.longitudinal.coast_throttle_kp` | 0.1 | — | 2 |
| `control.longitudinal.coast_throttle_max` | 0.12 | — | 2 |
| `control.longitudinal.straight_throttle_cap` | 0.45 | — | 2 |
| `control.longitudinal.straight_curvature_threshold` | 0.003 | — | 2 |
| `control.longitudinal.accel_tracking_enabled` | true | — | 2 |
| `control.longitudinal.accel_tracking_error_scale` | 0.35 | — | 2 |
| `control.longitudinal.accel_pid_kp` | 0.5 | — | 2 |
| `control.longitudinal.accel_pid_ki` | 0.02 | — | 2 |
| `control.longitudinal.accel_pid_kd` | 0.02 | — | 2 |
| `control.longitudinal.throttle_curve_gamma` | 1.4 | — | 2 |
| `control.longitudinal.throttle_curve_min` | 0.0 | — | 3 |
| `control.longitudinal.speed_drag_gain` | 0.035 | — | 2 |
| `control.longitudinal.accel_target_smoothing_alpha` | 0.86 | — | 2 |
| `control.longitudinal.continuous_accel_control` | true | — | 2 |
| `control.longitudinal.continuous_accel_deadband` | 0.05 | — | 2 |
| `control.longitudinal.startup_ramp_seconds` | 2.5 | s | 2 |
| `control.longitudinal.startup_accel_limit` | 0.5 | m/s² | 1 |
| `control.longitudinal.startup_speed_threshold` | 2.0 | m/s | 2 |
| `control.longitudinal.startup_throttle_cap` | 0.15 | — | 2 |
| `control.longitudinal.startup_disable_accel_feedforward` | true | — | 3 |
| `control.longitudinal.low_speed_accel_limit` | 0.6 | m/s² | 2 |
| `control.longitudinal.low_speed_speed_threshold` | 2.0 | m/s | 2 |
| `control.longitudinal.target_speed` | 12.0 | m/s | 1 |
| `control.longitudinal.max_speed` | 12.0 | m/s | 1 |
| `control.longitudinal.speed_smoothing` | 0.3 | — | 2 |
| `control.longitudinal.speed_deadband` | 0.03 | m/s | 3 |
| `control.longitudinal.throttle_limit_threshold` | 0.9 | — | 3 |
| `control.longitudinal.throttle_reduction_factor` | 0.15 | — | 3 |
| `control.longitudinal.brake_aggression` | 2.0 | — | 2 |
| `control.longitudinal.throttle_rate_limit` | 0.04 | /frame | 1 |
| `control.longitudinal.brake_rate_limit` | 0.05 | /frame | 1 |
| `control.longitudinal.throttle_smoothing_alpha` | 0.7 | — | 1 |
| `control.longitudinal.min_throttle_when_accel` | 0.1 | — | 2 |
| `control.longitudinal.min_throttle_hold` | 0.16 | — | 2 |
| `control.longitudinal.min_throttle_hold_speed` | 4.0 | m/s | 2 |
| `control.longitudinal.target_speed_slew_rate_up` | 1.5 | m/s² | 1 |
| `control.longitudinal.target_speed_slew_rate_down` | 1.8 | m/s² | 1 |
| `control.longitudinal.target_speed_slew_error_min` | 0.5 | m/s | 2 |
| `control.longitudinal.target_speed_slew_error_max` | 3.0 | m/s | 2 |
| `control.longitudinal.target_speed_slew_rate_up_min` | 0.6 | m/s² | 2 |
| `control.longitudinal.target_speed_slew_rate_up_max` | 2.0 | m/s² | 2 |
| `control.longitudinal.target_speed_slew_rate_down_min` | 1.0 | m/s² | 2 |
| `control.longitudinal.target_speed_slew_rate_down_max` | 2.0 | m/s² | 2 |
| `control.longitudinal.target_speed_stop_threshold` | 0.5 | m/s | 3 |
| `control.longitudinal.target_speed_restart_ramp_seconds` | 2.5 | s | 3 |
| `control.longitudinal.launch_throttle_stop_threshold` | 2.0 | m/s | 2 |
| `control.longitudinal.launch_throttle_ramp_seconds` | 2.5 | s | 2 |
| `control.longitudinal.launch_throttle_cap_min` | 0.15 | — | 2 |
| `control.longitudinal.launch_throttle_cap_max` | 0.55 | — | 2 |
| `control.longitudinal.launch_throttle_rearm_hysteresis` | 0.5 | — | 3 |
| `control.longitudinal.launch_throttle_stop_hold_seconds` | 0.5 | s | 3 |

### trajectory

| Parameter | Default | Unit | Tier |
|---|---|---|---|
| `trajectory.trajectory_source` | planner | — | 3 |
| `trajectory.lookahead_distance` | 20.0 | m | 2 |
| `trajectory.point_spacing` | 1.0 | m | 3 |
| `trajectory.target_speed` | 12.0 | m/s | 1 |
| `trajectory.reference_lookahead` | 9.0 | m | 1 |
| `trajectory.dynamic_reference_lookahead` | true | — | 1 |
| `trajectory.curve_scheduler_mode` | phase_shadow | — | 2 |
| `trajectory.reference_lookahead_speed_table` | (see §6) | — | 1 |
| `trajectory.reference_lookahead_speed_table_entry` | (see §6) | — | 2 |
| `trajectory.reference_lookahead_entry_source` | track | — | 2 |
| `trajectory.reference_lookahead_entry_preview_scale` | 1.5 | — | 2 |
| `trajectory.reference_lookahead_entry_track_name` | "" | — | 3 |
| `trajectory.reference_lookahead_entry_preview_only_fallback` | true | — | 2 |
| `trajectory.reference_lookahead_entry_preview_curvature_min` | 0.003 | — | 2 |
| `trajectory.reference_lookahead_entry_preview_curvature_max` | 0.015 | — | 2 |
| `trajectory.reference_lookahead_entry_hysteresis_on` | 0.010 | — | 2 |
| `trajectory.reference_lookahead_entry_hysteresis_off` | 0.008 | — | 2 |
| `trajectory.reference_lookahead_entry_distance_start_m` | 20.0 | m | 2 |
| `trajectory.reference_lookahead_entry_distance_end_m` | 6.0 | m | 2 |
| `trajectory.curve_phase_preview_curvature_min` | 0.003 | — | 2 |
| `trajectory.curve_phase_preview_curvature_max` | 0.015 | — | 2 |
| `trajectory.curve_phase_path_curvature_min` | 0.003 | — | 2 |
| `trajectory.curve_phase_path_curvature_max` | 0.015 | — | 2 |
| `trajectory.curve_phase_rise_min` | 0.0005 | — | 2 |
| `trajectory.curve_phase_rise_max` | 0.006 | — | 2 |
| `trajectory.curve_phase_time_to_curve_min_s` | 0.8 | s | 2 |
| `trajectory.curve_phase_time_to_curve_max_s` | 3.0 | s | 2 |
| `trajectory.curve_phase_confidence_min` | 0.35 | — | 2 |
| `trajectory.curve_phase_confidence_max` | 0.80 | — | 2 |
| `trajectory.curve_phase_confidence_floor_scale` | 0.6 | — | 2 |
| `trajectory.curve_phase_ema_alpha` | 0.45 | — | 2 |
| `trajectory.curve_phase_on` | 0.45 | — | 2 |
| `trajectory.curve_phase_off` | 0.30 | — | 2 |
| `trajectory.curve_phase_commit_on` | 0.55 | — | 2 |
| `trajectory.curve_phase_commit_min_frames` | 4 | frames | 2 |
| `trajectory.curve_phase_entry_floor` | 0.15 | — | 2 |
| `trajectory.curve_phase_commit_floor` | 0.30 | — | 2 |
| `trajectory.curve_phase_rearm_hold_frames` | 4 | frames | 2 |
| `trajectory.curve_context_preview_horizon_m` | 25.0 | m | 2 |
| `trajectory.curve_context_preview_step_m` | 1.0 | m | 3 |
| `trajectory.curve_context_curve_start_curvature` | 0.006 | — | 2 |
| `trajectory.curve_context_curvature_gain` | 6.0 | — | 1 |
| `trajectory.reference_lookahead_slew_rate_m_per_frame` | 0.25 | m/frame | 2 |
| `trajectory.reference_lookahead_slew_rate_shorten_m_per_frame` | 0.35 | m/frame | 2 |
| `trajectory.reference_lookahead_slew_rate_lengthen_m_per_frame` | 0.20 | m/frame | 2 |
| `trajectory.reference_lookahead_speed_table_straight` | [] | — | 3 |
| `trajectory.reference_lookahead_speed_table_curve` | [] | — | 3 |
| `trajectory.reference_lookahead_curve_blend_curvature_min` | 0.006 | — | 3 |
| `trajectory.reference_lookahead_curve_blend_curvature_max` | 0.015 | — | 3 |
| `trajectory.reference_lookahead_min` | 2.5 | m | 2 |
| `trajectory.reference_lookahead_scale_min` | 0.80 | — | 2 |
| `trajectory.reference_lookahead_speed_min` | 4.0 | m/s | 2 |
| `trajectory.reference_lookahead_speed_max` | 10.0 | m/s | 2 |
| `trajectory.reference_lookahead_curvature_min` | 0.002 | — | 2 |
| `trajectory.reference_lookahead_curvature_max` | 0.015 | — | 2 |
| `trajectory.reference_lookahead_tight_curvature_threshold` | 0.01 | — | 2 |
| `trajectory.reference_lookahead_tight_scale` | 0.35 | — | 2 |
| `trajectory.reference_lookahead_tight_blend_band` | 0.006 | — | 2 |
| `trajectory.curve_intent_speed_guardrail_enabled` | true | — | 2 |
| `trajectory.curve_intent_speed_guardrail_intent_min` | 0.55 | — | 2 |
| `trajectory.curve_intent_speed_guardrail_confidence_max` | 0.55 | — | 2 |
| `trajectory.curve_intent_speed_guardrail_cap_mps` | 7.0 | m/s | 2 |
| `trajectory.curve_anticipation_enabled` | true | — | 2 |
| `trajectory.curve_anticipation_shadow_only` | true | — | 3 (shadow) |
| `trajectory.curve_anticipation_single_owner_mode` | false | — | 3 (shadow) |
| `trajectory.curve_anticipation_score_ema_alpha` | 0.35 | — | 2 |
| `trajectory.curve_anticipation_score_on` | 0.65 | — | 2 |
| `trajectory.curve_anticipation_score_off` | 0.50 | — | 2 |
| `trajectory.curve_anticipation_on_frames` | 2 | frames | 2 |
| `trajectory.curve_anticipation_off_frames` | 2 | frames | 2 |
| `trajectory.curve_anticipation_curvature_min` | 0.0025 | — | 2 |
| `trajectory.curve_anticipation_curvature_max` | 0.015 | — | 2 |
| `trajectory.curve_anticipation_heading_delta_min_rad` | 0.14 | rad | 2 |
| `trajectory.curve_anticipation_heading_delta_max_rad` | 0.40 | rad | 2 |
| `trajectory.curve_anticipation_far_rise_min` | 0.30 | — | 2 |
| `trajectory.curve_anticipation_far_rise_max` | 1.00 | — | 2 |
| `trajectory.curve_anticipation_entry_weight_on` | 0.65 | — | 2 |
| `trajectory.curve_anticipation_entry_weight_full` | 0.90 | — | 2 |
| `trajectory.curve_anticipation_entry_weight_min_active` | 0.35 | — | 2 |
| `trajectory.curve_anticipation_entry_weight_max` | 1.0 | — | 2 |
| `trajectory.curve_anticipation_entry_weight_merge_with_preview` | true | — | 2 |
| `trajectory.use_vehicle_frame_lookahead_ref` | false | — | 3 |
| `trajectory.vehicle_frame_lookahead_scale` | 1.0 | — | 3 |
| `trajectory.vehicle_frame_lookahead_scale_left` | 1.0 | — | 3 |
| `trajectory.vehicle_frame_lookahead_scale_right` | 1.05 | — | 3 |
| `trajectory.curvature_smoothing_enabled` | true | — | 2 |
| `trajectory.curvature_smoothing_window_m` | 12.0 | m | 2 |
| `trajectory.curvature_smoothing_min_speed` | 2.0 | m/s | 2 |
| `trajectory.straight_speed_smoothing_alpha` | 0.2 | — | 2 |
| `trajectory.straight_speed_smoothing_curvature_threshold` | 0.003 | — | 2 |
| `trajectory.straight_speed_slew_rate_up` | 0.4 | — | 2 |
| `trajectory.straight_target_hold_seconds` | 1.0 | s | 2 |
| `trajectory.straight_target_hold_limit_delta` | 0.3 | m/s | 2 |
| `trajectory.target_speed_blend_window_m` | 20.0 | m | 3 |
| `trajectory.target_speed_blend_reset_seconds` | 0.5 | s | 3 |
| `trajectory.max_lateral_accel` | 3.0 | m/s² | 2 |
| `trajectory.min_curve_speed` | 3.0 | m/s | 2 |
| `trajectory.curvature_bins[0].max_lateral_accel` | 3.0 | m/s² | 2 |
| `trajectory.curvature_bins[0].min_curve_speed` | 3.0 | m/s | 2 |
| `trajectory.curvature_bins[1].max_lateral_accel` | 3.0 | m/s² | 2 |
| `trajectory.curvature_bins[1].min_curve_speed` | 3.0 | m/s | 2 |
| `trajectory.curvature_bins[2].max_lateral_accel` | 3.0 | m/s² | 2 |
| `trajectory.curvature_bins[2].min_curve_speed` | 3.0 | m/s | 2 |
| `trajectory.speed_limit_slew_rate` | 1.0 | — | 2 |
| `trajectory.speed_limit_slew_rate_up` | 2.5 | — | 2 |
| `trajectory.speed_limit_slew_rate_down` | 1.5 | — | 2 |
| `trajectory.speed_limit_preview_distance` | 12.0 | m | 2 |
| `trajectory.speed_limit_preview_decel` | 2.5 | m/s² | 2 |
| `trajectory.speed_limit_preview_stability_frames` | 10 | frames | 2 |
| `trajectory.speed_limit_preview_change_delta` | 0.3 | m/s | 2 |
| `trajectory.speed_limit_preview_hold_curvature_threshold` | 0.01 | — | 2 |
| `trajectory.speed_limit_preview_hold_release_delta` | 0.5 | m/s | 2 |
| `trajectory.speed_limit_preview_hold_release_frames` | 3 | frames | 2 |
| `trajectory.speed_limit_preview_hold_min_distance_margin` | 0.5 | m | 2 |
| `trajectory.curve_speed_preview_enabled` | true | — | 2 |
| `trajectory.curve_speed_preview_lookahead_scale` | 1.6 | — | 2 |
| `trajectory.curve_speed_preview_distance` | 10.0 | m | 2 |
| `trajectory.curve_speed_preview_decel` | 1.8 | m/s² | 2 |
| `trajectory.curve_speed_preview_min_curvature` | 0.002 | — | 2 |
| `trajectory.steering_speed_guard_enabled` | true | — | 2 |
| `trajectory.steering_speed_guard_threshold` | 0.55 | — | 2 |
| `trajectory.steering_speed_guard_scale` | 0.7 | — | 2 |
| `trajectory.steering_speed_guard_min_speed` | 3.0 | m/s | 2 |
| `trajectory.curvature_limit_min_abs` | 0.001 | — | 3 |
| `trajectory.min_speed_limit_ratio` | 0.6 | — | 2 |
| `trajectory.min_speed_floor` | 3.0 | m/s | 2 |
| `trajectory.min_speed_curvature_max` | 0.02 | — | 2 |
| `trajectory.speed_planner.enabled` | true | — | 2 |
| `trajectory.speed_planner.max_accel` | 1.2 | m/s² | 2 |
| `trajectory.speed_planner.max_decel` | 2.4 | m/s² | 2 |
| `trajectory.speed_planner.max_jerk` | 0.8 | m/s³ | 2 |
| `trajectory.speed_planner.max_jerk_min` | 0.5 | m/s³ | 2 |
| `trajectory.speed_planner.max_jerk_max` | 1.0 | m/s³ | 2 |
| `trajectory.speed_planner.jerk_error_min` | 0.3 | m/s | 2 |
| `trajectory.speed_planner.jerk_error_max` | 2.0 | m/s | 2 |
| `trajectory.speed_planner.use_speed_dependent_limits` | true | — | 2 |
| `trajectory.speed_planner.accel_speed_min/max` | 0.0/12.0 | m/s | 3 |
| `trajectory.speed_planner.max_accel_at_speed_min/max` | 1.2/1.2 | m/s² | 2 |
| `trajectory.speed_planner.max_decel_at_speed_min/max` | 2.5/2.5 | m/s² | 2 |
| `trajectory.speed_planner.enforce_desired_speed_slew` | false | — | 3 |
| `trajectory.speed_planner.min_speed` | 0.0 | m/s | 2 |
| `trajectory.speed_planner.launch_speed_floor` | 2.0 | m/s | 2 |
| `trajectory.speed_planner.launch_speed_floor_threshold` | 1.0 | m/s | 2 |
| `trajectory.speed_planner.reset_gap_seconds` | 0.5 | s | 2 |
| `trajectory.speed_planner.sync_speed_threshold` | 50.0 | — | 3 |
| `trajectory.speed_planner.speed_error_gain` | 0.0 | — | 3 |
| `trajectory.speed_planner.speed_error_max_delta` | 0.4 | m/s | 3 |
| `trajectory.speed_planner.speed_error_deadband` | 0.2 | m/s | 2 |
| `trajectory.speed_planner.speed_limit_bias` | -0.2 | m/s | 2 |
| `trajectory.speed_planner.sync_under_target` | false | — | 3 |
| `trajectory.speed_planner.sync_under_target_error` | 0.2 | m/s | 3 |
| `trajectory.image_width` | 640 | px | 3 |
| `trajectory.image_height` | 480 | px | 3 |
| `trajectory.camera_fov` | 66.0 | degrees | 3 |
| `trajectory.camera_height` | 1.4 | m | 3 |
| `trajectory.camera_offset_x` | 0.0 | m | 3 |
| `trajectory.distance_scaling_factor` | 0.875 | — | 3 |
| `trajectory.bias_correction_threshold` | 10.0 | — | 3 |
| `trajectory.reference_smoothing` | 0.75 | — | 1 |
| `trajectory.lane_smoothing_alpha` | 0.8 | — | 1 |
| `trajectory.center_spline_enabled` | true | — | 2 |
| `trajectory.center_spline_degree` | 2 | — | 3 |
| `trajectory.center_spline_samples` | 30 | — | 3 |
| `trajectory.center_spline_alpha` | 0.8 | — | 2 |
| `trajectory.x_clip_enabled` | true | — | 2 |
| `trajectory.x_clip_limit_m` | 15.0 | m | 2 |
| `trajectory.trajectory_eval_y_min_ratio` | 0.1 | — | 3 |
| `trajectory.centerline_midpoint_mode` | coeff_average | — | 3 |
| `trajectory.projection_model` | legacy | — | 3 |
| `trajectory.calibrated_fx_px` | 0.0 | px | 3 |
| `trajectory.calibrated_fy_px` | 0.0 | px | 3 |
| `trajectory.calibrated_cx_px` | 0.0 | px | 3 |
| `trajectory.calibrated_cy_px` | 0.0 | px | 3 |
| `trajectory.direct_reference_curvature_gain` | 2.0 | — | 2 |
| `trajectory.target_lane` | right | — | 3 |
| `trajectory.target_lane_width_m` | 3.6 | m | 3 |
| `trajectory.ref_sign_flip_threshold` | 0.12 | m | 2 |
| `trajectory.ref_x_rate_limit` | 0.22 | m/frame | 1 |
| `trajectory.ref_x_rate_limit_turn_heading_min_abs_rad` | 0.04 | rad | 2 |
| `trajectory.ref_x_rate_limit_turn_scale_max` | 4.0 | — | 2 |
| `trajectory.ref_sign_flip_disable_on_turn` | true | — | 2 |
| `trajectory.ref_sign_flip_disable_heading_min_abs_rad` | 0.02 | rad | 2 |
| `trajectory.ref_x_rate_limit_precurve_enabled` | true | — | 2 |
| `trajectory.ref_x_rate_limit_precurve_heading_min_abs_rad` | 0.03 | rad | 2 |
| `trajectory.ref_x_rate_limit_precurve_scale_max` | 6.0 | — | 2 |
| `trajectory.multi_lookahead_enabled` | true | — | 2 |
| `trajectory.multi_lookahead_far_scale` | 1.4 | — | 2 |
| `trajectory.multi_lookahead_blend_alpha` | 0.25 | — | 2 |
| `trajectory.multi_lookahead_curvature_threshold` | 0.01 | — | 2 |
| `trajectory.multi_lookahead_curve_heading_smoothing_alpha` | 0.75 | — | 2 |
| `trajectory.multi_lookahead_curve_heading_min_abs_rad` | 0.05 | rad | 2 |
| `trajectory.heading_zero_quad_threshold` | 0.005 | — | 2 |
| `trajectory.heading_zero_curvature_guard` | 0.001 | — | 2 |
| `trajectory.smoothing_alpha_curve_reduction` | 0.15 | — | 2 |
| `trajectory.ref_x_rate_limit_curvature_min` | 0.001 | — | 2 |
| `trajectory.ref_x_rate_limit_curvature_scale_max` | 5.0 | — | 2 |
| `trajectory.traj_heading_zero_gate_center_a_on_abs_max` | 0.004 | — | 3 |
| `trajectory.traj_heading_zero_gate_center_a_off_abs_max` | 0.008 | — | 3 |
| `trajectory.traj_heading_zero_gate_heading_on_abs_rad` | 0.035 | rad | 3 |
| `trajectory.traj_heading_zero_gate_heading_off_abs_rad` | 0.061 | rad | 3 |
| `trajectory.dynamic_effective_horizon_enabled` | true | — | 2 |
| `trajectory.dynamic_effective_horizon_min_m` | 3.0 | m | 2 |
| `trajectory.dynamic_effective_horizon_max_m` | 20.0 | m | 2 |
| `trajectory.dynamic_effective_horizon_speed_scale` | 0.35 | — | 2 |
| `trajectory.dynamic_effective_horizon_curvature_scale` | 0.15 | — | 2 |
| `trajectory.dynamic_effective_horizon_farfield_scale_min` | 0.85 | — | 2 |
| `trajectory.dynamic_effective_horizon_confidence_floor` | 0.6 | — | 2 |
| `trajectory.dynamic_effective_horizon_confidence_fallback` | 1.0 | — | 2 |
| `trajectory.far_band_contribution_cap_enabled` | true | — | 2 |
| `trajectory.far_band_contribution_cap_start_m` | 12.0 | m | 2 |
| `trajectory.far_band_contribution_cap_gain` | 0.45 | — | 2 |
| `trajectory.speed_horizon_guardrail_enabled` | true | — | 2 |
| `trajectory.speed_horizon_guardrail_time_headway_s` | 0.8 | s | 2 |
| `trajectory.speed_horizon_guardrail_margin_m` | 1.0 | m | 2 |
| `trajectory.speed_horizon_guardrail_min_speed_mps` | 3.0 | m/s | 2 |
| `trajectory.speed_horizon_guardrail_gain` | 1.0 | — | 2 |
| `trajectory.speed_governor.enabled` | true | — | 2 |
| `trajectory.speed_governor.comfort_governor_max_lat_accel_g` | 0.20 | g | 2 |
| `trajectory.speed_governor.comfort_governor_min_speed` | 3.0 | m/s | 2 |
| `trajectory.speed_governor.curvature_calibration_scale` | 2.5 | — | 2 |
| `trajectory.speed_governor.curvature_history_frames` | 5 | frames | 2 |
| `trajectory.speed_governor.curve_preview_enabled` | true | — | 2 |
| `trajectory.speed_governor.curve_preview_lookahead_scale` | 1.6 | — | 2 |
| `trajectory.speed_governor.curve_preview_max_decel_mps2` | 1.2 | m/s² | 2 |
| `trajectory.speed_governor.curve_preview_min_curvature` | 0.002 | — | 2 |
| `trajectory.lane_position_smoothing_alpha` | 0.20 | — | 2 |
| `trajectory.ref_point_jump_clamp_x` | 0.15 | m | 2 |
| `trajectory.reference_x_max_abs` | 3.2 | m | 2 |

### safety

| Parameter | Default | Unit | Tier |
|---|---|---|---|
| `safety.max_speed` | 12.0 | m/s | 1 |
| `safety.emergency_brake_threshold` | 1.5 | m/s | 1 |
| `safety.speed_prevention_threshold` | 0.9 | fraction | 2 |
| `safety.speed_prevention_brake_threshold` | 0.95 | fraction | 2 |
| `safety.speed_prevention_brake_amount` | 0.2 | fraction | 2 |
| `safety.max_lateral_error` | 2.0 | m | 1 |
| `safety.recovery_lateral_error` | 1.0 | m | 1 |
| `safety.emergency_stop_error` | 4.0 | m | 1 |
| `safety.emergency_stop_release_speed` | 0.2 | m/s | 2 |
| `safety.emergency_stop_end_run_after_seconds` | 3.0 | s | 2 |
| `safety.emergency_stop_use_gt_lane_boundaries` | true | — | 2 |
| `safety.emergency_stop_gt_lane_boundary_margin` | 0.05 | m | 2 |
| `safety.lane_width` | 7.0 | m | 1 |
| `safety.car_width` | 1.9 | m | 1 |
| `safety.allowed_outside_lane` | 1.0 | m | 1 |

### logging

| Parameter | Default | Unit | Tier |
|---|---|---|---|
| `logging.frame_log_interval` | 30 | frames | 3 |
| `logging.log_lateral_error` | true | — | 3 |
| `logging.log_reference_point` | true | — | 3 |

### perception

| Parameter | Default | Unit | Tier |
|---|---|---|---|
| `perception.record_segmentation_mask` | false | — | 3 |
| `perception.low_visibility_fallback_enabled` | true | — | 1 |
| `perception.low_visibility_fallback_blend_alpha` | 0.35 | — | 2 |
| `perception.low_visibility_fallback_max_consecutive_frames` | 8 | frames | 1 |
| `perception.low_visibility_fallback_center_shift_cap_m` | 0.25 | m | 2 |
| `perception.segmentation_fit_min_row_ratio` | 0.45 | fraction | 2 |
| `perception.segmentation_fit_max_row_ratio` | 0.75 | fraction | 2 |
| `perception.segmentation_ransac_enabled` | true | — | 2 |
| `perception.segmentation_ransac_residual_px` | 10.0 | px | 2 |
| `perception.segmentation_ransac_min_inliers` | 8 | — | 2 |
| `perception.segmentation_ransac_max_trials` | 100 | — | 3 |
| `perception.model_fallback_confidence_hard_threshold` | 0.1 | — | 1 |
| `perception.model_fallback_zero_lane_confidence_threshold` | 0.6 | — | 2 |
| `perception.lane_width_min_m` | 1.0 | m | 3 |
| `perception.lane_width_max_m` | 10.0 | m | 3 |
| `perception.lane_center_change_clamp_m` | 0.4 | m | 2 |
| `perception.lane_width_change_clamp_m` | 0.5 | m | 2 |
| `perception.lane_line_change_clamp_m` | 0.25 | m | 2 |
| `perception.lane_side_sign_enforce` | true | — | 2 |
| `perception.lane_side_sign_min_abs_m` | 0.25 | m | 2 |
| `perception.lane_side_sign_skip_curve_phase` | true | — | 2 |
| `perception.lane_side_sign_curve_curvature_min` | 0.006 | — | 2 |
| `perception.lane_side_sign_preview_curvature_min` | 0.006 | — | 2 |
| `perception.lane_center_ema_alpha` | 0.15 | — | 1 |
| `perception.lane_width_ema_alpha` | 0.2 | — | 2 |
| `perception.lane_center_gate_m` | 0.3 | m | 1 |
| `perception.lane_width_gate_m` | 0.6 | m | 2 |
| `perception.lane_center_ab_beta` | 0.12 | — | 2 |
| `perception.lane_width_ab_beta` | 0.08 | — | 2 |

### lateral_control (legacy — inactive)

| Parameter | Default | Unit | Tier |
|---|---|---|---|
| `lateral_control.heading_weight` | 0.5 | — | 3 (inactive) |
| `lateral_control.lateral_weight` | 0.5 | — | 3 (inactive) |
| `lateral_control.steering_smoothing_alpha` | 0.7 | — | 3 (inactive) |
| `lateral_control.steering_rate_scale_min` | 0.8 | — | 3 (inactive) |
| `lateral_control.steering_jerk_limit` | 0.0 | — | 3 (inactive) |
