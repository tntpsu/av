# Operational Design Domain (ODD)

Defines where, when, and how this AV system is designed to operate.

---

## System Capabilities

- **Primary function:** Lane-keeping on known closed-loop tracks
- **Speed governed:** 0–25 m/s (configurable per track; autobahn validated at 25 m/s)
- **Single scripted traffic actor supported for Step 5 ACC validation**
- **ACC lead following supported for one same-lane lead actor on validated highway scenarios**
- **Single-actor wrong-target rejection supported for:**
  - adjacent-lane same-direction actor
  - oncoming straight actor
- **No mixed traffic yet** — cannot currently follow a lead while a second traffic actor is present
- **No intersections, no pedestrians**
- **Simulation only** — Unity 6000.3 LTS environment

---

## Track Constraints

| Parameter | Limit | Source |
|---|---|---|
| Max grade | ±10% (GRADE_MAX_SAFE_PCT) | `tools/scoring_registry.py` |
| Min curve radius | R ≥ 10m (κ ≤ 0.10) | `tests/test_track_coverage.py` |
| Max curve radius | R ≤ 1000m (κ ≥ 0.001) | `tests/test_track_coverage.py` |
| Speed range | 0–25 m/s | `config/av_stack_config.yaml` (target_speed) |
| Road width | 7.2m standard | Track YAML default |
| Track type | Closed loop only | Elevation/heading must close |

---

## Sensor Assumptions

- **Single forward camera** at 30 FPS
- **Segmentation-based lane detection** (trained model, active by default)
- **CV fallback** available via `--use-cv` flag
- **Simulated forward radar/lead telemetry** for Step 5 ACC scenarios
- **Lead association is producer-side and single-target only**
- **No LIDAR or GPS**
- **Ground truth reporter** provides `roadGrade` from track geometry
- **Pitch/roll telemetry** reflects chassis orientation, not road surface (Unity WheelCollider suspension keeps chassis level)

---

## Control Modes

| Mode | Speed Range | Curvature | Activation |
|---|---|---|---|
| Stanley | v < 4.5 m/s | Any | Low-speed fallback (parking, hairpin entry) |
| Pure Pursuit (PP) | Any | κ ≥ 0.020 (R ≤ 50m) | Tight curves; MPC solver failure fallback |
| Linear MPC (LMPC) | v > 4.0 m/s | κ < 0.020 (R > 50m) | Primary controller for R50+ curves and straights |
| Nonlinear MPC (NMPC) | v > 21 m/s (production) | κ < 0.020 | High-speed regime (exact sin/tan bicycle model; SLSQP solver) |
| Regime switching | — | Hysteresis bands | PP↔LMPC: 30-frame hold, 15-frame blend. LMPC↔NMPC: 40-frame hold, 30-frame blend |

**Step 4 change (2026-03-17):** LMPC is now the primary lateral controller above ~4 m/s. Curvature guard (`mpc_max_curvature=0.020`) prevents MPC on R≤50m curves. Per-track overlays raise `mpc_min_speed_mps` where needed.

**Step 5 change (2026-03-23):** NMPC added as high-speed regime (scipy SLSQP, analytical adjoint gradient, 10–14ms solve). Validated at 25 m/s on autobahn_30 (97.5/100). `lmpc_nmpc_blend_frames=30` and `lmpc_nmpc_min_hold_frames=40` reduce regime chatter at the LMPC↔NMPC boundary.

**q_lat auto-derived from track curvature (Step 4+5):** Both `mpc_q_lat` (LMPC) and `nmpc_q_lat` (NMPC) are automatically scaled via centripetal acceleration formula `q_lat = base × (v²×κ / v_ref²×κ_ref)`. No per-track manual override needed.

### Grade Compensation

- **Feedforward** in both PP and MPC longitudinal paths
- **Source:** `roadGrade` from GroundTruthReporter (NOT pitch telemetry)
- **EMA smoothing:** `grade_ema_alpha` (default 0.3, hill tracks use 0.15)
- **MPC clamp:** `grade_clamp_rad = 0.15` (~8.5°)
- **FF gain:** `grade_ff_gain` (default 1.0, Unity physics calibrated at 1.8)
  - Unity WheelCollider applies rolling resistance + tire friction that scales with slope
  - Simple `g·sin(θ)` only models the gravity component
  - Gain of 1.8 compensates for the additional Unity physics forces

---

## Environmental Constraints

- **Dry road** — no wet/icy conditions modeled
- **Daylight** — no night/low-light scenarios
- **No weather effects** — no rain, fog, glare
- **At most one scripted traffic actor in runtime scenarios**
- **No mixed traffic** — no simultaneous lead + distractor traffic yet
- **No intersections or urban traffic interactions**
- **Simulation only** — Unity 2021.3 LTS, not real-world deployment

---

## Known Limitations

1. **Pitch telemetry is dead on slopes** — Unity WheelCollider suspension keeps chassis level. Use `roadGrade` (from GroundTruthReporter) as the authoritative grade source.
2. **Grade FF gain calibrated to Unity physics** — The 1.8× gain compensates for Unity-specific rolling resistance. Real-world deployment would need re-calibration.
3. **No banking support** — Road superelevation is not modeled. Lateral dynamics assume flat cross-section.
4. **Curvature preview requires map data** — Lane polynomial curvature amplifies noise quadratically at distance. Map curvature via `_map_curvature_at_distance()` is exact.
5. **Segmentation model requires training** — Default model must be trained on track-specific data. Untrained model falls back to CV pipeline.
6. **Wrong-target traffic coverage is still single-actor** — adjacent-lane and oncoming reject scenarios are supported in isolation, but not with a simultaneous lead vehicle.
7. **Mixed traffic is not yet in the ODD** — multi-actor association/ranking is still planned work.
8. **Wrong-target reject scenarios are contract checks, not clean trajectory-quality references** — current free-flow highway reject runs can expose an unrelated high-speed mild-curve lateral weakness.

---

## Validated Tracks

| Track | Grade Range | Min Radius | Speed Target | Config |
|---|---|---|---|---|
| s_loop | Flat | R40 | 12 m/s | `mpc_sloop.yaml` |
| highway_65 | Flat | R800 | 15 m/s | `mpc_highway.yaml` |
| hairpin_15 | Flat | R15 | 4 m/s (capped) | `mpc_hairpin.yaml` |
| mixed_radius | Flat | R40–R200 | 12 m/s | `mpc_mixed.yaml` |
| sweeping_highway | Flat | R300+ | 15 m/s | `mpc_highway.yaml` |
| hill_highway | ±5% (straights only) | R100 | 12 m/s | base config (no overlay — q_lat auto-derived) |
| autobahn_30 | Flat | R600 | 25 m/s | `mpc_autobahn.yaml` / `mpc_autobahn_nmpc_test.yaml` |

---

## Revision History

| Date | Change |
|---|---|
| 2026-03-16 | Initial ODD document. Step 3B: grade E2E fixes. |
| 2026-03-17 | Step 4: MPC as primary controller (curvature guard, new regime table). q_lat auto-derive. hill_highway validated at 89.6/100. |
| 2026-03-23 | Step 5: NMPC added (scipy SLSQP, analytical adjoint, v > 21 m/s). Speed ceiling raised to 25 m/s. autobahn_30 validated at 97.5/100. `lmpc_nmpc_blend_frames/min_hold_frames` added to regime selector. |
| 2026-03-27 | Step 5 ACC scope updated: single-actor wrong-target rejection supported for adjacent-lane same-direction and oncoming straight validation. Mixed traffic remains outside the current ODD. |
