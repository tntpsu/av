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
- **Simulation only** — Unity 6000.3 LTS environment (upgraded from 2021.3 LTS; see `project_unity6_shader_crash.md`)

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
| Pure Pursuit (PP) | Any | Any | Primary controller (Stage 2); PP recovery term handles perturbation |
| Linear MPC (LMPC) | v > 4.0 m/s | κ < 0.020 (R > 50m) | Straights and gentle curves on ACC scenarios (H2 currently regressed) |
| Nonlinear MPC (NMPC) | **DISABLED** | — | Kinematic model can't compete with PP; needs dynamic bicycle + sysid tire params (see `project_nmpc_sloop_tuning.md`) |
| Regime switching | — | Hysteresis bands | PP↔LMPC: 30-frame hold, 15-frame blend. Lateral-accel budget `κ×v²` unified selector (threshold 1.5) |

**Step 4 change (2026-03-17):** LMPC is now the primary lateral controller above ~4 m/s. Curvature guard (`mpc_max_curvature=0.020`) prevents MPC on R≤50m curves. Per-track overlays raise `mpc_min_speed_mps` where needed.

**Step 5 change (2026-03-23):** NMPC added as high-speed regime (scipy SLSQP, analytical adjoint gradient, 10–14ms solve). Validated at 25 m/s on autobahn_30 (97.5/100). `lmpc_nmpc_blend_frames=30` and `lmpc_nmpc_min_hold_frames=40` reduce regime chatter at the LMPC↔NMPC boundary.

**Stage 2 update (2026-04-09):** Regime inversion — PP is now primary for curves (not MPC). See `project_mpc_primary.md` for the reversal rationale. NMPC disabled pending dynamic bicycle model implementation.

**PP recovery term (2026-04-18):** Continuous smoothstep lateral-error correction upstream of rate/jerk limiters replaces orchestrator post-limiter multipliers. Hairpin_15 Control 80→100, jerk 30.08→18.0. See `project_pp_recovery_term.md`.

**Frenet MPC reference shadow telemetry (2026-04-19):** `control/mpc_e_lat_frenet_linearized_m` logs Frenet `d` every frame for diagnostic purposes. **NOT ACTIVE** — activation regressed H2 79→59 due to removed PD-preview damping in legacy lookahead signal. See `project_frenet_mpc_reference.md`.

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
- **Simulation only** — Unity 6000.3 LTS, not real-world deployment

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
9. **H2 ACC scenario regressed to 79.0** — commit 7e3caf0 (2026-04-05) fed lookahead-projection offset as `e_lat(0)`; Frenet shadow telemetry landed 2026-04-19 but activation blocked pending MPC retuning (removed preview damping caused oscillation runaway Q1→Q4).
10. **s_loop PP ceiling** — 0.111 m RMSE floor is kinematic model-plant mismatch on PP; LMPC cannot improve curves here. 2DOF / preview weighting implemented but DISABLED (no RMSE gain). See `project_rmse_structural.md`.
11. **NMPC disabled** — Kinematic bicycle model cannot compete with PP on validated tracks. Re-enabling requires dynamic bicycle model with sysid tire params. See `project_nmpc_sloop_tuning.md`.
12. **Unity 6 + macOS 26 shader crash** — `WarmupAllShaders` crashes on Mac Mini + macOS 26. Disabled in `ShaderPrewarmer.cs`; async shader compilation replaces it.
13. **Unity segfault ceiling** — After ~36 consecutive launch/kill cycles, Unity player segfaults. Rebuild between track-sweep batches. See `project_training_pipeline_stability.md`.

### Scoring threshold update (Stage 2)

All layers must be ≥95 to pass (not 60/80). See `feedback_layer_score_threshold.md`. This is stricter than the comfort-gate baseline and reflects the "all-layers-≥95 goal" currently tracked on 5 frozen tracks.

---

## Validated Tracks

| Track | Grade Range | Min Radius | Speed Target | Current Score | Config |
|---|---|---|---|---|---|
| s_loop | Flat | R40 | 12 m/s | 99.1 | `mpc_sloop.yaml` |
| highway_65 | Flat | R800 | 15 m/s | 99.5 | `mpc_highway.yaml` |
| hairpin_15 | Flat | R15 | 4 m/s (capped) | 98.7 (2026-04-18, +7.1 from PP recovery term) | `mpc_hairpin.yaml` |
| mixed_radius | Flat | R40–R200 | 12 m/s | 98.7 | `mpc_mixed.yaml` |
| sweeping_highway | Flat | R300+ | 15 m/s | — | `mpc_highway.yaml` |
| hill_highway | ±5% (straights only) | R100 | 12 m/s | 97.7 | base config (no overlay — q_lat auto-derived) |
| autobahn_30 | Flat | R600 | 25 m/s | 97.5 (NMPC baseline; NMPC now disabled) | `mpc_autobahn.yaml` |

### ACC Scenarios (Step 5, regressed — not in all-layers-≥95 goal)

| Scenario | Status | Notes |
|---|---|---|
| H2 (steady lead-follow) | **79.0** | Regressed from 98.1 by commit 7e3caf0; Frenet activation blocked |
| H5, H6, A1, G1, G2 | regressed | Likely shared root cause with H2; awaits MPC retuning |

---

## Revision History

| Date | Change |
|---|---|
| 2026-03-16 | Initial ODD document. Step 3B: grade E2E fixes. |
| 2026-03-17 | Step 4: MPC as primary controller (curvature guard, new regime table). q_lat auto-derive. hill_highway validated at 89.6/100. |
| 2026-03-23 | Step 5: NMPC added (scipy SLSQP, analytical adjoint, v > 21 m/s). Speed ceiling raised to 25 m/s. autobahn_30 validated at 97.5/100. `lmpc_nmpc_blend_frames/min_hold_frames` added to regime selector. |
| 2026-03-27 | Step 5 ACC scope updated: single-actor wrong-target rejection supported for adjacent-lane same-direction and oncoming straight validation. Mixed traffic remains outside the current ODD. |
| 2026-04-05 | Commit 7e3caf0 landed lookahead-projection `e_lat(0)` — caused H2 regression 98.1→79.0 (documented for traceability). |
| 2026-04-09 | Stage 2: regime inversion — PP is now primary for curves, MPC is secondary. NMPC disabled pending dynamic bicycle model. See `project_mpc_primary.md`. |
| 2026-04-18 | PP recovery term landed (hairpin_15 Control 80→100, jerk 30.08→18.0). Orchestrator post-limiter multipliers retired. Unity version upgrade: 2021.3 → 6000.3 LTS. Scoring threshold tightened to all-layers-≥95. |
| 2026-04-19 | Frenet MPC reference shadow-mode telemetry landed; active mode FAILED validation (H2 79→59 oscillation runaway). Shadow retained for diagnostic use; activation deferred pending MPC retuning. |
