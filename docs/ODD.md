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
| Linear MPC (LMPC) | v > 4.0 m/s | κ < 0.020 (R > 50m) | Straights and gentle curves on ACC scenarios |
| Nonlinear MPC (NMPC) | **DISABLED** | — | Kinematic model can't compete with PP; needs dynamic bicycle + sysid tire params (see `project_nmpc_sloop_tuning.md`) |
| Regime switching | — | Hysteresis bands | PP↔LMPC: 30-frame hold, 15-frame blend. Lateral-accel budget `κ×v²` unified selector (threshold 1.5) |

**Step 4 change (2026-03-17):** LMPC is now the primary lateral controller above ~4 m/s. Curvature guard (`mpc_max_curvature=0.020`) prevents MPC on R≤50m curves. Per-track overlays raise `mpc_min_speed_mps` where needed.

**Step 5 change (2026-03-23):** NMPC added as high-speed regime (scipy SLSQP, analytical adjoint gradient, 10–14ms solve). Validated at 25 m/s on autobahn_30 (97.5/100). `lmpc_nmpc_blend_frames=30` and `lmpc_nmpc_min_hold_frames=40` reduce regime chatter at the LMPC↔NMPC boundary.

**Stage 2 update (2026-04-09):** Regime inversion — PP is now primary for curves (not MPC). See `project_mpc_primary.md` for the reversal rationale. NMPC disabled pending dynamic bicycle model implementation.

**PP recovery term (2026-04-18):** Continuous smoothstep lateral-error correction upstream of rate/jerk limiters replaces orchestrator post-limiter multipliers. Hairpin_15 Control 80→100, jerk 30.08→18.0. See `project_pp_recovery_term.md`.

**Decomposed MPC reference (2026-04-20):** H2 regression resolved — `mpc_e_lat_reference_mode: at_car_gt_legacy` + `mpc_ff_decomposition_enabled: true` separate the bundled `lookahead_offset = d + Ld·sin(e_h) + Ld²·κ/2` into independently-tunable cost terms. H2 restored 79 → 99.4 with ACC active. Frenet shadow telemetry retained for diagnostics. See `project_h2_damping_on_straights.md` and plan `.claude/plans/decomposed-mpc-reference.md`.

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
9. **ACC stop-scenario brake authority insufficient** — H5 (stop-go) and G2 (stop-on-grade) collide on hard-deceleration half-cycles: ACC state machine correctly enters `COLLAPSED_GAP_STOP + acc_request_estop` but `idm_comfortable_decel_mps2: 3.0` is not enough to prevent gap collapse to 0.10m. H5 scores 59.0, G2 scores 79.0. Separate investigation from the lateral stack; shared root cause across both scenarios. H2 (steady lead-follow) resolved to 99.4 2026-04-20 via decomposed-MPC fix. A1 (autobahn steady) scores 99.6 but records 12.4% radar detection rate because lead is usually beyond the 60m `detection_range_m` — expected physics.
10. **s_loop PP ceiling** — 0.111 m RMSE floor is kinematic model-plant mismatch on PP; LMPC cannot improve curves here. 2DOF / preview weighting implemented but DISABLED (no RMSE gain). See `project_rmse_structural.md`.
11. **NMPC disabled** — Kinematic bicycle model cannot compete with PP on validated tracks. Re-enabling requires dynamic bicycle model with sysid tire params. See `project_nmpc_sloop_tuning.md`.
12. **Unity 6 + macOS 26 shader crash** — `WarmupAllShaders` crashes on Mac Mini + macOS 26. Disabled in `ShaderPrewarmer.cs`; async shader compilation replaces it.
13. **Unity segfault ceiling (partially refuted 2026-04-20)** — Historical observation of segfaults after ~36 consecutive launch/kill cycles. Empirical 7-cycle sweep with zero segfaults suggests the ceiling was caused by synchronous `WarmupAllShaders` (fixed in `ShaderPrewarmer.cs`, see `project_unity6_shader_crash.md` and `project_unity_cycle_ceiling_lifted.md`). Rebuild-every-3 ritual may be unnecessary; full 36+ cycle validation still pending.

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

### ACC Scenarios (Step 5, re-validated 2026-04-20 with `acc.enabled: true`)

| Scenario | Status | Score | Notes |
|---|---|---|---|
| H2 (steady lead-follow) | **PASS** | 99.4 | Restored from 79.0 via decomposed-MPC fix (61cc446) + ACC activation (92391bb) |
| H6 (close-gap approach) | **PASS** | 97.5 | Min gap 12m, min TTC 7.67s; no collisions |
| A1 (autobahn steady) | **PASS** | 99.6 | Lead mostly beyond 60m `detection_range_m` (12.4% detection rate — expected) |
| G1 (grade following) | **PASS** | 96.2 | Traj 86 from pre-existing grade residual (not ACC-related) |
| H5 (stop-go sinusoidal) | **BLOCKED** | 59.0 | Harness bimodal failure reproduced twice (curve-cap=100%, target pinned 2.09 m/s, bridge 34s stale); ACC never engages to exercise brake authority. Deferred pending third-hit escalation — see `project_h5_harness_deferred.md` |
| G2 (stop-on-grade) | **PASS (Safety)** | 79.0 | 885 → **2 e-stops** after `acc-idm-accel-plumbing` plan landed 2026-04-21 (390f057). Score capped at 79 by pre-existing SignalIntegrity heading-suppression −25 (T-HEAD-SUPPR) and residual 2 Safety e-stops from late-transitioning ACC state machine (T-ACC-STATE-LATE). Pre-emergency IDM min-gate landed 2026-04-22 shadow-mode (533cd40, plan `greedy-swimming-naur.md`); active-mode validation pending Phase G2. Expected 2 → 0 e-stops on activation. |

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
| 2026-04-20 | Decomposed-MPC reference landed (61cc446): H2 restored 79→99.4. ACC capability flag flipped to `enabled: true` (92391bb) and 6 scenarios re-validated — H2/H6/A1/G1 PASS, H5/G2 FAIL on stop-collision brake-authority (separate investigation). |
| 2026-04-21 | ACC emergency brake authority landed (plan `acc-idm-accel-plumbing.md`, commits A/B/C/C.1/D through 390f057). Peels comfort limiters (decel/jerk clip, cooldown scale, accel EMA) off brake path on `EMERGENCY_BRAKE`/`TTC_ESTOP`/`COLLAPSED_GAP_STOP`. G2 e-stops 885 → 2 (99.8% reduction). H5 validation blocked on harness bimodal failure; two-hit tripwire set. Follow-ups filed: T-HEAD-SUPPR, T-ACC-STATE-LATE, T-ACC-UNIFY, T-B1-HDF5-PLUMBING, T-H5-HARNESS. |
| 2026-04-22 | ACC pre-emergency IDM min-gate landed shadow-mode (commit 533cd40, plan `greedy-swimming-naur.md`). Routes `acc_idm_accel_mps2` as pre-limiter `min(speed_p, idm)` clip during `ACC_ACTIVE`/`CUTOUT` states — covers the 10–15 frame window before state flips to EMERGENCY_BRAKE. Phase 0.5 confirmed G2 onsets show ≥10 consecutive would-gate frames each (max Δ +5.24 m/s²); H2 cruise no-op (max Δ +0.13 m/s²). Default mode `shadow` (telemetry-only); active-mode Phase G2 validation pending. Expected G2: 2 → 0 e-stops. |
