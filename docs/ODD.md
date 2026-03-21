# Operational Design Domain (ODD)

Defines where, when, and how this AV system is designed to operate.

---

## System Capabilities

- **Primary function:** Lane-keeping on known closed-loop tracks
- **Speed governed:** 0–15 m/s (configurable per track)
- **No intersections, no other vehicles, no pedestrians**
- **Simulation only** — Unity 2021.3 LTS environment

---

## Track Constraints

| Parameter | Limit | Source |
|---|---|---|
| Max grade | ±10% (GRADE_MAX_SAFE_PCT) | `tools/scoring_registry.py` |
| Min curve radius | R ≥ 10m (κ ≤ 0.10) | `tests/test_track_coverage.py` |
| Max curve radius | R ≤ 1000m (κ ≥ 0.001) | `tests/test_track_coverage.py` |
| Speed range | 0–15 m/s | `control/mpc_controller.py` (v_max) |
| Road width | 7.2m standard | Track YAML default |
| Track type | Closed loop only | Elevation/heading must close |

---

## Sensor Assumptions

- **Single forward camera** at 30 FPS
- **Segmentation-based lane detection** (trained model, active by default)
- **CV fallback** available via `--use-cv` flag
- **No LIDAR, radar, or GPS**
- **Ground truth reporter** provides `roadGrade` from track geometry
- **Pitch/roll telemetry** reflects chassis orientation, not road surface (Unity WheelCollider suspension keeps chassis level)

---

## Control Modes

| Mode | Speed Range | Curvature | Activation |
|---|---|---|---|
| Stanley | v < 4.5 m/s | Any | Low-speed fallback (parking, hairpin entry) |
| Pure Pursuit (PP) | Any | κ ≥ 0.020 (R ≤ 50m) | Tight curves; MPC solver failure fallback |
| Linear MPC | v > 4.0 m/s | κ < 0.020 (R > 50m) | Primary controller for R50+ curves and straights |
| Regime switching | — | Hysteresis bands | PP ↔ MPC: speed hysteresis + curvature guard + 30-frame hold |

**Step 4 change (2026-03-17):** MPC is now the primary lateral controller above ~4 m/s, not a highway-only mode. The curvature guard (`mpc_max_curvature=0.020`) prevents MPC on R≤50m curves where linear MPC validity breaks down. Per-track config overlays can raise the `mpc_min_speed_mps` threshold for tracks where MPC should not activate (e.g., s_loop/hairpin: 10.0 m/s; mixed_radius: 9.0 m/s).

**q_lat auto-derived from track curvature (Step 4):** `mpc_q_lat` is automatically scaled from the base value (0.5 at κ=0.002) using `q_lat = base × (κ_mpc_active / 0.002)^0.75`. No per-track manual override needed.

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
- **No dynamic obstacles** — no other vehicles, pedestrians, or debris
- **Simulation only** — Unity 2021.3 LTS, not real-world deployment

---

## Known Limitations

1. **Pitch telemetry is dead on slopes** — Unity WheelCollider suspension keeps chassis level. Use `roadGrade` (from GroundTruthReporter) as the authoritative grade source.
2. **Grade FF gain calibrated to Unity physics** — The 1.8× gain compensates for Unity-specific rolling resistance. Real-world deployment would need re-calibration.
3. **No banking support** — Road superelevation is not modeled. Lateral dynamics assume flat cross-section.
4. **Curvature preview requires map data** — Lane polynomial curvature amplifies noise quadratically at distance. Map curvature via `_map_curvature_at_distance()` is exact.
5. **Segmentation model requires training** — Default model must be trained on track-specific data. Untrained model falls back to CV pipeline.

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

---

## Revision History

| Date | Change |
|---|---|
| 2026-03-16 | Initial ODD document. Step 3B: grade E2E fixes. |
| 2026-03-17 | Step 4: MPC as primary controller (curvature guard, new regime table). q_lat auto-derive. hill_highway validated at 89.6/100. |
