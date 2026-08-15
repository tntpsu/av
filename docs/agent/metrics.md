# Metrics — What We Measure, Why, and What's Missing

**Last updated:** 2026-08-15
**Companion docs:** `docs/agent/performance.md` (cadence/latency), `docs/ODD.md` (operating envelope)

This is the reasoning layer behind `tools/scoring_registry.py`. The registry
holds the numbers; this holds *why those numbers*, and where they mislead.

---

## 1. The scoring model in one table

Overall score = weighted sum of six layers, then capped if any layer goes yellow.

| Layer | Weight | Answers | Main deductions |
|---|---|---|---|
| Safety | 29.4% | Did we avoid harm? | Out-of-lane events, emergency stops, collisions |
| Trajectory | 27.6% | Did we follow the path? | Lateral Error RMSE (curv-adj), Lateral P95 |
| Control | 14.7% | Was actuation clean? | Steering jerk, oscillation growth |
| Perception | 12.9% | Did we see correctly? | Lane detection, stale data, instability, jitter |
| LongitudinalComfort | 7.4% | Was speed control smooth? | Accel P95, jerk P95 |
| SignalIntegrity | 8.0% | Were signals trustworthy? | Heading suppression, reference jitter |

**Pass bar: every layer ≥ 95.** The 60/80 thresholds in
`test_scoring_regression.py` are drift-detection floors, not the pass bar — see
`CLAUDE.md`.

---

## 2. Gate thresholds and their basis

| Gate | Value | Basis | Confidence |
|---|---|---|---|
| Accel P95 | ≤ 3.0 m/s² | Comfort convention | Reasonable |
| Jerk P95 | ≤ 6.0 m/s³ | Comfort convention | Reasonable |
| Lateral RMSE (adj) | ≤ 0.40 m | Lane-keeping requirement | Reasonable |
| Centered frames | ≥ 70% | Project convention | Arbitrary |
| Steering jerk | ≤ 20 (norm/s²) | Actuator limit | Empirical |
| Emergency stops | 0 | Safety | Correct |
| ACC TTC min | ≥ 2.0 s | Safety convention | Reasonable |
| ACC collisions | 0 | Safety | Correct |
| ACC detection | ≥ 95% | Sensor requirement | Reasonable |
| ACC gap RMSE | ≤ 35 m | IDM equilibrium physics | Derived — good |
| Layer pass | ≥ 95 | User policy | Deliberate |

---

## 3. Known metric defects — history

Every one of these produced a confident, wrong number. Recorded so the class of
error is recognisable, not just the instances.

| Metric | Defect | Effect | Status |
|---|---|---|---|
| `_sign_flips_per_min` | Hardcoded `fps=30` on a 13 FPS stack | Every rate inflated **2.3×** | Fixed 2026-08-14 |
| `_sign_flips_per_min` | Zero guard `s[s==0]=0` was a **no-op** | `+ → 0 → +` counted 2 spurious flips | Fixed 2026-08-14 |
| `_sign_flips_per_min` | No magnitude dead-band | Counts zero-crossings of a signal idling at −0.01 m/s² | **Open** — see §5 |
| `stream_front_unity_dt_ms` | Subtracts Unity sim clock from a different clock | Reported 9.3 s "staleness"; true frame age 39.5 ms | Open (T-PERF-METRIC-1) |
| `cadence_breakdown` QUEUE_BACKLOG | Fires on `depth ≈ capacity` for a ring buffer designed to sit full | False alarm | Open (T-PERF-METRIC-2) |
| `cadence_breakdown` PACKET_FALLBACK | Flags configured `packet_shadow` mode as a fault | False alarm | Open (T-PERF-METRIC-2) |

**The recurring pattern: a metric names a physical quantity but computes
something else.** Before trusting any diagnostic, read the function that emits
it (`feedback_read_source_function_before_designing_fix`).

---

## 4. Saturating caps destroy ranking

| Cap | Value | Consequence |
|---|---|---|
| `ACC_SCORE_GAP_RMSE_PENALTY_CAP` | 50 pts | 55 m and 200 m gap error score identically |
| `ACC_SCORE_OSC_PENALTY_CAP` | 30 pts | Every oscillating scenario looks equally bad |
| `ACC_SCORE_JERK_PENALTY_CAP` | 50 pts | — |
| `ACC_SCORE_HUNTING_PENALTY_CAP` | 30 pts | — |
| `ACC_SCORE_BANGBANG_PENALTY_CAP` | 20 pts | — |
| `STEERING_JERK_PENALTY_CAP` | 18.0 | hairpin pinned here in both A/B arms |

A cap means **the worst performers are indistinguishable, and improvement is
invisible until you drop below the cap** — precisely the region where guidance
matters most. Measured 2026-08-14: 3 of 10 ACC scenarios were still at a cap
even after the fps fix.

**Proposed:** replace hard caps with soft-knee (log or asymptotic) scaling so
severity keeps ranking. Filed as T-METRIC-UNCAP.

---

## 5. Frequency blindness — the largest validity gap

Every comfort gate is an **amplitude percentile**. Human discomfort depends far
more on **frequency**.

| Band | Range | What lives there | Our coverage |
|---|---|---|---|
| Motion sickness (ISO 2631-1 MSDV, `W_f`) | 0.1–0.5 Hz | Slow weave, speed hunting | **None until 2026-08-15** |
| Ride comfort (ISO 2631-1, `W_d`/`W_k`) | 0.5–80 Hz | Road roughness, shake | Accel/jerk P95 (unweighted) |

Measured on `highway_h3` (`recording_20260814_154040`):

| Axis | Dominant frequency | In sickness band? |
|---|---|---|
| Lateral path weave | **0.243 Hz** (4.1 s period) | **Yes** |
| Longitudinal speed | **0.157 Hz** (6.4 s period) | **Yes** |

Both sit inside the motion-sickness band and **below** the band our P95 gates
address. A 0.24 Hz weave and a 3 Hz shake with identical P95 score identically
but feel completely different — one causes nausea, the other feels like a rough
road. This is why a weave visible to the naked eye was invisible to every
existing number.

**Addressed:** `tools/analyze/analyze_ride_comfort.py` (MSDV, `a_w` RMS,
dominant frequency, both axes). Tests: `tests/test_ride_comfort.py`.
Not yet wired into the scoring layers — see §7.

---

## 6. What we do well (vs industry)

| Practice | Industry reference | Us |
|---|---|---|
| Module-level attribution | Nuro module metrics | ✅ six-layer breakdown |
| Solver/compute health | standard | ✅ solve time, feasibility, fallback, regime chatter |
| Collision / TTC / near-miss | standard | ✅ present and correct |
| Frozen regression baselines | standard | ✅ `scoring_baselines.json` + golden recordings |
| Per-recording provenance | standard | ✅ HDF5 `recording_provenance` |

---

## 7. Open gaps

| Gap | Why it matters | Task |
|---|---|---|
| MSDV not in the scoring layers | Metric exists but doesn't gate anything | T-METRIC-MSDV-WIRE |
| Speed / progress unscored | RMSE is the top deduction and slowing down always reduces it — a one-way ratchet | T-SCORE-SPEED-COMPLIANCE |
| Saturating caps | Worst cases unrankable | T-METRIC-UNCAP |
| Sign-flip dead-band | Counts noise around zero | T-METRIC-DEADBAND |
| ODD-conditioned metrics | Waymo decomposes goals → per-ODD behavioural metrics; we are per-track | — |
| Human-referenced benchmark | Waymo benchmarks against attentive-human response | — (arguably out of scope for a sim research stack) |

---

## 8. How to add a metric here

1. Name the **physical quantity** first. If you cannot state it, the metric is a proxy — see the `feedback_tuning_vs_architecture` smell list.
2. Derive rates from **measured** timestamps. Never hardcode a frame rate; this stack runs at ~13 FPS, not 30.
3. Give sign-based metrics a **magnitude dead-band**, or they measure noise.
4. Prefer **soft-knee** penalties over hard caps, so severity keeps ranking.
5. Add a regression test that **fails on the bug you are preventing**, not just on the happy path. The 28 pre-existing ACC tests all passed both before and after a 2.5× error.
