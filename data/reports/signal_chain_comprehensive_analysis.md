# Comprehensive Signal Chain Analysis Report
**Date:** 2026-02-20  
**Recordings analyzed:** 3 (pre-change baseline, all captured before A1-C3 code changes)  
**Tools used:** analyze_signal_chain.py, build_failure_packet.py, analyze_curve_entry_feasibility.py, PhilViz summary_analyzer, PhilViz issue_detector

---

## 1. Signal Chain Health (Baseline — Pre-Change)

| Metric | recording_130304 | recording_125604 | recording_195826 |
|--------|-------------------|-------------------|-------------------|
| Total Frames | 378 | 484 | 611 |
| Failure Frame | 378 (offroad) | 484 (offroad) | None |
| Failure Time (s) | 22.1 | 27.3 | — |
| Signal Delay (frames) | **176** | **154** | N/A |
| Heading Zeroed (total) | 170 frames | 148 frames | 71 frames |
| Heading Zero % During Curve | **22.0%** | **21.7%** | 8.8% |
| Ctrl Rate Limited | 49 frames | 96 frames | 0 |
| Ctrl Jerk Limited | **154 frames** | **234 frames** | 0 |
| Speed at Curve Entry | 2.6 m/s | 2.8 m/s | — |
| Overspeed | No | No | — |
| Classification | mixed-or-unclear | mixed-or-unclear | — |

### Key Finding: Signal Delay is 154-176 Frames (~2.5-3.0 Seconds)

This is the time between trajectory heading exceeding 0.01 rad and control feedforward exceeding 0.01. At 60 Hz, this means the steering wheel doesn't respond for 2.5-3 seconds after the trajectory planner sees the curve. At 2.6 m/s, the car travels ~7 meters before responding.

## 2. Root Cause: Where Does Heading Get Suppressed?

Per-frame analysis of recording_130304 around curve approach (frames 155-220):

- **Frames 155-171**: Heading raw = 0.03-0.11 rad (real curve signal), but heading_zero_gate=1
  - Curvature at these frames: 0.0001-0.0016 (below A1 guard of 0.002)
  - **Root cause:** quadratic_coeff < 0.005 triggers heading zeroing; curvature is too low to trip the guard
- **Frame 168**: First frame where heading_zero_gate=0 (curvature=0.0016)
- **Frame 187**: First frame where control feedforward > 0.01
- **Frames 200-208**: Steering output reaches 0.13-0.17 (turning right)
- **Frame 209**: **STEERING REVERSES** to -0.046 (suddenly turns left)
  - Lateral error crosses zero at frame 204 (-0.004m), continues negative to -0.077m at 209
  - Heading error crosses zero at frame 204 as well
  - The PID feedback overcorrects, fighting the feedforward
- **Frames 209-229**: Steering oscillates negative while feedforward stays positive 0.014-0.016
- **Frames 350-378**: Car is at max steering (0.88), lateral error grows to -1.47m → offroad failure

### Critical Observation: Steering Reversal at Frame 209

The failure pattern is not just "too late" — the controller overshoots past center, then the PID integral (accumulated from straight driving) creates a sign conflict between feedforward (+0.016) and total steering output (-0.046 to -0.157). The C3 integral flush is designed specifically for this pattern.

## 3. Jerk Limiting is the Dominant Limiter

| Recording | Jerk Limited Frames | Rate Limited Frames | Hard Clip Frames |
|-----------|---------------------|---------------------|------------------|
| 130304 | 154 (40.7%) | 49 (13.0%) | 32 (8.5%) |
| 125604 | 234 (48.3%) | 96 (19.8%) | 0 (0.0%) |

Limiter code 2 (jerk limit) dominates throughout both recordings. The issue detector confirms: "jerk_limit dominates (77.3-77.7% of limited frames)."

## 4. Speed is NOT the Primary Problem (Yet)

Despite the plan's emphasis on speed overshoot, these recordings show:
- Speed at curve entry: 2.6-2.8 m/s (very slow — likely decelerated from earlier events)
- v_max_feasible at entry: 21.0 m/s (ample margin)
- Turn feasibility margin: 0.23g (well under 0.25g limit)
- Speed-limited frames: 0%

**However**, this is because these runs fail before reaching full speed. The B1/B2 speed management will matter on runs that survive longer and approach 8-12 m/s.

## 5. PhilViz Summary Scores

| Layer | recording_130304 | recording_125604 |
|-------|-------------------|-------------------|
| **Overall** | **85.1** | **59.0** (capped) |
| Perception | 100.0 ✅ | 96.0 ✅ |
| Trajectory | 87.7 ✅ | 53.2 🔴 (cap trigger) |
| Control | 77.2 🟡 | 78.9 🟡 |
| LongitudinalComfort | 68.8 🟡 | 60.0 🟡 |
| Safety | 80.0 ✅ | 80.0 ✅ |
| SignalIntegrity | 100.0 ✅ | 100.0 ✅ |

Recording 125604 is capped at 59.0 due to Trajectory layer being red (53.2). The trajectory score is driven by high lateral error RMSE (0.536m vs 0.233m).

## 6. Issue Timeline

**Recording 130304:**
1. F161: Negative steering-reference correlation (medium) — controller fighting trajectory
2. F361: Steering limiter dominant: jerk_limit 77.7% (high)
3. F362: Straight sign mismatch 16 frames (high) — perception stale
4. F378: Emergency stop (critical)

**Recording 125604:**
1. F139: Negative control correlation -0.447 (medium)
2. F169: Negative control correlation -0.904 (high) — **steering directly opposing reference**
3. F199: Negative control correlation -0.646 (high)
4. F301: Jerk limit dominant 77.3% (high)
5. F483: Centerline cross left (high)
6. F484: Emergency stop (critical)

## 7. What the Code Changes Should Fix

| Change | Addresses | Expected Impact |
|--------|-----------|-----------------|
| **A1: Curvature guard** (now 0.001) | Heading zeroing on curve approach | ~10 frames earlier heading signal when curvature > 0.001 |
| **A2: Heading from history** | Heading=0 fallback on lane_positions path | Heading available even without lane_coeffs |
| **A3: Curvature-aware smoothing** | Serial smoothing delay | Reduced alpha during curve approach → faster signal propagation |
| **A4: Curvature preview** | No early warning of upcoming curve | Control gets 5-10 frame curvature advance notice |
| **B1/B2: Speed cap** | Speed overshoot (not dominant here) | Will matter when runs survive long enough to build speed |
| **C1: Lower curvature smoothing** | 3-4 frame delay in feedforward | Faster feedforward response |
| **C2: Preview feedforward blend** | Feedforward lag | 30% preview blend primes feedforward early |
| **C3: Integral flush** | Steering reversal at frame 209 | Straight-line bias cleared on curve entry |

## 8. Data-Driven Config Adjustment Made

Based on this analysis:
- **`heading_zero_curvature_guard`: 0.002 → 0.001** — recordings show heading is zeroed at curvature 0.0005-0.0015, so the original 0.002 guard wouldn't have prevented any of the zeroing events

## 9. Required Next Steps (All Require Unity)

1. **Start Unity + Bridge** → run one canonical s-loop recording with the new config
2. **Run `analyze_signal_chain.py`** on the new recording → confirm signal delay < 50 frames (was 154-176)
3. **Check PhilViz Summary** → confirm no new red layers
4. **Check for steering reversal** → the C3 integral flush should prevent the frame-209 pattern
5. **If first turn clears**: run D2 (5-repeat canonical A/B, Mann-Whitney p<0.10)
6. **If first turn still fails**: use PhilViz signal chain panel to identify which suppression stage is still binding

## 10. Architecture Escalation Readiness

If D2 shows TTF < 20s with all changes, the plan mandates architecture checkpoint:
- Evaluate MPC for lateral control
- Evaluate predictive trajectory with path-integrated curvature
- Evaluate speed-steering co-optimization
- Document decision in ROADMAP.md
