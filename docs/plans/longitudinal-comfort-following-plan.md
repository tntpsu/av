# Longitudinal Comfort And Following Plan

## Goal

Improve ACC following quality on clean transport runs by addressing two distinct problems:

1. following-distance convergence is too conservative
2. comfort scoring is overly dominated by raw measured jerk spikes that do not match commanded behavior

Primary reference run:
- `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_230150.h5`

## Current Evidence

From the validated run:

- ACC active: `96.9%`
- state mode: `ACC_ACTIVE`
- collisions: `0`
- near misses: `0`
- TTC p05: `26.58 s`
- minimum gap: `27.3 m`

But:

- actual gap p50: `30.96 m`
- target gap p50: `20.27 m`
- gap error p50: `+9.55 m`
- frames below target gap: `0%`
- frames above target gap + `10 m`: `49.5%`

So the system is safe, but too conservative.

Comfort evidence on the same run:

- ACC jerk P95 raw: `215.8 m/s^3`
- filtered jerk p95: `2.007 m/s^3`
- commanded jerk p95: `0.312 m/s^3`

And hotspot frames show:

- target speeds are nearly flat
- ACC target speed is nearly flat
- raw accel command stays near zero
- measured acceleration flips sharply across short sample intervals

So the current ACC comfort failure is mostly a metric-contract problem plus limiter-transition / speed-quantization behavior, not aggressive commanded ACC behavior.

## Workstreams

### L1. Following-Convergence Contract

Goal:
- make ACC converge toward the intended following gap instead of sitting far long

Files:
- `/Users/philiptullai/Documents/Coding/av/control/acc_controller.py`
- `/Users/philiptullai/Documents/Coding/av/av_stack/orchestrator.py`
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`

Implementation:
1. trace the exact ACC target-gap policy and whether the current IDM/headway tuning is intentionally conservative
2. add direct convergence metrics:
   - `acc_gap_error_abs_p50_m`
   - `acc_gap_error_abs_p95_m`
   - `acc_gap_above_target_plus_2m_rate`
   - `acc_gap_above_target_plus_5m_rate`
   - `acc_gap_above_target_plus_10m_rate`
3. classify ACC behavior by regime:
   - closing
   - tracking
   - over-conservative trailing
4. only tune controller behavior after the metrics say the policy is actually too conservative

Acceptance:
- safe gap remains safe
- median gap error drops materially
- excessive “far trailing” rate drops materially

### L2. ACC Jerk Metric Contract Cleanup

Goal:
- stop treating raw measured jerk as the primary comfort truth when commanded behavior is calm

Files:
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
- `/Users/philiptullai/Documents/Coding/av/scoring_registry.py`
- longitudinal scoring/summary code paths after inspection

Implementation:
1. make metric roles explicit:
   - commanded jerk: control-quality gate
   - filtered measured jerk: comfort-quality gate or secondary gate
   - raw jerk: diagnostic only
2. add attribution-specific rollups:
   - limiter-transition dominated
   - speed-estimation quantization dominated
   - true command-induced jerk
3. ensure the top-level ACC comfort deduction uses the right metric role
4. keep raw jerk visible, but do not let it dominate the score unless it reflects actual control behavior

Acceptance:
- clean steady following runs no longer fail comfort on raw spikes alone
- true aggressive ACC still fails

### L3. Longitudinal Hotspot Attribution Hardening

Goal:
- directly explain why the worst jerk frames happened

Files:
- `/Users/philiptullai/Documents/Coding/av/tools/drive_summary_core.py`
- `/Users/philiptullai/Documents/Coding/av/tools/analyze/analyze_drive_overall.py`
- `/Users/philiptullai/Documents/Coding/av/tools/debug_visualizer/visualizer.js`

Implementation:
1. add a dedicated `ACC Comfort Contract` block
2. surface:
   - gap error statistics
   - ACC target-speed stability
   - commanded accel stability
   - limiter-transition rate
   - speed-quantization / irregular-sample coupling rate
3. distinguish:
   - safe but over-conservative following
   - unstable following
   - scoring artifact

Acceptance:
- the user can tell directly whether the problem is policy, controller, or metric

## Validation Order

1. instrument L1 and L2 metrics without changing runtime behavior
2. rerun:
   - `/Users/philiptullai/Documents/Coding/av/data/recordings/recording_20260326_230150.h5` style highway ACC run
3. decide whether to tune ACC policy or only revise comfort scoring
4. only then change ACC behavior if convergence is truly too conservative

## Output Standard

For the next clean run, the tools should answer these directly:

1. Is the car safe behind the lead?
2. Is it converging to the intended target gap?
3. Is comfort failing because of commanded behavior or because of measurement/limiter artifacts?
4. Which exact metric is the gate, and why?
