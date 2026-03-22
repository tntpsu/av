# Grade–lateral observability

**Plan:** `docs/plans/GRADE_LATERAL_PLAN.md`  
**Roadmap:** Stage 6 — G6-L0–L3 (L4 = policy tuning after metrics).

## What was added

| Layer | Artifact |
|-------|----------|
| Telemetry | `control/lateral_grade_damping`, `control/lateral_error_smoothing_alpha_effective` (per frame), plus existing `vehicle/road_grade` |
| Analytics | `tools/grade_lateral_analysis.py` → `grade_lateral_v1` JSON |
| CLI | `python tools/analyze_grade_lateral.py <recording.h5> [--json]` |
| Summary | `analyze_recording_summary(path, include_grade_lateral=True)` adds `grade_lateral` key |
| PhilViz | `GET /api/recording/<file>/grade-lateral?pre_failure_only=true&grade_threshold=0.02` — Chain tab **Grade–Lateral Breakdown** |

## Usage

```bash
# Terminal (same JSON as API)
python tools/analyze_grade_lateral.py data/recordings/your_run.h5 --json

# Where |lateral_error| is worst on *flat* grade (frame ranges for PhilViz jump-to-frame)
python tools/analyze_grade_lateral_flat_focus.py data/recordings/your_run.h5 --out data/reports/flat_focus.json

# Optional block inside full drive summary (extra work: reads HDF5 again)
python -c "from pathlib import Path; from tools.drive_summary_core import analyze_recording_summary; \
print(analyze_recording_summary(Path('data/recordings/your_run.h5'), include_grade_lateral=True).get('grade_lateral'))"
```

PhilViz **Chain → Grade–Lateral Breakdown** also loads flat-focus (API `GET .../grade-lateral-flat-focus`) and lists ranges with **→** jump to frame.

## Pre-failure policy

Default analysis clips to `executive_summary.failure_frame` when `pre_failure_only` is true (matches oscillation attribution). Override by passing `failure_frame` into `analyze_grade_lateral` or using `--no-pre-failure-only` on the CLI.

## Policy tuning (G6-L4)

See **`docs/GRADE_LATERAL_TUNING.md`** — A/B protocol, primary knob `grade_steering_damping_gain`, exit criteria.

## References

- `docs/CONFIG_GUIDE.md` — sign/units for `road_grade` and control telemetry  
- `docs/OSCILLATION_ATTRIBUTION.md` — shared failure-window semantics  
