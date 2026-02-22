# AV Stack — Agent Memory: Project Overview

**Last updated:** 2026-02-21
**Current milestone:** S1-M39 (Longitudinal comfort tuning)

---

## Purpose

A complete autonomous vehicle (AV) pipeline running inside a **Unity 2021.3 LTS** simulation. The stack handles perception, trajectory planning, and vehicle control for lane-keeping. This is a research/development system — not production safety-critical.

---

## Tech Stack

| Layer | Language / Libraries |
|---|---|
| Simulation | Unity 2021.3 LTS (C#) |
| Core stack | Python 3 |
| ML / Vision | PyTorch ≥ 2.0, OpenCV ≥ 4.8, scikit-learn |
| API bridge | FastAPI + uvicorn (async) |
| Data storage | HDF5 via h5py |
| Numerics | NumPy, SciPy, Pandas |
| Testing | pytest (98 test files, 126+ passing tests) |
| Visualization | Flask-based web tool (PhilViz), Matplotlib |
| Config | YAML (config/av_stack_config.yaml, ~17 KB) |

---

## Major Folders

```
av/
├── av_stack.py                # Main orchestrator (5,420 lines) — DO NOT edit lightly
├── config/av_stack_config.yaml  # All tuning parameters (100+ params)
├── perception/                # Lane detection (ML + CV fallback)
├── trajectory/                # Path planning and speed planning
├── control/                   # PID / Pure Pursuit / Stanley controllers
├── bridge/                    # FastAPI Unity↔Python bridge
├── data/                      # HDF5 recorder and replay
├── tools/
│   ├── analyze/               # 50+ analysis scripts (primary diagnostic workflow)
│   └── debug_visualizer/      # PhilViz — web-based frame inspector (port 5001)
├── tests/                     # 98 test files
├── tracks/                    # YAML track definitions (s_loop, oval, highway_65…)
├── docs/                      # Documentation (ARCHITECTURE, TODO, ROADMAP, agent/)
└── unity/AVSimulation/        # Unity project (C# bridge, scene)
```

---

## Development Stage

**Stage 1: Lane-Keeping** — actively completing comfort metrics.
Stage 2 (multi-lane) and beyond are defined in `docs/ROADMAP.md` but not yet started.

---

## Key Entry Points

| Purpose | Command |
|---|---|
| Run AV stack | `./start_av_stack.sh --launch-unity --unity-auto-play --duration 60` |
| Ground truth baseline | `./start_ground_truth.sh --track-yaml tracks/s_loop.yml --duration 60` |
| Quick summary | `python tools/analyze/analyze_drive_overall.py --latest` |
| Frame inspector | `python tools/debug_visualizer/server.py` → http://localhost:5001 |
| A/B test | `python tools/analyze/run_ab_batch.py --config-a A.yaml --config-b B.yaml --runs 5` |

---

## Related Memory Files

- `docs/agent/architecture.md` — system components, data flow, execution sequence
- `docs/agent/current_state.md` — active work, incomplete items, known issues
- `docs/agent/tasks.md` — structured task list with priorities
