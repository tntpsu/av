Resolve a scenario alias and output the exact `start_av_stack.sh` command(s) to run it.

The user's request is: $ARGUMENTS

## Step 1 — Resolve the scenario alias

Look up the scenario in `.claude/docs/scenario_registry.md`. Accept fuzzy/natural language names:

| If user says... | Map to alias |
|----------------|--------------|
| "h2", "steady", "highway steady", "acc steady", "steady following" | `h2_steady` |
| "h3", "brake", "hard brake" | `h3_brake` |
| "h4", "accel away", "lead accelerates" | `h4_accel` |
| "h5", "stop go", "stop-go" | `h5_stopgo` |
| "h6", "close gap", "cut in" | `h6_close` |
| "h7", "catchup", "straight catchup" | `h7_catchup` |
| "h8", "curve catchup" | `h8_curve` |
| "a1", "autobahn steady" | `a1_steady` |
| "a2", "autobahn brake" | `a2_brake` |
| "g1", "grade following", "hill following" | `g1_grade` |
| "g2", "stop on grade", "hill stop" | `g2_stop` |
| "highway", "hwy", "65" | `highway` |
| "sloop", "s loop", "s_loop" | `s_loop` |
| "mixed", "mixed radius" | `mixed` |
| "hill", "hill highway" | `hill` |
| "autobahn", "high speed", "25mps" | `autobahn` |
| "hairpin" | `hairpin` |
| "all acc highway", "acc highway suite" | suite: `acc_highway` |
| "all acc", "full acc suite" | suite: `acc_all` |
| "standard tracks", "regression" | suite: `standard` |

If the alias is not recognized, list the 3 closest matches from the registry and ask for clarification.

## Step 2 — Determine build flags

- **ACC scenarios** (any of h2–h8, a1–a2, g1–g2): REQUIRE `--build-unity-player --skip-unity-build-if-clean`
- **Standard tracks** (s_loop, highway, mixed, hill, autobahn, hairpin, sweeping): use `--build-unity-player --skip-unity-build-if-clean` by default (safe to include)
- If user says "skip build" or "no rebuild" → use `--skip-unity-build-if-clean` only
- **NEVER omit `--track-yaml`** — omitting it causes the car to fall through the road (315,000N crash)

## Step 3 — Handle duration overrides

- If user specifies duration ("30 seconds", "2 minutes", "120s") → use that value
- Otherwise → use registry default

## Step 4 — Handle config overrides

- If user specifies a config ("with mpc_highway config", "using acc_autobahn") → use that config
- For `hill` track: **never add `--config`** unless explicitly overridden — it uses base config with auto-derived q_lat

## Step 5 — Output the command(s)

For a single scenario:
```bash
./start_av_stack.sh --build-unity-player --skip-unity-build-if-clean \
  --config config/<overlay>.yaml \
  --track-yaml tracks/<track>.yml \
  --duration <seconds>
```

For `hill` (no config overlay):
```bash
./start_av_stack.sh --build-unity-player --skip-unity-build-if-clean \
  --track-yaml tracks/hill_highway.yml --duration 90
```

For a suite (sequential), list all commands and note approximate total time (each scenario ~duration + 30s overhead).

## Step 6 — Remind about post-run analysis

After each run, suggest the right analysis command:
- ACC scenario: `python3 tools/analyze/acc_pipeline_analysis.py --latest`
- All scenarios: `python3 tools/analyze/analyze_drive_overall.py --latest`
- Or just: `/diagnose`

---

Scenario registry: `.claude/docs/scenario_registry.md`
