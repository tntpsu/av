# Script Runbook

Single source of truth for what each top-level script does, when to use it, and whether it launches Unity.

If you are unsure which command to run, start here first.

## Quick Intent Map

- Run full AV stack (perception -> trajectory -> control): `./start_av_stack.sh`
- Run GT drive on a specific track (launch Unity player + GT follower): `./start_ground_truth.sh --track-yaml tracks/s_loop.yml`
- Run GT follower only (bridge + follower, Unity already running): `./start_ground_truth_follower.sh`
- Offline trajectory-lock replay (no Unity needed): `python tools/analyze/replay_trajectory_locked.py ...`
- Offline control-lock replay (no Unity needed): `python tools/analyze/replay_control_locked.py ...`
- Stage-4 latency/noise suite (offline): `python tools/analyze/run_latency_noise_suite.py ...`
- Stage-5 counterfactual layer swap (offline): `python tools/analyze/counterfactual_layer_swap.py ...`

## Startup Scripts

### `start_av_stack.sh`

- **Purpose:** Start bridge + AV stack with optional Unity launch/build flags.
- **Unity launch behavior:** Does not launch Unity unless you pass `--launch-unity` or `--run-unity-player`.
- **Perception default:** Segmentation by default; override with `--use-cv`.
- **Use when:** You want closed-loop AV behavior through your current stack.

### `start_ground_truth.sh`

- **Purpose:** One-command GT drive run using Unity player + ground truth follower.
- **Unity launch behavior:** Launches Unity player.
- **Track selection:** Yes (`--track-yaml tracks/s_loop.yml`, `tracks/oval.yml`, etc.).
- **Perception default:** Segmentation by default; override with `--use-cv`.
- **GT lane semantics default:** `--gt-centerline-as-left-lane=true` (single-lane semantics; expected GT width ~= half road width on `s_loop`).
- **Strict GT options:** `--strict-gt-pose` (zero manual controls) and `--strict-gt-feedback` (fail on GT feedback integrity gaps).
- **Fast record option:** `--fast-record` (GT-only fast path that skips full stack compute and records camera + vehicle + GT control).
- **State-lag experiment:** `--gt-record-state-lag-frames=N` (fast-record only) offsets recorded vehicle state by N frames for camera/state phase-alignment A/B tests.
- **JPEG load experiment:** `--gt-jpeg-quality=Q` (10-100) lowers camera encoding cost for reversible capture-pipeline A/B tests.
- **Upload worker default:** `--gt-camera-send-async=true` queues camera uploads through one worker coroutine by default to reduce capture/send jitter in GT recordings.
- **GT rotation source experiment:** `--gt-rotation-from-road-frame=true` uses closest road-frame tangent for GT rotation/velocity direction (reversible A/B for heading bias checks).
- **GT sync capture default:** `--gt-sync-capture=true` (capture on physics ticks with deterministic timestamps for GT recordings).
- **GT fixed timestep override:** `--gt-sync-fixed-delta=0.033333333` (GT sync runs force `Time.fixedDeltaTime` to ~30 Hz unless overridden).
- **Top-down experiment switch:** `--gt-disable-topdown=true` disables top-down capture for reversible A/B throughput testing.
- **Projection diagnostics default:** Unity now emits right-lane fiducials (vehicle-frame + Unity `WorldToScreenPoint` pixels) for front-camera projection validation in GT runs.
- **Cadence behavior:** GT follower loop is deadline-paced (~30 Hz target) and skips duplicate camera timestamps to reduce frame-time jitter in recordings.
- **Runtime logging default:** `--log-level=error` (benchmark-safe default). Use `--diagnostic-logging` or `--log-level=debug` only for diagnosis runs.
- **Stream sync policy default:** `--stream-sync-policy=aligned`. For replay-oriented throughput captures with full stack active, prefer `--stream-sync-policy=latest`.
- **Use when:** You want a clean GT-drive recording on a chosen track.

### `start_ground_truth_follower.sh`

- **Purpose:** Start bridge + `tools/ground_truth_follower.py` only.
- **Unity launch behavior:** Does not launch Unity.
- **Track selection:** No direct track flag (track comes from already-running Unity instance).
- **Use when:** Unity is already running and you only need GT follower app startup.

### `tools/promote_golden_gt.sh`

- **Purpose:** Promote a recording to canonical golden GT naming for replay baselines.
- **Default behavior:** Copies source file into `data/recordings/` using `golden_gt_<date>_<track>_<sync-policy>_<duration>.h5`.
- **Safety default:** Non-destructive (`--mode copy`), with optional `--mode move`.
- **Overwrite behavior:** Requires `--force` to replace existing destination.
- **Use when:** You want to refresh or add officially tagged golden GT files without manual renaming.

### `launch_unity.sh`

- **Purpose:** Unity launch helper.
- **Use when:** You want manual Unity launch flow separate from AV/GT script wrappers.

### `stop_av_stack.sh`

- **Purpose:** Stop running AV stack processes.

## Scheduled / Automation Scripts

### `tools/nightly/run.sh`

- **Purpose:** Wrapper invoked by launchd at 2am local time to run the nightly test-fix agent. Pulls `main`, invokes `claude -p` with `tools/nightly/PROMPT.md` under a hard 60-min watchdog timeout, logs to `~/av_runtime/logs/nightly/<date>.log`, and emails a completion summary on every exit.
- **Unity launch behavior:** No Unity. Pytest only (now uses `pytest -n auto` via pytest-xdist for ~7× speedup).
- **Auth/permissions:** Inherits user shell `gh`/`git` auth. Runs `claude -p --permission-mode bypassPermissions --strict-mcp-config --mcp-config '{"mcpServers":{}}'` (MCP disabled — Google Calendar OAuth re-auth hangs under launchd's no-TTY context) with a $5 budget cap and a 3600s wall-clock timeout. Exports `AV_NIGHTLY_RUN=1` so hardware-sensitive perf tests can self-skip.
- **Use when:** Triggered automatically by launchd; do not run manually unless smoke-testing — it will open a real PR if it finds fixable failures.
- **Install:** `cp tools/nightly/com.philtullai.av-nightly.plist ~/Library/LaunchAgents/ && launchctl load ~/Library/LaunchAgents/com.philtullai.av-nightly.plist`
- **Uninstall:** `launchctl unload ~/Library/LaunchAgents/com.philtullai.av-nightly.plist`
- **Companion files:** `tools/nightly/PROMPT.md` (agent prompt), `tools/nightly/RUBRIC.md` (classification rules), `tools/nightly/notify.py` (email helper).
- **Email subject composition:** `compose_subject()` in `notify_on_exit` parses `data/reports/nightly_test_report.txt` (Fixed/Real-breaks/Flaky counts) and `data/reports/nightly_status.txt` (delivery= field) directly, instead of relying on the agent printing a literal summary line to stdout. Falls back to log-grep then exit-code-synthesis if the report file is missing.

### `tools/nightly/sweep/run.sh`

- **Purpose:** Wrapper invoked by launchd at 3am local time (right after the 2am fix-tests job) to run a cross-track regression sweep. Pulls `main`, invokes `claude -p` with `tools/nightly/sweep/PROMPT.md`, which drives the `/sweep` slash command across all 6 tracks and compares scores to `tests/fixtures/scoring_baselines.json`. Logs to `~/av_runtime/logs/sweep/<date>.log`, emails a regression digest.
- **Unity launch behavior:** May launch Unity via `start_av_stack.sh` — but the prompt defaults to `--quick` mode (analyze most recent recordings, no fresh launches) because Unity stability under launchd's no-GUI Background context is unvalidated. Stops on second consecutive Unity failure.
- **Auth/permissions:** Same MCP-disabled flags as fix-tests. $10 budget cap, 5400s (90-min) hard wall-clock timeout. Exports `AV_NIGHTLY_RUN=1`.
- **Use when:** Triggered automatically by launchd; can be manually invoked via `launchctl start com.philtullai.av-sweep` for end-to-end validation.
- **Install:** `cp tools/nightly/sweep/com.philtullai.av-sweep.plist ~/Library/LaunchAgents/ && launchctl load ~/Library/LaunchAgents/com.philtullai.av-sweep.plist`
- **Uninstall:** `launchctl unload ~/Library/LaunchAgents/com.philtullai.av-sweep.plist`
- **Read-only by design:** Never commits or opens PRs. Reports regressions to the email; the human decides whether to investigate.
- **Companion files:** `tools/nightly/sweep/PROMPT.md`, `.claude/commands/sweep.md` (the playbook), `tools/nightly/notify.py`.
- **Email subject composition:** `compose_subject()` in `notify_on_exit` parses `data/reports/sweep_status.txt` directly — counts done tracks, sums regressions (delta < -2.0), counts FLAG= markers, identifies worst-delta track. Falls back to log-grep then exit-code-synthesis if the heartbeat is missing.

### `tools/nightly/acc-sweep/run.sh`

- **Purpose:** Wrapper invoked by launchd at 4am local time daily (after fix-tests at 2am, lateral sweep at 3am) to verify ACC scenarios in `tracks/scenarios/*.yml` against the gate criteria embedded in each scenario's YAML header. Drives the `/acc-sweep` slash command. Logs to `~/av_runtime/logs/acc-sweep/<date>.log`.
- **Unity launch behavior:** May launch Unity if `--fresh` is requested, but the PROMPT.md hard-bans `--fresh` for unattended runs (no GUI under launchd; 14 sequential Unity launches would be a stability hazard). Default is `--quick` mode: analyze the most recent matching recording per scenario, mark SKIPPED if no recording newer than 7d.
- **Auth/permissions:** Same MCP-disabled flags as fix-tests/sweep. $10 budget cap, 5400s (90-min) hard wall-clock timeout. Exports `AV_NIGHTLY_RUN=1`.
- **Use when:** Triggered automatically by launchd; can be manually invoked via `launchctl start com.philtullai.av-acc-sweep` or `/acc-sweep --fresh` for end-to-end validation during the day.
- **Install:** `cp tools/nightly/acc-sweep/com.philtullai.av-acc-sweep.plist ~/Library/LaunchAgents/ && launchctl load ~/Library/LaunchAgents/com.philtullai.av-acc-sweep.plist`
- **Uninstall:** `launchctl unload ~/Library/LaunchAgents/com.philtullai.av-acc-sweep.plist`
- **Read-only by design:** Never commits or opens PRs. Per-scenario PASS/FAIL/WARN/SKIPPED/AMBIGUOUS verdicts in the email digest; `data/reports/acc_sweep_report.txt` has the full report.
- **Companion files:** `tools/nightly/acc-sweep/PROMPT.md`, `.claude/commands/acc-sweep.md` (the playbook), `tools/nightly/notify.py`.
- **Email subject composition:** `compose_subject()` in `notify_on_exit` parses `data/reports/acc_sweep_status.txt` per-scenario lines (`scenario_<name>_done verdict=<X>`), counts each verdict type, computes gate=PASS (no FAILs) or gate=FAIL.
- **Known V1 limitation:** ACC scenarios share `track_id` with their base track in `recording_provenance`. Disambiguation is best-effort (filename + ACC-data-presence + recency). A `recording_provenance.scenario_id` field is on the deferred roadmap (see `docs/agent/tasks.md`).

### `tools/nightly/process-health/run.sh`

- **Purpose:** Wrapper invoked by launchd every Sunday at 5am (moved from 4am to make room for daily acc-sweep at 4am) to generate a weekly process-health Pareto digest. Pulls `main`, invokes `claude -p` with `tools/nightly/process-health/PROMPT.md`, which drives the `/process-health` slash command. Reads `data/reports/improvement_log.json`, computes Paretos by `process_stage` and detection efficiency, and emails the digest.
- **Unity launch behavior:** No Unity. JSON read + counts only.
- **Auth/permissions:** Same MCP-disabled flags. $3 budget cap, 1800s (30-min) wall-clock timeout. Exports `AV_NIGHTLY_RUN=1`.
- **Use when:** Triggered automatically Sundays. Skip if `data/reports/improvement_log.json` is missing — wrapper will report "log empty" and exit cleanly.
- **Install:** `cp tools/nightly/process-health/com.philtullai.av-process-health.plist ~/Library/LaunchAgents/ && launchctl load ~/Library/LaunchAgents/com.philtullai.av-process-health.plist`
- **Uninstall:** `launchctl unload ~/Library/LaunchAgents/com.philtullai.av-process-health.plist`
- **Companion files:** `tools/nightly/process-health/PROMPT.md`, `.claude/commands/process-health.md`, `tools/nightly/notify.py`.

### `tools/nightly/notify.py`

- **Purpose:** Send a Gmail SMTP email notification for the nightly job. Reads creds from `/Users/philtullai/ai-agents/duckAgent/.env` (reusing duckAgent's existing Gmail SMTP config). Pure stdlib — no pip deps.
- **Use when:** Invoked from `tools/nightly/run.sh`'s EXIT trap; not normally run manually.
- **Manual test:** `echo "test body" | python3 tools/nightly/notify.py "test subject"`
- **Required env (loaded from duckAgent's .env if not already set):** `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS`, and one of `INSIGHTS_EMAIL` / `EMAIL_TO`.

## Analysis / Replay Scripts (Offline)

These scripts replay recordings offline and do not require Unity runtime interaction.

### `tools/analyze/replay_trajectory_locked.py`

- **Purpose:** Lock trajectory reference to source recording while re-running control path.
- **Default perception mode:** Segmentation default.
- **CV override:** `--use-cv`.

### `tools/analyze/replay_control_locked.py`

- **Purpose:** Lock control outputs (steer/throttle/brake) to source recording.
- **Default perception mode:** Segmentation default.
- **CV override:** `--use-cv`.

### `tools/analyze/run_latency_noise_suite.py`

- **Purpose:** Stage-4 deterministic latency/noise stress matrix on trajectory-lock replay.
- **Default perception mode:** Segmentation default.
- **CV override:** `--use-cv`.

### `tools/analyze/counterfactual_layer_swap.py`

- **Purpose:** Stage-5 matrix (trajectory-lock + control-lock) and attribution scorecard.
- **Default perception mode:** Segmentation default.
- **CV override:** `--use-cv`.

### `tools/analyze/analyze_drive_overall.py`

- **Purpose:** PRIMARY end-to-end drive evaluation tool. Combines path-tracking accuracy, control smoothness, perception quality, trajectory quality, system health, and safety metrics into one comprehensive report.
- **Use when:** You want a single-command verdict on whether a recording reflects healthy or degraded behavior across the whole stack.

## Debug Visualizer (PhilViz)

The `tools/debug_visualizer/` tree powers the in-browser playback + diagnostics dashboard. Backend modules expose health and triage data to the visualizer; the server is the entry point.

### `tools/debug_visualizer/server.py`

- **Purpose:** Python server for the debug visualizer. Converts HDF5 recordings to JSON, serves camera frames, and exposes the backend modules below over HTTP.
- **Use when:** You want to open a recording in the visualizer UI for frame-level inspection.

### `tools/debug_visualizer/backend/issue_detector.py`

- **Purpose:** Automatically flag problematic frames in a recording — extreme polynomial coefficients, high lateral error, perception failures, emergency stops, heading jumps.
- **Use when:** You need a per-frame issue list rather than a global drive-summary verdict.

### `tools/debug_visualizer/backend/layer_health.py`

- **Purpose:** PhilViz Phase 3 layer-health scoring. Computes per-frame health scores (0.0–1.0) for each stack layer (Perception, Trajectory, Control) via a weighted linear combination of normalized signals.
- **Use when:** You want a glanceable per-layer health timeline alongside the recording.

### `tools/debug_visualizer/backend/triage_engine.py`

- **Purpose:** PhilViz Phase 5 triage. Matches failure patterns against a library of known issue signatures, computes per-layer attribution, and generates an ordered action checklist (each item linked to a code pointer, config lever, and fix hint).
- **Use when:** You want the visualizer to suggest *what to do next* about a flagged failure, not just identify it.

## Guardrails

- Segmentation is the default across startup/replay tooling unless `--use-cv` is explicitly set.
- For segmentation mode, scripts validate checkpoint presence and fail fast if missing.
- For track-specific GT runs, prefer `start_ground_truth.sh` with explicit `--track-yaml`.

## Stable Reminder Pattern

To make this easy to rediscover in future sessions:

- Keep this file path stable: `docs/SCRIPT_RUNBOOK.md`.
- Keep links to this file in:
  - `README.md`
  - `docs/README.md`
  - `docs/README_STARTUP.md`

## Enforcement Rule

This runbook is enforced by automation:

- Local pre-commit hook: `runbook-sync` (see `.pre-commit-config.yaml`)
- CI gate: "Enforce script runbook updates" (see `.github/workflows/tests.yml`)

If script-like files change without updating `docs/SCRIPT_RUNBOOK.md`, the check fails.

