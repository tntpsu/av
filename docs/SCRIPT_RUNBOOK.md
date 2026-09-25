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

### `build_unity_player.sh`

- **Purpose:** Build the macOS Unity player (`unity/AVSimulation/mybuild.app`) via
  `BuildPlayerCLI.BuildMacPlayer` in batchmode. Invoked automatically by
  `start_av_stack.sh`; rarely run by hand.
- **Unity launch behavior:** Runs Unity in `-batchmode -nographics -quit`. **Requires an
  activated Unity Editor licence** — builds fail with exit 198 and
  `"No valid Unity Editor license found"` if the licence has lapsed. Running an
  already-built player does not need a licence.
- **`--skip-if-clean`:** Skips the build when the Unity project is git-clean **and**
  `mybuild.app.stamp` records the same `HEAD:unity/AVSimulation` tree hash as the current
  HEAD. The stamp is written automatically after each successful build.
  - Do **not** reintroduce an mtime comparison here: the Unity build rewrites
    `ProjectSettings/*.asset`, so source mtimes are always ≥ the build output and a
    timestamp check can never pass. A prior version also stat'd the `.app` *directory*,
    whose mtime tracks only its immediate entries and read 2026-04-16 while the actual
    build was 2026-05-06. Between the two, `--skip-if-clean` never fired and every run
    was forced into a build — which, with the licence lapsed, meant no runs at all.
  - **Seeding an existing player:** if you have a known-good player but no stamp, run
    `git rev-parse "HEAD:unity/AVSimulation" > unity/AVSimulation/mybuild.app.stamp`.
    The stamp is gitignored.
- **Use when:** Unity C# / Assets / ProjectSettings changed and you need a fresh player.

### `stop_av_stack.sh`

- **Purpose:** Stop running AV stack processes.

## Scheduled / Automation Scripts

### `tools/nightly/run.sh`

- **Purpose:** Wrapper invoked by launchd at 2am local time to run the nightly test-fix agent. Pulls `main`, invokes `claude -p` with `tools/nightly/PROMPT.md` under a hard 60-min watchdog timeout, logs to `~/av_runtime/logs/nightly/<date>.log`, and emails a completion summary on every exit.
- **Unity launch behavior:** No Unity. Pytest only (uses `pytest -n auto --dist loadfile` via pytest-xdist for ~7× speedup; `--dist loadfile` groups same-module tests onto one worker so MPCController's cached state doesn't leak between parallel workers).
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
- **Email subject composition:** `compose_subject()` in `notify_on_exit` parses `data/reports/sweep_status.txt` directly — counts done tracks, sums regressions (delta < -2.0), counts FLAG= markers, identifies worst-delta track. **Gate verdict** is read from `data/reports/sweep_report.txt`'s canonical `GATE: ...` line so the subject reflects the same rule the agent applied (layer-≥95 AND no regressions), not a wrapper-side recomputation. Falls back to log-grep then exit-code-synthesis if files are missing.

### `tools/nightly/acc-sweep/run.sh`

- **Purpose:** Wrapper invoked by launchd at 4am local time daily (after fix-tests at 2am, lateral sweep at 3am) to verify ACC scenarios in `tracks/scenarios/*.yml` against the gate criteria embedded in each scenario's YAML header. Drives the `/acc-sweep` slash command. Logs to `~/av_runtime/logs/acc-sweep/<date>.log`.
- **Unity launch behavior:** Step 2 default is `--quick` mode (analyze most recent matching recording per scenario, mark SKIPPED if none newer than 7d). Step 2.5 then auto-invokes `/e2e tracks/scenarios/<name>.yml` for SKIPPED/FAIL/WARN scenarios up to a soft cap of `max_fresh_runs_per_night = 5` Unity launches — converts "no data" results into actual verdicts. Cap protects the 90-min wall ceiling and bounds Unity-launch-under-launchd risk. Priority order: SKIPPED first, then FAIL, then WARN.
- **Pre-flight cadence smoke test (added 2026-05-06, repaired 2026-08-12):** Before invoking `claude -p`, the wrapper runs a 10-second `start_av_stack.sh --force --skip-unity-build-if-clean` on `highway_65` and parses `camera/timestamps` for `dt_p95`. If `dt_p95 > 250ms` it sets `AV_NIGHTLY_NO_FRESH_E2E=1`; PROMPT.md honors that env var and skips Step 2.5 entirely, marking would-be re-seeds as `verdict=SKIPPED reason="preflight_cadence_fail"`. Purpose is to stop the 4-AM-Unity-starvation pattern (2026-05-05 G1 diagnosis) from poisoning the email with infra-bug-masquerading-as-control-bug data.
  - **This gate failed 94 consecutive nights (2026-05-06 → 2026-08-11) and produced a total ACC data blackout.** Three independent defects, all fixed 2026-08-12:
    1. **Threshold was unpassable.** Camera delivery is quantized to 76.9 ms (13 FPS) steps; starvation drops whole frames rather than lowering the base rate. Measured p95: healthy 76.9–153.8 ms, starved 364.3 ms. The old `100ms` threshold sat *below* the healthy p95, so it could only pass with zero dropped frames in 10 s. Recalibrated to **250 ms**, which sits between the two populations with margin. Note `p50` is 76.9 ms on healthy *and* starved runs — **do not gate on p50, it does not discriminate.**
    2. **No `--skip-unity-build-if-clean`,** so the smoke test attempted a full Unity build nightly. While the Editor licence was lapsed, every build failed and no recording was produced.
    3. **Measured the wrong file.** It read `files[-1]` (newest `*.h5` by mtime) regardless of whether the smoke test created it, so when the run failed it silently scored a *previous* recording. It now requires a recording newer than the smoke test's start time, else returns `9999` and fails honestly.
  - **If you change the threshold, recalibrate against both populations** — a known-healthy interactive run and a known-starved one. The numbers are in the comment block in `run.sh`.
  - **Two further fixes 2026-08-13, after the repaired gate ran for real:**
    4. **`caffeinate -di` now wraps the smoke test, not just `claude -p`.** It previously covered only the agent invocation, so at 4 AM the smoke test launched Unity into a display-asleep, GPU-throttled context and measured **409 ms** — versus **153 ms** for the identical command run interactively. The gate was measuring starvation it had caused itself, and that false FAIL then blocked the caffeinated agent run that followed. **Any Unity launch in a nightly wrapper must be inside `caffeinate`.**
    5. **Smoke-test recordings now go to `$RUNTIME/preflight_recordings`,** not `data/recordings`. These are ~200-frame / 10-second stubs; left in the main pool they become the newest recording for `highway_65`, and `latest_per_track()` in the nightly sweep then scores a 17-second straight-only run as a full lap (observed 2026-08-13: a 216-frame run scored 99.9, flagged untrustworthy in the report). The measurement reads from `PREFLIGHT_REC_DIR`.
- **Sleep prevention:** The `claude -p` invocation is wrapped in `caffeinate -di` so macOS idle/display sleep doesn't kick in mid-run. Covers the entire subprocess tree including any Unity processes spawned by Step 2.5 `/e2e`.
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
- **Email subject composition:** `compose_ph_subject()` parses `data/reports/process_health_<DATE>.md` for `Total entries: **N**` and the first non-zero stage in the Process Stage Pareto, building `PROCESS_HEALTH stage=Design(64%) entries=14`. Falls back to log-grep then exit-code if the report file is missing.
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

### `av_stack/fall_detector.py` (module, not a script)

- **Purpose:** detects the vehicle falling off the world. Tracks are a bare `MeshCollider` exactly `roadWidth` (7.2 m) wide with **no terrain, ground plane, or shoulder** — leaving the road means falling into void.
- **Confirmed occurrence:** `recording_20260506_042710.h5` (H9, 2026-05-06) fell 46.7 m (pos.y 0.80 → −45.93, never recovered) with 12.5 m lateral offset and 172° roll. Nothing detected it as a fall; it scored as an ordinary out-of-lane + e-stop.
- **Two signatures, either sufficient:** descent rate above `descent_rate_mps` (4.0) for 3 consecutive frames, or all four `wheel_contact_normal_y` at 0 for `airborne_frames` (5).
- **Do NOT use an absolute drop threshold.** The first implementation did and false-positived on BOTH graded tracks — hill_highway and hill_g1 legitimately descend ~5 m. Descent *rate* is the physical discriminator: max_speed × max_grade = 25 × 0.10 = 2.5 m/s is the ceiling for any legitimate descent, while the H9 fall reached 47.65 m/s and the steepest grade peaks at 0.72 m/s.
- **Runtime:** `av_stack/orchestrator.py` ends the run `post_fall_grace_s` (2.0 s) after a fall, so the event is captured but the remaining Unity time is not spent recording a vehicle falling.
- **Config:** `stack.fall_detection.{enabled,descent_rate_mps,airborne_frames,post_fall_grace_s}`. Set `enabled: false` to disable without a code revert.
- Tests: `tests/test_fall_detector.py` (15) — the grade negative-cases are the important half.

### `tools/analyze/analyze_ride_comfort.py`

- **Purpose:** ISO 2631-1 frequency-weighted ride comfort — MSDV (Motion Sickness Dose Value), `a_w` RMS, and dominant oscillation frequency, for both the lateral (path weave) and longitudinal (speed hunting) axes.
- **Unity launch behavior:** None — offline over an existing recording.
- **Invocation:** `--file <path>` or `--latest`, optional `--window T0 T1` to restrict to a time range.
- **Why it exists:** every other comfort gate is an amplitude percentile and therefore frequency-blind. Measured on `highway_h3`, the visible lane weave is **0.243 Hz** — inside the ISO motion-sickness band (0.1–0.5 Hz) and *below* the ride-comfort band (0.5–80 Hz) that accel/jerk P95 target. A 0.24 Hz weave and a 3 Hz shake with identical P95 score the same and feel nothing alike.
- **Discriminates where the ACC oscillation metric saturates:** across 5 A/B pairs it separated the two MPC reference modes cleanly (MSDV median 8.35 → 5.86, ISO verdict "uncomfortable" → "fairly uncomfortable").
- **Not yet wired into the scoring layers** — it reports, it does not gate. See T-METRIC-MSDV-WIRE.
- Tests: `tests/test_ride_comfort.py` (18).

### `tools/analyze/run_ab_batch.py`

- **Purpose:** Batch A/B runner for a single config parameter, with robust median/quartile stats over N paired trials. This is the promotion gate for config changes (CLAUDE.md requires ≥5 runs).
- **Unity launch behavior:** Launches Unity once per trial — `2 × --repeats` runs total.
- **Invocation:** `--param <dotted.key> --a <baseline> --b <treatment> --repeats 5 --track-yaml tracks/<t>.yml [--config <overlay>]`. Resolve the dotted key against the loaded YAML; e.g. `trajectory.mpc.mpc_e_lat_reference_mode`.
- **ALWAYS check `first_failure_frame` in the summaries before trusting any metric.** If it is not `n/a`, runs were terminating early and every other statistic is computed over a truncated prefix. On 2026-08-13 a hairpin batch reported a clean-looking `lateral_error_rmse 0.055 vs 0.029` that was measured over ~20 frames before an e-stop in all 10 trials.
- **2026-08-13 fixes:**
  1. `--fixed-start-t` defaulted to `0.0` and was *always* passed through as `--start-t`, silently overriding each track's own `start_distance`. `hairpin_15.yml` declares `start_distance: 3.0`; at `t=0.0` the car starts off-lane and every trial e-stopped at ~frame 19 with `gt_left_offroad`, invalidating the batch while still printing plausible numbers. Default is now `None` — the track YAML wins unless you explicitly pass a value.
  2. Added `--skip-unity-build-if-clean` to the per-trial launch. An A/B never changes Unity C#, and rebuilding the player every trial dominated wall time.
- **Use when:** promoting any config change. Do not promote on single runs — see the hairpin case above, where a single-run 0.6-point "gate crossing" was run-to-run noise that 5 pairs refuted.

### `tools/analyze/run_gate_and_triage.py`

- **Purpose:** Acceptance-gate evaluation plus the triage engine's pattern detectors (20+ known signatures). Writes a bundle to `data/reports/gates/<UTC>_gate/` containing `decision.json`, `gate_report.json`, `triage_packets/<recording>.json` and `failure_packets/<recording>/packet.json`.
- **Unity launch behavior:** None by default (offline over existing recordings). Only launches Unity with `--execute-gates`.
- **Invocation:** takes **`--recordings <path> --recording-track-ids <track_id>`**, *not* a positional path and *not* `--latest`. A bare positional path errors with `unrecognized arguments`.
- **Where the answer is:** `triage_packets/*.json` → `what_failed_first` (phase/frame/type/description) and `trigger_reasons`. `failure_packets/*/packet.json` → `root_cause_bucket`. The stdout JSON only carries counts.
- **2026-08-13 fix:** `_extract_run_metrics` used `float(curve_intent_diag.get("<key>", 0.0))` for four `curve_intent_*` fields. `.get(key, default)` returns the *stored* value when the key exists with a `None` value, so the default never applied and the tool crashed with `TypeError: float() argument must be ... not 'NoneType'` on any recording whose curve-intent diagnostics were unpopulated. Now uses the `(… or 0.0)` idiom already used elsewhere in the same function.
- **Use when:** diagnosing any recording — run it *before* manual investigation, since its detectors catch known patterns instantly.

### `tools/analyze/counterfactual_layer_swap.py`

- **Purpose:** Stage-5 matrix (trajectory-lock + control-lock) and attribution scorecard.
- **Default perception mode:** Segmentation default.
- **CV override:** `--use-cv`.

### `tools/analyze/validate_nmpc_cold_start.py`

- **Purpose:** Phase 0 oracle for the NMPC sign-determinism fix (2026-05-02). Reproduces the SLSQP cold-start saddle behavior at `kappa=0` with non-zero `e_lat` and shows the proposed seed value across a sweep of `(e_lat, gain)` pairs. Used to confirm root cause and tune the cold-start seed gain before touching the controller code.
- **Use when:** Investigating any future BLAS-determinism regression in NMPC, or when tuning `cold_start_e_lat_seed_gain` / `cold_start_e_lat_seed_max_frac` in `NMPCParams`.
- **Manual:** `python3 tools/analyze/validate_nmpc_cold_start.py --gain-sweep`

### `tools/analyze/analyze_drive_overall.py`

- **Purpose:** PRIMARY end-to-end drive evaluation tool. Combines path-tracking accuracy, control smoothness, perception quality, trajectory quality, system health, and safety metrics into one comprehensive report.
- **Use when:** You want a single-command verdict on whether a recording reflects healthy or degraded behavior across the whole stack.
### `tools/speed_utilization.py`

- **Purpose:** Speed-limit utilisation — did the car use the road it was given? Reports `v / speed_limit` (product truth) and `v / min(speed_limit, target_speed)` (system truth; isolates governor / tracking-budget defects from the deliberately low research target) over eligible frames (after 10 s startup, not ACC-following, not e-stopped, not braking toward a lower posted limit), plus % of eligible time under 0.70 and the **binding cap** at those frames (target / velocity_profile / curve_cap / comfort). Added 2026-09-25 (T-METRIC-SPEED-UTILISATION) after hill_highway scored 97.6 while driving 15 mph on a 25 mph road and s_loop turned out to run at half its posted limit. **Report-only** — no score is changed; proposed gate: vs-allowed median ≥ 0.85 and ≤ 10 % of time under 0.70.
- **Use when:** `python3 tools/speed_utilization.py <recording.h5>`, or via `analyze_drive_overall.py` (section "SPEED UTILISATION"). Nightly lateral sweep reports the vs-allowed median per track.
- **Tests:** `tests/test_speed_utilization.py` (synthetic HDF5; eligibility rules, both ratios, attribution).


## Debug Visualizer (PhilViz)

The `tools/debug_visualizer/` tree powers the in-browser playback + diagnostics dashboard. Backend modules expose health and triage data to the visualizer; the server is the entry point.
- **Speed Utilisation section (2026-09-25):** report-only line from `tools/speed_utilization.py` — see that entry. Wrapped in try/except so a metric failure can never break the report.

### `tools/debug_visualizer/backend/dashboards.py`

- **Purpose:** Backend parsers for the PhilViz Dashboards page (added 2026-05-02). Reads the same heartbeat/report files the nightly job wrappers consume to compose email subjects (`data/reports/{nightly,sweep,acc_sweep}_status.txt` and `data/reports/{nightly_test,sweep,acc_sweep}_report.txt`) and returns structured JSON for the `/dashboards` mobile-friendly view.
- **Use when:** Imported by `tools/debug_visualizer/server.py`. Read-only; never writes to data/reports.
- **Endpoints:** `GET /api/dashboards/all`, `/api/dashboards/sweep`, `/api/dashboards/acc-sweep`, `/api/dashboards/fix-tests`, `/api/tracks/with-metadata`.

### `tools/analyze/acc_pipeline_analysis.py`

- **Purpose:** ACC pipeline diagnostic CLI. Five analysis cards: (1) Radar Health, (2) IDM State, (3) Safety Layer, (4) Worst Frames, (5) **Composite ACC Score** (added 2026-05-05 — Proposal A). Card 5 emits a continuous 0-100 score with three weighted sub-layers (Safety 50%, Tracking 30%, Behavior 20%) alongside the existing PASS/FAIL gate verdicts. Behavior catches longitudinal oscillation, jerk, and bang-bang that gate verdicts miss.
- **Use when:** Invoked by humans for triage of a specific recording (`python3 tools/analyze/acc_pipeline_analysis.py --latest`). Also referenced from `tools/nightly/acc-sweep/PROMPT.md` Step 6 — the nightly agent runs it per scenario to populate the Score column in the email report.
- **Score computation:** `_compute_acc_score(d) -> dict` is a pure function returning structured deductions; `_card5_acc_score(d)` is the human view that prints to stdout. Future callers (PhilViz cards, triage stage B) should import `_compute_acc_score` rather than re-parse text. Score=`None` when ACC ran for fewer than `ACC_SCORE_MIN_ACTIVE_FRAMES = 30` frames (run died too early to score meaningfully).
- **Constants:** all `ACC_SCORE_*` thresholds and weights live in `tools/scoring_registry.py` (single source of truth).
- **Tests:** `tests/test_acc_score.py` (9 tests covering clean run, collision-zero, TTC penalty, oscillation→Behavior, e-stop edge counting, missing-control-fields renorm).
- **Collision + frame changes (2026-09-22, T-ACC-SCORER-COLLISION-BLIND / T-ACC-RADAR-FRAME):** Card 3 prints `Physical Contact: N frame(s)` from `vehicle/lead_collision_detected`, and Card 5 now counts those frames as collisions (`n_collision`, `n_contact_frames` in the score dict) — `ACC_SCORE_COLLISION_FORCES_ZERO` finally fires. The `Collision Frames (distance < 0)` line is kept for schema compatibility but is structurally always 0 (Unity's override clamps the range to 0.1 m). Near-miss (Card 3 and Card 5) is computed on the bumper gap `distance − ACC_RADAR_RANGE_OFFSET_M` (4.43 m; `radar_fwd_distance_m` is centre-to-centre). `tools/drive_summary_core.py` near-miss uses the same offset. Frame is era-aware: `recording_provenance.radar_range_offset_m` (0 for pre-09-22 files) → correction = `ACC_RADAR_RANGE_OFFSET_M − offset_recorded` (the recorded `radar_fwd_distance_m` is the sensor's filtered gap AFTER the offset, orchestrator.py:9760). E-stop events exclude `EMERGENCY_BRAKE` frames (B1 reflex, not e-stops); near-miss excludes standstill (speed < 0.5). Historical G2/G1/H5 composites re-score to 0 — see `tools/nightly/acc-sweep/PROMPT.md` § Scoring changes.

### `tools/debug_visualizer/backend/skills_runner.py`

- **Purpose:** Backend module for the PhilViz Skills page (added 2026-05-02). Discovers slash commands from `.claude/commands/*.md`, spawns `claude -p` subprocesses, buffers their output line-by-line keyed by `job_id`, and supports cancel via SIGTERM. Subprocesses are detached from the HTTP connection so mobile users can disconnect/reconnect without killing in-flight skills.
- **Use when:** Imported by `tools/debug_visualizer/server.py`. Not invoked directly. Hot-path used by `/api/skills/{list,run,jobs,stream/<id>,cancel/<id>}`.
- **MCP/auth:** spawns claude-p with `--strict-mcp-config --mcp-config '{"mcpServers":{}}' --permission-mode bypassPermissions --no-session-persistence` (matches nightly wrapper pattern). Hardcoded budget cap `$5.00`. Sets `AV_NIGHTLY_RUN=1` so hardware-sensitive tests self-skip.
- **Storage:** in-process job dict, `MAX_BUFFER_LINES=5000` per job. No persistence across server restarts (acceptable for V1).

### `tools/debug_visualizer/server.py` — Skills + Tracks + Configs + Sites APIs (added 2026-05-02)

The PhilViz Flask server gained these endpoints to power the Skills page UI:
- `GET /api/skills/list` — discover slash commands from `.claude/commands/*.md`
- `POST /api/skills/run` — spawn `claude -p` with the chosen skill (returns `job_id`)
- `GET /api/skills/jobs` — recent runs (last 20)
- `GET /api/skills/stream/<job_id>` — SSE stream of subprocess output (reconnect-safe via `?last=`)
- `POST /api/skills/cancel/<job_id>` — SIGTERM the subprocess group
- `GET /api/tracks/list` — `tracks/*.yml` and `tracks/scenarios/*.yml` for the picker
- `GET /api/configs/list` — `config/*.yaml` for the config picker
- `GET /api/sites/list` — reads `~/.philviz_sites.json` (auto-creates with sensible defaults if missing) for the cross-site nav drawer

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


### `tools/analyze/acc_pipeline_analysis.py`

- **Purpose:** Five-card ACC analysis (radar health, IDM state, safety layer, worst frames, composite 0–100 score). Invoked by `/acc-sweep` and the nightly ACC job.
- **Unity launch behavior:** None — offline over an existing recording.
- **Invocation:** `--file <path>` or `--latest`. A bare positional path errors.
- **Silently returns no score** when `acc_active_pct < ACC_MIN_ACTIVE_FRAME_RATE` (~8–10%). For rejection scenarios (H9 adjacent-lane, H10 oncoming) and free-flow (H11) that is the *correct* outcome — ACC is meant to stay disengaged. **Do not read "n/a" as FAIL for those.**
- **2026-08-14 fix — `_sign_flips_per_min` inflated every rate ~2.5×:**
  1. `fps` defaulted to `30.0` and was never passed by either call site, but this stack captures at **~13 FPS** (`docs/agent/performance.md`). Duration was understated 2.3×, so every per-minute rate was overstated by the same factor. Now takes the measured rate from `d["fps"]`, derived from `camera/timestamps` in the loader.
  2. The zero guard `s[s == 0] = 0` was a **no-op** — it assigned 0 to elements already 0, so the documented "treat zeros as same sign as previous" never happened and a signal passing through exact zero counted two spurious flips.
  - Card 2's "IDM Sign Changes" had the same hardcoded 30.0, plus a `max(1.0, …)` that floored duration at one minute and under-reported any shorter run.
  - **Effect:** the oscillation penalty was pegged at its −30 cap on *every* ACC scenario, so the score could neither rank scenarios nor detect improvement. After the fix, `highway_h3_hard_brake` reads 43.0/min instead of 109.5 and scores 90.6 GREEN instead of 87.2 YELLOW; H5 and H6 rise to ~99.5.
  - Regression tests: `tests/test_acc_score.py::TestSignFlipsPerMin`. The 28 pre-existing ACC tests passed both before and after the bug — none exercised the rate denominator.
- **If you add a per-minute metric here, take fps from `d["fps"]`. Never hardcode 30.**
