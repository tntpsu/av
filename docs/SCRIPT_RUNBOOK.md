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

