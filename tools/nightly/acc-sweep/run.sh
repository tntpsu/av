#!/bin/bash
# Nightly ACC scenario sweep. Invoked by launchd at 4am local time
# (after fix-tests at 2am and lateral sweep at 3am). Drives the
# `/acc-sweep` slash command across all scenarios in tracks/scenarios/
# and verifies each against gate criteria parsed from its file header.

set -uo pipefail

export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH
export AV_NIGHTLY_RUN=1

REPO=/Users/philtullai/av
RUNTIME=/Users/philtullai/av_runtime
LOG_DIR="$RUNTIME/logs/acc-sweep"
DATE=$(date +%Y-%m-%d)
LOG="$LOG_DIR/$DATE.log"
CLAUDE_TIMEOUT=5400  # 90 min — same ceiling as lateral sweep; --quick mode keeps actual runtime far below

mkdir -p "$LOG_DIR"

cd "$REPO" || { echo "FATAL: cannot cd to $REPO" > "$LOG"; exit 1; }

# Compose email subject. Tries three sources in order:
#   1. Parse data/reports/acc_sweep_status.txt heartbeat (per-scenario verdicts).
#   2. Grep the log for a `^DATE ACC_SWEEP ...` line (legacy fallback).
#   3. Synthesize from exit code.
compose_subject() {
  local exit_code=$1
  local hb="$REPO/data/reports/acc_sweep_status.txt"
  if [ -r "$hb" ] && grep -q '^scenario_.*_done' "$hb"; then
    local total passed failed warned skipped ambiguous gate
    total=$(grep -c '^scenario_.*_done' "$hb" 2>/dev/null || true)
    passed=$(grep -cE '^scenario_.*_done verdict=PASS' "$hb" 2>/dev/null || true)
    failed=$(grep -cE '^scenario_.*_done verdict=FAIL' "$hb" 2>/dev/null || true)
    warned=$(grep -cE '^scenario_.*_done verdict=WARN' "$hb" 2>/dev/null || true)
    skipped=$(grep -cE '^scenario_.*_done verdict=SKIPPED' "$hb" 2>/dev/null || true)
    ambiguous=$(grep -cE '^scenario_.*_done verdict=AMBIGUOUS' "$hb" 2>/dev/null || true)
    # Gate verdict: prefer the canonical "GATE: ..." line from acc_sweep_report.txt
    # so the wrapper subject matches the agent's actual conclusion. Same
    # pattern as sweep/run.sh — avoids the wrapper-side recomputation drift
    # that bit us on the lateral sweep on 2026-05-03.
    local report="$REPO/data/reports/acc_sweep_report.txt"
    gate=""
    if [ -r "$report" ]; then
      gate=$(grep -E '^GATE:' "$report" | head -1 | awk '{print $2}')
    fi
    if [ -z "$gate" ]; then
      if [ "$failed" -eq 0 ]; then gate=PASS; else gate=FAIL; fi
    fi
    echo "av acc-sweep $DATE: ACC_SWEEP gate=$gate pass=$passed fail=$failed warn=$warned skip=$skipped amb=$ambiguous total=$total"
    return
  fi
  local legacy
  legacy=$(grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2} ACC_SWEEP ' "$LOG" 2>/dev/null | tail -1)
  if [ -n "$legacy" ]; then
    echo "av acc-sweep $DATE: $legacy"
    return
  fi
  case "$exit_code" in
    124|143|137) echo "av acc-sweep $DATE: TIMED OUT after ${CLAUDE_TIMEOUT}s" ;;
    0)           echo "av acc-sweep $DATE: complete (exit=0, no heartbeat parsed)" ;;
    *)           echo "av acc-sweep $DATE: WRAPPER FAILED (exit=$exit_code)" ;;
  esac
}

notify_on_exit() {
  local exit_code=$?
  local subject
  subject=$(compose_subject "$exit_code")
  tail -n 100 "$LOG" 2>/dev/null \
    | python3 "$REPO/tools/nightly/notify.py" "$subject" --job acc-sweep --log "$LOG" \
    >>"$LOG" 2>&1 \
    || echo "notify.py failed (continuing)" >>"$LOG"
}
trap notify_on_exit EXIT

{
  echo "=== acc-sweep run start $(date -Iseconds) ==="
  echo "host:    $(hostname)"
  echo "pwd:     $(pwd)"
  echo "claude:  $(command -v claude || echo MISSING)"
  echo "git:     $(command -v git)"
  echo "git head before pull: $(git rev-parse --short HEAD)"
  echo

  echo "--- git fetch + pull ---"
  git fetch origin
  git checkout main
  git pull --ff-only origin main
  echo "git head after pull:  $(git rev-parse --short HEAD)"
  echo

  echo "--- pre-flight cadence smoke test (10s on highway_65) ---"
  # Diagnoses the Unity-starvation-under-launchd issue (2026-05-05). If Unity
  # can't sustain a workable frame rate in this 10-second smoke test, Step 2.5
  # fresh /e2e launches will produce garbage data (truncated runs scoring n/a).
  # Better to abort the fresh-run budget cleanly than to spend 5 Unity launches
  # on data that won't score.
  #
  # THRESHOLD: dt_p95 ≤ 250ms. Calibrated 2026-08-12 against real recordings on
  # both sides. Camera delivery is QUANTIZED to 76.9ms (13 FPS) steps, so
  # starvation does not lower the base rate — it drops whole frames, producing
  # multiples of 76.9ms. Measured:
  #
  #     starved (2026-05-06, under launchd)  p50 76.9  p90 331.7  p95 364.3
  #     healthy (2026-08-12, s_loop)         p50 76.9  p90  76.9  p95  76.9
  #     healthy (2026-08-12, hill_highway)   p50 76.9  p90 153.8  p95 153.8
  #     healthy (2026-08-12, highway_65)     p50 76.9  p90 153.8  p95 153.8
  #
  # p50 does NOT discriminate (76.9 in every case). p95 does: healthy tops out
  # at 153.8 (one dropped frame), starved is 364.3 (~4 dropped). 250ms sits
  # between them with margin on both sides.
  #
  # The previous 100ms threshold was BELOW the healthy p95 of 153.8, so it could
  # only pass if zero frames dropped in 10 seconds. It was not a strict gate, it
  # was an unpassable one — which is why it failed 94 consecutive nights.
  #
  # 2026-08-12 — two further defects fixed here; both made this gate fire on
  # evidence it never actually gathered:
  #
  #   1. The smoke run omitted --skip-unity-build-if-clean, so it attempted a
  #      full Unity player build every night. While the Editor licence was
  #      lapsed every build failed, so no recording was produced at all.
  #   2. The measurement then read `files[-1]` — the newest *.h5 by mtime —
  #      regardless of whether this smoke test produced it. With no new
  #      recording it silently scored a PREVIOUS run, so the gate reported a
  #      cadence number for a file it had not created. (Same provenance trap as
  #      feedback_field_naming_for_single_writer_provenance.)
  #
  # The run now skips the build when the player already matches HEAD, and the
  # measurement REQUIRES a recording newer than the smoke test's start time —
  # otherwise it returns 9999 and fails honestly.
  PREFLIGHT_LOG="$RUNTIME/logs/acc-sweep/preflight-$DATE.log"
  PREFLIGHT_REC_DIR="$RUNTIME/preflight_recordings"
  mkdir -p "$PREFLIGHT_REC_DIR"
  PREFLIGHT_START=$(date +%s)
  # caffeinate MUST wrap this too, not just the `claude -p` call below. Without
  # it the smoke test launches Unity into a display-asleep context at 4am, the
  # GPU is throttled, and the gate measures starvation it caused itself. On
  # 2026-08-13 this read 409ms here versus 153ms for the identical command run
  # interactively — and that false FAIL then blocked the caffeinated agent run
  # that followed. The protection was being applied one step too late.
  #
  # --recording_dir keeps these 10s smoke runs OUT of data/recordings. They are
  # ~200-frame stubs; left in the main pool they become the "latest" recording
  # for highway_65 and the nightly sweep scores a 17s straight-only run as if it
  # were a full lap (observed 2026-08-13: 216-frame run scored 99.9, flagged as
  # untrustworthy in the report).
  caffeinate -di ./start_av_stack.sh --force --skip-unity-build-if-clean \
    --duration 10 --track-yaml tracks/highway_65.yml \
    --recording_dir "$PREFLIGHT_REC_DIR" > "$PREFLIGHT_LOG" 2>&1 || true
  PREFLIGHT_DT_P95=$(PREFLIGHT_START="$PREFLIGHT_START" PREFLIGHT_REC_DIR="$PREFLIGHT_REC_DIR" /opt/homebrew/bin/python3 - <<'PYEOF' 2>/dev/null || echo 9999
import glob, os, sys
import h5py, numpy as np
start = float(os.environ.get("PREFLIGHT_START", "0"))
files = sorted(glob.glob(os.environ.get("PREFLIGHT_REC_DIR", "data/recordings") + "/*.h5"), key=os.path.getmtime)
# Only accept a recording this smoke test actually produced. Scoring an older
# file would report cadence for a run that never happened.
files = [f for f in files if os.path.getmtime(f) >= start]
if not files:
    print(9999); sys.exit(0)
with h5py.File(files[-1], "r") as h:
    if "camera/timestamps" not in h:
        print(9999); sys.exit(0)
    ts = h["camera/timestamps"][:]
if len(ts) < 10:
    print(9999); sys.exit(0)
print(int(np.percentile(np.diff(ts), 95) * 1000))
PYEOF
)
  echo "preflight dt_p95: ${PREFLIGHT_DT_P95}ms (threshold ≤250ms)"
  if [ "${PREFLIGHT_DT_P95:-9999}" -gt 250 ]; then
    export AV_NIGHTLY_NO_FRESH_E2E=1
    echo "PRE-FLIGHT FAIL — Step 2.5 fresh /e2e disabled tonight (env var set)"
  else
    echo "pre-flight PASS — Step 2.5 fresh /e2e budget available"
  fi
  echo

  echo "--- claude -p (model=sonnet-5, budget=\$10, max=${CLAUDE_TIMEOUT}s) ---"
  # MCP servers disabled per launchd-OAuth-hang fix.
  # caffeinate -di prevents macOS idle/display sleep during the run; covers
  # the entire subprocess tree including any Unity processes that /e2e spawns.
  caffeinate -di claude -p \
    --model claude-sonnet-5 \
    --output-format text \
    --permission-mode bypassPermissions \
    --max-budget-usd 10.00 \
    --no-session-persistence \
    --strict-mcp-config \
    --mcp-config '{"mcpServers":{}}' \
    --tools "Bash,Edit,Read,Write,Glob,Grep,TodoWrite" \
    < tools/nightly/acc-sweep/PROMPT.md &
  CLAUDE_PID=$!

  ( sleep "$CLAUDE_TIMEOUT"
    if kill -0 "$CLAUDE_PID" 2>/dev/null; then
      echo
      echo "=== TIMEOUT after ${CLAUDE_TIMEOUT}s — killing pid=$CLAUDE_PID ==="
      kill -TERM "$CLAUDE_PID" 2>/dev/null
      sleep 5
      kill -KILL "$CLAUDE_PID" 2>/dev/null
    fi
  ) &
  WATCHDOG_PID=$!

  wait "$CLAUDE_PID"
  EXIT=$?
  kill "$WATCHDOG_PID" 2>/dev/null
  wait "$WATCHDOG_PID" 2>/dev/null
  echo

  echo "=== acc-sweep run end exit=$EXIT $(date -Iseconds) ==="
} > "$LOG" 2>&1

ln -sf "$LOG" "$LOG_DIR/latest.log"
exit ${EXIT:-1}
