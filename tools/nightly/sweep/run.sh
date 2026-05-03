#!/bin/bash
# Nightly cross-track regression sweep. Invoked by launchd at 3am local time.
# Runs `claude -p` against tools/nightly/sweep/PROMPT.md, which drives the
# `/sweep` slash command across all 6 tracks and compares against baselines.
# Logs to ~/av_runtime/logs/sweep/<date>.log, emails completion summary.

set -uo pipefail
# DO NOT set -e: we want to capture exit code, not abort early.

# launchd starts processes with a sparse PATH — set it explicitly.
export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH

# Mark this as a nightly run so hardware-sensitive perf tests can self-skip.
export AV_NIGHTLY_RUN=1

REPO=/Users/philtullai/av
RUNTIME=/Users/philtullai/av_runtime
LOG_DIR="$RUNTIME/logs/sweep"
DATE=$(date +%Y-%m-%d)
LOG="$LOG_DIR/$DATE.log"
CLAUDE_TIMEOUT=5400  # 90 min hard ceiling — 6 Unity launches + rebuilds + analysis

mkdir -p "$LOG_DIR"

cd "$REPO" || { echo "FATAL: cannot cd to $REPO" > "$LOG"; exit 1; }

# Compose email subject. Tries three sources in order:
#   1. Parse data/reports/sweep_status.txt (the agent's per-track heartbeat
#      lines: `step_track_<name>_done score=<N> baseline=<N> delta=<N> [FLAG=...]`)
#      — most reliable, doesn't depend on the agent printing the summary literally.
#   2. Grep the log for a `^DATE SWEEP ...` line (legacy fallback).
#   3. Synthesize from exit code (TIMED OUT / WRAPPER FAILED / complete).
compose_subject() {
  local exit_code=$1
  local hb="$REPO/data/reports/sweep_status.txt"
  if [ -r "$hb" ] && grep -q 'step_track_.*_done' "$hb"; then
    local passed regressions flags worst_track worst_delta gate
    # Per /sweep skill: regression = delta < -2.0. Flags are surfaced by the
    # agent in the FLAG= field (it applies the per-layer threshold itself).
    passed=$(grep -c '^step_track_.*_done' "$hb" 2>/dev/null || true)
    regressions=$(awk '/^step_track_.*_done/ {
      if (match($0, /delta=(-?[0-9.]+)/)) {
        d = substr($0, RSTART+6, RLENGTH-6) + 0
        if (d < -2.0) c++
      }
    } END {print c+0}' "$hb")
    flags=$(grep -cE '^step_track_.*_done.*FLAG=' "$hb" 2>/dev/null || true)
    read -r worst_track worst_delta < <(awk '
      /^step_track_.*_done/ {
        name=$1; sub(/^step_track_/,"",name); sub(/_done/,"",name);
        if (match($0, /delta=(-?[0-9.]+)/)) {
          d = substr($0, RSTART+6, RLENGTH-6) + 0
          if (NR==1 || d < min_d) { min_d = d; min_n = name }
        }
      } END { if (min_n) print min_n, min_d; else print "none 0.0" }' "$hb")
    # Gate verdict: prefer the canonical "GATE: ..." line from sweep_report.txt
    # (which applies the layer->=95 rule too). Fall back to regressions-only
    # check if the report file isn't present.
    local report="$REPO/data/reports/sweep_report.txt"
    gate=""
    if [ -r "$report" ]; then
      gate=$(grep -E '^GATE:' "$report" | head -1 | awk '{print $2}')
    fi
    if [ -z "$gate" ]; then
      if [ "$regressions" -eq 0 ]; then gate=PASS; else gate=FAIL; fi
    fi
    echo "av sweep $DATE: SWEEP gate=$gate regressions=$regressions flags=$flags passed=$passed/6 worst=${worst_track}(${worst_delta})"
    return
  fi
  local legacy
  legacy=$(grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2} SWEEP ' "$LOG" 2>/dev/null | tail -1)
  if [ -n "$legacy" ]; then
    echo "av sweep $DATE: $legacy"
    return
  fi
  case "$exit_code" in
    124|143|137) echo "av sweep $DATE: TIMED OUT after ${CLAUDE_TIMEOUT}s" ;;
    0)           echo "av sweep $DATE: complete (exit=0, no heartbeat parsed)" ;;
    *)           echo "av sweep $DATE: WRAPPER FAILED (exit=$exit_code)" ;;
  esac
}

notify_on_exit() {
  local exit_code=$?
  local subject
  subject=$(compose_subject "$exit_code")
  tail -n 100 "$LOG" 2>/dev/null \
    | python3 "$REPO/tools/nightly/notify.py" "$subject" \
    >>"$LOG" 2>&1 \
    || echo "notify.py failed (continuing)" >>"$LOG"
}
trap notify_on_exit EXIT

{
  echo "=== sweep run start $(date -Iseconds) ==="
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

  echo "--- claude -p (model=sonnet-4-6, budget=\$10, max=${CLAUDE_TIMEOUT}s) ---"
  # MCP servers disabled: under launchd's non-interactive context, an
  # MCP server needing OAuth re-auth (e.g. Google Calendar) hangs the
  # whole process — strict-mcp-config + empty inline config skips them.
  claude -p \
    --model claude-sonnet-4-6 \
    --output-format text \
    --permission-mode bypassPermissions \
    --max-budget-usd 10.00 \
    --no-session-persistence \
    --strict-mcp-config \
    --mcp-config '{"mcpServers":{}}' \
    --tools "Bash,Edit,Read,Write,Glob,Grep,TodoWrite" \
    < tools/nightly/sweep/PROMPT.md &
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

  echo "=== sweep run end exit=$EXIT $(date -Iseconds) ==="
} > "$LOG" 2>&1

ln -sf "$LOG" "$LOG_DIR/latest.log"
exit ${EXIT:-1}
