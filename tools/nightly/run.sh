#!/bin/bash
# Nightly test-fix wrapper. Invoked by launchd at 3am local time.
# Sets up env, pulls main, runs `claude -p` against tools/nightly/PROMPT.md
# under a hard 30-min timeout, logs to ~/av_runtime/logs/nightly/<date>.log,
# and emails a completion summary on every exit (success or failure).

set -uo pipefail
# DO NOT set -e: we want to capture exit code, not abort early.

# launchd starts processes with a sparse PATH — set it explicitly.
export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH

# Mark this as a nightly run so hardware-sensitive perf tests can self-skip;
# Mac mini under 3am+pytest+stack-spawn load can't honor sub-20ms budgets.
export AV_NIGHTLY_RUN=1

REPO=/Users/philtullai/av
RUNTIME=/Users/philtullai/av_runtime
LOG_DIR="$RUNTIME/logs/nightly"
DATE=$(date +%Y-%m-%d)
LOG="$LOG_DIR/$DATE.log"
CLAUDE_TIMEOUT=3600  # 60 min hard ceiling — accommodates 2x pytest (~11 min each) + analysis

mkdir -p "$LOG_DIR"

cd "$REPO" || { echo "FATAL: cannot cd to $REPO" > "$LOG"; exit 1; }

# Compose email subject. Tries three sources in order:
#   1. Parse data/reports/nightly_test_report.txt (the agent's structured
#      report — most reliable, doesn't depend on the agent printing the
#      summary line literally to stdout).
#   2. Grep the log for a `^DATE FIXES=...` line (legacy fallback).
#   3. Synthesize from exit code (TIMED OUT / WRAPPER FAILED / complete).
compose_subject() {
  local exit_code=$1
  local report="$REPO/data/reports/nightly_test_report.txt"
  if [ -r "$report" ]; then
    local fixed real_breaks flaky delivery
    fixed=$(grep -E '^Fixed:' "$report" | head -1 | awk '{print $2}')
    real_breaks=$(grep -E '^Real breaks' "$report" | head -1 | awk '{print $4}')
    flaky=$(grep -E '^Flaky:' "$report" | head -1 | awk '{print $2}')
    delivery=$(grep -E 'step6_done delivery=' "$REPO/data/reports/nightly_status.txt" 2>/dev/null \
                 | tail -1 | sed -nE 's/.*delivery=([a-z_]+).*/\1/p')
    [ -z "$delivery" ] && delivery=local_only
    if [ -n "$fixed" ] && [ -n "$real_breaks" ] && [ -n "$flaky" ]; then
      echo "av nightly $DATE: FIXES=$fixed REAL_BREAKS=$real_breaks FLAKY=$flaky delivery=$delivery"
      return
    fi
  fi
  local legacy
  legacy=$(grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2} FIXES=' "$LOG" 2>/dev/null | tail -1)
  if [ -n "$legacy" ]; then
    echo "av nightly $DATE: $legacy"
    return
  fi
  case "$exit_code" in
    124|143|137) echo "av nightly $DATE: TIMED OUT after ${CLAUDE_TIMEOUT}s" ;;
    0)           echo "av nightly $DATE: complete (exit=0, no report parsed)" ;;
    *)           echo "av nightly $DATE: WRAPPER FAILED (exit=$exit_code)" ;;
  esac
}

# Email on every exit, even unexpected ones (git conflict, kill, etc).
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
  echo "=== nightly run start $(date -Iseconds) ==="
  echo "host:    $(hostname)"
  echo "pwd:     $(pwd)"
  echo "claude:  $(command -v claude || echo MISSING)"
  echo "git:     $(command -v git)"
  echo "gh:      $(command -v gh || echo MISSING)"
  echo "git head before pull: $(git rev-parse --short HEAD)"
  echo

  echo "--- git fetch + pull ---"
  git fetch origin
  git checkout main
  git pull --ff-only origin main
  echo "git head after pull:  $(git rev-parse --short HEAD)"
  echo

  echo "--- claude -p (model=sonnet-4-6, budget=\$5, max=${CLAUDE_TIMEOUT}s) ---"
  # Background claude (stdin redirected from PROMPT.md to avoid the
  # variadic-flag collision with --tools that bit us before).
  # MCP servers disabled: under launchd's non-interactive context, an
  # MCP server needing OAuth re-auth (e.g. Google Calendar) hangs the
  # whole process — strict-mcp-config + empty inline config skips them.
  claude -p \
    --model claude-sonnet-4-6 \
    --output-format text \
    --permission-mode bypassPermissions \
    --max-budget-usd 5.00 \
    --no-session-persistence \
    --strict-mcp-config \
    --mcp-config '{"mcpServers":{}}' \
    --tools "Bash,Edit,Read,Write,Glob,Grep,TodoWrite" \
    < tools/nightly/PROMPT.md &
  CLAUDE_PID=$!

  # Watchdog: SIGTERM after CLAUDE_TIMEOUT, SIGKILL if still alive 5s later.
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
  # Cancel watchdog if claude finished on its own.
  kill "$WATCHDOG_PID" 2>/dev/null
  wait "$WATCHDOG_PID" 2>/dev/null
  echo

  echo "=== nightly run end exit=$EXIT $(date -Iseconds) ==="
} > "$LOG" 2>&1

ln -sf "$LOG" "$LOG_DIR/latest.log"
exit ${EXIT:-1}
