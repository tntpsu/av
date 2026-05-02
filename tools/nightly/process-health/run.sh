#!/bin/bash
# Weekly process-health Pareto digest. Invoked by launchd Sundays at 4am.
# Runs `claude -p` against tools/nightly/process-health/PROMPT.md, which
# drives the `/process-health` slash command and emits a Pareto of where
# issues originate based on the improvement_log.json.

set -uo pipefail
# DO NOT set -e: we want to capture exit code, not abort early.

export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH
export AV_NIGHTLY_RUN=1

REPO=/Users/philtullai/av
RUNTIME=/Users/philtullai/av_runtime
LOG_DIR="$RUNTIME/logs/process-health"
DATE=$(date +%Y-%m-%d)
LOG="$LOG_DIR/$DATE.log"
CLAUDE_TIMEOUT=1800  # 30 min ceiling — generous; this is mostly file reads

mkdir -p "$LOG_DIR"

cd "$REPO" || { echo "FATAL: cannot cd to $REPO" > "$LOG"; exit 1; }

notify_on_exit() {
  local exit_code=$?
  local summary
  summary=$(grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2} PROCESS_HEALTH ' "$LOG" 2>/dev/null | tail -1)
  local subject
  if [ -n "$summary" ]; then
    subject="av process-health $DATE: $summary"
  elif [ "$exit_code" = "124" ] || [ "$exit_code" = "143" ] || [ "$exit_code" = "137" ]; then
    subject="av process-health $DATE: TIMED OUT after ${CLAUDE_TIMEOUT}s"
  elif [ "$exit_code" = "0" ]; then
    subject="av process-health $DATE: complete (exit=0, no summary line)"
  else
    subject="av process-health $DATE: WRAPPER FAILED (exit=$exit_code)"
  fi
  tail -n 200 "$LOG" 2>/dev/null \
    | python3 "$REPO/tools/nightly/notify.py" "$subject" \
    >>"$LOG" 2>&1 \
    || echo "notify.py failed (continuing)" >>"$LOG"
}
trap notify_on_exit EXIT

{
  echo "=== process-health run start $(date -Iseconds) ==="
  echo "host:    $(hostname)"
  echo "pwd:     $(pwd)"
  echo "claude:  $(command -v claude || echo MISSING)"
  echo "git head before pull: $(git rev-parse --short HEAD)"
  echo

  echo "--- git fetch + pull ---"
  git fetch origin
  git checkout main
  git pull --ff-only origin main
  echo "git head after pull:  $(git rev-parse --short HEAD)"
  echo

  echo "--- claude -p (model=sonnet-4-6, budget=\$3, max=${CLAUDE_TIMEOUT}s) ---"
  claude -p \
    --model claude-sonnet-4-6 \
    --output-format text \
    --permission-mode bypassPermissions \
    --max-budget-usd 3.00 \
    --no-session-persistence \
    --strict-mcp-config \
    --mcp-config '{"mcpServers":{}}' \
    --tools "Bash,Read,Write,Glob,Grep,TodoWrite" \
    < tools/nightly/process-health/PROMPT.md &
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

  echo "=== process-health run end exit=$EXIT $(date -Iseconds) ==="
} > "$LOG" 2>&1

ln -sf "$LOG" "$LOG_DIR/latest.log"
exit ${EXIT:-1}
