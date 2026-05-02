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

# Email on every exit, even unexpected ones.
notify_on_exit() {
  local exit_code=$?
  local summary
  summary=$(grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2} SWEEP ' "$LOG" 2>/dev/null | tail -1)
  local subject
  if [ -n "$summary" ]; then
    subject="av sweep $DATE: $summary"
  elif [ "$exit_code" = "124" ] || [ "$exit_code" = "143" ] || [ "$exit_code" = "137" ]; then
    subject="av sweep $DATE: TIMED OUT after ${CLAUDE_TIMEOUT}s"
  elif [ "$exit_code" = "0" ]; then
    subject="av sweep $DATE: complete (exit=0, no summary line)"
  else
    subject="av sweep $DATE: WRAPPER FAILED (exit=$exit_code)"
  fi
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
