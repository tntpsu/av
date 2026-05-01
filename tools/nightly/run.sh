#!/bin/bash
# Nightly test-fix wrapper. Invoked by launchd at 3am local time.
# Sets up env, pulls main, runs `claude -p` against tools/nightly/PROMPT.md
# under a hard 30-min timeout, logs to ~/av_runtime/logs/nightly/<date>.log,
# and emails a completion summary on every exit (success or failure).

set -uo pipefail
# DO NOT set -e: we want to capture exit code, not abort early.

# launchd starts processes with a sparse PATH — set it explicitly.
export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH

REPO=/Users/philtullai/av
RUNTIME=/Users/philtullai/av_runtime
LOG_DIR="$RUNTIME/logs/nightly"
DATE=$(date +%Y-%m-%d)
LOG="$LOG_DIR/$DATE.log"
CLAUDE_TIMEOUT=1800  # 30 min hard ceiling

mkdir -p "$LOG_DIR"

cd "$REPO" || { echo "FATAL: cannot cd to $REPO" > "$LOG"; exit 1; }

# Email on every exit, even unexpected ones (git conflict, kill, etc).
notify_on_exit() {
  local exit_code=$?
  local summary
  summary=$(grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2} FIXES=' "$LOG" 2>/dev/null | tail -1)
  local subject
  if [ -n "$summary" ]; then
    subject="av nightly $DATE: $summary"
  elif [ "$exit_code" = "124" ] || [ "$exit_code" = "143" ] || [ "$exit_code" = "137" ]; then
    subject="av nightly $DATE: TIMED OUT after ${CLAUDE_TIMEOUT}s"
  elif [ "$exit_code" = "0" ]; then
    subject="av nightly $DATE: complete (exit=0, no summary line)"
  else
    subject="av nightly $DATE: WRAPPER FAILED (exit=$exit_code)"
  fi
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
