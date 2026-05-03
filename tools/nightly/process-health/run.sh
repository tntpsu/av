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

# Compose subject in three tiers:
#   1. Parse data/reports/process_health_<DATE>.md for the canonical metrics.
#   2. Grep the log for a `^DATE PROCESS_HEALTH ...` line (legacy fallback).
#   3. Synthesize from exit code (TIMED OUT / WRAPPER FAILED / complete).
compose_ph_subject() {
  local exit_code=$1
  local report="$REPO/data/reports/process_health_$DATE.md"
  if [ -r "$report" ]; then
    local total top_stage top_pct
    # "Total entries: **14**"
    total=$(grep -oE 'Total entries:\s+\*\*[0-9]+\*\*' "$report" | head -1 | grep -oE '[0-9]+')
    # First non-zero stage line in the Pareto (e.g. "Design  ████  64%  (9)").
    # Skip the "Process Stage Pareto" header and any 0% rows.
    read -r top_stage top_pct < <(awk '
      /Process Stage Pareto/ { in_p = 1; next }
      in_p && /[A-Za-z].*[0-9]+%.*\([0-9]+\)/ {
        # extract: first word as stage; first NN% as pct
        match($0, /[0-9]+%/)
        pct_str = substr($0, RSTART, RLENGTH-1) + 0
        if (pct_str > 0) { print $1, pct_str; exit }
      }
    ' "$report")
    if [ -n "$total" ] && [ -n "$top_stage" ]; then
      echo "av process-health $DATE: PROCESS_HEALTH stage=${top_stage}(${top_pct}%) entries=${total}"
      return
    fi
  fi
  local legacy
  legacy=$(grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2} PROCESS_HEALTH ' "$LOG" 2>/dev/null | tail -1)
  if [ -n "$legacy" ]; then
    echo "av process-health $DATE: $legacy"
    return
  fi
  case "$exit_code" in
    124|143|137) echo "av process-health $DATE: TIMED OUT after ${CLAUDE_TIMEOUT}s" ;;
    0)           echo "av process-health $DATE: complete (exit=0, no report parsed)" ;;
    *)           echo "av process-health $DATE: WRAPPER FAILED (exit=$exit_code)" ;;
  esac
}

notify_on_exit() {
  local exit_code=$?
  local subject
  subject=$(compose_ph_subject "$exit_code")
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
