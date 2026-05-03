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
    | python3 "$REPO/tools/nightly/notify.py" "$subject" \
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

  echo "--- claude -p (model=sonnet-4-6, budget=\$10, max=${CLAUDE_TIMEOUT}s) ---"
  # MCP servers disabled per launchd-OAuth-hang fix.
  claude -p \
    --model claude-sonnet-4-6 \
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
