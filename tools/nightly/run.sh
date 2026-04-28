#!/bin/bash
# Nightly test-fix wrapper. Invoked by launchd at 3am local time.
# Sets up env, pulls main, runs `claude -p` against tools/nightly/PROMPT.md,
# logs everything to ~/av_runtime/logs/nightly/<date>.log.

set -uo pipefail
# DO NOT set -e: we want to capture exit code, not abort early.

# launchd starts processes with a sparse PATH — set it explicitly.
export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH

REPO=/Users/philtullai/av
RUNTIME=/Users/philtullai/av_runtime
LOG_DIR="$RUNTIME/logs/nightly"
DATE=$(date +%Y-%m-%d)
LOG="$LOG_DIR/$DATE.log"

mkdir -p "$LOG_DIR"

cd "$REPO" || { echo "FATAL: cannot cd to $REPO" > "$LOG"; exit 1; }

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

  echo "--- claude -p (model=sonnet-4-6, budget=\$5) ---"
  # Pipe prompt via stdin: --tools is a variadic flag and would otherwise
  # consume the positional prompt argument as additional tool names.
  cat tools/nightly/PROMPT.md | claude -p \
    --model claude-sonnet-4-6 \
    --output-format text \
    --permission-mode bypassPermissions \
    --max-budget-usd 5.00 \
    --no-session-persistence \
    --tools "Bash,Edit,Read,Write,Glob,Grep,TodoWrite"
  EXIT=$?
  echo

  echo "=== nightly run end exit=$EXIT $(date -Iseconds) ==="
} > "$LOG" 2>&1

ln -sf "$LOG" "$LOG_DIR/latest.log"
exit ${EXIT:-1}
