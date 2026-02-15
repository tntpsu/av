#!/usr/bin/env python3
"""
Fail if script changes are made without updating docs/SCRIPT_RUNBOOK.md.

Usage:
  python tools/check_runbook_sync.py [changed_file1 changed_file2 ...]

If no files are passed, it falls back to:
  git diff --name-only HEAD~1...HEAD
"""

from __future__ import annotations

import fnmatch
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNBOOK_PATH = "docs/SCRIPT_RUNBOOK.md"

# Keep this intentionally broad for script-like entry points.
SCRIPT_PATTERNS = [
    "*.sh",
    "tools/**/*.py",
    "tools/**/*.sh",
]


def _is_script_path(rel_path: str) -> bool:
    norm = rel_path.strip().replace("\\", "/")
    if not norm:
        return False
    if norm == RUNBOOK_PATH:
        return False
    return any(fnmatch.fnmatch(norm, pattern) for pattern in SCRIPT_PATTERNS)


def _fallback_changed_files() -> list[str]:
    try:
        out = subprocess.check_output(
            ["git", "diff", "--name-only", "HEAD~1...HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
        )
    except Exception:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def main() -> int:
    changed = [p.strip() for p in sys.argv[1:] if p.strip()]
    if not changed:
        changed = _fallback_changed_files()
    if not changed:
        return 0

    script_changes = [p for p in changed if _is_script_path(p)]
    runbook_changed = RUNBOOK_PATH in {p.replace("\\", "/") for p in changed}

    if script_changes and not runbook_changed:
        print("ERROR: Script changes detected without runbook update.", file=sys.stderr)
        print(f"Please update `{RUNBOOK_PATH}` in the same change.", file=sys.stderr)
        print("Changed script-like paths:", file=sys.stderr)
        for p in script_changes:
            print(f"  - {p}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

