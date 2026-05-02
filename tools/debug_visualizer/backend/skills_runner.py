"""Skills runner backend for PhilViz Skills page.

Provides:
  - Skill discovery from `.claude/commands/*.md`
  - Detached subprocess pool for `claude -p` invocations
  - Live output buffering keyed by job_id (so mobile users can disconnect/reconnect)
  - Cancel via SIGTERM

Subprocess lifecycle is intentionally detached from any HTTP connection so
that closing the browser tab (locking phone, VPN drop) does not kill the
running skill. Output is buffered in-memory with a per-job cap.

V1 scope: in-process, no persistence across server restarts. If the server
restarts mid-skill, jobs are orphaned (subprocesses keep running but become
unowned). For V1, this is acceptable — server restarts are rare.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
COMMANDS_DIR = REPO_ROOT / ".claude" / "commands"
CLAUDE_BIN = "/opt/homebrew/bin/claude"

# Per-job output line buffer cap. Skills emit ~hundreds of lines normally;
# 5000 is enough for /sweep + /e2e while bounding memory.
MAX_BUFFER_LINES = 5000

# Hard timeout for a single skill invocation (matches nightly wrapper).
DEFAULT_TIMEOUT_S = 5400  # 90 min

# Per-skill budget cap.
DEFAULT_BUDGET_USD = 5.00


@dataclass
class SkillJob:
    job_id: str
    skill: str
    args: str
    started_at: float
    proc: Optional[subprocess.Popen] = None
    buffer: deque = field(default_factory=lambda: deque(maxlen=MAX_BUFFER_LINES))
    status: str = "running"          # running | completed | cancelled | failed
    exit_code: Optional[int] = None
    completed_at: Optional[float] = None
    lock: threading.Lock = field(default_factory=threading.Lock)


# Module-level job registry (in-process).
_jobs: dict[str, SkillJob] = {}
_jobs_lock = threading.Lock()


def list_skills() -> list[dict]:
    """Discover available skills from .claude/commands/*.md.

    Returns a list of {name, description, takes_args} dicts.
    Description is the first non-empty line of the file (trimmed).
    takes_args is True if the file contains '$ARGUMENTS'.
    """
    skills = []
    if not COMMANDS_DIR.is_dir():
        return skills
    for path in sorted(COMMANDS_DIR.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        # First non-empty line is the description.
        description = ""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                description = stripped
                break
        skills.append({
            "name": path.stem,
            "description": description[:240],
            "takes_args": "$ARGUMENTS" in text,
        })
    return skills


def _build_prompt(skill: str, args: str) -> Optional[str]:
    """Build the prompt body to feed to `claude -p` stdin.

    Substitutes `$ARGUMENTS` with the user-supplied args (empty string if none).
    Returns None if the skill file does not exist.
    """
    path = COMMANDS_DIR / f"{skill}.md"
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    return text.replace("$ARGUMENTS", args)


def _reader_thread(job: SkillJob) -> None:
    """Read subprocess stdout line-by-line into the job's buffer."""
    if job.proc is None or job.proc.stdout is None:
        return
    try:
        for raw in iter(job.proc.stdout.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            with job.lock:
                job.buffer.append(line)
    except Exception as e:
        with job.lock:
            job.buffer.append(f"[reader-error] {e!r}")
    finally:
        # Wait for process to fully exit so we can capture the return code.
        if job.proc is not None:
            job.proc.wait()
        with job.lock:
            if job.status == "running":
                job.status = "completed" if (job.proc and job.proc.returncode == 0) else "failed"
            job.exit_code = job.proc.returncode if job.proc else -1
            job.completed_at = time.time()
            job.buffer.append(f"[skill-runner] exit={job.exit_code} status={job.status}")


def start_job(skill: str, args: str = "",
              budget_usd: float = DEFAULT_BUDGET_USD) -> Optional[str]:
    """Spawn `claude -p` for the given skill, return job_id (or None if skill unknown)."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", skill):
        return None  # reject paths/escapes
    prompt = _build_prompt(skill, args)
    if prompt is None:
        return None

    job_id = uuid.uuid4().hex[:12]
    job = SkillJob(job_id=job_id, skill=skill, args=args, started_at=time.time())

    cmd = [
        CLAUDE_BIN, "-p",
        "--model", "claude-sonnet-4-6",
        "--output-format", "text",
        "--permission-mode", "bypassPermissions",
        "--max-budget-usd", f"{budget_usd:.2f}",
        "--no-session-persistence",
        "--strict-mcp-config",
        "--mcp-config", '{"mcpServers":{}}',
        "--tools", "Bash,Edit,Read,Write,Glob,Grep,TodoWrite",
    ]

    env = os.environ.copy()
    env["AV_NIGHTLY_RUN"] = "1"  # Hardware-sensitive perf tests self-skip.

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT),
        env=env,
        bufsize=1,           # line-buffered
        start_new_session=True,  # detach from server's process group → SIGTERM only when explicit
    )
    job.proc = proc

    # Pipe the prompt via stdin then close.
    try:
        if proc.stdin is not None:
            proc.stdin.write(prompt.encode("utf-8"))
            proc.stdin.close()
    except Exception as e:
        with job.lock:
            job.buffer.append(f"[skill-runner] stdin write failed: {e!r}")

    with _jobs_lock:
        _jobs[job_id] = job

    threading.Thread(target=_reader_thread, args=(job,), daemon=True).start()
    return job_id


def get_job(job_id: str) -> Optional[SkillJob]:
    with _jobs_lock:
        return _jobs.get(job_id)


def list_jobs(limit: int = 20) -> list[dict]:
    """Return a snapshot of recent jobs (most recent first)."""
    with _jobs_lock:
        snapshot = list(_jobs.values())
    snapshot.sort(key=lambda j: j.started_at, reverse=True)
    out = []
    for j in snapshot[:limit]:
        with j.lock:
            out.append({
                "job_id": j.job_id,
                "skill": j.skill,
                "args": j.args,
                "status": j.status,
                "exit_code": j.exit_code,
                "started_at": j.started_at,
                "completed_at": j.completed_at,
                "lines": len(j.buffer),
            })
    return out


def cancel_job(job_id: str) -> bool:
    job = get_job(job_id)
    if job is None or job.proc is None:
        return False
    if job.status != "running":
        return False
    try:
        # Kill the whole process group — claude often spawns child processes.
        os.killpg(os.getpgid(job.proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    with job.lock:
        job.status = "cancelled"
        job.buffer.append("[skill-runner] cancel requested via SIGTERM")
    return True


def stream_output(job_id: str, last_index: int = 0):
    """Generator that yields buffered output lines as they appear.

    Polls the buffer; sleeps briefly between polls. Exits when the job
    completes AND all buffered lines have been flushed. Caller is expected
    to wrap each yielded line in SSE format (`data: ...\\n\\n`).
    """
    job = get_job(job_id)
    if job is None:
        yield "[skill-runner] unknown job_id"
        return

    while True:
        with job.lock:
            current_lines = list(job.buffer)
            done = job.status != "running"
        # Yield any new lines past last_index.
        if last_index < len(current_lines):
            for line in current_lines[last_index:]:
                yield line
            last_index = len(current_lines)
        if done and last_index >= len(current_lines):
            break
        time.sleep(0.5)
