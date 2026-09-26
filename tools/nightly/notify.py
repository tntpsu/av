#!/usr/bin/env python3
"""Send the nightly job email.

Reuses the Gmail SMTP credentials that duckAgent already has configured at
/Users/philtullai/ai-agents/duckAgent/.env (SMTP_HOST/PORT/USER/PASS, EMAIL_TO).
Pure stdlib — no pip deps required.

Two modes:

  Legacy (no --job): the body on stdin is sent as plain text — the old
  "tail -n 100 of the log" email.

  Report (--job <nightly|sweep|acc-sweep|process-health>): the job's report in
  data/reports/ is rendered by tools/nightly/report_render.py into a
  professional HTML email (GATE badge, KPIs, Next steps first, results table,
  changes, failure cards) with a plain-text alternative; the stdin log tail
  becomes the appendix and the fallback if the report cannot be parsed.

Usage:
    tail -n 100 run.log | python3 tools/nightly/notify.py "subject" --job acc-sweep --log run.log
    python3 tools/nightly/notify.py "subject" --job sweep --preview /tmp/sweep.html   # render only
    echo "body" | python3 tools/nightly/notify.py "subject"                            # legacy
"""
import argparse
import os
import smtplib
import sys
from email.message import EmailMessage
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from tools.nightly.report_render import render as _render_report   # noqa: E402
except Exception:  # renderer must never take the email down with it
    _render_report = None

DUCK_ENV = Path("/Users/philtullai/ai-agents/duckAgent/.env")


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def _build_message(subject: str, body: str, job: str | None, log_path: str | None):
    """Return (text, html_or_None)."""
    if not job or _render_report is None:
        return body or "(no body)", None
    tail = body
    if not tail.strip() and log_path and Path(log_path).exists():
        tail = "\n".join(Path(log_path).read_text(errors="replace").splitlines()[-100:])
    try:
        text, html = _render_report(job, subject, tail, meta={"Log": log_path or ""})
        return text, html
    except Exception as e:  # degrade to the legacy body rather than fail to send
        return (body or "(no body)") + f"\n\n[report render failed: {type(e).__name__}: {e}]", None


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("subject", nargs="?")
    ap.add_argument("--job", choices=["nightly", "sweep", "acc-sweep", "process-health"])
    ap.add_argument("--log")
    ap.add_argument("--preview", help="write the HTML to this path and exit without sending")
    ap.add_argument("--to", help="override recipient")
    args = ap.parse_args()
    if not args.subject:
        print("usage: notify.py <subject> [--job JOB] [--log PATH] [--preview OUT.html] [body via stdin]", file=sys.stderr)
        return 2

    body = "" if sys.stdin.isatty() else sys.stdin.read()
    text, html = _build_message(args.subject, body, args.job, args.log)

    if args.preview:
        Path(args.preview).write_text(html or f"<pre>{text}</pre>")
        print(f"notify.py: wrote preview {args.preview} (not sent)")
        return 0

    load_dotenv(DUCK_ENV)

    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER")
    pw = os.environ.get("SMTP_PASS")
    to = args.to or os.environ.get("INSIGHTS_EMAIL") or os.environ.get("EMAIL_TO") or user

    if not (user and pw and to):
        print("notify.py: SMTP_USER/SMTP_PASS/EMAIL_TO not set", file=sys.stderr)
        return 1

    msg = EmailMessage()
    msg["From"] = user
    msg["To"] = to
    msg["Subject"] = args.subject
    msg.set_content(text)
    if html:
        msg.add_alternative(html, subtype="html")

    with smtplib.SMTP(host, port) as s:
        s.starttls()
        s.login(user, pw)
        s.send_message(msg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
