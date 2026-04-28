#!/usr/bin/env python3
"""Send email notification for the nightly test-fix job.

Reuses the Gmail SMTP credentials that duckAgent already has configured at
/Users/philtullai/ai-agents/duckAgent/.env (SMTP_HOST/PORT/USER/PASS, EMAIL_TO).
Pure stdlib — no pip deps required.

Usage:
    python3 tools/nightly/notify.py "subject line" < body.txt
    echo "body" | python3 tools/nightly/notify.py "subject line"
"""
import os
import smtplib
import sys
from email.message import EmailMessage
from pathlib import Path

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


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: notify.py <subject> [body via stdin]", file=sys.stderr)
        return 2

    load_dotenv(DUCK_ENV)

    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER")
    pw = os.environ.get("SMTP_PASS")
    to = os.environ.get("INSIGHTS_EMAIL") or os.environ.get("EMAIL_TO") or user

    if not (user and pw and to):
        print("notify.py: SMTP_USER/SMTP_PASS/EMAIL_TO not set", file=sys.stderr)
        return 1

    subject = sys.argv[1]
    body = sys.stdin.read() or "(no body)"

    msg = EmailMessage()
    msg["From"] = user
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(host, port) as s:
        s.starttls()
        s.login(user, pw)
        s.send_message(msg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
