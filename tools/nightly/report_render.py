#!/usr/bin/env python3
"""Render the nightly job reports as a professional HTML email (plus plain text).

The wrappers used to mail `tail -n 100` of the log — a splash of text. This module
reads the job's report files in `data/reports/` and renders:

    header  — job, date, GATE badge, KPI chips (pass/fail/warn, preflight, model)
    next    — "Next steps" first: every `Action:` / `Next action:` line, de-duplicated
    results — the per-scenario / per-track table with coloured verdicts
    changes — "Changes from last night" block, verbatim
    detail  — one card per FAIL/WARN with Root cause and Action
    appendix — the raw report and the last lines of the log, small and monospaced

Everything is best-effort: any parse failure degrades to the raw report (or the
log tail) inside the same shell, never to a missing email. Parsers are keyed on
the structures the PROMPTs contract (`GATE:` line, table rows, `Action:` /
`Root cause:` lines, `Changes from Night-N:` block) — keep those in sync.

Usage (from notify.py):
    text, html = render(job, subject, reports_dir, log_tail)
Standalone preview:
    python3 tools/nightly/report_render.py acc-sweep --preview /tmp/acc.html
"""
from __future__ import annotations

import html
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS = REPO_ROOT / "data" / "reports"

JOB_TITLES = {
    "nightly": "Nightly test-fix",
    "sweep": "Lateral cross-track sweep",
    "acc-sweep": "ACC scenario sweep",
    "process-health": "Process health (weekly)",
}

VERDICT_COLOURS = {
    "PASS": ("#0f7b3e", "#e6f4ea"),
    "FAIL": ("#b3261e", "#fce8e6"),
    "WARN": ("#9a6700", "#fff4d6"),
    "FLAG": ("#9a6700", "#fff4d6"),
    "SKIP": ("#5f6368", "#f1f3f4"),
    "SKIPPED": ("#5f6368", "#f1f3f4"),
    "AMB": ("#5f6368", "#f1f3f4"),
    "AMBIGUOUS": ("#5f6368", "#f1f3f4"),
    "UNKNOWN": ("#5f6368", "#f1f3f4"),
}


@dataclass
class Card:
    title: str
    lines: List[str] = field(default_factory=list)      # body lines (verbatim)
    root_cause: str = ""
    action: str = ""
    verdict: str = ""


@dataclass
class Report:
    job: str
    date: str
    gate: str = "UNKNOWN"
    kpis: List[Tuple[str, str]] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    verdict_col: Optional[int] = None
    next_steps: List[str] = field(default_factory=list)
    changes: List[str] = field(default_factory=list)
    cards: List[Card] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)      # banners (frozen pool, contamination …)
    raw: str = ""
    parse_error: str = ""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _first(pattern: str, text: str, flags=re.M) -> str:
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else ""


def _dedupe(items: List[str]) -> List[str]:
    seen, out = set(), []
    for it in items:
        key = re.sub(r"\s+", " ", it.strip().lower())
        if key and key not in seen:
            seen.add(key)
            out.append(it.strip())
    return out


def _block_after(heading_regex: str, text: str, stop_regex: str = r"^\s*$") -> List[str]:
    """Lines following a heading line until a blank line (or `stop_regex`)."""
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if re.search(heading_regex, ln):
            out: List[str] = []
            for nxt in lines[i + 1:]:
                if re.match(stop_regex, nxt):
                    break
                out.append(nxt.rstrip())
            return out
    return []


def _continuation(lines: List[str], start: int) -> str:
    """Join a `Key: value` line with its indented continuation lines."""
    first = lines[start]
    val = first.split(":", 1)[1].strip() if ":" in first else first.strip()
    indent = len(first) - len(first.lstrip())
    j = start + 1
    while j < len(lines):
        nxt = lines[j]
        if not nxt.strip():
            break
        ind = len(nxt) - len(nxt.lstrip())
        if ind < indent or re.match(r"^\s*[A-Z][A-Za-z /-]{1,24}:\s", nxt):
            break
        val += " " + nxt.strip()
        j += 1
    return val


_ACTION_LEAD = re.compile(
    r"^(Fix(?: options)?|Human(?: decision(?: needed)?)?|Recommend(?:ed|ation)?|Re-seed|Action|Next(?: action| steps?)?|"
    r"Requires|Suggest(?:ed|ion)?|Decision|Escalate|Investigate|Schedule|To unfreeze|→)\b", re.I)


def _extract_actions(text: str) -> List[str]:
    """Sentences that read as things a human should do next.

    The agents phrase these many ways (`Action:`, `Fix:`, `Human decision needed:`,
    `Re-seed:`, `Requires …`, `→ …`). Split on sentence/line boundaries and keep
    the ones that start with an action lead-in."""
    out: List[str] = []
    flat = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z→(])|\s(?=(?:Fix|Human decision|Re-seed|Action|Next action|Recommend)[a-z ]*:)", flat)
    for part in parts:
        part = part.strip(" -•")
        if len(part) < 12:
            continue
        if _ACTION_LEAD.match(part):
            out.append(part.rstrip("."))
    return out


def _clean_card_title(sid: str, rest: str) -> str:
    rest = rest.strip()
    m = re.match(r"^([a-z][a-z0-9_]+)\)\s*(.*)$", rest)          # "name) — FAIL [..]"
    if m:
        return f"{sid} — {m.group(1)} {m.group(2)}".strip()
    return f"{sid} — {rest}"


# ---------------------------------------------------------------------------
# parsers
# ---------------------------------------------------------------------------

# Row shapes seen so far: "A1  (autobahn_a1_steady)  autobahn_30  0.99d  FAIL  n/a  note" (night 43)
# and "A1  autobahn_a1_steady  autobahn_30  2d  FAIL  n/a  note" (night 44). Parens optional.
_ACC_ROW = re.compile(
    r"^(?P<id>[AGH]\d{1,2})\s+\(?(?P<name>[a-z][a-z0-9_]+)\)?\s+(?P<base>[a-z][a-z0-9_]+)\s+(?P<age>\S+)\s+"
    r"(?P<verdict>PASS|FAIL|WARN|SKIP(?:PED)?|AMB(?:IGUOUS)?)\s+(?P<score>\S+)\s*(?P<sub>.*)$"
)


def parse_acc_sweep(text: str, date: str) -> Report:
    r = Report(job="acc-sweep", date=date, raw=text)
    r.gate = _first(r"^GATE:\s*(\w+)", text) or "UNKNOWN"
    night = _first(r"\(Night\s*(\d+)\)", text)
    mode = _first(r"^Mode:\s*(.+)$", text)
    counts = _first(r"^(pass=\d+.*total=\d+)", text, re.M | re.I)
    frame = _first(r"^(?:Scoring frame|Post-09-22 scoring):\s*(.+)$", text)
    if night:
        r.kpis.append(("Night", night))
    for kv in re.findall(r"(pass|fail|warn|skip|amb)=(\d+)", counts, re.I):
        r.kpis.append((kv[0].upper(), kv[1]))
    if mode:
        r.kpis.append(("Mode", mode))
    if frame:
        r.notes.append("Scoring: " + frame)

    r.columns = ["Scenario", "Base track", "Recording", "Verdict", "Score", "Sub-scores / note"]
    r.verdict_col = 3
    for ln in text.splitlines():
        m = _ACC_ROW.match(ln.strip())
        if m:
            r.rows.append([f"{m['id']} {m['name']}", m["base"], m["age"], m["verdict"], m["score"], m["sub"].strip()])

    r.changes = [ln.strip() for ln in _block_after(r"^(Changes (from|vs)|DELTA vs|What changed|CHANGES)", text) if ln.strip()]
    contact = re.findall(r"^\s*Physical Contact:.*$", text, re.M)
    r.notes.extend(c.strip() for c in contact[:3])

    # Failure / warning cards: blocks headed "A1 — name [recording, age]" or "A1 (name) — FAIL [..]"
    lines = text.splitlines()
    head_re = re.compile(r"^(?P<id>[AGH]\d{1,2})\s+(?:—|\(|-)\s*(?P<rest>.*)$")
    i = 0
    while i < len(lines):
        m = head_re.match(lines[i].strip())
        if m and not _ACC_ROW.match(lines[i].strip()) and i + 1 < len(lines) and lines[i + 1].startswith("  "):
            card = Card(title=_clean_card_title(m['id'], m['rest']))
            j = i + 1
            while j < len(lines) and (lines[j].startswith("  ") or not lines[j].strip()):
                if not lines[j].strip() and j + 1 < len(lines) and not lines[j + 1].startswith("  "):
                    break
                s = lines[j].strip()
                if re.match(r"^(Root cause|Root):", s):
                    card.root_cause = _continuation(lines, j)
                elif re.match(r"^Action:", s):
                    card.action = _continuation(lines, j)
                    r.next_steps.append(f"{m['id']}: {card.action}")
                elif s:
                    card.lines.append(s)
                j += 1
            vm = re.search(r"\b(PASS|FAIL|WARN)\b", m["rest"]) or re.search(r"\b(FAIL|WARN)\b", " ".join(card.lines[:3]))
            card.verdict = vm.group(1) if vm else ""
            if not card.action:
                found = _extract_actions(" ".join([card.root_cause] + card.lines))
                if found:
                    card.action = "; ".join(found)
                    r.next_steps.extend(f"{m['id']}: {a}" for a in found)
            r.cards.append(card)
            i = j
            continue
        i += 1
    nxt = re.findall(r"^\s*(?:Next action|Next steps?|Recommend(?:ation)?):\s*(.+)$", text, re.M | re.I)
    r.next_steps.extend(nxt)
    r.next_steps = _dedupe(r.next_steps)
    return r


# Baseline may carry a footnote marker ("91.4*" = stale baseline); gate may have a note.
_SWEEP_ROW = re.compile(
    r"^(?P<track>[a-z][a-z0-9_]+)\s+(?P<base>[\d.]+\*?)\s+(?P<now>[\d.]+)\s+(?P<delta>[+-]?[\d.]+)\s+"
    r"(?P<traj>[\d.]+)?\s*(?P<util>[\d.]+)?\s*(?P<gate>PASS|FAIL|FLAG)(?P<note>.*)$"
)


def parse_sweep(text: str, date: str) -> Report:
    r = Report(job="sweep", date=date, raw=text)
    r.gate = _first(r"^GATE:\s*(\w+)", text) or "UNKNOWN"
    night = _first(r"\(night\s*(\d+)\)", text, re.M | re.I)
    if night:
        r.kpis.append(("Night", night))
    frozen = _first(r"^(⚠\s*FROZEN RECORDINGS.*)$", text)
    if frozen:
        r.notes.append(frozen)
    for key in ("Regressions", "New e-stops", "Flags"):
        v = _first(rf"^{key}[^:]*:\s*(.+)$", text)
        if v:
            r.kpis.append((key, v))
    has_util = bool(re.search(r"\bUtil\b", text))
    r.columns = ["Track", "Baseline", "Now", "Delta", "Trajectory"] + (["Util"] if has_util else []) + ["Gate"]
    r.verdict_col = len(r.columns) - 1
    for ln in text.splitlines():
        m = _SWEEP_ROW.match(ln.strip())
        if m:
            row = [m["track"], m["base"], m["now"], m["delta"], m["traj"] or ""]
            if has_util:
                row.append(m["util"] or "")
            row.append((m["gate"] + (m["note"] or "")).strip())
            r.rows.append(row)
    r.changes = [ln.strip() for ln in _block_after(r"^Standing failures", text) if ln.strip()]
    for ln in text.splitlines():
        s = ln.strip()
        if re.match(r"^(Next action|Do NOT tune|→)", s):
            r.next_steps.append(s)
        elif re.match(r"^(DIAGNOSIS|STALENESS NOTE|Layer failures)", s):
            r.notes.append(s)
            r.next_steps.extend(_extract_actions(s.split(":", 1)[-1]))
    util_lines = re.findall(r"^\s*(?:Speed Utilisation|Util)[^\n]*$", text, re.M)
    r.notes.extend(u.strip() for u in util_lines[:6])
    r.next_steps = _dedupe(r.next_steps)
    return r


def parse_nightly(text: str, status: str, date: str) -> Report:
    r = Report(job="nightly", date=date, raw=text)
    for key in ("Total", "Passed", "Fixed", "Real breaks \\(unfixed\\)", "Flaky"):
        v = _first(rf"^{key}:\s*(\d+)", text)
        if v:
            r.kpis.append((key.replace("\\", "").replace(" (unfixed)", ""), v))
    real = _first(r"^Real breaks \(unfixed\):\s*(\d+)", text)
    r.gate = "PASS" if real == "0" else ("FAIL" if real else "UNKNOWN")
    delivery = _first(r"delivery=(\w+)", status)
    if delivery:
        r.kpis.append(("Delivery", delivery))
    r.columns = ["Class", "Test", "Reason"]
    r.verdict_col = 0
    section = ""
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("## "):
            section = ln[3:].strip()
            continue
        m = re.match(r"^\s*\[(FIXED|FLAKY|REAL_BREAK)\]\s+(\S+)", ln)
        if m:
            reason = ""
            for nxt in lines[i + 1:i + 4]:
                if nxt.strip().startswith("—"):
                    reason = nxt.strip().lstrip("—").strip()
                    break
            cls = {"FIXED": "PASS", "FLAKY": "WARN", "REAL_BREAK": "FAIL"}[m.group(1)]
            r.rows.append([cls, m.group(2), reason])
            if m.group(1) == "REAL_BREAK":
                r.next_steps.append(f"Review REAL_BREAK {m.group(2)} — {reason}")
    if not r.next_steps:
        r.next_steps.append("No real breaks. Flaky set is the known xdist cases (feedback_mpc_tests_xdist_flaky).")
    return r


def parse_process_health(text: str, date: str) -> Report:
    r = Report(job="process-health", date=date, raw=text)
    r.gate = "INFO"
    entries = _first(r"\((\d+) entries", text)
    if entries:
        r.kpis.append(("Log entries", entries))
    heads = re.findall(r"^##\s+(.+)$", text, re.M)
    r.changes = [f"§ {h}" for h in heads[:8]]
    recs = _block_after(r"^##\s+RECOMMEND", text, stop_regex=r"^##\s")
    r.next_steps = _dedupe([ln.strip("-• ").strip() for ln in recs if ln.strip()][:8])
    return r


def load_report(job: str, reports_dir: Path = REPORTS, date: Optional[str] = None) -> Report:
    date = date or datetime.now().strftime("%Y-%m-%d")
    try:
        if job == "acc-sweep":
            return parse_acc_sweep(_read(reports_dir / "acc_sweep_report.txt"), date)
        if job == "sweep":
            return parse_sweep(_read(reports_dir / "sweep_report.txt"), date)
        if job == "nightly":
            return parse_nightly(_read(reports_dir / "nightly_test_report.txt"), _read(reports_dir / "nightly_status.txt"), date)
        if job == "process-health":
            files = sorted(reports_dir.glob("process_health_*.md"))
            return parse_process_health(_read(files[-1]) if files else "", date)
    except Exception as e:  # never lose the email over a parse bug
        r = Report(job=job, date=date)
        r.parse_error = f"{type(e).__name__}: {e}"
        return r
    return Report(job=job, date=date)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _badge(verdict: str) -> str:
    v = (verdict or "UNKNOWN").upper().split()[0]
    fg, bg = VERDICT_COLOURS.get(v, VERDICT_COLOURS["UNKNOWN"])
    return (f'<span style="display:inline-block;padding:2px 10px;border-radius:12px;font-weight:600;'
            f'font-size:12px;color:{fg};background:{bg};letter-spacing:.3px">{html.escape(verdict)}</span>')


def _table(columns: List[str], rows: List[List[str]], verdict_col: Optional[int]) -> str:
    if not rows:
        return '<p style="color:#5f6368;margin:6px 0">No table rows found in the report.</p>'
    th = "".join(f'<th style="text-align:left;padding:8px 10px;border-bottom:2px solid #dadce0;font-size:12px;'
                 f'color:#5f6368;text-transform:uppercase;letter-spacing:.4px">{html.escape(c)}</th>' for c in columns)
    trs = []
    for i, row in enumerate(rows):
        tds = []
        for j, cell in enumerate(row):
            content = _badge(cell) if (verdict_col is not None and j == verdict_col) else html.escape(cell)
            mono = ' font-family:SFMono-Regular,Menlo,Consolas,monospace;font-size:12px' if j not in (0, verdict_col) else ''
            tds.append(f'<td style="padding:8px 10px;border-bottom:1px solid #eceff1;vertical-align:top;{mono}">{content}</td>')
        bg = "#fafafa" if i % 2 else "#ffffff"
        trs.append(f'<tr style="background:{bg}">{"".join(tds)}</tr>')
    return f'<table cellpadding="0" cellspacing="0" style="border-collapse:collapse;width:100%">{th and "<tr>"+th+"</tr>"}{"".join(trs)}</table>'


def _section(title: str, body: str) -> str:
    return (f'<h2 style="font-size:15px;margin:26px 0 8px;color:#202124;border-left:4px solid #1a73e8;'
            f'padding-left:10px">{html.escape(title)}</h2>{body}')


def _pre(text: str, max_lines: int = 400) -> str:
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
        text = "\n".join(["… (truncated)"] + lines)
    return (f'<pre style="font-family:SFMono-Regular,Menlo,Consolas,monospace;font-size:11.5px;line-height:1.35;'
            f'background:#f8f9fa;border:1px solid #e0e0e0;border-radius:6px;padding:12px;overflow-x:auto;'
            f'white-space:pre-wrap;word-break:break-word;color:#3c4043">{html.escape(text)}</pre>')


def render_html(rep: Report, subject: str, log_tail: str, meta: Dict[str, str]) -> str:
    title = JOB_TITLES.get(rep.job, rep.job)
    kpis = "".join(
        f'<span style="display:inline-block;margin:0 8px 8px 0;padding:6px 10px;background:#f1f3f4;border-radius:8px;'
        f'font-size:12.5px;color:#3c4043"><b>{html.escape(k)}</b>&nbsp;{html.escape(v)}</span>'
        for k, v in rep.kpis + [(k, v) for k, v in meta.items() if v]
    )
    notes = "".join(f'<div style="margin:6px 0;padding:10px 12px;background:#fff8e1;border-left:4px solid #f9ab00;'
                    f'border-radius:4px;font-size:13px">{html.escape(n)}</div>' for n in rep.notes)
    if rep.parse_error:
        notes += (f'<div style="margin:6px 0;padding:10px 12px;background:#fce8e6;border-left:4px solid #b3261e;'
                  f'border-radius:4px;font-size:13px">Report parse failed ({html.escape(rep.parse_error)}) — raw report and log below.</div>')

    steps = "".join(f'<li style="margin:6px 0">{html.escape(s)}</li>' for s in rep.next_steps) or \
        '<li style="color:#5f6368">No explicit actions in the report.</li>'
    next_html = _section("Next steps", f'<ol style="margin:4px 0 0 18px;padding:0;font-size:14px;line-height:1.45">{steps}</ol>')

    results_html = _section("Results", _table(rep.columns, rep.rows, rep.verdict_col))

    changes_html = ""
    if rep.changes:
        changes_html = _section("Changes vs last night" if rep.job != "process-health" else "Sections",
                                _pre("\n".join(rep.changes), 60))

    cards_html = ""
    if rep.cards:
        cs = []
        for c in rep.cards:
            fg, bg = VERDICT_COLOURS.get(c.verdict or "UNKNOWN", VERDICT_COLOURS["UNKNOWN"])
            body = ""
            if c.root_cause:
                body += f'<div style="margin:6px 0"><b>Root cause.</b> {html.escape(c.root_cause)}</div>'
            if c.action:
                body += f'<div style="margin:6px 0"><b>Action.</b> {html.escape(c.action)}</div>'
            if c.lines:
                body += _pre("\n".join(c.lines), 30)
            cs.append(f'<div style="margin:10px 0;padding:12px 14px;border:1px solid #e0e0e0;border-left:5px solid {fg};'
                      f'border-radius:6px;background:#fff"><div style="font-weight:600;margin-bottom:4px">{html.escape(c.title)}</div>{body}</div>')
        cards_html = _section("Failures and warnings — detail", "".join(cs))

    appendix = _section("Raw report", _pre(rep.raw or "(report file missing)", 400))
    if log_tail.strip():
        appendix += _section("Log tail", _pre(log_tail, 80))

    fg, bg = VERDICT_COLOURS.get(rep.gate.upper(), VERDICT_COLOURS["UNKNOWN"])
    return f"""<!doctype html><html><body style="margin:0;padding:0;background:#eef1f4">
<div style="max-width:880px;margin:0 auto;padding:20px 16px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#202124">
  <div style="background:#ffffff;border-radius:10px;box-shadow:0 1px 3px rgba(0,0,0,.08);overflow:hidden">
    <div style="padding:18px 22px;background:#202124;color:#fff">
      <div style="font-size:12px;letter-spacing:.6px;text-transform:uppercase;color:#bdc1c6">AV stack · {html.escape(title)}</div>
      <div style="display:flex;align-items:center;justify-content:space-between;margin-top:6px">
        <div style="font-size:20px;font-weight:600">{html.escape(rep.date)}</div>
        <div style="padding:4px 14px;border-radius:14px;font-weight:700;font-size:13px;color:{fg};background:{bg}">GATE {html.escape(rep.gate)}</div>
      </div>
      <div style="font-size:12.5px;color:#bdc1c6;margin-top:6px">{html.escape(subject)}</div>
    </div>
    <div style="padding:16px 22px 24px">
      <div style="margin-bottom:6px">{kpis}</div>
      {notes}
      {next_html}
      {results_html}
      {changes_html}
      {cards_html}
      {appendix}
      <div style="margin-top:22px;font-size:11.5px;color:#80868b">Generated {html.escape(datetime.now().strftime('%Y-%m-%d %H:%M %Z').strip())} by tools/nightly/report_render.py · reports in data/reports/ · this email is also available as plain text.</div>
    </div>
  </div>
</div></body></html>"""


def render_text(rep: Report, subject: str, log_tail: str) -> str:
    out = [subject, "=" * len(subject), f"GATE: {rep.gate}", ""]
    if rep.kpis:
        out.append("  ".join(f"{k}={v}" for k, v in rep.kpis)); out.append("")
    if rep.next_steps:
        out.append("NEXT STEPS"); out += [f"  {i}. {s}" for i, s in enumerate(rep.next_steps, 1)]; out.append("")
    out.append("REPORT"); out.append(rep.raw or "(report file missing)"); out.append("")
    if log_tail.strip():
        out.append("LOG TAIL"); out.append(log_tail)
    return "\n".join(out)


def render(job: str, subject: str, log_tail: str = "", reports_dir: Path = REPORTS,
           meta: Optional[Dict[str, str]] = None, date: Optional[str] = None) -> Tuple[str, str]:
    rep = load_report(job, reports_dir, date)
    meta = dict(meta or {})
    if log_tail:
        pf = _first(r"preflight dt_p95:\s*(\d+ms[^\n]*)", log_tail)
        if pf:
            meta.setdefault("Preflight", pf)
        model = _first(r"model=([\w.-]+)", log_tail)
        if model:
            meta.setdefault("Model", model)
    return render_text(rep, subject, log_tail), render_html(rep, subject, log_tail, meta)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("job", choices=sorted(JOB_TITLES))
    ap.add_argument("--preview", help="write HTML here instead of printing text")
    ap.add_argument("--log", help="log file whose tail becomes the appendix")
    a = ap.parse_args()
    tail = ""
    if a.log and Path(a.log).exists():
        tail = "\n".join(Path(a.log).read_text(errors="replace").splitlines()[-100:])
    text, htm = render(a.job, f"preview: {a.job}", tail)
    if a.preview:
        Path(a.preview).write_text(htm)
        print(f"wrote {a.preview}")
    else:
        print(text)
    sys.exit(0)
