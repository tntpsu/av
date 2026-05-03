"""Dashboard data parsers for the PhilViz mobile-friendly results views.

Reads the same heartbeat/report files the nightly job wrappers use to
compose email subjects (`tools/nightly/{run.sh, sweep/run.sh, acc-sweep/run.sh}`)
and returns structured JSON for the /dashboards page.

Data sources (all in `data/reports/`):
  - sweep_status.txt           — per-track lines from the lateral sweep
  - sweep_report.txt           — full prose report
  - acc_sweep_status.txt       — per-scenario verdict lines
  - acc_sweep_report.txt       — full prose report
  - nightly_status.txt         — fix-tests heartbeat
  - nightly_test_report.txt    — fix-tests structured report

These files are produced by the nightly automation and persisted on disk;
this module is read-only.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[3]
REPORTS = REPO / "data" / "reports"


def _file_stat(path: Path) -> dict:
    if not path.is_file():
        return {"exists": False, "mtime": None, "size": 0}
    st = path.stat()
    return {"exists": True, "mtime": st.st_mtime, "size": st.st_size}


def _read_text(path: Path, max_bytes: int = 200_000) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[:max_bytes]


def parse_sweep_status() -> dict:
    """Parse data/reports/sweep_status.txt — lateral sweep heartbeat.

    Per-track lines look like:
      step_track_s_loop_done score=99.0 baseline=99.1 delta=-0.1 2026-05-02T03:02:44-04:00
      step_track_hairpin_15_done score=98.6 baseline=98.7 delta=-0.1 FLAG=trajectory_94.9 ...
    """
    hb = REPORTS / "sweep_status.txt"
    text = _read_text(hb)
    tracks: list[dict] = []
    started_at: Optional[str] = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("step0_started"):
            m = re.search(r"(\d{4}-\d{2}-\d{2}T[\d:.\-+Z]+)", line)
            if m:
                started_at = m.group(1)
            continue
        m = re.match(r"^step_track_(?P<name>[a-zA-Z0-9_]+)_done\s+(?P<rest>.+)$", line)
        if not m:
            continue
        name = m.group("name")
        rest = m.group("rest")
        score = _extract_float(rest, r"score=(-?[0-9.]+)")
        baseline = _extract_float(rest, r"baseline=(-?[0-9.]+)")
        delta = _extract_float(rest, r"delta=(-?[0-9.]+)")
        flag_m = re.search(r"FLAG=([^\s]+)", rest)
        flag = flag_m.group(1) if flag_m else None
        tracks.append({
            "name": name,
            "score": score,
            "baseline": baseline,
            "delta": delta,
            "flag": flag,
        })
    # Enrich each track with its layer scores parsed from sweep_report.txt
    layer_scores = _parse_lateral_layer_scores()
    for t in tracks:
        t["layers"] = layer_scores.get(t["name"], {})
    regressions = sum(1 for t in tracks if t["delta"] is not None and t["delta"] < -2.0)
    flags = sum(1 for t in tracks if t["flag"])
    gate = "FAIL" if regressions > 0 else ("FLAG" if flags > 0 else "PASS")
    worst = min(
        (t for t in tracks if t["delta"] is not None),
        key=lambda t: t["delta"],
        default=None,
    )
    return {
        "kind": "sweep",
        "started_at": started_at,
        "tracks": tracks,
        "summary": {
            "gate": gate,
            "passed": len(tracks),
            "regressions": regressions,
            "flags": flags,
            "worst_track": worst["name"] if worst else None,
            "worst_delta": worst["delta"] if worst else None,
        },
        "file": _file_stat(hb),
    }


def _parse_lateral_layer_scores() -> dict:
    """Extract per-track layer scores from sweep_report.txt.

    Looks for the 'Layer scores summary' table:
        Track            Safety  Trajectory  Control  Perception  LongComfort  SigInt
        s_loop           100.0   96.5        100.0    100.0       100.0        100.0
    Returns: {track_name: {layer: score, ...}, ...}
    """
    text = _read_text(REPORTS / "sweep_report.txt")
    out: dict = {}
    in_table = False
    headers: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if "Layer scores summary" in line:
            in_table = True
            continue
        if not in_table:
            continue
        # Header row defines column names
        if not headers and line.lstrip().startswith("Track"):
            headers = line.split()
            continue
        # Blank line ends the table
        if not line.strip():
            if headers:
                break
            continue
        # Data row
        parts = line.split()
        if len(parts) < 2:
            continue
        track = parts[0]
        # Map remaining columns to header names (skip 'Track' header column)
        layer_cols = headers[1:] if len(headers) > 1 else []
        for i, col in enumerate(layer_cols, start=1):
            if i < len(parts):
                try:
                    out.setdefault(track, {})[col] = float(parts[i])
                except ValueError:
                    pass
    return out


def parse_acc_sweep_status() -> dict:
    """Parse data/reports/acc_sweep_status.txt — ACC scenario verdicts.

    Per-scenario lines:
      scenario_h2_done verdict=PASS reason="TTC 2.4s OK" 2026-05-02T...
      scenario_g2_done verdict=FAIL reason="TTC dropped to 1.7s" ...
      scenario_h5_done verdict=SKIPPED reason="no recording newer than 7d" ...
    """
    hb = REPORTS / "acc_sweep_status.txt"
    text = _read_text(hb)
    scenarios: list[dict] = []
    started_at: Optional[str] = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("step0_started"):
            m = re.search(r"(\d{4}-\d{2}-\d{2}T[\d:.\-+Z]+)", line)
            if m:
                started_at = m.group(1)
            continue
        m = re.match(r"^scenario_(?P<name>[a-zA-Z0-9_]+)_done\s+(?P<rest>.+)$", line)
        if not m:
            continue
        name = m.group("name")
        rest = m.group("rest")
        verdict_m = re.search(r"verdict=([A-Z_]+)", rest)
        verdict = verdict_m.group(1) if verdict_m else "UNKNOWN"
        reason_m = re.search(r'reason="([^"]+)"', rest)
        reason = reason_m.group(1) if reason_m else ""
        scenarios.append({"name": name, "verdict": verdict, "reason": reason})
    # Enrich each scenario with its multi-line detail block from acc_sweep_report.txt
    details = _parse_acc_scenario_details()
    for s in scenarios:
        # Match by short name (scenario_A1_done → "A1") or by full name
        key_short = s["name"]
        # The report lists by short codename like "A1" — try both forms
        s["detail"] = details.get(key_short) or details.get(_short_acc_code(key_short)) or ""
    counts = {v: 0 for v in ("PASS", "FAIL", "WARN", "SKIPPED", "AMBIGUOUS", "UNKNOWN")}
    for s in scenarios:
        counts[s["verdict"]] = counts.get(s["verdict"], 0) + 1
    gate = "FAIL" if counts.get("FAIL", 0) > 0 else "PASS"
    return {
        "kind": "acc-sweep",
        "started_at": started_at,
        "scenarios": scenarios,
        "summary": {
            "gate": gate,
            "total": len(scenarios),
            **counts,
        },
        "file": _file_stat(hb),
    }


def _short_acc_code(name: str) -> str:
    """Extract short scenario code from longer names (e.g. 'autobahn_a1_steady' → 'A1')."""
    m = re.search(r"(?:^|_)([aAhHgG])(\d+)(?:_|$)", name)
    if m:
        return f"{m.group(1).upper()}{m.group(2)}"
    return name.upper()


def _parse_acc_scenario_details() -> dict:
    """Extract per-scenario multi-line detail blocks from acc_sweep_report.txt.

    Looks for sections like:
        Failures (1):
          A1 (autobahn_a1_steady) — recording_20260502_163502.h5
            <indented multi-line explanation>
        Warnings (1):
          A2 (autobahn_a2_hard_brake) — recording_...
            <multi-line>
        Skipped (12):
          H2, H3, H4 — base: highway_65
            ...

    Returns: {short_code: detail_text, ...}
    """
    text = _read_text(REPORTS / "acc_sweep_report.txt")
    out: dict = {}
    current_codes: list[str] = []
    current_lines: list[str] = []
    in_section = False
    section_re = re.compile(r"^(Failures|Warnings|Skipped|Ambiguous)\s*\(\d+\)\s*:")
    # Header line for a scenario block. Two shapes:
    #   "  A1 (autobahn_a1_steady) — recording_20260502_163502.h5"
    #   "  H2, H3, H4 — base: highway_65"  (grouped skip)
    block_re = re.compile(r"^\s{2}(?P<codes>[A-Z]\d+(?:,\s*[A-Z]\d+)*)\s*(\(|—)")

    def _flush():
        if current_codes and current_lines:
            joined = "\n".join(current_lines).rstrip()
            for code in current_codes:
                out[code] = joined

    for raw in text.splitlines():
        line = raw.rstrip()
        if section_re.match(line.strip()):
            _flush()
            current_codes = []
            current_lines = []
            in_section = True
            continue
        if not in_section:
            continue
        m = block_re.match(line)
        if m:
            _flush()
            current_codes = [c.strip() for c in m.group("codes").split(",")]
            current_lines = [line.lstrip()]
            continue
        # Continuation — indented body of the current block
        if current_codes and (line.startswith("    ") or line.startswith("\t")):
            current_lines.append(line.lstrip())
            continue
        if current_codes and line.strip() == "":
            current_lines.append("")
            continue
        # Anything else ends the current block
        _flush()
        current_codes = []
        current_lines = []
    _flush()
    return out


def parse_fix_tests_status() -> dict:
    """Parse data/reports/{nightly_status.txt, nightly_test_report.txt}."""
    hb = REPORTS / "nightly_status.txt"
    report = REPORTS / "nightly_test_report.txt"
    hb_text = _read_text(hb)
    rep_text = _read_text(report)

    started_at: Optional[str] = None
    delivery: Optional[str] = None
    for raw in hb_text.splitlines():
        line = raw.strip()
        if line.startswith("step0_started"):
            m = re.search(r"(\d{4}-\d{2}-\d{2}T[\d:.\-+Z]+)", line)
            if m:
                started_at = m.group(1)
        if line.startswith("step6_done delivery="):
            m = re.search(r"delivery=([a-z_]+)", line)
            if m:
                delivery = m.group(1)

    fixed = real_breaks = flaky = total = passed_n = None
    flaky_list: list[dict] = []
    for raw in rep_text.splitlines():
        if raw.startswith("Total:"):
            total = _extract_int(raw, r"Total:\s*(\d+)")
        elif raw.startswith("Passed:"):
            passed_n = _extract_int(raw, r"Passed:\s*(\d+)")
        elif raw.startswith("Fixed:"):
            fixed = _extract_int(raw, r"Fixed:\s*(\d+)")
        elif raw.startswith("Real breaks"):
            real_breaks = _extract_int(raw, r":\s*(\d+)")
        elif raw.startswith("Flaky:"):
            flaky = _extract_int(raw, r"Flaky:\s*(\d+)")
        elif raw.lstrip().startswith("[tests/") and "::" in raw:
            flaky_list.append({"test": raw.strip().lstrip("[").rstrip("]")})

    return {
        "kind": "fix-tests",
        "started_at": started_at,
        "summary": {
            "total": total,
            "passed": passed_n,
            "fixed": fixed,
            "real_breaks": real_breaks,
            "flaky": flaky,
            "delivery": delivery,
        },
        "flaky_list": flaky_list[:20],
        "file_status": _file_stat(hb),
        "file_report": _file_stat(report),
    }


def list_tracks_with_metadata() -> dict:
    """List tracks/scenarios with their latest recording metadata.

    Used by the standalone /tracks picker page. For each track, finds the
    most recent .h5 in data/recordings/ that *might* match (best-effort —
    matches by track_id substring in the filename or just returns the
    most recent overall as a fallback).
    """
    tracks_dir = REPO / "tracks"
    rec_dir = REPO / "data" / "recordings"

    base = sorted(p.stem for p in tracks_dir.glob("*.yml")) if tracks_dir.is_dir() else []
    scen_dir = tracks_dir / "scenarios"
    scenarios = sorted(p.stem for p in scen_dir.glob("*.yml")) if scen_dir.is_dir() else []

    # Index recordings by (mtime desc) so we can return most recent first
    recordings = []
    if rec_dir.is_dir():
        for p in rec_dir.glob("*.h5"):
            try:
                recordings.append({"name": p.name, "mtime": p.stat().st_mtime, "size": p.stat().st_size})
            except OSError:
                pass
        recordings.sort(key=lambda r: r["mtime"], reverse=True)

    def _latest_for(track_name: str) -> Optional[dict]:
        # Heuristic: if any recording filename mentions the track name, prefer it.
        # Otherwise just report None — the user can pick a recording explicitly later.
        for r in recordings:
            if track_name in r["name"]:
                return r
        return None

    return {
        "tracks": [{"name": t, "kind": "base", "latest_recording": _latest_for(t)} for t in base],
        "scenarios": [{"name": t, "kind": "scenario", "latest_recording": _latest_for(t)} for t in scenarios],
    }


# ---- helpers ----

def _extract_float(text: str, pattern: str) -> Optional[float]:
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except (ValueError, IndexError):
        return None


def _extract_int(text: str, pattern: str) -> Optional[int]:
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except (ValueError, IndexError):
        return None
