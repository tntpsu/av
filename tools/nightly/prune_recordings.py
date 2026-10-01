#!/usr/bin/env python3
"""Nightly recording retention (2026-10-01, after the disk hit 99 %).

Keeps:  every recording referenced anywhere in the repo (tests/fixtures, tests, tools,
        docs, .claude — goldens, ACC references, doc citations), every recording newer
        than --keep-days, the newest --keep-per-track per track_id, and anything written
        in the last 10 minutes. Everything else under data/recordings/*.h5 is deleted.
Writes: data/reports/prune/prune_<date>.json (manifest) and prints one summary line
        the nightly log/email can carry:
        RECORDINGS_PRUNE deleted=<n> freed_gb=<x> pool_gb=<y> pool_files=<n> free_gb=<z>

Usage:
    python3 tools/nightly/prune_recordings.py            # dry run (default)
    python3 tools/nightly/prune_recordings.py --apply
    python3 tools/nightly/prune_recordings.py --apply --keep-days 14 --keep-per-track 3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REC_DIR = REPO / "data" / "recordings"
MANIFEST_DIR = REPO / "data" / "reports" / "prune"
REF_GLOBS = ("tests/fixtures/*.json", "tests/*.py", "tools/**/*.py", "docs/**/*.md",
             ".claude/**/*.md", "CLAUDE.md", "config/**/*.yaml", "tracks/**/*.yml")
_NAME_RE = re.compile(r"recording_\d{8}_\d{6}\.h5")
MIN_AGE_S = 600.0  # never touch a file the recorder may still be writing


def referenced_recordings(repo: Path = REPO) -> set[str]:
    ref: set[str] = set()
    for pat in REF_GLOBS:
        for f in glob.glob(str(repo / pat), recursive=True):
            try:
                ref.update(_NAME_RE.findall(Path(f).read_text(errors="ignore")))
            except OSError:
                continue
    return ref


def track_id_of(path: Path) -> str:
    try:
        import h5py  # local import: keep the script usable without h5py for dry runs
        with h5py.File(path, "r") as f:
            if "metadata" in f.attrs:
                meta = json.loads(f.attrs["metadata"])
                return str(meta.get("recording_provenance", {}).get("track_id", "unknown"))
    except Exception:
        pass
    return "unknown"


def plan(rec_dir: Path = REC_DIR, *, keep_days: float, keep_per_track: int, now: float | None = None,
         refs: set[str] | None = None) -> dict:
    now = time.time() if now is None else now
    refs = referenced_recordings() if refs is None else refs
    files = sorted(rec_dir.glob("recording_*.h5"), key=os.path.getmtime)
    newest_per_track: dict[str, list[Path]] = {}
    for p in reversed(files):
        tid = track_id_of(p)
        bucket = newest_per_track.setdefault(tid, [])
        if len(bucket) < keep_per_track:
            bucket.append(p)
    protected_newest = {p.name for b in newest_per_track.values() for p in b}
    delete, keep = [], []
    for p in files:
        age_d = (now - os.path.getmtime(p)) / 86400.0
        reason = None
        if p.name in refs:
            reason = "referenced"
        elif (now - os.path.getmtime(p)) < MIN_AGE_S:
            reason = "being_written"
        elif age_d <= keep_days:
            reason = "recent"
        elif p.name in protected_newest:
            reason = "newest_per_track"
        entry = {"file": p.name, "bytes": p.stat().st_size, "age_days": round(age_d, 1)}
        (keep if reason else delete).append({**entry, "reason": reason} if reason else entry)
    return {"delete": delete, "keep": keep}


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="delete (default: dry run)")
    ap.add_argument("--keep-days", type=float, default=14.0)
    ap.add_argument("--keep-per-track", type=int, default=3)
    ap.add_argument("--recordings-dir", default=str(REC_DIR))
    args = ap.parse_args(argv[1:])
    rec_dir = Path(args.recordings_dir)
    result = plan(rec_dir, keep_days=args.keep_days, keep_per_track=args.keep_per_track)
    freed = 0
    for e in result["delete"]:
        if args.apply:
            try:
                (rec_dir / e["file"]).unlink()
                freed += e["bytes"]
            except OSError as exc:
                e["error"] = str(exc)
        else:
            freed += e["bytes"]
    pool = [p.stat().st_size for p in rec_dir.glob("*.h5")]
    free_gb = shutil.disk_usage(rec_dir).free / 2**30
    date = time.strftime("%Y-%m-%d")
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest = MANIFEST_DIR / f"prune_{date}.json"
    manifest.write_text(json.dumps({"date": date, "applied": args.apply, "keep_days": args.keep_days,
                                    "keep_per_track": args.keep_per_track, **result}, indent=1) + "\n")
    mode = "deleted" if args.apply else "would_delete"
    print(f"RECORDINGS_PRUNE {mode}={len(result['delete'])} freed_gb={freed / 2**30:.1f} "
          f"pool_gb={sum(pool) / 2**30:.1f} pool_files={len(pool)} free_gb={free_gb:.1f} manifest={_rel(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
