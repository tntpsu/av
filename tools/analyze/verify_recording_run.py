#!/usr/bin/env python3
"""
Verify that a recording's observed cadence/queue match the stack params embedded at record time.

Reads:
  - HDF5 attrs ``metadata`` JSON: ``stack_transport`` (written by AVStack when recording)
  - ``meta/runtime_config_json``: ``stack`` subset (if present)
  - ``control/timestamps`` or e2e mono: mean control rate
  - ``vehicle/stream_front_queue_depth``: max/p95 vs expected bridge cap

Usage:
  python tools/analyze/verify_recording_run.py path/to/recording.h5
  python tools/analyze/verify_recording_run.py --latest
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORDINGS = REPO_ROOT / "data" / "recordings"


def _load_metadata(f: h5py.File) -> dict[str, Any]:
    raw = f.attrs.get("metadata")
    if raw is None:
        return {}
    try:
        s = raw.decode("utf-8", "ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
        return json.loads(s) if s.strip() else {}
    except json.JSONDecodeError:
        return {}


def _finite_stats(a: np.ndarray) -> tuple[Optional[float], Optional[float], Optional[float]]:
    x = np.asarray(a, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None, None, None
    return float(np.mean(x)), float(np.percentile(x, 95)), float(np.max(x))


def verify(path: Path) -> int:
    if not path.is_file():
        print(f"ERROR: not found: {path}", file=sys.stderr)
        return 2

    with h5py.File(path, "r") as f:
        meta = _load_metadata(f)
        stack_tr = meta.get("stack_transport")
        if isinstance(stack_tr, dict):
            print("=== Embedded stack_transport (Python at record start) ===")
            for k in sorted(stack_tr.keys()):
                print(f"  {k}: {stack_tr[k]}")
        else:
            print("=== stack_transport: MISSING (recording older than this feature) ===")

        rcj = None
        if "meta/runtime_config_json" in f:
            try:
                raw = f["meta/runtime_config_json"][0]
                s = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                rcj = json.loads(s)
            except Exception as exc:
                print(f"=== meta/runtime_config_json: parse error ({exc}) ===")
        if isinstance(rcj, dict) and "stack" in rcj:
            print("=== meta/runtime_config_json stack ===")
            for k, v in sorted((rcj.get("stack") or {}).items()):
                if v is not None:
                    print(f"  {k}: {v}")

        ctrl_ts = None
        if "control/timestamps" in f:
            ctrl_ts = np.asarray(f["control/timestamps"][:], dtype=np.float64)
        sent = None
        if "control/e2e_control_sent_mono_s" in f:
            sent = np.asarray(f["control/e2e_control_sent_mono_s"][:], dtype=np.float64)

        wall_ms: list[float] = []
        if sent is not None and sent.size > 1:
            for i in range(1, min(len(sent), 50000)):
                if np.isfinite(sent[i]) and np.isfinite(sent[i - 1]):
                    wall_ms.append((sent[i] - sent[i - 1]) * 1000.0)
        elif ctrl_ts is not None and ctrl_ts.size > 1:
            for i in range(1, min(len(ctrl_ts), 50000)):
                if np.isfinite(ctrl_ts[i]) and np.isfinite(ctrl_ts[i - 1]):
                    wall_ms.append((ctrl_ts[i] - ctrl_ts[i - 1]) * 1000.0)

        mean_hz: Optional[float] = None
        if wall_ms:
            arr = np.array(wall_ms, dtype=np.float64)
            mean_ms = float(np.mean(arr))
            p95_ms = float(np.percentile(arr, 95))
            mean_hz = 1000.0 / mean_ms if mean_ms > 1e-6 else None
            print("\n=== Observed control loop (wall period) ===")
            print(f"  samples: {len(wall_ms)}")
            print(f"  mean_period_ms: {mean_ms:.2f}  -> mean_hz: {mean_hz:.2f}" if mean_hz else "")
            print(f"  p95_period_ms: {p95_ms:.2f}")

        qd = None
        if "vehicle/stream_front_queue_depth" in f:
            qd = np.asarray(f["vehicle/stream_front_queue_depth"][:], dtype=np.float64)
        if qd is not None and qd.size:
            qm, qp95, qmax = _finite_stats(qd)
            print("\n=== stream_front_queue_depth ===")
            print(f"  mean: {qm:.2f}  p95: {qp95:.2f}  max: {qmax:.2f}")

            cap = None
            if isinstance(stack_tr, dict):
                env_cap = stack_tr.get("AV_BRIDGE_MAX_CAMERA_QUEUE")
                yaml_cap = stack_tr.get("bridge_max_camera_queue_yaml")
                if env_cap is not None and str(env_cap).strip().isdigit():
                    cap = int(env_cap)
                elif yaml_cap is not None:
                    try:
                        cap = int(yaml_cap)
                    except (TypeError, ValueError):
                        pass
            if cap is not None and qmax is not None:
                headroom = cap - qmax
                if headroom < 0:
                    print(f"  CHECK: max queue {qmax:.0f} > cap {cap} (unexpected)")
                elif headroom == 0 and qp95 is not None and qp95 >= cap - 0.5:
                    print(f"  NOTE: queue saturated at cap {cap} (consumer not faster than producer)")
                else:
                    print(f"  OK: headroom vs cap {cap} (max {qmax:.0f}, not pinned)")
            elif cap is None and qmax is not None and qp95 is not None and qmax == qp95:
                print(
                    "  NOTE: queue max == p95 (flat top) — likely pinned at bridge cap; "
                    "re-run with new recordings to embed stack_transport, or check AV_BRIDGE_MAX_CAMERA_QUEUE."
                )

        target = None
        if isinstance(stack_tr, dict) and stack_tr.get("target_loop_hz") is not None:
            try:
                target = float(stack_tr["target_loop_hz"])
            except (TypeError, ValueError):
                pass
        if target is not None and wall_ms and mean_hz is not None:
            print("\n=== vs target_loop_hz ===")
            print(f"  target_hz: {target:.2f}  observed_mean_hz: {mean_hz:.2f}")
            if mean_hz + 0.5 < target:
                print(
                    f"  WARN: mean loop slower than target (~{target - mean_hz:.2f} Hz short). "
                    "Lower Unity targetFPS + target_loop_hz or reduce per-frame work."
                )
            elif mean_hz > target + 1.0:
                print("  NOTE: mean loop faster than target (sleep/pacing may be limiting upper bound)")

    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Verify recording cadence vs embedded stack params.")
    p.add_argument("recording", nargs="?", help="Path to .h5")
    p.add_argument("--latest", action="store_true", help="Use newest recording in data/recordings")
    args = p.parse_args()

    if args.latest:
        cands = sorted(RECORDINGS.glob("recording_*.h5"), key=lambda x: x.stat().st_mtime)
        path = cands[-1] if cands else None
    elif args.recording:
        path = Path(args.recording)
    else:
        cands = sorted(RECORDINGS.glob("recording_*.h5"), key=lambda x: x.stat().st_mtime)
        path = cands[-1] if cands else None

    if path is None:
        raise SystemExit("No recording path; pass a .h5 or use --latest")
    print(f"Recording: {path.resolve()}\n")
    raise SystemExit(verify(path))


if __name__ == "__main__":
    main()
