#!/usr/bin/env python3
"""Track scoreboard — extract and display scores from latest recordings.

Usage:
    python tools/analyze/scoreboard.py              # latest per track
    python tools/analyze/scoreboard.py s_loop       # specific track
    python tools/analyze/scoreboard.py 3            # 3 most recent
    python tools/analyze/scoreboard.py path/to.h5   # specific file
    python tools/analyze/scoreboard.py --json       # machine-readable output
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.drive_summary_core import analyze_recording_summary


def extract_scores(recordings_dir: Path, filter_track: str = None,
                   limit: int = None, specific_file: Path = None) -> list:
    """Extract scores from recordings, one per track unless filtered."""
    if specific_file:
        recs = [specific_file]
    else:
        recs = sorted(recordings_dir.glob("*.h5"),
                      key=lambda p: p.stat().st_mtime, reverse=True)

    seen_tracks = set()
    results = []
    for rec in recs[:50]:
        try:
            summary = analyze_recording_summary(rec, analyze_to_failure=True)
            track_id = (summary.get("run_intent") or {}).get("track_id") or "unknown"

            if track_id == "unknown":
                continue  # skip crashed/empty recordings
            if filter_track and track_id != filter_track:
                continue
            if not specific_file and not limit and track_id in seen_tracks:
                continue
            seen_tracks.add(track_id)

            ls = summary.get("layer_scores", {})
            c = summary.get("comfort", {})
            es = summary.get("executive_summary", {})

            # Oscillation growth deduction from Control layer
            ctrl_bd = summary.get("layer_score_breakdown", {}).get("Control", {})
            osc_ded = 0.0
            for d in ctrl_bd.get("deductions", []):
                if "Oscillation Growth" in d.get("name", ""):
                    osc_ded = d.get("value", 0.0)

            # Collect non-zero deductions per layer
            deductions = {}
            for layer, bd in summary.get("layer_score_breakdown", {}).items():
                for dd in bd.get("deductions", []):
                    if dd["value"] > 0.5:
                        deductions.setdefault(layer, []).append(
                            {"name": dd["name"], "value": round(dd["value"], 1)}
                        )

            results.append({
                "file": rec.name,
                "track": track_id,
                "score": round(es.get("overall_score", 0), 1),
                "cap": es.get("score_breakdown", {}).get("cap_reason", "none"),
                "duration_s": round(es.get("drive_duration", 0), 1),
                "frames": int(es.get("total_frames", 0)),
                "layers": {k: round(v, 1) for k, v in ls.items()},
                "accel_p95": round(c.get("acceleration_p95_filtered", 0), 3),
                "jerk_p95": round(c.get("commanded_jerk_p95", 0), 3),
                "steer_jerk": round(c.get("steering_jerk_max", 0), 1),
                "lat_rmse": round(summary.get("path_tracking", {}).get(
                    "lateral_error_rmse", 0), 3),
                "centered_pct": round(summary.get("path_tracking", {}).get(
                    "time_in_lane_centered", 0), 1),
                "osc_ded": round(osc_ded, 1),
                "estops": int(c.get("emergency_stops", 0)),
                "deductions": deductions,
                "issues": es.get("key_issues", [])[:3],
            })

            if limit and len(results) >= limit:
                break
        except Exception:
            pass

    results.sort(key=lambda r: r["score"])
    return results


def _check(val, gate): return "✓" if val <= gate else "✗"
def _check_min(val, gate): return "✓" if val >= gate else "✗"
def _layer_mark(score): return "✓" if score >= 95 else "✗"


def _load_baselines() -> dict:
    """Load frozen baselines from scoring_baselines.json."""
    bl_path = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "scoring_baselines.json"
    if bl_path.exists():
        with open(bl_path) as f:
            return json.load(f).get("tracks", {})
    return {}


def _delta_str(val, baseline, higher_is_better=True, fmt=".1f"):
    """Format a delta string like '+2.1' or '-0.3' with direction indicator."""
    delta = val - baseline
    if abs(delta) < 0.05:
        return ""
    sign = "+" if delta > 0 else ""
    # Green arrow if improving, red if regressing
    good = (delta > 0) if higher_is_better else (delta < 0)
    arrow = "▲" if good else "▼"
    return f" {arrow}{sign}{delta:{fmt}}"


def print_scoreboard(results: list):
    """Print human-readable scoreboard."""
    if not results:
        print("No recordings found. Run the stack first:")
        print("  ./start_av_stack.sh --duration 60 --track-yaml tracks/s_loop.yml")
        return

    baselines = _load_baselines()

    print("TRACK SCOREBOARD")
    print("════════════════════════════════════════════════════════════════════")

    all_comfort_clear = 0
    all_layers_pass = 0
    total_estops = 0
    # For Pareto
    all_deductions = {}

    for r in results:
        cap_str = f"  (capped: {r['cap']})" if r["cap"] != "none" else ""
        status = "PASS" if r["score"] >= 95 else "NEEDS WORK"
        bl = baselines.get(r["track"], {})

        print(f"\nTrack: {r['track']}  │  Recording: {r['file']}")
        score_delta = _delta_str(r["score"], bl.get("overall_score", r["score"])) if bl else ""
        print(f"Overall: {r['score']}/100  {status}{score_delta}{cap_str}")
        rmse_delta = _delta_str(r["lat_rmse"], bl.get("lateral_error_adj_rmse", r["lat_rmse"]),
                                higher_is_better=False, fmt=".3f") if bl else ""
        print(f"Drive: {r['duration_s']}s  │  {r['frames']} frames  │  "
              f"RMSE {r['lat_rmse']}m{rmse_delta}  │  Centered {r['centered_pct']}%")

        # Layer scores (exclude LongitudinalComfort — shown in Comfort section)
        print("\n  Layer Scores:")
        layer_order = ["Safety", "Perception", "Trajectory", "Control", "SignalIntegrity"]
        track_all_pass = True
        for layer in layer_order:
            s = r["layers"].get(layer, 0)
            mark = _layer_mark(s)
            if s < 95:
                track_all_pass = False
            print(f"    {layer:<20s} {s:5.1f}  {mark}")

        if track_all_pass:
            all_layers_pass += 1

        # Unified Comfort
        accel_ok = r["accel_p95"] <= 3.0
        jerk_ok = r["jerk_p95"] <= 6.0
        steer_ok = r["steer_jerk"] <= 18.0
        comfort_clear = accel_ok and jerk_ok and steer_ok
        if comfort_clear:
            all_comfort_clear += 1

        print("\n  Comfort (unified):")
        print(f"    Longitudinal:  accel_p95 {r['accel_p95']:.2f} m/s² {'✓' if accel_ok else '✗'}"
              f"  │  jerk_p95 {r['jerk_p95']:.2f} m/s³ {'✓' if jerk_ok else '✗'}")
        osc_str = f"  │  osc_growth -{r['osc_ded']}" if r["osc_ded"] > 0 else ""
        print(f"    Lateral:       steer_jerk {r['steer_jerk']:.1f} {'✓' if steer_ok else '✗'}"
              f"{osc_str}")

        # Top deductions
        has_deductions = False
        for layer in layer_order:
            deds = r["deductions"].get(layer, [])
            if deds:
                if not has_deductions:
                    print("\n  Top Deductions:")
                    has_deductions = True
                parts = ", ".join(f"{d['name']} -{d['value']}" for d in deds[:2])
                print(f"    {layer}: {parts}")

        # Also check LongitudinalComfort deductions
        lc_deds = r["deductions"].get("LongitudinalComfort", [])
        if lc_deds:
            if not has_deductions:
                print("\n  Top Deductions:")
            parts = ", ".join(f"{d['name']} -{d['value']}" for d in lc_deds[:2])
            print(f"    LongitudinalComfort: {parts}")

        # Issues
        if r["issues"]:
            print(f"\n  Issues: {'; '.join(r['issues'][:3])}")

        total_estops += r["estops"]

        # Aggregate deductions for Pareto
        for layer, deds in r["deductions"].items():
            for d in deds:
                key = d["name"]
                all_deductions.setdefault(key, {"total": 0, "tracks": set()})
                all_deductions[key]["total"] += d["value"]
                all_deductions[key]["tracks"].add(r["track"])

        print("────────────────────────────────────────────────────────────────────")

    # ── Compact comparison table ──────────────────────────────────────────
    n = len(results)
    # Sort by score descending for the table
    table_results = sorted(results, key=lambda r: -r["score"])

    print("\n\nSCORE TABLE")
    print("═══════════════════════════════════════════════════════════════════════════════════════════════════════")
    print(f"{'Track':<18s} {'Score':>5s} {'Δbl':>6s}  {'Time':>5s}  {'RMSE':>5s} {'Δbl':>7s}  {'Ctr%':>4s}  "
          f"{'Safe':>4s}  {'Perc':>4s}  {'Traj':>4s}  {'Ctrl':>4s}  {'SigI':>4s}  "
          f"{'StJrk':>5s}  {'E-st':>4s}")
    print("─" * 105)
    for r in table_results:
        ls = r["layers"]
        bl = baselines.get(r["track"], {})
        gate = "✓" if r["score"] >= 95 else "✗"

        # Score delta
        if bl and "overall_score" in bl:
            sd = r["score"] - bl["overall_score"]
            score_d = f"{sd:+.1f}" if abs(sd) >= 0.05 else "  —"
        else:
            score_d = "  —"

        # RMSE delta (lower is better)
        if bl and "lateral_error_adj_rmse" in bl:
            rd = r["lat_rmse"] - bl["lateral_error_adj_rmse"]
            rmse_d = f"{rd:+.3f}" if abs(rd) >= 0.001 else "   —"
        else:
            rmse_d = "   —"

        print(f"{r['track']:<18s} {r['score']:5.1f}{gate} {score_d:>6s} "
              f"{r['duration_s']:5.0f}s {r['lat_rmse']:5.3f} {rmse_d:>7s} {r['centered_pct']:4.0f}%  "
              f"{ls.get('Safety', 0):4.0f}  {ls.get('Perception', 0):4.0f}  "
              f"{ls.get('Trajectory', 0):4.0f}  {ls.get('Control', 0):4.0f}  "
              f"{ls.get('SignalIntegrity', 0):4.0f}  "
              f"{r['steer_jerk']:5.1f}  {r['estops']:4d}")
    print("─" * 105)
    print(f"{'SUMMARY':<18s}       "
          f"{all_layers_pass}/{n} all layers ≥ 95  │  "
          f"{sum(1 for r in results if r['score'] >= 95)}/{n} overall ≥ 95  │  "
          f"Comfort: {all_comfort_clear}/{n}  │  E-stops: {total_estops}")

    # Pareto top 3
    if all_deductions:
        ranked = sorted(all_deductions.items(), key=lambda x: -x[1]["total"])[:3]
        print(f"\nTOP 3 HIGHEST-ROI FIXES")
        print("════════════════════════════════════════")
        for i, (name, info) in enumerate(ranked, 1):
            tracks = ", ".join(sorted(info["tracks"]))
            print(f"{i}. {name} — {info['total']:.1f} pts across"
                  f" {len(info['tracks'])} tracks ({tracks})")
        print("\nRun /pareto for full cross-track analysis.")


def main():
    parser = argparse.ArgumentParser(description="Track scoreboard")
    parser.add_argument("filter", nargs="?", default=None,
                        help="Track name, recording path, or count")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--recordings-dir", default="data/recordings",
                        help="Recordings directory")
    args = parser.parse_args()

    rec_dir = Path(args.recordings_dir)
    filter_track = None
    limit = None
    specific_file = None

    if args.filter:
        p = Path(args.filter)
        if p.suffix == ".h5" and p.exists():
            specific_file = p
        elif args.filter.isdigit():
            limit = int(args.filter)
        else:
            # Normalize track name
            filter_track = args.filter.replace("-", "_").replace(" ", "_")

    results = extract_scores(rec_dir, filter_track, limit, specific_file)

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        print_scoreboard(results)


if __name__ == "__main__":
    main()
