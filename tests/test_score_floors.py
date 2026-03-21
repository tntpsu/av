"""Score floor registry tests.

These tests assert that the overall score and key layer scores from the
latest recording of each track stay above the floors committed in
tests/scores.json.

WHY THIS MATTERS
----------------
Without score floor tests, a config change that fixes one bug can silently
regress another layer and only reveal itself after a Unity run.  By
committing floors after each validated run, any regression is caught
locally before the next drive.

HOW TO USE
----------
1. After a good Unity run, check the score breakdown in the analyzer.
2. Update tests/scores.json with the new floors (raise, never lower):

       python tools/update_score_floor.py --track sloop --score 92.5

3. Run pytest locally to confirm the floors hold on the latest recording.

SKIPPING IN CI
--------------
These tests require HDF5 recording files which are not committed to git.
They automatically skip when no recording is found for a track, so CI
passes without recordings.

TRACK DETECTION
---------------
The test finds the latest recording matching a track by scanning
data/recordings/*.h5 and reading the 'track_name' attribute (or falling
back to filename heuristics).
"""

import json
import sys
import pytest
from pathlib import Path

_ROOT      = Path(__file__).resolve().parent.parent
_FLOORS    = _ROOT / "tests" / "scores.json"
_REC_DIR   = _ROOT / "data" / "recordings"

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_floors() -> dict:
    with open(_FLOORS) as f:
        return json.load(f)["floors"]


def _find_latest_recording(track: str) -> Path | None:
    """Return the most recently modified .h5 recording that is likely for *track*."""
    if not _REC_DIR.exists():
        return None
    candidates = []
    for p in _REC_DIR.glob("recording_*.h5"):
        # Quick heuristic: check if track name appears in the first 4 KB of
        # the file's string attributes (without loading the whole file)
        try:
            import h5py
            with h5py.File(p, "r") as f:
                # Prefer an explicit track_name attribute
                if "track_name" in f.attrs:
                    if track.lower() in str(f.attrs["track_name"]).lower():
                        candidates.append(p)
                    continue
                # Fall back: look for track keywords in top-level group names
                all_keys = " ".join(f.keys()).lower()
                if track.lower().replace("_", "") in all_keys.replace("_", ""):
                    candidates.append(p)
                    continue
                # Last resort: filename heuristic
                if track.lower().replace("_", "") in p.name.lower().replace("_", ""):
                    candidates.append(p)
        except Exception:
            continue

    if not candidates:
        # Accept any recent recording (user runs one track at a time)
        all_recs = sorted(_REC_DIR.glob("recording_*.h5"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        return all_recs[0] if all_recs else None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _run_analysis(recording_path: Path) -> dict:
    """Run drive_summary_core analysis and return the full result dict."""
    from drive_summary_core import analyze_recording_summary  # type: ignore
    result = analyze_recording_summary(recording_path)
    return result


def _is_cadence_invalid(result: dict) -> bool:
    """Return True if the run is flagged cadence-invalid by the analyzer.

    Cadence-invalid runs are explicitly excluded from tuning comparison
    (marked 'Run excluded from tuning' in key_issues).  They should not
    be used to update or enforce score floors because their scores are
    distorted by E2E latency spikes, not by code regressions.
    """
    issues = result.get("executive_summary", {}).get("key_issues", [])
    return any("cadence" in str(i).lower() for i in issues)


# ---------------------------------------------------------------------------
# Parametrized score floor tests
# ---------------------------------------------------------------------------

def _floor_params():
    """Generate (track, floor_dict) pairs from scores.json."""
    floors = _load_floors()
    return [(track, floor) for track, floor in floors.items()]


@pytest.mark.parametrize("track,floor", _floor_params())
def test_overall_score_meets_floor(track, floor):
    """Overall score for the latest recording must be ≥ the committed floor."""
    recording = _find_latest_recording(track)
    if recording is None:
        pytest.skip(f"No recording found for track '{track}' in {_REC_DIR}")

    result   = _run_analysis(recording)

    if _is_cadence_invalid(result):
        pytest.skip(
            f"Track '{track}': latest recording {recording.name} is cadence-invalid "
            f"(E2E latency spikes distort the score). "
            f"Re-run in Unity to get a valid comparison against the floor."
        )

    score    = result.get("executive_summary", {}).get("overall_score")

    if score is None:
        pytest.skip(f"overall_score not found in analysis result for {recording.name}")

    floor_val = float(floor["overall"])
    assert float(score) >= floor_val, (
        f"Track '{track}': overall score {score:.1f} < floor {floor_val:.1f}. "
        f"Recording: {recording.name}. "
        f"Update scores.json if this is a known regression and the floor needs lowering."
    )


@pytest.mark.parametrize("track,floor", _floor_params())
def test_layer_scores_meet_floors(track, floor):
    """Each layer score in the committed floor must be met."""
    recording = _find_latest_recording(track)
    if recording is None:
        pytest.skip(f"No recording found for track '{track}' in {_REC_DIR}")

    result = _run_analysis(recording)

    if _is_cadence_invalid(result):
        pytest.skip(
            f"Track '{track}': latest recording {recording.name} is cadence-invalid. "
            f"Skipping layer floor check — re-run in Unity for a valid comparison."
        )

    layers = result.get("layer_breakdowns", {})

    if not layers:
        pytest.skip(f"layer_breakdowns not found in result for {recording.name}")

    layer_names = [k for k in floor if not k.startswith(("overall", "notes", "_"))]
    failures = []
    for layer_name in layer_names:
        if layer_name not in layers:
            continue
        actual = float(layers[layer_name].get("final_score", 0.0))
        minimum = float(floor[layer_name])
        if actual < minimum:
            failures.append(f"  {layer_name}: {actual:.1f} < floor {minimum:.1f}")

    assert not failures, (
        f"Track '{track}' layer score regressions in {recording.name}:\n"
        + "\n".join(failures)
    )
