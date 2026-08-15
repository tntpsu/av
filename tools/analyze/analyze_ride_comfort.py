"""
Frequency-weighted ride comfort (ISO 2631-1) for AV recordings.

WHY THIS EXISTS
---------------
Every comfort gate in this project is an amplitude percentile: accel P95 <= 3.0,
jerk P95 <= 6.0, lateral P95 <= 0.40. Human discomfort depends far more on the
FREQUENCY of motion than on its peak amplitude, so those gates cannot tell a
0.24 Hz wander from a 3 Hz shake with the same P95 — one causes nausea, the
other feels like a rough road.

Measured on highway_h3 (recording_20260814_154040): the lane-offset weave that
was visible to the naked eye sits at 0.243 Hz — inside ISO 2631-1's
motion-sickness band (0.1-0.5 Hz) and entirely BELOW the ride-comfort band
(0.5-80 Hz) our P95 gates target. It was invisible to every existing metric.

WHAT IT COMPUTES
----------------
1. MSDV  (Motion Sickness Dose Value, ISO 2631-1 §8) — the sickness metric.
   MSDV = sqrt( integral over t of a_w(t)^2 dt ), where a_w is acceleration
   frequency-weighted by W_f (peaks ~0.16-0.25 Hz). Units m/s^1.5.
2. a_w RMS — frequency-weighted RMS acceleration (ISO 2631-1 comfort scale).
3. Dominant oscillation frequency and amplitude of the lateral path.

Both lateral (steering weave) and longitudinal (speed hunting) axes are
reported, because "oscillation" in this project has meant both at different
times and conflating them has already cost a day of investigation.

Usage:
    python3 tools/analyze/analyze_ride_comfort.py --latest
    python3 tools/analyze/analyze_ride_comfort.py --file data/recordings/x.h5
    python3 tools/analyze/analyze_ride_comfort.py --file x.h5 --window 60 130
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

# ── ISO 2631-1 bands ────────────────────────────────────────────────────────
MSDV_BAND_HZ = (0.1, 0.5)      # motion sickness (W_f weighting)
COMFORT_BAND_HZ = (0.5, 80.0)  # ride comfort (W_d / W_k weighting)

# ISO 2631-1 comfort reaction scale, a_w RMS in m/s^2 (Annex C, informative).
COMFORT_SCALE = [
    (0.315, "not uncomfortable"),
    (0.63,  "a little uncomfortable"),
    (1.0,   "fairly uncomfortable"),
    (1.6,   "uncomfortable"),
    (2.5,   "very uncomfortable"),
    (float("inf"), "extremely uncomfortable"),
]


def wf_weighting(freqs: np.ndarray) -> np.ndarray:
    """ISO 2631-1 W_f motion-sickness frequency weighting (magnitude).

    Band-pass peaking near 0.16-0.25 Hz, rolling off either side. This is the
    simplified magnitude response; the full standard specifies a transfer
    function, but the shape is what matters for ranking runs against each other.
    """
    f = np.asarray(freqs, dtype=float)
    w = np.zeros_like(f)
    nz = f > 0
    # Peak at ~0.2 Hz, -3dB near 0.1 and 0.5 Hz.
    f0, q = 0.2, 0.9
    with np.errstate(divide="ignore", invalid="ignore"):
        w[nz] = 1.0 / np.sqrt(1.0 + (q * (f[nz] / f0 - f0 / f[nz])) ** 2)
    return w


def _weighted_accel(sig: np.ndarray, fs: float, weighting) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a frequency weighting to `sig` via FFT. Returns (a_w, freqs, spectrum)."""
    n = sig.size
    x = sig - np.mean(sig)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    a_w = np.fft.irfft(spec * weighting(freqs), n=n)
    return a_w, freqs, np.abs(spec)


def compute_msdv(accel: np.ndarray, fs: float) -> float:
    """MSDV per ISO 2631-1: sqrt(∫ a_w(t)^2 dt). Units m/s^1.5."""
    if accel.size < 4 or fs <= 0:
        return float("nan")
    a_w, _, _ = _weighted_accel(accel, fs, wf_weighting)
    return float(np.sqrt(np.sum(a_w ** 2) / fs))


def dominant_frequency(sig: np.ndarray, fs: float, fmin: float = 0.05) -> tuple[float, float]:
    """Dominant frequency (Hz) and its peak-to-peak amplitude in signal units."""
    if sig.size < 8 or fs <= 0:
        return float("nan"), float("nan")
    x = (sig - np.mean(sig)) * np.hanning(sig.size)
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(sig.size, 1.0 / fs)
    band = freqs >= fmin
    if not band.any():
        return float("nan"), float("nan")
    idx = np.argmax(spec[band])
    return float(freqs[band][idx]), float(np.max(sig) - np.min(sig))


def comfort_verdict(aw_rms: float) -> str:
    for limit, label in COMFORT_SCALE:
        if aw_rms < limit:
            return label
    return COMFORT_SCALE[-1][1]


def _load(path: Path, window: tuple[float, float] | None):
    with h5py.File(path, "r") as f:
        ts = np.asarray(f["camera/timestamps"][:], dtype=float)
        t = ts - ts[0]
        def get(*keys):
            for k in keys:
                if k in f:
                    return np.asarray(f[k][:], dtype=float)
            return None
        lat_off = get("ground_truth/ego_lane_cross_track_road_frame_at_car",
                      "vehicle/road_frame_lateral_offset")
        speed = get("vehicle/speed", "vehicle/speed_mps")
        lat_acc = get("vehicle/lateral_accel", "control/regime_lateral_accel_mps2")
    m = np.ones(t.size, dtype=bool)
    if window is not None:
        m = (t >= window[0]) & (t <= window[1])

    def sel(a):
        return a[m] if a is not None else None

    return t[m], sel(lat_off), sel(speed), sel(lat_acc)


def main() -> int:
    ap = argparse.ArgumentParser(description="ISO 2631-1 frequency-weighted ride comfort")
    ap.add_argument("--file")
    ap.add_argument("--latest", action="store_true")
    ap.add_argument("--window", nargs=2, type=float, metavar=("T0", "T1"),
                    help="restrict to a time window in seconds")
    args = ap.parse_args()

    if args.latest or not args.file:
        recs = sorted((REPO_ROOT / "data" / "recordings").glob("*.h5"),
                      key=lambda p: p.stat().st_mtime)
        if not recs:
            print("no recordings found")
            return 1
        path = recs[-1]
    else:
        path = Path(args.file)

    t, lat_off, speed, lat_acc = _load(path, tuple(args.window) if args.window else None)
    if t.size < 8:
        print("not enough frames")
        return 1
    fs = (t.size - 1) / (t[-1] - t[0])

    print("=" * 72)
    print("  RIDE COMFORT — ISO 2631-1 frequency-weighted")
    print("=" * 72)
    print(f"  Recording: {path.name}")
    print(f"  Window:    {t[0]:.1f}–{t[-1]:.1f}s   sample rate {fs:.1f} Hz")
    print()
    print(f"  MSDV band (motion sickness): {MSDV_BAND_HZ[0]}–{MSDV_BAND_HZ[1]} Hz")
    print(f"  Comfort band (ride quality): {COMFORT_BAND_HZ[0]}–{COMFORT_BAND_HZ[1]} Hz")
    print()

    # ── Lateral axis (steering weave) ───────────────────────────────────────
    if lat_off is not None and lat_off.size > 8:
        # Second derivative of lane offset ≈ lateral acceleration of the path.
        a_lat = np.gradient(np.gradient(lat_off, 1.0 / fs), 1.0 / fs)
        f_dom, pk_pk = dominant_frequency(lat_off, fs)
        msdv = compute_msdv(a_lat, fs)
        a_w, _, _ = _weighted_accel(a_lat, fs, wf_weighting)
        aw_rms = float(np.sqrt(np.mean(a_w ** 2)))
        in_band = MSDV_BAND_HZ[0] <= f_dom <= MSDV_BAND_HZ[1]
        print("  LATERAL (path weave)")
        print(f"    dominant frequency     {f_dom:8.3f} Hz   (period {1/f_dom:5.1f}s)"
              f"   {'← IN MOTION-SICKNESS BAND' if in_band else ''}")
        print(f"    peak-to-peak offset    {pk_pk:8.2f} m")
        print(f"    MSDV                   {msdv:8.3f} m/s^1.5")
        print(f"    a_w RMS                {aw_rms:8.3f} m/s²   → {comfort_verdict(aw_rms)}")
        print()

    # ── Longitudinal axis (speed hunting) ───────────────────────────────────
    if speed is not None and speed.size > 8:
        a_lon = np.gradient(speed, 1.0 / fs)
        f_dom, pk_pk = dominant_frequency(speed, fs)
        msdv = compute_msdv(a_lon, fs)
        a_w, _, _ = _weighted_accel(a_lon, fs, wf_weighting)
        aw_rms = float(np.sqrt(np.mean(a_w ** 2)))
        in_band = MSDV_BAND_HZ[0] <= f_dom <= MSDV_BAND_HZ[1]
        print("  LONGITUDINAL (speed hunting)")
        print(f"    dominant frequency     {f_dom:8.3f} Hz   (period {1/f_dom:5.1f}s)"
              f"   {'← IN MOTION-SICKNESS BAND' if in_band else ''}")
        print(f"    peak-to-peak speed     {pk_pk:8.2f} m/s")
        print(f"    MSDV                   {msdv:8.3f} m/s^1.5")
        print(f"    a_w RMS                {aw_rms:8.3f} m/s²   → {comfort_verdict(aw_rms)}")
        print()

    print("  NOTE: amplitude-only gates (accel P95, jerk P95, lateral P95) cannot")
    print("  distinguish these frequencies. A 0.24 Hz weave and a 3 Hz shake with")
    print("  identical P95 score the same but feel completely different.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
