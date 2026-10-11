#!/usr/bin/env python3
"""Closed-loop plant identification from a steering-injection recording (T-LATERAL-LMPC-SPEED-MARGIN).

The run added a known multisine r(t) to the steering command after the controller
(`experiment.steering_injection`, tools/sysid_injection.py). With u = controller command
(`control/steering`), the plant input is u + r and every output y satisfies y = P·(u + r), so
    P(f) = S_yr / (S_ur + S_rr)
estimated at the tone frequencies from one periodogram over the excitation window (tone bins only,
so the controller's own activity between tones does not bias the estimate). Outputs:
    P_d  : steering command → at-car cross-track (m per unit)
    P_yr : steering command → yaw rate (rad/s per unit)
    A    : command → actual steer angle (deg per unit, actuator)
r is regenerated from the recording's runtime config and `vehicle/unity_time`, time-aligned by
cross-correlation against (vehicle/steering_input − control/steering).

Usage:
    python3 tools/analyze/sysid_steer_injection.py <recording.h5> [more.h5 ...]  [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from tools.sysid_injection import InjectionConfig  # noqa: E402


def _tone_xspec(x: np.ndarray, y: np.ndarray, dt: float, tones: np.ndarray, half_bins: int = 0):
    """Cross-spectrum S_yx evaluated EXACTLY at each tone (direct projection with a Hann taper), which
    is exact for a deterministic multisine; FFT bins (1/T) are not centred on log-spaced tones."""
    n = len(x); w = np.hanning(n); t = np.arange(n) * dt
    xw = (x - x.mean()) * w; yw = (y - y.mean()) * w
    out = []
    for fk in tones:
        ek = np.exp(-2j * math.pi * fk * t)
        X = np.sum(xw * ek); Y = np.sum(yw * ek)
        out.append(Y * np.conj(X))
    return np.array(out)


def identify(path: str) -> dict:
    with h5py.File(path, "r") as h:
        rc = json.loads(h["meta/runtime_config_json"][0]) if "meta/runtime_config_json" in h else {}
        cfg = InjectionConfig.from_config(rc)
        if not cfg.enabled:
            # The recorder's runtime snapshot omits whole sections; rebuild the merged config from the overlay.
            try:
                from av_stack.config import load_config
                ov = rc.get("config_overlay_path")
                cfg = InjectionConfig.from_config(load_config(str(REPO_ROOT / ov) if ov and not str(ov).startswith("/") else ov))
            except Exception:
                pass
        if not cfg.enabled:
            cfg = InjectionConfig(enabled=True)   # base defaults; overlays only flip `enabled`
        t = h["vehicle/unity_time"][()].astype(float); v = h["vehicle/speed"][()].astype(float)
        u = h["control/steering"][()].astype(float); sin_ = h["vehicle/steering_input"][()].astype(float)
        d = h["ground_truth/selected_lane_cross_track_road_frame_at_car"][()].astype(float)
        act = -np.radians(h["vehicle/steering_angle_actual"][()].astype(float))        # Unity feedback sign
        hd = np.unwrap(np.radians(h["vehicle/car_heading_deg"][()].astype(float)))
    dt = float(np.median(np.diff(t))); yaw = np.gradient(hd, t)
    # recorded excitation: what Unity received minus what the controller commanded, lagged by the pipeline
    rec = np.full_like(u, np.nan)
    lag = 2
    rec[:-lag] = sin_[lag:] - u[:-lag]
    rec = np.nan_to_num(rec)
    side = REPO_ROOT / "data" / "reports" / "sysid" / f"injection_{Path(path).stem}.npz"
    if side.exists():
        # The sidecar is the excitation actually applied, stamped with Python monotonic time; the recording's
        # clocks are Unity-side and the recorder skips frames, so align by ONE time offset found by
        # cross-correlation (a multisine gives a sharp peak), coarse then fine.
        z = np.load(side); mono = z["mono"]; rz = z["r"]; sent = z["steering_sent"]; order = np.argsort(mono); mono, rz, sent = mono[order], rz[order], sent[order]
        # Align on the FULL sent command vs what Unity recorded receiving (same signal, two clocks) — far
        # cleaner than the command-minus-input difference. Coarse then fine offset search.
        base = float(mono[0] - t[0]); best = None; si_c = sin_ - sin_.mean()
        for step, span in ((0.05, 4.0), (0.005, 0.1)):
            centre = base if best is None else best[1]
            for off in np.arange(centre - span, centre + span, step):
                ss = np.interp(t + off, mono, sent, left=np.nan, right=np.nan); ok = np.isfinite(ss)
                if ok.sum() < 200: continue
                a = ss[ok] - ss[ok].mean(); b = si_c[ok] - si_c[ok].mean()
                c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                if best is None or c > best[0]: best = (c, off)
        corr, off = best; r = np.interp(t + off, mono, rz, left=0.0, right=0.0); shift = float(off - base); source = "sidecar (sent↔input aligned)"
    else:
        # regenerate r and align by cross-correlation against the recorded difference
        tau0 = t - t[0]; best = None
        for shift in np.arange(-3.0, 3.0, dt):
            r = cfg.series(tau0 - shift, v)
            if np.count_nonzero(r) < 100: continue
            c = float(np.dot(r, rec) / (np.linalg.norm(r) * np.linalg.norm(rec) + 1e-12))
            if best is None or c > best[0]: best = (c, shift, r)
        corr, shift, r = best; source = "regenerated"
    m = r != 0.0
    i0, i1 = int(np.flatnonzero(m)[0]), int(np.flatnonzero(m)[-1]) + 1
    sl = slice(i0, i1); tones = cfg.tones_hz()
    Srr = _tone_xspec(r[sl], r[sl], dt, tones); Sur = _tone_xspec(r[sl], u[sl], dt, tones)
    denom = Sur + Srr
    P_d = _tone_xspec(r[sl], d[sl], dt, tones) / denom
    P_yr = _tone_xspec(r[sl], yaw[sl], dt, tones) / denom
    A = _tone_xspec(r[sl], act[sl], dt, tones) / denom
    G_ur = Sur / Srr                                           # controller response to the disturbance (loop sensitivity)
    Sdr = _tone_xspec(r[sl], d[sl], dt, tones); Sar = _tone_xspec(r[sl], act[sl], dt, tones)
    P_per_rad = Sdr / Sar                                      # lateral position per rad of ACTUAL steer angle (actuator removed)
    vm = float(np.mean(v[sl]))
    # mid-band summary (0.35-0.55 Hz, the R200 mode): median over tones in band
    band = (tones >= 0.33) & (tones <= 0.58)
    k_peak = int(np.argmax(np.abs(G_ur)))
    return {
        "file": Path(path).name, "speed_mean_mps": vm, "frames": int(i1 - i0), "dt_s": dt,
        "r_alignment_corr": corr, "r_shift_s": float(shift), "r_source": source, "injection": cfg.__dict__,
        "d_rms_window_m": float(np.sqrt(np.mean(d[sl] ** 2))), "steering_input_p95": float(np.percentile(np.abs(sin_[sl]), 95)),
        "tones_hz": tones.tolist(),
        "P_d_abs_m_per_unit": np.abs(P_d).tolist(), "P_d_phase_deg": np.degrees(np.angle(P_d)).tolist(),
        "P_yaw_abs_radps_per_unit": np.abs(P_yr).tolist(), "P_yaw_phase_deg": np.degrees(np.angle(P_yr)).tolist(),
        "A_abs_deg_per_unit": np.degrees(np.abs(A)).tolist(), "A_phase_deg": np.degrees(np.angle(A)).tolist(),
        "G_ur_abs": np.abs(G_ur).tolist(), "P_per_rad_abs_m": np.abs(P_per_rad).tolist(), "P_per_rad_phase_deg": np.degrees(np.angle(P_per_rad)).tolist(),
        "summary": {"sens_peak": float(np.abs(G_ur[k_peak])), "sens_peak_hz": float(tones[k_peak]),
                    "P_d_midband_m_per_unit": float(np.median(np.abs(P_d[band]))) if band.any() else None,
                    "A_midband_deg_per_unit": float(np.median(np.degrees(np.abs(A[band])))) if band.any() else None,
                    "P_per_rad_midband_m": float(np.median(np.abs(P_per_rad[band]))) if band.any() else None,
                    "kin_P_per_rad_midband_m": float(np.median([abs(kinematic_model(vm, fk, cmd_gain_deg=180/math.pi)) for fk in tones[band]])) if band.any() else None}, "snr_r_db": (10 * np.log10(np.abs(Srr) / max(1e-12, np.median(np.abs(_tone_xspec(rec[sl] - r[sl], rec[sl] - r[sl], dt, tones)))))).tolist(),
    }


def kinematic_model(v: float, f: float, L=2.5, Kus=0.0046, tau=0.25, delay=0.10, cmd_gain_deg=None) -> complex:
    """steer command → lateral position, kinematic + understeer + first-order actuator + delay."""
    w = 2 * math.pi * f; s = 1j * w
    ms = math.radians(cmd_gain_deg) if cmd_gain_deg else float(np.interp(v, [0, 12], [0.5236, 0.2793])) * 0.85
    Leff = L * (1 + Kus * v * v)
    return ms * (v / Leff) / s * v / s * (1 / (1 + s * tau)) * np.exp(-s * delay)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("recordings", nargs="+"); ap.add_argument("--json", default=None)
    a = ap.parse_args(); results = []
    for p in a.recordings:
        r = identify(p); results.append(r)
        print(f"\n== {r['file']}  v={r['speed_mean_mps']:.1f} m/s  frames {r['frames']}  r {r['r_source']} corr {r['r_alignment_corr']:.2f} (shift {r['r_shift_s']:+.2f}s)  d_rms {r['d_rms_window_m']:.3f} m  steer_input p95 {r['steering_input_p95']:.2f}")
        sm = r["summary"]; print(f"   SUMMARY mid-band 0.33-0.58 Hz: |P_d| {sm['P_d_midband_m_per_unit']:.3f} m/unit  A {sm['A_midband_deg_per_unit']:.1f}°/unit  |P| per rad {sm['P_per_rad_midband_m']:.2f} m (kin+act model {sm['kin_P_per_rad_midband_m']:.2f})  sensitivity peak {sm['sens_peak']:.1f}x @ {sm['sens_peak_hz']:.2f} Hz")
        print(f"   {'f Hz':>6s} {'|P_d| m/u':>10s} {'∠P_d':>7s} {'kin model':>10s} {'ratio':>6s} | {'|P_yaw|':>8s} {'∠':>6s} | {'A °/u':>6s} {'∠A':>6s} | {'|G_ur|':>6s} {'SNR dB':>6s}")
        for i, fk in enumerate(r["tones_hz"]):
            km = abs(kinematic_model(r["speed_mean_mps"], fk, cmd_gain_deg=r["A_abs_deg_per_unit"][i] if r["A_abs_deg_per_unit"][i] > 1 else None))
            print(f"   {fk:6.3f} {r['P_d_abs_m_per_unit'][i]:10.3f} {r['P_d_phase_deg'][i]:7.0f} {km:10.3f} {r['P_d_abs_m_per_unit'][i]/max(km,1e-9):6.2f} | {r['P_yaw_abs_radps_per_unit'][i]:8.3f} {r['P_yaw_phase_deg'][i]:6.0f} | {r['A_abs_deg_per_unit'][i]:6.1f} {r['A_phase_deg'][i]:6.0f} | {r['G_ur_abs'][i]:6.2f} {r['snr_r_db'][i]:6.1f}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True); Path(a.json).write_text(json.dumps(results, indent=1))
        print("wrote", a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
