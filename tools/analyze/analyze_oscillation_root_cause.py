"""
Deep analysis of oscillation to identify root cause.
Systematically investigates different parts of the system to find what's causing oscillation.

Supports both PID and MPC regimes. Curvature-segmented analysis identifies whether
oscillation is a straight-line or curve-tracking problem.

Usage:
    python tools/analyze/analyze_oscillation_root_cause.py [recording.h5] [--plot]
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
import sys

try:
    from scipy import signal as scipy_signal
    from scipy.fft import fft, fftfreq
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _safe(f, key):
    """Read HDF5 dataset as float array, returning None if missing."""
    if key not in f:
        return None
    return np.asarray(f[key][:], dtype=float)


def analyze_oscillation_root_cause(recording_file: str, plot: bool = False):
    """Deep dive into oscillation to find root cause."""

    print("=" * 80)
    print("OSCILLATION ROOT CAUSE ANALYSIS")
    print("=" * 80)
    print()

    with h5py.File(recording_file, 'r') as f:
        # Core signals
        lateral_error = np.asarray(f['control/lateral_error'][:], dtype=float)
        steering = np.asarray(f['control/steering'][:], dtype=float)
        heading_error = _safe(f, 'control/heading_error')
        vehicle_speed = _safe(f, 'vehicle/speed')
        regime = _safe(f, 'control/regime')

        # MPC signals
        mpc_e_lat = _safe(f, 'control/mpc_e_lat')
        smith_raw = _safe(f, 'control/mpc_smith_raw_e_lat')
        smith_pred = _safe(f, 'control/mpc_smith_e_lat_predicted')
        recovery = _safe(f, 'control/mpc_recovery_mode_suppressed')
        r_steer_rate_eff = _safe(f, 'control/mpc_r_steer_rate_effective')

        # Trajectory signals
        ref_x = _safe(f, 'trajectory/reference_point_x')
        raw_ref_x = _safe(f, 'trajectory/diag_raw_ref_x')
        smooth_ref_x = _safe(f, 'trajectory/diag_smoothed_ref_x')
        curvature = _safe(f, 'trajectory/reference_point_curvature')

        # PID signals (legacy, for PP analysis)
        pid_integral = _safe(f, 'control/pid_integral')

        timestamps = _safe(f, 'vehicle/timestamps')
        if timestamps is None:
            timestamps = _safe(f, 'control/timestamps')
        if timestamps is not None:
            time_seconds = timestamps - timestamps[0]
        else:
            time_seconds = np.arange(len(lateral_error)) / 30.0

        dt = np.mean(np.diff(time_seconds)) if len(time_seconds) > 1 else 1.0 / 30.0
        n = len(lateral_error)

        # Build masks
        mpc_mask = regime >= 0.5 if regime is not None else np.zeros(n, dtype=bool)
        pp_mask = ~mpc_mask
        curve_mask = np.abs(curvature) > 0.0005 if curvature is not None else np.zeros(n, dtype=bool)
        straight_mask = ~curve_mask
        straight_mpc = straight_mask & mpc_mask
        curve_mpc = curve_mask & mpc_mask

        # ── 1. Frequency analysis ──
        print("1. OSCILLATION CHARACTERISTICS")
        print("-" * 80)

        sign_changes = int(np.sum(np.diff(np.sign(lateral_error)) != 0))
        duration = float(time_seconds[-1]) if len(time_seconds) > 0 else 1.0
        osc_freq = sign_changes / duration if duration > 0 else 0
        print(f"   Sign-change frequency: {osc_freq:.2f} Hz ({sign_changes} changes over {duration:.1f}s)")

        if HAS_SCIPY and len(lateral_error) > 10:
            centered = lateral_error - np.mean(lateral_error)
            fft_vals = fft(centered)
            fft_freqs = fftfreq(len(centered), dt)
            pos_freqs = fft_freqs[:len(fft_freqs) // 2]
            pos_fft = np.abs(fft_vals[:len(fft_vals) // 2])
            if len(pos_fft) > 1:
                dom_idx = np.argmax(pos_fft[1:]) + 1
                print(f"   Dominant FFT frequency: {pos_freqs[dom_idx]:.2f} Hz (magnitude: {pos_fft[dom_idx]:.2f})")
                top3 = np.argsort(pos_fft[1:])[-3:][::-1] + 1
                for rank, idx in enumerate(top3, 1):
                    if idx < len(pos_freqs):
                        print(f"     #{rank}: {pos_freqs[idx]:.3f} Hz  mag={pos_fft[idx]:.2f}")

        # Growth rate
        if n > 60:
            window = max(10, n // 20)
            rms = np.array([np.sqrt(np.mean(lateral_error[max(0, i - window):i + 1] ** 2))
                            for i in range(n)])
            first_q = rms[:n // 4]
            last_q = rms[3 * n // 4:]
            growth = (np.mean(last_q) - np.mean(first_q)) / duration if duration > 0 else 0
            print(f"   RMS growth: {np.mean(first_q):.3f}m → {np.mean(last_q):.3f}m  slope={growth:.4f} m/s")
            runaway = "YES" if growth > 0.001 else "NO"
            print(f"   Amplitude runaway: {runaway}")
        print()

        # ── 2. Regime breakdown ──
        print("2. REGIME BREAKDOWN")
        print("-" * 80)
        print(f"   MPC frames: {int(mpc_mask.sum())}/{n} ({100 * mpc_mask.mean():.1f}%)")
        print(f"   PP frames:  {int(pp_mask.sum())}/{n} ({100 * pp_mask.mean():.1f}%)")
        print(f"   Curve frames: {int(curve_mask.sum())}  Straight: {int(straight_mask.sum())}")
        print(f"   Curve+MPC: {int(curve_mpc.sum())}  Straight+MPC: {int(straight_mpc.sum())}")
        print()

        # ── 3. Segmented lateral error ──
        print("3. LATERAL ERROR BY SEGMENT")
        print("-" * 80)
        for label, mask in [("All", np.ones(n, dtype=bool)),
                            ("Straight+MPC", straight_mpc),
                            ("Curve+MPC", curve_mpc),
                            ("PP", pp_mask)]:
            if mask.sum() > 0:
                ae = np.abs(lateral_error[mask])
                print(f"   {label:15s}  P50={np.median(ae):.3f}  P95={np.percentile(ae, 95):.3f}  "
                      f"Max={np.max(ae):.3f}  RMSE={np.sqrt(np.mean(ae ** 2)):.3f}  ({int(mask.sum())} frames)")
        print()

        # ── 4. Trajectory smoothing lag ──
        if raw_ref_x is not None and smooth_ref_x is not None:
            m = min(len(raw_ref_x), len(smooth_ref_x), n)
            lag = smooth_ref_x[:m] - raw_ref_x[:m]

            print("4. TRAJECTORY SMOOTHING LAG")
            print("-" * 80)
            for label, mask in [("All", np.ones(m, dtype=bool)),
                                ("Straight+MPC", straight_mpc[:m]),
                                ("Curve+MPC", curve_mpc[:m])]:
                if mask.sum() > 0:
                    al = np.abs(lag[mask])
                    print(f"   {label:15s}  P50={np.median(al):.4f}m  P95={np.percentile(al, 95):.4f}m  "
                          f"Max={np.max(al):.4f}m")

            # Lag as % of error — the smoking-gun metric
            cm = curve_mpc[:m]
            if cm.sum() > 10:
                abs_lag_c = np.abs(lag[cm])
                abs_elat_c = np.abs(lateral_error[:m][cm])
                safe_elat = np.maximum(abs_elat_c, 1e-6)
                lag_pct = abs_lag_c / safe_elat
                corr = float(np.corrcoef(abs_lag_c, abs_elat_c)[0, 1])
                print()
                print(f"   Curve lag as % of error:  P50={100 * np.median(lag_pct):.1f}%  "
                      f"P95={100 * np.percentile(lag_pct, 95):.1f}%")
                causal = "CAUSAL" if abs(corr) > 0.5 else "weak" if abs(corr) > 0.2 else "none"
                print(f"   Lag-error correlation:    {corr:.3f}  ({causal})")
                dominates = np.percentile(lag_pct, 95) > 0.5
                print(f"   >>> Smoothing lag dominates curve error: {'YES — FIX EMA FLOOR' if dominates else 'NO'}")
            print()

        # ── 5. MPC error pipeline ──
        if smith_raw is not None and mpc_e_lat is not None and mpc_mask.sum() > 0:
            print("5. MPC ERROR PIPELINE (raw → Smith → attenuation → MPC input)")
            print("-" * 80)
            m = min(len(smith_raw), len(mpc_e_lat), n)
            for label, mask in [("All MPC", mpc_mask[:m]),
                                ("Curve+MPC", curve_mpc[:m]),
                                ("Straight+MPC", straight_mpc[:m])]:
                if mask.sum() > 0:
                    raw = np.abs(smith_raw[:m][mask])
                    final = np.abs(mpc_e_lat[:m][mask])
                    ratio = np.median(final / np.maximum(raw, 1e-6))
                    print(f"   {label:15s}  raw_P50={np.median(raw):.3f}  final_P50={np.median(final):.3f}  "
                          f"ratio={ratio:.3f}")
            print()

        # ── 6. Recovery mode ──
        if recovery is not None and mpc_mask.sum() > 0:
            print("6. RECOVERY MODE SUPPRESSION")
            print("-" * 80)
            rec = recovery[:n]
            rec_all = float(rec[mpc_mask].mean()) * 100
            print(f"   All MPC:       {rec_all:.1f}%  ({int(rec[mpc_mask].sum())}/{int(mpc_mask.sum())})")
            if curve_mpc.sum() > 0:
                rec_c = float(rec[curve_mpc].mean()) * 100
                flag = "ELEVATED" if rec_c > 10 else "OK"
                print(f"   Curve+MPC:     {rec_c:.1f}%  [{flag}]")
            if straight_mpc.sum() > 0:
                rec_s = float(rec[straight_mpc].mean()) * 100
                print(f"   Straight+MPC:  {rec_s:.1f}%")
            print()

        # ── 7. Sign agreement (Smith predictor relevance) ──
        if heading_error is not None and mpc_mask.sum() > 0:
            print("7. SIGN AGREEMENT (Smith predictor relevance)")
            print("-" * 80)
            m = min(len(heading_error), n)
            for label, mask in [("All MPC", mpc_mask[:m]),
                                ("Curve+MPC", curve_mpc[:m]),
                                ("Straight+MPC", straight_mpc[:m])]:
                if mask.sum() > 0:
                    sa = (lateral_error[:m][mask] * heading_error[:m][mask]) >= 0
                    print(f"   {label:15s}  agree={100 * sa.mean():.1f}%  "
                          f"({'guard inactive — no-op' if sa.mean() > 0.95 else 'guard active'})")
            print()

        # ── 8. Phase analysis ──
        print("8. PHASE ANALYSIS (steering vs error)")
        print("-" * 80)
        if n > 10:
            e_norm = (lateral_error - np.mean(lateral_error)) / (np.std(lateral_error) + 1e-10)
            s_norm = (steering - np.mean(steering)) / (np.std(steering) + 1e-10)
            corr = np.correlate(e_norm, s_norm, mode='full')
            lags = np.arange(-n + 1, n)
            max_idx = np.argmax(np.abs(corr))
            max_lag = lags[max_idx]
            lag_time = max_lag * dt
            print(f"   Max correlation: {corr[max_idx]:.3f} at lag={max_lag} frames ({lag_time:.3f}s)")
            if abs(lag_time) < 0.1:
                print(f"   Steering and error are in phase")
            elif lag_time > 0:
                print(f"   Steering LAGS error by {lag_time:.3f}s")
            else:
                print(f"   Steering LEADS error by {abs(lag_time):.3f}s")
        print()

        # ── 9. Speed context ──
        if vehicle_speed is not None:
            print("9. SPEED CONTEXT")
            print("-" * 80)
            for label, mask in [("All", np.ones(n, dtype=bool)),
                                ("Straight+MPC", straight_mpc),
                                ("Curve+MPC", curve_mpc)]:
                if mask.sum() > 0:
                    sp = vehicle_speed[:n][mask]
                    print(f"   {label:15s}  P50={np.median(sp):.1f}  P95={np.percentile(sp, 95):.1f}  "
                          f"Max={np.max(sp):.1f} m/s")
            print()

        # ── 10. r_steer_rate scheduling ──
        if r_steer_rate_eff is not None and mpc_mask.sum() > 0:
            print("-" * 60)
            print("r_STEER_RATE SCHEDULING")
            print("-" * 60)
            rsr = r_steer_rate_eff[:n]
            rsr_mpc = rsr[mpc_mask]
            print(f"   Mean effective r_steer_rate: {np.mean(rsr_mpc):.3f}")
            print(f"   P95  effective r_steer_rate: {np.percentile(rsr_mpc, 95):.3f}")
            print(f"   Min/Max: {np.min(rsr_mpc):.3f} / {np.max(rsr_mpc):.3f}")
            if vehicle_speed is not None:
                sp_mpc = vehicle_speed[:n][mpc_mask]
                if len(sp_mpc) > 10:
                    cc = np.corrcoef(sp_mpc, rsr_mpc)[0, 1]
                    print(f"   Speed-r_steer_rate correlation: {cc:.3f}")
            # Count band transitions (r_steer_rate changes > 0.1)
            diffs = np.abs(np.diff(rsr_mpc))
            transitions = int(np.sum(diffs > 0.1))
            print(f"   Band transitions: {transitions}")
            if np.max(rsr_mpc) - np.min(rsr_mpc) < 0.01:
                print("   ⚠ Scheduling appears INACTIVE (constant r_steer_rate)")
            print()

        # ── 10b. e_lat speed attenuation — REMOVED (2026-03-31) ──
        # Speed-adaptive e_lat attenuation was removed. Root cause of oscillation was
        # the e_lat ramp limiter (0.05 m/frame), not insufficient gain reduction.
        # r_steer_rate scheduling handles v/L gain compensation correctly via QP cost.

        # ── 11. Automated root cause attribution ──
        print("=" * 80)
        print("ROOT CAUSE ATTRIBUTION")
        print("=" * 80)
        findings = []

        # Check smoothing lag dominance
        if raw_ref_x is not None and smooth_ref_x is not None:
            m = min(len(raw_ref_x), len(smooth_ref_x), n)
            lag = smooth_ref_x[:m] - raw_ref_x[:m]
            cm = curve_mpc[:m]
            if cm.sum() > 10:
                abs_lag_c = np.abs(lag[cm])
                abs_elat_c = np.abs(lateral_error[:m][cm])
                lag_pct_p95 = np.percentile(abs_lag_c / np.maximum(abs_elat_c, 1e-6), 95)
                if lag_pct_p95 > 0.5:
                    findings.append(("HIGH", f"EMA smoothing lag dominates curve error "
                                     f"(lag={100 * lag_pct_p95:.0f}% of error at P95). "
                                     f"Fix: curvature-gate the EMA floor."))

        # Check recovery mode
        if recovery is not None and curve_mpc.sum() > 0:
            rec_curve = float(recovery[:n][curve_mpc].mean()) * 100
            if rec_curve > 10:
                findings.append(("HIGH", f"Recovery mode elevated on curves: {rec_curve:.0f}% "
                                 f"(baseline ~3%). MPC knows it's in trouble."))

        # Check Smith predictor relevance
        if heading_error is not None and mpc_mask.sum() > 0:
            m = min(len(heading_error), n)
            sa = (lateral_error[:m][mpc_mask[:m]] * heading_error[:m][mpc_mask[:m]]) >= 0
            if sa.mean() > 0.95:
                findings.append(("INFO", f"Smith predictor guard is a no-op "
                                 f"({100 * sa.mean():.0f}% sign agreement). "
                                 f"Consider removing or making it curvature-aware."))

        # Check amplitude runaway
        if n > 60:
            window = max(10, n // 20)
            rms = np.array([np.sqrt(np.mean(lateral_error[max(0, i - window):i + 1] ** 2))
                            for i in range(n)])
            first_q = rms[:n // 4]
            last_q = rms[3 * n // 4:]
            growth = (np.mean(last_q) - np.mean(first_q)) / duration
            if growth > 0.003:
                findings.append(("HIGH", f"Oscillation amplitude runaway "
                                 f"(growth={growth:.4f} m/s). "
                                 f"Loop gain exceeds 1.0 at operating speed."))
            elif growth > 0.001:
                findings.append(("MEDIUM", f"Oscillation amplitude growing "
                                 f"(growth={growth:.4f} m/s)."))

        if findings:
            for sev, msg in findings:
                print(f"   [{sev:6s}] {msg}")
        else:
            print("   No clear root cause identified from signal analysis.")
        print()

        # ── 12. Inter-frame diagnostics ──
        if_active = _safe(f, 'control/interframe_active')
        if if_active is not None and np.any(if_active > 0.5):
            print("12. INTER-FRAME CONTROL EXTRAPOLATION")
            print("-" * 80)
            if_mask = if_active > 0.5
            if_count = int(np.sum(if_mask))
            if_total = _safe(f, 'control/interframe_total_count')
            if_updates = _safe(f, 'control/interframe_updates_this_cycle')
            if_dt = _safe(f, 'control/interframe_dt_actual')
            if_e_lat = _safe(f, 'control/interframe_last_e_lat')

            total = int(if_total[-1]) if if_total is not None and len(if_total) > 0 else if_count
            camera_frames = n
            effective_hz = (camera_frames + total) / max(1.0, time_seconds[-1]) if len(time_seconds) > 0 else 0
            camera_hz = camera_frames / max(1.0, time_seconds[-1]) if len(time_seconds) > 0 else 0

            # Target rate: camera + max_interframe_updates per camera cycle
            # Default: 3 updates/cycle → target ≈ camera_hz * (1 + 3)
            target_hz = camera_hz * 4.0 if camera_hz > 0 else 30.0
            rate_ratio = effective_hz / target_hz if target_hz > 0 else 0.0

            print(f"  Camera frames:       {camera_frames}")
            print(f"  Inter-frame updates: {total}")
            print(f"  Camera rate:         {camera_hz:.1f} Hz")
            print(f"  Effective rate:      {effective_hz:.1f} Hz (camera + inter-frame)")
            print(f"  Target rate:         {target_hz:.1f} Hz (camera × 4)")
            if rate_ratio < 0.6:
                print(f"  >>> RATE THROTTLED: {rate_ratio:.0%} of target — check for hidden sleeps in inter-frame path")
            elif rate_ratio < 0.85:
                print(f"  >>> Rate below target: {rate_ratio:.0%} — possibly limited by Unity physics update rate")

            if if_updates is not None:
                up_active = if_updates[if_mask]
                if len(up_active) > 0:
                    at_max = float(np.mean(up_active >= 3.0) * 100.0)
                    print(f"  Updates/cycle:       P50={np.median(up_active):.0f}  "
                          f"mean={np.mean(up_active):.1f}  at_max={at_max:.0f}%")

            if if_dt is not None:
                dt_active = if_dt[if_mask]
                if len(dt_active) > 0:
                    dt_p50_ms = np.median(dt_active) * 1000.0
                    dt_p95_ms = np.percentile(dt_active, 95) * 1000.0
                    print(f"  Inter-frame dt:      P50={dt_p50_ms:.1f}ms  "
                          f"P95={dt_p95_ms:.1f}ms")
                    # Anomaly: if P50 dt ≈ camera interval, inter-frame is sleeping too long
                    camera_dt_ms = (1000.0 / camera_hz) if camera_hz > 0 else 133.0
                    if dt_p50_ms > camera_dt_ms * 0.7:
                        print(f"  >>> DT ANOMALY: P50 dt ({dt_p50_ms:.0f}ms) ≈ camera interval "
                              f"({camera_dt_ms:.0f}ms) — inter-frame path likely contains a sleep "
                              f"at camera rate instead of inter-frame rate")

            if if_e_lat is not None and mpc_e_lat is not None:
                el_if = if_e_lat[if_mask]
                # Compare inter-frame e_lat to next camera mpc_e_lat
                divergence = np.abs(if_e_lat[:-1] - mpc_e_lat[1:])[if_mask[:-1]]
                if len(divergence) > 0:
                    print(f"  GT divergence:       mean={np.mean(divergence):.4f}m  "
                          f"P95={np.percentile(divergence, 95):.4f}m  "
                          f"max={np.max(divergence):.4f}m")
                    pct_over = float(np.mean(divergence > 0.2) * 100.0)
                    if pct_over > 0:
                        print(f"  Divergence > 0.2m:   {pct_over:.1f}% of inter-frame updates")
            print()
        else:
            print("12. INTER-FRAME CONTROL EXTRAPOLATION")
            print("-" * 80)
            print("  Inter-frame not active in this recording.")
            print()

    # Plot if requested
    if plot and HAS_MATPLOTLIB:
        _plot_oscillation(recording_file)
    elif plot and not HAS_MATPLOTLIB:
        print("WARNING: --plot requested but matplotlib not installed. Skipping plots.")


def _plot_oscillation(recording_file: str):
    """Generate matplotlib plots for visual diagnosis."""
    import matplotlib.pyplot as plt

    with h5py.File(recording_file, 'r') as f:
        e_lat = np.asarray(f['control/lateral_error'][:], dtype=float)
        steer = np.asarray(f['control/steering'][:], dtype=float)
        speed = _safe(f, 'vehicle/speed')
        curv = _safe(f, 'trajectory/reference_point_curvature')
        raw_x = _safe(f, 'trajectory/diag_raw_ref_x')
        smooth_x = _safe(f, 'trajectory/diag_smoothed_ref_x')

    n = len(e_lat)
    frames = np.arange(n)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(frames, e_lat, label='lateral_error', linewidth=0.8)
    axes[0].set_ylabel('Lateral Error (m)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(frames, steer, label='steering', linewidth=0.8, color='orange')
    axes[1].set_ylabel('Steering')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if raw_x is not None and smooth_x is not None:
        m = min(len(raw_x), len(smooth_x), n)
        axes[2].plot(frames[:m], raw_x[:m], label='raw_ref_x', linewidth=0.8, alpha=0.7)
        axes[2].plot(frames[:m], smooth_x[:m], label='smooth_ref_x', linewidth=0.8)
        axes[2].plot(frames[:m], smooth_x[:m] - raw_x[:m], label='smoothing_lag', linewidth=0.8, color='red')
        axes[2].set_ylabel('Reference X / Lag (m)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    if curv is not None:
        m = min(len(curv), n)
        ax3 = axes[3]
        ax3.plot(frames[:m], curv[:m], label='curvature', linewidth=0.8, color='green')
        ax3.axhline(y=0.0005, color='gray', linestyle='--', alpha=0.5, label='curve threshold')
        ax3.set_ylabel('Curvature (1/m)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Frame')
    plt.suptitle(f'Oscillation Root Cause — {Path(recording_file).name}')
    plt.tight_layout()
    plt.savefig('oscillation_root_cause.png', dpi=150)
    print(f"Plot saved to oscillation_root_cause.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oscillation root cause analysis")
    parser.add_argument("recording", nargs="?", default=None, help="Recording HDF5 file")
    parser.add_argument("--plot", action="store_true", help="Generate matplotlib plots (requires matplotlib)")
    parser.add_argument("--latest", action="store_true", help="Use latest recording")
    args = parser.parse_args()

    if args.recording:
        recording_file = args.recording
    elif args.latest or args.recording is None:
        recordings = sorted(Path('data/recordings').glob('*.h5'), key=lambda p: p.stat().st_mtime, reverse=True)
        if recordings:
            recording_file = str(recordings[0])
            print(f"Using latest recording: {recording_file}\n")
        else:
            print("No recordings found.")
            sys.exit(1)
    else:
        print("Usage: python tools/analyze/analyze_oscillation_root_cause.py [recording.h5] [--plot]")
        sys.exit(1)

    analyze_oscillation_root_cause(recording_file, plot=args.plot)
