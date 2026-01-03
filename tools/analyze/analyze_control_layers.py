#!/usr/bin/env python3
"""
Analyze control layer architecture and identify potential issues from stacked controllers/filters.
"""

import sys
from pathlib import Path
import h5py
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq

def analyze_control_layers(recording_file: str):
    """Analyze control layers and their interactions."""
    
    print("=" * 80)
    print("CONTROL LAYER ARCHITECTURE ANALYSIS")
    print("=" * 80)
    print()
    
    with h5py.File(recording_file, 'r') as f:
        # Load data
        lateral_error = np.array(f['control/lateral_error'][:])
        steering = np.array(f['control/steering'][:])
        ref_x = np.array(f.get('trajectory/reference_point_x', [])[:]) if 'trajectory/reference_point_x' in f else None
        ref_x_raw = np.array(f.get('trajectory/reference_point_raw_x', [])[:]) if 'trajectory/reference_point_raw_x' in f else None
        
        timestamps = f.get('control/timestamps', f.get('camera/timestamps', None))
        if timestamps is not None:
            timestamps = np.array(timestamps[:])
            time = timestamps - timestamps[0]
        else:
            time = np.arange(len(lateral_error)) / 30.0
        
        dt = np.mean(np.diff(time)) if len(time) > 1 else 1.0/30.0
        
        print("CONTROL LAYER ARCHITECTURE:")
        print("-" * 80)
        print("Layer 1: Lane Detection Smoothing")
        print("  - Location: trajectory/models/trajectory_planner.py")
        print("  - Type: Exponential smoothing (alpha=0.8)")
        print("  - Purpose: Smooth lane coefficients over time")
        print("  - Delay: ~1-2 frames")
        print()
        
        print("Layer 2: Reference Point Smoothing")
        print("  - Location: trajectory/inference.py")
        print("  - Type: Exponential smoothing (alpha=0.80)")
        print("  - Purpose: Smooth reference point to reduce noise")
        print("  - Delay: ~2-3 frames")
        print()
        
        print("Layer 3: PID Controller")
        print("  - Location: control/pid_controller.py")
        print("  - Type: PID with integral decay")
        print("  - Purpose: Compute steering from error")
        print("  - Delay: ~0 frames (immediate)")
        print()
        
        print("Layer 4: Steering Rate Limiting")
        print("  - Location: control/pid_controller.py")
        print("  - Type: Rate limiter (max 0.1 per frame)")
        print("  - Purpose: Prevent sudden steering changes")
        print("  - Delay: ~1 frame")
        print()
        
        print("Layer 5: Steering Saturation")
        print("  - Location: control/pid_controller.py")
        print("  - Type: Clipping (max ±0.5)")
        print("  - Purpose: Limit steering range")
        print("  - Delay: ~0 frames (immediate)")
        print()
        
        print("=" * 80)
        print("TOTAL DELAY ANALYSIS:")
        print("=" * 80)
        print()
        
        # Calculate total delay
        total_delay_frames = 1 + 2 + 0 + 1 + 0  # Sum of delays
        total_delay_seconds = total_delay_frames * dt
        print(f"Estimated total delay: {total_delay_frames} frames ({total_delay_seconds:.3f}s)")
        print()
        
        # Check for phase issues
        if ref_x is not None and ref_x_raw is not None and len(ref_x) == len(lateral_error):
            print("PHASE ANALYSIS:")
            print("-" * 80)
            
            # Cross-correlation between raw and smoothed reference
            ref_x_centered = ref_x - np.mean(ref_x)
            ref_x_raw_centered = ref_x_raw - np.mean(ref_x_raw)
            
            correlation = np.correlate(ref_x_centered, ref_x_raw_centered, mode='full')
            lags = np.arange(-len(ref_x)+1, len(ref_x))
            max_corr_idx = np.argmax(np.abs(correlation))
            max_lag = lags[max_corr_idx]
            lag_time = max_lag * dt
            
            print(f"Reference smoothing lag: {max_lag} frames ({lag_time:.3f}s)")
            if abs(lag_time) > 0.1:
                print(f"  ⚠️  Significant delay from smoothing!")
            print()
            
            # Cross-correlation between error and steering
            error_centered = lateral_error - np.mean(lateral_error)
            steering_centered = steering - np.mean(steering)
            
            correlation = np.correlate(error_centered, steering_centered, mode='full')
            lags = np.arange(-len(lateral_error)+1, len(lateral_error))
            max_corr_idx = np.argmax(np.abs(correlation))
            max_lag = lags[max_corr_idx]
            lag_time = max_lag * dt
            
            print(f"Error-to-steering lag: {max_lag} frames ({lag_time:.3f}s)")
            if abs(lag_time) > 0.1:
                print(f"  ⚠️  Significant delay in control response!")
            print()
        
        # Frequency response analysis
        print("FREQUENCY RESPONSE ANALYSIS:")
        print("-" * 80)
        
        if len(lateral_error) > 10:
            # FFT of error
            error_fft = fft(lateral_error - np.mean(lateral_error))
            error_freqs = fftfreq(len(lateral_error), dt)
            error_power = np.abs(error_fft)
            
            # FFT of steering
            steering_fft = fft(steering - np.mean(steering))
            steering_freqs = fftfreq(len(steering), dt)
            steering_power = np.abs(steering_fft)
            
            # Find dominant frequencies
            positive_freqs = error_freqs[:len(error_freqs)//2]
            error_power_pos = error_power[:len(error_power)//2]
            steering_power_pos = steering_power[:len(steering_power)//2]
            
            error_dominant_idx = np.argmax(error_power_pos[1:]) + 1
            steering_dominant_idx = np.argmax(steering_power_pos[1:]) + 1
            
            error_dominant_freq = positive_freqs[error_dominant_idx]
            steering_dominant_freq = positive_freqs[steering_dominant_idx]
            
            print(f"Error dominant frequency: {error_dominant_freq:.2f} Hz")
            print(f"Steering dominant frequency: {steering_dominant_freq:.2f} Hz")
            
            # Check if frequencies match (should for good control)
            freq_diff = abs(error_dominant_freq - steering_dominant_freq)
            if freq_diff < 0.5:
                print(f"  ✓ Frequencies match (good control)")
            else:
                print(f"  ⚠️  Frequency mismatch: {freq_diff:.2f} Hz (possible phase issues)")
            print()
        
        # Layer interaction analysis
        print("LAYER INTERACTION ISSUES:")
        print("-" * 80)
        
        issues = []
        
        # Issue 1: Multiple smoothing layers
        if ref_x is not None:
            ref_x_std = np.std(ref_x)
            if ref_x_raw is not None:
                ref_x_raw_std = np.std(ref_x_raw)
                smoothing_reduction = ref_x_raw_std / ref_x_std if ref_x_std > 0 else 0
                print(f"Reference smoothing reduction: {smoothing_reduction:.2f}x")
                if smoothing_reduction > 3.0:
                    issues.append("Excessive reference smoothing may cause delay")
        
        # Issue 2: Rate limiting + smoothing
        steering_changes = np.abs(np.diff(steering))
        max_change = np.max(steering_changes)
        mean_change = np.mean(steering_changes)
        print(f"Steering change rate: mean={mean_change:.4f}, max={max_change:.4f}")
        if max_change > 0.1:
            print(f"  ⚠️  Rate limiting may be too restrictive (max allowed: 0.1)")
            issues.append("Steering rate limiting may be too restrictive")
        
        # Issue 3: Integral accumulation with smoothing
        pid_integral = f.get('control/pid_integral', None)
        if pid_integral is not None:
            pid_integral = np.array(pid_integral[:])
            first_third = np.abs(pid_integral[:len(pid_integral)//3])
            last_third = np.abs(pid_integral[2*len(pid_integral)//3:])
            accumulation = np.mean(last_third) / np.mean(first_third) if np.mean(first_third) > 0 else 0
            if accumulation > 1.5:
                issues.append(f"PID integral accumulating ({accumulation:.2f}x) despite smoothing")
        
        print()
        if issues:
            print("IDENTIFIED ISSUES:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("✓ No major layer interaction issues detected")
        print()
        
        # Recommendations
        print("=" * 80)
        print("RECOMMENDATIONS:")
        print("=" * 80)
        print()
        print("1. REDUCE SMOOTHING LAYERS:")
        print("   - Consider reducing reference_smoothing: 0.80 → 0.75")
        print("   - Or reduce lane_smoothing_alpha: 0.8 → 0.75")
        print("   - Each smoothing layer adds delay (~2-3 frames)")
        print()
        print("2. OPTIMIZE RATE LIMITING:")
        print("   - Current: 0.1 per frame (3.0 per second)")
        print("   - If steering changes are hitting limit frequently, consider:")
        print("     - Increase to 0.15 per frame (4.5 per second)")
        print("     - Or reduce smoothing to allow faster response")
        print()
        print("3. ADDRESS INTEGRAL ACCUMULATION:")
        print("   - Current decay: 0.90 per frame when integral > 0.01")
        print("   - Try: 0.85 per frame (more aggressive)")
        print("   - Or: Lower threshold to 0.005 (catch earlier)")
        print()
        print("4. CONSIDER REMOVING ONE SMOOTHING LAYER:")
        print("   - Option A: Remove reference point smoothing (keep lane smoothing)")
        print("   - Option B: Remove lane smoothing (keep reference smoothing)")
        print("   - Test which provides better stability with less delay")
        print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        recordings = sorted(Path('data/recordings').glob('*.h5'), key=lambda p: p.stat().st_mtime, reverse=True)
        if recordings:
            recording_file = str(recordings[0])
            print(f"Using latest recording: {recording_file}\n")
        else:
            print("No recordings found. Usage: python tools/analyze_control_layers.py <recording.h5>")
            sys.exit(1)
    else:
        recording_file = sys.argv[1]
    
    analyze_control_layers(recording_file)

