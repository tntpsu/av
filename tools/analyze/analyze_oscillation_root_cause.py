"""
Deep analysis of oscillation to identify root cause.
Systematically investigates different parts of the system to find what's causing oscillation.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy import signal
from scipy.fft import fft, fftfreq

def analyze_oscillation_root_cause(recording_file: str):
    """Deep dive into oscillation to find root cause."""
    
    print("=" * 80)
    print("OSCILLATION ROOT CAUSE ANALYSIS")
    print("=" * 80)
    print()
    
    with h5py.File(recording_file, 'r') as f:
        # Load data
        lateral_error = f['control/lateral_error'][:]
        steering = f['control/steering'][:]
        heading_error = f.get('control/heading_error', None)
        total_error = f.get('control/total_error', None)
        pid_integral = f.get('control/pid_integral', None)
        pid_derivative = f.get('control/pid_derivative', None)
        ref_x = f.get('trajectory/reference_point_x', None)
        ref_heading = f.get('trajectory/reference_point_heading', None)
        vehicle_speed = f.get('vehicle/speed', None)
        
        if heading_error is not None:
            heading_error = heading_error[:]
        if total_error is not None:
            total_error = total_error[:]
        if pid_integral is not None:
            pid_integral = pid_integral[:]
        if pid_derivative is not None:
            pid_derivative = pid_derivative[:]
        if ref_x is not None:
            ref_x = ref_x[:]
        if ref_heading is not None:
            ref_heading = ref_heading[:]
        if vehicle_speed is not None:
            vehicle_speed = vehicle_speed[:]
        
        timestamps = f.get('control/timestamps', f.get('camera/timestamps', None))
        if timestamps is not None:
            timestamps = timestamps[:]
            time_seconds = (timestamps - timestamps[0]) if len(timestamps) > 0 else np.arange(len(lateral_error)) / 30.0
        else:
            time_seconds = np.arange(len(lateral_error)) / 30.0
        
        dt = np.mean(np.diff(time_seconds)) if len(time_seconds) > 1 else 1.0/30.0
        
        print("OSCILLATION CHARACTERISTICS:")
        print("-" * 80)
        
        # Frequency analysis
        abs_lateral_error = np.abs(lateral_error)
        sign_changes = np.sum(np.diff(np.sign(lateral_error)) != 0)
        oscillation_freq = sign_changes / time_seconds[-1] if time_seconds[-1] > 0 else 0
        print(f"  Oscillation frequency: {oscillation_freq:.2f} Hz")
        print(f"  Sign changes: {sign_changes} over {time_seconds[-1]:.2f}s")
        print()
        
        # FFT analysis to find dominant frequencies
        if len(lateral_error) > 10:
            # Remove DC component
            lateral_error_centered = lateral_error - np.mean(lateral_error)
            fft_vals = fft(lateral_error_centered)
            fft_freqs = fftfreq(len(lateral_error_centered), dt)
            
            # Get positive frequencies only
            positive_freqs = fft_freqs[:len(fft_freqs)//2]
            positive_fft = np.abs(fft_vals[:len(fft_vals)//2])
            
            # Find dominant frequency
            dominant_idx = np.argmax(positive_fft[1:]) + 1  # Skip DC
            dominant_freq = positive_freqs[dominant_idx]
            dominant_magnitude = positive_fft[dominant_idx]
            
            print(f"  Dominant frequency (FFT): {dominant_freq:.2f} Hz")
            print(f"  Dominant magnitude: {dominant_magnitude:.2f}")
            
            # Find top 3 frequencies
            top3_indices = np.argsort(positive_fft[1:])[-3:][::-1] + 1
            print(f"  Top 3 frequencies:")
            for i, idx in enumerate(top3_indices, 1):
                if idx < len(positive_freqs):
                    print(f"    {i}. {positive_freqs[idx]:.2f} Hz (magnitude: {positive_fft[idx]:.2f})")
            print()
        
        # Phase analysis: Check if steering leads or lags error
        print("PHASE ANALYSIS (Steering vs Error):")
        print("-" * 80)
        if len(lateral_error) > 10 and len(steering) > 10:
            # Cross-correlation to find phase relationship
            # Normalize signals
            error_norm = (lateral_error - np.mean(lateral_error)) / (np.std(lateral_error) + 1e-10)
            steering_norm = (steering - np.mean(steering)) / (np.std(steering) + 1e-10)
            
            # Cross-correlation
            correlation = np.correlate(error_norm, steering_norm, mode='full')
            lags = np.arange(-len(lateral_error)+1, len(lateral_error))
            
            # Find lag with maximum correlation
            max_corr_idx = np.argmax(np.abs(correlation))
            max_lag = lags[max_corr_idx]
            max_corr = correlation[max_corr_idx]
            
            lag_time = max_lag * dt
            print(f"  Max correlation: {max_corr:.3f} at lag: {max_lag} frames ({lag_time:.3f}s)")
            
            if abs(lag_time) < 0.1:
                print(f"  ✓ Steering and error are in phase (no significant delay)")
            elif lag_time > 0:
                print(f"  ⚠️  Steering LAGS error by {lag_time:.3f}s (delayed response)")
            else:
                print(f"  ⚠️  Steering LEADS error by {abs(lag_time):.3f}s (anticipatory)")
            print()
        
        # Check for controller fighting itself
        print("CONTROLLER STABILITY ANALYSIS:")
        print("-" * 80)
        
        # Check if steering is proportional to error (should be)
        significant = (np.abs(lateral_error) > 0.05) & (np.abs(steering) > 0.02)
        if np.sum(significant) > 10:
            error_significant = lateral_error[significant]
            steering_significant = steering[significant]
            
            # Linear regression: steering = k * error
            correlation_coef = np.corrcoef(error_significant, steering_significant)[0, 1]
            print(f"  Steering-error correlation: {correlation_coef:.3f}")
            
            if correlation_coef > 0.7:
                print(f"  ✓ Strong positive correlation (steering follows error correctly)")
            elif correlation_coef > 0.3:
                print(f"  ⚠️  Weak correlation (steering not strongly following error)")
            else:
                print(f"  ⚠️  Very weak correlation (steering not following error)")
            
            # Calculate effective gain
            if np.std(error_significant) > 0.001:
                effective_gain = np.std(steering_significant) / np.std(error_significant)
                print(f"  Effective gain (steering/error): {effective_gain:.3f}")
                print(f"  (Config kp=0.5, but effective gain may differ due to total_error calculation)")
        print()
        
        # PID component analysis
        if total_error is not None and pid_integral is not None:
            print("PID COMPONENT ANALYSIS:")
            print("-" * 80)
            
            # Estimate PID components (approximate)
            # P component: proportional to error
            # I component: integral
            # D component: derivative of error
            
            # Check if integral is contributing to oscillation
            integral_abs = np.abs(pid_integral)
            error_abs = np.abs(lateral_error)
            
            # Correlation between integral and error
            if len(integral_abs) > 10:
                integral_error_corr = np.corrcoef(integral_abs, error_abs)[0, 1]
                print(f"  Integral-error correlation: {integral_error_corr:.3f}")
                
                if integral_error_corr > 0.5:
                    print(f"  ⚠️  High correlation - integral may be contributing to oscillation")
                else:
                    print(f"  ✓ Low correlation - integral not driving oscillation")
            
            # Check integral accumulation
            first_third = integral_abs[:len(integral_abs)//3]
            last_third = integral_abs[2*len(integral_abs)//3:]
            accumulation = np.mean(last_third) / np.mean(first_third) if np.mean(first_third) > 0 else 0
            print(f"  Integral accumulation: {accumulation:.2f}x")
            print()
        
        # Trajectory stability analysis
        if ref_x is not None:
            print("TRAJECTORY STABILITY ANALYSIS:")
            print("-" * 80)
            
            ref_x_std = np.std(ref_x)
            ref_x_mean = np.mean(np.abs(ref_x))
            print(f"  ref_x std: {ref_x_std:.4f} m")
            print(f"  ref_x mean abs: {ref_x_mean:.4f} m")
            
            # Check if trajectory is oscillating
            ref_x_sign_changes = np.sum(np.diff(np.sign(ref_x)) != 0)
            ref_x_oscillation_freq = ref_x_sign_changes / time_seconds[-1] if time_seconds[-1] > 0 else 0
            print(f"  ref_x oscillation frequency: {ref_x_oscillation_freq:.2f} Hz")
            
            # Correlation between trajectory and error
            if len(ref_x) == len(lateral_error):
                trajectory_error_corr = np.corrcoef(ref_x, lateral_error)[0, 1]
                print(f"  Trajectory-error correlation: {trajectory_error_corr:.3f}")
                
                if abs(trajectory_error_corr) > 0.7:
                    print(f"  ⚠️  High correlation - trajectory oscillation may be driving error oscillation")
                elif abs(trajectory_error_corr) > 0.3:
                    print(f"  ⚠️  Moderate correlation - trajectory may contribute to oscillation")
                else:
                    print(f"  ✓ Low correlation - trajectory not driving oscillation")
            print()
        
        # Heading error analysis
        if heading_error is not None:
            print("HEADING ERROR ANALYSIS:")
            print("-" * 80)
            
            heading_error_abs = np.abs(heading_error)
            heading_oscillation = np.sum(np.diff(np.sign(heading_error)) != 0)
            heading_oscillation_freq = heading_oscillation / time_seconds[-1] if time_seconds[-1] > 0 else 0
            print(f"  Heading error oscillation frequency: {heading_oscillation_freq:.2f} Hz")
            
            # Correlation with lateral error
            if len(heading_error) == len(lateral_error):
                heading_lateral_corr = np.corrcoef(heading_error, lateral_error)[0, 1]
                print(f"  Heading-lateral error correlation: {heading_lateral_corr:.3f}")
                
                if abs(heading_lateral_corr) > 0.7:
                    print(f"  ⚠️  High correlation - heading error may be driving lateral oscillation")
            print()
        
        # System response analysis
        print("SYSTEM RESPONSE ANALYSIS:")
        print("-" * 80)
        
        # Check if system is over-damped, under-damped, or critically damped
        # Look at error response to steering changes
        if len(lateral_error) > 20:
            # Find peaks in error
            error_peaks, _ = signal.find_peaks(abs_lateral_error, height=0.05, distance=5)
            
            if len(error_peaks) > 2:
                # Calculate damping ratio (simplified)
                peak_values = abs_lateral_error[error_peaks]
                if len(peak_values) > 1:
                    # Check if peaks are decreasing (damped) or constant/increasing (under-damped)
                    peak_ratio = peak_values[-1] / peak_values[0] if peak_values[0] > 0 else 0
                    
                    if peak_ratio < 0.5:
                        print(f"  ✓ Over-damped: Peaks decreasing ({peak_ratio:.2f}x)")
                    elif peak_ratio < 1.0:
                        print(f"  ⚠️  Under-damped: Peaks decreasing slowly ({peak_ratio:.2f}x)")
                    else:
                        print(f"  ⚠️  Unstable: Peaks not decreasing ({peak_ratio:.2f}x)")
        print()
        
        # Root cause hypotheses
        print("=" * 80)
        print("ROOT CAUSE HYPOTHESES (Prioritized):")
        print("=" * 80)
        
        hypotheses = []
        confidence = []
        
        # Hypothesis 1: Controller gain too high
        if total_error is not None:
            significant = (np.abs(lateral_error) > 0.05) & (np.abs(steering) > 0.02)
            if np.sum(significant) > 10:
                error_sig = lateral_error[significant]
                steering_sig = steering[significant]
                if np.std(error_sig) > 0.001:
                    effective_gain = np.std(steering_sig) / np.std(error_sig)
                    if effective_gain > 2.0:
                        hypotheses.append("Controller gain too high (effective gain > 2.0)")
                        confidence.append(0.8)
        
        # Hypothesis 2: PID integral windup
        if pid_integral is not None:
            first_third = np.abs(pid_integral[:len(pid_integral)//3])
            last_third = np.abs(pid_integral[2*len(pid_integral)//3:])
            accumulation = np.mean(last_third) / np.mean(first_third) if np.mean(first_third) > 0 else 0
            if accumulation > 1.5:
                hypotheses.append(f"PID integral windup ({accumulation:.2f}x accumulation)")
                confidence.append(0.9)
        
        # Hypothesis 3: Trajectory oscillation
        if ref_x is not None:
            ref_x_sign_changes = np.sum(np.diff(np.sign(ref_x)) != 0)
            ref_x_oscillation_freq = ref_x_sign_changes / time_seconds[-1] if time_seconds[-1] > 0 else 0
            if ref_x_oscillation_freq > 5.0:
                hypotheses.append(f"Trajectory oscillation ({ref_x_oscillation_freq:.2f} Hz)")
                confidence.append(0.7)
            
            if len(ref_x) == len(lateral_error):
                trajectory_error_corr = np.corrcoef(ref_x, lateral_error)[0, 1]
                if abs(trajectory_error_corr) > 0.7:
                    hypotheses.append("Trajectory oscillation driving error oscillation")
                    confidence.append(0.8)
        
        # Hypothesis 4: Insufficient damping
        if pid_derivative is not None:
            # Check if derivative term is active
            derivative_abs = np.abs(pid_derivative)
            if np.mean(derivative_abs) < 0.01:
                hypotheses.append("Insufficient damping (derivative term too small)")
                confidence.append(0.6)
        
        # Hypothesis 5: Delayed response
        if len(lateral_error) > 10 and len(steering) > 10:
            error_norm = (lateral_error - np.mean(lateral_error)) / (np.std(lateral_error) + 1e-10)
            steering_norm = (steering - np.mean(steering)) / (np.std(steering) + 1e-10)
            correlation = np.correlate(error_norm, steering_norm, mode='full')
            lags = np.arange(-len(lateral_error)+1, len(lateral_error))
            max_corr_idx = np.argmax(np.abs(correlation))
            max_lag = lags[max_corr_idx]
            lag_time = max_lag * dt
            if abs(lag_time) > 0.1:
                hypotheses.append(f"Delayed response (lag: {lag_time:.3f}s)")
                confidence.append(0.7)
        
        # Hypothesis 6: Reference point smoothing delay
        if ref_x is not None and len(ref_x) > 10:
            # Check if ref_x is smoothed (has less variation than it should)
            # Compare ref_x variation to what we'd expect from lane detection
            ref_x_std = np.std(ref_x)
            if ref_x_std < 0.01:  # Very low variation suggests heavy smoothing
                hypotheses.append("Reference point over-smoothed (causing delayed response)")
                confidence.append(0.6)
        
        # Sort by confidence
        if hypotheses:
            sorted_hypotheses = sorted(zip(hypotheses, confidence), key=lambda x: x[1], reverse=True)
            for i, (hyp, conf) in enumerate(sorted_hypotheses, 1):
                print(f"  {i}. [{conf:.1f}] {hyp}")
        else:
            print("  No clear hypotheses identified")
        print()
        
        # Investigation plan
        print("=" * 80)
        print("INVESTIGATION PLAN:")
        print("=" * 80)
        print()
        print("STEP 1: Verify controller gains")
        print("  - Check config: kp, ki, kd values")
        print("  - Measure effective gain (steering/error ratio)")
        print("  - Compare to theoretical gain")
        print()
        print("STEP 2: Analyze PID components separately")
        print("  - Check if P, I, or D term is driving oscillation")
        print("  - Verify integral reset/decay is working")
        print("  - Check derivative term magnitude")
        print()
        print("STEP 3: Check trajectory stability")
        print("  - Verify ref_x is stable (not oscillating)")
        print("  - Check if trajectory smoothing is causing delay")
        print("  - Verify lane detection is stable")
        print()
        print("STEP 4: System response analysis")
        print("  - Check phase relationship (steering vs error)")
        print("  - Verify response time is appropriate")
        print("  - Check for delays in pipeline")
        print()
        print("STEP 5: Test with modified parameters")
        print("  - Reduce kp to see if oscillation decreases")
        print("  - Increase kd to add more damping")
        print("  - Adjust smoothing parameters")
        print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        recordings = sorted(Path('data/recordings').glob('*.h5'), key=lambda p: p.stat().st_mtime, reverse=True)
        if recordings:
            recording_file = str(recordings[0])
            print(f"Using latest recording: {recording_file}\n")
        else:
            print("No recordings found. Usage: python tools/analyze_oscillation_root_cause.py <recording.h5>")
            sys.exit(1)
    else:
        recording_file = sys.argv[1]
    
    analyze_oscillation_root_cause(recording_file)

