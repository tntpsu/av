#!/usr/bin/env python3
"""
Comprehensive analysis tool for AV stack recordings.
Combines ALL analysis functionality into a single unified tool.

This is the PRIMARY analysis tool - use this for all recording analysis.

Includes:
- Basic metrics and statistics
- Control system analysis (PID, steering, speed)
- Oscillation and convergence analysis
- Root cause diagnostics (perception, trajectory, controller)
- Prioritized recommendations

Replaces all other analysis tools:
- analyze_recording.py
- analyze_recording_advanced.py
- analyze_lateral_error_convergence.py
- analyze_oscillation_root_cause.py
- analyze_heading_opposition.py
- automated_diagnostic_framework.py
- automated_root_cause_diagnostic.py
- analyze_root_cause.py

Usage:
    python tools/analyze_recording_comprehensive.py [recording_file]
    python tools/analyze_recording_comprehensive.py --latest
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import h5py
import numpy as np
from typing import Optional, Dict, List, Tuple
import argparse
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy import stats

# Note: analyze_recording_advanced.py was removed - functions implemented inline below


class ComprehensiveAnalyzer:
    """Comprehensive analysis of AV stack recordings."""
    
    def __init__(self, recording_path: Path):
        """Initialize analyzer with recording path."""
        self.recording_path = recording_path
        self.data = {}
        self.diagnostics = []
        
    def load_data(self):
        """Load all data from recording."""
        with h5py.File(self.recording_path, 'r') as f:
            # Vehicle data
            if 'vehicle/position' in f:
                self.data['position'] = np.array(f['vehicle/position'][:])
                self.data['speed'] = np.array(f['vehicle/speed'][:])
                self.data['rotation'] = np.array(f['vehicle/rotation'][:])
                self.data['timestamps'] = np.array(f['vehicle/timestamps'][:])
            else:
                print("⚠️  No vehicle data found")
                return False
            
            # Control data - convert to numpy arrays immediately
            self.data['steering'] = np.array(f['control/steering'][:])
            self.data['throttle'] = np.array(f['control/throttle'][:]) if 'control/throttle' in f else None
            self.data['brake'] = np.array(f['control/brake'][:]) if 'control/brake' in f else None
            self.data['lateral_error'] = np.array(f['control/lateral_error'][:]) if 'control/lateral_error' in f else None
            self.data['heading_error'] = np.array(f['control/heading_error'][:]) if 'control/heading_error' in f else None
            self.data['total_error'] = np.array(f['control/total_error'][:]) if 'control/total_error' in f else None
            self.data['pid_integral'] = np.array(f['control/pid_integral'][:]) if 'control/pid_integral' in f else None
            self.data['pid_derivative'] = np.array(f['control/pid_derivative'][:]) if 'control/pid_derivative' in f else None
            
            # Trajectory data
            if 'trajectory/reference_point_x' in f:
                self.data['ref_x'] = np.array(f['trajectory/reference_point_x'][:])
                self.data['ref_y'] = np.array(f['trajectory/reference_point_y'][:]) if 'trajectory/reference_point_y' in f else None
                self.data['ref_heading'] = np.array(f['trajectory/reference_point_heading'][:]) if 'trajectory/reference_point_heading' in f else None
            else:
                self.data['ref_x'] = None
                self.data['ref_y'] = None
                self.data['ref_heading'] = None
            
            # Perception data
            self.data['left_lane_x'] = np.array(f['perception/left_lane_x'][:]) if 'perception/left_lane_x' in f else None
            self.data['right_lane_x'] = np.array(f['perception/right_lane_x'][:]) if 'perception/right_lane_x' in f else None
            
            # Calculate time axis
            if len(self.data['timestamps']) > 0:
                self.data['time'] = self.data['timestamps'] - self.data['timestamps'][0]
            else:
                self.data['time'] = np.arange(len(self.data['position'])) / 30.0
            
            self.data['dt'] = np.mean(np.diff(self.data['time'])) if len(self.data['time']) > 1 else 1.0/30.0
            
            return True
    
    def run_comprehensive_analysis(self):
        """Run all analysis sections."""
        print("=" * 80)
        print("COMPREHENSIVE RECORDING ANALYSIS")
        print(f"Recording: {self.recording_path.name}")
        print("=" * 80)
        print()
        
        if not self.load_data():
            return
        
        # 1. Basic Analysis (from analyze_recording_advanced.py)
        print("=" * 80)
        print("SECTION 1: BASIC ANALYSIS")
        print("=" * 80)
        print()
        self._analyze_basic()
        
        # 2. Control System Analysis
        print("=" * 80)
        print("SECTION 2: CONTROL SYSTEM ANALYSIS")
        print("=" * 80)
        print()
        self._analyze_control_system()
        
        # 3. Oscillation Analysis
        print("=" * 80)
        print("SECTION 3: OSCILLATION ANALYSIS")
        print("=" * 80)
        print()
        self._analyze_oscillation()
        
        # 4. Convergence Analysis
        print("=" * 80)
        print("SECTION 4: CONVERGENCE ANALYSIS")
        print("=" * 80)
        print()
        self._analyze_convergence()
        
        # 5. Root Cause Diagnostics
        print("=" * 80)
        print("SECTION 5: ROOT CAUSE DIAGNOSTICS")
        print("=" * 80)
        print()
        self._analyze_root_cause()
        
        # 6. Trajectory Analysis
        print("=" * 80)
        print("SECTION 5: TRAJECTORY ANALYSIS")
        print("=" * 80)
        print()
        self._analyze_trajectory()
        
        # 6. Automated Diagnostics
        print("=" * 80)
        print("SECTION 6: AUTOMATED DIAGNOSTICS")
        print("=" * 80)
        print()
        self._run_automated_diagnostics()
        
        # 7. Recommendations
        print("=" * 80)
        print("SECTION 7: RECOMMENDATIONS")
        print("=" * 80)
        print()
        self._print_recommendations()
    
    def _analyze_basic(self):
        """Basic analysis - basic metrics and statistics."""
        print("BASIC METRICS:")
        print("-" * 80)
        
        lateral_error = self.data.get('lateral_error')
        steering = self.data.get('steering')
        speed = self.data.get('speed')
        
        if lateral_error is not None:
            abs_error = np.abs(lateral_error)
            print(f"  Lateral Error:")
            print(f"    Mean: {np.mean(abs_error):.4f} m")
            print(f"    Std:  {np.std(abs_error):.4f} m")
            print(f"    Max:  {np.max(abs_error):.4f} m")
            print(f"    RMSE: {np.sqrt(np.mean(lateral_error**2)):.4f} m")
        
        if steering is not None:
            print(f"  Steering:")
            print(f"    Mean: {np.mean(np.abs(steering)):.4f}")
            print(f"    Std:  {np.std(steering):.4f}")
            print(f"    Max:  {np.max(np.abs(steering)):.4f}")
        
        if speed is not None:
            print(f"  Speed:")
            print(f"    Mean: {np.mean(speed):.2f} m/s")
            print(f"    Std:  {np.std(speed):.2f} m/s")
            print(f"    Max:  {np.max(speed):.2f} m/s")
        
        print()
    
    def _analyze_control_system(self):
        """Analyze control system performance."""
        lateral_error = self.data.get('lateral_error')
        steering = self.data.get('steering')
        heading_error = self.data.get('heading_error')
        total_error = self.data.get('total_error')
        pid_integral = self.data.get('pid_integral')
        pid_derivative = self.data.get('pid_derivative')
        
        if lateral_error is None:
            print("⚠️  No lateral error data")
            return
        
        print("CONTROL PERFORMANCE:")
        print("-" * 80)
        abs_lateral_error = np.abs(lateral_error)
        print(f"  Mean lateral error: {np.mean(abs_lateral_error):.4f} m")
        print(f"  Median lateral error: {np.median(abs_lateral_error):.4f} m")
        print(f"  Std lateral error: {np.std(abs_lateral_error):.4f} m")
        print(f"  Max lateral error: {np.max(abs_lateral_error):.4f} m")
        print()
        
        # PID Analysis
        if pid_integral is not None:
            print("PID INTEGRAL:")
            print("-" * 80)
            abs_integral = np.abs(pid_integral)
            print(f"  Mean integral: {np.mean(abs_integral):.4f}")
            print(f"  Max integral: {np.max(abs_integral):.4f}")
            
            # Check accumulation (consistent calculation across all sections)
            n = len(abs_integral)
            first_third = abs_integral[:n//3]
            last_third = abs_integral[2*n//3:]
            accumulation = np.mean(last_third) / np.mean(first_third) if np.mean(first_third) > 0 else 0
            print(f"  Accumulation ratio: {accumulation:.2f}x")
            if accumulation > 1.5:
                print(f"  ⚠️  INTEGRAL ACCUMULATING: {accumulation:.2f}x (should be < 1.5x)")
                self.diagnostics.append({
                    'issue': 'PID Integral Windup',
                    'severity': 'high',
                    'value': accumulation,
                    'fix': 'Lower integral decay threshold (0.01), increase decay factor (0.90), more frequent resets'
                })
            print()
        
        # Steering direction
        if steering is not None:
            print("STEERING DIRECTION:")
            print("-" * 80)
            significant = (abs_lateral_error > 0.1) & (np.abs(steering) > 0.05)
            if np.sum(significant) > 10:
                correct = ((lateral_error[significant] > 0) & (steering[significant] > 0)) | \
                         ((lateral_error[significant] < 0) & (steering[significant] < 0))
                correct_pct = 100 * np.sum(correct) / np.sum(significant)
                print(f"  Correct steering direction: {correct_pct:.1f}%")
                if correct_pct < 70:
                    print(f"  ⚠️  LOW CORRECT DIRECTION: {correct_pct:.1f}% (should be > 70%)")
                    self.diagnostics.append({
                        'issue': 'Steering Direction Errors',
                        'severity': 'critical',
                        'value': correct_pct,
                        'fix': 'Check coordinate system transformations, verify lateral_error calculation'
                    })
            print()
    
    def _analyze_oscillation(self):
        """Analyze oscillation patterns."""
        lateral_error = self.data.get('lateral_error')
        if lateral_error is None:
            print("⚠️  No lateral error data")
            return
        
        time = self.data['time']
        dt = self.data['dt']
        
        print("OSCILLATION CHARACTERISTICS:")
        print("-" * 80)
        
        # Frequency analysis (consistent calculation)
        sign_changes = np.sum(np.diff(np.sign(lateral_error)) != 0)
        total_time = time[-1] if len(time) > 0 and time[-1] > 0 else len(lateral_error) * self.data['dt']
        oscillation_freq = sign_changes / total_time if total_time > 0 else 0
        print(f"  Sign change frequency: {oscillation_freq:.2f} Hz")
        
        # FFT analysis
        if len(lateral_error) > 10:
            error_centered = lateral_error - np.mean(lateral_error)
            fft_vals = fft(error_centered)
            fft_freqs = fftfreq(len(error_centered), dt)
            positive_freqs = fft_freqs[:len(fft_freqs)//2]
            positive_fft = np.abs(fft_vals[:len(fft_vals)//2])
            dominant_idx = np.argmax(positive_fft[1:]) + 1
            dominant_freq = positive_freqs[dominant_idx]
            print(f"  Dominant frequency (FFT): {dominant_freq:.2f} Hz")
            
            if oscillation_freq > 5.0:
                print(f"  ⚠️  HIGH OSCILLATION: {oscillation_freq:.2f} Hz (should be < 5 Hz)")
                self.diagnostics.append({
                    'issue': 'High Oscillation',
                    'severity': 'high',
                    'value': oscillation_freq,
                    'fix': 'Reduce kp gain, increase kd gain, fix PID integral windup'
                })
        print()
    
    def _analyze_convergence(self):
        """Analyze error convergence."""
        lateral_error = self.data.get('lateral_error')
        if lateral_error is None:
            print("⚠️  No lateral error data")
            return
        
        abs_error = np.abs(lateral_error)
        
        print("CONVERGENCE ANALYSIS:")
        print("-" * 80)
        
        # Split into quarters
        quarter_len = len(lateral_error) // 4
        q1 = abs_error[:quarter_len]
        q4 = abs_error[3*quarter_len:]
        q1_to_q4 = np.mean(q4) / np.mean(q1) if np.mean(q1) > 0 else 0
        
        print(f"  Q1 mean error: {np.mean(q1):.4f} m")
        print(f"  Q4 mean error: {np.mean(q4):.4f} m")
        print(f"  Q1→Q4 change: {q1_to_q4:.2f}x")
        
        if q1_to_q4 > 1.1:
            print(f"  ⚠️  ERROR INCREASING: {q1_to_q4:.2f}x (should decrease)")
            self.diagnostics.append({
                'issue': 'Error Not Converging',
                'severity': 'high',
                'value': q1_to_q4,
                'fix': 'Fix PID integral windup, check for trajectory drift'
            })
        elif q1_to_q4 < 0.9:
            print(f"  ✓ Error decreasing: {q1_to_q4:.2f}x")
        else:
            print(f"  ⚠️  Error stable (not converging)")
        print()
    
    def _analyze_root_cause(self):
        """Analyze root causes by component (perception, trajectory, controller)."""
        left_lane_x = self.data.get('left_lane_x')
        right_lane_x = self.data.get('right_lane_x')
        ref_x = self.data.get('ref_x')
        lateral_error = self.data.get('lateral_error')
        steering = self.data.get('steering')
        pid_integral = self.data.get('pid_integral')
        
        root_cause_scores = {}
        
        # 1. PERCEPTION DIAGNOSIS
        print("PERCEPTION (Lane Detection):")
        print("-" * 80)
        if left_lane_x is not None and right_lane_x is not None:
            lane_center = (left_lane_x + right_lane_x) / 2.0
            lane_width = right_lane_x - left_lane_x
            
            lane_center_std = np.std(lane_center)
            lane_width_mean = np.mean(lane_width)
            lane_width_std = np.std(lane_width)
            
            print(f"  Lane center std: {lane_center_std:.4f}m")
            print(f"  Lane width: {lane_width_mean:.2f}m (expected ~3.5m)")
            print(f"  Lane width std: {lane_width_std:.4f}m")
            
            perception_score = 0.0
            if lane_center_std > 0.1:
                print(f"  ⚠️  Lane center unstable (std={lane_center_std:.4f}m)")
                perception_score += 0.5
            if abs(lane_width_mean - 3.5) > 0.5:
                print(f"  ⚠️  Lane width incorrect ({lane_width_mean:.2f}m, expected ~3.5m)")
                perception_score += 0.5
            if lane_width_std > 0.3:
                print(f"  ⚠️  Lane width inconsistent (std={lane_width_std:.4f}m)")
                perception_score += 0.3
            
            root_cause_scores['Perception'] = min(1.0, perception_score)
        else:
            print("  ⚠️  No perception data")
            root_cause_scores['Perception'] = 0.5
        print()
        
        # 2. TRAJECTORY DIAGNOSIS
        print("TRAJECTORY:")
        print("-" * 80)
        if ref_x is not None:
            ref_x_std = np.std(ref_x)
            ref_x_mean = np.mean(ref_x)
            ref_x_sign_changes = np.sum(np.diff(np.sign(ref_x)) != 0)
            ref_x_osc = ref_x_sign_changes / self.data['time'][-1] if self.data['time'][-1] > 0 else 0
            
            print(f"  ref_x std: {ref_x_std:.4f}m")
            print(f"  ref_x mean: {ref_x_mean:.4f}m")
            print(f"  Oscillation: {ref_x_osc:.2f} Hz")
            
            trajectory_score = 0.0
            if ref_x_osc > 3.0:
                print(f"  ⚠️  High oscillation ({ref_x_osc:.2f} Hz)")
                trajectory_score += 0.5
            if abs(ref_x_mean) > 0.1:
                print(f"  ⚠️  Biased ({ref_x_mean:.4f}m from center)")
                trajectory_score += 0.3
            if ref_x_std > 0.2:
                print(f"  ⚠️  High variance (std={ref_x_std:.4f}m)")
                trajectory_score += 0.2
            
            # Check offset from lane center
            if left_lane_x is not None and right_lane_x is not None:
                lane_center = (left_lane_x + right_lane_x) / 2.0
                trajectory_offset = ref_x - lane_center
                mean_offset = np.mean(np.abs(trajectory_offset))
                print(f"  Offset from lane center: {mean_offset:.4f}m")
                if mean_offset > 0.1:
                    print(f"  ⚠️  Not centered ({mean_offset:.4f}m)")
                    trajectory_score += 0.3
            
            root_cause_scores['Trajectory'] = min(1.0, trajectory_score)
        else:
            print("  ⚠️  No trajectory data")
            root_cause_scores['Trajectory'] = 0.5
        print()
        
        # 3. CONTROLLER DIAGNOSIS
        print("CONTROLLER:")
        print("-" * 80)
        if lateral_error is not None and ref_x is not None and steering is not None:
            # Correlation with trajectory
            correlation = np.corrcoef(ref_x, lateral_error)[0, 1] if len(ref_x) > 1 else 0.0
            print(f"  Correlation with trajectory: {correlation:.3f} (1.0 = perfect tracking)")
            
            # Error oscillation (consistent with oscillation analysis)
            error_sign_changes = np.sum(np.diff(np.sign(lateral_error)) != 0)
            total_time = self.data['time'][-1] if len(self.data['time']) > 0 and self.data['time'][-1] > 0 else len(lateral_error) * self.data['dt']
            error_osc = error_sign_changes / total_time if total_time > 0 else 0
            print(f"  Error oscillation: {error_osc:.2f} Hz")
            
            # Integral accumulation (consistent with control system analysis)
            if pid_integral is not None:
                abs_integral = np.abs(pid_integral)
                n = len(abs_integral)
                first_third = abs_integral[:n//3]
                last_third = abs_integral[2*n//3:]
                accumulation = np.mean(last_third) / np.mean(first_third) if np.mean(first_third) > 0 else 0
                print(f"  Integral accumulation: {accumulation:.2f}x")
            else:
                accumulation = 0
            
            controller_score = 0.0
            if correlation < 0.5:
                print(f"  ⚠️  Low correlation - not following trajectory")
                controller_score += 0.5
            else:
                # High correlation means controller is working - issue is upstream
                controller_score = 0.1  # Low score = not root cause
            if error_osc > 10.0:
                print(f"  ⚠️  High error oscillation ({error_osc:.2f} Hz)")
                controller_score += 0.3
            if accumulation > 1.5:
                print(f"  ⚠️  Integral accumulating ({accumulation:.2f}x)")
                controller_score += 0.2
            
            root_cause_scores['Controller'] = min(1.0, controller_score)
        else:
            print("  ⚠️  Missing controller data")
            root_cause_scores['Controller'] = 0.5
        print()
        
        # ROOT CAUSE PRIORITY
        print("ROOT CAUSE PRIORITY:")
        print("-" * 80)
        sorted_scores = sorted(root_cause_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (component, score) in enumerate(sorted_scores, 1):
            icon = "🔴" if score > 0.7 else "🟠" if score > 0.4 else "🟡" if score > 0.1 else "✅"
            print(f"  {i}. {icon} {component}: {score:.2f} (higher = more likely root cause)")
        print()
        
        # RECOMMENDATION
        top_component = sorted_scores[0][0]
        print(f"💡 RECOMMENDATION: Fix {top_component} first (root cause score: {sorted_scores[0][1]:.2f})")
        if top_component == "Perception":
            print("   → Fix coordinate conversion, improve lane detection stability")
        elif top_component == "Trajectory":
            print("   → Reduce smoothing, improve bias correction")
        elif top_component == "Controller":
            print("   → Tune PID gains, improve integral reset/decay")
        print()
    
    def _analyze_trajectory(self):
        """Analyze trajectory stability."""
        ref_x = self.data.get('ref_x')
        left_lane_x = self.data.get('left_lane_x')
        right_lane_x = self.data.get('right_lane_x')
        
        if ref_x is None:
            print("⚠️  No trajectory data")
            return
        
        print("TRAJECTORY STABILITY:")
        print("-" * 80)
        ref_x_std = np.std(ref_x)
        print(f"  ref_x std: {ref_x_std:.4f} m")
        
        # Check oscillation
        ref_x_sign_changes = np.sum(np.diff(np.sign(ref_x)) != 0)
        ref_x_oscillation_freq = ref_x_sign_changes / self.data['time'][-1] if self.data['time'][-1] > 0 else 0
        print(f"  ref_x oscillation frequency: {ref_x_oscillation_freq:.2f} Hz")
        
        if ref_x_oscillation_freq > 3.0:
            print(f"  ⚠️  TRAJECTORY OSCILLATING: {ref_x_oscillation_freq:.2f} Hz")
            self.diagnostics.append({
                'issue': 'Trajectory Oscillation',
                'severity': 'medium',
                'value': ref_x_oscillation_freq,
                'fix': 'Increase lane smoothing, reduce reference point smoothing'
            })
        
        # Check centering
        if left_lane_x is not None and right_lane_x is not None:
            lane_center = (left_lane_x + right_lane_x) / 2.0
            trajectory_offset = ref_x - lane_center
            mean_offset = np.mean(np.abs(trajectory_offset))
            print(f"  Trajectory offset from lane center: {mean_offset:.4f} m")
            
            if mean_offset > 0.2:
                print(f"  ⚠️  TRAJECTORY NOT CENTERED: {mean_offset:.4f}m")
                self.diagnostics.append({
                    'issue': 'Trajectory Not Centered',
                    'severity': 'medium',
                    'value': mean_offset,
                    'fix': 'Improve bias correction logic, check lane detection'
                })
        print()
    
    def _run_automated_diagnostics(self):
        """Run automated diagnostic checks."""
        if not self.diagnostics:
            print("✓ No issues detected")
            return
        
        print("ISSUES DETECTED:")
        print("-" * 80)
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        self.diagnostics.sort(key=lambda x: severity_order.get(x['severity'], 4))
        
        for i, diag in enumerate(self.diagnostics, 1):
            severity_icon = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}.get(diag['severity'], '⚪')
            print(f"  {i}. {severity_icon} [{diag['severity'].upper()}] {diag['issue']}")
            print(f"     Value: {diag['value']}")
            print(f"     Fix: {diag['fix']}")
            print()
    
    def _print_recommendations(self):
        """Print prioritized recommendations."""
        if not self.diagnostics:
            print("✓ System appears to be operating well")
            return
        
        # Group by severity
        critical = [d for d in self.diagnostics if d['severity'] == 'critical']
        high = [d for d in self.diagnostics if d['severity'] == 'high']
        medium = [d for d in self.diagnostics if d['severity'] == 'medium']
        
        if critical:
            print("PRIORITY 1 (CRITICAL):")
            for diag in critical:
                print(f"  - {diag['issue']}: {diag['fix']}")
            print()
        
        if high:
            print("PRIORITY 2 (HIGH):")
            for diag in high:
                print(f"  - {diag['issue']}: {diag['fix']}")
            print()
        
        if medium:
            print("PRIORITY 3 (MEDIUM):")
            for diag in medium:
                print(f"  - {diag['issue']}: {diag['fix']}")
            print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Comprehensive analysis of AV stack recordings')
    parser.add_argument('recording', nargs='?', type=Path, help='Path to recording file (.h5)')
    parser.add_argument('--latest', action='store_true', help='Use latest recording')
    
    args = parser.parse_args()
    
    if args.latest or args.recording is None:
        recordings = sorted(Path('data/recordings').glob('*.h5'), key=lambda p: p.stat().st_mtime, reverse=True)
        if recordings:
            recording_path = recordings[0]
            print(f"Using latest recording: {recording_path.name}\n")
        else:
            print("No recordings found")
            sys.exit(1)
    else:
        recording_path = args.recording
    
    if not recording_path.exists():
        print(f"Recording not found: {recording_path}")
        sys.exit(1)
    
    analyzer = ComprehensiveAnalyzer(recording_path)
    analyzer.run_comprehensive_analysis()


if __name__ == '__main__':
    main()

