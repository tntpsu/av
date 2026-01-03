#!/usr/bin/env python3
"""
Analyze when and why heading error opposes lateral error.
This helps understand steering direction errors.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import h5py
import numpy as np
import argparse
from scipy import stats


def extract_heading_from_quaternion(rotation: np.ndarray) -> float:
    """Extract heading (yaw) from quaternion rotation."""
    x, y, z, w = rotation[0], rotation[1], rotation[2], rotation[3]
    siny_cosp = 2.0 * (w * y + z * x)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw


def analyze_heading_opposition(recording_path: Path):
    """Analyze when heading error opposes lateral error."""
    print("=" * 80)
    print("HEADING ERROR OPPOSITION ANALYSIS")
    print("=" * 80)
    print()
    print(f"Analyzing: {recording_path.name}")
    print()
    
    with h5py.File(recording_path, 'r') as f:
        # Load data
        lat_err = f['control/lateral_error'][:]
        heading_err = f['control/heading_error'][:]
        total_err = f['control/total_error'][:]
        steering = f['control/steering'][:]
        ref_x = f['trajectory/reference_point_x'][:]
        ref_heading = f['trajectory/reference_point_heading'][:]
        
        # Check if we have lane data
        has_lane_data = 'perception/left_lane_x' in f and 'perception/right_lane_x' in f
        
        if has_lane_data:
            left_lane_x = f['perception/left_lane_x'][:]
            right_lane_x = f['perception/right_lane_x'][:]
            lane_center_x = (left_lane_x + right_lane_x) / 2.0
        else:
            left_lane_x = None
            right_lane_x = None
            lane_center_x = None
        
        # Vehicle state
        positions = f['vehicle/position'][:]
        rotations = f['vehicle/rotation'][:]
        speeds = f['vehicle/speed'][:]
        
        # Extract vehicle headings
        vehicle_headings = []
        for rot in rotations:
            heading = extract_heading_from_quaternion(rot)
            vehicle_headings.append(heading)
        vehicle_headings = np.array(vehicle_headings)
        
        num_frames = len(lat_err)
        
        print("1. OVERALL STATISTICS")
        print("-" * 80)
        print()
        
        # Count when heading opposes lateral
        heading_opposes_lat = (heading_err > 0) != (lat_err > 0)
        num_opposes = np.sum(heading_opposes_lat)
        pct_opposes = 100 * num_opposes / num_frames
        
        print(f"Frames where heading OPPOSES lateral: {num_opposes}/{num_frames} ({pct_opposes:.1f}%)")
        print(f"Frames where heading MATCHES lateral: {num_frames - num_opposes}/{num_frames} ({100 - pct_opposes:.1f}%)")
        print()
        
        # Analyze steering direction errors
        significant = (np.abs(lat_err) > 0.1) & (np.abs(steering) > 0.05)
        sig_idx = np.where(significant)[0]
        
        if len(sig_idx) > 0:
            lat_err_sig = lat_err[sig_idx]
            heading_err_sig = heading_err[sig_idx]
            steering_sig = steering[sig_idx]
            total_err_sig = total_err[sig_idx]
            heading_opposes_sig = heading_opposes_lat[sig_idx]
            
            # Wrong direction: opposite sign
            wrong_dir = ((lat_err_sig > 0) & (steering_sig < 0)) | \
                       ((lat_err_sig < 0) & (steering_sig > 0))
            
            wrong_when_opposes = np.sum(wrong_dir & heading_opposes_sig)
            wrong_when_matches = np.sum(wrong_dir & ~heading_opposes_sig)
            total_when_opposes = np.sum(heading_opposes_sig)
            total_when_matches = np.sum(~heading_opposes_sig)
            
            print(f"Significant errors: {len(sig_idx)} frames")
            print(f"Wrong steering when heading OPPOSES: {wrong_when_opposes}/{total_when_opposes} ({100*wrong_when_opposes/total_when_opposes if total_when_opposes > 0 else 0:.1f}%)")
            print(f"Wrong steering when heading MATCHES: {wrong_when_matches}/{total_when_matches} ({100*wrong_when_matches/total_when_matches if total_when_matches > 0 else 0:.1f}%)")
            print()
        
        print("2. WHEN DOES HEADING OPPOSE LATERAL?")
        print("-" * 80)
        print()
        
        # Analyze conditions when heading opposes lateral
        opposes_mask = heading_opposes_lat
        matches_mask = ~heading_opposes_lat
        
        print("When Heading OPPOSES Lateral:")
        print(f"  Mean lateral error: {np.mean(lat_err[opposes_mask]):.4f}m")
        print(f"  Mean heading error: {np.mean(heading_err[opposes_mask]):.4f} rad ({np.degrees(np.mean(heading_err[opposes_mask])):.2f}°)")
        print(f"  Mean ref_x: {np.mean(ref_x[opposes_mask]):.4f}m")
        print(f"  Mean ref_heading: {np.mean(ref_heading[opposes_mask]):.4f} rad ({np.degrees(np.mean(ref_heading[opposes_mask])):.2f}°)")
        print(f"  Mean vehicle heading: {np.mean(vehicle_headings[opposes_mask]):.4f} rad ({np.degrees(np.mean(vehicle_headings[opposes_mask])):.2f}°)")
        print(f"  Mean speed: {np.mean(speeds[opposes_mask]):.2f} m/s")
        print()
        
        print("When Heading MATCHES Lateral:")
        print(f"  Mean lateral error: {np.mean(lat_err[matches_mask]):.4f}m")
        print(f"  Mean heading error: {np.mean(heading_err[matches_mask]):.4f} rad ({np.degrees(np.mean(heading_err[matches_mask])):.2f}°)")
        print(f"  Mean ref_x: {np.mean(ref_x[matches_mask]):.4f}m")
        print(f"  Mean ref_heading: {np.mean(ref_heading[matches_mask]):.4f} rad ({np.degrees(np.mean(ref_heading[matches_mask])):.2f}°)")
        print(f"  Mean vehicle heading: {np.mean(vehicle_headings[matches_mask]):.4f} rad ({np.degrees(np.mean(vehicle_headings[matches_mask])):.2f}°)")
        print(f"  Mean speed: {np.mean(speeds[matches_mask]):.2f} m/s")
        print()
        
        print("3. HEADING ERROR CALCULATION ANALYSIS")
        print("-" * 80)
        print()
        
        # heading_error = ref_heading - vehicle_heading
        calculated_heading_err = ref_heading - vehicle_headings
        
        print(f"Calculated heading_error (ref_heading - vehicle_heading):")
        print(f"  Mean: {np.mean(calculated_heading_err):.4f} rad ({np.degrees(np.mean(calculated_heading_err)):.2f}°)")
        print(f"  Std: {np.std(calculated_heading_err):.4f} rad")
        print()
        
        print(f"Recorded heading_error:")
        print(f"  Mean: {np.mean(heading_err):.4f} rad ({np.degrees(np.mean(heading_err)):.2f}°)")
        print(f"  Std: {np.std(heading_err):.4f} rad")
        print()
        
        # Check if they match
        diff = np.abs(calculated_heading_err - heading_err)
        print(f"Difference (calculated vs recorded):")
        print(f"  Mean: {np.mean(diff):.4f} rad ({np.degrees(np.mean(diff)):.2f}°)")
        print(f"  Max: {np.max(diff):.4f} rad ({np.degrees(np.max(diff)):.2f}°)")
        if np.mean(diff) > 0.01:
            print(f"  ⚠️  Large difference - heading_error calculation may be wrong!")
        print()
        
        print("4. REFERENCE POINT ANALYSIS")
        print("-" * 80)
        print()
        
        print(f"Reference Point (ref_x):")
        print(f"  Mean: {np.mean(ref_x):.4f}m")
        print(f"  Std: {np.std(ref_x):.4f}m")
        print(f"  Range: [{np.min(ref_x):.4f}, {np.max(ref_x):.4f}]m")
        print()
        
        if has_lane_data:
            print(f"Lane Center (at lookahead):")
            print(f"  Mean: {np.mean(lane_center_x):.4f}m")
            print(f"  Std: {np.std(lane_center_x):.4f}m")
            print()
            
            center_diff = ref_x - lane_center_x
            print(f"Trajectory vs Lane Center:")
            print(f"  Mean difference: {np.mean(center_diff):.4f}m")
            print(f"  Std difference: {np.std(center_diff):.4f}m")
            print()
            
            # Check if bias correlates with heading opposition
            opposes_center_diff = np.mean(center_diff[opposes_mask])
            matches_center_diff = np.mean(center_diff[matches_mask])
            
            print(f"Center difference when heading OPPOSES: {opposes_center_diff:.4f}m")
            print(f"Center difference when heading MATCHES: {matches_center_diff:.4f}m")
            print()
        
        print(f"Reference Heading (ref_heading):")
        print(f"  Mean: {np.mean(ref_heading):.4f} rad ({np.degrees(np.mean(ref_heading)):.2f}°)")
        print(f"  Std: {np.std(ref_heading):.4f} rad")
        print(f"  Range: [{np.min(ref_heading):.4f}, {np.max(ref_heading):.4f}] rad")
        print()
        
        print("5. TEMPORAL ANALYSIS")
        print("-" * 80)
        print()
        
        # Analyze by time segments
        thirds = num_frames // 3
        segments = [
            (0, thirds, 'First third'),
            (thirds, 2*thirds, 'Middle third'),
            (2*thirds, num_frames, 'Last third')
        ]
        
        for start, end, label in segments:
            seg_opposes = np.sum(heading_opposes_lat[start:end])
            seg_total = end - start
            pct = 100 * seg_opposes / seg_total if seg_total > 0 else 0
            
            print(f"{label}: {seg_opposes}/{seg_total} ({pct:.1f}%) oppose")
        
        print()
        
        print("6. CORRELATIONS")
        print("-" * 80)
        print()
        
        # Correlation between ref_x and heading opposition
        ref_x_corr = np.corrcoef(ref_x, heading_opposes_lat.astype(float))[0, 1]
        print(f"ref_x ↔ Heading Opposition: {ref_x_corr:.3f}")
        
        # Correlation between ref_heading and heading opposition
        ref_heading_corr = np.corrcoef(ref_heading, heading_opposes_lat.astype(float))[0, 1]
        print(f"ref_heading ↔ Heading Opposition: {ref_heading_corr:.3f}")
        
        # Correlation between speed and heading opposition
        speed_corr = np.corrcoef(speeds, heading_opposes_lat.astype(float))[0, 1]
        print(f"Speed ↔ Heading Opposition: {speed_corr:.3f}")
        
        # Correlation between lateral error magnitude and heading opposition
        lat_err_abs_corr = np.corrcoef(np.abs(lat_err), heading_opposes_lat.astype(float))[0, 1]
        print(f"|Lateral Error| ↔ Heading Opposition: {lat_err_abs_corr:.3f}")
        
        print()
        
        print("7. ROOT CAUSE HYPOTHESES")
        print("-" * 80)
        print()
        
        hypotheses = []
        
        # Hypothesis 1: ref_heading is wrong
        if abs(np.mean(ref_heading)) > 0.1:  # > 5.7°
            hypotheses.append(f"⚠️  ref_heading has persistent bias ({np.degrees(np.mean(ref_heading)):.2f}°) - may cause heading error to oppose lateral")
        
        # Hypothesis 2: ref_x bias causes heading error
        if has_lane_data and abs(np.mean(center_diff)) > 0.05:
            hypotheses.append(f"⚠️  Trajectory not centered ({np.mean(center_diff):.4f}m offset) - may cause heading error to oppose lateral")
        
        # Hypothesis 3: Vehicle heading is wrong
        if abs(np.mean(vehicle_headings)) > 0.1:
            hypotheses.append(f"⚠️  Vehicle heading has persistent bias ({np.degrees(np.mean(vehicle_headings)):.2f}°) - may cause heading error calculation issues")
        
        # Hypothesis 4: Heading error calculation is wrong
        if np.mean(diff) > 0.01:
            hypotheses.append(f"⚠️  Heading error calculation mismatch ({np.degrees(np.mean(diff)):.2f}° difference) - calculation may be wrong")
        
        if len(hypotheses) > 0:
            for i, hyp in enumerate(hypotheses, 1):
                print(f"{i}. {hyp}")
        else:
            print("✅ No obvious root causes identified")
        
        print()
        print("8. RECOMMENDATIONS")
        print("-" * 80)
        print()
        
        recommendations = []
        
        if abs(np.mean(ref_heading)) > 0.1:
            recommendations.append("Fix ref_heading calculation - persistent bias detected")
        
        if has_lane_data and abs(np.mean(center_diff)) > 0.05:
            recommendations.append("Fix trajectory centering - trajectory not between lanes")
        
        if np.mean(diff) > 0.01:
            recommendations.append("Verify heading_error calculation - mismatch detected")
        
        if pct_opposes > 50:
            recommendations.append("Investigate why heading opposes lateral >50% of time")
        
        if len(recommendations) > 0:
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
        else:
            print("✅ No immediate recommendations")
        
        print()


def main():
    parser = argparse.ArgumentParser(description='Analyze heading error opposition')
    parser.add_argument('recording', type=str, nargs='?', default=None,
                       help='Path to recording file (default: latest)')
    parser.add_argument('--latest', action='store_true',
                       help='Analyze latest recording')
    
    args = parser.parse_args()
    
    recordings_dir = Path('data/recordings')
    
    if args.recording:
        recording_path = Path(args.recording)
    elif args.latest or args.recording is None:
        recordings = sorted(recordings_dir.glob('*.h5'), 
                          key=lambda p: p.stat().st_mtime, reverse=True)
        if not recordings:
            print('❌ No recordings found in data/recordings/')
            return
        recording_path = recordings[0]
    else:
        print('❌ Please specify a recording file or use --latest')
        return
    
    if not recording_path.exists():
        print(f'❌ Recording not found: {recording_path}')
        return
    
    analyze_heading_opposition(recording_path)


if __name__ == '__main__':
    main()

