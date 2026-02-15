#!/usr/bin/env python3
"""
Analyze curvature fix: Compare path curvature to ground truth and check feedforward activation.
Stops analysis at out-of-bounds point.
"""

import sys
from pathlib import Path
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def analyze_curvature_fix(recording_path: str):
    """Analyze if curvature fix improved path curvature values."""
    print(f"Analyzing: {recording_path}\n")
    
    with h5py.File(recording_path, 'r') as f:
        # Find out-of-bounds point
        out_of_bounds_frame = None
        if 'control/lateral_error' in f:
            lat_err = f['control/lateral_error'][:]
            for i in range(len(lat_err)):
                if abs(lat_err[i]) > 1.5:  # Out of bounds threshold
                    out_of_bounds_frame = i
                    break
        
        if out_of_bounds_frame:
            end_frame = out_of_bounds_frame
            print(f"⚠️  Out of bounds at frame {out_of_bounds_frame}")
            print(f"Analyzing frames 0-{end_frame} (stopping before out-of-bounds)\n")
        else:
            end_frame = len(f['control/steering']) if 'control/steering' in f else len(f['vehicle/timestamps'])
            print(f"Analyzing all {end_frame} frames\n")
        
        # Get data
        if 'ground_truth/path_curvature' in f:
            gt_curv = f['ground_truth/path_curvature'][:end_frame]
        else:
            print("No ground truth curvature data")
            return
        
        if 'control/path_curvature_input' in f:
            path_curv = f['control/path_curvature_input'][:end_frame]
        else:
            print("No path curvature input data")
            return
        
        if 'control/feedforward_steering' in f:
            ff_steer = f['control/feedforward_steering'][:end_frame]
        else:
            ff_steer = np.zeros(end_frame)
        
        steering = f['control/steering'][:end_frame] if 'control/steering' in f else np.zeros(end_frame)
        lat_err = f['control/lateral_error'][:end_frame] if 'control/lateral_error' in f else np.zeros(end_frame)
        
        # Find curve section
        curve_start = None
        for i in range(len(gt_curv)):
            if abs(gt_curv[i]) > 0.01:
                curve_start = i
                break
        
        print("="*70)
        print("OVERALL STATISTICS (all frames):")
        print("="*70)
        print(f"GT curvature: mean={np.mean(np.abs(gt_curv)):.4f}, max={np.max(np.abs(gt_curv)):.4f}")
        print(f"Path curvature: mean={np.mean(np.abs(path_curv)):.4f}, max={np.max(np.abs(path_curv)):.4f}")
        if np.mean(np.abs(gt_curv)) > 0:
            ratio = np.mean(np.abs(path_curv)) / np.mean(np.abs(gt_curv))
            print(f"Ratio (path/GT): {ratio:.1%}")
        print(f"Feedforward steering: mean={np.mean(np.abs(ff_steer)):.4f}, max={np.max(np.abs(ff_steer)):.4f}")
        print(f"Total steering: mean={np.mean(np.abs(steering)):.4f}, max={np.max(np.abs(steering)):.4f}")
        print(f"Lateral error: mean={np.mean(np.abs(lat_err)):.4f}, max={np.max(np.abs(lat_err)):.4f}")
        
        if curve_start:
            print(f"\n" + "="*70)
            print(f"CURVE SECTION (frames {curve_start}-{min(curve_start+50, end_frame)}):")
            print("="*70)
            curve_end = min(curve_start + 50, end_frame)
            curve_gt = gt_curv[curve_start:curve_end]
            curve_path = path_curv[curve_start:curve_end]
            curve_ff = ff_steer[curve_start:curve_end]
            curve_steering = steering[curve_start:curve_end]
            curve_lat_err = lat_err[curve_start:curve_end]
            
            print(f"GT curvature: mean={np.mean(np.abs(curve_gt)):.4f}, max={np.max(np.abs(curve_gt)):.4f}")
            print(f"Path curvature: mean={np.mean(np.abs(curve_path)):.4f}, max={np.max(np.abs(curve_path)):.4f}")
            if np.mean(np.abs(curve_gt)) > 0:
                ratio = np.mean(np.abs(curve_path)) / np.mean(np.abs(curve_gt))
                print(f"Ratio (path/GT): {ratio:.1%} {'✓ IMPROVED' if ratio > 0.1 else '✗ Still too low'}")
            print(f"Feedforward steering: mean={np.mean(np.abs(curve_ff)):.4f}, max={np.max(np.abs(curve_ff)):.4f}")
            print(f"Total steering: mean={np.mean(np.abs(curve_steering)):.4f}, max={np.max(np.abs(curve_steering)):.4f}")
            print(f"Lateral error: mean={np.mean(np.abs(curve_lat_err)):.4f}, max={np.max(np.abs(curve_lat_err)):.4f}")
            
            # Check if feedforward is activating
            ff_active_frames = np.sum(np.abs(curve_ff) > 0.01)
            print(f"\nFeedforward activation: {ff_active_frames}/{len(curve_ff)} frames ({100*ff_active_frames/len(curve_ff):.0f}%)")
            if ff_active_frames == 0:
                print("  ⚠️  Feedforward still not activating!")
            else:
                print(f"  ✓ Feedforward is activating")
        
        # Frame-by-frame around curve entry
        if curve_start:
            print(f"\n" + "="*70)
            print(f"CURVE ENTRY DETAIL (frames {curve_start}-{min(curve_start+20, end_frame)}):")
            print("="*70)
            print(f"{'Frame':<8} {'GT_Curv':<10} {'Path_Curv':<12} {'FF_Steer':<10} {'Total_Steer':<12} {'Lat_Err':<10}")
            print("-"*70)
            for i in range(curve_start, min(curve_start+20, end_frame)):
                print(f"{i:<8} {gt_curv[i]:<10.4f} {path_curv[i]:<12.4f} {ff_steer[i]:<10.3f} {steering[i]:<12.3f} {lat_err[i]:<10.3f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze curvature fix results")
    parser.add_argument("recording", nargs="?", help="Recording file (default: latest)")
    args = parser.parse_args()
    
    if args.recording:
        recording_path = args.recording
    else:
        recordings = sorted(Path("data/recordings").glob("*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not recordings:
            print("No recordings found")
            sys.exit(1)
        recording_path = str(recordings[0])
    
    analyze_curvature_fix(recording_path)
