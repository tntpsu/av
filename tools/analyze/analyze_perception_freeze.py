#!/usr/bin/env python3
"""
Analyze perception data around a specific frame to diagnose why perception stopped updating.
"""

import h5py
import numpy as np
import sys
from pathlib import Path

def analyze_perception_freeze(recording_file: str, start_frame: int = 150, end_frame: int = 200):
    """
    Analyze perception data around a specific frame range.
    
    Args:
        recording_file: Path to HDF5 recording file
        start_frame: Starting frame to analyze
        end_frame: Ending frame to analyze
    """
    recording_path = Path(recording_file)
    if not recording_path.exists():
        print(f"ERROR: Recording file not found: {recording_file}")
        return
    
    print("=" * 80)
    print(f"ANALYZING PERCEPTION DATA: Frames {start_frame} to {end_frame}")
    print("=" * 80)
    print(f"Recording: {recording_path.name}")
    print()
    
    with h5py.File(recording_path, 'r') as f:
        # Check what datasets are available
        print("Available datasets:")
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
        f.visititems(print_structure)
        print()
        
        # Get perception data
        if 'perception/timestamps' not in f:
            print("ERROR: No perception timestamps found in recording")
            return
        
        timestamps = f['perception/timestamps'][:]
        total_frames = len(timestamps)
        
        print(f"Total frames in recording: {total_frames}")
        print(f"Analyzing frames {start_frame} to {min(end_frame, total_frames-1)}")
        print()
        
        # Get perception lane positions
        left_lane_x = None
        right_lane_x = None
        if 'perception/left_lane_line_x' in f:
            left_lane_x = f['perception/left_lane_line_x'][:]
        elif 'perception/left_lane_x' in f:
            left_lane_x = f['perception/left_lane_x'][:]
        
        if 'perception/right_lane_line_x' in f:
            right_lane_x = f['perception/right_lane_line_x'][:]
        elif 'perception/right_lane_x' in f:
            right_lane_x = f['perception/right_lane_x'][:]
        
        # Get lane width
        lane_width = None
        if 'perception/lane_width' in f:
            lane_width = f['perception/lane_width'][:]
        
        # Get vehicle state timestamps for comparison
        vehicle_timestamps = None
        if 'vehicle/timestamps' in f:
            vehicle_timestamps = f['vehicle/timestamps'][:]
        
        # Analyze frames
        print("FRAME-BY-FRAME ANALYSIS:")
        print("-" * 80)
        print(f"{'Frame':<6} {'Timestamp':<12} {'dt':<8} {'Left X':<10} {'Right X':<10} {'Width':<10} {'Status'}")
        print("-" * 80)
        
        prev_timestamp = None
        prev_left = None
        prev_right = None
        
        for i in range(start_frame, min(end_frame + 1, total_frames)):
            ts = float(timestamps[i])
            dt = ts - prev_timestamp if prev_timestamp is not None else 0.0
            
            left_val = float(left_lane_x[i]) if left_lane_x is not None and i < len(left_lane_x) else None
            right_val = float(right_lane_x[i]) if right_lane_x is not None and i < len(right_lane_x) else None
            width_val = float(lane_width[i]) if lane_width is not None and i < len(lane_width) else None
            
            # Check if values changed
            left_changed = (prev_left is not None and abs(left_val - prev_left) > 0.001) if (left_val is not None and prev_left is not None) else True
            right_changed = (prev_right is not None and abs(right_val - prev_right) > 0.001) if (right_val is not None and prev_right is not None) else True
            
            status = []
            if dt > 0.5:
                status.append("TIME_GAP")
            if not left_changed and not right_changed:
                status.append("FROZEN")
            if left_val is None or right_val is None:
                status.append("MISSING")
            if not status:
                status.append("OK")
            
            status_str = ", ".join(status)
            
            left_str = f"{left_val:.3f}" if left_val is not None else "None"
            right_str = f"{right_val:.3f}" if right_val is not None else "None"
            width_str = f"{width_val:.3f}" if width_val is not None else "None"
            
            print(f"{i:<6} {ts:<12.3f} {dt:<8.3f} {left_str:<10} {right_str:<10} {width_str:<10} {status_str}")
            
            prev_timestamp = ts
            prev_left = left_val
            prev_right = right_val
        
        print("-" * 80)
        print()
        
        # Summary statistics
        print("SUMMARY:")
        print("-" * 80)
        
        # Count frozen frames
        frozen_count = 0
        time_gap_count = 0
        missing_count = 0
        
        for i in range(start_frame, min(end_frame + 1, total_frames)):
            if i >= len(timestamps):
                break
            
            ts = float(timestamps[i])
            dt = ts - prev_timestamp if prev_timestamp is not None and i > start_frame else 0.0
            
            if dt > 0.5:
                time_gap_count += 1
            
            if left_lane_x is not None and right_lane_x is not None and i < len(left_lane_x) and i < len(right_lane_x):
                if i > start_frame:
                    prev_left = float(left_lane_x[i-1])
                    prev_right = float(right_lane_x[i-1])
                    curr_left = float(left_lane_x[i])
                    curr_right = float(right_lane_x[i])
                    
                    if abs(curr_left - prev_left) < 0.001 and abs(curr_right - prev_right) < 0.001:
                        frozen_count += 1
            else:
                missing_count += 1
            
            prev_timestamp = ts
        
        total_analyzed = min(end_frame + 1, total_frames) - start_frame
        print(f"Total frames analyzed: {total_analyzed}")
        print(f"Frozen frames (no change): {frozen_count} ({100*frozen_count/total_analyzed:.1f}%)")
        print(f"Time gaps (>0.5s): {time_gap_count} ({100*time_gap_count/total_analyzed:.1f}%)")
        print(f"Missing data: {missing_count} ({100*missing_count/total_analyzed:.1f}%)")
        print()
        
        # Check vehicle state timestamps for comparison
        if vehicle_timestamps is not None:
            print("VEHICLE STATE COMPARISON:")
            print("-" * 80)
            print(f"Vehicle timestamps available: {len(vehicle_timestamps)} frames")
            
            # Find matching vehicle state for frame 157
            if 157 < len(timestamps):
                perception_ts = float(timestamps[157])
                # Find closest vehicle timestamp
                vehicle_idx = np.argmin(np.abs(vehicle_timestamps - perception_ts))
                vehicle_ts = float(vehicle_timestamps[vehicle_idx])
                print(f"Frame 157: perception_ts={perception_ts:.3f}, vehicle_ts={vehicle_ts:.3f}, diff={abs(perception_ts - vehicle_ts):.3f}s")
                print()
        
        # Check for patterns
        print("PATTERN ANALYSIS:")
        print("-" * 80)
        
        # Find first frozen frame
        first_frozen = None
        for i in range(start_frame, min(end_frame + 1, total_frames)):
            if i >= len(timestamps) or i == 0:
                continue
            
            if left_lane_x is not None and right_lane_x is not None and i < len(left_lane_x) and i < len(right_lane_x):
                prev_left = float(left_lane_x[i-1])
                prev_right = float(right_lane_x[i-1])
                curr_left = float(left_lane_x[i])
                curr_right = float(right_lane_x[i])
                
                if abs(curr_left - prev_left) < 0.001 and abs(curr_right - prev_right) < 0.001:
                    first_frozen = i
                    break
        
        if first_frozen:
            print(f"First frozen frame: {first_frozen}")
            print(f"  Timestamp: {float(timestamps[first_frozen]):.3f}")
            if first_frozen > 0:
                prev_ts = float(timestamps[first_frozen-1])
                curr_ts = float(timestamps[first_frozen])
                dt = curr_ts - prev_ts
                print(f"  Time since previous frame: {dt:.3f}s")
                if dt > 0.5:
                    print(f"  ⚠️  LARGE TIME GAP - Unity likely paused!")
                else:
                    print(f"  ⚠️  NO TIME GAP - Perception likely failed!")
        else:
            print("No frozen frames detected in this range")
        print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/analyze_perception_freeze.py <recording_file.h5> [start_frame] [end_frame]")
        print("Example: python tools/analyze_perception_freeze.py data/recordings/recording_20260103_112942.h5 150 200")
        sys.exit(1)
    
    recording_file = sys.argv[1]
    start_frame = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    end_frame = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    
    analyze_perception_freeze(recording_file, start_frame, end_frame)

