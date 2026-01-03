"""
Quick script to check what data exists in a recording.
Can be run without h5py if needed.
"""

import sys
from pathlib import Path

def check_recording(recording_file: str):
    """Check what data exists in recording."""
    
    print("=" * 70)
    print(f"CHECKING RECORDING: {recording_file}")
    print("=" * 70)
    print()
    
    try:
        import h5py
        import numpy as np
    except ImportError:
        print("⚠️  h5py/numpy not available")
        print("Install with: pip install h5py numpy")
        return
    
    try:
        with h5py.File(recording_file, 'r') as f:
            print("Available datasets:")
            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    shape = obj.shape
                    dtype = obj.dtype
                    size = obj.size
                    print(f"  {name}: shape={shape}, dtype={dtype}, size={size}")
            f.visititems(print_structure)
            print()
            
            # Check key datasets
            print("Key Data Checks:")
            print("-" * 70)
            
            checks = [
                ('camera/images', 'Camera frames'),
                ('vehicle/position', 'Vehicle positions'),
                ('vehicle/steering_angle', 'Steering angles'),
                ('control/steering', 'Control steering'),
                ('ground_truth/path_curvature', 'Path curvature'),
                ('ground_truth/lane_center_x', 'Lane center'),
                ('unity_feedback/ground_truth_mode_active', 'GT mode active'),
            ]
            
            for key, desc in checks:
                if key in f:
                    data = f[key]
                    print(f"  ✓ {desc}: {len(data)} samples")
                    if len(data) > 0:
                        if 'curvature' in key:
                            avg = np.mean(data[:])
                            print(f"      Average: {avg:.6f}")
                        elif 'steering' in key or 'angle' in key:
                            avg = np.mean(data[:])
                            print(f"      Average: {avg:.3f}")
                else:
                    print(f"  ✗ {desc}: NOT FOUND")
            
            print()
            print("=" * 70)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Find latest recording
        recordings_dir = Path("data/recordings")
        recordings = sorted(recordings_dir.glob("*.h5"), 
                          key=lambda p: p.stat().st_mtime, reverse=True)
        if recordings:
            recording_file = str(recordings[0])
            print(f"Using latest recording: {recording_file}\n")
        else:
            print("No recordings found!")
            sys.exit(1)
    else:
        recording_file = sys.argv[1]
    
    check_recording(recording_file)

