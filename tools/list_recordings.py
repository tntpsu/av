"""
List recordings with their types and metadata.

Helps identify which recordings are manual drive, ground truth follower, etc.
"""

import sys
import h5py
import json
from pathlib import Path
from datetime import datetime

def list_recordings(recordings_dir: str = "data/recordings", show_all: bool = False):
    """
    List recordings with their types.
    
    Args:
        recordings_dir: Directory containing recordings
        show_all: If True, show all recordings. If False, show only recent ones.
    """
    recordings_path = Path(recordings_dir)
    if not recordings_path.exists():
        print(f"Recordings directory not found: {recordings_dir}")
        return
    
    recordings = sorted(recordings_path.glob("*.h5"), 
                       key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not recordings:
        print("No recordings found!")
        return
    
    if not show_all:
        recordings = recordings[:10]  # Show last 10
    
    print("="*80)
    print(f"RECORDINGS ({len(recordings)} shown)")
    print("="*80)
    print(f"{'Type':<25} {'Name':<35} {'Frames':<8} {'Date':<20}")
    print("-"*80)
    
    for rec in recordings:
        try:
            with h5py.File(rec, 'r') as f:
                metadata = {}
                if "metadata" in f.attrs:
                    metadata = json.loads(f.attrs["metadata"])
                
                rec_type = metadata.get("recording_type", "unknown")
                num_frames = metadata.get("total_frames", len(f.get("camera/images", [])))
                start_time = metadata.get("recording_start_time", "")
                
                # Format date
                if start_time:
                    try:
                        dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                        date_str = dt.strftime("%Y-%m-%d %H:%M")
                    except:
                        date_str = start_time[:16] if len(start_time) > 16 else start_time
                else:
                    date_str = "unknown"
                
                # Color code by type
                type_display = rec_type
                if rec_type == "gt_drive" or rec_type == "ground_truth_follower":  # Support old name
                    type_display = "🔬 GT Drive"
                elif rec_type == "perception_replay" or rec_type == "reprocessed":  # Support old name
                    type_display = "🔄 Perception Replay"
                elif rec_type == "manual":
                    type_display = "👤 Manual"
                elif rec_type == "av_stack":
                    type_display = "🚗 AV Stack"
                else:
                    type_display = f"❓ {rec_type}"
                
                print(f"{type_display:<25} {rec.stem:<35} {num_frames:<8} {date_str:<20}")
        
        except Exception as e:
            print(f"{'ERROR':<25} {rec.stem:<35} {'-':<8} {str(e)[:20]}")
    
    print("="*80)
    print("\nTypes:")
    print("  🔬 GT Drive         - Drive using ground truth (creates test recording)")
    print("  🔄 Perception Replay - Replay perception stack against recording")
    print("  👤 Manual           - Manually driven")
    print("  🚗 AV Stack         - Normal AV stack operation")
    print("  ❓ unknown          - No type metadata (old recording)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="List recordings with their types")
    parser.add_argument("--all", action="store_true",
                       help="Show all recordings (default: last 10)")
    parser.add_argument("--dir", type=str, default="data/recordings",
                       help="Recordings directory")
    
    args = parser.parse_args()
    
    list_recordings(args.dir, show_all=args.all)

