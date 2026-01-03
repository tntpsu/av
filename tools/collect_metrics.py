#!/usr/bin/env python3
"""
Metrics collection and analysis tool for AV Stack.
Analyzes logs and generates performance reports.
"""

import re
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class FrameMetrics:
    """Metrics for a single frame."""
    frame_id: int
    lanes_detected: int
    confidence: float
    speed: float
    heading: float
    steering: float
    throttle: float
    brake: float
    has_trajectory: bool
    ref_x: Optional[float] = None
    ref_y: Optional[float] = None
    ref_heading: Optional[float] = None


@dataclass
class SystemMetrics:
    """Aggregated system metrics."""
    total_frames: int
    lane_detection_rate: float
    avg_confidence: float
    avg_speed: float
    speed_std: float
    avg_steering: float
    steering_std: float
    avg_throttle: float
    avg_brake: float
    trajectory_availability: float
    frames_with_movement: int
    frames_with_extreme_steering: int
    frames_with_extreme_speed: int
    avg_lateral_offset: Optional[float] = None
    avg_ref_heading_error: Optional[float] = None
    frames_with_ref_point: int = 0


def parse_av_stack_log(log_file: str) -> List[FrameMetrics]:
    """Parse AV stack log file and extract frame metrics."""
    frames = []
    
    # Pattern: Frame 30: Lanes=2, Conf=0.50, Speed=8.31m/s, Heading=0.4°, Ref=(-0.07, 8.00, 83.8°), Steering=0.236, Throttle=0.011, Brake=0.000
    pattern = re.compile(
        r'Frame (\d+): Lanes=(\d+), Conf=([\d.]+), '
        r'Speed=([\d.]+)m/s, Heading=([\d.-]+)°'
        r'(?:, Ref=\(([\d.-]+), ([\d.-]+), ([\d.-]+)°\))?'
        r', Steering=([\d.-]+), Throttle=([\d.]+), Brake=([\d.]+)'
    )
    
    with open(log_file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                groups = match.groups()
                frame_id, lanes, conf, speed, heading = groups[0:5]
                ref_x, ref_y, ref_heading = groups[5:8] if len(groups) > 8 else (None, None, None)
                steering, throttle, brake = groups[8:11] if len(groups) > 8 else groups[5:8]
                
                # Check if trajectory was available (if we see steering/throttle, assume trajectory)
                has_trajectory = True  # Could be improved by checking for "No trajectory" messages
                
                frames.append(FrameMetrics(
                    frame_id=int(frame_id),
                    lanes_detected=int(lanes),
                    confidence=float(conf),
                    speed=float(speed),
                    heading=float(heading),
                    steering=float(steering),
                    throttle=float(throttle),
                    brake=float(brake),
                    has_trajectory=has_trajectory,
                    ref_x=float(ref_x) if ref_x else None,
                    ref_y=float(ref_y) if ref_y else None,
                    ref_heading=float(ref_heading) if ref_heading else None
                ))
    
    return frames


def calculate_metrics(frames: List[FrameMetrics]) -> SystemMetrics:
    """Calculate aggregated metrics from frame data."""
    if not frames:
        return SystemMetrics(
            total_frames=0, lane_detection_rate=0.0, avg_confidence=0.0,
            avg_speed=0.0, speed_std=0.0, avg_steering=0.0, steering_std=0.0,
            avg_throttle=0.0, avg_brake=0.0, trajectory_availability=0.0,
            frames_with_movement=0, frames_with_extreme_steering=0, frames_with_extreme_speed=0
        )
    
    total = len(frames)
    
    # Lane detection
    frames_with_lanes = sum(1 for f in frames if f.lanes_detected > 0)
    lane_detection_rate = frames_with_lanes / total
    
    # Confidence
    avg_confidence = sum(f.confidence for f in frames) / total
    
    # Speed
    speeds = [f.speed for f in frames]
    avg_speed = sum(speeds) / total
    speed_variance = sum((s - avg_speed) ** 2 for s in speeds) / total
    speed_std = speed_variance ** 0.5
    frames_with_movement = sum(1 for s in speeds if s > 0.1)
    frames_with_extreme_speed = sum(1 for s in speeds if s > 15.0)
    
    # Steering
    steerings = [f.steering for f in frames]
    avg_steering = sum(steerings) / total
    steering_variance = sum((s - avg_steering) ** 2 for s in steerings) / total
    steering_std = steering_variance ** 0.5
    frames_with_extreme_steering = sum(1 for s in steerings if abs(s) > 0.9)
    
    # Throttle/Brake
    avg_throttle = sum(f.throttle for f in frames) / total
    avg_brake = sum(f.brake for f in frames) / total
    
    # Trajectory
    frames_with_trajectory = sum(1 for f in frames if f.has_trajectory)
    trajectory_availability = frames_with_trajectory / total
    
    # Reference point metrics
    frames_with_ref = [f for f in frames if f.ref_x is not None]
    frames_with_ref_point = len(frames_with_ref)
    avg_lateral_offset = None
    avg_ref_heading_error = None
    
    if frames_with_ref:
        # Lateral offset (ref_x) - how far from center
        lateral_offsets = [abs(f.ref_x) for f in frames_with_ref]
        avg_lateral_offset = sum(lateral_offsets) / len(lateral_offsets)
        
        # Reference heading error (should be close to 0 for straight road)
        ref_heading_errors = [abs(f.ref_heading) for f in frames_with_ref if f.ref_heading is not None]
        if ref_heading_errors:
            avg_ref_heading_error = sum(ref_heading_errors) / len(ref_heading_errors)
    
    return SystemMetrics(
        total_frames=total,
        lane_detection_rate=lane_detection_rate,
        avg_confidence=avg_confidence,
        avg_speed=avg_speed,
        speed_std=speed_std,
        avg_steering=avg_steering,
        steering_std=steering_std,
        avg_throttle=avg_throttle,
        avg_brake=avg_brake,
        trajectory_availability=trajectory_availability,
        frames_with_movement=frames_with_movement,
        frames_with_extreme_steering=frames_with_extreme_steering,
        frames_with_extreme_speed=frames_with_extreme_speed,
        avg_lateral_offset=avg_lateral_offset,
        avg_ref_heading_error=avg_ref_heading_error,
        frames_with_ref_point=frames_with_ref_point
    )


def print_metrics_report(metrics: SystemMetrics, frames: List[FrameMetrics]):
    """Print formatted metrics report."""
    print("=" * 60)
    print("AV STACK METRICS REPORT")
    print("=" * 60)
    print()
    
    print("📊 SYSTEM PERFORMANCE")
    print(f"  Total Frames Processed: {metrics.total_frames}")
    print(f"  Trajectory Availability: {metrics.trajectory_availability * 100:.1f}%")
    print()
    
    print("🛣️  LANE DETECTION")
    print(f"  Detection Rate: {metrics.lane_detection_rate * 100:.1f}%")
    print(f"  Average Confidence: {metrics.avg_confidence:.3f}")
    if metrics.lane_detection_rate == 0:
        print("  ⚠️  WARNING: No lanes detected!")
    print()
    
    print("🚗 VEHICLE CONTROL")
    print(f"  Average Speed: {metrics.avg_speed:.2f} m/s ({metrics.avg_speed * 3.6:.1f} km/h)")
    print(f"  Speed StdDev: {metrics.speed_std:.2f} m/s")
    print(f"  Frames with Movement: {metrics.frames_with_movement}/{metrics.total_frames}")
    if metrics.frames_with_movement == 0:
        print("  ⚠️  WARNING: Vehicle not moving!")
    if metrics.frames_with_extreme_speed > 0:
        print(f"  ⚠️  WARNING: {metrics.frames_with_extreme_speed} frames with speed > 15 m/s")
    print()
    
    print("🎮 CONTROL COMMANDS")
    print(f"  Average Steering: {metrics.avg_steering:.3f}")
    print(f"  Steering StdDev: {metrics.steering_std:.3f} (lower = smoother)")
    if metrics.steering_std < 0.1:
        print("  ✅ Steering is stable")
    elif metrics.steering_std > 0.5:
        print("  ⚠️  WARNING: Steering is oscillating!")
    print(f"  Frames with Extreme Steering: {metrics.frames_with_extreme_steering}")
    print(f"  Average Throttle: {metrics.avg_throttle:.3f}")
    print(f"  Average Brake: {metrics.avg_brake:.3f}")
    print()
    
    if metrics.frames_with_ref_point > 0:
        print("🎯 TRAJECTORY PLANNING")
        print(f"  Frames with Reference Point: {metrics.frames_with_ref_point}/{metrics.total_frames}")
        if metrics.avg_lateral_offset is not None:
            print(f"  Average Lateral Offset: {metrics.avg_lateral_offset:.3f} m")
            if metrics.avg_lateral_offset > 1.0:
                print("  ⚠️  WARNING: Car is far from lane center!")
            elif metrics.avg_lateral_offset < 0.3:
                print("  ✅ Car is well-centered")
        if metrics.avg_ref_heading_error is not None:
            print(f"  Average Reference Heading Error: {metrics.avg_ref_heading_error:.1f}°")
            if metrics.avg_ref_heading_error > 10.0:
                print("  ⚠️  WARNING: Reference heading is incorrect!")
        print()
    
    print("✅ SUCCESS CRITERIA")
    criteria = {
        "Lane Detection Rate > 50%": metrics.lane_detection_rate > 0.5,
        "Vehicle Moving": metrics.frames_with_movement > 0,
        "Steering Smooth (StdDev < 0.2)": metrics.steering_std < 0.2,
        "No Extreme Speed": metrics.frames_with_extreme_speed == 0,
        "Trajectory Available": metrics.trajectory_availability > 0.9,
    }
    
    for criterion, passed in criteria.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {criterion}")
    
    passed_count = sum(criteria.values())
    print(f"\n  Overall: {passed_count}/{len(criteria)} criteria met")
    print()
    
    print("=" * 60)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect and analyze AV Stack metrics')
    # Default to tmp/logs/av_stack.log in project root
    default_log = Path(__file__).parent / 'tmp' / 'logs' / 'av_stack.log'
    parser.add_argument('--log', type=str, default=str(default_log),
                       help='Path to AV stack log file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for metrics')
    parser.add_argument('--json', action='store_true',
                       help='Output in JSON format')
    
    args = parser.parse_args()
    
    if not Path(args.log).exists():
        print(f"Error: Log file not found: {args.log}")
        sys.exit(1)
    
    # Parse frames
    frames = parse_av_stack_log(args.log)
    
    if not frames:
        print("Error: No frame data found in log file")
        sys.exit(1)
    
    # Calculate metrics
    metrics = calculate_metrics(frames)
    
    # Output
    if args.json:
        output = {
            'metrics': asdict(metrics),
            'frame_count': len(frames)
        }
        print(json.dumps(output, indent=2))
    else:
        print_metrics_report(metrics, frames)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({
                    'metrics': asdict(metrics),
                    'frames': [asdict(frame) for frame in frames]
                }, f, indent=2)
            print(f"\nMetrics saved to: {args.output}")


if __name__ == '__main__':
    main()

