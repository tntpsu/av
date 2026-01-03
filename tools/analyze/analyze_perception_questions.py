"""
Analyze perception data to systematically answer the 7 key questions.

This script takes a recording and provides detailed analysis for each question.
"""

import sys
import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_perception_ground_truth import PerceptionGroundTruthTest


def analyze_question_1(test: PerceptionGroundTruthTest) -> Dict:
    """
    Question 1: Are we detecting the correct lane lines?
    
    Checks:
    - Detection rate
    - What we're detecting (width, positions)
    - Comparison to ground truth
    """
    print("\n" + "="*80)
    print("QUESTION 1: Are we detecting the correct lane lines?")
    print("="*80)
    
    valid_frames = np.where(test.num_lanes_detected >= 2)[0]
    detection_rate = len(valid_frames) / test.num_frames
    
    print(f"\nDetection Rate: {detection_rate*100:.1f}% ({len(valid_frames)}/{test.num_frames} frames)")
    
    if len(valid_frames) == 0:
        print("❌ CRITICAL: No frames with valid lane detection!")
        return {'status': 'FAIL', 'detection_rate': 0.0}
    
    # Analyze what we're detecting
    print(f"\nDetected Lane Characteristics (when detected):")
    print(f"  Left lane range: {np.min(test.perception_left[valid_frames]):.3f}m to {np.max(test.perception_left[valid_frames]):.3f}m")
    print(f"  Right lane range: {np.min(test.perception_right[valid_frames]):.3f}m to {np.max(test.perception_right[valid_frames]):.3f}m")
    print(f"  Width range: {np.min(test.perception_width[valid_frames]):.3f}m to {np.max(test.perception_width[valid_frames]):.3f}m")
    print(f"  Width mean: {np.mean(test.perception_width[valid_frames]):.3f}m")
    
    print(f"\nGround Truth Lane Characteristics:")
    print(f"  Left lane range: {np.min(test.gt_left[valid_frames]):.3f}m to {np.max(test.gt_left[valid_frames]):.3f}m")
    print(f"  Right lane range: {np.min(test.gt_right[valid_frames]):.3f}m to {np.max(test.gt_right[valid_frames]):.3f}m")
    print(f"  Width range: {np.min(test.gt_width[valid_frames]):.3f}m to {np.max(test.gt_width[valid_frames]):.3f}m")
    print(f"  Width mean: {np.mean(test.gt_width[valid_frames]):.3f}m")
    
    # Check if width is correct
    width_error = np.abs(test.perception_width[valid_frames] - test.gt_width[valid_frames])
    mean_width_error = np.mean(width_error)
    
    print(f"\nWidth Accuracy:")
    print(f"  Mean absolute error: {mean_width_error:.3f}m")
    print(f"  Max error: {np.max(width_error):.3f}m")
    
    if detection_rate < 0.8:
        status = 'WARN'
        print(f"⚠️  WARNING: Detection rate is low ({detection_rate*100:.1f}%)")
    elif mean_width_error > 0.5:
        status = 'WARN'
        print(f"⚠️  WARNING: Width error is large ({mean_width_error:.3f}m)")
    else:
        status = 'PASS'
        print(f"✓ Detection rate and width accuracy are acceptable")
    
    return {
        'status': status,
        'detection_rate': detection_rate,
        'mean_width_error': mean_width_error,
        'detected_width_mean': np.mean(test.perception_width[valid_frames]),
        'gt_width_mean': np.mean(test.gt_width[valid_frames])
    }


def analyze_question_2(test: PerceptionGroundTruthTest) -> Dict:
    """
    Question 2: Are lane positions accurate in vehicle coordinates?
    """
    print("\n" + "="*80)
    print("QUESTION 2: Are lane positions accurate in vehicle coordinates?")
    print("="*80)
    
    valid_frames = np.where(test.num_lanes_detected >= 2)[0]
    
    if len(valid_frames) == 0:
        print("❌ Cannot analyze - no valid detections")
        return {'status': 'FAIL'}
    
    left_errors = test.perception_left[valid_frames] - test.gt_left[valid_frames]
    right_errors = test.perception_right[valid_frames] - test.gt_right[valid_frames]
    center_errors = (test.perception_left[valid_frames] + test.perception_right[valid_frames]) / 2.0 - test.gt_center[valid_frames]
    
    print(f"\nPosition Errors (detected - ground truth):")
    print(f"  Left lane:")
    print(f"    Mean absolute error: {np.mean(np.abs(left_errors)):.3f}m")
    print(f"    Mean error: {np.mean(left_errors):.3f}m (bias)")
    print(f"    Std deviation: {np.std(left_errors):.3f}m")
    print(f"    Max error: {np.max(np.abs(left_errors)):.3f}m")
    
    print(f"  Right lane:")
    print(f"    Mean absolute error: {np.mean(np.abs(right_errors)):.3f}m")
    print(f"    Mean error: {np.mean(right_errors):.3f}m (bias)")
    print(f"    Std deviation: {np.std(right_errors):.3f}m")
    print(f"    Max error: {np.max(np.abs(right_errors)):.3f}m")
    
    print(f"  Center (computed from lanes):")
    print(f"    Mean absolute error: {np.mean(np.abs(center_errors)):.3f}m")
    print(f"    Mean error: {np.mean(center_errors):.3f}m (bias)")
    print(f"    Std deviation: {np.std(center_errors):.3f}m")
    print(f"    Max error: {np.max(np.abs(center_errors)):.3f}m")
    
    # Check for systematic bias
    left_bias = np.mean(left_errors)
    right_bias = np.mean(right_errors)
    
    if abs(left_bias) > 0.2 or abs(right_bias) > 0.2:
        status = 'WARN'
        print(f"\n⚠️  WARNING: Systematic bias detected!")
        print(f"    Left bias: {left_bias:.3f}m, Right bias: {right_bias:.3f}m")
        print(f"    This suggests coordinate conversion may be incorrect")
    elif np.mean(np.abs(left_errors)) > 0.5 or np.mean(np.abs(right_errors)) > 0.5:
        status = 'WARN'
        print(f"\n⚠️  WARNING: Large position errors detected")
    else:
        status = 'PASS'
        print(f"\n✓ Position accuracy is acceptable")
    
    return {
        'status': status,
        'left_mae': np.mean(np.abs(left_errors)),
        'right_mae': np.mean(np.abs(right_errors)),
        'center_mae': np.mean(np.abs(center_errors)),
        'left_bias': left_bias,
        'right_bias': right_bias
    }


def analyze_question_3(test: PerceptionGroundTruthTest) -> Dict:
    """
    Question 3: Is lane width correct?
    """
    print("\n" + "="*80)
    print("QUESTION 3: Is lane width correct?")
    print("="*80)
    
    valid_frames = np.where(test.num_lanes_detected >= 2)[0]
    
    if len(valid_frames) == 0:
        print("❌ Cannot analyze - no valid detections")
        return {'status': 'FAIL'}
    
    width_errors = test.perception_width[valid_frames] - test.gt_width[valid_frames]
    
    print(f"\nWidth Error Analysis:")
    print(f"  Mean absolute error: {np.mean(np.abs(width_errors)):.3f}m")
    print(f"  Mean error: {np.mean(width_errors):.3f}m (bias)")
    print(f"  Std deviation: {np.std(width_errors):.3f}m")
    print(f"  Max error: {np.max(np.abs(width_errors)):.3f}m")
    
    print(f"\nDetected vs Ground Truth:")
    print(f"  Detected width: {np.mean(test.perception_width[valid_frames]):.3f}m ± {np.std(test.perception_width[valid_frames]):.3f}m")
    print(f"  Ground truth width: {np.mean(test.gt_width[valid_frames]):.3f}m ± {np.std(test.gt_width[valid_frames]):.3f}m")
    print(f"  Expected width: ~7.0m (single lane road)")
    
    # Check if width is systematically wrong
    width_bias = np.mean(width_errors)
    width_mae = np.mean(np.abs(width_errors))
    
    if abs(width_bias) > 1.0:
        status = 'FAIL'
        print(f"\n❌ CRITICAL: Width is systematically wrong!")
        print(f"    Bias: {width_bias:.3f}m ({width_bias/7.0*100:.1f}% of expected width)")
        print(f"    This suggests coordinate conversion is incorrect")
    elif width_mae > 0.5:
        status = 'WARN'
        print(f"\n⚠️  WARNING: Width error is large ({width_mae:.3f}m)")
    else:
        status = 'PASS'
        print(f"\n✓ Width accuracy is acceptable")
    
    return {
        'status': status,
        'width_mae': width_mae,
        'width_bias': width_bias,
        'detected_width_mean': np.mean(test.perception_width[valid_frames]),
        'gt_width_mean': np.mean(test.gt_width[valid_frames])
    }


def analyze_question_4(test: PerceptionGroundTruthTest) -> Dict:
    """
    Question 4: Is heading calculation correct?
    """
    print("\n" + "="*80)
    print("QUESTION 4: Is heading calculation correct?")
    print("="*80)
    
    valid_frames = np.where(test.num_lanes_detected >= 2)[0]
    
    if len(valid_frames) == 0:
        print("❌ Cannot analyze - no valid detections")
        return {'status': 'FAIL'}
    
    # Compute expected heading from ground truth
    expected_headings = []
    for i in valid_frames:
        expected = test.compute_expected_heading(i)
        expected_headings.append(expected)
    
    expected_headings = np.array(expected_headings)
    actual_headings = test.heading[valid_frames]
    heading_errors = actual_headings - expected_headings
    
    print(f"\nHeading Analysis:")
    print(f"  Actual heading: {np.mean(np.abs(actual_headings)):.1f}° ± {np.std(actual_headings):.1f}°")
    print(f"  Expected heading: {np.mean(np.abs(expected_headings)):.1f}° ± {np.std(expected_headings):.1f}°")
    print(f"  Mean absolute error: {np.mean(np.abs(heading_errors)):.1f}°")
    print(f"  Max error: {np.max(np.abs(heading_errors)):.1f}°")
    
    # For a 50m radius curve, expected heading should be ~2.86°
    expected_curve_heading = 2.86
    actual_mean_heading = np.mean(np.abs(actual_headings))
    
    print(f"\nCurve Analysis:")
    print(f"  Expected heading for 50m curve: ~{expected_curve_heading:.2f}°")
    print(f"  Actual mean heading: {actual_mean_heading:.2f}°")
    print(f"  Ratio: {actual_mean_heading/expected_curve_heading:.1f}x")
    
    if actual_mean_heading > expected_curve_heading * 2.0:
        status = 'FAIL'
        print(f"\n❌ CRITICAL: Heading is {actual_mean_heading/expected_curve_heading:.1f}x too high!")
        print(f"    This suggests coordinate conversion or heading calculation is wrong")
    elif np.mean(np.abs(heading_errors)) > 5.0:
        status = 'WARN'
        print(f"\n⚠️  WARNING: Heading error is large ({np.mean(np.abs(heading_errors)):.1f}°)")
    else:
        status = 'PASS'
        print(f"\n✓ Heading accuracy is acceptable")
    
    return {
        'status': status,
        'heading_mae': np.mean(np.abs(heading_errors)),
        'actual_mean_heading': actual_mean_heading,
        'expected_mean_heading': np.mean(np.abs(expected_headings)),
        'heading_ratio': actual_mean_heading / expected_curve_heading if expected_curve_heading > 0 else 0
    }


def analyze_question_5(test: PerceptionGroundTruthTest) -> Dict:
    """
    Question 5: Are we handling curves correctly?
    """
    print("\n" + "="*80)
    print("QUESTION 5: Are we handling curves correctly?")
    print("="*80)
    
    # Identify curve sections (where heading is significant)
    curve_threshold = 2.0  # degrees
    curve_frames = np.where(np.abs(test.heading) > np.radians(curve_threshold))[0]
    straight_frames = np.where(np.abs(test.heading) <= np.radians(curve_threshold))[0]
    
    valid_curve_frames = [f for f in curve_frames if test.num_lanes_detected[f] >= 2]
    valid_straight_frames = [f for f in straight_frames if test.num_lanes_detected[f] >= 2]
    
    print(f"\nFrame Classification:")
    print(f"  Curve frames (heading > {curve_threshold}°): {len(curve_frames)}")
    print(f"  Straight frames (heading ≤ {curve_threshold}°): {len(straight_frames)}")
    print(f"  Valid curve detections: {len(valid_curve_frames)}")
    print(f"  Valid straight detections: {len(valid_straight_frames)}")
    
    if len(valid_curve_frames) == 0:
        print("❌ Cannot analyze curves - no valid detections on curves")
        return {'status': 'FAIL'}
    
    # Compare detection on curves vs straights
    curve_detection_rate = len(valid_curve_frames) / len(curve_frames) if len(curve_frames) > 0 else 0
    straight_detection_rate = len(valid_straight_frames) / len(straight_frames) if len(straight_frames) > 0 else 0
    
    print(f"\nDetection Rate Comparison:")
    print(f"  Curves: {curve_detection_rate*100:.1f}%")
    print(f"  Straights: {straight_detection_rate*100:.1f}%")
    
    # Position errors on curves
    if len(valid_curve_frames) > 0:
        curve_left_errors = test.perception_left[valid_curve_frames] - test.gt_left[valid_curve_frames]
        curve_right_errors = test.perception_right[valid_curve_frames] - test.gt_right[valid_curve_frames]
        curve_width_errors = test.perception_width[valid_curve_frames] - test.gt_width[valid_curve_frames]
        
        print(f"\nPosition Errors on Curves:")
        print(f"  Left lane MAE: {np.mean(np.abs(curve_left_errors)):.3f}m")
        print(f"  Right lane MAE: {np.mean(np.abs(curve_right_errors)):.3f}m")
        print(f"  Width MAE: {np.mean(np.abs(curve_width_errors)):.3f}m")
    
    if len(valid_straight_frames) > 0:
        straight_left_errors = test.perception_left[valid_straight_frames] - test.gt_left[valid_straight_frames]
        straight_right_errors = test.perception_right[valid_straight_frames] - test.gt_right[valid_straight_frames]
        straight_width_errors = test.perception_width[valid_straight_frames] - test.gt_width[valid_straight_frames]
        
        print(f"\nPosition Errors on Straights:")
        print(f"  Left lane MAE: {np.mean(np.abs(straight_left_errors)):.3f}m")
        print(f"  Right lane MAE: {np.mean(np.abs(straight_right_errors)):.3f}m")
        print(f"  Width MAE: {np.mean(np.abs(straight_width_errors)):.3f}m")
    
    if curve_detection_rate < 0.7:
        status = 'WARN'
        print(f"\n⚠️  WARNING: Detection rate on curves is low ({curve_detection_rate*100:.1f}%)")
    elif len(valid_curve_frames) > 0 and np.mean(np.abs(curve_width_errors)) > 0.5:
        status = 'WARN'
        print(f"\n⚠️  WARNING: Width error on curves is large")
    else:
        status = 'PASS'
        print(f"\n✓ Curve handling is acceptable")
    
    return {
        'status': status,
        'curve_detection_rate': curve_detection_rate,
        'straight_detection_rate': straight_detection_rate
    }


def analyze_question_6(test: PerceptionGroundTruthTest) -> Dict:
    """
    Question 6: Are we handling straight roads correctly?
    """
    print("\n" + "="*80)
    print("QUESTION 6: Are we handling straight roads correctly?")
    print("="*80)
    
    # Identify straight sections
    straight_threshold = 1.0  # degrees
    straight_frames = np.where(np.abs(test.heading) <= np.radians(straight_threshold))[0]
    valid_straight_frames = [f for f in straight_frames if test.num_lanes_detected[f] >= 2]
    
    print(f"\nStraight Road Analysis:")
    print(f"  Straight frames (heading ≤ {straight_threshold}°): {len(straight_frames)}")
    print(f"  Valid detections: {len(valid_straight_frames)}")
    
    if len(valid_straight_frames) == 0:
        print("❌ Cannot analyze - no valid detections on straights")
        return {'status': 'FAIL'}
    
    # Check heading on straights (should be near 0)
    straight_headings = test.heading[valid_straight_frames]
    mean_heading = np.mean(np.abs(straight_headings))
    
    print(f"\nHeading on Straights:")
    print(f"  Mean absolute heading: {np.degrees(mean_heading):.2f}°")
    print(f"  Max heading: {np.max(np.abs(straight_headings)):.2f}°")
    print(f"  Expected: ~0°")
    
    # Check position stability
    straight_left = test.perception_left[valid_straight_frames]
    straight_right = test.perception_right[valid_straight_frames]
    left_std = np.std(straight_left)
    right_std = np.std(straight_right)
    
    print(f"\nPosition Stability on Straights:")
    print(f"  Left lane std: {left_std:.3f}m")
    print(f"  Right lane std: {right_std:.3f}m")
    print(f"  Expected: Low variance (stable detection)")
    
    if mean_heading > np.radians(2.0):
        status = 'WARN'
        print(f"\n⚠️  WARNING: Heading on straights is too high ({np.degrees(mean_heading):.2f}°)")
    elif left_std > 0.3 or right_std > 0.3:
        status = 'WARN'
        print(f"\n⚠️  WARNING: Position is not stable on straights")
    else:
        status = 'PASS'
        print(f"\n✓ Straight road handling is acceptable")
    
    return {
        'status': status,
        'mean_heading_straight': np.degrees(mean_heading),
        'left_std': left_std,
        'right_std': right_std
    }


def analyze_question_7(test: PerceptionGroundTruthTest) -> Dict:
    """
    Question 7: Is there temporal consistency?
    """
    print("\n" + "="*80)
    print("QUESTION 7: Is there temporal consistency?")
    print("="*80)
    
    valid_frames = np.where(test.num_lanes_detected >= 2)[0]
    
    if len(valid_frames) < 2:
        print("❌ Cannot analyze - need at least 2 valid frames")
        return {'status': 'FAIL'}
    
    # Compute frame-to-frame changes
    frame_changes_left = []
    frame_changes_right = []
    frame_changes_width = []
    
    for i in range(len(valid_frames) - 1):
        curr_idx = valid_frames[i]
        next_idx = valid_frames[i + 1]
        
        if next_idx == curr_idx + 1:  # Consecutive frames
            frame_changes_left.append(abs(test.perception_left[next_idx] - test.perception_left[curr_idx]))
            frame_changes_right.append(abs(test.perception_right[next_idx] - test.perception_right[curr_idx]))
            frame_changes_width.append(abs(test.perception_width[next_idx] - test.perception_width[curr_idx]))
    
    if len(frame_changes_left) == 0:
        print("❌ Cannot analyze - no consecutive valid frames")
        return {'status': 'FAIL'}
    
    frame_changes_left = np.array(frame_changes_left)
    frame_changes_right = np.array(frame_changes_right)
    frame_changes_width = np.array(frame_changes_width)
    
    print(f"\nFrame-to-Frame Changes (consecutive valid frames):")
    print(f"  Left lane:")
    print(f"    Mean change: {np.mean(frame_changes_left):.3f}m")
    print(f"    Max change: {np.max(frame_changes_left):.3f}m")
    print(f"  Right lane:")
    print(f"    Mean change: {np.mean(frame_changes_right):.3f}m")
    print(f"    Max change: {np.max(frame_changes_right):.3f}m")
    print(f"  Width:")
    print(f"    Mean change: {np.mean(frame_changes_width):.3f}m")
    print(f"    Max change: {np.max(frame_changes_width):.3f}m")
    
    # Check for sudden jumps (outliers)
    large_jumps = np.sum(frame_changes_left > 0.5) + np.sum(frame_changes_right > 0.5)
    jump_rate = large_jumps / (len(frame_changes_left) * 2)
    
    print(f"\nTemporal Stability:")
    print(f"  Large jumps (>0.5m): {large_jumps} ({jump_rate*100:.1f}% of frame pairs)")
    print(f"  Expected: Low jump rate (smooth transitions)")
    
    if jump_rate > 0.1:
        status = 'WARN'
        print(f"\n⚠️  WARNING: High jump rate ({jump_rate*100:.1f}%) - temporal consistency is poor")
    elif np.mean(frame_changes_left) > 0.2 or np.mean(frame_changes_right) > 0.2:
        status = 'WARN'
        print(f"\n⚠️  WARNING: Large frame-to-frame changes")
    else:
        status = 'PASS'
        print(f"\n✓ Temporal consistency is acceptable")
    
    return {
        'status': status,
        'mean_frame_change_left': np.mean(frame_changes_left),
        'mean_frame_change_right': np.mean(frame_changes_right),
        'jump_rate': jump_rate
    }


def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze perception data to answer 7 key questions")
    parser.add_argument("recording_file", nargs='?', default=None,
                       help="Path to recording file (default: latest)")
    
    args = parser.parse_args()
    
    # Find recording file
    if args.recording_file:
        recording_file = args.recording_file
    else:
        recordings_dir = Path("data/recordings")
        recordings = sorted(recordings_dir.glob("*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not recordings:
            print("No recordings found!")
            return 1
        recording_file = str(recordings[0])
        print(f"Using latest recording: {recordings[0].name}\n")
    
    # Load test
    test = PerceptionGroundTruthTest(recording_file)
    
    # Show recording metadata
    try:
        with h5py.File(recording_file, 'r') as f:
            if "metadata" in f.attrs:
                import json
                metadata = json.loads(f.attrs["metadata"])
                rec_type = metadata.get("recording_type", "unknown")
                print(f"\nRecording Type: {rec_type}")
                if rec_type == "perception_replay" and "source_recording" in metadata:
                    print(f"Source: {metadata['source_recording']}")
                print()
    except:
        pass
    
    try:
        # Analyze each question
        results = {}
        results['q1'] = analyze_question_1(test)
        results['q2'] = analyze_question_2(test)
        results['q3'] = analyze_question_3(test)
        results['q4'] = analyze_question_4(test)
        results['q5'] = analyze_question_5(test)
        results['q6'] = analyze_question_6(test)
        results['q7'] = analyze_question_7(test)
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        for i, (key, result) in enumerate(results.items(), 1):
            status = result.get('status', 'UNKNOWN')
            status_symbol = '✓' if status == 'PASS' else '⚠️' if status == 'WARN' else '❌'
            print(f"{status_symbol} Question {i}: {status}")
        
        print("\n" + "="*80)
        
    finally:
        test.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

