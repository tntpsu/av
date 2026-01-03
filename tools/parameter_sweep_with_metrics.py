#!/usr/bin/env python3
"""
Parameter Sweep with Performance Metrics

Tests multiple parameter combinations and ranks them by actual performance metrics
(integral accumulation, oscillation frequency, error convergence) rather than just pass/fail.

Usage:
    python tools/parameter_sweep_with_metrics.py
"""

import subprocess
import sys
import yaml
import shutil
import numpy as np
from pathlib import Path
import json
import re

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from control.pid_controller import LateralController


def backup_config():
    """Backup current config."""
    config_file = project_root / "config" / "av_stack_config.yaml"
    backup_file = project_root / "config" / "av_stack_config.yaml.backup"
    shutil.copy(config_file, backup_file)
    return backup_file


def restore_config(backup_file):
    """Restore config from backup."""
    config_file = project_root / "config" / "av_stack_config.yaml"
    shutil.copy(backup_file, config_file)
    backup_file.unlink()


def measure_integral_accumulation(kp, kd, ki):
    """
    Measure PID integral accumulation using realistic simulation.
    Returns accumulation ratio (last_third / first_third).
    """
    controller = LateralController(kp=kp, ki=ki, kd=kd, deadband=0.01)
    
    dt = 0.033
    num_frames = 600  # 20 seconds
    vehicle_speed = 8.0
    
    # Initial state
    vehicle_x = 0.0
    vehicle_y = 0.0
    vehicle_heading = 0.0
    initial_offset = 0.2  # Start 20cm off center
    vehicle_x = initial_offset
    
    # Reference tracking
    smoothed_ref_x = 0.0
    smoothing_alpha = 0.80  # Like Unity
    persistent_bias = 0.05  # 5cm persistent bias
    
    integrals = []
    lateral_errors = []
    np.random.seed(42)  # Deterministic
    
    for frame in range(num_frames):
        # Simulate varying reference with noise (like Unity)
        noise = np.random.normal(0, 0.02)  # 2cm noise
        actual_ref_x = noise + persistent_bias
        
        # Apply smoothing delay
        smoothed_ref_x = smoothing_alpha * smoothed_ref_x + (1 - smoothing_alpha) * actual_ref_x
        reference_x = smoothed_ref_x
        
        # Reference point
        reference_point = {
            'x': reference_x - vehicle_x,
            'y': 10.0,
            'heading': 0.0 - vehicle_heading,
            'velocity': vehicle_speed
        }
        
        # Compute steering
        result = controller.compute_steering(
            current_heading=vehicle_heading,
            reference_point=reference_point,
            vehicle_position=np.array([vehicle_x, vehicle_y]),
            return_metadata=True
        )
        
        integrals.append(abs(result['pid_integral']))
        lateral_errors.append(abs(result['lateral_error']))
        
        # Vehicle dynamics
        steering = result['steering']
        vehicle_heading += steering * 0.1 * dt
        vehicle_heading = np.arctan2(np.sin(vehicle_heading), np.cos(vehicle_heading))
        vehicle_x += np.sin(vehicle_heading) * vehicle_speed * dt
        vehicle_y += np.cos(vehicle_heading) * vehicle_speed * dt
    
    # Calculate accumulation
    first_third = integrals[:num_frames//3]
    last_third = integrals[2*num_frames//3:]
    mean_first = np.mean(first_third) if first_third else 0.001
    mean_last = np.mean(last_third) if last_third else 0.001
    accumulation = mean_last / mean_first if mean_first > 0 else 999
    
    # Calculate mean error
    mean_error = np.mean(lateral_errors)
    
    # Calculate oscillation (sign changes per second)
    sign_changes = sum(1 for i in range(1, len(lateral_errors)) 
                      if np.sign(lateral_errors[i]) != np.sign(lateral_errors[i-1]))
    oscillation_hz = sign_changes / (num_frames * dt)
    
    return {
        'accumulation': accumulation,
        'mean_error': mean_error,
        'oscillation_hz': oscillation_hz
    }


def calculate_score(metrics):
    """
    Calculate a composite score (lower is better).
    Penalizes high accumulation, high error, and high oscillation.
    """
    accumulation = metrics['accumulation']
    mean_error = metrics['mean_error']
    oscillation = metrics['oscillation_hz']
    
    # Ideal: accumulation < 1.5, error < 0.05m, oscillation < 5 Hz
    # Penalize deviations from ideal
    accumulation_penalty = max(0, (accumulation - 1.0) * 50)  # 50 points per 1x over 1.0
    error_penalty = max(0, (mean_error - 0.05) * 100)  # 100 points per 0.01m over 0.05m
    oscillation_penalty = max(0, (oscillation - 5.0) * 10)  # 10 points per Hz over 5 Hz
    
    total_score = accumulation_penalty + error_penalty + oscillation_penalty
    return total_score


def parameter_sweep():
    """Perform parameter sweep with performance metrics."""
    print("="*60)
    print("PARAMETER SWEEP WITH PERFORMANCE METRICS")
    print("="*60)
    print("\nTesting multiple parameter combinations...")
    print("Measuring: integral accumulation, mean error, oscillation\n")
    
    # Focused parameter ranges (around current best)
    kp_values = [0.3, 0.35, 0.4, 0.45, 0.5]
    kd_values = [0.4, 0.45, 0.5, 0.55, 0.6]
    ki_values = [0.002, 0.003, 0.004]
    
    all_results = []
    
    total_combinations = len(kp_values) * len(kd_values) * len(ki_values)
    current = 0
    
    for kp in kp_values:
        for kd in kd_values:
            for ki in ki_values:
                current += 1
                print(f"[{current}/{total_combinations}] kp={kp}, kd={kd}, ki={ki}...", end=" ")
                
                try:
                    metrics = measure_integral_accumulation(kp, kd, ki)
                    score = calculate_score(metrics)
                    
                    result = {
                        'kp': kp,
                        'kd': kd,
                        'ki': ki,
                        'accumulation': metrics['accumulation'],
                        'mean_error': metrics['mean_error'],
                        'oscillation_hz': metrics['oscillation_hz'],
                        'score': score
                    }
                    all_results.append(result)
                    
                    print(f"acc={metrics['accumulation']:.2f}x, err={metrics['mean_error']:.4f}m, "
                          f"osc={metrics['oscillation_hz']:.2f}Hz, score={score:.1f}")
                except Exception as e:
                    print(f"ERROR: {e}")
                    result = {
                        'kp': kp,
                        'kd': kd,
                        'ki': ki,
                        'accumulation': 999,
                        'mean_error': 999,
                        'oscillation_hz': 999,
                        'score': 9999
                    }
                    all_results.append(result)
    
    # Sort by score (lower is better)
    all_results.sort(key=lambda x: x['score'])
    
    # Print summary
    print("\n" + "="*60)
    print("PARAMETER SWEEP SUMMARY")
    print("="*60)
    
    print(f"\nTotal combinations tested: {len(all_results)}")
    print(f"\nTop 10 Configurations (by composite score):")
    print("-" * 60)
    print(f"{'Rank':<6} {'kp':<6} {'kd':<6} {'ki':<8} {'Accum':<8} {'Error':<8} {'Osc':<8} {'Score':<8}")
    print("-" * 60)
    
    for i, config in enumerate(all_results[:10], 1):
        print(f"{i:<6} {config['kp']:<6.2f} {config['kd']:<6.2f} {config['ki']:<8.4f} "
              f"{config['accumulation']:<8.2f} {config['mean_error']:<8.4f} "
              f"{config['oscillation_hz']:<8.2f} {config['score']:<8.1f}")
    
    best = all_results[0]
    print(f"\n🏆 BEST CONFIGURATION:")
    print(f"   kp={best['kp']}")
    print(f"   kd={best['kd']}")
    print(f"   ki={best['ki']}")
    print(f"   Accumulation: {best['accumulation']:.2f}x (ideal: <1.5x)")
    print(f"   Mean Error: {best['mean_error']:.4f}m (ideal: <0.05m)")
    print(f"   Oscillation: {best['oscillation_hz']:.2f}Hz (ideal: <5Hz)")
    print(f"   Composite Score: {best['score']:.1f} (lower is better)")
    
    # Save results
    output_file = project_root / "tmp" / "parameter_sweep_metrics.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'best_config': best,
            'all_results': all_results,
            'top_10': all_results[:10]
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Apply best config
    print(f"\n💡 To apply best config, update config/av_stack_config.yaml:")
    print(f"   kp: {best['kp']}")
    print(f"   kd: {best['kd']}")
    print(f"   ki: {best['ki']}")


if __name__ == "__main__":
    parameter_sweep()


