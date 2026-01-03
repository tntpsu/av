#!/usr/bin/env python3
"""
Controller Tuning Tool using Unit Tests

This script runs controller performance tests to evaluate parameter configurations
without needing Unity. Much faster than full Unity runs!

Usage:
    python tools/tune_controller_with_tests.py
    python tools/tune_controller_with_tests.py --kp 0.4 --kd 0.5 --ki 0.003
"""

import argparse
import sys
import subprocess
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_test(test_name, extra_args=None):
    """Run a specific test and return pass/fail."""
    cmd = ["pytest", f"tests/{test_name}", "-v", "--tb=short"]
    if extra_args:
        cmd.extend(extra_args)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr


def run_controller_tests(kp=None, kd=None, ki=None):
    """
    Run controller performance tests with given parameters.
    
    Returns dict with test results.
    """
    # Set environment variables for test parameters
    env = {}
    if kp is not None:
        env['TEST_KP'] = str(kp)
    if kd is not None:
        env['TEST_KD'] = str(kd)
    if ki is not None:
        env['TEST_KI'] = str(ki)
    
    results = {}
    
    # Test 1: PID Integral Accumulation
    print("Running PID Integral Accumulation Tests...")
    passed, stdout, stderr = run_test("test_pid_integral_accumulation.py", env=env)
    results['integral_accumulation'] = {
        'passed': passed,
        'stdout': stdout,
        'stderr': stderr
    }
    
    # Test 2: Oscillation Detection
    print("Running Oscillation Detection Tests...")
    passed, stdout, stderr = run_test("test_oscillation_detection.py", env=env)
    results['oscillation'] = {
        'passed': passed,
        'stdout': stdout,
        'stderr': stderr
    }
    
    # Test 3: Integration Scenarios (convergence, stability)
    print("Running Integration Scenario Tests...")
    passed, stdout, stderr = run_test("test_integration_scenarios.py", env=env)
    results['integration'] = {
        'passed': passed,
        'stdout': stdout,
        'stderr': stderr
    }
    
    return results


def evaluate_configuration(kp, kd, ki):
    """
    Evaluate a controller configuration using tests.
    
    Returns a score and detailed results.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating Configuration: kp={kp}, kd={kd}, ki={ki}")
    print(f"{'='*60}\n")
    
    results = run_controller_tests(kp=kp, kd=kd, ki=ki)
    
    # Calculate score
    score = 0
    max_score = 0
    
    # Integral accumulation (weight: 3)
    max_score += 3
    if results['integral_accumulation']['passed']:
        score += 3
        print("✅ PID Integral Accumulation: PASSED")
    else:
        print("❌ PID Integral Accumulation: FAILED")
    
    # Oscillation (weight: 3)
    max_score += 3
    if results['oscillation']['passed']:
        score += 3
        print("✅ Oscillation Detection: PASSED")
    else:
        print("❌ Oscillation Detection: FAILED")
    
    # Integration (weight: 2)
    max_score += 2
    if results['integration']['passed']:
        score += 2
        print("✅ Integration Scenarios: PASSED")
    else:
        print("❌ Integration Scenarios: FAILED")
    
    score_percent = (score / max_score) * 100 if max_score > 0 else 0
    
    print(f"\nScore: {score}/{max_score} ({score_percent:.1f}%)")
    
    return {
        'kp': kp,
        'kd': kd,
        'ki': ki,
        'score': score,
        'max_score': max_score,
        'score_percent': score_percent,
        'results': results
    }


def parameter_sweep():
    """
    Perform a parameter sweep to find optimal values.
    """
    print("="*60)
    print("CONTROLLER PARAMETER SWEEP")
    print("="*60)
    print("\nThis will test multiple parameter combinations...")
    print("(This may take a few minutes)\n")
    
    # Parameter ranges to test
    kp_values = [0.3, 0.35, 0.4, 0.45, 0.5]
    kd_values = [0.4, 0.45, 0.5, 0.55, 0.6]
    ki_values = [0.002, 0.003, 0.004, 0.005]
    
    best_config = None
    best_score = -1
    all_results = []
    
    total_combinations = len(kp_values) * len(kd_values) * len(ki_values)
    current = 0
    
    for kp in kp_values:
        for kd in kd_values:
            for ki in ki_values:
                current += 1
                print(f"\n[{current}/{total_combinations}] Testing kp={kp}, kd={kd}, ki={ki}")
                
                result = evaluate_configuration(kp, kd, ki)
                all_results.append(result)
                
                if result['score_percent'] > best_score:
                    best_score = result['score_percent']
                    best_config = result
                    print(f"⭐ NEW BEST: Score {best_score:.1f}%")
    
    # Print summary
    print("\n" + "="*60)
    print("PARAMETER SWEEP SUMMARY")
    print("="*60)
    
    # Sort by score
    all_results.sort(key=lambda x: x['score_percent'], reverse=True)
    
    print("\nTop 5 Configurations:")
    for i, config in enumerate(all_results[:5], 1):
        print(f"{i}. kp={config['kp']}, kd={config['kd']}, ki={config['ki']}: "
              f"{config['score_percent']:.1f}%")
    
    print(f"\n🏆 BEST CONFIGURATION:")
    print(f"   kp={best_config['kp']}")
    print(f"   kd={best_config['kd']}")
    print(f"   ki={best_config['ki']}")
    print(f"   Score: {best_config['score_percent']:.1f}%")
    
    return best_config, all_results


def main():
    parser = argparse.ArgumentParser(
        description="Tune controller parameters using unit tests"
    )
    parser.add_argument('--kp', type=float, help='Proportional gain')
    parser.add_argument('--kd', type=float, help='Derivative gain')
    parser.add_argument('--ki', type=float, help='Integral gain')
    parser.add_argument('--sweep', action='store_true', 
                       help='Perform parameter sweep')
    
    args = parser.parse_args()
    
    if args.sweep:
        best_config, all_results = parameter_sweep()
        
        # Save results
        output_file = project_root / "tmp" / "controller_tuning_results.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump({
                'best_config': best_config,
                'all_results': all_results
            }, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    else:
        # Use current config or provided values
        from config.config_loader import load_config
        config = load_config()
        
        kp = args.kp if args.kp is not None else config['control']['lateral']['kp']
        kd = args.kd if args.kd is not None else config['control']['lateral']['kd']
        ki = args.ki if args.ki is not None else config['control']['lateral']['ki']
        
        result = evaluate_configuration(kp, kd, ki)
        
        print("\n" + "="*60)
        print("RECOMMENDATION:")
        print("="*60)
        if result['score_percent'] >= 80:
            print("✅ Configuration looks good! Consider running Unity tests to verify.")
        elif result['score_percent'] >= 60:
            print("⚠️  Configuration has some issues. Consider tuning parameters.")
        else:
            print("❌ Configuration has significant issues. Parameter tuning required.")


if __name__ == "__main__":
    main()


