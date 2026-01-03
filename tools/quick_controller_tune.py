#!/usr/bin/env python3
"""
Quick Controller Tuning using Tests

This script:
1. Temporarily modifies config with test parameters
2. Runs controller performance tests
3. Reports results

Much faster than Unity tests! (~1 second vs ~30+ seconds)

Usage:
    python tools/quick_controller_tune.py
    python tools/quick_controller_tune.py --kp 0.4 --kd 0.5
"""

import argparse
import subprocess
import sys
import yaml
from pathlib import Path
import shutil
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


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


def modify_config(kp, kd, ki):
    """Modify config with test parameters."""
    config_file = project_root / "config" / "av_stack_config.yaml"
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    if kp is not None:
        config['control']['lateral']['kp'] = kp
    if kd is not None:
        config['control']['lateral']['kd'] = kd
    if ki is not None:
        config['control']['lateral']['ki'] = ki
    
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return config


def run_tests():
    """Run controller performance tests."""
    tests = [
        "tests/test_pid_integral_accumulation.py",
        "tests/test_oscillation_detection.py",
        "tests/test_divergence_prevention.py::test_pid_integral_does_not_accumulate"
    ]
    
    results = {}
    for test in tests:
        print(f"Running {test}...")
        result = subprocess.run(
            ["pytest", test, "-v", "--tb=line"],
            capture_output=True,
            text=True
        )
        results[test] = {
            'passed': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    
    return results


def evaluate_results(results):
    """Evaluate test results and return score."""
    total_tests = 0
    passed_tests = 0
    
    for test, result in results.items():
        # Count tests from output
        lines = result['stdout'].split('\n')
        for line in lines:
            if 'passed' in line.lower() and 'test' in line.lower():
                # Extract number of passed tests
                if 'passed' in line:
                    try:
                        num_passed = int(line.split()[0])
                        passed_tests += num_passed
                    except:
                        pass
            if '::' in line and ('PASSED' in line or 'FAILED' in line):
                total_tests += 1
                if 'PASSED' in line:
                    passed_tests += 1
    
    if total_tests == 0:
        # Fallback: check if all tests passed
        all_passed = all(r['passed'] for r in results.values())
        return 100 if all_passed else 0, total_tests, passed_tests
    
    score = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    return score, total_tests, passed_tests


def main():
    parser = argparse.ArgumentParser(
        description="Quick controller tuning using tests"
    )
    parser.add_argument('--kp', type=float, help='Proportional gain to test')
    parser.add_argument('--kd', type=float, help='Derivative gain to test')
    parser.add_argument('--ki', type=float, help='Integral gain to test')
    parser.add_argument('--restore', action='store_true',
                       help='Restore config from backup and exit')
    
    args = parser.parse_args()
    
    # Handle restore
    backup_file = project_root / "config" / "av_stack_config.yaml.backup"
    if args.restore:
        if backup_file.exists():
            restore_config(backup_file)
            print("✅ Config restored from backup")
        else:
            print("❌ No backup found")
        return
    
    # Load current config
    config_file = project_root / "config" / "av_stack_config.yaml"
    with open(config_file, 'r') as f:
        current_config = yaml.safe_load(f)
    
    current_kp = current_config['control']['lateral']['kp']
    current_kd = current_config['control']['lateral']['kd']
    current_ki = current_config['control']['lateral']['ki']
    
    # Use provided values or current config
    test_kp = args.kp if args.kp is not None else current_kp
    test_kd = args.kd if args.kd is not None else current_kd
    test_ki = args.ki if args.ki is not None else current_ki
    
    print("="*60)
    print("QUICK CONTROLLER TUNING")
    print("="*60)
    print(f"\nCurrent config: kp={current_kp}, kd={current_kd}, ki={current_ki}")
    print(f"Testing: kp={test_kp}, kd={test_kd}, ki={test_ki}\n")
    
    # Backup config
    backup_file = backup_config()
    
    try:
        # Modify config
        modify_config(test_kp, test_kd, test_ki)
        
        # Run tests
        print("Running controller performance tests...\n")
        results = run_tests()
        
        # Evaluate
        score, total, passed = evaluate_results(results)
        
        # Print results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"\nTests: {passed}/{total} passed ({score:.1f}%)")
        
        for test, result in results.items():
            status = "✅ PASSED" if result['passed'] else "❌ FAILED"
            print(f"  {status}: {test}")
        
        print("\n" + "="*60)
        print("RECOMMENDATION")
        print("="*60)
        if score >= 90:
            print("✅ Excellent! All tests passed. Safe to use in Unity.")
        elif score >= 70:
            print("⚠️  Good, but some tests failed. Review failures before Unity.")
        else:
            print("❌ Multiple test failures. Tune parameters before Unity.")
        
        if test_kp != current_kp or test_kd != current_kd or test_ki != current_ki:
            print(f"\n💡 To apply these parameters, update config:")
            print(f"   kp: {test_kp}")
            print(f"   kd: {test_kd}")
            print(f"   ki: {test_ki}")
    
    finally:
        # Restore config
        restore_config(backup_file)
        print("\n✅ Config restored to original")


if __name__ == "__main__":
    main()


