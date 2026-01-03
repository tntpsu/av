#!/usr/bin/env python3
"""
Parameter Sweep for Controller Tuning

Tests multiple parameter combinations using realistic tests to find optimal values.
Much faster than Unity (0.05s per test vs 30s+ per Unity run).

Usage:
    python tools/parameter_sweep_tuning.py
"""

import subprocess
import sys
import yaml
import shutil
from pathlib import Path
import json

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
    
    config['control']['lateral']['kp'] = kp
    config['control']['lateral']['kd'] = kd
    config['control']['lateral']['ki'] = ki
    
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return config


def run_tests():
    """Run controller performance tests and return pass/fail."""
    tests = [
        "tests/test_pid_integral_accumulation_realistic.py",
        "tests/test_oscillation_detection.py::TestOscillationDetection::test_no_high_frequency_oscillation"
    ]
    
    all_passed = True
    for test in tests:
        result = subprocess.run(
            ["pytest", test, "-v", "--tb=line", "-q"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            all_passed = False
    
    return all_passed


def parameter_sweep():
    """Perform parameter sweep to find optimal values."""
    print("="*60)
    print("PARAMETER SWEEP FOR CONTROLLER TUNING")
    print("="*60)
    print("\nTesting multiple parameter combinations...")
    print("(This will take a few minutes)\n")
    
    # Parameter ranges to test
    kp_values = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
    kd_values = [0.4, 0.45, 0.5, 0.55, 0.6]
    ki_values = [0.002, 0.003, 0.004, 0.005]
    
    best_config = None
    best_score = -1
    all_results = []
    
    total_combinations = len(kp_values) * len(kd_values) * len(ki_values)
    current = 0
    
    # Backup config
    backup_file = backup_config()
    
    try:
        for kp in kp_values:
            for kd in kd_values:
                for ki in ki_values:
                    current += 1
                    print(f"[{current}/{total_combinations}] Testing kp={kp}, kd={kd}, ki={ki}...", end=" ")
                    
                    # Modify config
                    modify_config(kp, kd, ki)
                    
                    # Run tests
                    passed = run_tests()
                    
                    if passed:
                        score = 100  # All tests passed
                        print(f"✅ PASSED (score: {score})")
                    else:
                        score = 0  # Tests failed
                        print(f"❌ FAILED (score: {score})")
                    
                    result = {
                        'kp': kp,
                        'kd': kd,
                        'ki': ki,
                        'score': score,
                        'passed': passed
                    }
                    all_results.append(result)
                    
                    if score > best_score:
                        best_score = score
                        best_config = result
                        if passed:
                            print(f"  ⭐ NEW BEST!")
        
        # Restore config
        restore_config(backup_file)
        
        # Print summary
        print("\n" + "="*60)
        print("PARAMETER SWEEP SUMMARY")
        print("="*60)
        
        # Sort by score
        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Count passing configs
        passing_configs = [r for r in all_results if r['passed']]
        
        print(f"\nTotal combinations tested: {len(all_results)}")
        print(f"Passing configurations: {len(passing_configs)}")
        
        if passing_configs:
            print("\nTop 10 Passing Configurations:")
            for i, config in enumerate(passing_configs[:10], 1):
                print(f"{i}. kp={config['kp']}, kd={config['kd']}, ki={config['ki']}")
            
            print(f"\n🏆 BEST CONFIGURATION:")
            print(f"   kp={best_config['kp']}")
            print(f"   kd={best_config['kd']}")
            print(f"   ki={best_config['ki']}")
            print(f"   Score: {best_config['score']}%")
            
            # Save results
            output_file = project_root / "tmp" / "parameter_sweep_results.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump({
                    'best_config': best_config,
                    'all_results': all_results,
                    'passing_configs': passing_configs
                }, f, indent=2)
            
            print(f"\nResults saved to: {output_file}")
            
            # Apply best config
            print(f"\n💡 To apply best config, update config/av_stack_config.yaml:")
            print(f"   kp: {best_config['kp']}")
            print(f"   kd: {best_config['kd']}")
            print(f"   ki: {best_config['ki']}")
        else:
            print("\n⚠️  No configurations passed all tests!")
            print("   Consider relaxing test thresholds or investigating issues.")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Sweep interrupted by user")
        restore_config(backup_file)
    except Exception as e:
        print(f"\n\n❌ Error during sweep: {e}")
        restore_config(backup_file)
        raise


if __name__ == "__main__":
    parameter_sweep()


