#!/usr/bin/env python3
"""
Installation Test Script

This script verifies that all required dependencies are correctly installed.
Run this after setting up the environment to ensure everything is working.
"""

import sys
import importlib


def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    display_name = package_name or module_name
    try:
        importlib.import_module(module_name)
        print(f"✓ {display_name}")
        return True
    except ImportError as e:
        print(f"✗ {display_name}: {e}")
        return False


def check_version(module_name, min_version=None):
    """Check module version."""
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        version_str = f" (version: {version})"
        
        if min_version and version != 'unknown':
            # Simple version comparison (works for most cases)
            if version >= min_version:
                print(f"✓ {module_name}{version_str}")
                return True
            else:
                print(f"✗ {module_name}{version_str} - minimum required: {min_version}")
                return False
        else:
            print(f"✓ {module_name}{version_str}")
            return True
    except ImportError as e:
        print(f"✗ {module_name}: {e}")
        return False


def main():
    """Run all installation checks."""
    print("=" * 60)
    print("Drone Hybrid RL+PID - Installation Test")
    print("=" * 60)
    
    print(f"\nPython version: {sys.version}")
    
    # Core dependencies
    print("\n--- Core Dependencies ---")
    checks = []
    checks.append(check_version('numpy', '1.21.0'))
    checks.append(check_version('torch', '1.12.0'))
    checks.append(check_import('gymnasium'))
    checks.append(check_import('stable_baselines3'))
    
    # Simulation
    print("\n--- Simulation ---")
    checks.append(check_import('pybullet'))
    checks.append(check_import('gym_pybullet_drones', 'gym-pybullet-drones'))
    
    # Real drone
    print("\n--- Real Drone Interface ---")
    checks.append(check_import('djitellopy'))
    checks.append(check_import('cv2', 'opencv-python'))
    
    # Utilities
    print("\n--- Utilities ---")
    checks.append(check_import('yaml', 'pyyaml'))
    checks.append(check_import('matplotlib'))
    checks.append(check_import('pandas'))
    checks.append(check_import('scipy'))
    checks.append(check_import('tqdm'))
    
    # Optional
    print("\n--- Optional Dependencies ---")
    check_import('wandb')
    check_import('tensorboard')
    check_import('plotly')
    
    # Summary
    print("\n" + "=" * 60)
    total_required = len(checks)
    passed = sum(checks)
    
    if passed == total_required:
        print(f"✓ All {total_required} required checks passed!")
        print("\nYou're ready to start training!")
        print("\nQuick start:")
        print("  python scripts/train_pid.py")
        print("  python scripts/train_rl.py --timesteps 500000")
        return 0
    else:
        print(f"✗ {total_required - passed} check(s) failed")
        print("\nPlease install missing dependencies:")
        print("  pip install -r requirements.txt")
        print("or")
        print("  conda env create -f environment.yml")
        return 1


if __name__ == '__main__':
    sys.exit(main())
