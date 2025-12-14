#!/usr/bin/env python3
"""
Update PID Gains After Tuning

After running --tune-pid mode, use this script to update default gains across the codebase.

Usage:
    python scripts/update_pid_gains.py 0.6 0.7
    # Updates kp=0.6, max_vel=0.7 as new defaults
"""

import sys
import re
from pathlib import Path


def update_file(filepath, kp, max_vel):
    """Update PID gains in a specific file"""
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    
    # Pattern 1: VelocityPIDController(kp=X, max_vel=Y)
    content = re.sub(
        r'VelocityPIDController\(kp=[\d.]+,\s*max_vel=[\d.]+\)',
        f'VelocityPIDController(kp={kp}, max_vel={max_vel})',
        content
    )
    
    # Pattern 2: def __init__(self, ..., kp=X, max_vel=Y, ...)
    content = re.sub(
        r"(trajectory_type='[\w]+',\s*)kp=[\d.]+,\s*max_vel=[\d.]+",
        rf"\1kp={kp}, max_vel={max_vel}",
        content
    )
    
    # Pattern 3: Shell script defaults
    content = re.sub(
        r'kp=\$\{kp:-[\d.]+\}',
        f'kp=${{kp:-{kp}}}',
        content
    )
    content = re.sub(
        r'max_vel=\$\{max_vel:-[\d.]+\}',
        f'max_vel=${{max_vel:-{max_vel}}}',
        content
    )
    
    # Pattern 4: Default in class definition
    content = re.sub(
        r'def __init__\(self,\s*kp=[\d.]+,\s*max_vel=[\d.]+\)',
        f'def __init__(self, kp={kp}, max_vel={max_vel})',
        content
    )
    
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False


def main():
    if len(sys.argv) != 3:
        print("Usage: python scripts/update_pid_gains.py <kp> <max_vel>")
        print("Example: python scripts/update_pid_gains.py 0.6 0.7")
        sys.exit(1)
    
    try:
        kp = float(sys.argv[1])
        max_vel = float(sys.argv[2])
    except ValueError:
        print("Error: kp and max_vel must be numbers")
        sys.exit(1)
    
    print(f"\n🔧 Updating PID gains to: kp={kp}, max_vel={max_vel}")
    print("="*60)
    
    files_to_update = [
        'src/controllers/pid_controller.py',
        'src/real_drone/run_tello.py',
        'scripts/autonomous_data_collection.py',
        'collect_autonomous.sh',
    ]
    
    updated = []
    skipped = []
    
    for filepath in files_to_update:
        full_path = Path(filepath)
        if not full_path.exists():
            print(f"⚠️  {filepath} - NOT FOUND")
            continue
        
        if update_file(full_path, kp, max_vel):
            print(f"✓ {filepath}")
            updated.append(filepath)
        else:
            print(f"○ {filepath} - No changes needed")
            skipped.append(filepath)
    
    print("\n" + "="*60)
    print(f"✓ Updated {len(updated)} files")
    
    if updated:
        print("\nUpdated files:")
        for f in updated:
            print(f"  • {f}")
    
    print("\n📋 Next steps:")
    print("  1. Test with: ./collect_autonomous.sh")
    print("  2. Verify tracking error improved")
    print("  3. Collect training data with new gains")


if __name__ == '__main__':
    main()
