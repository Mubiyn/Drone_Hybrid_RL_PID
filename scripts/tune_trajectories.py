#!/usr/bin/env python3
"""
Tune PID Gains for Each Trajectory

Tests different PID configurations to find optimal gains for each trajectory type.
Runs multiple quick tests and reports the best configuration.
"""

import argparse
import time
import numpy as np
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from djitellopy import Tello


def load_trajectory(traj_file):
    """Load trajectory from pickle file"""
    with open(traj_file, 'rb') as f:
        return pickle.load(f)


def test_trajectory_tracking(tello, trajectory, kp, max_vel, duration=20.0):
    """
    Test a single PID configuration
    
    Returns tracking error statistics
    """
    from src.controllers.pid_controller import VelocityPIDController
    
    controller = VelocityPIDController(kp=kp, max_vel=max_vel)
    
    waypoints = trajectory['waypoints']
    waypoint_times = trajectory['waypoint_times']
    
    # Special handling for hover (single waypoint)
    is_hover = len(waypoints) == 1
    
    if is_hover:
        # For hover, just use the single target position
        target_pos_hover = waypoints[0]
        print(f"  Hover mode: target = {target_pos_hover}")
    else:
        # Split into approach and loop for normal trajectories
        approach_end_idx = 0
        for i in range(len(waypoints)):
            if abs(waypoints[i, 2] - waypoints[-1, 2]) < 0.05:
                approach_end_idx = i
                break
        
        approach_duration = waypoint_times[approach_end_idx] if approach_end_idx > 0 else 0.0
        loop_duration = waypoint_times[-1] - approach_duration
        
        print(f"  Approach: {approach_duration:.1f}s, Loop: {loop_duration:.1f}s")
    
    # Takeoff
    tello.takeoff()
    time.sleep(3)
    
    states = []
    errors = []
    
    start_time = time.time()
    rate = 20  # Hz
    dt = 1.0 / rate
    
    try:
        while time.time() - start_time < duration:
            loop_start = time.time()
            t_elapsed = time.time() - start_time
            
            # Get current state
            height = tello.get_height() / 100.0
            
            # Simple state (no MoCap needed for tuning)
            state = np.array([0, 0, height, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            
            # Get target from trajectory
            if is_hover:
                # Hover: always use same target position
                target_pos = target_pos_hover
                target_vel = np.array([0, 0, 0])
            elif approach_end_idx > 0 and t_elapsed < approach_duration:
                # Approach phase
                idx = int((t_elapsed / approach_duration) * approach_end_idx)
                idx = min(idx, approach_end_idx)
                target_pos = waypoints[idx]
                if idx < approach_end_idx:
                    target_vel = (waypoints[idx + 1] - waypoints[idx]) / (waypoint_times[idx + 1] - waypoint_times[idx])
                else:
                    target_vel = np.array([0, 0, 0])
            else:
                # Loop phase
                t_loop = (t_elapsed - approach_duration) % loop_duration
                loop_waypoints = waypoints[approach_end_idx:]
                loop_times = waypoint_times[approach_end_idx:] - approach_duration
                
                idx = 0
                for i in range(len(loop_times) - 1):
                    if loop_times[i] <= t_loop <= loop_times[i + 1]:
                        idx = i
                        break
                
                idx = min(idx, len(loop_waypoints) - 2)
                alpha = (t_loop - loop_times[idx]) / (loop_times[idx + 1] - loop_times[idx])
                alpha = np.clip(alpha, 0, 1)
                
                target_pos = loop_waypoints[idx] + alpha * (loop_waypoints[idx + 1] - loop_waypoints[idx])
                target_vel = (loop_waypoints[idx + 1] - loop_waypoints[idx]) / (loop_times[idx + 1] - loop_times[idx])
            
            # Compute error
            error = np.linalg.norm(state[0:3] - target_pos)
            errors.append(error)
            states.append(state.copy())
            
            # Get control action (velocity command)
            target = np.concatenate([target_pos, target_vel])
            
            # Compute velocity command (only needs obs and target_pos)
            action = controller.compute_control(state, target_pos)
            
            # Convert to Tello commands (body frame)
            vx, vy, vz, yaw_rate = action
            vx_cm = int(vx * 100)
            vy_cm = int(vy * 100)
            vz_cm = int(vz * 100)
            yaw_deg = int(np.degrees(yaw_rate))
            
            tello.send_rc_control(vy_cm, vx_cm, vz_cm, yaw_deg)
            
            # Maintain control rate
            elapsed = time.time() - loop_start
            sleep_time = max(0, dt - elapsed)
            time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\n  Interrupted!")
    except Exception as e:
        print(f"\n  Error: {e}")
    finally:
        # Stop and land
        tello.send_rc_control(0, 0, 0, 0)
        time.sleep(0.2)
        tello.land()
        time.sleep(3)
    
    if len(errors) == 0:
        return None
    
    errors = np.array(errors)
    return {
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'max_error': np.max(errors),
        'samples': len(errors),
    }


def tune_trajectory(traj_name, traj_file, num_tests=5):
    """
    Tune PID gains for a specific trajectory
    
    Tests multiple configurations and finds the best one.
    """
    print(f"\n{'='*70}")
    print(f"TUNING: {traj_name.upper()}")
    print(f"{'='*70}\n")
    
    trajectory = load_trajectory(traj_file)
    print(f"Trajectory: {len(trajectory['waypoints'])} waypoints, {trajectory['waypoint_times'][-1]:.1f}s duration")
    
    # Connect to Tello
    print("\nConnecting to Tello...")
    tello = Tello()
    try:
        tello.connect()
        battery = tello.get_battery()
        print(f"✓ Connected! Battery: {battery}%")
        
        if battery < 10:
            print("✗ Battery too low! Need at least 10% for tuning.")
            return None
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return None
    
    # Test configurations (conservative for tuning)
    configs = [
        (0.4, 0.5),  # Conservative
        (0.5, 0.6),  # Moderate
        (0.6, 0.7),  # Moderate-aggressive
        (0.7, 0.8),  # Aggressive
        (0.8, 0.9),  # Very aggressive
    ][:num_tests]
    
    results = []
    
    print(f"\nTesting {len(configs)} configurations...")
    print("Press Ctrl+C during a test to skip to next configuration\n")
    
    for idx, (kp, max_vel) in enumerate(configs, 1):
        print(f"\n{'-'*70}")
        print(f"Test {idx}/{len(configs)}: kp={kp}, max_vel={max_vel}")
        print(f"{'-'*70}")
        
        battery = tello.get_battery()
        print(f"Battery: {battery}%")
        
        if battery < 25:
            print("✗ Battery too low to continue tuning")
            break
        
        input("Press ENTER to start test (or Ctrl+C to skip)...")
        
        result = test_trajectory_tracking(tello, trajectory, kp, max_vel, duration=10.0)
        
        if result:
            result['kp'] = kp
            result['max_vel'] = max_vel
            results.append(result)
            
            print(f"\n✓ Results:")
            print(f"  Mean Error: {result['mean_error']:.4f}m")
            print(f"  Std Error:  {result['std_error']:.4f}m")
            print(f"  Max Error:  {result['max_error']:.4f}m")
        else:
            print("✗ Test failed or no data collected")
        
        print("\nRest period: 5 seconds...")
        time.sleep(5)
    
    # Cleanup
    try:
        tello.end()
    except:
        pass
    
    if not results:
        print("\n✗ No successful tests!")
        return None
    
    # Find best configuration
    results.sort(key=lambda x: x['mean_error'])
    
    print(f"\n{'='*70}")
    print(f"TUNING RESULTS: {traj_name.upper()}")
    print(f"{'='*70}\n")
    
    print(f"{'Rank':<6} {'kp':<6} {'max_vel':<10} {'Mean Error':<12} {'Std Error':<12} {'Max Error':<12}")
    print(f"{'-'*70}")
    for idx, r in enumerate(results, 1):
        print(f"{idx:<6} {r['kp']:<6} {r['max_vel']:<10} {r['mean_error']:<12.4f} {r['std_error']:<12.4f} {r['max_error']:<12.4f}")
    
    best = results[0]
    print(f"\n{'='*70}")
    print(f"BEST CONFIGURATION:")
    print(f"  kp = {best['kp']}")
    print(f"  max_vel = {best['max_vel']}")
    print(f"  Mean Error: {best['mean_error']:.4f}m")
    print(f"{'='*70}\n")
    
    return best


def main():
    parser = argparse.ArgumentParser(description="Tune PID gains for trajectories")
    parser.add_argument('--trajectory', type=str, 
                       choices=['circle', 'square', 'figure8', 'spiral', 'hover'],
                       help='Trajectory to tune (if not specified, tune all)')
    parser.add_argument('--num-tests', type=int, default=5,
                       help='Number of configurations to test per trajectory')
    parser.add_argument('--output', type=str, default='tuning_results.txt',
                       help='Output file for tuning results')
    
    args = parser.parse_args()
    
    traj_dir = Path('data/expert_trajectories')
    
    if args.trajectory:
        trajectories = [args.trajectory]
    else:
        trajectories = ['circle', 'square', 'figure8', 'spiral', 'hover']
    
    all_results = {}
    
    for traj_name in trajectories:
        traj_file = traj_dir / f'perfect_{traj_name}_trajectory.pkl'
        
        if not traj_file.exists():
            print(f"✗ Trajectory file not found: {traj_file}")
            continue
        
        best = tune_trajectory(traj_name, traj_file, args.num_tests)
        
        if best:
            all_results[traj_name] = best
        
        if traj_name != trajectories[-1]:
            print("\nRest between trajectories: 10 seconds...")
            time.sleep(10)
    
    # Save results
    output_file = Path(args.output)
    with open(output_file, 'w') as f:
        f.write("PID Tuning Results\n")
        f.write("=" * 70 + "\n\n")
        
        for traj_name, result in all_results.items():
            f.write(f"{traj_name.upper()}:\n")
            f.write(f"  kp = {result['kp']}\n")
            f.write(f"  max_vel = {result['max_vel']}\n")
            f.write(f"  Mean Error: {result['mean_error']:.4f}m\n")
            f.write(f"  Std Error: {result['std_error']:.4f}m\n")
            f.write(f"  Max Error: {result['max_error']:.4f}m\n")
            f.write("\n")
    
    print(f"\n✓ Tuning results saved to: {output_file}")
    
    print("\n" + "="*70)
    print("RECOMMENDED GAINS FOR DATA COLLECTION:")
    print("="*70)
    for traj_name, result in all_results.items():
        print(f"{traj_name:8}: kp={result['kp']}, max_vel={result['max_vel']}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
