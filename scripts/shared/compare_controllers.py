#!/usr/bin/env python3
"""
Compare Open-Loop vs Hybrid RL on Real Tello

Tests both controllers with the same trajectory and compares performance.
This is the final test for your project: Does Hybrid RL outperform open-loop
under domain randomization (added weight, wind, etc.)?
"""

import sys
import time
import numpy as np
import argparse
from pathlib import Path
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))

from djitellopy import Tello
from stable_baselines3 import PPO


def load_trajectory(trajectory_file):
    """Load trajectory from pkl file"""
    with open(trajectory_file, 'rb') as f:
        traj = pickle.load(f)
    return traj


def get_tello_state(tello):
    """Get 12D state from Tello sensors (matches training data format)"""
    state = np.zeros(12)
    
    # Position [x, y, z] - using barometer for Z, dead reckoning for X,Y
    state[2] = tello.get_height() / 100.0  # cm to m
    # X, Y are relative (no absolute position without MoCap)
    state[0] = 0.0  # Relative X
    state[1] = 0.0  # Relative Y
    
    # Orientation [roll, pitch, yaw]
    state[3] = np.radians(tello.get_roll())
    state[4] = np.radians(tello.get_pitch())
    state[5] = np.radians(tello.get_yaw())
    
    # Velocity [vx, vy, vz]
    state[6] = tello.get_speed_x() / 100.0  # dm/s to m/s
    state[7] = tello.get_speed_y() / 100.0
    state[8] = tello.get_speed_z() / 100.0
    
    # Angular velocity [wx, wy, wz] - not available from Tello, set to 0
    state[9:12] = 0.0
    
    return state


def interpolate_trajectory(waypoints, waypoint_times, t_elapsed, approach_duration, loop_duration, approach_end_idx):
    """Interpolate trajectory (matches autonomous_data_collection.py logic)"""
    # During approach phase
    if approach_end_idx > 0 and t_elapsed < approach_duration:
        t_traj = t_elapsed
        times = waypoint_times[:approach_end_idx+1]
        wps = waypoints[:approach_end_idx+1]
    else:
        # Loop phase
        t_after_approach = t_elapsed - approach_duration
        t_traj = t_after_approach % loop_duration
        # Adjust times and waypoints for loop
        times = waypoint_times[approach_end_idx:] - waypoint_times[approach_end_idx]
        wps = waypoints[approach_end_idx:]
    
    # Interpolate
    idx = np.searchsorted(times, t_traj)
    if idx == 0:
        return wps[0], np.zeros(3)
    elif idx >= len(wps):
        return wps[-1], np.zeros(3)
    else:
        alpha = (t_traj - times[idx-1]) / (times[idx] - times[idx-1])
        target_pos = (1 - alpha) * wps[idx-1] + alpha * wps[idx]
        
        # Velocity
        if idx < len(wps) - 1:
            dt = times[idx+1] - times[idx]
            target_vel = (wps[idx+1] - wps[idx]) / dt
        else:
            dt = times[idx] - times[idx-1]
            target_vel = (wps[idx] - wps[idx-1]) / dt
        
        return target_pos, target_vel


def run_open_loop(tello, trajectory_file, duration):
    """Open-loop trajectory following (baseline)"""
    print(f"\n{'='*60}")
    print("OPEN-LOOP CONTROLLER (Baseline)")
    print(f"{'='*60}\n")
    
    traj = load_trajectory(trajectory_file)
    waypoints = traj['waypoints']
    waypoint_times = traj['waypoint_times']
    
    # Determine approach vs loop phase
    approach_end_idx = 0
    for i in range(len(waypoints)):
        if abs(waypoints[i, 2] - waypoints[-1, 2]) < 0.05:
            approach_end_idx = i
            break
    
    approach_duration = waypoint_times[approach_end_idx] if approach_end_idx > 0 else 0.0
    loop_duration = waypoint_times[-1] - approach_duration
    
    print(f"Trajectory: {traj.get('trajectory_label', 'unknown')}")
    print(f"Approach: {approach_duration:.1f}s, Loop: {loop_duration:.1f}s")
    print(f"Duration: {duration}s\n")
    
    print("Takeoff...")
    tello.takeoff()
    time.sleep(3)
    
    start_time = time.time()
    control_rate = 20  # Hz
    dt = 1.0 / control_rate
    
    errors = []
    
    try:
        while time.time() - start_time < duration:
            loop_start = time.time()
            t_elapsed = time.time() - start_time
            
            # Get state
            state = get_tello_state(tello)
            
            # Get target
            target_pos, target_vel = interpolate_trajectory(
                waypoints, waypoint_times, t_elapsed, approach_duration, loop_duration, approach_end_idx
            )
            
            # Transform to body frame
            yaw = state[5]
            vx_body = target_vel[0] * np.cos(yaw) + target_vel[1] * np.sin(yaw)
            vy_body = -target_vel[0] * np.sin(yaw) + target_vel[1] * np.cos(yaw)
            
            # Send RC commands
            lr_cmd = int(np.clip(vy_body * 100, -100, 100))
            fb_cmd = int(np.clip(vx_body * 100, -100, 100))
            ud_cmd = int(np.clip(target_vel[2] * 100, -100, 100))
            
            tello.send_rc_control(lr_cmd, fb_cmd, ud_cmd, 0)
            
            # Track error (Z only, since X/Y unknown)
            z_error = abs(state[2] - target_pos[2])
            errors.append(z_error)
            
            if int(t_elapsed) % 5 == 0 and t_elapsed - int(t_elapsed) < dt:
                print(f"t={t_elapsed:.1f}s | Z: {state[2]:.2f}m | Target Z: {target_pos[2]:.2f}m | "
                      f"Cmd: [LR:{lr_cmd}, FB:{fb_cmd}, UD:{ud_cmd}] | Bat: {tello.get_battery()}%")
            
            # Maintain rate
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    
    except KeyboardInterrupt:
        print("\n⚠️  Stopped by user")
    
    finally:
        print("\nLanding...")
        tello.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5)
        tello.land()
        time.sleep(2)
    
    return errors


def run_hybrid(tello, model_path, trajectory_file, duration):
    """Hybrid RL controller"""
    print(f"\n{'='*60}")
    print("HYBRID RL CONTROLLER")
    print(f"{'='*60}\n")
    
    # Load model
    model = PPO.load(model_path)
    print(f"Loaded model: {model_path}\n")
    
    traj = load_trajectory(trajectory_file)
    waypoints = traj['waypoints']
    waypoint_times = traj['waypoint_times']
    
    # Same trajectory parsing as open-loop
    approach_end_idx = 0
    for i in range(len(waypoints)):
        if abs(waypoints[i, 2] - waypoints[-1, 2]) < 0.05:
            approach_end_idx = i
            break
    
    approach_duration = waypoint_times[approach_end_idx] if approach_end_idx > 0 else 0.0
    loop_duration = waypoint_times[-1] - approach_duration
    
    print("Takeoff...")
    tello.takeoff()
    time.sleep(3)
    
    start_time = time.time()
    control_rate = 20
    dt = 1.0 / control_rate
    
    errors = []
    
    try:
        while time.time() - start_time < duration:
            loop_start = time.time()
            t_elapsed = time.time() - start_time
            
            # Get state
            state = get_tello_state(tello)
            
            # Get target
            target_pos, target_vel = interpolate_trajectory(
                waypoints, waypoint_times, t_elapsed, approach_duration, loop_duration, approach_end_idx
            )
            
            # Build 18D observation for model
            pos_error = target_pos - state[0:3]
            vel_error = target_vel - state[6:9]
            obs_18d = np.concatenate([state, pos_error, vel_error])
            
            # Get RL action
            action, _ = model.predict(obs_18d.reshape(1, -1), deterministic=True)
            action = action[0]  # [vx, vy, vz, yaw_rate]
            
            # Transform to body frame
            yaw = state[5]
            vx_body = action[0] * np.cos(yaw) + action[1] * np.sin(yaw)
            vy_body = -action[0] * np.sin(yaw) + action[1] * np.cos(yaw)
            
            # Send commands
            lr_cmd = int(np.clip(vy_body * 100, -100, 100))
            fb_cmd = int(np.clip(vx_body * 100, -100, 100))
            ud_cmd = int(np.clip(action[2] * 100, -100, 100))
            yaw_cmd = int(np.clip(action[3] * 100, -100, 100))
            
            tello.send_rc_control(lr_cmd, fb_cmd, ud_cmd, yaw_cmd)
            
            # Track error
            z_error = abs(state[2] - target_pos[2])
            errors.append(z_error)
            
            if int(t_elapsed) % 5 == 0 and t_elapsed - int(t_elapsed) < dt:
                print(f"t={t_elapsed:.1f}s | Z: {state[2]:.2f}m | Target Z: {target_pos[2]:.2f}m | "
                      f"Cmd: [LR:{lr_cmd}, FB:{fb_cmd}, UD:{ud_cmd}] | Bat: {tello.get_battery()}%")
            
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    
    except KeyboardInterrupt:
        print("\n⚠️  Stopped by user")
    
    finally:
        print("\nLanding...")
        tello.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5)
        tello.land()
        time.sleep(2)
    
    return errors


def main():
    parser = argparse.ArgumentParser(description='Compare Open-Loop vs Hybrid RL on Tello')
    
    parser.add_argument('--controller', type=str, required=True,
                        choices=['open-loop', 'hybrid', 'both'],
                        help='Controller to test')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained Hybrid model (required for hybrid/both)')
    parser.add_argument('--trajectory-file', type=str, required=True,
                        help='Path to trajectory .pkl file')
    parser.add_argument('--duration', type=float, default=20.0,
                        help='Flight duration in seconds')
    parser.add_argument('--perturbation', type=str, default=None,
                        help='Type of perturbation (e.g., "weight_5g", "wind", etc.)')
    
    args = parser.parse_args()
    
    # Validate
    if not Path(args.trajectory_file).exists():
        print(f" Trajectory file not found: {args.trajectory_file}")
        return
    
    if args.controller in ['hybrid', 'both']:
        if args.model is None:
            print(" --model required for hybrid controller")
            return
        if not Path(args.model).exists():
            if Path(args.model + '.zip').exists():
                args.model = args.model + '.zip'
            else:
                print(f" Model not found: {args.model}")
                return
    
    # Connect to Tello
    print("\n🔌 Connecting to Tello...")
    tello = Tello()
    tello.connect()
    print(f"✓ Connected | Battery: {tello.get_battery()}%")
    
    if args.perturbation:
        print(f"\n⚠️  Perturbation: {args.perturbation}")
        print("   Make sure perturbation is applied before starting!")
        input("   Press Enter to continue...")
    
    results = {}
    
    try:
        if args.controller in ['open-loop', 'both']:
            print("\n" + "="*60)
            print("TEST 1: OPEN-LOOP BASELINE")
            print("="*60)
            input("Press Enter to start open-loop flight...")
            
            errors_openloop = run_open_loop(tello, args.trajectory_file, args.duration)
            results['open-loop'] = {
                'mean_error': np.mean(errors_openloop),
                'max_error': np.max(errors_openloop),
                'std_error': np.std(errors_openloop)
            }
            
            if args.controller == 'both':
                print(f"\n⏸️  Rest for battery recovery...")
                time.sleep(30)
        
        if args.controller in ['hybrid', 'both']:
            print("\n" + "="*60)
            print("TEST 2: HYBRID RL")
            print("="*60)
            input("Press Enter to start hybrid flight...")
            
            errors_hybrid = run_hybrid(tello, args.model, args.trajectory_file, args.duration)
            results['hybrid'] = {
                'mean_error': np.mean(errors_hybrid),
                'max_error': np.max(errors_hybrid),
                'std_error': np.std(errors_hybrid)
            }
    
    finally:
        tello.end()
    
    # Summary
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    for controller, metrics in results.items():
        print(f"\n{controller.upper()}:")
        print(f"  Mean Z Error: {metrics['mean_error']:.3f}m ± {metrics['std_error']:.3f}m")
        print(f"  Max Z Error:  {metrics['max_error']:.3f}m")
    
    if len(results) == 2:
        improvement = (results['open-loop']['mean_error'] - results['hybrid']['mean_error']) / results['open-loop']['mean_error'] * 100
        print(f"\n{'='*60}")
        if improvement > 0:
            print(f"🎉 Hybrid RL is {improvement:.1f}% better than open-loop!")
        else:
            print(f"⚠️  Open-loop is {-improvement:.1f}% better than Hybrid RL")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
