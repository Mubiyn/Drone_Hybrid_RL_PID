#!/usr/bin/env python3
"""
Test All Controllers with Domain Randomization (Wind Perturbations)

Tests both PID baseline and Hybrid RL controllers under wind disturbances.
Collects data for comprehensive performance comparison.
"""

import sys
import time
import argparse
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from djitellopy import Tello

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.controllers.pid_controller import VelocityPIDController


def load_trajectory(filepath):
    """Load trajectory from pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_tello_state(tello):
    """Get current Tello state (12D)"""
    vx = tello.get_speed_x() / 100.0
    vy = tello.get_speed_y() / 100.0
    vz = tello.get_speed_z() / 100.0
    
    pitch = tello.get_pitch() * np.pi / 180.0
    roll = tello.get_roll() * np.pi / 180.0
    yaw = tello.get_yaw() * np.pi / 180.0
    
    height = tello.get_height() / 100.0
    
    wx = wy = wz = 0.0
    
    return np.array([
        0, 0, height,
        roll, pitch, yaw,
        vx, vy, vz,
        wx, wy, wz
    ])


def interpolate_trajectory(waypoints, waypoint_times, t_elapsed, approach_duration, loop_duration, approach_end_idx):
    """Interpolate trajectory position and velocity"""
    if t_elapsed < approach_duration:
        t_phase = t_elapsed
        times_phase = waypoint_times[:approach_end_idx+1]
        waypoints_phase = waypoints[:approach_end_idx+1]
    else:
        t_in_loop = (t_elapsed - approach_duration) % loop_duration
        t_phase = approach_duration + t_in_loop
        times_phase = waypoint_times[approach_end_idx:]
        waypoints_phase = waypoints[approach_end_idx:]
    
    idx = np.searchsorted(times_phase, t_phase)
    if idx == 0:
        return waypoints_phase[0], np.zeros(3)
    if idx >= len(times_phase):
        return waypoints_phase[-1], np.zeros(3)
    
    t0, t1 = times_phase[idx-1], times_phase[idx]
    p0, p1 = waypoints_phase[idx-1], waypoints_phase[idx]
    alpha = (t_phase - t0) / (t1 - t0) if t1 > t0 else 0
    
    pos = p0 + alpha * (p1 - p0)
    vel = (p1 - p0) / (t1 - t0) if t1 > t0 else np.zeros(3)
    
    return pos, vel


def test_pid_controller(tello, trajectory_file, trajectory_type, duration, perturbation_type):
    """Test PID baseline controller"""
    print(f"\n{'='*60}")
    print(f"PID CONTROLLER - {trajectory_type.upper()}")
    print(f"Perturbation: {perturbation_type}")
    print(f"{'='*60}\n")
    
    # Load trajectory
    traj = load_trajectory(trajectory_file)
    waypoints = traj['waypoints']
    waypoint_times = traj['waypoint_times']
    
    # Tuned PID gains
    TUNED_GAINS = {
        'circle': {'kp': 0.8, 'max_vel': 0.9},
        'square': {'kp': 0.7, 'max_vel': 0.8},
        'figure8': {'kp': 0.4, 'max_vel': 0.5},
        'spiral': {'kp': 0.8, 'max_vel': 0.9},
        'hover': {'kp': 0.6, 'max_vel': 0.7}
    }
    
    gains = TUNED_GAINS.get(trajectory_type, {'kp': 1.0, 'max_vel': 1.0})
    pid_controller = VelocityPIDController(kp=gains['kp'], max_vel=gains['max_vel'])
    
    # Parse trajectory
    approach_end_idx = 0
    for i in range(len(waypoints)):
        if abs(waypoints[i, 2] - waypoints[-1, 2]) < 0.05:
            approach_end_idx = i
            break
    
    approach_duration = waypoint_times[approach_end_idx] if approach_end_idx > 0 else 0.0
    loop_duration = waypoint_times[-1] - approach_duration
    
    # Takeoff
    print("Takeoff...")
    tello.takeoff()
    time.sleep(3)
    
    # Initialize tracking
    position = np.array([0.0, 0.0, tello.get_height() / 100.0])
    last_time = time.time()
    start_time = time.time()
    control_rate = 20
    dt = 1.0 / control_rate
    
    # Data collection
    states_history = []
    targets_history = []
    actions_history = []
    errors_history = []
    battery_history = []
    timestamps = []
    
    print("Starting PID control...\n")
    
    try:
        while time.time() - start_time < duration:
            loop_start = time.time()
            t_elapsed = time.time() - start_time
            
            # Get state
            state = get_tello_state(tello)
            current_time = time.time()
            dt_actual = current_time - last_time
            position += state[6:9] * dt_actual
            position[2] = state[2]
            state[0:3] = position
            last_time = current_time
            
            # Get target
            target_pos, target_vel = interpolate_trajectory(
                waypoints, waypoint_times, t_elapsed,
                approach_duration, loop_duration, approach_end_idx
            )
            
            # Compute PID command
            pid_velocity = pid_controller.compute_control(state, target_pos)
            
            # Transform to body frame
            yaw = state[5]
            vx_body = pid_velocity[0] * np.cos(yaw) + pid_velocity[1] * np.sin(yaw)
            vy_body = -pid_velocity[0] * np.sin(yaw) + pid_velocity[1] * np.cos(yaw)
            
            # Send commands
            lr_cmd = int(np.clip(vy_body * 100, -100, 100))
            fb_cmd = int(np.clip(vx_body * 100, -100, 100))
            ud_cmd = int(np.clip(pid_velocity[2] * 100, -100, 100))
            yaw_cmd = int(np.clip(pid_velocity[3] * 100, -100, 100))
            
            tello.send_rc_control(lr_cmd, fb_cmd, ud_cmd, yaw_cmd)
            
            # Track data
            # Use Z-axis error only (from barometer - accurate)
            # Don't use X/Y position (dead reckoning is unreliable)
            z_error = abs(state[2] - target_pos[2])
            vel_magnitude = np.linalg.norm(state[6:9])
            
            states_history.append(state.copy())
            targets_history.append(target_pos.copy())
            actions_history.append(np.array([lr_cmd, fb_cmd, ud_cmd, yaw_cmd]))
            errors_history.append(z_error)  # Only Z-axis error
            battery_history.append(tello.get_battery())
            timestamps.append(t_elapsed)
            
            # Status
            if int(t_elapsed) % 5 == 0 and t_elapsed - int(t_elapsed) < dt:
                print(f"t={t_elapsed:.1f}s | Error: {errors_history[-1]:.3f}m | "
                      f"Battery: {battery_history[-1]}% | Height: {state[2]:.2f}m")
            
            # Rate limiting
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted!")
    
    finally:
        print("\n\nLanding...")
        tello.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5)
        tello.land()
        time.sleep(2)
    
    # Save data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"data/flight_logs/pid_{perturbation_type}_{trajectory_type}_{timestamp}.pkl"
    
    data = {
        'controller': 'pid',
        'trajectory_type': trajectory_type,
        'perturbation_type': perturbation_type,
        'states': np.array(states_history),
        'targets': np.array(targets_history),
        'actions': np.array(actions_history),
        'errors': np.array(errors_history),
        'battery_history': battery_history,
        'timestamps': timestamps,
        'metadata': {
            'duration': duration,
            'kp': gains['kp'],
            'max_vel': gains['max_vel'],
            'mean_error': np.mean(errors_history),
            'max_error': np.max(errors_history),
            'battery_used': battery_history[0] - battery_history[-1]
        }
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\n✓ Data saved: {output_file}")
    print(f"\nMean error: {data['metadata']['mean_error']:.3f}m")
    print(f"Max error: {data['metadata']['max_error']:.3f}m")
    print(f"Battery used: {data['metadata']['battery_used']}%\n")
    
    return output_file


def test_hybrid_controller(tello, model_path, trajectory_file, trajectory_type, duration, perturbation_type):
    """Test Hybrid RL controller"""
    print(f"\n{'='*60}")
    print(f"HYBRID RL CONTROLLER - {trajectory_type.upper()}")
    print(f"Perturbation: {perturbation_type}")
    print(f"{'='*60}\n")
    
    # Load model
    model = PPO.load(model_path)
    print(f"✓ Loaded model: {model_path}")
    
    # Load trajectory
    traj = load_trajectory(trajectory_file)
    waypoints = traj['waypoints']
    waypoint_times = traj['waypoint_times']
    
    # Tuned PID gains (for baseline)
    TUNED_GAINS = {
        'circle': {'kp': 0.8, 'max_vel': 0.9},
        'square': {'kp': 0.7, 'max_vel': 0.8},
        'figure8': {'kp': 0.4, 'max_vel': 0.5},
        'spiral': {'kp': 0.8, 'max_vel': 0.9},
        'hover': {'kp': 0.6, 'max_vel': 0.7}
    }
    
    gains = TUNED_GAINS.get(trajectory_type, {'kp': 1.0, 'max_vel': 1.0})
    pid_controller = VelocityPIDController(kp=gains['kp'], max_vel=gains['max_vel'])
    
    # Parse trajectory
    approach_end_idx = 0
    for i in range(len(waypoints)):
        if abs(waypoints[i, 2] - waypoints[-1, 2]) < 0.05:
            approach_end_idx = i
            break
    
    approach_duration = waypoint_times[approach_end_idx] if approach_end_idx > 0 else 0.0
    loop_duration = waypoint_times[-1] - approach_duration
    
    # Takeoff
    print("Takeoff...")
    tello.takeoff()
    time.sleep(3)
    
    # Initialize tracking
    position = np.array([0.0, 0.0, tello.get_height() / 100.0])
    last_time = time.time()
    start_time = time.time()
    control_rate = 20
    dt = 1.0 / control_rate
    
    # Data collection
    states_history = []
    targets_history = []
    actions_history = []
    errors_history = []
    battery_history = []
    timestamps = []
    
    print("Starting hybrid control...\n")
    
    try:
        while time.time() - start_time < duration:
            loop_start = time.time()
            t_elapsed = time.time() - start_time
            
            # Get state
            state = get_tello_state(tello)
            current_time = time.time()
            dt_actual = current_time - last_time
            position += state[6:9] * dt_actual
            position[2] = state[2]
            state[0:3] = position
            last_time = current_time
            
            # Get target
            target_pos, target_vel = interpolate_trajectory(
                waypoints, waypoint_times, t_elapsed,
                approach_duration, loop_duration, approach_end_idx
            )
            
            # PID baseline
            pid_velocity = pid_controller.compute_control(state, target_pos)
            
            # RL residual
            pos_error = target_pos - state[0:3]
            vel_error = target_vel - state[6:9]
            obs_18d = np.concatenate([state, pos_error, vel_error])
            
            rpm_residual, _ = model.predict(obs_18d.reshape(1, -1), deterministic=True)
            rpm_residual = rpm_residual[0]
            
            # Convert RPM to velocity correction
            rpm_to_vel_scale = 0.0001
            vel_correction = rpm_residual[:3] * rpm_to_vel_scale
            
            # Combined velocity
            combined_vel = pid_velocity[:3] + vel_correction
            yaw_rate = pid_velocity[3]
            
            # Transform to body frame
            yaw = state[5]
            vx_body = combined_vel[0] * np.cos(yaw) + combined_vel[1] * np.sin(yaw)
            vy_body = -combined_vel[0] * np.sin(yaw) + combined_vel[1] * np.cos(yaw)
            
            # Send commands
            lr_cmd = int(np.clip(vy_body * 100, -100, 100))
            fb_cmd = int(np.clip(vx_body * 100, -100, 100))
            ud_cmd = int(np.clip(combined_vel[2] * 100, -100, 100))
            yaw_cmd = int(np.clip(yaw_rate * 100, -100, 100))
            
            tello.send_rc_control(lr_cmd, fb_cmd, ud_cmd, yaw_cmd)
            
            # Track data
            # Use Z-axis error only (from barometer - accurate)
            # Don't use X/Y position (dead reckoning is unreliable)
            z_error = abs(state[2] - target_pos[2])
            
            states_history.append(state.copy())
            targets_history.append(target_pos.copy())
            actions_history.append(np.array([lr_cmd, fb_cmd, ud_cmd, yaw_cmd]))
            errors_history.append(z_error)  # Only Z-axis error (reliable)
            battery_history.append(tello.get_battery())
            timestamps.append(t_elapsed)
            
            # Status
            if int(t_elapsed) % 5 == 0 and t_elapsed - int(t_elapsed) < dt:
                print(f"t={t_elapsed:.1f}s | Error: {errors_history[-1]:.3f}m | "
                      f"Battery: {battery_history[-1]}% | Height: {state[2]:.2f}m")
            
            # Rate limiting
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted!")
    
    finally:
        print("\n\nLanding...")
        tello.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5)
        tello.land()
        time.sleep(2)
    
    # Save data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"data/flight_logs/hybrid_{perturbation_type}_{trajectory_type}_{timestamp}.pkl"
    
    data = {
        'controller': 'hybrid',
        'trajectory_type': trajectory_type,
        'perturbation_type': perturbation_type,
        'states': np.array(states_history),
        'targets': np.array(targets_history),
        'actions': np.array(actions_history),
        'errors': np.array(errors_history),
        'battery_history': battery_history,
        'timestamps': timestamps,
        'metadata': {
            'duration': duration,
            'model_path': str(model_path),
            'mean_error': np.mean(errors_history),
            'max_error': np.max(errors_history),
            'battery_used': battery_history[0] - battery_history[-1]
        }
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\n✓ Data saved: {output_file}")
    print(f"\nMean error: {data['metadata']['mean_error']:.3f}m")
    print(f"Max error: {data['metadata']['max_error']:.3f}m")
    print(f"Battery used: {data['metadata']['battery_used']}%\n")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Test Controllers with Perturbations')
    
    parser.add_argument('--controller', type=str, required=True,
                        choices=['pid', 'hybrid', 'both'],
                        help='Controller to test')
    parser.add_argument('--trajectory', type=str, required=True,
                        choices=['circle', 'hover', 'spiral', 'figure8'],
                        help='Trajectory type')
    parser.add_argument('--perturbation', type=str, default='wind',
                        choices=['none', 'wind', 'weight', 'wind_weight'],
                        help='Type of perturbation')
    parser.add_argument('--duration', type=int, default=20,
                        help='Flight duration in seconds')
    
    args = parser.parse_args()
    
    # Build paths
    trajectory_file = f"data/expert_trajectories/perfect_{args.trajectory}_trajectory.pkl"
    
    # Connect to Tello
    tello = Tello()
    tello.connect()
    
    print(f"\n{'='*60}")
    print(f"TELLO STATUS")
    print(f"{'='*60}")
    print(f"Battery: {tello.get_battery()}%")
    print(f"Temperature: {tello.get_temperature()}°C")
    print(f"{'='*60}\n")
    
    if tello.get_battery() < 20:
        print("⚠️  Battery too low!")
        return
    
    # Test PID
    if args.controller in ['pid', 'both']:
        input(f"\nReady to test PID with {args.perturbation}? Press Enter...")
        test_pid_controller(tello, trajectory_file, args.trajectory, args.duration, args.perturbation)
        
        if args.controller == 'both':
            print("\n" + "="*60)
            print("Waiting 30 seconds before next test...")
            print("="*60 + "\n")
            time.sleep(30)
    
    # Test Hybrid
    if args.controller in ['hybrid', 'both']:
        # Find model
        model_dir = Path(f"logs/hybrid/{args.trajectory}")
        model_dirs = sorted(model_dir.glob("rl_only_*"))
        if not model_dirs:
            print(f"❌ No trained models found for {args.trajectory}")
            return
        
        latest_model_dir = model_dirs[-1]
        model_path = latest_model_dir / "final_model.zip"
        
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            return
        
        input(f"\nReady to test Hybrid with {args.perturbation}? Press Enter...")
        test_hybrid_controller(tello, str(model_path), trajectory_file, args.trajectory, args.duration, args.perturbation)
    
    print("\n✅ Testing complete!\n")


if __name__ == '__main__':
    main()
