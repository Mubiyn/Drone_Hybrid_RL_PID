#!/usr/bin/env python3
"""
Test Hybrid RL Controller on Real Tello Drone

Deploys RPM-trained hybrid models to real Tello by using the PID baseline
and adding learned residual corrections.
"""

import sys
import time
import argparse
import pickle
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from djitellopy import Tello

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.controllers.pid_controller import VelocityPIDController


def load_trajectory(filepath):
    """Load trajectory from pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_tello_state(tello):
    """Get current Tello state (12D)"""
    # Position (relative to takeoff, integrated from velocity)
    # For real deployment, we use optical flow velocity
    vx = tello.get_speed_x() / 100.0  # cm/s to m/s
    vy = tello.get_speed_y() / 100.0
    vz = tello.get_speed_z() / 100.0
    
    # Orientation (Tello doesn't provide roll/pitch, only yaw)
    pitch = tello.get_pitch() * np.pi / 180.0
    roll = tello.get_roll() * np.pi / 180.0
    yaw = tello.get_yaw() * np.pi / 180.0
    
    # Height from barometer
    height = tello.get_height() / 100.0  # cm to m
    
    # Angular velocity (not available from Tello, use zeros)
    wx = wy = wz = 0.0
    
    # We don't have absolute position, use integrated position
    # This is stored globally
    return np.array([
        0, 0, height,  # x, y, z (only z is accurate from barometer)
        roll, pitch, yaw,
        vx, vy, vz,
        wx, wy, wz
    ])


def interpolate_trajectory(waypoints, waypoint_times, t_elapsed, approach_duration, loop_duration, approach_end_idx):
    """Interpolate trajectory position and velocity"""
    if t_elapsed < approach_duration:
        # Approach phase
        t_phase = t_elapsed
        times_phase = waypoint_times[:approach_end_idx+1]
        waypoints_phase = waypoints[:approach_end_idx+1]
    else:
        # Loop phase
        t_in_loop = (t_elapsed - approach_duration) % loop_duration
        t_phase = approach_duration + t_in_loop
        times_phase = waypoint_times[approach_end_idx:]
        waypoints_phase = waypoints[approach_end_idx:]
    
    # Find interpolation indices
    idx = np.searchsorted(times_phase, t_phase)
    if idx == 0:
        return waypoints_phase[0], np.zeros(3)
    if idx >= len(times_phase):
        return waypoints_phase[-1], np.zeros(3)
    
    # Linear interpolation
    t0, t1 = times_phase[idx-1], times_phase[idx]
    p0, p1 = waypoints_phase[idx-1], waypoints_phase[idx]
    alpha = (t_phase - t0) / (t1 - t0) if t1 > t0 else 0
    
    pos = p0 + alpha * (p1 - p0)
    vel = (p1 - p0) / (t1 - t0) if t1 > t0 else np.zeros(3)
    
    return pos, vel


def run_hybrid_controller(tello, model_path, trajectory_file, trajectory_type, duration=20):
    """
    Run hybrid RL controller on real Tello.
    
    Strategy:
    1. Use trajectory-specific tuned VelocityPID as baseline
    2. Model predicts RPM residuals
    3. Convert RPM residuals to velocity corrections
    4. Send combined velocity to Tello
    """
    print(f"\n{'='*60}")
    print(f"HYBRID RL CONTROLLER - {trajectory_type.upper()}")
    print(f"{'='*60}\n")
    
    # Load model
    model = PPO.load(model_path)
    print(f"✓ Loaded model: {model_path}")
    
    # Load trajectory
    traj = load_trajectory(trajectory_file)
    waypoints = traj['waypoints']
    waypoint_times = traj['waypoint_times']
    print(f"✓ Loaded trajectory: {len(waypoints)} waypoints, {waypoint_times[-1]:.1f}s duration\n")
    
    # Tuned PID gains (same as used in training)
    TUNED_GAINS = {
        'circle': {'kp': 0.8, 'max_vel': 0.9},
        'square': {'kp': 0.7, 'max_vel': 0.8},
        'figure8': {'kp': 0.4, 'max_vel': 0.5},
        'spiral': {'kp': 0.8, 'max_vel': 0.9},
        'hover': {'kp': 0.6, 'max_vel': 0.7}
    }
    
    gains = TUNED_GAINS.get(trajectory_type, {'kp': 1.0, 'max_vel': 1.0})
    pid_controller = VelocityPIDController(kp=gains['kp'], max_vel=gains['max_vel'])
    print(f"PID baseline: kp={gains['kp']}, max_vel={gains['max_vel']}")
    
    # Parse trajectory into approach and loop phases
    approach_end_idx = 0
    for i in range(len(waypoints)):
        if abs(waypoints[i, 2] - waypoints[-1, 2]) < 0.05:
            approach_end_idx = i
            break
    
    approach_duration = waypoint_times[approach_end_idx] if approach_end_idx > 0 else 0.0
    loop_duration = waypoint_times[-1] - approach_duration
    print(f"Approach: {approach_duration:.1f}s, Loop: {loop_duration:.1f}s\n")
    
    # Connect and takeoff
    print("Connecting to Tello...")
    input("Press Enter when ready for takeoff...")
    
    print("\nTakeoff...")
    tello.takeoff()
    time.sleep(3)
    
    # Track position via dead reckoning (same as autonomous flights)
    position = np.array([0.0, 0.0, tello.get_height() / 100.0])
    last_time = time.time()
    
    start_time = time.time()
    control_rate = 20  # Hz
    dt = 1.0 / control_rate
    
    errors = []
    battery_history = []
    
    print("Starting hybrid control...\n")
    
    try:
        while time.time() - start_time < duration:
            loop_start = time.time()
            t_elapsed = time.time() - start_time
            
            # Get state
            state = get_tello_state(tello)
            
            # Update position estimate via dead reckoning
            current_time = time.time()
            dt_actual = current_time - last_time
            position += state[6:9] * dt_actual  # Integrate velocity
            position[2] = state[2]  # Use barometer for height
            state[0:3] = position  # Update position in state
            last_time = current_time
            
            # Get target
            target_pos, target_vel = interpolate_trajectory(
                waypoints, waypoint_times, t_elapsed, 
                approach_duration, loop_duration, approach_end_idx
            )
            
            # Compute PID baseline velocity command
            pid_velocity = pid_controller.compute_control(state, target_pos)
            
            # Build 18D observation for RL model
            pos_error = target_pos - state[0:3]
            vel_error = target_vel - state[6:9]
            obs_18d = np.concatenate([state, pos_error, vel_error])
            
            # Get RL residual (trained on RPM, but we need velocity correction)
            # The model outputs 4D RPM residuals, we convert to velocity adjustments
            rpm_residual, _ = model.predict(obs_18d.reshape(1, -1), deterministic=True)
            rpm_residual = rpm_residual[0]
            
            # Convert RPM residual to velocity correction (heuristic scaling)
            # RPM range: ~16000 (hover), residual scale: 100 RPM
            # Velocity range: ~1 m/s, so 100 RPM ≈ 0.01 m/s
            rpm_to_vel_scale = 0.0001  # 100 RPM → 0.01 m/s
            vel_correction = rpm_residual[:3] * rpm_to_vel_scale
            
            # Combined velocity command
            combined_vel = pid_velocity[:3] + vel_correction
            yaw_rate = pid_velocity[3]  # Use PID yaw
            
            # Transform to body frame
            yaw = state[5]
            vx_body = combined_vel[0] * np.cos(yaw) + combined_vel[1] * np.sin(yaw)
            vy_body = -combined_vel[0] * np.sin(yaw) + combined_vel[1] * np.cos(yaw)
            
            # Send commands to Tello
            lr_cmd = int(np.clip(vy_body * 100, -100, 100))
            fb_cmd = int(np.clip(vx_body * 100, -100, 100))
            ud_cmd = int(np.clip(combined_vel[2] * 100, -100, 100))
            yaw_cmd = int(np.clip(yaw_rate * 100, -100, 100))
            
            tello.send_rc_control(lr_cmd, fb_cmd, ud_cmd, yaw_cmd)
            
            # Track metrics
            pos_err_mag = np.linalg.norm(pos_error)
            errors.append(pos_err_mag)
            battery_history.append(tello.get_battery())
            
            # Status update
            if int(t_elapsed) % 5 == 0 and t_elapsed - int(t_elapsed) < dt:
                print(f"t={t_elapsed:.1f}s | Pos error: {pos_err_mag:.3f}m | "
                      f"Battery: {tello.get_battery()}% | "
                      f"Height: {state[2]:.2f}m")
            
            # Rate limiting
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
    
    finally:
        print("\n\nLanding...")
        tello.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5)
        tello.land()
        time.sleep(2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("FLIGHT SUMMARY")
    print(f"{'='*60}")
    print(f"Duration: {time.time() - start_time:.1f}s")
    print(f"Mean position error: {np.mean(errors):.3f}m")
    print(f"Max position error: {np.max(errors):.3f}m")
    print(f"Battery used: {battery_history[0] - battery_history[-1]}%")
    print(f"{'='*60}\n")
    
    return errors, battery_history


def main():
    parser = argparse.ArgumentParser(description='Test Hybrid RL on Real Tello')
    
    parser.add_argument('--trajectory', type=str, required=True,
                        choices=['circle', 'hover', 'spiral', 'figure8'],
                        help='Trajectory type')
    parser.add_argument('--duration', type=int, default=20,
                        help='Flight duration in seconds')
    
    args = parser.parse_args()
    
    # Build paths
    trajectory_file = f"data/expert_trajectories/perfect_{args.trajectory}_trajectory.pkl"
    
    # Find latest model for this trajectory
    model_dir = Path(f"logs/hybrid/{args.trajectory}")
    model_dirs = sorted(model_dir.glob("rl_only_*"))
    if not model_dirs:
        print(f" No trained models found for {args.trajectory}")
        return
    
    latest_model_dir = model_dirs[-1]
    model_path = latest_model_dir / "final_model.zip"
    
    if not model_path.exists():
        print(f" Model not found: {model_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"HYBRID RL TEST - {args.trajectory.upper()}")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Trajectory: {trajectory_file}")
    print(f"Duration: {args.duration}s")
    print(f"{'='*60}\n")
    
    # Connect to Tello
    tello = Tello()
    tello.connect()
    
    print(f"✓ Connected to Tello")
    print(f"  Battery: {tello.get_battery()}%")
    print(f"  Temperature: {tello.get_temperature()}°C\n")
    
    if tello.get_battery() < 20:
        print("⚠️  Battery too low! Please charge before flying.")
        return
    
    # Run test
    run_hybrid_controller(tello, str(model_path), trajectory_file, args.trajectory, args.duration)


if __name__ == '__main__':
    main()
