#!/usr/bin/env python3
"""
Prepare Autonomous Flight Data for Hybrid RL Training

Takes autonomous flight data (open-loop trajectory following) and prepares
it for training the Hybrid RL controller in simulation.

The workflow:
1. Real drone flies trajectories open-loop (baseline performance)
2. Extract trajectories and performance metrics
3. Train Hybrid RL in simulation to match/exceed baseline
4. Test both on real drone with domain randomization (weight, obstacles, etc.)
"""

import pickle
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt


def analyze_flight_data(flight_file):
    """Analyze a single autonomous flight"""
    with open(flight_file, 'rb') as f:
        data = pickle.load(f)
    
    states = np.array(data['states'])
    actions = np.array(data['actions'])
    timestamps = np.array(data['timestamps'])
    
    # Extract positions and velocities
    positions = states[:, 0:3]
    velocities = states[:, 6:9]
    
    # Calculate drift (position at end vs start)
    drift = np.linalg.norm(positions[-1, 0:2] - positions[0, 0:2])
    
    # Calculate smoothness (jerk = derivative of acceleration)
    if len(actions) > 2:
        jerk = np.diff(actions[:, :3], axis=0)
        smoothness = np.mean(np.linalg.norm(jerk, axis=1))
    else:
        smoothness = 0
    
    metrics = {
        'duration': timestamps[-1] if len(timestamps) > 0 else 0,
        'num_samples': len(states),
        'drift_xy': drift,
        'avg_velocity': np.mean(np.linalg.norm(velocities[:, :2], axis=1)),
        'max_velocity': np.max(np.linalg.norm(velocities, axis=1)),
        'smoothness': smoothness,
        'trajectory_label': data.get('trajectory_label', 'unknown')
    }
    
    return metrics, data


def prepare_for_simulation(trajectory_files, output_dir):
    """Prepare trajectory files for simulation training"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("PREPARING DATA FOR HYBRID RL TRAINING")
    print(f"{'='*60}\n")
    
    for traj_file in trajectory_files:
        traj_file = Path(traj_file)
        print(f"Processing: {traj_file.name}")
        
        with open(traj_file, 'rb') as f:
            traj = pickle.load(f)
        
        # Extract trajectory info
        waypoints = traj['waypoints']
        waypoint_times = traj['waypoint_times']
        duration = traj['duration']
        
        print(f"  Waypoints: {len(waypoints)}")
        print(f"  Duration: {duration:.1f}s")
        
        # Save in format expected by gym environment
        sim_traj = {
            'positions': waypoints,
            'times': waypoint_times,
            'duration': duration,
            'label': traj.get('trajectory_label', 'unknown')
        }
        
        output_file = output_dir / f"sim_{traj_file.name}"
        with open(output_file, 'wb') as f:
            pickle.dump(sim_traj, f)
        print(f"  ✓ Saved: {output_file}\n")


def main():
    parser = argparse.ArgumentParser(description='Prepare data for Hybrid RL training')
    parser.add_argument('--flight-data-dir', type=str, default='data/tello_flights',
                        help='Directory with autonomous flight data')
    parser.add_argument('--trajectory-dir', type=str, default='data/expert_trajectories',
                        help='Directory with trajectory files')
    parser.add_argument('--output-dir', type=str, default='data/simulation_trajectories',
                        help='Output directory for simulation-ready data')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze flight performance')
    
    args = parser.parse_args()
    
    if args.analyze:
        # Analyze all autonomous flights
        flight_dir = Path(args.flight_data_dir)
        flight_files = sorted(flight_dir.glob("autonomous_*.pkl"))
        
        print(f"\n{'='*60}")
        print("AUTONOMOUS FLIGHT ANALYSIS")
        print(f"{'='*60}\n")
        
        all_metrics = []
        for flight_file in flight_files:
            metrics, data = analyze_flight_data(flight_file)
            all_metrics.append(metrics)
            
            print(f"{flight_file.name}:")
            print(f"  Trajectory: {metrics['trajectory_label']}")
            print(f"  Duration: {metrics['duration']:.1f}s")
            print(f"  Samples: {metrics['num_samples']}")
            print(f"  XY Drift: {metrics['drift_xy']:.3f}m")
            print(f"  Avg velocity: {metrics['avg_velocity']:.2f} m/s")
            print(f"  Smoothness: {metrics['smoothness']:.3f}\n")
        
        # Summary
        if all_metrics:
            avg_drift = np.mean([m['drift_xy'] for m in all_metrics])
            avg_vel = np.mean([m['avg_velocity'] for m in all_metrics])
            print(f"\n{'='*60}")
            print(f"SUMMARY ({len(all_metrics)} flights)")
            print(f"{'='*60}")
            print(f"Average XY drift: {avg_drift:.3f}m")
            print(f"Average velocity: {avg_vel:.2f} m/s")
            print(f"{'='*60}\n")
    
    # Prepare trajectory files for simulation
    traj_dir = Path(args.trajectory_dir)
    trajectory_files = sorted(traj_dir.glob("perfect_*.pkl"))
    
    if trajectory_files:
        prepare_for_simulation(trajectory_files, args.output_dir)
    
    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    print("1. Train Hybrid RL in simulation using these trajectories")
    print("2. Use domain randomization (mass, drag, inertia)")
    print("3. Test trained Hybrid vs Open-loop PID on real Tello")
    print("4. Compare performance under perturbations (added weight, etc.)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
