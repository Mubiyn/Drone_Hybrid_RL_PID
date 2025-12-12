#!/usr/bin/env python3
"""
Analyze collected Tello flight data
Generates statistics and visualizations of recorded flights.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


def load_flight_data(data_dir="data/tello_flights"):
    """Load all flight data files from directory"""
    data_dir = Path(data_dir)
    flight_files = sorted(data_dir.glob("flight_*.pkl"))
    
    print(f"Found {len(flight_files)} flight recordings")
    
    flights = []
    for f in flight_files:
        with open(f, 'rb') as file:
            data = pickle.load(file)
            flights.append({
                'filename': f.name,
                'data': data
            })
            print(f"  {f.name}: {data['samples']} samples, {data['duration']:.1f}s")
            
    return flights


def analyze_flight(flight_data):
    """Analyze single flight dataset"""
    states = flight_data['states']
    actions = flight_data['actions']
    timestamps = flight_data['timestamps']
    
    # Extract components
    positions = states[:, 0:3]
    orientations = states[:, 3:6]
    velocities = states[:, 6:9]
    
    # Statistics
    stats = {
        'duration': flight_data['duration'],
        'samples': flight_data['samples'],
        'sample_rate': flight_data['samples'] / flight_data['duration'],
        
        # Position stats
        'pos_range': {
            'x': (positions[:, 0].min(), positions[:, 0].max()),
            'y': (positions[:, 1].min(), positions[:, 1].max()),
            'z': (positions[:, 2].min(), positions[:, 2].max())
        },
        
        # Velocity stats
        'vel_mean': velocities.mean(axis=0),
        'vel_std': velocities.std(axis=0),
        'vel_max': np.abs(velocities).max(axis=0),
        
        # Action stats
        'action_mean': actions.mean(axis=0),
        'action_std': actions.std(axis=0),
        'action_max': np.abs(actions).max(axis=0)
    }
    
    return stats


def plot_flight(flight_data, save_path=None):
    """Visualize flight trajectory and states"""
    states = flight_data['states']
    actions = flight_data['actions']
    timestamps = flight_data['timestamps']
    
    positions = states[:, 0:3]
    velocities = states[:, 6:9]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 3D trajectory
    ax = fig.add_subplot(2, 3, 1, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', alpha=0.6)
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
               c='g', s=100, marker='o', label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
               c='r', s=100, marker='x', label='End')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Flight Trajectory')
    ax.legend()
    
    # XY trajectory
    axes[0, 1].plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.6)
    axes[0, 1].scatter(positions[0, 0], positions[0, 1], c='g', s=100, label='Start')
    axes[0, 1].scatter(positions[-1, 0], positions[-1, 1], c='r', s=100, marker='x', label='End')
    axes[0, 1].set_xlabel('X (m)')
    axes[0, 1].set_ylabel('Y (m)')
    axes[0, 1].set_title('XY Trajectory')
    axes[0, 1].grid(True)
    axes[0, 1].axis('equal')
    axes[0, 1].legend()
    
    # Altitude over time
    axes[0, 2].plot(timestamps, positions[:, 2], 'b-')
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Altitude (m)')
    axes[0, 2].set_title('Altitude Profile')
    axes[0, 2].grid(True)
    
    # Velocities over time
    axes[1, 0].plot(timestamps, velocities[:, 0], label='vx')
    axes[1, 0].plot(timestamps, velocities[:, 1], label='vy')
    axes[1, 0].plot(timestamps, velocities[:, 2], label='vz')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Velocity (m/s)')
    axes[1, 0].set_title('Velocities')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Actions over time
    axes[1, 1].plot(timestamps, actions[:, 0], label='Forward/Back')
    axes[1, 1].plot(timestamps, actions[:, 1], label='Left/Right')
    axes[1, 1].plot(timestamps, actions[:, 2], label='Up/Down')
    axes[1, 1].plot(timestamps, actions[:, 3], label='Yaw')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Action (normalized)')
    axes[1, 1].set_title('Control Actions')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Action distribution
    axes[1, 2].hist(np.linalg.norm(actions[:, 0:3], axis=1), bins=30, alpha=0.7)
    axes[1, 2].set_xlabel('Action Magnitude')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Action Distribution')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")
    else:
        plt.show()
        
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze collected Tello flight data")
    parser.add_argument('--data-dir', type=str, default='data/tello_flights',
                        help='Directory with flight data')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots for each flight')
    parser.add_argument('--summary', action='store_true',
                        help='Print summary statistics')
    
    args = parser.parse_args()
    
    flights = load_flight_data(args.data_dir)
    
    if len(flights) == 0:
        print("No flight data found!")
        return
        
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    total_samples = 0
    total_duration = 0
    
    for i, flight in enumerate(flights):
        data = flight['data']
        stats = analyze_flight(data)
        
        total_samples += stats['samples']
        total_duration += stats['duration']
        
        if args.summary:
            print(f"\nFlight {i+1}: {flight['filename']}")
            print(f"  Duration: {stats['duration']:.1f}s")
            print(f"  Samples: {stats['samples']} ({stats['sample_rate']:.1f} Hz)")
            print(f"  Position range:")
            print(f"    X: [{stats['pos_range']['x'][0]:.2f}, {stats['pos_range']['x'][1]:.2f}] m")
            print(f"    Y: [{stats['pos_range']['y'][0]:.2f}, {stats['pos_range']['y'][1]:.2f}] m")
            print(f"    Z: [{stats['pos_range']['z'][0]:.2f}, {stats['pos_range']['z'][1]:.2f}] m")
            print(f"  Max velocity: {stats['vel_max'][0]:.2f}, {stats['vel_max'][1]:.2f}, {stats['vel_max'][2]:.2f} m/s")
            
        if args.plot:
            plot_path = Path(args.data_dir) / f"analysis_{flight['filename'].replace('.pkl', '.png')}"
            plot_flight(data, save_path=plot_path)
            
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total flights: {len(flights)}")
    print(f"Total samples: {total_samples}")
    print(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"Average sample rate: {total_samples/total_duration:.1f} Hz")
    print("\nDataset ready for training!")


if __name__ == '__main__':
    main()
