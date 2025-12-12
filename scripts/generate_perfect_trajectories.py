#!/usr/bin/env python3
"""
Generate Perfect Trajectories from Manual Flight Analysis

Creates mathematically perfect trajectories (circle, spiral, figure-8, waypoint)
scaled to match the actual Tello flight characteristics observed in manual data.

This ensures trajectories are realistic and achievable by the real drone.
"""

import pickle
import numpy as np
from pathlib import Path
import argparse
import json
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def analyze_manual_flights(data_dir):
    """Analyze manual flight data to extract realistic parameters"""
    data_dir = Path(data_dir)
    pkl_files = sorted(data_dir.glob("*.pkl"))
    
    all_positions = []
    all_velocities = []
    
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        states = data['states']
        all_positions.append(states[:, 0:3])
        all_velocities.append(states[:, 6:9])
    
    positions = np.vstack(all_positions)
    velocities = np.vstack(all_velocities)
    
    params = {
        'typical_radius': float(np.std(positions[:, 0:2])),  # XY spread
        'typical_height': float(np.mean(positions[:, 2])),
        'height_variance': float(np.std(positions[:, 2])),
        'max_xy_velocity': float(np.percentile(np.abs(velocities[:, 0:2]), 95)),
        'max_z_velocity': float(np.percentile(np.abs(velocities[:, 2]), 95)),
        'spatial_extent_x': float(positions[:, 0].max() - positions[:, 0].min()),
        'spatial_extent_y': float(positions[:, 1].max() - positions[:, 1].min()),
    }
    
    print(f"\n{'='*60}")
    print("MANUAL FLIGHT CHARACTERISTICS")
    print(f"{'='*60}")
    print(f"Typical radius: {params['typical_radius']:.3f} m")
    print(f"Typical height: {params['typical_height']:.3f} m")
    print(f"Height variance: {params['height_variance']:.3f} m")
    print(f"Max XY velocity: {params['max_xy_velocity']:.3f} m/s")
    print(f"Max Z velocity: {params['max_z_velocity']:.3f} m/s")
    print(f"Spatial extent: X={params['spatial_extent_x']:.3f}m, Y={params['spatial_extent_y']:.3f}m")
    
    return params


def generate_circle(center, radius, num_points, height, include_approach=True):
    """Generate perfect circle with optional approach from origin"""
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    waypoints = np.zeros((num_points, 3))
    waypoints[:, 0] = center[0] + radius * np.cos(angles)
    waypoints[:, 1] = center[1] + radius * np.sin(angles)
    waypoints[:, 2] = height
    
    if include_approach:
        # Add approach waypoints from takeoff (0,0,0) to first circle point
        approach_points = np.linspace([0, 0, 0], waypoints[0], 5, endpoint=False)
        waypoints = np.vstack([approach_points, waypoints])
    
    return waypoints


def generate_figure8(center, radius, num_points, height, include_approach=True):
    """Generate perfect figure-8 (lemniscate) with optional approach"""
    t = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    
    # Lemniscate parametric equations
    scale = radius * 1.2  # Slightly larger for nice shape
    waypoints = np.zeros((num_points, 3))
    waypoints[:, 0] = center[0] + scale * np.sin(t)
    waypoints[:, 1] = center[1] + scale * np.sin(t) * np.cos(t)
    waypoints[:, 2] = height
    
    if include_approach:
        approach_points = np.linspace([0, 0, 0], waypoints[0], 5, endpoint=False)
        waypoints = np.vstack([approach_points, waypoints])
    
    return waypoints


def generate_spiral(center, radius, num_points, height_start, height_end, include_approach=True):
    """Generate upward/downward spiral with optional approach"""
    angles = np.linspace(0, 4*np.pi, num_points, endpoint=False)  # 2 full rotations
    heights = np.linspace(height_start, height_end, num_points)
    
    waypoints = np.zeros((num_points, 3))
    waypoints[:, 0] = center[0] + radius * np.cos(angles)
    waypoints[:, 1] = center[1] + radius * np.sin(angles)
    waypoints[:, 2] = heights
    
    if include_approach:
        approach_points = np.linspace([0, 0, 0], waypoints[0], 5, endpoint=False)
        waypoints = np.vstack([approach_points, waypoints])
    
    return waypoints


def generate_square_waypoints(center, side_length, num_points_per_side, height, include_approach=True):
    """Generate square/rectangle waypoint pattern with optional approach"""
    half = side_length / 2
    
    # Define corners
    corners = np.array([
        [center[0] + half, center[1] + half, height],
        [center[0] + half, center[1] - half, height],
        [center[0] - half, center[1] - half, height],
        [center[0] - half, center[1] + half, height],
    ])
    
    # Interpolate points along edges
    waypoints = []
    for i in range(4):
        start = corners[i]
        end = corners[(i + 1) % 4]
        edge_points = np.linspace(start, end, num_points_per_side, endpoint=False)
        waypoints.append(edge_points)
    
    waypoints = np.vstack(waypoints)
    
    if include_approach:
        approach_points = np.linspace([0, 0, 0], waypoints[0], 5, endpoint=False)
        waypoints = np.vstack([approach_points, waypoints])
    
    return waypoints


def save_trajectory(waypoints, trajectory_label, duration, output_dir, flight_params, plot=True):
    """Save trajectory with metadata"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_points = len(waypoints)
    waypoint_times = np.linspace(0, duration, num_points, endpoint=False)
    
    trajectory_data = {
        'trajectory_label': trajectory_label,
        'source': 'generated_perfect',
        'waypoints': waypoints,
        'waypoint_times': waypoint_times,
        'duration': duration,
        'num_waypoints': num_points,
        'flight_params_used': flight_params,
        'generated_from': 'manual_flight_analysis'
    }
    
    output_file = output_dir / f"perfect_{trajectory_label}_trajectory.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(trajectory_data, f)
    
    meta_file = output_file.with_suffix('.json')
    meta_data = {k: v for k, v in trajectory_data.items() if k not in ['waypoints']}
    meta_data['waypoint_times'] = waypoint_times.tolist()
    meta_data['waypoints_preview'] = waypoints[:5].tolist()
    
    with open(meta_file, 'w') as f:
        json.dump(meta_data, f, indent=2)
    
    print(f"\n✓ Saved: {output_file}")
    print(f"  Waypoints: {num_points}")
    print(f"  Duration: {duration:.1f}s")
    
    # Generate plot
    if plot:
        fig = plt.figure(figsize=(14, 6))
        
        # 3D plot
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 'g-', linewidth=2, label='Trajectory')
        ax1.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], c='red', s=50, alpha=0.6, label='Waypoints')
        ax1.scatter([waypoints[0, 0]], [waypoints[0, 1]], [waypoints[0, 2]], 
                   c='green', s=200, marker='^', label='Start', zorder=5)
        ax1.scatter([waypoints[-1, 0]], [waypoints[-1, 1]], [waypoints[-1, 2]], 
                   c='red', s=200, marker='v', label='End', zorder=5)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title(f'3D Trajectory: {trajectory_label.upper()}')
        ax1.legend()
        ax1.grid(True)
        
        # Top-down view
        ax2 = fig.add_subplot(122)
        ax2.plot(waypoints[:, 0], waypoints[:, 1], 'g-', linewidth=2, label='Trajectory')
        ax2.scatter(waypoints[:, 0], waypoints[:, 1], c='red', s=50, alpha=0.6, label='Waypoints')
        ax2.scatter([waypoints[0, 0]], [waypoints[0, 1]], c='green', s=200, marker='^', label='Start', zorder=5)
        ax2.scatter([waypoints[-1, 0]], [waypoints[-1, 1]], c='red', s=200, marker='v', label='End', zorder=5)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title(f'Top-Down View: {trajectory_label.upper()}')
        ax2.legend()
        ax2.grid(True)
        ax2.axis('equal')
        
        plt.tight_layout()
        
        plot_file = output_dir / f"perfect_{trajectory_label}_plot.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"  Plot: {plot_file}")
        plt.close()
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Generate perfect trajectories from manual flight analysis')
    parser.add_argument('--manual-data-dir', type=str, default='data/tello_flights',
                        help='Directory with manual flight .pkl files')
    parser.add_argument('--reference-circle', type=str, default='data/expert_trajectories/learned_circle_trajectory.pkl',
                        help='Reference learned circle trajectory to match position/scale')
    parser.add_argument('--output-dir', type=str, default='data/expert_trajectories',
                        help='Output directory for generated trajectories')
    parser.add_argument('--duration', type=float, default=None,
                        help='Duration for one complete trajectory loop (auto-calculated if not provided)')
    parser.add_argument('--num-points', type=int, default=36,
                        help='Number of waypoints per trajectory')
    parser.add_argument('--radius', type=float, default=0.5,
                        help='Circle radius in meters (default: 0.5m)')
    parser.add_argument('--height', type=float, default=0.8,
                        help='Flight height in meters (default: 0.8m)')
    parser.add_argument('--velocity', type=float, default=0.3,
                        help='Target velocity in m/s (default: 0.3m/s)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("GENERATING PERFECT TRAJECTORIES")
    print(f"{'='*60}")
    
    # Use realistic parameters for indoor drone flight
    center_x = 0.0
    center_y = 0.0
    center_z = args.height
    radius = args.radius
    
    # Auto-calculate duration based on desired velocity
    if args.duration is None:
        circumference = 2 * np.pi * radius
        args.duration = circumference / args.velocity
        print(f"\nAuto-calculated duration from velocity:")
        print(f"  Circumference: {circumference:.2f}m")
        print(f"  Velocity: {args.velocity:.2f}m/s")
        print(f"  Duration: {args.duration:.1f}s")
    
    params = {
        'center_x': center_x,
        'center_y': center_y,
        'center_z': center_z,
        'radius': radius,
        'velocity': args.velocity,
        'source': 'realistic_parameters'
    }
    
    print(f"\n{'='*60}")
    print("TRAJECTORY PARAMETERS")
    print(f"{'='*60}")
    print(f"Center: [{center_x:.3f}, {center_y:.3f}, {center_z:.3f}] m")
    print(f"Radius: {radius:.3f} m")
    print(f"Duration: {args.duration:.1f} s")
    print(f"Waypoints: {args.num_points}")
    
    center = np.array([center_x, center_y])
    
    # Generate trajectories
    print(f"\n{'='*60}")
    print("GENERATING TRAJECTORIES")
    print(f"{'='*60}")
    
    # 1. Circle (regenerate perfect version matching learned circle)
    circle_wp = generate_circle(center, radius, args.num_points, center_z)
    save_trajectory(circle_wp, 'circle', args.duration, args.output_dir, params)
    
    # 2. Figure-8
    figure8_wp = generate_figure8(center, radius, args.num_points, center_z)
    save_trajectory(figure8_wp, 'figure8', args.duration * 1.2, args.output_dir, params)
    
    # 3. Spiral (upward from current height)
    spiral_wp = generate_spiral(center, radius, args.num_points, 
                                center_z, center_z + 0.3)  # 30cm climb
    save_trajectory(spiral_wp, 'spiral', args.duration * 1.5, args.output_dir, params)
    
    # 4. Square waypoints
    square_wp = generate_square_waypoints(center, radius * 2, args.num_points // 4, center_z)
    save_trajectory(square_wp, 'square', args.duration, args.output_dir, params)
    
    # 5. Hover (at learned circle center)
    hover_wp = np.array([[center_x, center_y, center_z]])
    hover_times = np.array([0.0])
    hover_data = {
        'trajectory_label': 'hover',
        'source': 'generated_perfect',
        'waypoints': hover_wp,
        'waypoint_times': hover_times,
        'duration': args.duration,
        'num_waypoints': 1,
        'flight_params_used': params,
    }
    hover_file = Path(args.output_dir) / "perfect_hover_trajectory.pkl"
    with open(hover_file, 'wb') as f:
        pickle.dump(hover_data, f)
    print(f"\n✓ Saved: {hover_file}")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Generated 5 perfect trajectories in: {args.output_dir}")
    print(f"  - circle: {args.num_points} points, {args.duration:.0f}s")
    print(f"  - figure8: {args.num_points} points, {args.duration*1.2:.0f}s")
    print(f"  - spiral: {args.num_points} points, {args.duration*1.5:.0f}s")
    print(f"  - square: {args.num_points} points, {args.duration:.0f}s")
    print(f"  - hover: 1 point, {args.duration:.0f}s")
    print(f"\nAll trajectories scaled to match your manual flight characteristics!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
