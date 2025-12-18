#!/usr/bin/env python3
"""
Train PID from Manual Data

Extracts trajectory from manually collected flight data and tunes PID controller
to autonomously follow the same trajectory.

Workflow:
1. Load manual flight data (positions over time)
2. Extract trajectory waypoints with timestamps
3. Create trajectory file for autonomous replay
4. Optionally: Test PID tracking in simulation before real drone
"""

import pickle
import numpy as np
from pathlib import Path
import argparse
import json
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


def fit_circle_3d(points):
    """Fit a circle to 3D points using least squares
    
    Returns:
        center: [x, y, z] center of circle
        radius: radius of circle
        normal: normal vector to plane of circle
    """
    # Use mean Z as the circle plane height
    z_plane = np.mean(points[:, 2])
    points_2d = points[:, 0:2]
    
    # Fit circle in 2D (XY plane)
    def circle_residuals(params, points):
        xc, yc, r = params
        return np.sqrt((points[:, 0] - xc)**2 + (points[:, 1] - yc)**2) - r
    
    # Initial guess: center at mean, radius from std
    x0 = [np.mean(points_2d[:, 0]), np.mean(points_2d[:, 1]), 
          np.std(points_2d)]
    
    result = least_squares(circle_residuals, x0, args=(points_2d,))
    xc, yc, radius = result.x
    
    center = np.array([xc, yc, z_plane])
    
    return center, radius


def generate_perfect_circle(center, radius, num_points=50, z_variance=0.0):
    """Generate a perfect circle trajectory
    
    Args:
        center: [x, y, z] center position
        radius: circle radius
        num_points: number of waypoints
        z_variance: altitude variation (0 for flat circle)
    
    Returns:
        waypoints: (num_points, 3) array of circle waypoints
    """
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    
    waypoints = np.zeros((num_points, 3))
    waypoints[:, 0] = center[0] + radius * np.cos(angles)
    waypoints[:, 1] = center[1] + radius * np.sin(angles)
    waypoints[:, 2] = center[2] + z_variance * np.sin(2*angles)  # Optional spiral
    
    return waypoints


class TrajectoryExtractor:
    def __init__(self, smoothing_window=5):
        self.smoothing_window = smoothing_window
    
    def smooth_trajectory(self, positions, timestamps):
        """Apply smoothing to reduce noise in manual flight data"""
        if len(positions) < self.smoothing_window:
            return positions, timestamps
        
        smoothed = np.copy(positions)
        for i in range(len(positions)):
            start_idx = max(0, i - self.smoothing_window // 2)
            end_idx = min(len(positions), i + self.smoothing_window // 2 + 1)
            smoothed[i] = np.mean(positions[start_idx:end_idx], axis=0)
        
        return smoothed, timestamps
    
    def extract_waypoints(self, positions, timestamps, waypoint_spacing=0.1):
        """Extract keyframe waypoints from continuous trajectory
        
        Args:
            positions: (N, 3) array of XYZ positions
            timestamps: (N,) array of timestamps
            waypoint_spacing: Min distance between waypoints (meters)
        
        Returns:
            waypoints: (M, 3) array of waypoint positions
            waypoint_times: (M,) array of waypoint timestamps
        """
        waypoints = [positions[0]]
        waypoint_times = [timestamps[0]]
        
        for i in range(1, len(positions)):
            dist = np.linalg.norm(positions[i] - waypoints[-1])
            if dist >= waypoint_spacing:
                waypoints.append(positions[i])
                waypoint_times.append(timestamps[i])
        
        waypoints.append(positions[-1])
        waypoint_times.append(timestamps[-1])
        
        return np.array(waypoints), np.array(waypoint_times)
    
    def fit_spline(self, waypoints, waypoint_times):
        """Fit cubic spline through waypoints for smooth trajectory
        
        Returns:
            spline: CubicSpline object (callable: spline(t) -> [x, y, z])
        """
        if len(waypoints) < 4:
            print(f"⚠️  Warning: Only {len(waypoints)} waypoints, spline may be unstable")
        
        spline = CubicSpline(waypoint_times, waypoints, bc_type='natural')
        return spline


def load_manual_data(pkl_file):
    """Load manual flight data"""
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data


def visualize_trajectory(original_pos, smoothed_pos, waypoints, save_path=None):
    """Plot 3D trajectory comparison"""
    fig = plt.figure(figsize=(12, 8))
    
    ax = fig.add_subplot(121, projection='3d')
    ax.plot(original_pos[:, 0], original_pos[:, 1], original_pos[:, 2], 
            'b-', alpha=0.3, label='Original', linewidth=1)
    ax.plot(smoothed_pos[:, 0], smoothed_pos[:, 1], smoothed_pos[:, 2], 
            'g-', alpha=0.6, label='Smoothed', linewidth=2)
    ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 
               c='r', s=100, marker='o', label='Waypoints', zorder=5)
    
    ax.scatter([original_pos[0, 0]], [original_pos[0, 1]], [original_pos[0, 2]], 
               c='green', s=200, marker='^', label='Start', zorder=6)
    ax.scatter([original_pos[-1, 0]], [original_pos[-1, 1]], [original_pos[-1, 2]], 
               c='red', s=200, marker='v', label='End', zorder=6)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory')
    ax.legend()
    ax.grid(True)
    
    ax2 = fig.add_subplot(122)
    ax2.plot(original_pos[:, 0], original_pos[:, 1], 'b-', alpha=0.3, label='Original', linewidth=1)
    ax2.plot(smoothed_pos[:, 0], smoothed_pos[:, 1], 'g-', alpha=0.6, label='Smoothed', linewidth=2)
    ax2.scatter(waypoints[:, 0], waypoints[:, 1], c='r', s=100, marker='o', label='Waypoints', zorder=5)
    ax2.scatter([original_pos[0, 0]], [original_pos[0, 1]], c='green', s=200, marker='^', label='Start', zorder=6)
    ax2.scatter([original_pos[-1, 0]], [original_pos[-1, 1]], c='red', s=200, marker='v', label='End', zorder=6)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top-Down View (XY)')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory plot to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Extract trajectory from manual flight data')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to manual flight .pkl file')
    parser.add_argument('--output', type=str, default='data/expert_trajectories',
                        help='Output directory for learned trajectory')
    parser.add_argument('--smooth-window', type=int, default=5,
                        help='Smoothing window size (samples)')
    parser.add_argument('--waypoint-spacing', type=float, default=0.1,
                        help='Minimum distance between waypoints (meters)')
    parser.add_argument('--visualize', action='store_true',
                        help='Show trajectory visualization')
    parser.add_argument('--extract-one-loop', action='store_true',
                        help='Extract only the first complete loop (for repeated circles)')
    parser.add_argument('--loop-duration', type=float, default=None,
                        help='Duration of one loop in seconds (auto-detect if not specified)')
    parser.add_argument('--extra-smooth', type=int, default=None,
                        help='Additional aggressive smoothing window (applied after initial smoothing)')
    parser.add_argument('--fit-perfect-circle', action='store_true',
                        help='Fit a perfect circle to the data and generate clean waypoints')
    parser.add_argument('--circle-points', type=int, default=36,
                        help='Number of waypoints for perfect circle (default: 36 = every 10 degrees)')
    
    args = parser.parse_args()
    
    pkl_file = Path(args.data)
    if not pkl_file.exists():
        print(f" File not found: {pkl_file}")
        return
    
    print(f"\n{'='*60}")
    print(f"Loading manual flight data: {pkl_file.name}")
    print(f"{'='*60}")
    
    data = load_manual_data(pkl_file)
    
    states = data['states']
    timestamps = data['timestamps']
    trajectory_label = data.get('trajectory_label', 'unknown')
    
    positions = states[:, 0:3]
    
    print(f"Trajectory: {trajectory_label}")
    print(f"Duration: {data['duration']:.1f}s")
    print(f"Samples: {len(positions)}")
    print(f"Sample rate: {len(positions)/data['duration']:.1f} Hz")
    
    print(f"\nPosition range:")
    print(f"  X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}] m")
    print(f"  Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}] m")
    print(f"  Z: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}] m")
    
    print(f"\n{'='*60}")
    print("Extracting trajectory...")
    print(f"{'='*60}")
    
    # Extract one loop if requested
    if args.extract_one_loop:
        if args.loop_duration:
            loop_samples = int(args.loop_duration * (len(positions) / data['duration']))
        else:
            # Auto-detect loop: find when drone returns close to start
            start_pos = positions[0]
            distances = np.linalg.norm(positions - start_pos, axis=1)
            # Find first return to start (after moving away)
            moved_away = np.where(distances > 0.15)[0]  # Moved >15cm from start
            if len(moved_away) > 0:
                after_move = moved_away[-1]  # Last point far from start
                returned = np.where(distances[after_move:] < 0.1)[0]  # Returned within 10cm
                if len(returned) > 0:
                    loop_samples = after_move + returned[0]
                    print(f"Auto-detected loop duration: {timestamps[loop_samples] - timestamps[0]:.1f}s")
                else:
                    loop_samples = len(positions) // 2  # Use first half
                    print(f"⚠️  Could not auto-detect loop, using first half")
            else:
                loop_samples = len(positions)
                print(f"⚠️  No clear loop detected, using all data")
        
        positions = positions[:loop_samples]
        timestamps = timestamps[:loop_samples]
        print(f"✓ Extracted one loop: {loop_samples} samples ({timestamps[-1] - timestamps[0]:.1f}s)")
    
    extractor = TrajectoryExtractor(smoothing_window=args.smooth_window)
    
    smoothed_pos, smoothed_times = extractor.smooth_trajectory(positions, timestamps)
    print(f"✓ Applied smoothing (window={args.smooth_window})")
    
    # Apply extra aggressive smoothing if requested
    if args.extra_smooth:
        for i in range(len(smoothed_pos)):
            start_idx = max(0, i - args.extra_smooth // 2)
            end_idx = min(len(smoothed_pos), i + args.extra_smooth // 2 + 1)
            smoothed_pos[i] = np.mean(smoothed_pos[start_idx:end_idx], axis=0)
        print(f"✓ Applied extra smoothing (window={args.extra_smooth})")
    
    # Fit perfect circle if requested
    if args.fit_perfect_circle:
        print(f"\n{'='*60}")
        print("Fitting Perfect Circle...")
        print(f"{'='*60}")
        
        center, radius = fit_circle_3d(smoothed_pos)
        print(f"Circle center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}] m")
        print(f"Circle radius: {radius:.3f} m")
        
        # Generate perfect circle waypoints
        waypoints = generate_perfect_circle(center, radius, num_points=args.circle_points)
        
        # Create uniform time spacing for one loop
        duration = smoothed_times[-1] - smoothed_times[0]
        waypoint_times = np.linspace(0, duration, args.circle_points, endpoint=False)
        
        print(f"✓ Generated {len(waypoints)} perfect circle waypoints")
        
        # For visualization, keep original/smoothed
        original_pos_for_plot = positions
        smoothed_pos_for_plot = smoothed_pos
    else:
        waypoints, waypoint_times = extractor.extract_waypoints(
            smoothed_pos, smoothed_times, waypoint_spacing=args.waypoint_spacing
        )
        print(f"✓ Extracted {len(waypoints)} waypoints (spacing={args.waypoint_spacing}m)")
        
        original_pos_for_plot = positions
        smoothed_pos_for_plot = smoothed_pos
    
    spline = extractor.fit_spline(waypoints, waypoint_times)
    print(f"✓ Fitted cubic spline")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_name = f"learned_{trajectory_label}_trajectory"
    output_file = output_dir / f"{output_name}.pkl"
    
    learned_trajectory = {
        'trajectory_label': trajectory_label,
        'source_file': str(pkl_file),
        'waypoints': waypoints,
        'waypoint_times': waypoint_times,
        'duration': waypoint_times[-1] - waypoint_times[0],
        'num_waypoints': len(waypoints),
        'smoothing_window': args.smooth_window,
        'waypoint_spacing': args.waypoint_spacing,
        'spline_coefficients': {
            't': waypoint_times.tolist(),
            'c': [spline.c[:, i].tolist() for i in range(3)]
        }
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(learned_trajectory, f)
    
    meta_file = output_file.with_suffix('.json')
    with open(meta_file, 'w') as f:
        meta_data = {k: v for k, v in learned_trajectory.items() 
                     if k != 'spline_coefficients'}
        meta_data['waypoint_times'] = waypoint_times.tolist()
        meta_data['waypoints'] = waypoints.tolist()
        json.dump(meta_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"SAVED LEARNED TRAJECTORY")
    print(f"{'='*60}")
    print(f"File: {output_file}")
    print(f"Metadata: {meta_file}")
    print(f"Waypoints: {len(waypoints)}")
    print(f"Duration: {learned_trajectory['duration']:.1f}s")
    
    if args.visualize:
        print(f"\nGenerating visualization...")
        plot_file = output_dir / f"{output_name}_plot.png"
        visualize_trajectory(original_pos_for_plot, smoothed_pos_for_plot, waypoints, save_path=plot_file)
    
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print(f"{'='*60}")
    print(f"1. Test trajectory in simulation:")
    print(f"   python scripts/test_pid_trajectory_sim.py --trajectory {output_file}")
    print(f"\n2. Test on real Tello drone:")
    print(f"   python src/hardware/run_tello.py --controller pid --trajectory {output_file}")
    print(f"\n3. Train Hybrid model to imitate manual actions:")
    print(f"   python scripts/train_hybrid_from_manual.py --data {pkl_file}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
