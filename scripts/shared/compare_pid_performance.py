#!/usr/bin/env python3
"""
Compare PID Tracking Performance
Analyzes multiple autonomous flights and ranks PID configurations.

Usage:
    python scripts/compare_pid_performance.py data/tello_flights/autonomous_*.pkl
    python scripts/compare_pid_performance.py --trajectory hover
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict


def load_flight(filepath):
    """Load flight data from pickle file"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def compute_tracking_metrics(data):
    """Compute tracking performance metrics"""
    states = data['states']
    actions = data['actions']
    timestamps = data['timestamps']
    trajectory_type = data.get('trajectory_type', 'unknown')
    controller = data.get('controller', {})
    
    positions = states[:, 0:3]
    velocities = states[:, 6:9]
    
    # Compute target trajectory
    from scripts.autonomous_data_collection import TrajectoryGenerator
    
    targets = []
    for t in timestamps:
        if trajectory_type == 'circle':
            target_pos, _ = TrajectoryGenerator.circle(t)
        elif trajectory_type == 'figure8':
            target_pos, _ = TrajectoryGenerator.figure8(t)
        elif trajectory_type == 'spiral':
            target_pos, _ = TrajectoryGenerator.spiral(t)
        elif trajectory_type == 'waypoint':
            target_pos, _ = TrajectoryGenerator.waypoint(t, [
                (0.5, 0.5, 1.0), (0.5, -0.5, 1.2),
                (-0.5, -0.5, 1.0), (-0.5, 0.5, 0.8)
            ])
        elif trajectory_type == 'hover':
            target_pos, _ = TrajectoryGenerator.hover(t)
        else:
            target_pos = np.zeros(3)
        targets.append(target_pos)
    
    targets = np.array(targets)
    
    # Tracking errors
    errors = np.linalg.norm(positions - targets, axis=1)
    
    # Metrics
    metrics = {
        'trajectory': trajectory_type,
        'kp': controller.get('kp', 0),
        'max_vel': controller.get('max_vel', 0),
        'duration': timestamps[-1] - timestamps[0],
        'samples': len(states),
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'median_error': np.median(errors),
        'max_error': np.max(errors),
        'rmse': np.sqrt(np.mean(errors**2)),
        'settling_time': compute_settling_time(errors),
        'overshoot': compute_overshoot(positions, targets),
        'control_effort': np.mean(np.abs(actions)),
        'velocity_utilization': np.mean(np.linalg.norm(velocities, axis=1)) / controller.get('max_vel', 1.0),
    }
    
    return metrics, errors, positions, targets


def compute_settling_time(errors, threshold=0.1):
    """Time to settle within threshold (seconds)"""
    settled = errors < threshold
    if not np.any(settled):
        return np.inf
    
    first_settled = np.argmax(settled)
    return first_settled * (1.0 / 20.0)  # Assume 20 Hz


def compute_overshoot(positions, targets):
    """Maximum overshoot as percentage"""
    errors = np.abs(positions - targets)
    max_overshoot = np.max(errors, axis=0)
    target_range = np.ptp(targets, axis=0)
    
    # Avoid division by zero
    target_range = np.where(target_range > 0.01, target_range, 1.0)
    
    overshoot_pct = (max_overshoot / target_range) * 100
    return np.mean(overshoot_pct)


def plot_comparison(results, output_dir=None):
    """Plot comparison across different PID configurations"""
    
    # Group by trajectory type
    by_trajectory = defaultdict(list)
    for r in results:
        by_trajectory[r['trajectory']].append(r)
    
    for traj_type, traj_results in by_trajectory.items():
        if len(traj_results) < 2:
            continue
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'PID Performance Comparison - {traj_type.upper()}', fontsize=14, fontweight='bold')
        
        # Extract data
        kp_vals = [r['kp'] for r in traj_results]
        max_vel_vals = [r['max_vel'] for r in traj_results]
        mean_errors = [r['mean_error'] for r in traj_results]
        rmse_vals = [r['rmse'] for r in traj_results]
        settling_times = [r['settling_time'] if r['settling_time'] != np.inf else 30 for r in traj_results]
        overshoots = [r['overshoot'] for r in traj_results]
        
        # Create labels
        labels = [f"kp={r['kp']:.1f}, v={r['max_vel']:.1f}" for r in traj_results]
        x_pos = np.arange(len(labels))
        
        # Plot 1: Mean Error
        ax1 = axes[0, 0]
        bars1 = ax1.bar(x_pos, mean_errors, color='steelblue')
        ax1.set_ylabel('Mean Error (m)', fontweight='bold')
        ax1.set_title('Tracking Error')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # Highlight best
        best_idx = np.argmin(mean_errors)
        bars1[best_idx].set_color('green')
        
        # Plot 2: RMSE
        ax2 = axes[0, 1]
        bars2 = ax2.bar(x_pos, rmse_vals, color='coral')
        ax2.set_ylabel('RMSE (m)', fontweight='bold')
        ax2.set_title('Root Mean Square Error')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        bars2[best_idx].set_color('green')
        
        # Plot 3: Settling Time
        ax3 = axes[1, 0]
        bars3 = ax3.bar(x_pos, settling_times, color='mediumpurple')
        ax3.set_ylabel('Settling Time (s)', fontweight='bold')
        ax3.set_title('Time to Settle (<0.1m error)')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(labels, rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Overshoot
        ax4 = axes[1, 1]
        bars4 = ax4.bar(x_pos, overshoots, color='orange')
        ax4.set_ylabel('Overshoot (%)', fontweight='bold')
        ax4.set_title('Maximum Overshoot')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(labels, rotation=45, ha='right')
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path / f'pid_comparison_{traj_type}.png', dpi=150, bbox_inches='tight')
            print(f"   Saved: {output_path / f'pid_comparison_{traj_type}.png'}")
        else:
            plt.show()


def print_summary_table(results):
    """Print formatted comparison table"""
    print("\n" + "="*120)
    print("PID PERFORMANCE COMPARISON")
    print("="*120)
    
    # Group by trajectory
    by_trajectory = defaultdict(list)
    for r in results:
        by_trajectory[r['trajectory']].append(r)
    
    for traj_type, traj_results in by_trajectory.items():
        print(f"\n TRAJECTORY: {traj_type.upper()}")
        print("-"*120)
        
        # Header
        print(f"{'kp':>5} {'max_vel':>8} {'Samples':>8} {'Mean Err':>10} {'Std Err':>10} "
              f"{'RMSE':>10} {'Max Err':>10} {'Settle':>8} {'Overshoot':>10} {'Effort':>10}")
        print("-"*120)
        
        # Sort by mean error
        sorted_results = sorted(traj_results, key=lambda x: x['mean_error'])
        
        for i, r in enumerate(sorted_results):
            marker = "🏆" if i == 0 else "  "
            print(f"{marker} {r['kp']:4.1f} {r['max_vel']:7.1f} {r['samples']:7d} "
                  f"{r['mean_error']:9.3f} {r['std_error']:9.3f} {r['rmse']:9.3f} "
                  f"{r['max_error']:9.3f} {r['settling_time']:7.1f}s {r['overshoot']:9.1f}% "
                  f"{r['control_effort']:9.3f}")
        
        # Best configuration
        best = sorted_results[0]
        print("\n" + "🏆 BEST CONFIGURATION:")
        print(f"   kp={best['kp']:.1f}, max_vel={best['max_vel']:.1f} m/s")
        print(f"   Mean error: {best['mean_error']:.3f} ± {best['std_error']:.3f} m")
        print(f"   RMSE: {best['rmse']:.3f} m")
        print(f"   Settling time: {best['settling_time']:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Compare PID tracking performance")
    parser.add_argument('files', nargs='*', help='Flight data files to compare')
    parser.add_argument('--trajectory', type=str, help='Filter by trajectory type')
    parser.add_argument('--plot', action='store_true', help='Generate comparison plots')
    parser.add_argument('--output', type=str, help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Find files
    if args.files:
        files = [Path(f) for f in args.files]
    else:
        # Default: search data directory
        files = list(Path('data/tello_flights').glob('autonomous_*.pkl'))
    
    if not files:
        print(" No flight data files found!")
        print("   Run autonomous data collection first:")
        print("   ./collect_autonomous.sh")
        return
    
    print(f"\n Found {len(files)} flight data files")
    
    # Load and analyze all flights
    results = []
    for filepath in files:
        try:
            data = load_flight(filepath)
            metrics, _, _, _ = compute_tracking_metrics(data)
            
            # Filter by trajectory if specified
            if args.trajectory and metrics['trajectory'] != args.trajectory:
                continue
            
            results.append(metrics)
            print(f"   ✓ {filepath.name}")
        except Exception as e:
            print(f"   ✗ {filepath.name}: {e}")
    
    if not results:
        print("\n No valid flight data to analyze!")
        return
    
    print(f"\n✓ Analyzed {len(results)} flights")
    
    # Print summary
    print_summary_table(results)
    
    # Generate plots
    if args.plot:
        print("\n Generating comparison plots...")
        plot_comparison(results, output_dir=args.output)
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()
