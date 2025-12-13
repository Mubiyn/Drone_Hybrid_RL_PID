#!/usr/bin/env python3
"""
Analyze Autonomous Flight Data

Loads autonomous flight data and analyzes flight performance:
- Position distribution
- Velocity characteristics  
- Flight stability
- Visualization of actual trajectories
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_flight_data(pkl_file):
    """Load a single flight data file"""
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data


def load_trajectory_file(traj_type):
    """Load the reference trajectory file"""
    traj_file = Path(f'data/expert_trajectories/perfect_{traj_type}_trajectory.pkl')
    if traj_file.exists():
        with open(traj_file, 'rb') as f:
            return pickle.load(f)
    return None


def analyze_flight(data, trajectory_name, reference_traj=None):
    """Analyze a single flight"""
    states = data['states']
    actions = data['actions']
    
    # Extract position and velocity
    actual_pos = states[:, 0:3]
    actual_vel = states[:, 6:9]
    
    # Calculate basic statistics
    pos_std = np.std(actual_pos, axis=0)
    vel_mean = np.mean(np.linalg.norm(actual_vel, axis=1))
    vel_max = np.max(np.linalg.norm(actual_vel, axis=1))
    
    results = {
        'trajectory': trajectory_name,
        'duration': data['metadata']['duration'],
        'samples': len(states),
        'actual_pos': actual_pos,
        'actual_vel': actual_vel,
        'actions': actions,
        'pos_std': pos_std,
        'vel_mean': vel_mean,
        'vel_max': vel_max,
        'battery_drop': data['metadata']['battery_start'] - data['metadata']['battery_end'],
    }
    
    # If reference trajectory available, compute tracking error
    if reference_traj is not None:
        target_waypoints = reference_traj['waypoints']
        # Find closest waypoint for each state
        errors = []
        for pos in actual_pos:
            distances = np.linalg.norm(target_waypoints - pos, axis=1)
            errors.append(np.min(distances))
        
        results['tracking_error'] = np.array(errors)
        results['error_mean'] = np.mean(errors)
        results['error_std'] = np.std(errors)
        results['error_max'] = np.max(errors)
        results['target_waypoints'] = target_waypoints
    
    return results


def plot_trajectory_3d(results_list, trajectory_type, output_dir):
    """Plot 3D trajectory comparison"""
    fig = plt.figure(figsize=(15, 5))
    
    # Load reference trajectory if available
    ref_traj = load_trajectory_file(trajectory_type)
    
    for idx, results in enumerate(results_list):
        ax = fig.add_subplot(1, len(results_list), idx + 1, projection='3d')
        
        actual = results['actual_pos']
        
        # Plot reference if available
        if 'target_waypoints' in results:
            target = results['target_waypoints']
            ax.plot(target[:, 0], target[:, 1], target[:, 2], 
                    'g--', linewidth=2, label='Target', alpha=0.7)
        
        # Plot actual trajectory
        ax.plot(actual[:, 0], actual[:, 1], actual[:, 2], 
                'b-', linewidth=1.5, label='Actual', alpha=0.8)
        
        # Mark start and end
        ax.scatter(actual[0, 0], actual[0, 1], actual[0, 2], 
                  c='green', s=100, marker='o', label='Start')
        ax.scatter(actual[-1, 0], actual[-1, 1], actual[-1, 2], 
                  c='red', s=100, marker='x', label='End')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        
        if 'error_mean' in results:
            ax.set_title(f'Flight {idx+1}\nError: {results["error_mean"]:.3f}m')
        else:
            ax.set_title(f'Flight {idx+1}\nStd: [{results["pos_std"][0]:.3f}, {results["pos_std"][1]:.3f}, {results["pos_std"][2]:.3f}]m')
        
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Equal aspect ratio
        max_range = np.array([
            actual[:, 0].max() - actual[:, 0].min(),
            actual[:, 1].max() - actual[:, 1].min(),
            actual[:, 2].max() - actual[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (actual[:, 0].max() + actual[:, 0].min()) * 0.5
        mid_y = (actual[:, 1].max() + actual[:, 1].min()) * 0.5
        mid_z = (actual[:, 2].max() + actual[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.suptitle(f'{trajectory_type.upper()} Trajectory - 3D View', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_dir / f'{trajectory_type}_3d_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_tracking_error(results_list, trajectory_type, output_dir):
    """Plot tracking error over time or velocity/position stats"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    has_tracking = 'tracking_error' in results_list[0]
    
    for idx, results in enumerate(results_list):
        time = np.arange(len(results['actual_pos'])) / 20.0  # 20 Hz
        
        if has_tracking:
            # Tracking error over time
            axes[0, 0].plot(time, results['tracking_error'], 
                           label=f'Flight {idx+1}', alpha=0.7)
        
        # Position over time
        axes[0, 1].plot(time, results['actual_pos'][:, 0], 
                       label=f'Flight {idx+1} X', alpha=0.7, linestyle='-')
        axes[1, 0].plot(time, results['actual_pos'][:, 1], 
                       label=f'Flight {idx+1} Y', alpha=0.7, linestyle='-')
        axes[1, 1].plot(time, results['actual_pos'][:, 2], 
                       label=f'Flight {idx+1} Z', alpha=0.7, linestyle='-')
    
    if has_tracking:
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Tracking Error (m)')
        axes[0, 0].set_title('Distance to Nearest Target Waypoint')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'No reference\ntrajectory', 
                       ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Tracking Error (N/A)')
    
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('X Position (m)')
    axes[0, 1].set_title('X-axis Position')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Y Position (m)')
    axes[1, 0].set_title('Y-axis Position')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Z Position (m)')
    axes[1, 1].set_title('Z-axis Position (Height)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{trajectory_type.upper()} - Flight Analysis', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_dir / f'{trajectory_type}_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def main():
    data_dir = Path('data/flight_logs')
    output_dir = Path('results/autonomous_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all autonomous flight files
    pkl_files = sorted(data_dir.glob('autonomous_*.pkl'))
    
    if not pkl_files:
        print("✗ No autonomous flight data found in data/flight_logs/")
        return
    
    print(f"\n{'='*60}")
    print("AUTONOMOUS FLIGHT ANALYSIS")
    print(f"{'='*60}\n")
    print(f"Found {len(pkl_files)} flight recordings\n")
    
    # Group by trajectory type
    by_trajectory = defaultdict(list)
    for pkl_file in pkl_files:
        # Extract trajectory type from filename
        # Format: autonomous_learned_<type>_<timestamp>.pkl
        parts = pkl_file.stem.split('_')
        if len(parts) >= 3:
            traj_type = parts[2]  # circle, square, figure8, spiral, hover
            by_trajectory[traj_type].append(pkl_file)
    
    # Analyze each trajectory type
    all_results = {}
    
    for traj_type, files in sorted(by_trajectory.items()):
        print(f"\n{'-'*60}")
        print(f"Trajectory: {traj_type.upper()}")
        print(f"{'-'*60}")
        print(f"Flights: {len(files)}\n")
        
        results_list = []
        
        # Load reference trajectory once
        ref_traj = load_trajectory_file(traj_type)
        if ref_traj:
            print(f"Reference trajectory loaded: {len(ref_traj['waypoints'])} waypoints\n")
        else:
            print(f"No reference trajectory found\n")
        
        for idx, pkl_file in enumerate(files, 1):
            data = load_flight_data(pkl_file)
            results = analyze_flight(data, traj_type, ref_traj)
            results_list.append(results)
            
            print(f"Flight {idx}: {pkl_file.name}")
            print(f"  Duration: {results['duration']:.1f}s")
            print(f"  Samples: {results['samples']}")
            print(f"  Battery drop: {results['battery_drop']}%")
            print(f"  Velocity (avg/max): {results['vel_mean']:.3f} / {results['vel_max']:.3f} m/s")
            print(f"  Position Std: X={results['pos_std'][0]:.4f}, Y={results['pos_std'][1]:.4f}, Z={results['pos_std'][2]:.4f}m")
            
            if 'error_mean' in results:
                print(f"  Tracking Error: {results['error_mean']:.4f} ± {results['error_std']:.4f}m")
                print(f"  Max Error: {results['error_max']:.4f}m")
            
            print()
        
        # Overall statistics
        if 'error_mean' in results_list[0]:
            mean_errors = [r['error_mean'] for r in results_list]
            print(f"Overall Tracking Performance:")
            print(f"  Mean Error: {np.mean(mean_errors):.4f} ± {np.std(mean_errors):.4f}m")
            print(f"  Best Flight: {np.min(mean_errors):.4f}m")
            print(f"  Worst Flight: {np.max(mean_errors):.4f}m")
        else:
            print(f"Overall Flight Characteristics:")
            vel_means = [r['vel_mean'] for r in results_list]
            print(f"  Avg Velocity: {np.mean(vel_means):.3f} ± {np.std(vel_means):.3f}m/s")
        
        # Generate plots
        print(f"\nGenerating plots...")
        plot_trajectory_3d(results_list, traj_type, output_dir)
        plot_tracking_error(results_list, traj_type, output_dir)
        
        all_results[traj_type] = results_list
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY - All Trajectories")
    print(f"{'='*60}\n")
    
    summary_data = []
    for traj_type, results_list in sorted(all_results.items()):
        if 'error_mean' in results_list[0]:
            mean_errors = [r['error_mean'] for r in results_list]
            summary_data.append({
                'trajectory': traj_type,
                'flights': len(results_list),
                'avg_error': np.mean(mean_errors),
                'std_error': np.std(mean_errors),
            })
        else:
            vel_means = [r['vel_mean'] for r in results_list]
            summary_data.append({
                'trajectory': traj_type,
                'flights': len(results_list),
                'avg_vel': np.mean(vel_means),
                'std_vel': np.std(vel_means),
            })
    
    if summary_data and 'avg_error' in summary_data[0]:
        print(f"{'Trajectory':<12} {'Flights':<8} {'Avg Error (m)':<15} {'Std (m)':<10}")
        print(f"{'-'*60}")
        for item in summary_data:
            print(f"{item['trajectory']:<12} {item['flights']:<8} "
                  f"{item['avg_error']:<15.4f} {item['std_error']:<10.4f}")
    else:
        print(f"{'Trajectory':<12} {'Flights':<8} {'Avg Vel (m/s)':<15} {'Std (m/s)':<10}")
        print(f"{'-'*60}")
        for item in summary_data:
            print(f"{item['trajectory']:<12} {item['flights']:<8} "
                  f"{item.get('avg_vel', 0):<15.3f} {item.get('std_vel', 0):<10.3f}")
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! Plots saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
