#!/usr/bin/env python3
"""
Comprehensive Analysis of PID vs Hybrid Controllers Under Perturbations

Generates all plots and performance comparisons for domain randomization experiments.
"""

import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

sns.set_style("whitegrid")
sns.set_palette("husl")


def load_flight_data(filepath):
    """Load flight data from pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def plot_3d_trajectory(data, title, output_file):
    """Plot 3D trajectory: desired vs actual"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    states = data['states']
    targets = data['targets']
    
    # Plot desired trajectory
    ax.plot(targets[:, 0], targets[:, 1], targets[:, 2],
            'b-', linewidth=2, label='Desired', alpha=0.7)
    
    # Plot actual trajectory
    ax.plot(states[:, 0], states[:, 1], states[:, 2],
            'r-', linewidth=2, label='Actual', alpha=0.7)
    
    # Mark start and end
    ax.scatter([targets[0, 0]], [targets[0, 1]], [targets[0, 2]],
               c='g', s=100, marker='o', label='Start')
    ax.scatter([targets[-1, 0]], [targets[-1, 1]], [targets[-1, 2]],
               c='r', s=100, marker='x', label='End')
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def plot_tracking_performance(data, title, output_file):
    """Plot tracking error, battery, and control commands over time"""
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 2, figure=fig)
    
    timestamps = data['timestamps']
    states = data['states']
    targets = data['targets']
    errors = data['errors']  # Now Z-axis error only
    battery = data['battery_history']
    actions = data['actions']
    
    # Height tracking (Z-axis - reliable from barometer)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(timestamps, states[:, 2], 'b-', label='Z Actual', linewidth=2)
    ax1.plot(timestamps, targets[:, 2], 'r--', label='Z Desired', linewidth=2, alpha=0.7)
    ax1.fill_between(timestamps, states[:, 2], targets[:, 2], alpha=0.3)
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Height (m)', fontsize=11)
    ax1.set_title('Height Tracking (Barometer - Reliable)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Z-axis tracking error
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(timestamps, errors, 'r-', linewidth=2)
    ax2.axhline(y=np.mean(errors), color='b', linestyle='--', 
                label=f'Mean: {np.mean(errors):.3f}m', linewidth=1.5)
    ax2.fill_between(timestamps, 0, errors, alpha=0.3)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Z-Axis Error (m)', fontsize=11)
    ax2.set_title('Height Tracking Error (Reliable Metric)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Battery consumption
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(timestamps, battery, 'g-', linewidth=2)
    battery_used = battery[0] - battery[-1]
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Battery (%)', fontsize=11)
    ax3.set_title(f'Battery (Used: {battery_used}%)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Control commands
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(timestamps, actions[:, 0], label='Left/Right', linewidth=1.5)
    ax4.plot(timestamps, actions[:, 1], label='Forward/Back', linewidth=1.5)
    ax4.plot(timestamps, actions[:, 2], label='Up/Down', linewidth=1.5)
    ax4.plot(timestamps, actions[:, 3], label='Yaw Rate', linewidth=1.5)
    ax4.set_xlabel('Time (s)', fontsize=11)
    ax4.set_ylabel('Command (-100 to 100)', fontsize=11)
    ax4.set_title('Control Commands', fontsize=12, fontweight='bold')
    ax4.legend(ncol=4, fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-105, 105)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def plot_comparison(pid_data, hybrid_data, trajectory_type, perturbation_type, output_dir):
    """Compare PID vs Hybrid performance"""
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(4, 2, figure=fig)
    
    # Add warning about X/Y position
    fig.text(0.5, 0.97, 'Note: X/Y positions use dead reckoning (unreliable). Focus on Z-axis error and control smoothness.',
             ha='center', fontsize=10, style='italic', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Trajectories comparison (3D) - for visualization only
    ax1 = fig.add_subplot(gs[0, :], projection='3d')
    
    targets = pid_data['targets']
    ax1.plot(targets[:, 0], targets[:, 1], targets[:, 2],
             'k--', linewidth=2, label='Desired', alpha=0.8)
    ax1.plot(pid_data['states'][:, 0], pid_data['states'][:, 1], pid_data['states'][:, 2],
             'r-', linewidth=2, label='PID', alpha=0.7)
    ax1.plot(hybrid_data['states'][:, 0], hybrid_data['states'][:, 1], hybrid_data['states'][:, 2],
             'b-', linewidth=2, label='Hybrid RL', alpha=0.7)
    
    ax1.set_xlabel('X (m - unreliable)', fontsize=10)
    ax1.set_ylabel('Y (m - unreliable)', fontsize=10)
    ax1.set_zlabel('Z (m - barometer)', fontsize=10)
    ax1.set_title(f'3D Trajectory - {trajectory_type.upper()} (X/Y from dead reckoning)', 
                  fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Z-axis tracking error comparison (RELIABLE)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(pid_data['timestamps'], pid_data['errors'], 'r-', 
             label=f'PID (mean: {np.mean(pid_data["errors"]):.3f}m)', linewidth=2)
    ax2.plot(hybrid_data['timestamps'], hybrid_data['errors'], 'b-',
             label=f'Hybrid (mean: {np.mean(hybrid_data["errors"]):.3f}m)', linewidth=2)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Z-Axis Error (m)', fontsize=11)
    ax2.set_title('Height Tracking Error (Reliable)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Error distribution
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(pid_data['errors'], bins=30, alpha=0.6, label='PID', color='r', density=True)
    ax3.hist(hybrid_data['errors'], bins=30, alpha=0.6, label='Hybrid', color='b', density=True)
    ax3.set_xlabel('Z-Axis Error (m)', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Control smoothness (command variance - lower is smoother)
    ax4 = fig.add_subplot(gs[2, 0])
    pid_smoothness = np.std(np.diff(pid_data['actions'], axis=0), axis=0)
    hybrid_smoothness = np.std(np.diff(hybrid_data['actions'], axis=0), axis=0)
    
    cmd_labels = ['LR', 'FB', 'UD', 'Yaw']
    x = np.arange(len(cmd_labels))
    width = 0.35
    ax4.bar(x - width/2, pid_smoothness, width, label='PID', color='r', alpha=0.7)
    ax4.bar(x + width/2, hybrid_smoothness, width, label='Hybrid', color='b', alpha=0.7)
    ax4.set_ylabel('Command Variance', fontsize=11)
    ax4.set_title('Control Smoothness (Lower is Better)', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(cmd_labels, fontsize=10)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Battery comparison
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(pid_data['timestamps'], pid_data['battery_history'], 'r-', 
             label=f'PID (used: {pid_data["battery_history"][0] - pid_data["battery_history"][-1]}%)',
             linewidth=2)
    ax5.plot(hybrid_data['timestamps'], hybrid_data['battery_history'], 'b-',
             label=f'Hybrid (used: {hybrid_data["battery_history"][0] - hybrid_data["battery_history"][-1]}%)',
             linewidth=2)
    ax5.set_xlabel('Time (s)', fontsize=11)
    ax5.set_ylabel('Battery (%)', fontsize=11)
    ax5.set_title('Battery Consumption', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # Performance metrics bar chart (Z-error based)
    ax6 = fig.add_subplot(gs[3, :])
    metrics = ['Mean Z-Error\n(m)', 'Max Z-Error\n(m)', 'Std Z-Error\n(m)', 
               'Control Smoothness\n(avg variance)', 'Battery\nUsed (%)']
    pid_metrics = [
        np.mean(pid_data['errors']),
        np.max(pid_data['errors']),
        np.std(pid_data['errors']),
        np.mean(np.std(np.diff(pid_data['actions'], axis=0), axis=0)),
        pid_data['battery_history'][0] - pid_data['battery_history'][-1]
    ]
    hybrid_metrics = [
        np.mean(hybrid_data['errors']),
        np.max(hybrid_data['errors']),
        np.std(hybrid_data['errors']),
        np.mean(np.std(np.diff(hybrid_data['actions'], axis=0), axis=0)),
        hybrid_data['battery_history'][0] - hybrid_data['battery_history'][-1]
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax6.bar(x - width/2, pid_metrics, width, label='PID', color='r', alpha=0.7)
    ax6.bar(x + width/2, hybrid_metrics, width, label='Hybrid', color='b', alpha=0.7)
    ax6.set_ylabel('Value', fontsize=11)
    ax6.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics, fontsize=9)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Calculate improvement
    improvement = ((np.mean(pid_data['errors']) - np.mean(hybrid_data['errors'])) / 
                   np.mean(pid_data['errors']) * 100)
    
    plt.suptitle(f'{trajectory_type.upper()} - {perturbation_type.upper()} Perturbation\n'
                 f'Hybrid RL Improvement: {improvement:+.1f}%',
                 fontsize=14, fontweight='bold', y=0.998)
    plt.tight_layout()
    
    output_file = output_dir / f'comparison_{trajectory_type}_{perturbation_type}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")
    
    return improvement


def generate_summary_report(all_results, output_dir):
    """Generate summary comparison across all trajectories"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    trajectories = list(all_results.keys())
    
    # Mean error comparison
    ax1 = axes[0, 0]
    pid_errors = [all_results[t]['pid_mean_error'] for t in trajectories]
    hybrid_errors = [all_results[t]['hybrid_mean_error'] for t in trajectories]
    
    x = np.arange(len(trajectories))
    width = 0.35
    ax1.bar(x - width/2, pid_errors, width, label='PID', color='r', alpha=0.7)
    ax1.bar(x + width/2, hybrid_errors, width, label='Hybrid', color='b', alpha=0.7)
    ax1.set_ylabel('Mean Position Error (m)', fontsize=11)
    ax1.set_title('Mean Tracking Error', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([t.capitalize() for t in trajectories], fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Max error comparison
    ax2 = axes[0, 1]
    pid_max = [all_results[t]['pid_max_error'] for t in trajectories]
    hybrid_max = [all_results[t]['hybrid_max_error'] for t in trajectories]
    
    ax2.bar(x - width/2, pid_max, width, label='PID', color='r', alpha=0.7)
    ax2.bar(x + width/2, hybrid_max, width, label='Hybrid', color='b', alpha=0.7)
    ax2.set_ylabel('Max Position Error (m)', fontsize=11)
    ax2.set_title('Maximum Tracking Error', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([t.capitalize() for t in trajectories], fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Improvement percentage
    ax3 = axes[1, 0]
    improvements = [all_results[t]['improvement'] for t in trajectories]
    colors = ['g' if imp > 0 else 'r' for imp in improvements]
    ax3.bar(x, improvements, color=colors, alpha=0.7)
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax3.set_ylabel('Improvement (%)', fontsize=11)
    ax3.set_title('Hybrid RL Improvement over PID', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([t.capitalize() for t in trajectories], fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Battery usage
    ax4 = axes[1, 1]
    pid_battery = [all_results[t]['pid_battery'] for t in trajectories]
    hybrid_battery = [all_results[t]['hybrid_battery'] for t in trajectories]
    
    ax4.bar(x - width/2, pid_battery, width, label='PID', color='r', alpha=0.7)
    ax4.bar(x + width/2, hybrid_battery, width, label='Hybrid', color='b', alpha=0.7)
    ax4.set_ylabel('Battery Used (%)', fontsize=11)
    ax4.set_title('Battery Consumption', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([t.capitalize() for t in trajectories], fontsize=10)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Overall Performance Summary - Domain Randomization Tests',
                 fontsize=14, fontweight='bold', y=0.998)
    plt.tight_layout()
    
    output_file = output_dir / 'summary_all_trajectories.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_file}")


def main():
    print("\n" + "="*60)
    print("COMPREHENSIVE ANALYSIS - PID vs HYBRID RL")
    print("="*60 + "\n")
    
    # Setup directories
    data_dir = Path("data/flight_logs")
    output_dir = Path("results/perturbation_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all flight data files
    pid_files = sorted(data_dir.glob("pid_*.pkl"))
    hybrid_files = sorted(data_dir.glob("hybrid_*.pkl"))
    
    print(f"Found {len(pid_files)} PID flights")
    print(f"Found {len(hybrid_files)} Hybrid flights\n")
    
    # Group by trajectory and perturbation
    all_results = {}
    
    for pid_file in pid_files:
        pid_data = load_flight_data(pid_file)
        traj_type = pid_data['trajectory_type']
        pert_type = pid_data['perturbation_type']
        
        # Find matching hybrid file
        hybrid_file = None
        for hf in hybrid_files:
            hd = load_flight_data(hf)
            if hd['trajectory_type'] == traj_type and hd['perturbation_type'] == pert_type:
                hybrid_file = hf
                hybrid_data = hd
                break
        
        if not hybrid_file:
            print(f"⚠️  No matching hybrid data for {traj_type} + {pert_type}")
            continue
        
        print(f"Analyzing: {traj_type} + {pert_type}")
        
        # Create trajectory-specific output directory
        traj_output_dir = output_dir / traj_type
        traj_output_dir.mkdir(exist_ok=True)
        
        # Plot PID trajectory
        plot_3d_trajectory(
            pid_data,
            f'PID - {traj_type.upper()} ({pert_type})',
            traj_output_dir / f'pid_{traj_type}_{pert_type}_3d.png'
        )
        
        # Plot PID performance
        plot_tracking_performance(
            pid_data,
            f'PID Performance - {traj_type.upper()} ({pert_type})',
            traj_output_dir / f'pid_{traj_type}_{pert_type}_performance.png'
        )
        
        # Plot Hybrid trajectory
        plot_3d_trajectory(
            hybrid_data,
            f'Hybrid RL - {traj_type.upper()} ({pert_type})',
            traj_output_dir / f'hybrid_{traj_type}_{pert_type}_3d.png'
        )
        
        # Plot Hybrid performance
        plot_tracking_performance(
            hybrid_data,
            f'Hybrid RL Performance - {traj_type.upper()} ({pert_type})',
            traj_output_dir / f'hybrid_{traj_type}_{pert_type}_performance.png'
        )
        
        # Plot comparison
        improvement = plot_comparison(
            pid_data, hybrid_data, traj_type, pert_type, traj_output_dir
        )
        
        # Store results
        key = f"{traj_type}_{pert_type}"
        all_results[traj_type] = {
            'pid_mean_error': np.mean(pid_data['errors']),
            'pid_max_error': np.max(pid_data['errors']),
            'hybrid_mean_error': np.mean(hybrid_data['errors']),
            'hybrid_max_error': np.max(hybrid_data['errors']),
            'improvement': improvement,
            'pid_battery': pid_data['battery_history'][0] - pid_data['battery_history'][-1],
            'hybrid_battery': hybrid_data['battery_history'][0] - hybrid_data['battery_history'][-1]
        }
        
        print()
    
    # Generate overall summary
    if all_results:
        print("Generating summary report...")
        generate_summary_report(all_results, output_dir)
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
