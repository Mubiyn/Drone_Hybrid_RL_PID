#!/usr/bin/env python3
"""
Phase 1 Simulation Analysis

Analyzes perturbation test results and generates comparison plots.
"""

import sys
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_latest_results(results_dir):
    """Load the most recent test results"""
    result_files = sorted(results_dir.glob('perturbation_test_results_*.json'))
    
    if not result_files:
        print(f"✗ No result files found in {results_dir}")
        return None
    
    latest_file = result_files[-1]
    print(f"Loading results from: {latest_file.name}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    return data


def plot_tracking_error_comparison(results, output_dir):
    """Plot tracking error comparison across all trajectories"""
    
    # Group by trajectory
    trajectories = sorted(set(r['trajectory'] for r in results))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(trajectories))
    width = 0.2
    
    # Prepare data
    pid_baseline_errors = []
    pid_dr_errors = []
    hybrid_baseline_errors = []
    hybrid_dr_errors = []
    
    for traj in trajectories:
        traj_results = [r for r in results if r['trajectory'] == traj]
        
        pid_base = next((r for r in traj_results if r['controller'] == 'PID' 
                        and not r['domain_randomization']), None)
        pid_dr = next((r for r in traj_results if r['controller'] == 'PID' 
                      and r['domain_randomization']), None)
        hybrid_base = next((r for r in traj_results if r['controller'] == 'Hybrid' 
                           and not r['domain_randomization']), None)
        hybrid_dr = next((r for r in traj_results if r['controller'] == 'Hybrid' 
                         and r['domain_randomization']), None)
        
        pid_baseline_errors.append(pid_base['mean_tracking_error'] if pid_base else 0)
        pid_dr_errors.append(pid_dr['mean_tracking_error'] if pid_dr else 0)
        hybrid_baseline_errors.append(hybrid_base['mean_tracking_error'] if hybrid_base else 0)
        hybrid_dr_errors.append(hybrid_dr['mean_tracking_error'] if hybrid_dr else 0)
    
    # Plot bars
    ax.bar(x - 1.5*width, pid_baseline_errors, width, label='PID Baseline', 
           color='#3498db', alpha=0.8)
    ax.bar(x - 0.5*width, pid_dr_errors, width, label='PID + DR', 
           color='#3498db', alpha=0.5, hatch='///')
    ax.bar(x + 0.5*width, hybrid_baseline_errors, width, label='Hybrid Baseline', 
           color='#e74c3c', alpha=0.8)
    ax.bar(x + 1.5*width, hybrid_dr_errors, width, label='Hybrid + DR', 
           color='#e74c3c', alpha=0.5, hatch='///')
    
    ax.set_xlabel('Trajectory', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Tracking Error (m)', fontsize=12, fontweight='bold')
    ax.set_title('Phase 1: Tracking Error Comparison\nSimulation with Domain Randomization', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in trajectories])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tracking_error_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: tracking_error_comparison.png")


def plot_improvement_percentages(results, output_dir):
    """Plot improvement percentages for Hybrid vs PID"""
    
    trajectories = sorted(set(r['trajectory'] for r in results))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(trajectories))
    width = 0.35
    
    baseline_improvements = []
    dr_improvements = []
    
    for traj in trajectories:
        traj_results = [r for r in results if r['trajectory'] == traj]
        
        pid_base = next((r for r in traj_results if r['controller'] == 'PID' 
                        and not r['domain_randomization']), None)
        pid_dr = next((r for r in traj_results if r['controller'] == 'PID' 
                      and r['domain_randomization']), None)
        hybrid_base = next((r for r in traj_results if r['controller'] == 'Hybrid' 
                           and not r['domain_randomization']), None)
        hybrid_dr = next((r for r in traj_results if r['controller'] == 'Hybrid' 
                         and r['domain_randomization']), None)
        
        if pid_base and hybrid_base:
            base_imp = ((pid_base['mean_tracking_error'] - hybrid_base['mean_tracking_error']) 
                       / pid_base['mean_tracking_error'] * 100)
            baseline_improvements.append(base_imp)
        else:
            baseline_improvements.append(0)
        
        if pid_dr and hybrid_dr:
            dr_imp = ((pid_dr['mean_tracking_error'] - hybrid_dr['mean_tracking_error']) 
                     / pid_dr['mean_tracking_error'] * 100)
            dr_improvements.append(dr_imp)
        else:
            dr_improvements.append(0)
    
    # Plot bars
    bars1 = ax.bar(x - width/2, baseline_improvements, width, 
                   label='Without DR', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, dr_improvements, width, 
                   label='With DR', color='#f39c12', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:+.1f}%',
                   ha='center', va='bottom' if height > 0 else 'top', 
                   fontsize=9, fontweight='bold')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Trajectory', fontsize=12, fontweight='bold')
    ax.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_title('Phase 1: Hybrid vs PID Improvement\n(Positive = Hybrid Better)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in trajectories])
    ax.legend(loc='best', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_percentages.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: improvement_percentages.png")


def plot_control_smoothness(results, output_dir):
    """Plot control smoothness comparison"""
    
    trajectories = sorted(set(r['trajectory'] for r in results))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(trajectories))
    width = 0.2
    
    pid_baseline_smooth = []
    pid_dr_smooth = []
    hybrid_baseline_smooth = []
    hybrid_dr_smooth = []
    
    for traj in trajectories:
        traj_results = [r for r in results if r['trajectory'] == traj]
        
        pid_base = next((r for r in traj_results if r['controller'] == 'PID' 
                        and not r['domain_randomization']), None)
        pid_dr = next((r for r in traj_results if r['controller'] == 'PID' 
                      and r['domain_randomization']), None)
        hybrid_base = next((r for r in traj_results if r['controller'] == 'Hybrid' 
                           and not r['domain_randomization']), None)
        hybrid_dr = next((r for r in traj_results if r['controller'] == 'Hybrid' 
                         and r['domain_randomization']), None)
        
        pid_baseline_smooth.append(pid_base['mean_smoothness'] if pid_base else 0)
        pid_dr_smooth.append(pid_dr['mean_smoothness'] if pid_dr else 0)
        hybrid_baseline_smooth.append(hybrid_base['mean_smoothness'] if hybrid_base else 0)
        hybrid_dr_smooth.append(hybrid_dr['mean_smoothness'] if hybrid_dr else 0)
    
    ax.bar(x - 1.5*width, pid_baseline_smooth, width, label='PID Baseline', 
           color='#3498db', alpha=0.8)
    ax.bar(x - 0.5*width, pid_dr_smooth, width, label='PID + DR', 
           color='#3498db', alpha=0.5, hatch='///')
    ax.bar(x + 0.5*width, hybrid_baseline_smooth, width, label='Hybrid Baseline', 
           color='#e74c3c', alpha=0.8)
    ax.bar(x + 1.5*width, hybrid_dr_smooth, width, label='Hybrid + DR', 
           color='#e74c3c', alpha=0.5, hatch='///')
    
    ax.set_xlabel('Trajectory', fontsize=12, fontweight='bold')
    ax.set_ylabel('Control Smoothness (Variance)', fontsize=12, fontweight='bold')
    ax.set_title('Phase 1: Control Smoothness Comparison\n(Lower = Smoother)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in trajectories])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'control_smoothness_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: control_smoothness_comparison.png")


def generate_summary_report(results, output_dir):
    """Generate text summary report"""
    
    trajectories = sorted(set(r['trajectory'] for r in results))
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("PHASE 1: SIMULATION PERTURBATION TESTING - SUMMARY REPORT")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append(f"Models tested: models/hybrid_robust/")
    report_lines.append(f"Episodes per test: 5")
    report_lines.append(f"Domain Randomization: Mass ±30%, Inertia ±30%, Wind 0.15N")
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("DETAILED RESULTS BY TRAJECTORY")
    report_lines.append("="*80)
    report_lines.append("")
    
    for traj in trajectories:
        traj_results = [r for r in results if r['trajectory'] == traj]
        
        pid_base = next((r for r in traj_results if r['controller'] == 'PID' 
                        and not r['domain_randomization']), None)
        pid_dr = next((r for r in traj_results if r['controller'] == 'PID' 
                      and r['domain_randomization']), None)
        hybrid_base = next((r for r in traj_results if r['controller'] == 'Hybrid' 
                           and not r['domain_randomization']), None)
        hybrid_dr = next((r for r in traj_results if r['controller'] == 'Hybrid' 
                         and r['domain_randomization']), None)
        
        report_lines.append(f"{traj.upper()}")
        report_lines.append("-"*80)
        report_lines.append(f"{'Controller':<20} {'Condition':<15} {'Tracking Error':<20} {'Smoothness':<15}")
        report_lines.append("-"*80)
        
        if pid_base:
            report_lines.append(f"{'PID':<20} {'Baseline':<15} "
                              f"{pid_base['mean_tracking_error']:.4f} ± {pid_base['std_tracking_error']:.4f}m    "
                              f"{pid_base['mean_smoothness']:.2f}")
        if pid_dr:
            report_lines.append(f"{'PID':<20} {'With DR':<15} "
                              f"{pid_dr['mean_tracking_error']:.4f} ± {pid_dr['std_tracking_error']:.4f}m    "
                              f"{pid_dr['mean_smoothness']:.2f}")
        if hybrid_base:
            report_lines.append(f"{'Hybrid':<20} {'Baseline':<15} "
                              f"{hybrid_base['mean_tracking_error']:.4f} ± {hybrid_base['std_tracking_error']:.4f}m    "
                              f"{hybrid_base['mean_smoothness']:.2f}")
        if hybrid_dr:
            report_lines.append(f"{'Hybrid':<20} {'With DR':<15} "
                              f"{hybrid_dr['mean_tracking_error']:.4f} ± {hybrid_dr['std_tracking_error']:.4f}m    "
                              f"{hybrid_dr['mean_smoothness']:.2f}")
        
        report_lines.append("")
        
        # Calculate improvements
        if pid_base and hybrid_base:
            base_imp = ((pid_base['mean_tracking_error'] - hybrid_base['mean_tracking_error']) 
                       / pid_base['mean_tracking_error'] * 100)
            report_lines.append(f"Improvement (Baseline):  {base_imp:+.1f}%")
        
        if pid_dr and hybrid_dr:
            dr_imp = ((pid_dr['mean_tracking_error'] - hybrid_dr['mean_tracking_error']) 
                     / pid_dr['mean_tracking_error'] * 100)
            report_lines.append(f"Improvement (With DR):   {dr_imp:+.1f}%")
        
        report_lines.append("")
        report_lines.append("")
    
    # Overall summary
    report_lines.append("="*80)
    report_lines.append("OVERALL SUMMARY")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append(f"{'Trajectory':<15} {'Baseline Improvement':<25} {'DR Improvement':<25}")
    report_lines.append("-"*80)
    
    avg_base_imp = []
    avg_dr_imp = []
    
    for traj in trajectories:
        traj_results = [r for r in results if r['trajectory'] == traj]
        
        pid_base = next((r for r in traj_results if r['controller'] == 'PID' 
                        and not r['domain_randomization']), None)
        pid_dr = next((r for r in traj_results if r['controller'] == 'PID' 
                      and r['domain_randomization']), None)
        hybrid_base = next((r for r in traj_results if r['controller'] == 'Hybrid' 
                           and not r['domain_randomization']), None)
        hybrid_dr = next((r for r in traj_results if r['controller'] == 'Hybrid' 
                         and r['domain_randomization']), None)
        
        base_imp = 0
        dr_imp = 0
        
        if pid_base and hybrid_base:
            base_imp = ((pid_base['mean_tracking_error'] - hybrid_base['mean_tracking_error']) 
                       / pid_base['mean_tracking_error'] * 100)
            avg_base_imp.append(base_imp)
        
        if pid_dr and hybrid_dr:
            dr_imp = ((pid_dr['mean_tracking_error'] - hybrid_dr['mean_tracking_error']) 
                     / pid_dr['mean_tracking_error'] * 100)
            avg_dr_imp.append(dr_imp)
        
        report_lines.append(f"{traj.capitalize():<15} {base_imp:+6.1f}%                   {dr_imp:+6.1f}%")
    
    report_lines.append("-"*80)
    if avg_base_imp and avg_dr_imp:
        report_lines.append(f"{'AVERAGE':<15} {np.mean(avg_base_imp):+6.1f}%                   {np.mean(avg_dr_imp):+6.1f}%")
    report_lines.append("")
    
    report_lines.append("="*80)
    report_lines.append("KEY FINDINGS")
    report_lines.append("="*80)
    report_lines.append("")
    
    if avg_base_imp:
        avg_base = np.mean(avg_base_imp)
        if avg_base > 0:
            report_lines.append(f"✓ Hybrid controller shows {avg_base:.1f}% average improvement over PID baseline")
        else:
            report_lines.append(f"✗ Hybrid controller performs {abs(avg_base):.1f}% worse than PID baseline")
    
    if avg_dr_imp:
        avg_dr = np.mean(avg_dr_imp)
        if avg_dr > 0:
            report_lines.append(f"✓ With domain randomization, Hybrid shows {avg_dr:.1f}% average improvement")
        else:
            report_lines.append(f"✗ With domain randomization, Hybrid performs {abs(avg_dr):.1f}% worse")
    
    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Save report
    report_file = output_dir / 'summary_report.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ Saved: summary_report.txt")
    
    # Also print to console
    print("\n" + '\n'.join(report_lines))


def main():
    results_dir = Path('results/phase1_simulation/perturbation_tests')
    output_dir = Path('results/phase1_simulation/comparison_plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("PHASE 1: SIMULATION ANALYSIS")
    print(f"{'='*70}\n")
    
    # Load results
    data = load_latest_results(results_dir)
    if not data:
        return
    
    results = data['results']
    
    print(f"\nGenerating analysis plots...")
    print(f"Output directory: {output_dir}\n")
    
    # Generate plots
    plot_tracking_error_comparison(results, output_dir)
    plot_improvement_percentages(results, output_dir)
    plot_control_smoothness(results, output_dir)
    
    # Generate summary report
    generate_summary_report(results, output_dir)
    
    print(f"\n{'='*70}")
    print(f"✓ Analysis complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
