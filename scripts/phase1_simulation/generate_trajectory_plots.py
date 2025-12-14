#!/usr/bin/env python3
"""
Generate Phase 1 Trajectory Visualization Plots

Creates detailed trajectory plots for PID and Hybrid controllers under domain randomization,
using the EXACT SAME methodology as the original eval_comparison.py script that generated
the validated results in results/figures/.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.envs.BaseTrackAviary import BaseTrackAviary
from src.envs.RobustTrackAviary import RobustTrackAviary
from src.envs.HybridAviary import HybridAviary
from src.controllers.pid_controller import PIDController


def eval_controller(controller_type, model_path=None, trajectory_type='circle', duration=15.0, robust=True):
    """
    Evaluate a controller on a specific trajectory.
    EXACT COPY of eval_comparison.py methodology to ensure consistency.
    
    controller_type: 'pid' or 'hybrid'
    robust: If True, enables wind and mass randomization (SHOULD BE TRUE for validated results)
    
    Returns: (errors_array, trajectory_positions_array, target_positions_array, time_array)
    """
    print(f"  Evaluating {controller_type.upper()} on {trajectory_type} (Robust: {robust})...")
    
    # Setup Env Factory - EXACTLY like eval_comparison.py
    def make_env():
        if controller_type == 'hybrid':
            return HybridAviary(
                trajectory_type=trajectory_type,
                gui=False,
                record=False,
                domain_randomization=robust
            )
        else:  # PID
            if robust:
                return RobustTrackAviary(
                    trajectory_type=trajectory_type,
                    gui=False,
                    record=False,
                    domain_randomization=True
                )
            else:
                return BaseTrackAviary(
                    trajectory_type=trajectory_type,
                    gui=False,
                    record=False
                )
    
    # Create Vectorized Env
    env = DummyVecEnv([make_env])
    
    # Load Model
    model = None
    if controller_type == 'hybrid':
        if model_path and os.path.exists(model_path):
            try:
                model = PPO.load(model_path)
                print(f"    Loaded model from {model_path}")
                
                # Load VecNormalize stats if available
                vec_norm_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
                if os.path.exists(vec_norm_path):
                    print(f"    Loading normalization stats from {vec_norm_path}")
                    env = VecNormalize.load(vec_norm_path, env)
                    env.training = False
                    env.norm_reward = False
                else:
                    print("    No normalization stats found. Using raw environment.")
            except Exception as e:
                print(f"    Failed to load model from {model_path}: {e}")
                env.close()
                return None, None, None, None
        else:
            print(f"    Model path not found: {model_path}")
            env.close()
            return None, None, None, None
    
    # PID Controller (for 'pid' case)
    pid = None
    if controller_type == 'pid':
        pid = PIDController(freq=240)
    
    obs = env.reset()
    
    errors = []
    trajectory_trace = []
    target_trace = []
    time_trace = []
    
    # Run simulation - EXACTLY like eval_comparison.py
    steps = int(duration * 240)
    for i in range(steps):
        # obs shape can be (1, 18) or (1, 1, 18) depending on wrapping
        if obs.ndim == 3:
            current_obs = obs[0, 0]  # Extract from (1, 1, 18)
        else:
            current_obs = obs[0]  # Extract from (1, 18)
        
        # Obs structure: [x,y,z, r,p,y, vx,vy,vz, wx,wy,wz, pos_err(3), vel_err(3)]
        current_pos = current_obs[0:3]
        pos_error = current_obs[12:15]
        target_pos = current_pos + pos_error
        
        trajectory_trace.append(current_pos)
        target_trace.append(target_pos)
        errors.append(np.linalg.norm(pos_error))
        time_trace.append(i / 240.0)
        
        if controller_type == 'pid':
            current_vel = current_obs[6:9]
            vel_error = current_obs[15:18]
            target_vel = current_vel + vel_error
            # PID needs only the state (first 12 dims)
            state_obs = current_obs[0:12]
            action = pid.compute_control(state_obs, target_pos, target_vel)
            action = np.array([[action]])  # (1, 1, 4)
        elif model:
            action, _ = model.predict(obs, deterministic=True)
            # Ensure action is (1, 1, 4) if model outputs (1, 4)
            if action.ndim == 2:
                action = action.reshape(1, 1, 4)
        else:
            action = [env.action_space.sample()]
        
        obs, _, _, _ = env.step(action)
    
    env.close()
    return np.array(errors), np.array(trajectory_trace), np.array(target_trace), np.array(time_trace)


def plot_trajectory_comparison(trajectory_type, pid_data, hybrid_data, output_path):
    """
    Create detailed 6-panel comparison plot similar to Phase 2 perturbation analysis.
    """
    pid_err, pid_pos, pid_target, pid_time = pid_data
    hybrid_err, hybrid_pos, hybrid_target, hybrid_time = hybrid_data
    
    if pid_err is None or hybrid_err is None:
        print(f"  ✗ Skipping {trajectory_type} - missing data")
        return
    
    fig = plt.figure(figsize=(18, 10))
    
    # Calculate metrics
    pid_mean_err = np.mean(pid_err)
    hybrid_mean_err = np.mean(hybrid_err)
    improvement = ((pid_mean_err - hybrid_mean_err) / pid_mean_err) * 100
    
    # Panel 1: 3D Trajectory - PID
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(pid_target[:, 0], pid_target[:, 1], pid_target[:, 2],
             'g--', linewidth=2, label='Target', alpha=0.7)
    ax1.plot(pid_pos[:, 0], pid_pos[:, 1], pid_pos[:, 2],
             'b-', linewidth=1.5, label='Actual', alpha=0.8)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'PID Controller - {trajectory_type.capitalize()}\n3D Trajectory', 
                  fontsize=11, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Tracking Error Over Time
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(pid_time, pid_err, 'b-', label=f'PID (Mean: {pid_mean_err:.4f}m)', alpha=0.8)
    ax2.plot(hybrid_time, hybrid_err, 'r-', label=f'Hybrid (Mean: {hybrid_mean_err:.4f}m)', alpha=0.8)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position Error (m)')
    ax2.set_title('Tracking Error Over Time', fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: XY Plane View
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(pid_target[:, 0], pid_target[:, 1], 'g--', linewidth=2, label='Target', alpha=0.7)
    ax3.plot(pid_pos[:, 0], pid_pos[:, 1], 'b-', linewidth=1.5, label='PID', alpha=0.7)
    ax3.plot(hybrid_pos[:, 0], hybrid_pos[:, 1], 'r-', linewidth=1.5, label='Hybrid', alpha=0.7)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('XY Plane Comparison', fontsize=11, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # Panel 4: 3D Trajectory - Hybrid
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax4.plot(hybrid_target[:, 0], hybrid_target[:, 1], hybrid_target[:, 2],
             'g--', linewidth=2, label='Target', alpha=0.7)
    ax4.plot(hybrid_pos[:, 0], hybrid_pos[:, 1], hybrid_pos[:, 2],
             'r-', linewidth=1.5, label='Actual', alpha=0.8)
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_zlabel('Z (m)')
    ax4.set_title(f'Hybrid Controller - {trajectory_type.capitalize()}\n3D Trajectory', 
                  fontsize=11, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Error Comparison Bar Chart
    ax5 = fig.add_subplot(2, 3, 5)
    controllers = ['PID', 'Hybrid']
    mean_errors = [pid_mean_err, hybrid_mean_err]
    colors = ['#3498db', '#e74c3c']
    bars = ax5.bar(controllers, mean_errors, color=colors, alpha=0.7)
    ax5.set_ylabel('Mean Position Error (m)')
    ax5.set_title(f'Performance Comparison\nImprovement: {improvement:+.1f}%', 
                  fontsize=11, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}m',
                ha='center', va='bottom', fontsize=9)
    
    # Panel 6: Summary Statistics
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"""
PERFORMANCE SUMMARY
{trajectory_type.upper()} Trajectory with Domain Randomization (Robust)

PID Controller:
  • Mean Error: {pid_mean_err:.4f} m
  • Max Error:  {np.max(pid_err):.4f} m
  • Std Dev:    {np.std(pid_err):.4f} m

Hybrid RL-PID Controller:
  • Mean Error: {hybrid_mean_err:.4f} m
  • Max Error:  {np.max(hybrid_err):.4f} m
  • Std Dev:    {np.std(hybrid_err):.4f} m

Improvement: {improvement:+.2f}%
{'✓ Hybrid performs better' if improvement > 0 else '✗ PID performs better'}

Test Duration: {pid_time[-1]:.1f}s
Control Frequency: 240 Hz
Domain Randomization: ENABLED
  • Wind: ±0.15N constant bias
  • Mass: ±30% variation
  • Inertia: ±30% variation
"""
    
    ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Saved: {output_path.name}")
    print(f"      PID error: {pid_mean_err:.4f}m")
    print(f"      Hybrid error: {hybrid_mean_err:.4f}m")
    print(f"      Improvement: {improvement:+.1f}%")


def main():
    """
    Generate trajectory comparison plots for all 5 trajectories.
    Uses ROBUST mode to match the original validated eval_comparison.py results.
    """
    trajectories = ['circle', 'figure8', 'hover', 'spiral', 'waypoint']
    models_base = Path("models/hybrid_robust")
    output_dir = Path("results/phase1_simulation/trajectory_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    duration = 15.0  # seconds
    robust = True  # MUST BE TRUE to match original validated results
    
    print(f"\n{'='*60}")
    print("PHASE 1 TRAJECTORY VISUALIZATION")
    print(f"{'='*60}\n")
    print(f"Output directory: {output_dir}")
    print(f"Test duration: {duration}s")
    print(f"Domain randomization: {'ENABLED' if robust else 'DISABLED'} (Robust Mode)")
    print(f"Note: Using EXACT methodology as eval_comparison.py")
    print(f"\n{'='*60}\n")
    
    for traj in trajectories:
        print(f"Processing: {traj.upper()}")
        print(f"{'='*60}")
        
        model_path = models_base / traj / "final_model.zip"
        print(f"  Model: {model_path}")
        
        # Test PID
        pid_data = eval_controller('pid', trajectory_type=traj, duration=duration, robust=robust)
        
        # Test Hybrid
        hybrid_data = eval_controller('hybrid', model_path=str(model_path), 
                                     trajectory_type=traj, duration=duration, robust=robust)
        
        # Generate comparison plot
        output_path = output_dir / f'{traj}_trajectory_comparison.png'
        plot_trajectory_comparison(traj, pid_data, hybrid_data, output_path)
        print()
    
    print(f"{'='*60}")
    print(f"COMPLETE - All plots saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
