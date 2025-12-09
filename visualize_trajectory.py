#!/usr/bin/env python
"""
Visualize actual drone trajectory vs target trajectory
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.envs.HybridAviary import HybridAviary
from src.envs.RobustTrackAviary import RobustTrackAviary
from src.controllers.pid_controller import PIDController
import argparse

def visualize_trajectory(controller_type, model_path=None, trajectory_type='circle', duration=15.0):
    """
    Run simulation and collect trajectory data for visualization
    """
    print(f"Collecting trajectory: {controller_type} | {trajectory_type}")
    
    # Setup environment
    def make_env():
        if controller_type == 'hybrid':
            return HybridAviary(trajectory_type=trajectory_type, 
                               gui=False, 
                               record=False, 
                               domain_randomization=True)
        else:
            return RobustTrackAviary(trajectory_type=trajectory_type, 
                                    gui=False, 
                                    record=False, 
                                    domain_randomization=True)
    
    env = DummyVecEnv([make_env])
    
    # Load model if needed
    model = None
    if controller_type in ['hybrid', 'ppo'] and model_path:
        model = PPO.load(model_path)
        vec_norm_path = model_path.replace('final_model.zip', 'vec_normalize.pkl')
        try:
            env = VecNormalize.load(vec_norm_path, env)
            env.training = False
            env.norm_reward = False
        except:
            pass
    
    # PID controller
    pid = None
    if controller_type == 'pid':
        pid = PIDController(freq=240)
    
    # Run simulation
    obs = env.reset()
    steps = int(duration * 240)
    
    actual_positions = []
    target_positions = []
    
    for i in range(steps):
        # Handle obs shape
        if obs.ndim == 3:
            current_obs = obs[0, 0]
        else:
            current_obs = obs[0]
        
        # Extract positions
        current_pos = current_obs[0:3]
        pos_error = current_obs[12:15]
        target_pos = current_pos + pos_error
        
        actual_positions.append(current_pos.copy())
        target_positions.append(target_pos.copy())
        
        # Compute action
        if controller_type == 'pid':
            current_vel = current_obs[6:9]
            vel_error = current_obs[15:18]
            target_vel = current_vel + vel_error
            state_obs = current_obs[0:12]
            action = pid.compute_control(state_obs, target_pos, target_vel)
            action = np.array([[action]])
        elif model:
            action, _ = model.predict(obs, deterministic=True)
            if action.ndim == 2:
                action = action.reshape(1, 1, 4)
        else:
            action = [env.action_space.sample()]
        
        obs, _, _, _ = env.step(action)
    
    env.close()
    
    actual_positions = np.array(actual_positions)
    target_positions = np.array(target_positions)
    
    return actual_positions, target_positions

def plot_trajectories(trajectories_data, trajectory_type):
    """
    Plot 3D trajectories
    """
    fig = plt.figure(figsize=(16, 10))
    
    # 3D plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    
    # Plot target
    target = trajectories_data[0]['target']
    ax1.plot(target[:, 0], target[:, 1], target[:, 2], 
             'k--', linewidth=2, label='Target', alpha=0.7)
    
    # Plot each controller
    colors = {'PID': 'red', 'Hybrid': 'blue', 'PPO': 'green'}
    for data in trajectories_data:
        actual = data['actual']
        ax1.plot(actual[:, 0], actual[:, 1], actual[:, 2], 
                color=colors.get(data['name'], 'gray'), 
                linewidth=1.5, label=data['name'], alpha=0.8)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'3D Trajectory: {trajectory_type.upper()}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # XY plane
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(target[:, 0], target[:, 1], 'k--', linewidth=2, label='Target', alpha=0.7)
    for data in trajectories_data:
        actual = data['actual']
        ax2.plot(actual[:, 0], actual[:, 1], 
                color=colors.get(data['name'], 'gray'), 
                linewidth=1.5, label=data['name'], alpha=0.8)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Plane View')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # XZ plane
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(target[:, 0], target[:, 2], 'k--', linewidth=2, label='Target', alpha=0.7)
    for data in trajectories_data:
        actual = data['actual']
        ax3.plot(actual[:, 0], actual[:, 2], 
                color=colors.get(data['name'], 'gray'), 
                linewidth=1.5, label=data['name'], alpha=0.8)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('XZ Plane View')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Position errors over time
    ax4 = fig.add_subplot(2, 3, 4)
    time = np.arange(len(target)) / 240.0
    for data in trajectories_data:
        actual = data['actual']
        errors = np.linalg.norm(actual - target, axis=1)
        ax4.plot(time, errors, 
                color=colors.get(data['name'], 'gray'), 
                linewidth=1.5, label=f"{data['name']} (RMSE: {np.sqrt(np.mean(errors**2)):.3f}m)")
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Position Error (m)')
    ax4.set_title('Tracking Error Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Error statistics
    ax5 = fig.add_subplot(2, 3, 5)
    names = []
    rmse_values = []
    max_errors = []
    for data in trajectories_data:
        actual = data['actual']
        errors = np.linalg.norm(actual - target, axis=1)
        names.append(data['name'])
        rmse_values.append(np.sqrt(np.mean(errors**2)))
        max_errors.append(np.max(errors))
    
    x = np.arange(len(names))
    width = 0.35
    ax5.bar(x - width/2, rmse_values, width, label='RMSE', alpha=0.8)
    ax5.bar(x + width/2, max_errors, width, label='Max Error', alpha=0.8)
    ax5.set_ylabel('Error (m)')
    ax5.set_title('Error Statistics')
    ax5.set_xticks(x)
    ax5.set_xticklabels(names)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # YZ plane
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(target[:, 1], target[:, 2], 'k--', linewidth=2, label='Target', alpha=0.7)
    for data in trajectories_data:
        actual = data['actual']
        ax6.plot(actual[:, 1], actual[:, 2], 
                color=colors.get(data['name'], 'gray'), 
                linewidth=1.5, label=data['name'], alpha=0.8)
    ax6.set_xlabel('Y (m)')
    ax6.set_ylabel('Z (m)')
    ax6.set_title('YZ Plane View')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    filename = f"results/figures/trajectory_{trajectory_type}_comparison.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved trajectory plot to {filename}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", type=str, default="circle", help="Trajectory type")
    parser.add_argument("--duration", type=float, default=15.0, help="Duration in seconds")
    args = parser.parse_args()
    
    trajectories = []
    
    # Collect PID trajectory
    print("\n" + "="*50)
    print("Collecting PID trajectory...")
    actual, target = visualize_trajectory('pid', trajectory_type=args.traj, duration=args.duration)
    trajectories.append({'name': 'PID', 'actual': actual, 'target': target})
    
    # Collect Hybrid trajectory if model exists
    hybrid_model = f"models/hybrid_robust/{args.traj}/final_model.zip"
    import os
    if os.path.exists(hybrid_model):
        print("Collecting Hybrid trajectory...")
        actual, target = visualize_trajectory('hybrid', model_path=hybrid_model, 
                                             trajectory_type=args.traj, duration=args.duration)
        trajectories.append({'name': 'Hybrid', 'actual': actual, 'target': target})
    else:
        print(f"Hybrid model not found: {hybrid_model}")
    
    # Plot
    print("\nGenerating plots...")
    plot_trajectories(trajectories, args.traj)
