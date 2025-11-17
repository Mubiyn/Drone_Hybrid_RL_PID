#!/usr/bin/env python3
import os
import sys
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.utils import sync

from src.controllers.pid_controller import PIDController
from src.utils.trajectories import get_trajectory
from src.utils.metrics import evaluate_trajectory


def run_task(controller, task_name, gui=True, record=False):
    trajectory = get_trajectory(task_name)
    
    # Ground-start initialization: 3cm above ground (real-world deployment)
    # Matches actual Crazyflie placed on landing pad (2-3cm height to center)
    # Well above collision threshold (1.25cm) for safe takeoff
    initial_pos = np.array([0.0, 0.0, 0.03])
    
    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.array([initial_pos]),
        physics=Physics.PYB,
        gui=gui,
        record=record,
        pyb_freq=240,
        ctrl_freq=48
    )
    
    obs, info = env.reset()
    controller.reset()
    
    # No warmup - measure full trajectory including takeoff transient
    # This matches RL training where agent learns from t=0
    positions = []
    actions_taken = []
    targets = []
    
    for step in range(len(trajectory)):
        target_pos = trajectory[step]
        
        action = controller.compute_control(obs[0], target_pos)
        obs, reward, terminated, truncated, info = env.step(np.array([action]))
        
        positions.append(obs[0][0:3].copy())
        actions_taken.append(action.copy())
        targets.append(target_pos.copy())
        
        if gui:
            sync(step, env.CTRL_TIMESTEP, env.CTRL_FREQ)
        
        if terminated or truncated:
            break
    
    env.close()
    
    positions = np.array(positions)
    targets = np.array(targets)
    actions_taken = np.array(actions_taken)
    
    results = evaluate_trajectory(positions, targets, actions_taken)
    
    return positions, targets, actions_taken, results


def plot_results(positions, targets, task_name, results):
    fig = plt.figure(figsize=(15, 10))
    
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Actual')
    ax1.plot(targets[:, 0], targets[:, 1], targets[:, 2], 'r--', linewidth=1, alpha=0.5, label='Target')
    ax1.scatter(targets[0, 0], targets[0, 1], targets[0, 2], color='g', s=100, marker='o', label='Start')
    ax1.scatter(targets[-1, 0], targets[-1, 1], targets[-1, 2], color='r', s=100, marker='x', label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.legend()
    ax1.set_title(f'3D Trajectory - {task_name}')
    
    ax2 = fig.add_subplot(222)
    ax2.plot(positions[:, 0], label='X', linewidth=2)
    ax2.plot(positions[:, 1], label='Y', linewidth=2)
    ax2.plot(positions[:, 2], label='Z', linewidth=2)
    ax2.plot(targets[:, 0], 'r--', alpha=0.3, linewidth=1)
    ax2.plot(targets[:, 1], 'g--', alpha=0.3, linewidth=1)
    ax2.plot(targets[:, 2], 'b--', alpha=0.3, linewidth=1)
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Position (m)')
    ax2.legend()
    ax2.set_title('Position vs Time')
    ax2.grid(True, alpha=0.3)
    
    errors = np.linalg.norm(positions - targets, axis=1)
    ax3 = fig.add_subplot(223)
    ax3.plot(errors, linewidth=2)
    ax3.axhline(0.05, color='r', linestyle='--', alpha=0.5, label='5cm threshold')
    ax3.set_xlabel('Time step')
    ax3.set_ylabel('Position Error (m)')
    ax3.legend()
    ax3.set_title('Tracking Error')
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(224)
    metrics_text = f"RMSE: {results['rmse']:.4f} m\n"
    metrics_text += f"Max Error: {results['max_error']:.4f} m\n"
    metrics_text += f"Mean Error: {results['mean_error']:.4f} m\n"
    metrics_text += f"Final Error: {results['final_error']:.4f} m\n"
    if 'settling_time' in results:
        metrics_text += f"Settling Time: {results['settling_time']:.2f} s\n"
    if 'control_effort' in results:
        ce = results['control_effort']
        metrics_text += f"\nControl Effort:\n"
        metrics_text += f"  Mean: {ce['mean']:.2f}\n"
        metrics_text += f"  Max: {ce['max']:.2f}\n"
        metrics_text += f"  Saturation: {ce['saturation_pct']:.1f}%"
    
    ax4.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
             verticalalignment='center', transform=ax4.transAxes)
    ax4.axis('off')
    ax4.set_title('Performance Metrics')
    
    plt.tight_layout()
    
    os.makedirs('results/figures', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'results/figures/pid_{task_name}_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {filename}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Test PID Controller')
    parser.add_argument('--task', type=str, default='hover',
                       choices=['hover', 'hover_extended', 'waypoint_delivery', 'figure8', 'circle', 'emergency_landing'],
                       help='Task to test')
    parser.add_argument('--config', type=str, default='config/pid_hover_config.yaml')
    parser.add_argument('--gui', action='store_true', help='Show GUI')
    parser.add_argument('--record', action='store_true', help='Record video')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Testing PID on task: {args.task}")
    
    controller = PIDController(DroneModel.CF2X, config)
    positions, targets, actions, results = run_task(controller, args.task, args.gui, args.record)
    
    print("\n" + "="*60)
    print(f"PID CONTROLLER - {args.task.upper()} TASK RESULTS")
    print("="*60)
    print(f"RMSE: {results['rmse']:.4f} m")
    print(f"Max Error: {results['max_error']:.4f} m")
    print(f"Mean Error: {results['mean_error']:.4f} m")
    print(f"Final Error: {results['final_error']:.4f} m")
    if 'settling_time' in results:
        print(f"Settling Time: {results['settling_time']:.2f} s")
    if 'control_effort' in results:
        ce = results['control_effort']
        print(f"\nControl Effort:")
        print(f"  Mean: {ce['mean']:.2f}")
        print(f"  Max: {ce['max']:.2f}")
        print(f"  Saturation: {ce['saturation_pct']:.1f}%")
    print("="*60)
    
    os.makedirs('results/data', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    import json
    import pandas as pd
    
    json_path = f'results/data/pid_{args.task}_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump({
            'task': args.task,
            'timestamp': timestamp,
            'metrics': results,
            'config': config
        }, f, indent=2)
    print(f"Results saved: {json_path}")
    
    csv_path = f'results/data/pid_{args.task}_{timestamp}.csv'
    df = pd.DataFrame({
        'step': range(len(positions)),
        'pos_x': positions[:, 0],
        'pos_y': positions[:, 1],
        'pos_z': positions[:, 2],
        'target_x': targets[:, 0],
        'target_y': targets[:, 1],
        'target_z': targets[:, 2],
        'error': np.linalg.norm(positions - targets, axis=1),
        'action_0': actions[:, 0],
        'action_1': actions[:, 1],
        'action_2': actions[:, 2],
        'action_3': actions[:, 3]
    })
    df.to_csv(csv_path, index=False)
    print(f"Trajectory data saved: {csv_path}")
    
    plot_results(positions, targets, args.task, results)


if __name__ == '__main__':
    main()
