#!/usr/bin/env python3
import os
import sys
import yaml
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary

from src.controllers.pid_controller import PIDController
from src.utils.trajectories import get_trajectory
from src.utils.metrics import evaluate_trajectory
from src.testing.test_scenarios import OOD_SCENARIOS


def run_ood_test(controller, task_name, scenario_name, scenario_params, gui=True):
    import pybullet as p
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
        record=False,
        pyb_freq=240,
        ctrl_freq=48
    )
    
    obs, info = env.reset()
    
    new_mass = env.M * scenario_params['mass_multiplier']
    p.changeDynamics(env.DRONE_IDS[0], -1, mass=new_mass, physicsClientId=env.CLIENT)
    env.M = new_mass
    env.GRAVITY = env.G * env.M
    
    controller.reset()
    
    # No warmup - measure full trajectory including takeoff transient
    # This matches RL training where agent learns from t=0
    positions = []
    motor_efficiency = np.array(scenario_params['motor_efficiency'])
    wind_speed = scenario_params['wind_speed']
    
    for step in range(len(trajectory)):
        target_pos = trajectory[step]
        
        action = controller.compute_control(obs[0], target_pos)
        
        action_modified = action * motor_efficiency
        
        if wind_speed > 0:
            wind_force = np.array([wind_speed, 0, 0])
            p.applyExternalForce(env.DRONE_IDS[0], -1, wind_force, [0, 0, 0], 
                                p.WORLD_FRAME, physicsClientId=env.CLIENT)
        
        obs, reward, terminated, truncated, info = env.step(np.array([action_modified]))
        
        positions.append(obs[0][0:3].copy())
        
        if terminated or truncated:
            break
    
    env.close()
    
    positions = np.array(positions)
    targets = np.array(trajectory[:len(positions)])
    
    results = evaluate_trajectory(positions, targets)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test PID under OOD conditions')
    parser.add_argument('--task', type=str, default='hover',
                       choices=['hover', 'hover_extended', 'waypoint_delivery', 'figure8', 'circle', 'emergency_landing'])
    parser.add_argument('--config', type=str, default='config/pid_hover_config.yaml')
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--scenario', type=str, default='all',
                       help='Scenario name or "all"')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    controller = PIDController(DroneModel.CF2X, config)
    
    scenarios = [args.scenario] if args.scenario != 'all' else list(OOD_SCENARIOS.keys())
    
    all_results = {}
    
    print(f"\nTesting PID on {args.task} task")
    print("="*70)
    
    for scenario_name in scenarios:
        scenario = OOD_SCENARIOS[scenario_name]
        print(f"\nScenario: {scenario_name}")
        print(f"  {scenario['description']}")
        
        trial_results = []
        for trial in range(args.trials):
            results = run_ood_test(controller, args.task, scenario_name, scenario, gui=False)
            trial_results.append(results)
            print(f"  Trial {trial+1}/{args.trials}: RMSE={results['rmse']:.4f}m")
        
        rmse_values = [r['rmse'] for r in trial_results]
        max_errors = [r['max_error'] for r in trial_results]
        
        all_results[scenario_name] = {
            'rmse_mean': np.mean(rmse_values),
            'rmse_std': np.std(rmse_values),
            'max_error_mean': np.mean(max_errors),
            'max_error_std': np.std(max_errors),
            'trials': trial_results
        }
        
        print(f"  Average RMSE: {all_results[scenario_name]['rmse_mean']:.4f} ± {all_results[scenario_name]['rmse_std']:.4f}m")
    
    os.makedirs('results/data', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    json_path = f'results/data/pid_ood_{args.task}_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump({
            'task': args.task,
            'timestamp': timestamp,
            'trials': args.trials,
            'scenarios': all_results
        }, f, indent=2)
    print(f"\nResults saved: {json_path}")
    
    import pandas as pd
    csv_data = []
    for scenario_name, scenario_data in all_results.items():
        csv_data.append({
            'scenario': scenario_name,
            'rmse_mean': scenario_data['rmse_mean'],
            'rmse_std': scenario_data['rmse_std'],
            'max_error_mean': scenario_data['max_error_mean'],
            'max_error_std': scenario_data['max_error_std'],
            'trials': args.trials
        })
    
    df = pd.DataFrame(csv_data)
    csv_path = f'results/data/pid_ood_{args.task}_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV saved: {csv_path}")
    
    plot_comparison(all_results, args.task, timestamp)


def plot_comparison(results, task_name, timestamp):
    scenarios = list(results.keys())
    rmse_means = [results[s]['rmse_mean'] for s in scenarios]
    rmse_stds = [results[s]['rmse_std'] for s in scenarios]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(scenarios))
    bars = ax.bar(x, rmse_means, yerr=rmse_stds, capsize=5, alpha=0.7, color='steelblue')
    
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('RMSE (m)', fontsize=12)
    ax.set_title(f'PID Controller Performance - {task_name} Task', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, val, std) in enumerate(zip(bars, rmse_means, rmse_stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    filename = f'results/figures/pid_ood_comparison_{task_name}_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {filename}")
    plt.show()


if __name__ == '__main__':
    main()
