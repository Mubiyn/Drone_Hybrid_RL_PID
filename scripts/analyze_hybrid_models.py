#!/usr/bin/env python3
"""
Analyze Trained Hybrid RL Models

Evaluates all trained models in simulation and compares performance.
"""

import sys
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from src.envs.HybridAviary import HybridAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


def evaluate_model(model_path, trajectory_file, n_episodes=10, domain_randomization=False):
    """Evaluate a trained model"""
    
    # Load trajectory
    with open(trajectory_file, 'rb') as f:
        traj = pickle.load(f)
    trajectory_label = traj.get('trajectory_label', 'unknown')
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment
    env = HybridAviary(
        trajectory_type=trajectory_label,
        drone_model=DroneModel.CF2X,
        physics=Physics.PYB,
        freq=240,
        gui=False,
        domain_randomization=domain_randomization
    )
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {trajectory_label.upper()}")
    print(f"Model: {Path(model_path).parent.name}")
    print(f"Episodes: {n_episodes}")
    print(f"Domain Randomization: {domain_randomization}")
    print(f"{'='*60}\n")
    
    episode_rewards = []
    episode_lengths = []
    tracking_errors = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        errors = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            step_count += 1
            
            # Track position error
            pos_error = np.linalg.norm(obs[0][12:15])  # pos_err from observation
            errors.append(pos_error)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        tracking_errors.append(np.mean(errors))
        
        print(f"Episode {ep+1}/{n_episodes}: Reward={episode_reward:.2f}, "
              f"Steps={step_count}, Avg Error={np.mean(errors):.4f}m")
    
    env.close()
    
    results = {
        'trajectory': trajectory_label,
        'model_path': model_path,
        'n_episodes': n_episodes,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'mean_tracking_error': np.mean(tracking_errors),
        'std_tracking_error': np.std(tracking_errors),
        'min_tracking_error': np.min(tracking_errors),
        'max_tracking_error': np.max(tracking_errors),
    }
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {trajectory_label.upper()}")
    print(f"{'='*60}")
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Episode Length: {results['mean_length']:.1f} steps")
    print(f"Tracking Error: {results['mean_tracking_error']:.4f} ± {results['std_tracking_error']:.4f}m")
    print(f"Error Range: [{results['min_tracking_error']:.4f}, {results['max_tracking_error']:.4f}]m")
    print(f"{'='*60}\n")
    
    return results


def main():
    logs_dir = Path('logs/hybrid')
    traj_dir = Path('data/expert_trajectories')
    output_dir = Path('results/hybrid_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("HYBRID RL MODEL ANALYSIS")
    print(f"{'='*70}\n")
    
    # Find all trained models
    model_dirs = sorted(logs_dir.glob('*/bc_rl_*'))
    
    if not model_dirs:
        print("✗ No trained models found!")
        return
    
    print(f"Found {len(model_dirs)} trained models\n")
    
    # Group by trajectory
    models_by_traj = {}
    for model_dir in model_dirs:
        traj_type = model_dir.parent.name
        model_file = model_dir / 'final_model.zip'
        
        if model_file.exists():
            if traj_type not in models_by_traj:
                models_by_traj[traj_type] = []
            models_by_traj[traj_type].append(model_file)
    
    # Evaluate each trajectory (use latest model if multiple)
    all_results = []
    
    for traj_type in sorted(models_by_traj.keys()):
        models = models_by_traj[traj_type]
        latest_model = sorted(models)[-1]  # Use most recent
        
        traj_file = traj_dir / f'perfect_{traj_type}_trajectory.pkl'
        
        if not traj_file.exists():
            print(f"⚠️  Skipping {traj_type}: trajectory file not found")
            continue
        
        # Evaluate without domain randomization (baseline)
        results = evaluate_model(latest_model, traj_file, n_episodes=10, domain_randomization=False)
        all_results.append(results)
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY - All Trajectories (No Domain Randomization)")
    print(f"{'='*70}\n")
    
    print(f"{'Trajectory':<12} {'Mean Reward':<15} {'Tracking Error (m)':<25} {'Episode Length':<15}")
    print(f"{'-'*70}")
    
    for r in all_results:
        print(f"{r['trajectory']:<12} {r['mean_reward']:>6.2f} ± {r['std_reward']:<5.2f} "
              f"{r['mean_tracking_error']:>6.4f} ± {r['std_tracking_error']:<6.4f}  "
              f"{r['mean_length']:>8.1f}")
    
    # Compare with autonomous baseline
    print(f"\n{'='*70}")
    print("COMPARISON WITH AUTONOMOUS BASELINE")
    print(f"{'='*70}\n")
    
    # Load autonomous analysis results
    autonomous_errors = {
        'circle': 0.3720,
        'square': 0.4012,
        'figure8': 0.0192,
        'spiral': 0.3884,
        'hover': 0.1000,
    }
    
    print(f"{'Trajectory':<12} {'Autonomous Error':<18} {'Hybrid Error':<18} {'Improvement':<15}")
    print(f"{'-'*70}")
    
    for r in all_results:
        traj = r['trajectory']
        if traj in autonomous_errors:
            auto_error = autonomous_errors[traj]
            hybrid_error = r['mean_tracking_error']
            improvement = ((auto_error - hybrid_error) / auto_error) * 100
            
            print(f"{traj:<12} {auto_error:>6.4f}m          {hybrid_error:>6.4f}m          "
                  f"{improvement:>+6.1f}%")
    
    # Save results
    output_file = output_dir / 'hybrid_evaluation_summary.txt'
    with open(output_file, 'w') as f:
        f.write("HYBRID RL MODEL EVALUATION SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        for r in all_results:
            f.write(f"{r['trajectory'].upper()}\n")
            f.write(f"  Model: {Path(r['model_path']).parent.name}\n")
            f.write(f"  Mean Reward: {r['mean_reward']:.2f} ± {r['std_reward']:.2f}\n")
            f.write(f"  Tracking Error: {r['mean_tracking_error']:.4f} ± {r['std_tracking_error']:.4f}m\n")
            f.write(f"  Episode Length: {r['mean_length']:.1f} steps\n\n")
        
        f.write("\nCOMPARISON WITH AUTONOMOUS BASELINE\n")
        f.write("-"*70 + "\n")
        for r in all_results:
            traj = r['trajectory']
            if traj in autonomous_errors:
                auto_error = autonomous_errors[traj]
                hybrid_error = r['mean_tracking_error']
                improvement = ((auto_error - hybrid_error) / auto_error) * 100
                f.write(f"{traj}: {auto_error:.4f}m → {hybrid_error:.4f}m ({improvement:+.1f}%)\n")
    
    print(f"\n✓ Results saved to: {output_file}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
