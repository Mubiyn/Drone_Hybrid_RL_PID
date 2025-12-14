#!/usr/bin/env python3
"""
Test Trained Hybrid RL Controller

Tests the trained Hybrid controller in simulation and compares performance
against baseline open-loop trajectory following.
"""

import sys
import argparse
from pathlib import Path
import pickle
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from src.envs.HybridAviary import HybridAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


def test_hybrid_model(model_path, trajectory_file, num_episodes=5, gui=True, domain_randomization=True):
    """Test trained Hybrid model"""
    
    # Load trajectory
    with open(trajectory_file, 'rb') as f:
        traj = pickle.load(f)
    trajectory_label = traj.get('trajectory_label', 'custom')
    
    print(f"\n{'='*60}")
    print("TESTING HYBRID RL CONTROLLER")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Trajectory: {trajectory_label}")
    print(f"Episodes: {num_episodes}")
    print(f"Domain Randomization: {domain_randomization}")
    print(f"{'='*60}\n")
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment
    env = HybridAviary(
        trajectory_type=trajectory_label,
        drone_model=DroneModel.CF2X,
        physics=Physics.PYB,
        freq=240,
        gui=gui,
        record=False,
        domain_randomization=domain_randomization
    )
    
    # Run episodes
    episode_rewards = []
    episode_errors = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        errors = []
        step_count = 0
        
        print(f"\nEpisode {ep + 1}/{num_episodes}")
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            
            # Track error (position error from observation)
            if 'pos_error' in info:
                errors.append(info['pos_error'])
            
            step_count += 1
            
            if step_count % 100 == 0:
                print(f"  Step {step_count}, Reward: {total_reward:.2f}")
        
        episode_rewards.append(total_reward)
        if errors:
            avg_error = np.mean(errors)
            episode_errors.append(avg_error)
            print(f"  ✓ Episode {ep + 1} complete: Reward={total_reward:.2f}, Avg Error={avg_error:.4f}m")
        else:
            print(f"  ✓ Episode {ep + 1} complete: Reward={total_reward:.2f}")
    
    env.close()
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Episodes: {num_episodes}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    if episode_errors:
        print(f"Average Position Error: {np.mean(episode_errors):.4f}m ± {np.std(episode_errors):.4f}m")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Test trained Hybrid RL model')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.zip file)')
    parser.add_argument('--trajectory-file', type=str,
                        default='data/expert_trajectories/perfect_circle_trajectory.pkl',
                        help='Path to trajectory file')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of test episodes')
    parser.add_argument('--no-gui', action='store_true',
                        help='Disable GUI visualization')
    parser.add_argument('--domain-randomization', action='store_true', default=True,
                        help='Enable domain randomization')
    parser.add_argument('--no-domain-randomization', dest='domain_randomization',
                        action='store_false',
                        help='Disable domain randomization')
    
    args = parser.parse_args()
    
    # Validate
    if not Path(args.model).exists():
        # Try adding .zip extension
        if Path(args.model + '.zip').exists():
            args.model = args.model + '.zip'
        else:
            print(f" Model not found: {args.model}")
            return
    
    if not Path(args.trajectory_file).exists():
        print(f" Trajectory file not found: {args.trajectory_file}")
        return
    
    test_hybrid_model(
        args.model,
        args.trajectory_file,
        num_episodes=args.episodes,
        gui=not args.no_gui,
        domain_randomization=args.domain_randomization
    )


if __name__ == '__main__':
    main()
