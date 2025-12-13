#!/usr/bin/env python3
"""
Train Hybrid RL Controller (RL Only - No BC)

Trains directly with PPO + domain randomization.
The PID baseline provides initial behavior, RL learns residual corrections.
"""

import os
import sys
import argparse
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from gym_pybullet_drones.utils.enums import DroneModel, Physics

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def make_hybrid_env(trajectory_file, domain_randomization=True, gui=False):
    """Create environment factory"""
    def _init():
        from src.envs.HybridAviary import HybridAviary
        
        # Load trajectory
        with open(trajectory_file, 'rb') as f:
            traj = pickle.load(f)
        trajectory_label = traj.get('trajectory_label', 'circle')
        
        # HybridAviary now uses trajectory-specific tuned PID gains automatically
        env = HybridAviary(
            trajectory_type=trajectory_label,
            drone_model=DroneModel.CF2X,
            physics=Physics.PYB,
            freq=240,
            gui=gui,
            record=False,
            domain_randomization=domain_randomization
        )
        return env
    return _init


def train_with_rl(trajectory_file, args):
    """Train with RL from scratch (PID baseline provides initial behavior)"""
    
    print(f"\n{'='*60}")
    print("HYBRID RL TRAINING (RL ONLY)")
    print(f"{'='*60}")
    print(f"Trajectory: {trajectory_file}")
    print(f"Domain Randomization: {args.domain_randomization}")
    print(f"Total Steps: {args.train_steps:,}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"{'='*60}\n")
    
    # Load trajectory
    with open(trajectory_file, 'rb') as f:
        traj = pickle.load(f)
    trajectory_label = traj.get('trajectory_label', 'circle')
    
    # Create log directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"logs/hybrid/{trajectory_label}/rl_only_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create vectorized environment
    # Use DummyVecEnv on macOS to avoid multiprocessing issues
    if args.num_envs > 1:
        print(f"Using DummyVecEnv (sequential) for {args.num_envs} environments")
        print("(SubprocVecEnv has issues on macOS with numpy multiprocessing)")
        env = make_vec_env(
            make_hybrid_env(trajectory_file, args.domain_randomization, False),
            n_envs=args.num_envs,
            vec_env_cls=DummyVecEnv
        )
    else:
        env = make_vec_env(
            make_hybrid_env(trajectory_file, args.domain_randomization, args.gui),
            n_envs=1
        )
    
    # Create evaluation environment (no domain randomization for eval)
    eval_env = make_vec_env(
        make_hybrid_env(trajectory_file, domain_randomization=False, gui=False),
        n_envs=1
    )
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Create PPO model (train from scratch, PID provides baseline behavior)
    print("\nInitializing PPO model...")
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=args.learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=f"{log_dir}/tb"
    )
    
    # Train
    print(f"\nTraining PPO for {args.train_steps:,} steps...")
    print("The PID baseline provides initial behavior, RL learns residual corrections.\n")
    
    model.learn(
        total_timesteps=args.train_steps,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save final model
    final_model_path = f"{log_dir}/final_model"
    model.save(final_model_path)
    print(f"\n✓ Model saved to: {final_model_path}.zip")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Train Hybrid RL Controller (RL Only - No BC)'
    )
    
    parser.add_argument('--trajectory', type=str, default='circle',
                        choices=['circle', 'square', 'figure8', 'spiral', 'hover'],
                        help='Trajectory type to train')
    parser.add_argument('--train-steps', type=int, default=500000,
                        help='RL training timesteps')
    parser.add_argument('--num-envs', type=int, default=4,
                        help='Number of parallel environments for RL')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='PPO learning rate')
    parser.add_argument('--domain-randomization', action='store_true', default=True)
    parser.add_argument('--no-domain-randomization', dest='domain_randomization',
                        action='store_false')
    parser.add_argument('--gui', action='store_true',
                        help='Show GUI (only with --num-envs 1)')
    
    args = parser.parse_args()
    
    # Validate
    if args.gui and args.num_envs > 1:
        print("⚠️  GUI requires --num-envs 1")
        args.num_envs = 1
    
    # Build trajectory file path
    trajectory_file = f"data/expert_trajectories/perfect_{args.trajectory}_trajectory.pkl"
    
    if not os.path.exists(trajectory_file):
        print(f"❌ Trajectory file not found: {trajectory_file}")
        return
    
    # Train with RL
    model = train_with_rl(trajectory_file, args)
    
    print("\n✅ TRAINING COMPLETE!")
    print("   The Hybrid controller was trained purely with RL.")
    print("   The PID baseline provides initial behavior,")
    print("   RL learned residual corrections to improve tracking.\n")


if __name__ == '__main__':
    main()
