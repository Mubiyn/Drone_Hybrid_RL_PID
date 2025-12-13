#!/usr/bin/env python3
"""
Train Hybrid RL Controller with Perfect Trajectories

Uses the perfect trajectories generated from autonomous flights to train
a Hybrid (PID + RL) controller in simulation with domain randomization.

The goal: Train a controller that can outperform open-loop trajectory following
when faced with domain shifts (added mass, drag, wind, etc.)
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import pickle
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType


def load_trajectory(trajectory_file):
    """Load trajectory from pkl file"""
    with open(trajectory_file, 'rb') as f:
        traj = pickle.load(f)
    return traj


def make_hybrid_env(trajectory_file, domain_randomization=True, gui=False):
    """Create environment factory for parallel training"""
    def _init():
        # Import here to avoid issues with multiprocessing
        from src.envs.HybridAviary import HybridAviary
        
        # Load trajectory
        traj = load_trajectory(trajectory_file)
        trajectory_label = traj.get('trajectory_label', 'custom')
        
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


def train_hybrid(args):
    """Train Hybrid RL controller"""
    
    # Load trajectory to get label
    traj = load_trajectory(args.trajectory_file)
    trajectory_label = traj.get('trajectory_label', 'custom')
    
    print(f"\n{'='*60}")
    print("HYBRID RL TRAINING")
    print(f"{'='*60}")
    print(f"Trajectory: {trajectory_label}")
    print(f"File: {args.trajectory_file}")
    print(f"Waypoints: {traj['num_waypoints']}")
    print(f"Duration: {traj['duration']:.1f}s")
    print(f"Domain Randomization: {args.domain_randomization}")
    print(f"Parallel Envs: {args.num_envs}")
    print(f"Total Steps: {args.total_steps:,}")
    print(f"{'='*60}\n")
    
    # Create log directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"logs/hybrid/{trajectory_label}/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create vectorized training environment
    if args.num_envs > 1:
        env = make_vec_env(
            make_hybrid_env(args.trajectory_file, args.domain_randomization, gui=False),
            n_envs=args.num_envs,
            vec_env_cls=SubprocVecEnv
        )
    else:
        env = make_vec_env(
            make_hybrid_env(args.trajectory_file, args.domain_randomization, gui=args.gui),
            n_envs=1
        )
    
    # Create evaluation environment (single env, no GUI)
    eval_env = make_vec_env(
        make_hybrid_env(args.trajectory_file, args.domain_randomization, gui=False),
        n_envs=1
    )
    
    # Configure PPO for residual learning
    # Lower learning rate since PID does most of the work
    model_config = {
        'policy': 'MlpPolicy',
        'learning_rate': args.learning_rate,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,  # Encourage exploration
        'verbose': 1,
        'tensorboard_log': log_dir
    }
    
    print("Creating PPO model...")
    model = PPO(env=env, **model_config)
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // args.num_envs,  # Adjust for parallel envs
        save_path=log_dir,
        name_prefix="hybrid_model"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=10000 // args.num_envs,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    # Train
    print(f"\n🚀 Starting training...")
    print(f"   Monitor progress: tensorboard --logdir {log_dir}\n")
    
    try:
        model.learn(
            total_timesteps=args.total_steps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    
    # Save final model
    final_model_path = f"{log_dir}/final_model"
    model.save(final_model_path)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Model saved: {final_model_path}.zip")
    print(f"Logs: {log_dir}")
    print(f"\nTo test the model:")
    print(f"  python scripts/test_hybrid.py --model {final_model_path}.zip")
    print(f"{'='*60}\n")
    
    env.close()
    eval_env.close()


def main():
    parser = argparse.ArgumentParser(description='Train Hybrid RL controller with trajectory following')
    
    parser.add_argument('--trajectory-file', type=str, 
                        default='data/expert_trajectories/perfect_circle_trajectory.pkl',
                        help='Path to trajectory .pkl file')
    parser.add_argument('--total-steps', type=int, default=1_000_000,
                        help='Total training timesteps')
    parser.add_argument('--num-envs', type=int, default=4,
                        help='Number of parallel environments')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='PPO learning rate (lower for residual learning)')
    parser.add_argument('--domain-randomization', action='store_true', default=True,
                        help='Enable domain randomization (mass, inertia, wind)')
    parser.add_argument('--no-domain-randomization', dest='domain_randomization', 
                        action='store_false',
                        help='Disable domain randomization')
    parser.add_argument('--gui', action='store_true',
                        help='Show GUI (only works with --num-envs 1)')
    
    args = parser.parse_args()
    
    # Validate
    if args.gui and args.num_envs > 1:
        print("⚠️  GUI only works with --num-envs 1, setting num_envs=1")
        args.num_envs = 1
    
    if not Path(args.trajectory_file).exists():
        print(f"❌ Trajectory file not found: {args.trajectory_file}")
        return
    
    train_hybrid(args)


if __name__ == '__main__':
    main()
