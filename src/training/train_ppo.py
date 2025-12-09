import os
import time
from datetime import datetime
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from src.envs.BaseTrackAviary import BaseTrackAviary
from src.training.configs import PPO_CONFIG, TRAIN_CONFIG
from gym_pybullet_drones.utils.enums import DroneModel, Physics

def make_env(trajectory_type='hover'):
    def _init():
        env = BaseTrackAviary(trajectory_type=trajectory_type,
                              drone_model=DroneModel.CF2X,
                              physics=Physics.PYB,
                              freq=240,
                              gui=False,
                              record=False)
        return env
    return _init

def train(trajectory_type='hover'):
    # Create log dir
    log_dir = f"logs/ppo/{trajectory_type}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create environment
    # We can use vectorized env for speed
    env = make_vec_env(make_env(trajectory_type), n_envs=TRAIN_CONFIG["num_envs"])
    
    # Initialize PPO
    model = PPO(env=env, tensorboard_log=log_dir, **PPO_CONFIG)
    # model = PPO(env=env, tensorboard_log=None, **PPO_CONFIG)
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=log_dir, name_prefix="ppo_model")
    
    # Train
    print(f"Starting training for {trajectory_type}...")
    model.learn(total_timesteps=TRAIN_CONFIG["total_timesteps"], callback=checkpoint_callback)
    
    # Save final model
    model.save(f"{log_dir}/final_model")
    print(f"Training finished. Model saved to {log_dir}")

if __name__ == "__main__":
    # Train for hover first as a test
    train(trajectory_type='hover')
