import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from src.envs.HybridAviary import HybridAviary
from src.training.configs import PPO_CONFIG, TRAIN_CONFIG
from gym_pybullet_drones.utils.enums import DroneModel, Physics

def make_hybrid_env(trajectory_type='hover'):
    def _init():
        env = HybridAviary(trajectory_type=trajectory_type,
                           drone_model=DroneModel.CF2X,
                           physics=Physics.PYB,
                           freq=240,
                           gui=False,
                           record=False)
        return env
    return _init

def train_hybrid(trajectory_type='hover'):
    # Create log dir
    log_dir = f"logs/hybrid/{trajectory_type}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create environment
    env = make_vec_env(make_hybrid_env(trajectory_type), n_envs=TRAIN_CONFIG["num_envs"])
    
    # Initialize PPO
    # We might want to lower the learning rate or entropy coefficient for residual learning
    # as the PID does most of the work.
    hybrid_config = PPO_CONFIG.copy()
    hybrid_config["learning_rate"] = 1e-4 # Slower learning
    
    model = PPO(env=env, tensorboard_log=None, **hybrid_config)
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=log_dir, name_prefix="hybrid_model")
    
    # Train
    print(f"Starting Hybrid training for {trajectory_type}...")
    model.learn(total_timesteps=TRAIN_CONFIG["total_timesteps"], callback=checkpoint_callback)
    
    # Save final model
    model.save(f"{log_dir}/final_model")
    print(f"Training finished. Model saved to {log_dir}")

if __name__ == "__main__":
    # Train for hover first
    train_hybrid(trajectory_type='hover')
