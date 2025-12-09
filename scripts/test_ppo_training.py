from src.training.train_ppo import train
from src.training.configs import TRAIN_CONFIG

# Override config for testing
TRAIN_CONFIG["total_timesteps"] = 2048

if __name__ == "__main__":
    print("Testing PPO training pipeline...")
    train(trajectory_type='hover')
