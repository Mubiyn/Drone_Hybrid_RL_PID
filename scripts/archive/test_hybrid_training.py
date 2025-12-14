from src.training.train_hybrid import train_hybrid
from src.training.configs import TRAIN_CONFIG

# Override config for testing
TRAIN_CONFIG["total_timesteps"] = 2048

if __name__ == "__main__":
    print("Testing Hybrid training pipeline...")
    train_hybrid(trajectory_type='hover')
