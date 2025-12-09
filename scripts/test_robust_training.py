from src.training.train_robust import train_robust_hybrid
from src.training.configs import TRAIN_CONFIG

# Override config for testing
TRAIN_CONFIG["total_timesteps"] = 2048

if __name__ == "__main__":
    print("Testing Robust Hybrid training pipeline...")
    train_robust_hybrid(trajectory_type='hover')
