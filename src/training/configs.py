# PPO Hyperparameters
PPO_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,  # Small exploration bonus
    "verbose": 1,
}

# Training Config
TRAIN_CONFIG = {
    "total_timesteps": 5_000_000,  # Need longer for feedforward learning
    "num_envs": 8,  # Maximize parallel data collection
    "env_id": "BaseTrackAviary-v0",
}
