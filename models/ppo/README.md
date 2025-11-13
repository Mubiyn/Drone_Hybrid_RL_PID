# Trained Models

This directory contains trained models for the drone control system.

## Directory Structure

```
models/
├── ppo/                      # Pure PPO models
│   ├── best_hover.zip        # Best hovering policy
│   ├── best_waypoint.zip     # Best waypoint navigation policy
│   └── checkpoints/          # Training checkpoints
├── hybrid/                   # Hybrid PID+RL models
│   ├── best_model.zip        # Best hybrid model
│   ├── adaptive_*.zip        # Adaptive weighting variants
│   ├── switching_*.zip       # Switching variants
│   └── weighted_*.zip        # Fixed weighting variants
└── README.md                 # This file
```

## Model Format

All models are saved using Stable-Baselines3's `.zip` format, which includes:
- Policy network weights
- Value network weights
- Optimizer state
- Hyperparameters
- Normalization statistics

## Loading Models

### Load PPO Model

```python
from stable_baselines3 import PPO

# Load trained model
model = PPO.load('models/ppo/best_waypoint.zip')

# Use for inference
obs = env.reset()
action, _ = model.predict(obs, deterministic=True)
```

### Load Hybrid Model

```python
from src.controllers.hybrid_controller import HybridController

# Load hybrid controller
hybrid = HybridController.load('models/hybrid/best_model.zip')

# Use for control
obs = env.reset()
action = hybrid.get_action(obs)
```

## Model Download

If models are too large to include in the repository, download them from:

**Google Drive:** [Link to be added]

```bash
# Download using gdown
pip install gdown
gdown --id <file_id> -O models/ppo/best_waypoint.zip
```

## Training Your Own Models

See the main README.md for training instructions, or run:

```bash
# Train PID
python scripts/train_pid.py

# Train RL
python scripts/train_rl.py --timesteps 500000

# Train Hybrid
python scripts/train_hybrid.py --timesteps 300000
```

## Model Performance

| Model | Task | Success Rate | Avg Return | File Size |
|-------|------|--------------|------------|-----------|
| best_hover.zip | Hover | 97% | 465 | 2.3 MB |
| best_waypoint.zip | Waypoint | 88% | 412 | 2.3 MB |
| best_model.zip | Trajectory | 79% | 368 | 4.8 MB |

## Checkpoints

Training checkpoints are saved every 10,000 steps in `checkpoints/` subdirectories. These can be used to resume training or evaluate intermediate performance.

```bash
# Resume training from checkpoint
python scripts/train_rl.py --resume models/ppo/checkpoints/rl_model_100000_steps.zip
```
