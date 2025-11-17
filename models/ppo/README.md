# PPO Trained Models

This directory stores trained PPO model weights (`.zip` files).

## Structure
```
models/ppo/
├── hover/
│   ├── ppo_hover_50000_steps.zip
│   ├── best_model.zip
│   └── ppo_hover_final.zip
├── hover_extended/
├── waypoint_delivery/
├── figure8/
├── circle/
└── emergency_landing/
```

## Usage

Load a trained model:
```python
from stable_baselines3 import PPO

model = PPO.load("models/ppo/hover/ppo_hover_final.zip")
```

## Training

See `docs/PPO_TRAINING_GUIDE.md` for full training instructions.

Quick start:
```bash
python scripts/train_ppo.py --task hover --timesteps 1000000
```
