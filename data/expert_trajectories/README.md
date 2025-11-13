# Expert Demonstrations

This directory contains expert trajectory data for behavior cloning or curriculum learning.

## Data Format

Demonstrations are stored as NumPy arrays (`.npy`) or CSV files with the following structure:

### NumPy Format
```python
import numpy as np

# Load demonstrations
data = np.load('data/expert_trajectories/hover_demos.npy', allow_pickle=True)

# Data structure
demos = {
    'observations': [],  # List of observation arrays
    'actions': [],       # List of action arrays
    'rewards': [],       # List of reward arrays
    'dones': [],        # List of done flags
    'infos': []         # List of info dicts
}
```

### CSV Format
Columns: `timestamp, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, roll, pitch, yaw, rpm_1, rpm_2, rpm_3, rpm_4`

## Collecting Demonstrations

### Using PID Controller
```bash
# Collect trajectories using tuned PID
python scripts/collect_demos.py \
    --controller pid \
    --task hover \
    --num-episodes 50 \
    --output data/expert_trajectories/hover_demos.npy
```

### Manual Collection (Real Drone)
```bash
# Record human flight data from Tello
python scripts/record_flight.py \
    --duration 60 \
    --output data/expert_trajectories/manual_flight.npy
```

## Using Demonstrations

### Behavior Cloning
```python
from src.training.bc_trainer import BehaviorCloning

# Load demos
demos = np.load('data/expert_trajectories/hover_demos.npy', allow_pickle=True)

# Train BC policy
bc = BehaviorCloning()
bc.train(demos, epochs=100)
```

### Curriculum Learning
Use demonstrations to initialize RL training or as expert baselines.

## Available Demonstrations

| File | Task | Episodes | Quality | Size |
|------|------|----------|---------|------|
| hover_demos.npy | Hover | 50 | ⭐⭐⭐⭐⭐ | 1.2 MB |
| waypoint_demos.npy | Waypoint | 30 | ⭐⭐⭐⭐ | 2.5 MB |
| trajectory_demos.npy | Trajectory | 20 | ⭐⭐⭐⭐ | 3.8 MB |

## Data Collection Guidelines

1. **Quality over quantity:** Focus on successful demonstrations
2. **Diversity:** Include various initial conditions and disturbances
3. **Consistency:** Use the same observation/action spaces as RL training
4. **Validation:** Verify demonstrations achieve desired performance

## Download

If demonstrations are not included, download from:
[Google Drive link to be added]
