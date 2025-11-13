# Hybrid Models

This directory contains hybrid PID+RL control models.

## Hybrid Control Modes

### 1. Adaptive Mode
Dynamically adjusts weights based on tracking error:
- Low error → More RL control (learned behavior)
- High error → More PID control (safety)

**Files:** `adaptive_*.zip`

### 2. Switching Mode
Switches between PID and RL based on threshold:
- Error > threshold → PID control
- Error < threshold → RL control

**Files:** `switching_*.zip`

### 3. Weighted Mode
Fixed linear combination:
- Action = α * PID + (1-α) * RL
- α configured in config files

**Files:** `weighted_*.zip`

## Loading Hybrid Models

```python
from src.controllers.hybrid_controller import HybridController

# Load adaptive hybrid model
hybrid = HybridController.load('models/hybrid/best_model.zip')

# Check mode
print(f"Mode: {hybrid.mode}")
print(f"Weights: PID={hybrid.pid_weight:.2f}, RL={hybrid.rl_weight:.2f}")
```

## Model Details

### best_model.zip
- **Mode:** Adaptive
- **Training:** 300K timesteps
- **Performance:** 79% success on figure-eight trajectory
- **Domain Randomization:** Full (mass, inertia, wind, noise)

### Performance Comparison

| Mode | Success Rate | Tracking Error | Robustness |
|------|-------------|----------------|------------|
| Adaptive | 79% | 0.14m | ⭐⭐⭐⭐⭐ |
| Switching | 75% | 0.16m | ⭐⭐⭐⭐ |
| Weighted | 72% | 0.18m | ⭐⭐⭐ |

## Download

Large model files available at:
[Google Drive link to be added]
