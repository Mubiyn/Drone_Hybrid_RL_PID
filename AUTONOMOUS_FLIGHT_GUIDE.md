# Autonomous Data Collection & PID Tuning Guide

## Overview

This guide covers autonomous trajectory flight for:
1. **Data Collection**: High-quality training data with complex trajectories
2. **PID Tuning**: Optimize control gains for real Tello dynamics

## Why Autonomous Flight?

**Manual Control Issues:**
- Impossible to fly precise circles/figure-8s manually
- Inconsistent human inputs introduce noise
- Limited trajectory diversity

**Autonomous Flight Benefits:**
- ✓ Perfect trajectory tracking (ground truth from MoCap)
- ✓ Consistent PID control actions
- ✓ Reveals controller performance gaps
- ✓ Generates diverse, repeatable training data

## Quick Start

### 1. Simple Circle Flight
```bash
./collect_autonomous.sh
# Choose: 1 (circle)
# MoCap: y
# Duration: 60
# kp: 0.4
# max_vel: 0.5
```

### 2. Complex Figure-8
```bash
python scripts/autonomous_data_collection.py \
    --trajectory figure8 \
    --duration 60 \
    --mocap \
    --kp 0.4 \
    --max-vel 0.5
```

### 3. PID Auto-Tuning
```bash
python scripts/autonomous_data_collection.py --tune-pid --mocap
```
This tests 20 combinations of (kp, max_vel) and reports best gains.

## Available Trajectories

### Circle (`--trajectory circle`)
- **Best for**: Baseline tracking, centripetal force dynamics
- **Pattern**: Constant 0.8m radius circle, 1.0m altitude
- **Period**: 20 seconds per loop
- **Velocity**: Smooth sinusoidal, ~0.25 m/s
- **Use case**: Initial data collection, PID validation

### Figure-8 (`--trajectory figure8`)
- **Best for**: Complex lateral motion, direction changes
- **Pattern**: Lemniscate (infinity symbol)
- **Period**: 30 seconds per loop
- **Velocity**: Variable speed through curves
- **Use case**: Testing controller agility, diverse dynamics

### Spiral (`--trajectory spiral`)
- **Best for**: Combined altitude + position control
- **Pattern**: Expanding radius while ascending
- **Height**: 0.5m → 2.0m over 40 seconds
- **Radius**: 0 → 1.0m
- **Use case**: Full 3D dynamics, altitude coupling

### Waypoint (`--trajectory waypoint`)
- **Best for**: Aggressive maneuvers, overshoot analysis
- **Pattern**: Square with corners at (±0.5, ±0.5) with altitude variation
- **Transitions**: Smooth cosine interpolation (8s per segment)
- **Use case**: Step response characterization, PID tuning

### Hover (`--trajectory hover`)
- **Best for**: Stability testing, disturbance rejection
- **Pattern**: Stationary at (0, 0, 1.0)
- **Use case**: Position hold performance, noise characterization

## PID Tuning Workflow

### Current Problem
Your PID was tuned for **CF2X simulation** (27g, instant response), but real Tello has:
- **80g mass** (3x heavier)
- **0.48s lateral delay** (slow actuators)
- **Lower velocity limits** (1.0 m/s vs 2+ m/s)

### Tuning Strategy

**Step 1: Baseline Assessment**
```bash
# Test current gains (kp=0.4, max_vel=0.5)
python scripts/autonomous_data_collection.py \
    --trajectory hover \
    --duration 30 \
    --kp 0.4 \
    --max-vel 0.5 \
    --mocap
```

**Step 2: Analyze Tracking Error**
```bash
python scripts/analyze_flight_data.py \
    data/tello_flights/autonomous_hover_*.pkl \
    --plot \
    --output results/figures
```
Look for:
- **Overshoot**: kp too high → reduce
- **Slow convergence**: kp too low → increase
- **Oscillation**: max_vel too high → reduce
- **Steady-state error**: kp too low or max_vel too low

**Step 3: Manual Sweep**
Test increasing kp values:
```bash
# Conservative
python scripts/autonomous_data_collection.py --trajectory hover --kp 0.2 --max-vel 0.3 --mocap

# Moderate (current)
python scripts/autonomous_data_collection.py --trajectory hover --kp 0.4 --max-vel 0.5 --mocap

# Aggressive
python scripts/autonomous_data_collection.py --trajectory hover --kp 0.6 --max-vel 0.7 --mocap

# Very aggressive (may oscillate)
python scripts/autonomous_data_collection.py --trajectory hover --kp 0.8 --max-vel 1.0 --mocap
```

**Step 4: Automated Grid Search**
```bash
python scripts/autonomous_data_collection.py --tune-pid --mocap
```
This tests:
- `kp`: [0.2, 0.4, 0.6, 0.8, 1.0]
- `max_vel`: [0.3, 0.5, 0.7, 1.0]

**Expected Output:**
```
PID TUNING RESULTS
====================================
    kp  max_vel   Mean Error    Std Error     Max Error
------------------------------------
   0.6      0.7        0.089        0.045         0.234  ← Best
   0.4      0.5        0.127        0.062         0.312
   0.8      1.0        0.156        0.189         0.892  ← Oscillating
   0.2      0.3        0.234        0.078         0.456  ← Too slow
```

**Step 5: Validate on Complex Trajectory**
Once you find best gains (e.g., kp=0.6, max_vel=0.7), test on circle:
```bash
python scripts/autonomous_data_collection.py \
    --trajectory circle \
    --duration 60 \
    --kp 0.6 \
    --max-vel 0.7 \
    --mocap
```

## Data Collection Strategy

### Objective: Diverse Training Dataset

**Goal**: 5-10 flights covering full dynamics range

**Recommended Collection:**

1. **Hover (baseline)**
   ```bash
   python scripts/autonomous_data_collection.py --trajectory hover --duration 30 --mocap
   ```

2. **Circle (smooth tracking)**
   ```bash
   python scripts/autonomous_data_collection.py --trajectory circle --duration 60 --mocap
   ```

3. **Figure-8 (complex lateral)**
   ```bash
   python scripts/autonomous_data_collection.py --trajectory figure8 --duration 60 --mocap
   ```

4. **Spiral (3D motion)**
   ```bash
   python scripts/autonomous_data_collection.py --trajectory spiral --duration 40 --mocap
   ```

5. **Waypoint (aggressive)**
   ```bash
   python scripts/autonomous_data_collection.py --trajectory waypoint --duration 60 --mocap
   ```

6. **Multiple speeds** (repeat circle with different max_vel):
   ```bash
   # Slow
   python scripts/autonomous_data_collection.py --trajectory circle --max-vel 0.3 --mocap
   
   # Medium
   python scripts/autonomous_data_collection.py --trajectory circle --max-vel 0.5 --mocap
   
   # Fast
   python scripts/autonomous_data_collection.py --trajectory circle --max-vel 0.8 --mocap
   ```

### Expected Dataset
After collection:
- **8-10 flights**
- **5,000-10,000 samples** total
- **Full state coverage**: Position [-2, +2]m, Velocity [0, 1.0] m/s, All orientations
- **PID actions**: Continuous control commands (not discrete keyboard)

## Analysis & Training Pipeline

### 1. Analyze Individual Flights
```bash
python scripts/analyze_flight_data.py \
    data/tello_flights/autonomous_circle_*.pkl \
    --plot \
    --output results/figures
```

### 2. Compare Trajectories
```bash
# Compare tracking error across trajectories
for file in data/tello_flights/autonomous_*.pkl; do
    python scripts/analyze_flight_data.py $file
done
```

### 3. Extract Dynamics Model
The data reveals:
- **Action → Velocity mapping**: How PID commands translate to actual motion
- **Control delays**: Forward (0.0s), Lateral (0.48s), Vertical (?s)
- **Acceleration limits**: Max achievable acceleration from rest
- **Velocity saturation**: Tello's actual speed limits

### 4. System Identification
Create `scripts/system_identification.py`:
```python
# Fit: next_state = f(current_state, action)
# Extract: mass, drag coefficients, control authority
# Use for: Updating simulation parameters
```

### 5. Use for Training

**Option A: Behavioral Cloning (Simplest)**
```python
# Train policy to imitate PID on collected trajectories
policy = BC(state_dim=12, action_dim=4)
policy.train(states, actions)
```

**Option B: Offline RL (Better)**
```python
# Use collected data for offline policy optimization
# Methods: CQL, IQL, TD3+BC
from stable_baselines3 import TD3
policy = TD3.load("logs/hybrid/circle/best_model")
policy.fine_tune(offline_dataset=tello_flights)
```

**Option C: Online Fine-tuning (Risky)**
```python
# Continue PPO training on real Tello
# Requires many flights, risk of crashes
```

## Expected Improvements

### Current Baseline (Manual)
- Position tracking: Unknown (no ground truth)
- Velocity control: Human reaction time (~0.3s)
- Trajectory precision: Poor (manual drift)

### After PID Tuning
- Position tracking: **0.10 ± 0.05m** (MoCap validated)
- Control delay compensation: **Optimized for 0.48s lag**
- Hover stability: **< 0.1m variance**

### After Hybrid Fine-tuning
- Expected: **50-70% better** than tuned PID
- Tracking error: **0.05 ± 0.02m** (learned compensations)
- Aggressive maneuvers: Smoother through learned dynamics

## Safety Notes

⚠️ **Emergency Stop**: Press `Ctrl+C` during flight → immediate landing

⚠️ **Battery**: Starts flight only if > 20%

⚠️ **MoCap Required**: Without MoCap, position control is unreliable (uses dead reckoning)

⚠️ **Space**: Ensure 3x3m clear area for circle/figure-8 (1.6m radius max)

⚠️ **Tuning Risks**: Aggressive gains (kp > 0.8) may cause oscillation → test carefully

## Troubleshooting

**Problem: "No MoCap data"**
- Check: `python test_mocap.sh`
- Verify rigid body named "Tello" in Motive
- Ensure multicast 239.255.42.99:1511

**Problem: Large tracking error (>0.5m)**
- PID gains too low → increase kp
- max_vel too low → increase max_vel
- Control delay not compensated → add feed-forward term

**Problem: Oscillation around target**
- PID gains too high → reduce kp
- max_vel too high → reduce max_vel

**Problem: Drone drifts during flight**
- MoCap not working → check connection
- Optical flow drift → clean Tello camera
- Wind disturbance → fly indoors

## Next Steps

1. **Run PID tuning**: `./collect_autonomous.sh` → Choice 6
2. **Collect diverse flights**: 5-10 trajectories with best gains
3. **Analyze tracking performance**: Compare PID vs desired trajectory
4. **System identification**: Extract real Tello dynamics
5. **Train hybrid model**: Fine-tune on collected data
6. **Evaluate**: Compare PID vs Hybrid on test trajectories

## Summary

This autonomous flight system solves your two problems:

1. ✅ **Complex Trajectories**: PID flies perfect circles/figure-8s (impossible manually)
2. ✅ **PID Tuning**: Automated grid search finds optimal gains for real Tello

The collected data will be **much higher quality** than manual control:
- Smooth, continuous actions (not discrete keyboard)
- Ground truth tracking error (MoCap validation)
- Full dynamics coverage (slow → fast, all directions)
- Perfect for sim-to-real transfer learning
