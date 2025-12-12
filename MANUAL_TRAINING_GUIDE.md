# Manual Data Collection → Training → Testing Workflow

## Overview
Train PID and Hybrid controllers to autonomously imitate your manual flight data.

## Workflow Steps

### ✅ Step 1: Manual Data Collection (COMPLETED)
You've already collected 4 flight recordings:
- `flight_20251212_191949_hover.pkl` (1529 samples, 82s)
- `flight_20251212_193328_circle.pkl` (1366 samples, 73s)
- `flight_20251212_193406_circle.pkl` (218 samples, 12s)
- `flight_20251212_194341_figure8.pkl` (975 samples, 52s - actually circle)

**Total:** 4088 samples, 3.7 minutes of flight data

---

### 📊 Step 2A: Train PID from Manual Trajectory

Extract trajectory waypoints from your manual flight and create a learned trajectory file.

**Usage:**
```bash
# Extract trajectory from hover flight
python scripts/train_pid_from_manual.py \
  --data data/tello_flights/flight_20251212_191949_hover.pkl \
  --output data/expert_trajectories \
  --visualize

# Extract from circle flight
python scripts/train_pid_from_manual.py \
  --data data/tello_flights/flight_20251212_193328_circle.pkl \
  --output data/expert_trajectories \
  --smooth-window 5 \
  --waypoint-spacing 0.1 \
  --visualize
```

**Output:**
- `data/expert_trajectories/learned_hover_trajectory.pkl` - Trajectory waypoints with spline
- `data/expert_trajectories/learned_hover_trajectory.json` - Metadata
- `data/expert_trajectories/learned_hover_trajectory_plot.png` - Visualization

**Parameters:**
- `--smooth-window`: Smoothing window size (default: 5 samples)
- `--waypoint-spacing`: Min distance between waypoints in meters (default: 0.1)
- `--visualize`: Show 3D trajectory plot

---

### 🧠 Step 2B: Train Hybrid Model (Behavioral Cloning)

Train a neural network to directly imitate your manual control actions.

**Usage:**
```bash
# Train on all collected flights
python scripts/train_hybrid_from_manual.py \
  --data-dir data/tello_flights \
  --output-dir models/manual_bc \
  --epochs 100 \
  --batch-size 64 \
  --lr 1e-3
```

**Output:**
- `models/manual_bc/best_model.pth` - Trained neural network
- `models/manual_bc/metadata.json` - Training info
- `models/manual_bc/training_curves.png` - Loss plot

**Parameters:**
- `--epochs`: Training epochs (default: 100)
- `--batch-size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 1e-3)
- `--hidden-sizes`: Network architecture (default: [256, 256])

---

### 🚁 Step 3A: Test PID Following Learned Trajectory

**PID follows waypoints extracted from your manual flight**

```bash
# Test hover trajectory
python src/real_drone/run_tello.py \
  --controller pid \
  --trajectory-file data/expert_trajectories/learned_hover_trajectory.pkl \
  --duration 30

# Test circle trajectory
python src/real_drone/run_tello.py \
  --controller pid \
  --trajectory-file data/expert_trajectories/learned_circle_trajectory.pkl \
  --duration 30
```

**Expected behavior:**
- Drone autonomously flies the same path you flew manually
- PID controller tracks waypoints smoothly
- Should follow trajectory with <20cm error

---

### 🤖 Step 3B: Test Hybrid (Behavioral Cloning)

**Neural network directly imitates your manual actions**

```bash
# Test BC model
python src/real_drone/run_tello.py \
  --controller manual_bc \
  --model models/manual_bc/best_model.pth \
  --duration 30
```

**Expected behavior:**
- Network outputs actions similar to your manual commands
- No explicit trajectory - learns patterns from your flying style
- Should fly smoothly if trained well

---

## Quick Start Commands

```bash
# 1. Extract trajectory from best circle flight
python scripts/train_pid_from_manual.py \
  --data data/tello_flights/flight_20251212_193328_circle.pkl \
  --visualize

# 2. Train BC model on all flights
python scripts/train_hybrid_from_manual.py

# 3. Test PID with learned trajectory
python src/real_drone/run_tello.py \
  --controller pid \
  --trajectory-file data/expert_trajectories/learned_circle_trajectory.pkl \
  --duration 30

# 4. Test BC model
python src/real_drone/run_tello.py \
  --controller manual_bc \
  --model models/manual_bc/best_model.pth \
  --duration 30
```

---

## Comparison: PID vs Hybrid

| Controller | How it works | Pros | Cons |
|------------|--------------|------|------|
| **PID** | Follows extracted trajectory waypoints | Predictable, smooth, interpretable | Requires explicit trajectory |
| **Manual BC** | Imitates your actions directly | Learns your flying style, end-to-end | Black box, may overfit |

---

## Troubleshooting

### Training Issues

**Low validation loss but poor flight:**
- Try collecting more diverse data (more trajectories)
- Increase smoothing window for trajectory extraction
- Reduce learning rate for BC training

**BC model outputs erratic actions:**
- Check data quality (no sudden jumps in actions)
- Increase training epochs
- Add data augmentation (small noise to states)

### Flight Issues

**PID doesn't follow trajectory well:**
- Trajectory might be too aggressive (tight curves)
- Increase `--waypoint-spacing` for smoother path
- Tune PID gains in `VelocityPIDController`

**BC model flies unstably:**
- Model may have overfit to training data
- Try retraining with more regularization
- Collect more varied training flights

---

## Data Requirements

**Minimum for training:**
- ~1000 samples (1 minute at 18 Hz)
- At least 1 trajectory type

**Recommended:**
- 3000+ samples (3+ minutes)
- Multiple trajectory types (hover, circle, forward/back)
- Diverse maneuvers (turns, altitude changes)

**Your data:**
- ✅ 4088 samples - GOOD
- ✅ 3 trajectory types - GOOD
- ✅ Multiple recordings - GOOD

You have enough data to train both PID and BC successfully!

---

## Next Steps

1. **Extract trajectories** from your best flights (hover + circle)
2. **Train BC model** on all collected data
3. **Test PID** with learned trajectory on real drone
4. **Test BC model** on real drone
5. **Compare performance** - which flies more like your manual control?

Ready to fly! 🚁
