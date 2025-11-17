# Drone Hybrid RL+PID - Task Progress Tracker

**Project Goal**: Compare PID, PPO, and Hybrid controllers on real-world delivery drone scenarios under out-of-distribution conditions.

---

##  Six Main Tasks

### Task 1: Hover Stability
**Real-world**: Drone waiting to pick up package
- **Objective**: Maintain position at [0, 0, 1.0] for 10 seconds
- **Metrics**: Position variance, settling time, energy efficiency
- **Status**: IMPLEMENTED
- **Test**: `python scripts/train_pid.py --task hover --gui`

### Task 2: Waypoint Navigation (Delivery Route)
**Real-world**: Package delivery route
- **Objective**: Follow path: Home → Pickup → Delivery → Return
- **Waypoints**: [0,0,1] → [2,2,1.5] → [4,0,1] → [2,-2,1.5] → [0,0,1]
- **Metrics**: Tracking accuracy, completion time
- **Status**: IMPLEMENTED
- **Test**: `python scripts/train_pid.py --task waypoint_delivery --gui`

### Task 3: Figure-8 Trajectory
**Real-world**: Smooth flight in tight spaces
- **Objective**: Follow continuous curved path
- **Metrics**: Tracking smoothness, control effort
- **Status**: IMPLEMENTED
- **Test**: `python scripts/train_pid.py --task figure8 --gui`

### Task 4: Circle Trajectory
**Real-world**: Perimeter inspection or surveillance
- **Objective**: Fly circular path around center point
- **Metrics**: Consistent radius, smooth velocity
- **Status**: IMPLEMENTED
- **Test**: `python scripts/train_pid.py --task circle --gui`

### Task 5: Extended Hover Stability
**Real-world**: Drone waiting to pick up package (extended duration)
- **Objective**: Maintain position at [0, 0, 1.0] for 30 seconds
- **Metrics**: Position variance, energy efficiency, long-term stability
- **Test conditions**: nominal, +payload, +wind
- **Status**: IMPLEMENTED
- **Test**: `python scripts/train_pid.py --task hover_extended --gui`

### Task 6: Emergency Landing
**Real-world**: Battery critical or motor failure
- **Objective**: Descend from 2m to 0.1m safely in controlled manner
- **Metrics**: Descent rate control, no crash, vertical stability
- **Test conditions**: one motor at 50% efficiency
- **Status**: IMPLEMENTED
- **Test**: `python scripts/train_pid.py --task emergency_landing --gui`

---

##  Six OOD Test Scenarios

### 1. Nominal (Baseline)
- **Conditions**: Ideal - no disturbances
- **Purpose**: Establish baseline performance
- **Status**: READY

### 2. Heavy Payload
- **Conditions**: +20% mass (carrying package)
- **Purpose**: Test adaptability to mass changes
- **Status**: READY

### 3. Light Payload
- **Conditions**: -20% mass (after delivery)
- **Purpose**: Test opposite mass change
- **Status**: READY

### 4. Damaged Motor
- **Conditions**: Motor 3 at 70% efficiency
- **Purpose**: Test fault tolerance
- **Status**: READY

### 5. Strong Wind
- **Conditions**: 2 m/s wind gusts
- **Purpose**: Test environmental disturbances
- **Status**: READY

### 6. Combined Worst Case
- **Conditions**: Heavy payload + damaged motor + wind
- **Purpose**: Test extreme conditions
- **Status**: READY

### 7. Critical Motor Failure
- **Conditions**: Motor 3 at 50% efficiency (emergency scenario)
- **Purpose**: Test emergency landing capability
- **Status**: READY

**Test All**: `python scripts/test_ood.py --task hover --trials 3`

---

##  Week-by-Week Implementation Plan

### Week 1: PID Baseline ✅ COMPLETE
- [x] Environment setup and installation
- [x] PID controller implementation (using UTIAS DSL tuned gains)
- [x] Task definitions (6 trajectories)
- [x] Metrics and evaluation framework
- [x] OOD test scenarios (7 scenarios)
- [x] **Ground-start initialization (0.03m)** - matches real-world deployment
- [x] Data export to JSON and CSV formats
- [x] Run all 6 tasks with baseline PID
- [x] Run OOD tests on all 6 tasks (210 total tests: 6 tasks × 7 scenarios × 5 trials)
- [x] Complete baseline data collection

**Baseline PID Performance (Ground-Start, Nominal Conditions):**
- Hover: RMSE = 0.156m (includes 97cm climb from ground)
- Hover Extended: Data collected
- Waypoint Delivery: Data collected
- Figure-8: Data collected
- Circle: Data collected
- Emergency Landing: Data collected

**Key Finding**: PID baseline complete with realistic ground-start initialization. All data exported to `results/data/` (12 JSON + 12 CSV files). Ready for Week 2 PPO implementation.

**Key Finding**: PID excels in ideal conditions but fails catastrophically under wind disturbances and severely degrades with motor failures. This establishes clear motivation for RL/Hybrid approaches.

**Option 1: Automated (Recommended) - Run everything at once:**

```bash
# Run complete PID baseline collection (2-3 hours)
./scripts/run_all_pid.sh
```

**Option 2: Manual - Run commands individually:**

```bash
# STEP 1: Test each individual task (run one at a time, verify each works)
# All results saved to results/data/ in both JSON and CSV format
python scripts/train_pid.py --task hover                    # Sub-mm accuracy baseline
python scripts/train_pid.py --task hover_extended           # 30s long-term stability
python scripts/train_pid.py --task waypoint_delivery        # Delivery route tracking
python scripts/train_pid.py --task figure8                  # Smooth curved trajectory
python scripts/train_pid.py --task circle                   # Circular patrol
python scripts/train_pid.py --task emergency_landing        # Emergency descent

# STEP 2: Run OOD robustness tests (comprehensive evaluation)
# Tests all 7 scenarios, exports summary CSV + detailed JSON
python scripts/test_ood.py --task hover --trials 5                
python scripts/test_ood.py --task hover_extended --trials 5       
python scripts/test_ood.py --task waypoint_delivery --trials 5    
python scripts/test_ood.py --task figure8 --trials 5              
python scripts/test_ood.py --task circle --trials 5               
python scripts/test_ood.py --task emergency_landing --trials 5    

# STEP 3: Specific critical scenario tests
python scripts/test_ood.py --task emergency_landing --scenario critical_motor_failure --trials 3
python scripts/test_ood.py --task waypoint_delivery --scenario strong_wind --trials 3
```

**Data Output:**
- Individual task results: `results/data/pid_<task>_<timestamp>.json` and `.csv`
- OOD test results: `results/data/pid_ood_<task>_<timestamp>.json` and `.csv`
- Plots: `results/figures/`

---

### Week 2: RL Training

#### Phase 2a: Pure PPO (Current Focus)
- [x] **PPO infrastructure implemented**
  - Custom Gym environment with domain randomization
  - Task-specific reward functions for all 6 tasks
  - Ground-start initialization (0.03m) matching PID baseline
  - Automatic checkpointing and TensorBoard logging
- [ ] **TODO**: Train PPO on all 6 tasks (1M timesteps each, ~12-18 hours total)
  - [ ] Hover (2-3 hours)
  - [ ] Hover Extended (2-3 hours)
  - [ ] Waypoint Delivery (2-3 hours)
  - [ ] Figure-8 (2-3 hours)
  - [ ] Circle (2-3 hours)
  - [ ] Emergency Landing (2-3 hours)

#### Phase 2b: Hybrid Controller (After PPO Complete)
- [ ] **TODO**: Implement hybrid PID+RL controller
  - Architecture: `action = PID_output + α * RL_residual` (α = 0.3)
  - RL learns small corrections to PID baseline
  - Safer fallback if RL fails
- [ ] **TODO**: Train hybrid on all 6 tasks
- [ ] **TODO**: Compare training curves (PID vs PPO vs Hybrid)

**PPO Training Configuration:**
- Learning Rate: 3e-4
- Timesteps: 1M per task (~2-3 hours each)
- Domain Randomization: Mass ±30%, Wind 0-2m/s, Motors 70-100%
- Checkpoints: Every 50K steps
- TensorBoard: logs/ppo_<task>_<timestamp>/

**Option 1: Automated (Recommended) - Run everything at once:**

```bash
# Run complete PPO training (12-18 hours total)
# Automatically starts TensorBoard, trains all 6 tasks, evaluates each
./scripts/run_all_ppo.sh

# Monitor progress in another terminal
tensorboard --logdir logs/
# Then open: http://localhost:6006
```

**Option 2: Manual - Individual tasks:**

```bash
# Quick test (50K steps, ~5 minutes - for testing)
python scripts/train_ppo.py --task hover --timesteps 50000

# FULL TRAINING: All 6 tasks (1M steps each, ~2-3 hours per task)
# Run these commands sequentially or in separate terminals for parallel training
python scripts/train_ppo.py --task hover --timesteps 1000000
python scripts/train_ppo.py --task hover_extended --timesteps 1000000
python scripts/train_ppo.py --task waypoint_delivery --timesteps 1000000
python scripts/train_ppo.py --task figure8 --timesteps 1000000
python scripts/train_ppo.py --task circle --timesteps 1000000
python scripts/train_ppo.py --task emergency_landing --timesteps 1000000

# Monitor training progress
tensorboard --logdir logs/

# Evaluate trained model
python scripts/train_ppo.py --task hover --eval-only --model-path models/ppo/hover/ppo_hover_final.zip
```

**Model Output:**
- Training checkpoints: `models/ppo/<task>/ppo_<task>_<steps>_steps.zip`
- Best model: `models/ppo/<task>/best_model.zip`
- Final model: `models/ppo/<task>/ppo_<task>_final.zip`
- TensorBoard logs: `logs/ppo_<task>_<timestamp>/`

**Deliverables:**
- Trained PPO models for each task
- Trained Hybrid models for each task
- Training logs and TensorBoard data

---

### Week 3: Testing & Analysis
- [ ] **TODO**: Test PPO on all OOD scenarios
- [ ] **TODO**: Test Hybrid on all OOD scenarios
- [ ] **TODO**: Generate comparison plots
- [ ] **TODO**: Statistical analysis (3 controllers × 6 scenarios × 4 tasks)
- [ ] **TODO**: Create summary tables
- [ ] **TODO**: Real drone testing (if hardware available)

**Deliverables:**
- Complete performance comparison (24 test combinations)
- Plots and visualizations
- Statistical significance tests
- Final report

---

##  Expected Results Matrix

| Task | PID | PPO | Hybrid | Winner |
|------|-----|-----|--------|--------|
| **Hover - Nominal** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | PID/Hybrid |
| **Hover - Heavy** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Hybrid |
| **Hover - Wind** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Hybrid |
| **Waypoint - Nominal** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Hybrid |
| **Waypoint - Damaged** | ⭐ | ⭐⭐ | ⭐⭐⭐⭐ | Hybrid |
| **Figure8 - Combined** | ❌ | ⭐⭐ | ⭐⭐⭐⭐ | Hybrid |

**Key Hypothesis**: Hybrid will show 30-50% better RMSE in OOD scenarios

---

## 🔬 Performance Metrics Tracked

For each test run, we measure:
- **RMSE**: Root mean square error (primary metric)
- **Max Error**: Worst-case deviation
- **Mean Error**: Average tracking error
- **Final Error**: Error at trajectory end
- **Settling Time**: Time to reach target (hover only)
- **Control Effort**: Motor saturation and aggressiveness
- **Success Rate**: % of time within acceptable bounds

---

## Current Status

**Overall Progress**: 45% Complete

-  Infrastructure and environment setup
-  PID baseline implementation with **optimal tuned gains** (UTIAS DSL)
-  Task and scenario definitions
-  **CRITICAL**: Proper initialization, warmup, and data collection
-  Data collection with CSV/JSON export (in progress)
-  RL training (next)
-  Final testing and analysis (final week)

**Last Updated**: November 17, 2025

**Recent Achievement**: Fixed PID baseline - achieved **sub-millimeter hover accuracy** (0.0007m RMSE), proving this is professional-grade performance. PID now shows clear failure modes (90,000x degradation in wind) that motivate RL/Hybrid approaches.

---

##  Quick Commands Reference

```bash
# Activate environment
conda activate drone-rl-pid
cd /Users/MMD/Desktop/MLR/Drone_Hybrid_RL_PID

# Test single task with visualization
python scripts/train_pid.py --task <hover|hover_extended|waypoint_delivery|figure8|circle|emergency_landing> --gui

# Run OOD robustness comparison
python scripts/test_ood.py --task <task_name> --trials 5 --scenario all

# Test specific OOD scenario
python scripts/test_ood.py --task hover --scenario heavy_payload --trials 3
```

---

##  Notes

- All results automatically saved to `results/` directory
- Plots saved to `results/figures/`
- Use `--gui` flag for visualization during development
- Remove `--gui` for faster batch testing
- Increase `--trials` for more statistical confidence
