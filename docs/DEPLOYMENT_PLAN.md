# CLEAR PATH FORWARD: Real Drone Deployment

## Current Status
-  Hybrid model trained in simulation with domain randomization
-  Real Tello + MoCap setup working
-  Manual flight data collected
-  **Hybrid tested on real Tello → FAILED** (deployed final_model.zip instead of best_model.zip)
-  **Training bug discovered**: No `EvalCallback` → only saved `final_model.zip`, not `best_model.zip`
-  **Domain randomization gap**: Sim 27g ± 30%, Real 80g (OUTSIDE range)
-  PID not tuned for real Tello (using guessed kp=0.4, max_vel=0.5)

## Root Causes of Failure
1. **Wrong model deployed**: `final_model.zip` (last checkpoint) instead of `best_model.zip` (best performing)
2. **Sim-to-real gap**: 80g Tello outside [18.9-35.1g] randomization range
3. **Untuned PID baseline**: Conservative gains not optimized for 80g mass
4. **MoCap marker effects**: Markers add 5-15g mass + change inertia (natural domain randomization)

## Goal
**Fix training pipeline, then fine-tune hybrid model for real Tello deployment**

## CRITICAL INSIGHT: MoCap Markers = Natural Domain Randomization
⚠️ **The Tello with MoCap markers IS the deployment configuration!**
- **Added mass**: Markers increase weight (5-15g) → Total mass now 85-95g
- **Changed inertia**: Asymmetric marker placement shifts CoM
- **Drift observed**: Heavier drone behaves differently than bare 80g Tello
- **Implication**: Flight data collected WITH markers = training data for actual deployment

 **This is GOOD**: No sim-to-real gap for MoCap-equipped flights  
 **But**: If markers removed later, model won't transfer to bare Tello

**Recommendation**: Keep markers on Tello permanently OR collect separate data for bare Tello

---

## Phase 0: Fix Training Pipeline (CRITICAL - 5 min)

### Why
Current training saves only `final_model.zip` without validation → deployed wrong checkpoint

### What Was Fixed
 Added `EvalCallback` to `train_robust.py`:
- Evaluates model every 10k steps on validation env
- Saves `best_model.zip` based on highest reward
- Prevents deploying undertrained/overtrained models

### Impact
- **Previous training**: Only `final_model.zip` → may be suboptimal
- **Future training**: Both `best_model.zip` (deploy this) + `final_model.zip` (fallback)

### Action Required
**If re-training**: The fixed script will automatically save best model
**If fine-tuning**: Work with existing `final_model.zip` (we have no choice)

---

## Phase 1: PID Baseline  COMPLETED

### Results
**Auto-tuning completed** (13/20 tests, battery limited):

```
Best: kp=0.7, max_vel=0.7 → Mean error: 0.323m
```

**Tuned PID parameters for marker-equipped Tello (85-95g)**:
- Proportional gain: **kp = 0.7**
- Maximum velocity: **max_vel = 0.7 m/s**

### Steps

**1.1 Auto-tune PID**  DONE
```bash
python scripts/autonomous_data_collection.py --tune-pid --mocap
```

**1.2 Test PID on Real Tasks** ⏭️ SKIP (move directly to data collection)

**Deliverable**: Optimal PID gains identified for data collection

---

## Phase 2: Test Sim-Trained Hybrid (30 min)

### Why
See if domain randomization was sufficient for transfer

### Steps
## Phase 2: ~~Test Sim-Trained Hybrid~~ (SKIP - Already Failed)

### Status
 **User already tested sim-trained hybrid → FAILED**

### Why It Failed
1. Deployed `final_model.zip` instead of `best_model.zip` (training bug)
2. 80g Tello outside [18.9-35.1g] sim randomization range (3x mass difference)
3. PID baseline not tuned (conservative kp=0.4, max_vel=0.5)

### Lessons Learned
- Domain randomization alone insufficient (too far from real dynamics)
- Need validation during training (now fixed with `EvalCallback`)
- Must fine-tune on real data

### Decision
**Skip re-testing sim model** → Go directly to Phase 3A (fine-tuning)
Adapt sim-trained model to real Tello dynamics

### Steps

**3A.1 Collect Real Data with Tuned PID**
```bash
# Clean autonomous trajectories
python scripts/autonomous_data_collection.py --trajectory circle --kp 0.6 --max-vel 0.7 --mocap --duration 60
python scripts/autonomous_data_collection.py --trajectory figure8 --kp 0.6 --max-vel 0.7 --mocap --duration 60
python scripts/autonomous_data_collection.py --trajectory spiral --kp 0.6 --max-vel 0.7 --mocap --duration 40
## Phase 3A: Fine-tune Hybrid (CURRENT - Data Collection)

### Why
Sim-trained model failed on real Tello → must adapt to real dynamics (85-95g with markers)

### Prerequisites
 Phase 1 complete (PID tuned: kp=0.7, max_vel=0.7)
 `train_robust.py` fixed with `EvalCallback`

### Steps

**3A.1 Collect Real Data with Tuned PID** 🔄 IN PROGRESS
```bash
# Use tuned gains from Phase 1: kp=0.7, max_vel=0.7
python scripts/autonomous_data_collection.py --trajectory circle --kp 0.7 --max-vel 0.7 --mocap --duration 60
python scripts/autonomous_data_collection.py --trajectory figure8 --kp 0.7 --max-vel 0.7 --mocap --duration 60
python scripts/autonomous_data_collection.py --trajectory spiral --kp 0.7 --max-vel 0.7 --mocap --duration 40
python scripts/autonomous_data_collection.py --trajectory waypoint --kp 0.7 --max-vel 0.7 --mocap --duration 50
python scripts/autonomous_data_collection.py --trajectory hover --kp 0.7 --max-vel 0.7 --mocap --duration 30
```
- **Goal**: 5,000-10,000 total samples (3-5 diverse flights)
- **Quality**: MoCap ground truth states, continuous PID actions
- **Output**: `data/tello_flights/autonomous_{trajectory}_{timestamp}.pkl`
- **IMPORTANT**: Data collected WITH MoCap markers = training for deployment WITH markers
- **Mass**: ~85-95g (80g Tello + 5-15g markers) - heavier than bare Tello
- **Natural variation**: Marker placement creates real domain randomization (mass, inertia, drag)

**3A.2 Create Offline Fine-tuning Script** (MISSING - needs implementation)
```bash
# TODO: Create scripts/finetune_hybrid_offline.py
# Load pretrained: logs/hybrid_robust/{trajectory}/final_model.zip
# Load real data: data/tello_flights/autonomous_*.pkl
## Phase 3B: Final Validation

### Why
Demonstrate fine-tuned hybrid outperforms PID baseline on real tasks

### Prerequisites
 Phase 3A complete (fine-tuned model tested, better than PID)

### Steps

**3B.1 Extended Testing on All Real Tasks**
```bash
# Test 6 tasks defined in REAL_TASKS.md with multiple trials
# Task 1: Precision Hover (RMSE < 0.1m)
python src/real_drone/run_tello.py --controller pid --traj hover --duration 30 --mocap --trials 5
python src/real_drone/run_tello.py --controller hybrid --model logs/finetuned/best_model.zip --traj hover --duration 30 --mocap --trials 5

# Task 2: Waypoint Navigation (time < 60s, error < 0.15m per waypoint)
python src/real_drone/run_tello.py --controller pid --traj waypoint --mocap --trials 5
python src/real_drone/run_tello.py --controller hybrid --model logs/finetuned/best_model.zip --traj waypoint --mocap --trials 5

# Task 3: Return-to-Home (error < 0.15m)
# ... (implement remaining tasks)
**3B.3 Document Results**
- Create performance comparison table (PID vs Fine-tuned Hybrid)
- Generate trajectory visualization plots
- Write final report with statistical evidence

**3B.2 Statistical Analysis**
```bash
# TODO: Create scripts/analyze_final_results.py
# Compute: mean RMSE, completion time, success rate
# Statistics: paired t-tests, Cohen's d effect sizes
# Plots: comparison bar charts, trajectory overlays
```

**3B.3 Document Results**task.py \
    --task hover waypoint rth tracking \
    --controller pid hybrid \
    --trials 10 \
    --output results/final_evaluation.json
```

**3B.2 Statistical Analysis**
```bash
python scripts/analyze_final_results.py results/final_evaluation.json
# Outputs: t-tests, effect sizes, plots
```

**3B.3 Document Success**
- Record videos of hybrid vs PID
- Generate comparison plots
- Write up findings

---

## Task Definitions

### Task 1: Precision Hover
- **Objective**: Hold position (0, 0, 1.0) for 30s
- **Metrics**: RMSE, max deviation, recovery time after disturbance
- **Success**: RMSE < 0.1m

### Task 2: Waypoint Navigation
- **Objective**: Visit 4 waypoints in sequence
- **Waypoints**: [(1,0,1), (1,1,1.5), (0,1,1), (0,0,1)]
- **Metrics**: Completion time, tracking error
- **Success**: All waypoints within 0.2m, time < 60s

### Task 3: Return-to-Home (RTH)
- **Objective**: Return to start from random position
- **Metrics**: Time to home, final error
- **Success**: Final error < 0.15m

### Task 4: Moving Target Tracking (Advanced)
- **Objective**: Follow moving target (0.3 m/s random walk)
- **Metrics**: Mean tracking error
- **Success**: Error < 0.25m

---

## Timeline

### Fast Path (Hybrid works immediately)
- Phase 1: 2 hours (PID tuning + baseline)
- Phase 2: 30 min (test hybrid)
- Phase 3B: 1 hour (validation)
- **Total: ~4 hours**

### Slow Path (Need fine-tuning)
- Phase 1: 2 hours
- Phase 2: 30 min
- Phase 3A: 3-4 hours (data collection + training + testing)
- **Total: ~6-7 hours**

---

## What We're Testing

**Hypothesis**: Domain randomization in simulation was sufficient to bridge reality gap

**Test**: Deploy sim-trained hybrid directly on real Tello

**Outcomes**:
1. **Hybrid ≥ PID**: Domain randomization SUCCESS 
2. **Hybrid < PID, but stable**: Need fine-tuning (expected) ⚠️
3. **Hybrid unstable**: Reality gap too large, need re-training 

---

## Key Files Needed

Create these scripts:
1.  `autonomous_data_collection.py` - PID auto-tuning and data collection
2.  `test_real_task.py` - Evaluate controller on specific task
3.  `compare_results.py` - Statistical comparison of controllers
4.  `finetune_bc.py` - Behavioral cloning fine-tuning
5.  `finetune_offline_rl.py` - Offline RL fine-tuning (optional)

---

## Next Immediate Action

**Choice 1: Full automation (recommended)**
```bash
# Run complete pipeline
./run_real_deployment.sh
# This runs: PID tuning → baseline → hybrid test → decision
```

**Choice 2: Manual step-by-step**
```bash
# Start with PID tuning
python scripts/autonomous_data_collection.py --tune-pid --mocap
```

Which approach do you prefer?
