# Installation and Testing Checklist

This document provides step-by-step verification that the repository is properly set up and all functionality works correctly.

## Prerequisites Check

### System Requirements
- [ ] macOS, Linux, or Windows with WSL2
- [ ] Python 3.10 or higher installed
- [ ] Git installed
- [ ] (Optional) CUDA 11.7+ for GPU training
- [ ] (Optional) DJI Tello drone for Phase 2 testing

### Verify Python Version
```bash
python --version
# Should show: Python 3.10.x or higher
```

---

## Installation Testing

### Step 1: Clone Repository

```bash
git clone https://github.com/Mubiyn/Drone_Hybrid_RL_PID.git
cd Drone_Hybrid_RL_PID
```

**Verification**:
- [ ] Repository cloned successfully
- [ ] All files present (`ls -la` shows README.md, METHODOLOGY.md, etc.)

### Step 2: Create Environment

**Option A: Conda (Recommended)**
```bash
conda env create -f environment.yml
conda activate drone-hybrid-rl
```

**Option B: pip + venv**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Verification**:
- [ ] Environment created without errors
- [ ] Environment activated (prompt shows `(drone-hybrid-rl)` or `(venv)`)

### Step 3: Install gym-pybullet-drones

```bash
cd gym-pybullet-drones
pip install -e .
cd ..
```

**Verification**:
- [ ] Installation completed successfully
- [ ] No import errors when running:
  ```python
  python -c "import gym_pybullet_drones; print('Success!')"
  ```

### Step 4: Run Installation Test

```bash
python scripts/test_installation.py
```

**Expected Output**:
```
Testing environment setup...
✓ NumPy installed
✓ PyTorch installed  
✓ gym-pybullet-drones installed
✓ Stable-Baselines3 installed
✓ djitellopy installed
✓ All dependencies OK
Installation test passed!
```

**Verification**:
- [ ] All checks passed
- [ ] No import errors
- [ ] No missing dependencies

---

## Phase 1: Simulation Testing

### Step 5: Verify Models Exist

```bash
ls -la models/hybrid_robust/
```

**Expected**:
- [ ] circle/ directory exists
- [ ] figure8/ directory exists
- [ ] hover/ directory exists
- [ ] spiral/ directory exists
- [ ] waypoint/ directory exists

Each should contain:
- [ ] final_model.zip
- [ ] vec_normalize.pkl (if present)

### Step 6: Run Phase 1 Perturbation Tests

**Note**: This will take 2-3 hours to complete all 5 trajectories.

```bash
python scripts/phase1_simulation/test_simulation_perturbation.py
```

**Expected Behavior**:
- [ ] Tests all 5 trajectories (circle, figure8, hover, spiral, waypoint)
- [ ] 4 test conditions per trajectory (PID baseline, PID+DR, Hybrid baseline, Hybrid+DR)
- [ ] Displays progress for each test
- [ ] Saves results to `results/phase1_simulation/perturbation_tests/`
- [ ] No crashes or errors

**Expected Output**:
```
======================================================================
PHASE 1: SIMULATION PERTURBATION TESTING
======================================================================

=== CIRCLE ===
[1/4] Testing PID baseline...
✓ PID Baseline: Error=0.1470m
[2/4] Testing PID + DR...
✓ PID + DR: Error=0.2962m
[3/4] Testing Hybrid baseline...
✓ Hybrid Baseline: Error=0.1225m
[4/4] Testing Hybrid + DR...
✓ Hybrid + DR: Error=0.1472m

Improvement:
  Baseline: +16.7%
  With DR:  +50.3%
...
```

### Step 7: Generate Phase 1 Analysis Plots

```bash
python scripts/phase1_simulation/analyze_simulation_results.py
```

**Expected Behavior**:
- [ ] Loads latest test results JSON
- [ ] Generates 3 plots:
  - [ ] tracking_error_comparison.png
  - [ ] improvement_percentages.png
  - [ ] control_smoothness_comparison.png
- [ ] Creates text summary report
- [ ] Saves to `results/phase1_simulation/comparison_plots/`

**Verification**:
```bash
ls -la results/phase1_simulation/comparison_plots/
# Should show 3 PNG files
```

- [ ] All 3 plots created
- [ ] Plots display correctly (open and verify)
- [ ] Summary report makes sense

---

## Phase 2: Real Drone Setup (Optional)

### Step 8: Verify Tello Connection

**Prerequisites**:
- [ ] Tello drone powered on
- [ ] Connected to Tello WiFi (TELLO-XXXXXX)

```bash
python utils/test_tello_setup.py
```

**Expected Output**:
```
Connecting to Tello...
✓ Tello connected
✓ Battery: 85%
✓ Temperature: OK
Connection test passed!
```

**Verification**:
- [ ] Connects successfully
- [ ] Battery level reported
- [ ] No connection errors

### Step 9: Analyze Phase 2 Results

**Note**: This uses pre-recorded flight data.

```bash
python scripts/phase2_real_drone/analyze_perturbation_tests.py
```

**Expected Behavior**:
- [ ] Analyzes hover and spiral only (skips circle, figure8, square)
- [ ] Generates plots for each trajectory
- [ ] Creates comparison plots
- [ ] Saves to `results/phase2_real_drone/perturbation_analysis/`

**Verification**:
```bash
ls -la results/phase2_real_drone/perturbation_analysis/
```

- [ ] hover/ directory with plots
- [ ] spiral/ directory with plots
- [ ] summary_all_trajectories.png created

---

## Documentation Testing

### Step 10: Verify Documentation Files

- [ ] README.md exists and is well-formatted
- [ ] METHODOLOGY.md exists and is comprehensive
- [ ] RESULTS.md exists with actual data
- [ ] IMPLEMENTATION_PLAN.md shows progress
- [ ] All internal links work (click through in viewer)

### Step 11: Check Repository Organization

**Scripts**:
```bash
ls -la scripts/
```
- [ ] phase1_simulation/ exists with 2 scripts
- [ ] phase2_real_drone/ exists with 4 scripts
- [ ] shared/ exists with analysis scripts
- [ ] data_generation/ exists
- [ ] training_scripts/ exists
- [ ] archive/ exists
- [ ] README.md explains organization

**Results**:
```bash
ls -la results/
```
- [ ] phase1_simulation/ exists
- [ ] phase2_real_drone/ exists
- [ ] README.md explains organization
- [ ] Each subdirectory has README.md

**Models**:
```bash
ls -la models/
```
- [ ] hybrid_robust/ exists with 5 trajectories

**Logs**:
```bash
ls -la logs/
```
- [ ] hybrid_robust/ exists (Phase 1 training logs)
- [ ] hybrid_tello_drone/ exists (Phase 2 training logs)

### Step 12: Verify Clean Repository

**No loose files**:
```bash
ls | grep -E '\.(sh|py|txt)$'
```
- [ ] No loose shell scripts in root
- [ ] No loose Python files in root (except allowed ones)
- [ ] No temporary files

**No cache files**:
```bash
find . -name '__pycache__' -o -name '.DS_Store'
```
- [ ] No __pycache__ directories
- [ ] No .DS_Store files

---

## Common Issues and Solutions

### Issue: PyBullet fails to import on Apple Silicon

**Solution**:
```bash
conda install -c conda-forge pybullet
```

### Issue: GPU not detected

**Check**:
```python
import torch
print(torch.cuda.is_available())
```

**Solution**: Install CUDA-enabled PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu117
```

### Issue: Tello won't connect

**Troubleshooting**:
1. Verify WiFi connection: `ping 192.168.10.1`
2. Restart Tello drone
3. Check firewall settings
4. Try different USB WiFi adapter (if applicable)

### Issue: Test scripts fail with ModuleNotFoundError

**Solution**: Ensure environment is activated and gym-pybullet-drones installed:
```bash
conda activate drone-hybrid-rl
cd gym-pybullet-drones && pip install -e . && cd ..
```

### Issue: Plots don't display

**Solution**: Install matplotlib backend:
```bash
pip install PyQt5
# or
conda install pyqt
```

---

## Complete Workflow Test

### Quick Test (30 minutes)

1. [ ] Install environment
2. [ ] Run installation test
3. [ ] Verify models exist
4. [ ] Run ONE Phase 1 test (e.g., hover only)
5. [ ] Verify results saved
6. [ ] Check documentation files

### Full Test (3-4 hours)

1. [ ] Install environment
2. [ ] Run installation test
3. [ ] Run complete Phase 1 tests (all 5 trajectories)
4. [ ] Generate Phase 1 plots
5. [ ] Run Phase 2 analysis
6. [ ] Verify all documentation
7. [ ] Check repository organization
8. [ ] Test Tello connection (if available)

---

## External Review Checklist

Give this checklist to someone unfamiliar with the project:

### First Impressions
- [ ] README.md is clear and well-organized
- [ ] Installation instructions are easy to follow
- [ ] Project purpose is immediately clear
- [ ] Results are prominently displayed

### Installation
- [ ] Environment setup worked without issues
- [ ] Installation test passed
- [ ] No confusing error messages

### Running Code
- [ ] Scripts execute without modification
- [ ] Output is understandable
- [ ] Progress is clearly indicated
- [ ] Results are saved in expected locations

### Documentation
- [ ] METHODOLOGY.md explains approach clearly
- [ ] RESULTS.md presents findings well
- [ ] All technical terms explained
- [ ] Links to results/plots work

### Overall
- [ ] Repository is professional
- [ ] Code is well-organized
- [ ] Documentation is comprehensive
- [ ] Would recommend to others

---

## Sign-Off

### Tested By: _______________
### Date: _______________
### Environment: _______________
### Overall Result: [ ] PASS [ ] FAIL

### Notes:
```
(Add any issues encountered or suggestions for improvement)
```

---

*This checklist ensures the repository is fully functional and ready for external evaluation.*
