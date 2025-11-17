# Quick Start Guide

Fast reference for getting started with the Drone Hybrid RL+PID project.

## Installation (5 minutes)

```bash
# Clone and navigate
cd Task1_Drone_Hybrid_RL_PID

# Option 1: Conda (Recommended)
conda env create -f environment.yml
conda activate drone-hybrid-rl

# Option 2: pip + venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Verify
python scripts/test_installation.py
```

## Your Role

### Member 1 (PID & RL Implementation)
**Week 1:**
```bash
# Implement PID controller
src/controllers/pid_controller.py
scripts/train_pid.py
```

**Week 2:**
```bash
# Implement RL training
src/controllers/rl_policy.py
src/training/ppo_trainer.py
scripts/train_rl.py
```

**Week 3:**
```bash
# Simulation testing
src/testing/sim_tester.py
scripts/evaluate.py
```

### Member 2 (Domain Randomization & Hybrid)
**Week 1:**
```bash
# Implement domain randomization
src/training/domain_randomizer.py
scripts/tune_pid.py
notebooks/01_PID_Tuning.ipynb
```

**Week 2:**
```bash
# Implement hybrid controller
src/controllers/hybrid_controller.py
scripts/train_hybrid.py
```

**Week 3:**
```bash
# Real drone deployment
src/real_drone/tello_interface.py
src/real_drone/safe_deploy.py
scripts/deploy_real.py
```

### Member 3 (Testing & Metrics)
**Week 1:**
```bash
# Implement metrics and tests
src/testing/metrics.py
tests/test_pid.py
tests/test_domain_randomization.py
```

**Week 2:**
```bash
# RL and hybrid tests
tests/test_rl_policy.py
tests/test_hybrid.py
src/training/callbacks.py
```

**Week 3:**
```bash
# Documentation and final testing
docs/ARCHITECTURE.md
docs/PID_TUNING_GUIDE.md
# Run full test suite
pytest tests/ --cov=src
```

### Member 4 (Visualization & Analysis)
**Week 1:**
```bash
# Implement visualization tools
src/utils/visualization.py
src/utils/logging_utils.py
```

**Week 2:**
```bash
# Analysis notebooks
notebooks/02_RL_Training_Analysis.ipynb
notebooks/03_Hybrid_Comparison.ipynb
```

**Week 3:**
```bash
# Final documentation
docs/RL_TRAINING_GUIDE.md
docs/DEPLOYMENT_GUIDE.md
# Create result visualizations
```

## Common Tasks

### Train PID Controller
```bash
python scripts/train_pid.py --task hover --episodes 100
```

### Train RL Agent
```bash
python scripts/train_rl.py \
    --task waypoint \
    --timesteps 500000 \
    --domain-randomization

# Monitor training
tensorboard --logdir logs/tensorboard/
```

### Train Hybrid System
```bash
python scripts/train_hybrid.py \
    --pid-model models/ppo/best_hover.pkl \
    --timesteps 300000 \
    --hybrid-mode adaptive
```

### Evaluate Model
```bash
python scripts/evaluate.py \
    --model models/hybrid/best_model.zip \
    --n-episodes 50 \
    --render \
    --record-video
```

### Run Tests
```bash
# All tests
pytest tests/

# Specific test
pytest tests/test_pid.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Deploy to Real Drone
```bash
# Safe test mode
python scripts/deploy_real.py \
    --model models/hybrid/best_model.zip \
    --test-mode safe \
    --max-altitude 1.0

# Full deployment
python scripts/deploy_real.py \
    --model models/hybrid/best_model.zip \
    --task waypoint
```

## Git Workflow

```bash
# Start working
git checkout -b feature/pid-controller
# ... implement code ...

# Commit changes
git add src/controllers/pid_controller.py
git commit -m "Implement PID controller with tuning"

# Push to remote
git push origin feature/pid-controller

# Create pull request on GitHub
```

## File Locations

| What | Where |
|------|-------|
| Implementation Guide | `/Users/MMD/Downloads/Task1_Drone_Hybrid_RL_PID_Guide.md` |
| Configs | `config/*.yaml` |
| Source Code | `src/` |
| Scripts | `scripts/` |
| Tests | `tests/` |
| Notebooks | `notebooks/` |
| Models | `models/` |
| Results | `results/` |

## Configuration Files

### PID Hover
```bash
config/pid_hover_config.yaml
```

### RL Waypoint
```bash
config/rl_waypoint_config.yaml
```

### Hybrid Trajectory
```bash
config/hybrid_trajectory_config.yaml
```

### Domain Randomization
```bash
config/domain_randomization.yaml
```

## Quick Commands Reference

```bash
# Environment
conda activate drone-hybrid-rl              # Activate environment
python scripts/test_installation.py         # Verify installation

# Training
python scripts/train_pid.py                 # Train PID
python scripts/train_rl.py                  # Train RL
python scripts/train_hybrid.py              # Train hybrid

# Evaluation
python scripts/evaluate.py --model <path>   # Evaluate model
tensorboard --logdir logs/tensorboard/      # View training

# Testing
pytest tests/                               # Run all tests
pytest tests/test_pid.py -v                 # Run specific test

# Deployment
python scripts/deploy_real.py               # Deploy to Tello
```

## Troubleshooting

### Import Errors
```bash
# Make sure you're in project root
cd Task1_Drone_Hybrid_RL_PID

# Reinstall in editable mode
pip install -e .
```

### PyBullet Issues
```bash
# Specific version
pip install pybullet==3.2.5
```

### Tello Connection
```bash
# Check connection
python -c "from djitellopy import Tello; t = Tello(); t.connect()"

# Verify WiFi to Tello network
```

### CUDA Not Available
```bash
# Check PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall with CUDA
pip install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

## Daily Workflow

### Morning
1. Pull latest changes: `git pull origin main`
2. Check what to implement today (see PROJECT_STRUCTURE.md)
3. Create/checkout feature branch

### During Day
1. Implement assigned module
2. Write unit tests
3. Run tests locally
4. Update documentation

### Evening
1. Commit changes with clear message
2. Push to feature branch
3. Update team on progress
4. Plan tomorrow's tasks

## Week-by-Week Checklist

### Week 1
- [ ] PID controller implemented
- [ ] Domain randomization working
- [ ] Visualization tools ready
- [ ] PID successfully hovering
- [ ] Unit tests passing

### Week 2
- [ ] RL training script working
- [ ] Hybrid controller implemented
- [ ] Models training successfully
- [ ] TensorBoard monitoring active
- [ ] Analysis notebooks created

### Week 3
- [ ] Simulation testing complete
- [ ] Tello interface working
- [ ] Safe deployment tested
- [ ] All documentation complete
- [ ] Final report ready

## Help

- **Questions?** Check the main README.md
- **Bugs?** Open an issue on GitHub
- **Code examples?** See Task1_Drone_Hybrid_RL_PID_Guide.md
- **Architecture?** See docs/ARCHITECTURE.md (to be created)

## Useful Links

- [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones)
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [DJI Tello Python](https://github.com/damiafuentes/DJITelloPy)
- [PyBullet Quickstart](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA)

---

**Pro Tip:** Keep this file open in a separate tab for quick reference while working!

**Last Updated:** November 13, 2025
