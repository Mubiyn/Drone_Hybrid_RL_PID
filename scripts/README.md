# Scripts Directory

This directory contains all executable scripts for the Hybrid RL-PID drone control project, organized by purpose and phase.

## Directory Structure

```
scripts/
├── README.md                    # This file
├── test_installation.py         # Verify environment setup
│
├── phase1_simulation/          # Phase 1: Simulation validation
│   ├── test_simulation_perturbations.py
│   └── analyze_simulation_results.py
│
├── phase2_real_drone/          # Phase 2: Real Tello hardware
│   ├── test_hybrid_on_tello.py
│   ├── test_all_with_perturbations.py
│   ├── train_hybrid_rl_only.py
│   └── analyze_perturbation_tests.py
│
├── shared/                     # Shared analysis tools
│   ├── analyze_hybrid_models.py
│   ├── analyze_autonomous_flights.py
│   ├── analyze_flight_data.py
│   ├── compare_controllers.py
│   ├── compare_pid_performance.py
│   ├── autonomous_data_collection.py
│   └── prepare_hybrid_training_data.py
│
├── data_generation/            # Trajectory and data generation
│   ├── generate_perfect_trajectories.py
│   ├── tune_trajectories.py
│   └── update_pid_gains.py
│
├── training_scripts/           # Training utilities (historical)
│   ├── train_pid.py
│   ├── train_pid_from_manual.py
│   ├── train_hybrid_from_autonomous.py
│   ├── train_hybrid_from_manual.py
│   ├── train_hybrid_with_trajectories.py
│   └── update_domain_randomization.py
│
└── archive/                    # Old/deprecated scripts
    ├── test_hybrid.py
    ├── test_hybrid_training.py
    ├── test_pid_baseline.py
    ├── test_ppo_training.py
    ├── test_robust_training.py
    ├── detailed_data_proof.py
    ├── inspect_manual_data.py
    ├── show_data_source.py
    ├── collect_all_autonomous.sh
    └── collect_all_tuned.sh
```

## Quick Start

### Installation Verification
```bash
python test_installation.py
```

### Phase 1: Simulation Testing
```bash
# Run perturbation tests
python phase1_simulation/test_simulation_perturbations.py

# Analyze results
python phase1_simulation/analyze_simulation_results.py
```

### Phase 2: Real Drone Testing
```bash
# Test on Tello (ensure WiFi connected)
python phase2_real_drone/test_hybrid_on_tello.py --trajectory hover

# Test with perturbations
python phase2_real_drone/test_all_with_perturbations.py

# Analyze perturbation tests
python phase2_real_drone/analyze_perturbation_tests.py
```

### Shared Analysis
```bash
# Analyze hybrid models in simulation
python shared/analyze_hybrid_models.py

# Compare controllers
python shared/compare_controllers.py
```

## Script Categories

### Active Scripts (Phase 1 & 2)

**Phase 1 - Simulation**:
- `test_simulation_perturbations.py`: Test hybrid_robust models with domain randomization
- `analyze_simulation_results.py`: Generate plots and reports

**Phase 2 - Real Drone**:
- `test_hybrid_on_tello.py`: Deploy single trajectory to Tello
- `test_all_with_perturbations.py`: Test all trajectories with wind
- `train_hybrid_rl_only.py`: Train RL-only models for Tello
- `analyze_perturbation_tests.py`: Analyze real flight wind tests

**Shared**:
- `analyze_hybrid_models.py`: Comprehensive model analysis
- `analyze_autonomous_flights.py`: Autonomous flight analysis
- `compare_controllers.py`: PID vs Hybrid comparison

### Utility Scripts

**Data Generation**:
- `generate_perfect_trajectories.py`: Create reference trajectories
- `tune_trajectories.py`: Optimize trajectory parameters
- `update_pid_gains.py`: Tune PID gains per trajectory

**Training** (Historical):
- Various training scripts from development iterations
- Kept for reference and reproducibility
- Not needed for standard workflow

### Archived Scripts

Old experimental and debugging scripts kept for historical reference:
- Test scripts from early development
- Data inspection utilities
- Deprecated training approaches (e.g., BC+RL)

## Usage Patterns

### Standard Workflow

1. **Setup**: `test_installation.py`
2. **Phase 1**: Run simulation tests and analysis
3. **Phase 2**: Deploy to Tello and analyze results
4. **Analysis**: Use shared scripts for comparisons

### Development Workflow

1. Generate/update trajectories in `data_generation/`
2. Train models using scripts in `training_scripts/` or phase folders
3. Analyze using shared tools
4. Archive old scripts when superseded

## Notes

- Scripts assume you're running from the repository root
- Phase 2 scripts require DJI Tello WiFi connection
- GPU recommended but not required for training
- All scripts support `--help` flag for usage info

## Maintenance

- **Active scripts**: Keep in phase folders or shared
- **Superseded scripts**: Move to archive with date comment
- **Broken scripts**: Fix or remove after documenting issue
- **One-off scripts**: Add clear header comment explaining purpose
