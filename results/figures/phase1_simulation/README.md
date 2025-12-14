# Phase 1: Simulation Results

This directory contains results from simulation-based testing and validation using gym-pybullet-drones.

## Directory Structure

```
phase1_simulation/
├── perturbation_tests/     # Domain randomization testing results
│   └── perturbation_test_results_*.json
└── comparison_plots/       # Visualization of controller comparisons
    ├── tracking_error_comparison.png
    ├── improvement_percentages.png
    └── control_smoothness_comparison.png
```

## Models Tested

All tests use models from `models/hybrid_robust/`:
- Circle
- Figure8
- Hover
- Spiral
- Waypoint

## Test Configuration

**Phase 1 Configuration** (matches training):
- Residual scale: 200 RPM
- Mass variation: ±20%
- Inertia variation: ±20%
- Wind forces: 0.05N max

## Test Conditions

Each trajectory is tested under 4 conditions:
1. **PID Baseline**: PID controller without perturbations
2. **PID + DR**: PID controller with domain randomization
3. **Hybrid Baseline**: Hybrid RL controller without perturbations
4. **Hybrid + DR**: Hybrid RL controller with domain randomization

## Metrics

- **Tracking Error**: Mean position error (meters)
- **Control Smoothness**: Variance of action changes (lower is smoother)
- **Episode Reward**: RL environment reward signal
- **Episode Length**: Number of steps to complete trajectory

## Analysis Scripts

- `scripts/phase1_simulation/test_simulation_perturbations.py`: Run tests
- `scripts/phase1_simulation/analyze_simulation_results.py`: Generate plots and reports

## Expected Results

Hybrid RL should show improvements over PID, especially:
- Higher improvements on dynamic trajectories (circle, figure8, spiral)
- Stronger robustness with domain randomization
- Competitive or slightly better on hover task

## Notes

- Tests run 5 episodes per condition
- Results show mean ± std across episodes
- Domain randomization tests model robustness to real-world variations
