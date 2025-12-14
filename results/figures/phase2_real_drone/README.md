# Phase 2: Real Drone Results

This directory contains results from real DJI Tello hardware testing and deployment.

## Directory Structure

```
phase2_real_drone/
├── perturbation_analysis/   # Wind perturbation tests on Tello
├── autonomous_analysis/      # Autonomous flight analysis
├── model_analysis/           # RL model simulation analysis
│   └── hybrid_analysis/
└── README.md
```

## Models Tested

All tests use models from `logs/hybrid/rl_only_*/`:
- Circle (rl_only_circle3)
- Hover (rl_only_hover4)
- Spiral (rl_only_spiral4)

**Note**: Figure8 and Square were not successful on real hardware and are excluded.

## Test Configuration

**Phase 2 Configuration** (adjusted for real hardware):
- Residual scale: 100 RPM (reduced for stability)
- Mass variation: ±30% (increased for robustness)
- Inertia variation: ±30%
- Wind forces: 0.15N max (increased for real-world conditions)

## Test Types

### Perturbation Analysis
Tests Hybrid RL vs PID under simulated wind disturbances on real Tello drone:
- Wind direction: random
- Wind magnitude: 0-0.15N
- Metrics: tracking error, trajectory deviation, stability

### Autonomous Analysis
Tests autonomous flight performance:
- Full trajectory execution
- Position tracking accuracy
- Control smoothness
- Battery usage

### Model Analysis
Simulation-based analysis of RL models:
- Behavior visualization
- Policy analysis
- Transfer from simulation to reality

## Flight Data Structure

Real flight logs contain:
- States: [x, y, z, roll, pitch, yaw, vx, vy, vz]
- Targets: [x_target, y_target, z_target]
- Timestamps: milliseconds since flight start
- Note: Z-axis already tracked (not shrunk like Phase 1 simulation)

## Analysis Scripts

- `scripts/phase2_real_drone/test_hybrid_on_tello.py`: Deploy to Tello
- `scripts/phase2_real_drone/test_all_with_perturbations.py`: Test with perturbations
- `scripts/shared/analyze_hybrid_models.py`: Analyze rl_only models

## Real Hardware Constraints

- Tello weight: 80g
- Max payload: ~20g
- Optical flow positioning (no GPS)
- Barometer altitude sensor
- Limited control authority (100 RPM residual)
- Indoor flight only

## Expected Results

- Successful trajectories: Circle, Hover, Spiral
- Failed trajectories: Figure8, Square (too fast for Tello hardware)
- Hybrid should show improved tracking compared to PID
- Robustness to wind disturbances validated

## Notes

- All tests use djitellopy SDK
- Flight data logged at ~10-20 Hz
- Safety features: height limits, geofencing
- Manual emergency stop available
