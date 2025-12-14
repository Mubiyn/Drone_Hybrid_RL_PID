# Results Directory

This directory contains all experimental results from the Hybrid RL-PID drone control project.

## Organization

Results are organized by project phase:

### Phase 1: Simulation Validation
- **Directory**: `phase1_simulation/`
- **Purpose**: Validate hybrid_robust models in PyBullet simulation
- **Models**: `models/hybrid_robust/*`
- **Environment**: gym-pybullet-drones with CF2X model
- **Configuration**: residual_scale=200, DR ±20%/0.05N

See [phase1_simulation/README.md](phase1_simulation/README.md) for details.

### Phase 2: Real Drone Deployment
- **Directory**: `phase2_real_drone/`
- **Purpose**: Deploy and test on real DJI Tello hardware
- **Models**: `logs/hybrid/rl_only_*`
- **Hardware**: DJI Tello (80g quadrotor)
- **Configuration**: residual_scale=100, DR ±30%/0.15N

See [phase2_real_drone/README.md](phase2_real_drone/README.md) for details.

### Additional Directories

- **figures/**: Miscellaneous plots and figures
- **videos/**: Flight demonstration videos (see `videos/README.md`)

## Key Differences Between Phases

| Aspect | Phase 1 (Simulation) | Phase 2 (Real Drone) |
|--------|---------------------|---------------------|
| **Platform** | PyBullet simulation | DJI Tello hardware |
| **Models** | models/hybrid_robust/ | logs/hybrid/rl_only_* |
| **Residual Scale** | 200 RPM | 100 RPM |
| **Mass DR** | ±20% | ±30% |
| **Inertia DR** | ±20% | ±30% |
| **Wind DR** | 0.05N | 0.15N |
| **Trajectories** | All 5 | Circle, Hover, Spiral only |

## Configuration Evolution

The configuration parameters changed between phases for good reasons:

1. **Residual Scale Reduction** (200→100 RPM):
   - Tello has limited control authority
   - Lower residual prevents oscillations on real hardware
   - Circle trajectory was too aggressive with 200 RPM

2. **DR Increase** (±20%→±30%, 0.05N→0.15N):
   - Real world has more uncertainty than simulation
   - Increased robustness requirements for hardware deployment
   - Better generalization to varying conditions

## Analysis Tools

- **Phase 1 Scripts**: `scripts/phase1_simulation/`
- **Phase 2 Scripts**: `scripts/phase2_real_drone/`
- **Shared Analysis**: `scripts/shared/`

## Results Summary

### Phase 1 (Simulation)
-  Hybrid outperforms PID on dynamic trajectories
-  Strong robustness to domain randomization
-  All 5 trajectories successful in simulation

### Phase 2 (Real Drone)
-  Successful deployment on circle, hover, spiral
- ⚠️ Figure8 and square too aggressive for Tello hardware
-  Improved tracking over PID baseline
-  Robust to wind disturbances

## Accessing Results

1. **View plots**: Open PNG files in respective directories
2. **Read JSON data**: Use Python/JSON viewer for detailed metrics
3. **Watch videos**: See `videos/` directory (requires Google Drive link)
4. **Run analysis**: Execute scripts in `scripts/` directories

## Notes

- Results are timestamped (format: YYYYMMDD_HHMMSS)
- Multiple runs of same test create multiple JSON files
- Latest results are typically the most relevant
- All plots generated automatically by analysis scripts
