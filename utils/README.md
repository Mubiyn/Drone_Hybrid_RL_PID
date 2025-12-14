# Utility Scripts and Tools

This directory contains helper scripts and utilities used during development and testing.

## Contents

### Shell Scripts

**Data Collection**:
- `autonomous_quickstart.sh` - Quick autonomous data collection
- `collect_autonomous.sh` - Collect autonomous flight data
- `collect_tello_data.sh` - Tello-specific data collection

**Execution Helpers**:
- `run_demo.sh` - Run demonstration flights
- `run_evaluation.sh` - Run evaluation tests
- `run_playback.sh` - Playback recorded flights
- `run_real_drone.sh` - Real drone deployment helper
- `run_training.sh` - Training execution wrapper
- `train_all_hybrid.sh` - Batch train all hybrid models

**Diagnostics**:
- `network_diagnostics.sh` - Tello network connectivity tests
- `test_mocap.sh` - Motion capture system tests

### Python Utilities

**Visualization**:
- `visualize_trajectory.py` - Trajectory visualization tool

**Testing**:
- `test_tello_setup.py` - Verify Tello connection and setup

**Experimental**:
- `mocap_track.py` - Motion capture tracking (not used)

## Usage

These scripts are **supplementary tools** and not part of the main workflow documented in README.md.

### Example Usage

```bash
# Test Tello connection
python utils/test_tello_setup.py

# Visualize a trajectory
python utils/visualize_trajectory.py --file data/expert_trajectories/perfect_circle_trajectory.pkl

# Quick autonomous data collection
bash utils/autonomous_quickstart.sh
```

## Note

Most of these scripts were created during early development and may require updates to work with the current codebase. For the standard workflow, use the scripts in `scripts/phase1_simulation/` and `scripts/phase2_real_drone/` instead.

These utilities are kept for:
- Historical reference
- Potential future use cases
- Debugging and diagnostics
- One-off tasks not in main workflow
