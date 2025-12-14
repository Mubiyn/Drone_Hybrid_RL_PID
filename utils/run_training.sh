#!/bin/bash

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Export PYTHONPATH to include the project root and the gym-pybullet-drones submodule
export PYTHONPATH="$PYTHONPATH:$DIR:$DIR/gym-pybullet-drones"

# Check for arguments
TYPE=$1
TRAJ=${2:-all}  # Default to 'all' if not specified
MODE=${3:-robust} # Default to 'robust'

if [ "$MODE" == "nominal" ]; then
    FLAG="--nominal"
    echo "MODE: NOMINAL (No Domain Randomization)"
else
    FLAG=""
    echo "MODE: ROBUST (With Domain Randomization)"
fi

if [ "$TYPE" == "ppo" ]; then
    echo "================================================"
    echo "Starting PPO Training | Trajectory: $TRAJ"
    echo "================================================"
    python src/training/train_robust_ppo.py --traj "$TRAJ" $FLAG

elif [ "$TYPE" == "hybrid" ]; then
    echo "================================================"
    echo "Starting Hybrid Training | Trajectory: $TRAJ"
    echo "================================================"
    python src/training/train_robust.py --traj "$TRAJ" $FLAG

else
    echo "Usage: ./run_training.sh [ppo|hybrid] [trajectory] [mode]"
    echo ""
    echo "Options:"
    echo "  ppo    : Train the pure PPO baseline model"
    echo "  hybrid : Train the Hybrid (PID+RL) model"
    echo "  trajectory : all (default), hover, circle, figure8, spiral, waypoint"
    echo "  mode       : robust (default), nominal"
    echo ""
    echo "Example:"
    echo "  ./run_training.sh hybrid all robust"
    echo "  ./run_training.sh ppo circle nominal"
    exit 1
fi
