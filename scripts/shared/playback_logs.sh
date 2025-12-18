#!/bin/bash

# Quick playback script for visualizing trained models

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="$PYTHONPATH:$DIR:$DIR/gym-pybullet-drones"

TRAJ="${1:-circle}"
CONTROLLER="${2:-hybrid}"
DURATION="${3:-15}"

echo "================================================"
echo "Playback: $CONTROLLER on $TRAJ trajectory"
echo "Duration: ${DURATION}s | With Domain Randomization"
echo "================================================"

# Determine model path
if [ "$CONTROLLER" = "hybrid" ]; then
    MODEL_PATH="models/hybrid_robust/${TRAJ}/final_model.zip"
elif [ "$CONTROLLER" = "ppo" ]; then
    MODEL_PATH="models/ppo_robust/${TRAJ}/final_model.zip"
else
    MODEL_PATH=""
fi

# Build the command
if [ "$CONTROLLER" = "pid" ]; then
    python << EOF
from src.testing.demo_simulation import run_demo
run_demo('pid', trajectory_type='${TRAJ}', duration=${DURATION}, record=True, domain_randomization=True)
EOF
else
    if [ -f "$MODEL_PATH" ]; then
        echo "Loading model: $MODEL_PATH"
        python << EOF
from src.testing.demo_simulation import run_demo
run_demo('${CONTROLLER}', model_path='${MODEL_PATH}', trajectory_type='${TRAJ}', duration=${DURATION}, record=True, domain_randomization=True)
EOF
    else
        echo "ERROR: Model not found at $MODEL_PATH"
        echo "Please train the model first: ./run_training.sh ${CONTROLLER} ${TRAJ} robust"
        exit 1
    fi
fi

echo ""
echo "Video saved to results/videos/${CONTROLLER}/"
echo "================================================"
