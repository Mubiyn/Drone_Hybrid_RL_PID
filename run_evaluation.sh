#!/bin/bash

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Export PYTHONPATH to include the project root and the gym-pybullet-drones submodule
export PYTHONPATH="$PYTHONPATH:$DIR:$DIR/gym-pybullet-drones"

echo "================================================"
echo "Starting Comprehensive Evaluation"
echo "================================================"
echo "This will evaluate PID, PPO, and Hybrid controllers on all tasks."
echo "Results will be saved to results/figures/"
echo ""

# Check for robust flag
if [ "$1" == "--robust" ]; then
    echo "Running ROBUST evaluation (Wind + Mass Noise)..."
    python src/testing/eval_comparison.py --robust
else
    echo "Running NOMINAL evaluation (Ideal conditions)..."
    echo "To run robust evaluation, use: ./run_evaluation.sh --robust"
    python src/testing/eval_comparison.py
fi
