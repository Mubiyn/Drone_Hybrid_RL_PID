#!/bin/bash

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Export PYTHONPATH to include the project root and the gym-pybullet-drones submodule
export PYTHONPATH="$PYTHONPATH:$DIR:$DIR/gym-pybullet-drones"

echo "================================================"
echo "Starting Demo Simulation"
echo "================================================"

# Run the demo script and pass all arguments (e.g., --controller pid --record)
python src/testing/demo_simulation.py "$@"
