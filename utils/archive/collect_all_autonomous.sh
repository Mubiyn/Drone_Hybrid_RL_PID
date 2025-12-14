#!/bin/bash
# Collect autonomous flight data for all trajectories using open-loop PID control
# This data will be used for Hybrid RL training

set -e

echo "========================================="
echo "Autonomous Data Collection - All Tasks"
echo "========================================="
echo ""

# Configuration
DURATION=20  # 20 seconds per flight
OUTPUT_DIR="data/flight_logs"
TRAJ_DIR="data/expert_trajectories"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Trajectories to collect
TRAJECTORIES=("circle" "square" "figure8" "spiral" "hover")

# Collect data for each trajectory
for traj in "${TRAJECTORIES[@]}"; do
    echo ""
    echo "========================================="
    echo "Collecting: $traj"
    echo "========================================="
    
    # Check if trajectory file exists
    traj_file="$TRAJ_DIR/perfect_${traj}_trajectory.pkl"
    if [ ! -f "$traj_file" ]; then
        echo "✗ Trajectory file not found: $traj_file"
        echo "  Run: python scripts/generate_perfect_trajectories.py"
        exit 1
    fi
    
    echo "Trajectory file: $traj_file"
    echo "Duration: ${DURATION}s"
    echo ""
    echo "Ready to fly? Make sure:"
    echo "  1. Tello is powered on"
    echo "  2. Clear flight space"
    echo "  3. Battery > 50%"
    echo ""
    read -p "Press ENTER to start flight or Ctrl+C to skip..."
    
    # Run autonomous flight
    python scripts/autonomous_data_collection.py \
        --trajectory-file "$traj_file" \
        --duration "$DURATION" \
        --output-dir "$OUTPUT_DIR" \
        --rate 20
    
    echo ""
    echo "✓ $traj complete!"
    echo ""
    
    # Wait between flights for battery/setup
    if [ "$traj" != "hover" ]; then
        echo "Rest period: 10 seconds for battery recovery..."
        echo "Use this time to:"
        echo "  - Check battery level"
        echo "  - Reposition drone if needed"
        echo "  - Prepare for next flight"
        echo ""
        sleep 10
    fi
done

echo ""
echo "========================================="
echo "✓ All Autonomous Data Collection Complete!"
echo "========================================="
echo ""
echo "Collected data for:"
for traj in "${TRAJECTORIES[@]}"; do
    echo "  ✓ $traj"
done
echo ""
echo "Data saved to: $OUTPUT_DIR/autonomous_*.pkl"
echo ""
echo "Next step: Train Hybrid RL models"
echo "  ./train_all_hybrid.sh"
echo ""
