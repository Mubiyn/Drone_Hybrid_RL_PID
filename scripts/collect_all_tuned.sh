#!/bin/zsh
# Collect autonomous flight data with optimized PID gains per trajectory

set -e

echo "========================================="
echo "Autonomous Data Collection (Tuned)"
echo "========================================="
echo ""

# Configuration
DURATION=20
OUTPUT_DIR="data/flight_logs"
TRAJ_DIR="data/expert_trajectories"

mkdir -p "$OUTPUT_DIR"

# Tuned PID gains per trajectory (from tuning results)
# Format: trajectory kp max_vel
TRAJECTORIES=(
    "circle:0.8:0.9"
    "square:0.7:0.8"
    "figure8:0.4:0.5"
    "spiral:0.8:0.9"
    "hover:0.6:0.7"
)

echo "Using tuned PID gains:"
for entry in "${TRAJECTORIES[@]}"; do
    IFS=':' read -r traj kp max_vel <<< "$entry"
    echo "  $traj: kp=$kp, max_vel=$max_vel"
done
echo ""

# Collect data for each trajectory
for entry in "${TRAJECTORIES[@]}"; do
    IFS=':' read -r traj kp max_vel <<< "$entry"
    
    echo ""
    echo "========================================="
    echo "Collecting: $traj"
    echo "========================================="
    
    # Check if trajectory file exists
    traj_file="$TRAJ_DIR/perfect_${traj}_trajectory.pkl"
    if [ ! -f "$traj_file" ]; then
        echo "✗ Trajectory file not found: $traj_file"
        exit 1
    fi
    
    echo "Trajectory: $traj_file"
    echo "Duration: ${DURATION}s"
    echo "PID Gains: kp=$kp, max_vel=$max_vel"
    echo ""
    echo "Ready to fly?"
    echo ""
    read "?Press ENTER to start or Ctrl+C to skip..."
    
    # Run autonomous flight with tuned gains
    python scripts/autonomous_data_collection.py \
        --trajectory-file "$traj_file" \
        --duration "$DURATION" \
        --output-dir "$OUTPUT_DIR" \
        --kp "$kp" \
        --max-vel "$max_vel" \
        --rate 20
    
    echo ""
    echo "✓ $traj complete!"
    echo ""
    
    # Rest between flights
    if [ "$traj" != "hover" ]; then
        echo "Rest: 10 seconds..."
        sleep 10
    fi
done

echo ""
echo "========================================="
echo "✓ Data Collection Complete!"
echo "========================================="
echo ""
echo "Data saved to: $OUTPUT_DIR/autonomous_*.pkl"
echo ""
echo "Next steps:"
echo "  1. Analyze: python scripts/analyze_autonomous_flights.py"
echo "  2. Train: ./train_all_hybrid.sh"
echo ""
