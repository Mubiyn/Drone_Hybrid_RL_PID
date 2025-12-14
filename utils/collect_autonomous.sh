#!/bin/bash
# Quick launcher for autonomous data collection

echo "🚁 Autonomous Tello Data Collection"
echo "======================================"
echo ""
echo "Select trajectory:"
echo "  1) Circle (smooth, constant altitude)"
echo "  2) Figure-8 (complex lateral motion)"
echo "  3) Spiral (ascending with rotation)"
echo "  4) Waypoint (square pattern)"
echo "  5) Hover (stationary, for PID tuning)"
echo "  6) PID Tuning Mode (automatic gain optimization)"
echo ""
read -p "Choice [1-6]: " choice

case $choice in
    1) TRAJ="circle" ;;
    2) TRAJ="figure8" ;;
    3) TRAJ="spiral" ;;
    4) TRAJ="waypoint" ;;
    5) TRAJ="hover" ;;
    6) 
        echo ""
        echo "Running PID tuning mode..."
        python scripts/autonomous_data_collection.py --tune-pid --mocap
        exit 0
        ;;
    *) 
        echo "Invalid choice"
        exit 1
        ;;
esac

read -p "Use MoCap? [y/N]: " use_mocap
read -p "Duration (seconds) [60]: " duration
read -p "PID kp gain [0.4]: " kp
read -p "Max velocity (m/s) [0.5]: " max_vel

duration=${duration:-60}
kp=${kp:-0.4}
max_vel=${max_vel:-0.5}

MOCAP_FLAG=""
if [[ "$use_mocap" =~ ^[Yy]$ ]]; then
    MOCAP_FLAG="--mocap"
fi

echo ""
echo "Starting flight with:"
echo "  Trajectory: $TRAJ"
echo "  Duration: ${duration}s"
echo "  PID kp: $kp"
echo "  Max vel: $max_vel m/s"
echo "  MoCap: $use_mocap"
echo ""
read -p "Press Enter to start (Ctrl+C to cancel)..."

python scripts/autonomous_data_collection.py \
    --trajectory $TRAJ \
    --duration $duration \
    --kp $kp \
    --max-vel $max_vel \
    $MOCAP_FLAG

echo ""
echo "✓ Flight complete!"
