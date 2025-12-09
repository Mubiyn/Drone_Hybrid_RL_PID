#!/bin/bash

# Run Tello drone with trained controllers
# Usage: ./run_real_drone.sh [pid|hybrid] [trajectory] [duration] [--mocap] [--mocap-server IP] [--mocap-id ID]

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="$PYTHONPATH:$DIR:$DIR/gym-pybullet-drones"

CONTROLLER="${1:-pid}"
TRAJECTORY="${2:-hover}"
DURATION="${3:-10}"
MOCAP_ARGS=""

# Parse optional MoCap arguments (starting from 4th argument)
shift 3 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --mocap)
            MOCAP_ARGS="$MOCAP_ARGS --use-mocap"
            shift
            ;;
        --mocap-server)
            MOCAP_ARGS="$MOCAP_ARGS --mocap-server $2"
            shift 2
            ;;
        --mocap-id)
            MOCAP_ARGS="$MOCAP_ARGS --mocap-id $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "================================================"
echo "Tello Real Drone Controller"
echo "================================================"
echo "Controller: $CONTROLLER"
echo "Trajectory: $TRAJECTORY"
echo "Duration: ${DURATION}s"
if [[ -n "$MOCAP_ARGS" ]]; then
    echo "Motion Capture: ENABLED"
    echo "MoCap Options: $MOCAP_ARGS"
else
    echo "Motion Capture: DISABLED (using Tello sensors)"
fi
echo ""
echo "SAFETY CHECKLIST:"
echo "  [ ] Battery > 30%"
echo "  [ ] Clear flying space (3m x 3m minimum)"
echo "  [ ] Tello is powered on and visible"
echo "  [ ] Emergency stop ready (Ctrl+C)"
echo "  [ ] Drone on flat surface for takeoff"
if [[ -n "$MOCAP_ARGS" ]]; then
    echo "  [ ] OptiTrack streaming (test_optitrack.py passed)"
    echo "  [ ] Rigid body visible in Motive"
fi
echo ""
echo "================================================"

python src/real_drone/run_tello.py \
    --controller "$CONTROLLER" \
    --traj "$TRAJECTORY" \
    --duration "$DURATION" \
    $MOCAP_ARGS
