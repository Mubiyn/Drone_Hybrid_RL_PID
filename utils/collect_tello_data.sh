#!/bin/bash
# Launch manual Tello control for data collection

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="$PYTHONPATH:$DIR:$DIR/gym-pybullet-drones"

USE_MOCAP=""
if [ "$1" = "--mocap" ]; then
    USE_MOCAP="--mocap"
    echo "Using Motion Capture for ground truth"
fi

echo "================================================"
echo "Tello Manual Data Collection"
echo "================================================"
echo ""
echo "INSTRUCTIONS:"
echo "1. Connect to Tello WiFi"
echo "2. Press T to takeoff"
echo "3. Press R to start recording"
echo "4. Fly manually using keyboard"
echo "5. Press R again to stop recording"
echo "6. Press L to land"
echo "7. Repeat for multiple flights"
echo "8. Press ESC to quit"
echo ""
echo "Data will be saved to: data/tello_flights/"
echo ""
echo "================================================"

python src/real_drone/manual_data_collection.py $USE_MOCAP
