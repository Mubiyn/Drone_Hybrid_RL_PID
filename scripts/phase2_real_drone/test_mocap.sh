#!/bin/bash
# Test OptiTrack MoCap Connection

echo "============================================================"
echo "Testing OptiTrack Motion Capture Connection"
echo "============================================================"
echo ""
echo "Network Configuration:"
echo "  MoCap Server: 192.168.1.1"
echo "  Data Port: 1511"
echo "  Multicast: 239.255.42.99"
echo ""
echo "Make sure in Motive:"
echo "  1. View → Data Streaming Pane"
echo "  2. ✓ Broadcast Frame Data is enabled"
echo "  3. Local Interface is set correctly"
echo "  4. Your drone rigid body is created"
echo ""
echo "============================================================"
echo ""

python mocap_track.py \
  --mode multicast \
  --interface-ip 192.168.1.1 \
  --mcast-addr 239.255.42.99 \
  --data-port 1511 \
  --print-poses
