# Motion Capture Integration Guide

## Overview

The TelloWrapper supports Motion Capture (MoCap) systems for precise position tracking. This is optional but highly recommended for research/testing.

## Supported Systems

- **OptiTrack** (Motive software)
- **Vicon** (Tracker software)
- Any system that provides position/orientation data

## Why Use MoCap?

| Feature | Without MoCap | With MoCap |
|---------|---------------|------------|
| Position Accuracy | ±50cm (drift) | <1mm |
| Update Rate | 10 Hz | 100-360 Hz |
| XY Position | Dead reckoning (poor) | Ground truth |
| Z Position | Barometer (±10cm) | Sub-mm accuracy |
| Drift | Accumulates over time | None |

## Installation

### OptiTrack Setup

1. **Install NatNet SDK**
   ```bash
   pip install NatNetClient
   ```

2. **Configure Motive Software**
   - Create rigid body for drone (4+ markers)
   - Assign rigid body ID (e.g., ID=1)
   - Enable data streaming:
     - Edit → Preferences → Streaming
     - Enable "Broadcast Frame Data"
     - Local Interface: Your network adapter
     - Multicast Address: 239.255.42.99

3. **Update `mocap_client.py`**
   ```python
   from NatNetClient import NatNetClient
   
   client = NatNetClient()
   client.set_server_address("192.168.1.100")  # Motive computer IP
   client.run()
   ```

### Vicon Setup

1. **Install Vicon DataStream SDK**
   ```bash
   pip install pyvicon-datastream
   ```

2. **Update `mocap_client.py`**
   ```python
   from vicon_dssdk import ViconDataStream
   
   client = ViconDataStream.Client()
   client.Connect("192.168.1.100:801")  # Vicon computer IP
   client.EnableSegmentData()
   client.SetStreamMode(ViconDataStream.Client.StreamMode.EServerPush)
   ```

## Using MoCap with Tello

### Running with MoCap

```bash
# Enable MoCap flag
./run_real_drone.sh hybrid circle 20 --mocap
```

### Code Flow

1. **Initialization**
   ```python
   mocap = NatNetClient()
   mocap.start()
   tello = TelloWrapper(mocap_client=mocap)
   ```

2. **During Flight**
   ```python
   obs = tello.get_obs()
   # obs now contains MoCap position (instead of Tello's estimate)
   # Position: obs[0:3] ← from MoCap
   # Attitude: obs[3:6] ← from Tello sensors
   # Velocity: obs[6:9] ← estimated from MoCap position deltas
   ```

## Configuration

### Network Setup

**Default Settings:**
- **Multicast IP**: 239.255.42.99
- **Data Port**: 1511
- **Command Port**: 1510

**Custom Configuration:**
Edit `src/real_drone/mocap_client.py`:
```python
mocap = NatNetClient(
    server_ip="192.168.1.100",    # Your MoCap computer
    multicast_ip="239.255.42.99",
    data_port=1511
)
```

### Rigid Body Setup

1. **Marker Placement**
   - Attach 4+ reflective markers to Tello
   - Asymmetric pattern (for unique identification)
   - Markers should be visible from all angles

2. **Rigid Body Creation**
   - In Motive/Tracker: Select markers → Create Rigid Body
   - Assign ID (default code uses ID=1)
   - Name it (e.g., "Tello_Drone")

3. **Coordinate Frame**
   - OptiTrack/Vicon typically use: X-right, Y-up, Z-forward
   - Tello uses: X-forward, Y-left, Z-up
   - May need coordinate transformation (see below)

## Coordinate Transformation

If your MoCap coordinate system differs from Tello's:

```python
# In TelloWrapper.get_obs()
if self.use_mocap:
    mocap_pos = self.mocap_client.get_position()
    
    # Example: OptiTrack (X-right, Y-up, Z-forward) → Tello (X-forward, Y-left, Z-up)
    self.pos = np.array([
        mocap_pos[2],   # X (forward) ← MoCap Z
        -mocap_pos[0],  # Y (left) ← -MoCap X
        mocap_pos[1]    # Z (up) ← MoCap Y
    ])
```

## Troubleshooting

### No MoCap Data Received

**Check:**
1. Firewall allows UDP multicast on port 1511
2. Computer and MoCap system on same network
3. Motive/Tracker is streaming data
4. Rigid body is being tracked (visible in software)

**Test Connection:**
```bash
# In separate terminal
python -c "
from src.real_drone.mocap_client import NatNetClient
import time
mocap = NatNetClient()
mocap.start()
time.sleep(2)
pos = mocap.get_position()
print(f'Position: {pos}')
"
```

### Position Jumps/Noise

- **Occlusion**: Ensure markers always visible to cameras
- **Lighting**: Avoid reflective surfaces, direct sunlight
- **Marker Swap**: Use asymmetric marker pattern
- **Filtering**: Add low-pass filter to position data

### Latency Issues

- **High latency** (>20ms): Check network congestion
- **Jitter**: Use wired Ethernet instead of WiFi for MoCap computer
- **Prediction**: Can add position prediction based on velocity

## Without MoCap (Fallback)

If MoCap unavailable, system falls back to:
- Tello's internal height sensor (Z-axis only)
- Dead reckoning for XY (very inaccurate)
- Performance will be significantly degraded

**Recommendation:** For research/testing, MoCap is highly recommended!

## Performance Expectations

| Setup | Hover RMSE | Circle RMSE |
|-------|------------|-------------|
| Simulation | 0.17m | 0.39m |
| Real (No MoCap) | ~0.5-1.0m | Poor (drift) |
| Real (With MoCap) | ~0.2-0.4m | ~0.5-0.8m |

MoCap eliminates position drift and enables precise trajectory tracking!
