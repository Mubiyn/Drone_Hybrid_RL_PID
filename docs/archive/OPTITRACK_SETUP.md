# OptiTrack Setup Guide for Tello Drone

Complete guide for integrating OptiTrack motion capture with your Tello drone for precise position tracking.

## Table of Contents
- [Hardware Requirements](#hardware-requirements)
- [Network Setup](#network-setup)
- [Motive Configuration](#motive-configuration)
- [Rigid Body Creation](#rigid-body-creation)
- [Testing Connection](#testing-connection)
- [Running with MoCap](#running-with-mocap)
- [Troubleshooting](#troubleshooting)

## Hardware Requirements

### OptiTrack System
- **OptiTrack cameras**: 4+ cameras recommended for good coverage
- **Motive software**: Version 2.2+ (supports NatNet 3.0+)
- **Network**: Gigabit Ethernet recommended
- **Computer**: Running Motive (can be same machine or separate)

### Tello Drone
- **Retroreflective markers**: 4-5 markers (8-14mm diameter)
  - Available from OptiTrack or generic motion capture suppliers
  - Asymmetric placement for unique identification
- **Marker mounting**: Double-sided tape or custom 3D-printed holder
- **Suggested pattern**:
  ```
  Top view of Tello:
       [FRONT]
         M1
         
    M2   ·    M3    (· = center)
         
         M4
       [BACK]
  ```

## Network Setup

### Single Computer Setup (Motive + Tello on same machine)
```bash
# Motive will stream to localhost (127.0.0.1)
# No additional network configuration needed
```

### Multi-Computer Setup (Motive on separate machine)

1. **Connect both computers to same network** (WiFi or Ethernet)

2. **Find Motive computer IP address**:
   - Windows: `ipconfig` (look for IPv4 Address)
   - Mac: System Preferences → Network
   - Linux: `ip addr show`
   - Example: `192.168.1.100`

3. **Verify connectivity**:
   ```bash
   # On Tello control computer, ping Motive computer
   ping 192.168.1.100
   ```

4. **Configure firewall** (on Motive computer):
   - Allow incoming UDP on port **1511** (data stream)
   - Allow incoming UDP on port **1510** (commands)
   - Windows Firewall: Create inbound rule for UDP 1510-1511
   - Mac: System Preferences → Security & Privacy → Firewall → Options

## Motive Configuration

### 1. Camera Calibration
- Complete OptiTrack camera calibration (wanding + ground plane)
- Verify calibration quality in Motive

### 2. Data Streaming Setup

**In Motive:**

1. **Open Data Streaming Pane**:
   - View → Data Streaming Pane

2. **Configure Streaming**:
   ```
   ✓ Broadcast Frame Data (ENABLE THIS!)
   
   Stream Rigid Bodies: ON
   Stream Markers: OFF (optional, not needed)
   Stream Skeletons: OFF
   
   Local Interface: [Your network adapter]
   
   Up Axis: Y Up (or Z Up, depending on your setup)
   
   Multicast Interface: 239.255.42.99 (default)
   Command Port: 1510 (default)
   Data Port: 1511 (default)
   ```

3. **NatNet Version**:
   - Use NatNet 3.0 or higher
   - Version 4.0+ recommended for best compatibility

4. **Coordinate System**:
   - Note your coordinate system (Y-up or Z-up)
   - Our code assumes Z-up (standard for drones)
   - If Y-up: You'll need to swap axes in TelloWrapper

### 3. Verify Streaming
- Bottom of Data Streaming pane should show: **"Broadcasting frame data"**
- Frame counter should increment

## Rigid Body Creation

### 1. Place Markers on Tello

**Recommended 4-marker asymmetric pattern**:
```
Marker layout (top view):
- M1: Front center (15mm from front edge)
- M2: Left side (10mm from center)
- M3: Right side (10mm from center, 5mm forward of M2)
- M4: Back center (20mm from back edge)

This asymmetric pattern ensures unique identification
```

**Marker placement tips**:
- Clean Tello surface with alcohol before applying
- Use small markers (8mm) to reduce weight
- Ensure markers are firmly attached (won't fall off during flight)
- Avoid symmetrical patterns (causes identity swaps)

### 2. Create Rigid Body in Motive

1. **Select markers**:
   - Place Tello in capture volume
   - In Motive 3D view, Ctrl+Click to select all 4 markers
   - Make sure only Tello markers are selected

2. **Create rigid body**:
   - Right-click → Rigid Body → Create From Selected Markers
   - Name it: `Tello` (or `Drone1`, etc.)

3. **Set rigid body properties**:
   - **Rigid Body ID**: Note this number! (usually 1 for first body)
   - **Tracking Algorithm**: Best quality with fewest markers
   - **Pivot Point**: Set to center of Tello (between markers)
     - Right-click rigid body → Set Pivot Point → Center of Markers

4. **Verify tracking**:
   - Move Tello around
   - Check that rigid body stays tracked (doesn't swap with other objects)
   - Verify position values update smoothly

### 3. Define Coordinate Frame (Important!)

**Set Tello orientation**:
1. Place Tello on ground facing **forward** in your lab coordinate system
2. Right-click rigid body → Orient To → Current Position
3. This sets forward direction (X-axis in OptiTrack)

**Coordinate mapping** (OptiTrack → Tello):
- OptiTrack +X → Tello Forward
- OptiTrack +Y → Tello Left
- OptiTrack +Z → Tello Up

## Testing Connection

### 1. Test OptiTrack Streaming

```bash
# Test with Motive on same computer
python test_optitrack.py

# Test with Motive on another computer (e.g., 192.168.1.100)
python test_optitrack.py --server 192.168.1.100

# Test specific rigid body ID (e.g., ID 2)
python test_optitrack.py --id 2 --server 192.168.1.100

# Extended test (30 seconds)
python test_optitrack.py --duration 30
```

**Expected output**:
```
==============================================================
OptiTrack Connection Test
==============================================================
Server IP: 192.168.1.100
Rigid Body ID: 1
Duration: 10 seconds

Starting NatNet client...
NatNet Client initialized
  Server: 192.168.1.100
  Multicast: 239.255.42.99:1511
NatNet Client started
Waiting for OptiTrack data...

Scanning for rigid bodies...
✓ Found 1 rigid body(ies): [1]

==============================================================
Live Data for Rigid Body 1
==============================================================
Move your Tello to see position/orientation updates...
Press Ctrl+C to stop

Position: [ 0.523, -0.234,  0.056] m  Yaw:  45.3°  Vel: [ 0.00,  0.01,  0.00] m/s  ✓ TRACKED
...
```

### 2. Common Test Results

**✅ Success** - You see:
- Rigid body detected
- Position updates smoothly as you move Tello
- Yaw angle changes when you rotate Tello
- "✓ TRACKED" indicator

**❌ No rigid bodies detected**:
- Check Motive is streaming (see Data Streaming pane)
- Verify "Broadcast Frame Data" is enabled
- Check firewall allows UDP 1511

**❌ Wrong rigid body ID**:
- Note the detected IDs from test output
- Use `--id X` flag with correct ID

**❌ Position not updating**:
- Rigid body might be occluded (not enough markers visible)
- Move to area with better camera coverage

## Running with MoCap

### Basic Flight Test (Hover)

```bash
# Hover for 15 seconds with OptiTrack
./run_real_drone.sh hybrid hover 15 --mocap --mocap-server 192.168.1.100 --mocap-id 1
```

**Command breakdown**:
- `hybrid`: Use Hybrid RL controller
- `hover`: Hover trajectory
- `15`: Duration in seconds
- `--mocap`: Enable motion capture
- `--mocap-server 192.168.1.100`: Motive computer IP
- `--mocap-id 1`: Rigid body ID in Motive

### Advanced Trajectories

```bash
# Circle trajectory (20 seconds)
./run_real_drone.sh hybrid circle 20 --mocap --mocap-server 192.168.1.100 --mocap-id 1

# Figure-8 trajectory
./run_real_drone.sh hybrid figure8 30 --mocap --mocap-server 192.168.1.100 --mocap-id 1
```

### MoCap Parameters

Full command structure:
```bash
./run_real_drone.sh [CONTROLLER] [TRAJECTORY] [DURATION] [OPTIONS]

Controllers: pid, hybrid
Trajectories: hover, circle, figure8, spiral, waypoint

MoCap Options:
  --mocap                  Enable motion capture
  --mocap-server IP        Motive computer IP (default: 127.0.0.1)
  --mocap-id ID           Rigid body ID (default: 1)
```

## Troubleshooting

### Problem: No rigid bodies detected

**Causes & Solutions**:

1. **Motive not streaming**:
   ```
   ✓ Open View → Data Streaming Pane
   ✓ Enable "Broadcast Frame Data"
   ✓ Check "Broadcasting frame data" appears at bottom
   ```

2. **Network issues**:
   ```bash
   # Verify connectivity
   ping 192.168.1.100  # (Motive computer IP)
   
   # Check firewall (on Motive computer)
   # Windows: Allow UDP 1510-1511 inbound
   # Mac: System Prefs → Security → Firewall → Allow Motive
   ```

3. **Wrong multicast configuration**:
   - Ensure Motive uses multicast: `239.255.42.99`
   - Data port: `1511`

### Problem: Rigid body tracking lost during flight

**Causes**:
- **Marker occlusion**: Not enough cameras see markers
  - Solution: Add more cameras or reposition existing ones
  
- **Markers fell off**: Vibration/wind dislodged markers
  - Solution: Use stronger adhesive, lighter markers
  
- **Wrong marker ID**: Another object has similar marker pattern
  - Solution: Use asymmetric marker pattern

### Problem: Position data has high noise/jitter

**Causes & Solutions**:

1. **Camera calibration poor**:
   - Re-calibrate cameras with wanding
   - Check calibration quality in Motive

2. **Markers too small/reflective**:
   - Use larger markers (10-14mm)
   - Clean markers (dust reduces reflection)

3. **Insufficient lighting**:
   - OptiTrack needs dark room
   - Cover windows, turn off overhead lights

### Problem: Coordinate system mismatch

**Symptoms**: Drone moves backward when commanded forward

**Solution**: 
1. Check rigid body orientation in Motive
2. Re-orient rigid body:
   - Place Tello facing forward
   - Right-click → Orient To → Current Position
3. Verify coordinate mapping in test_optitrack.py

### Problem: High latency (delayed position)

**Causes**:
- Network congestion
- Motive processing lag
- Too many tracked objects

**Solutions**:
- Use dedicated Gigabit network
- Reduce Motive framerate to 120 Hz (from 240 Hz)
- Disable marker streaming (only stream rigid bodies)

## Performance Expectations

### With OptiTrack MoCap:
- **Position accuracy**: Sub-millimeter (0.1-0.5mm typical)
- **Update rate**: 120-240 Hz (depending on Motive settings)
- **Latency**: 5-15ms (network + processing)
- **Tracking RMSE**: 0.1-0.2m (limited by Tello control, not MoCap)

### Without MoCap (Tello sensors only):
- **Position accuracy**: ±0.5m (visual positioning system)
- **Drift**: 1-2m over 60 seconds
- **Tracking RMSE**: 0.5-1.5m (highly variable)

## Best Practices

1. **Calibrate OptiTrack regularly** (weekly if used frequently)

2. **Test MoCap before each flight session**:
   ```bash
   python test_optitrack.py --server 192.168.1.100 --duration 5
   ```

3. **Check marker placement** before flight (ensure all visible)

4. **Start with short flights** (10-15 seconds) to verify tracking

5. **Monitor tracking quality** in Motive during flight

6. **Keep capture volume clear** of other reflective objects

7. **Use safety net** - MoCap improves control but doesn't prevent crashes!

## Summary Checklist

Before first MoCap flight:

- [ ] OptiTrack cameras calibrated
- [ ] Motive streaming enabled (Data Streaming Pane)
- [ ] Rigid body created with 4+ markers
- [ ] Rigid body ID noted (e.g., 1)
- [ ] Coordinate frame oriented correctly
- [ ] Network connectivity verified (ping test)
- [ ] Firewall configured (UDP 1510-1511)
- [ ] `test_optitrack.py` shows tracking data
- [ ] Tello batteries charged
- [ ] Flight area clear and safe

**Then run**:
```bash
./run_real_drone.sh hybrid hover 10 --mocap --mocap-server 192.168.1.100 --mocap-id 1
```

🎯 **Expected result**: Tello hovers stably at (0, 0, 1.0m) with <0.2m RMSE!
