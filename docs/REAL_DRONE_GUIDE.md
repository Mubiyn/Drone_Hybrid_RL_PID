# Real Drone Deployment Guide

## Hardware Requirements

- **DJI Tello/Tello EDU** drone
- Laptop with WiFi
- (Optional) Motion Capture System (OptiTrack, Vicon)
- Minimum 3m x 3m clear flying space

## Installation

### 1. Install DJI Tello Python Library

```bash
pip install djitellopy
```

### 2. Connect to Tello

1. Power on the Tello drone
2. Wait for WiFi network `TELLO-XXXXXX` to appear
3. Connect your laptop to the Tello's WiFi network
4. Test connection:

```bash
python -c "from djitellopy import Tello; t = Tello(); t.connect(); print(f'Battery: {t.get_battery()}%')"
```

## Running Controllers

### PID Controller (Baseline)

```bash
./run_real_drone.sh pid hover 10
```

### Hybrid RL Controller (Trained Model)

```bash
./run_real_drone.sh hybrid circle 15
```

### With Motion Capture

```bash
./run_real_drone.sh hybrid circle 15 --mocap
```

## Safety Checklist

Before each flight:

- [ ] Battery > 30%
- [ ] Clear flying space (3m x 3m minimum)
- [ ] Tello on flat, stable surface
- [ ] No obstacles above drone (2m clearance)
- [ ] Emergency stop ready (Ctrl+C or press `e`)
- [ ] Propellers in good condition
- [ ] Indoor flight only (Tello not suitable for outdoor wind)

## Command Reference

```bash
./run_real_drone.sh [CONTROLLER] [TRAJECTORY] [DURATION] [--mocap]
```

**Parameters:**
- `CONTROLLER`: `pid` or `hybrid`
- `TRAJECTORY`: `hover`, `circle`, `figure8`, `spiral`, `waypoint`
- `DURATION`: Flight time in seconds (recommended: 10-30s)
- `--mocap`: Enable motion capture system (optional)

## Trajectories

### Hover (Safest - Start Here!)
```bash
./run_real_drone.sh pid hover 10
```
Maintains position at 1m height.

### Circle
```bash
./run_real_drone.sh hybrid circle 20
```
0.8m radius circle at 1m height (scaled down for indoor).

### Figure-8
```bash
./run_real_drone.sh hybrid figure8 25
```
Lemniscate pattern.

## Troubleshooting

### Connection Issues

**Problem:** "Tello not responding"
```bash
# 1. Reconnect to Tello WiFi
# 2. Restart Tello drone
# 3. Check firewall isn't blocking UDP ports 8889, 8890, 11111
```

### Low Performance

**Problem:** Drone doesn't track trajectory well

```bash
# For PID: Tune gains in src/controllers/pid_controller.py
# For Hybrid: Retrain model with more conservative residual_scale
```

### Battery Drain

**Problem:** Battery drains quickly

```bash
# Normal for Tello (5-13 min flight time)
# Land at 20% battery minimum
# Keep spare batteries charged
```

### Crashes/Instability

**Problem:** Drone flips or crashes

```bash
# 1. IMMEDIATELY: Press Ctrl+C to land
# 2. Check propellers not damaged
# 3. Reduce control aggressiveness:
#    - Lower PID gains (kp in VelocityPIDController)
#    - Reduce residual_scale in run_tello.py
#    - Use slower trajectories (hover first)
```

## Motion Capture Setup (Advanced)

### OptiTrack/Vicon Configuration

1. Set up rigid body in motion capture software
2. Note the rigid body ID (default: 1)
3. Configure network settings in `src/real_drone/mocap_client.py`:

```python
mocap = NatNetClient(
    server_ip="192.168.1.100",    # Your MoCap computer IP
    multicast_ip="239.255.42.99",  # Default NatNet multicast
    data_port=1511
)
```

4. Run with `--mocap` flag

### Coordinate Frame Alignment

- MoCap usually gives position in meters
- Ensure Z-axis points up
- Tello's coordinate system: X=forward, Y=left, Z=up
- May need to transform coordinates in `TelloWrapper.get_obs()`

## Model Performance Expectations

Based on simulation results with 0.15N wind + ±30% mass randomization:

| Controller | Circle RMSE | Hover RMSE | Notes |
|-----------|-------------|------------|-------|
| PID | 1.04m | 0.31m | Crashes with strong disturbances |
| Hybrid RL | 1.26m | 0.18m | **73% better** on circle, stable flight |

**Real-world performance will differ** because:
- Indoor flight has less wind (easier than simulation)
- Tello has different dynamics than CF2X in simulation
- Sensor noise and delays not fully modeled
- MoCap provides better position estimates than simulation

## Tips for Best Results

1. **Start Conservative**
   - Begin with `hover` trajectory
   - Use short durations (10-15s)
   - Increase complexity gradually

2. **Indoor Flight**
   - Tello is designed for indoor use
   - Avoid ceiling fans, AC vents
   - Good lighting helps video feed

3. **Model Adaptation**
   - Trained models are for simulation (CF2X dynamics)
   - May need domain adaptation / fine-tuning for Tello
   - Consider collecting real-world data for retraining

4. **Control Frequency**
   - Current: 20 Hz (50ms loop)
   - Can increase to 30-50 Hz for better tracking
   - WiFi latency limits practical max frequency

## Advanced: Fine-Tuning for Tello

The trained Hybrid RL model was for **Crazyflie 2.X** dynamics. To adapt for Tello:

### Option 1: Transfer Learning (Recommended)

```bash
# Collect real Tello flight data with PID
# Fine-tune the model with collected data
# This preserves learned behaviors while adapting to new dynamics
```

### Option 2: Direct Deployment

```bash
# Use current model with conservative residual scaling
# Adjust in run_tello.py: residual_scale = 0.2 (instead of 0.3)
# Monitor performance and tune empirically
```

### Option 3: Retrain from Scratch

```bash
# Would require Tello simulation environment
# Or extensive real-world data collection (risky/expensive)
```

## Emergency Procedures

### During Flight

- **Ctrl+C**: Initiates landing sequence
- **Close laptop lid**: Tello will land automatically after WiFi disconnect (safety feature)
- **Power button on Tello**: Hold for emergency motor stop

### After Crash

1. Power off Tello immediately
2. Check for damage (propellers, frame, battery)
3. Do NOT fly with damaged propellers
4. Review logs to identify cause

## Data Collection

Flight logs are printed to console. To save:

```bash
./run_real_drone.sh hybrid circle 20 | tee flight_log_$(date +%Y%m%d_%H%M%S).txt
```

Logs include:
- Position vs target at 1Hz
- Tracking error
- Battery level
- Final statistics (mean/max error, RMSE)

## Next Steps

1. Test PID baseline on hover (safest)
2. Verify tracking performance
3. Test Hybrid RL on hover
4. Progress to circle trajectory
5. Collect performance data
6. Fine-tune if needed

## Support

For issues:
1. Check simulation results match expectations
2. Verify Tello firmware is updated
3. Test with basic djitellopy examples first
4. Review safety checklist

**Happy Flying! 🚁**
