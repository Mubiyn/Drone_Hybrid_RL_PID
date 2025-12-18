# Video Demonstrations

This directory contains video demonstrations of the Hybrid RL-PID drone control system in action.

---

##  Real Drone Flight Videos (DJI Tello Hardware)

 **Full Flight Recordings**: [Google Drive Folder](https://drive.google.com/drive/folders/1SYW4yN6jDDRTXFW3YVjBrWCday0e9ai-?usp=sharing) 

**Includes**:
- Hover trajectory with wind perturbations (PID vs Hybrid comparison)
- Spiral trajectory with wind perturbations (PID vs Hybrid comparison)
- Multiple test runs demonstrating repeatability
- Side-by-side performance comparisons

**These are the most important videos** - they show real hardware validation of the hybrid controller!

---

##  Simulation Videos (PyBullet)

All simulation videos show trajectory tracking in the gym-pybullet-drones environment.

### Hybrid RL-PID Controller

#### Circle Trajectory

<video src="https://github.com/Mubiyn/Drone_Hybrid_RL_PID/raw/main/results/videos/hybrid/circle.mp4" controls></video>

*Hybrid controller tracking a circular trajectory* • [Download](hybrid/circle.mp4)

#### Figure-8 Trajectory

<video src="https://github.com/Mubiyn/Drone_Hybrid_RL_PID/raw/main/results/videos/hybrid/figure8.mp4" controls></video>

*Hybrid controller tracking a figure-8 pattern* • [Download](hybrid/figure8.mp4)

#### Hover Trajectory

<video src="https://github.com/Mubiyn/Drone_Hybrid_RL_PID/raw/main/results/videos/hybrid/hover.mp4" controls></video>

*Hybrid controller maintaining stationary hover position* • [Download](hybrid/hover.mp4)

#### Waypoint Trajectory

<video src="https://github.com/Mubiyn/Drone_Hybrid_RL_PID/raw/main/results/videos/hybrid/waypoint.mp4" controls></video>

*Hybrid controller navigating through waypoints* • [Download](hybrid/waypoint.mp4)

---

### PID Baseline Controller

<details>
<summary><b>Click to view PID baseline videos</b></summary>

#### Circle Trajectory

<video src="https://github.com/Mubiyn/Drone_Hybrid_RL_PID/raw/main/results/videos/pid/circle.mp4" controls></video>

*PID baseline tracking a circular trajectory* • [Download](pid/circle.mp4)

#### Figure-8 Trajectory

<video src="https://github.com/Mubiyn/Drone_Hybrid_RL_PID/raw/main/results/videos/pid/figure8.mp4" controls></video>

*PID baseline tracking a figure-8 pattern* • [Download](pid/figure8.mp4)

#### Hover Trajectory

<video src="https://github.com/Mubiyn/Drone_Hybrid_RL_PID/raw/main/results/videos/pid/hover.mp4" controls></video>

*PID baseline maintaining stationary hover position* • [Download](pid/hover.mp4)

#### Waypoint Trajectory

<video src="https://github.com/Mubiyn/Drone_Hybrid_RL_PID/raw/main/results/videos/pid/waypoint.mp4" controls></video>

*PID baseline navigating through waypoints* • [Download](pid/waypoint.mp4)

</details>

---

## Directory Structure

```
results/videos/
├── README.md          # This file
├── hybrid/            # Hybrid RL-PID controller videos
│   ├── circle.mp4     (661 KB)
│   ├── figure8.mp4    (653 KB)
│   ├── hover.mp4      (598 KB)
│   └── waypoint.mp4   (617 KB)
└── pid/               # PID baseline controller videos
    ├── circle.mp4     (656 KB)
    ├── figure8.mp4    (648 KB)
    ├── hover.mp4      (136 KB)
    └── waypoint.mp4   (615 KB)
```

---

##  Key Observations from Videos

### Hybrid RL-PID Advantages (Visible in Videos)

1. **Smoother Trajectory Tracking**: Watch how the hybrid controller follows curves more smoothly on circle and figure-8
2. **Better Wind Rejection**: Real drone videos show tighter position hold during wind disturbances
3. **Reduced Oscillations**: Less wobbling compared to pure PID, especially on dynamic trajectories
4. **Faster Convergence**: Hybrid reaches target positions more quickly

### PID Baseline Characteristics

1. **Reliable but Conservative**: Stable performance but more oscillation
2. **Good for Simple Tasks**: Competitive on hover (stationary task)
3. **Struggles with Dynamics**: More tracking error on circle and spiral
4. **Wind Sensitivity**: Real drone videos show larger deviations during perturbations

---

##  Related Documentation

- **[Results Summary](../../docs/05_results.md)** - Detailed performance metrics and analysis
- **[Figures Gallery](../figures/)** - Plots and visualizations
- **[Main README](../../README.md)** - Project overview

---

**[ Back to Results](../README.md)** | **[ Documentation](../../docs/README.md)** | **[ Home](../../README.md)**
