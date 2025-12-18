# System Architecture

This document explains the technical architecture of the Hybrid RL-PID drone control system.

---

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│              Hybrid RL-PID Control System               │
└─────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
   ┌────▼────┐                          ┌────▼────┐
   │ Phase 1 │                          │ Phase 2 │
   │  Sim    │                          │Hardware │
   └────┬────┘                          └────┬────┘
        │                                     │
        │                                     │
   PyBullet                              DJI Tello
   CF2X Sim                              Real Drone
```

---

## Component Architecture

### 1. Control Architecture

```
Input: Target Position & Current State
           │
           ▼
    ┌─────────────────┐
    │  Observation    │ ◄── 12-dim state vector
    │   Processing    │     [pos, quat, vel, ang_vel]
    └────────┬────────┘
             │
    ┌────────▼─────────┐
    │  Error Vector    │ ◄── Position & velocity errors
    │   Computation    │     (3 + 3 = 6 dims)
    └────────┬─────────┘
             │
      ┌──────┴──────┐
      │             │
┌─────▼─────┐  ┌───▼────┐
│    PID    │  │   RL   │
│ Baseline  │  │ Policy │
│  Control  │  │ (PPO)  │
└─────┬─────┘  └───┬────┘
      │            │
      │  4-dim     │ 4-dim
      │  velocity  │ residual
      │            │
      └─────┬──────┘
            │
      ┌─────▼──────┐
      │   Hybrid   │
      │ Combination│ ◄── action = pid + α * rl_residual
      └─────┬──────┘
            │
         Output: 4-dim velocity command [vx, vy, vz, yaw]
            │
            ▼
      ┌─────────────┐
      │  Low-Level  │
      │  Attitude   │
      │  Control    │ (Firmware)
      └──────┬──────┘
             │
             ▼
        Motor Commands
```

---

## Key Components

### A. PID Controller (`src/controllers/pid_controller.py`)

**Purpose**: Provides stable baseline control

**Algorithm**: Proportional control with velocity saturation

```python
class VelocityPIDController:
    def __init__(self, kp=0.4, max_vel=0.5):
        self.kp = kp          # Proportional gain
        self.max_vel = max_vel # Velocity limit
    
    def compute_control(self, obs, target_pos):
        # Extract current position
        current_pos = obs[0:3]
        
        # Compute position error
        pos_error = target_pos - current_pos
        
        # Proportional control
        vel_command = self.kp * pos_error
        
        # Saturate velocity
        vel_norm = np.linalg.norm(vel_command)
        if vel_norm > self.max_vel:
            vel_command = vel_command / vel_norm * self.max_vel
        
        return [vel_command[0], vel_command[1], vel_command[2], 0.0]
```

**Gains**:
- Simulation: `kp=0.8`, `max_vel=1.0`
- Hardware: `kp=0.4`, `max_vel=0.5` (more conservative)

---

### B. RL Policy (Stable-Baselines3 PPO)

**Purpose**: Learn residual corrections to PID

**Network Architecture**:
```
Input: 18-dim observation
  ├── 12-dim state [pos, quat, vel, ang_vel]
  ├── 3-dim position error
  └── 3-dim velocity error

Hidden Layers:
  ├── Dense(256) + ReLU
  └── Dense(256) + ReLU

Output: 4-dim residual [vx_res, vy_res, vz_res, yaw_res]
```

**Training Algorithm**: PPO (Proximal Policy Optimization)
- Learning rate: 3e-4
- Batch size: 64
- Total timesteps: 500,000
- Network: [256, 256] MLP

---

### C. Hybrid Combiner

**Phase 1 (Simulation)**:
```python
residual_scale = 200  # RPM units

action = pid_baseline + (rl_residual * residual_scale)
```

**Phase 2 (Hardware)**:
```python
residual_scale = 0.1  # Velocity units (reduced for stability)

action_vel = pid_vel.copy()
action_vel[0:3] += rl_action[0:3] * residual_scale  # xyz only
# Keep PID's yaw control
```

---

## Environment Architecture

### Training Environment (`src/envs/HybridAviary.py`)

```python
class HybridAviary(BaseRLAviary):
    def __init__(self, ...):
        # Observation space: 18 dims
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(18,)
        )
        
        # Action space: 4-dim residuals
        self.action_space = spaces.Box(
            low=-1, 
            high=1, 
            shape=(4,)
        )
    
    def _computeObs(self):
        # 12-dim state + 3-dim pos_error + 3-dim vel_error
        return np.concatenate([state, pos_err, vel_err])
    
    def _computeReward(self):
        # Negative position error (closer = higher reward)
        return -np.linalg.norm(pos_error)
    
    def step(self, action):
        # Combine PID + RL residual
        pid_action = self.pid.compute_control(obs, target)
        hybrid_action = pid_action + action * residual_scale
        
        # Apply to drone
        return self._apply_action(hybrid_action)
```

---

## Domain Randomization

### Phase 1 (Simulation Validation)
```python
mass_multiplier = np.random.uniform(0.8, 1.2)      # ±20%
inertia_multiplier = np.random.uniform(0.8, 1.2)   # ±20%
wind_force = np.random.uniform(-0.05, 0.05, 3)     # ±0.05N
```

### Phase 2 (Hardware Transfer)
```python
mass_multiplier = np.random.uniform(0.7, 1.3)      # ±30%
inertia_multiplier = np.random.uniform(0.7, 1.3)   # ±30%
wind_force = np.random.uniform(-0.15, 0.15, 3)     # ±0.15N
```

**Why Increased DR?**
- Real world has more uncertainty than simulation
- Better generalization to varying conditions
- Improved robustness to wind and sensor noise

---

## Data Flow

### Training Loop

```
1. Reset environment with randomized parameters
   ↓
2. Generate reference trajectory
   ↓
3. For each timestep:
   a. Get observation (18-dim)
   b. Compute PID baseline
   c. RL policy predicts residual
   d. Combine: action = PID + α * residual
   e. Apply action to simulation
   f. Compute reward (negative position error)
   g. Store transition in buffer
   ↓
4. After N steps, update policy (PPO)
   ↓
5. Repeat until convergence
```

### Inference (Deployment)

```
1. Initialize drone (real or simulated)
   ↓
2. Load trained RL model
   ↓
3. Control loop (20 Hz):
   a. Get current state from drone
   b. Compute target from trajectory
   c. Build 18-dim observation
   d. PID computes baseline velocity
   e. RL predicts residual
   f. Combine and send to drone
   ↓
4. Repeat until trajectory complete
```

---

## File Organization

### Core Source Files

```
src/
├── controllers/
│   ├── pid_controller.py          # PID implementation
│   └── hybrid_controller.py       # Hybrid combination logic
│
├── envs/
│   ├── HybridAviary.py            # Training environment
│   └── BaseTrackAviary.py         # Base tracking environment
│
├── training/
│   └── train_hybrid.py            # Training pipeline
│
├── testing/
│   └── test_rl.py                 # Evaluation scripts
│
└── hardware/
    ├── TelloWrapper.py            # Tello interface
    ├── run_tello.py               # Hardware deployment
    └── mocap_client.py            # Motion capture integration
```

---

## Communication Flow (Hardware)

```
┌──────────────┐         UDP           ┌──────────────┐
│   Computer   │◄─────────────────────►│  DJI Tello   │
│              │    8889 (commands)    │              │
│  Python      │                       │  Onboard     │
│  Controller  │    8890 (state)       │  Computer    │
└──────┬───────┘                       └──────────────┘
       │
       │ (Optional)
       ▼
┌──────────────┐      Ethernet/WiFi    ┌──────────────┐
│   MoCap PC   │◄─────────────────────►│  OptiTrack   │
│              │    NatNet protocol    │  Cameras     │
└──────────────┘                       └──────────────┘
```

**Tello Communication**:
- Command port: 8889 (send velocity commands)
- State port: 8890 (receive telemetry)
- Video port: 11111 (camera stream, not used)

**MoCap (Optional)**:
- NatNet protocol over LAN
- Provides high-accuracy 6-DOF pose (mm precision)
- Used for ground truth validation

---

## Model Architecture

### PPO Policy Network

```
┌─────────────────────┐
│  Input: 18-dim obs  │
└──────────┬──────────┘
           │
    ┌──────▼──────┐
    │ Dense(256)  │
    │   + ReLU    │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │ Dense(256)  │
    │   + ReLU    │
    └──────┬──────┘
           │
      ┌────┴────┐
      │         │
┌─────▼─────┐  ┌▼──────┐
│  Policy   │  │ Value │
│  Head     │  │ Head  │
│ Dense(4)  │  │Dense(1)│
│  + Tanh   │  │        │
└─────┬─────┘  └───┬────┘
      │            │
      │            │
   Actions      Value
  (residuals)  Estimate
```

**Parameters**:
- Total params: ~197K
- Input: 18 → Hidden: 256 → 256 → Output: 4
- Activation: ReLU (hidden), Tanh (output)
- Optimizer: Adam (lr=3e-4)

---

## Trajectory Generation

```python
class TrajectoryGenerator:
    def get_target(self, t):
        """Get target position at time t"""
        
        if trajectory == 'hover':
            return [0, 0, 1]  # Static position
        
        elif trajectory == 'circle':
            x = radius * np.cos(2*pi*t/period)
            y = radius * np.sin(2*pi*t/period)
            z = height
            return [x, y, z]
        
        elif trajectory == 'spiral':
            angle = 2*pi*t/period
            r = radius * (t/period)  # Growing radius
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            z = height + 0.5 * (t/period)  # Rising
            return [x, y, z]
        
        # ... (figure8, waypoint, etc.)
```

**Trajectories**:
- Hover: Static position hold
- Circle: Constant radius circular motion
- Spiral: Expanding circular motion with climb
- Figure8: Lemniscate curve
- Waypoint: Multi-point navigation

---

## Coordinate Systems

### Simulation (PyBullet)
- **Origin**: World frame (0, 0, 0)
- **X-axis**: Forward
- **Y-axis**: Right  
- **Z-axis**: Up (gravity: -9.81 m/s²)
- **Orientation**: Quaternion [x, y, z, w]

### Hardware (Tello)
- **Origin**: Takeoff position
- **X-axis**: Forward (drone front)
- **Y-axis**: Right
- **Z-axis**: Up
- **Orientation**: Inferred from velocity

---

## Performance Considerations

### Training Efficiency
- **CPU Training**: ~1-2 hours per trajectory (500K steps)
- **GPU Training**: ~20-30 minutes per trajectory
- **Parallelization**: 4-8 environments recommended
- **Memory**: ~4GB RAM per environment

### Inference Speed
- **Simulation**: 240 Hz (real-time)
- **Hardware**: 20 Hz (Tello UDP latency)
- **Model latency**: <5ms per prediction
- **Total latency**: ~50ms (hardware limited)

---

## Safety Features

### Simulation
- Automatic episode termination on crash
- Position limits: ±5m
- Velocity limits: ±2 m/s
- Episode timeout: 15 seconds

### Hardware
- Battery check before takeoff (>50%)
- Emergency land on connection loss
- Velocity saturation (max 0.5 m/s)
- Manual abort (Ctrl+C)
- Automatic landing after trajectory

---

## Next Steps

- **[Methodology](03_methodology.md)** - Training methodology and approach
- **[Hardware Setup](06_hardware_setup.md)** - Deploy to real drone
- **[Results](05_results.md)** - Experimental analysis and findings

---

**Questions?** Open an [issue on GitHub](https://github.com/Mubiyn/Drone_Hybrid_RL_PID/issues)
