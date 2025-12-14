# Methodology: Hybrid RL-PID Drone Control

This document provides a comprehensive overview of the methodology, design decisions, and iterative development process behind the Hybrid RL-PID drone control system.

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Hybrid Control Architecture](#hybrid-control-architecture)
3. [Two-Phase Development Approach](#two-phase-development-approach)
4. [Domain Randomization Strategy](#domain-randomization-strategy)
5. [Training Methodology](#training-methodology)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Evolution of Methodology](#evolution-of-methodology)
8. [Lessons Learned](#lessons-learned)

---

## Problem Statement

### Motivation

Traditional PID controllers are:
- ✅ **Stable and predictable** for simple tasks
- ✅ **Easy to understand** and tune
- ❌ **Limited adaptability** to changing conditions
- ❌ **Suboptimal** for complex, dynamic trajectories
- ❌ **Require manual tuning** for each task

Reinforcement Learning offers:
- ✅ **Learning from experience** to optimize performance
- ✅ **Adaptation to perturbations**
- ✅ **Automatic policy optimization**
- ❌ **Sample inefficiency** (needs lots of data)
- ❌ **Instability during learning**
- ❌ **Sim-to-real transfer challenges**

### Our Solution: Hybrid RL-PID

Combine the strengths of both approaches:

```
Control Output = PID(state, target) + RL(state, target)
                 └─ Stability ─┘     └─ Optimization ─┘
```

**Key Insight**: Let PID provide baseline stability, let RL learn residual corrections to improve tracking.

---

## Hybrid Control Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Hybrid Controller                       │
│                                                              │
│  ┌──────────────┐                  ┌───────────────────┐   │
│  │              │                  │                   │   │
│  │     PID      │  RPM_pid        │    RL Policy      │   │
│  │  Controller  │  ───────────┐   │      (PPO)        │   │
│  │              │             │   │                   │   │
│  └──────────────┘             ├──►│  Residual         │   │
│         ▲                     │   │  Corrections      │   │
│         │                     │   │                   │   │
│         │  State + Target     │   └───────────────────┘   │
│         │                     │            │              │
│  ┌──────┴──────────────────┐  │            │ RPM_residual│
│  │                         │  │            │              │
│  │   Environment           │  │  ┌─────────▼────────┐    │
│  │   (Drone Simulation     │  └──┤   RPM_total =    │    │
│  │    or Real Hardware)    │     │   RPM_pid +      │    │
│  │                         │     │   residual_scale │    │
│  └─────────────────────────┘     │   × RPM_residual │    │
│                                   └──────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Components

#### 1. PID Controller

**Role**: Provide baseline stability and safety

**Implementation**: `src/controllers/pid_controller.py`

**Gains** (trajectory-specific):
- **Circle**: kp=0.8, max_vel=0.9
- **Figure8**: kp=0.4, max_vel=0.5 (requires smoother control)
- **Hover**: kp=0.6, max_vel=0.7
- **Spiral**: kp=0.8, max_vel=0.9
- **Waypoint**: kp=0.7, max_vel=0.8

**Output**: RPM commands for 4 motors (range: ~16000-65000 RPM)

#### 2. RL Policy (PPO)

**Role**: Learn residual corrections to improve PID performance

**Algorithm**: Proximal Policy Optimization (PPO)
- **Why PPO**: Sample-efficient, stable, on-policy
- **Framework**: Stable-Baselines3
- **Network**: MLP (64, 64 hidden units)
- **Activation**: tanh

**Observation Space** (20-dim):
- Position (x, y, z)
- Velocity (vx, vy, vz)
- Orientation (roll, pitch, yaw)
- Angular velocity (wx, wy, wz)
- Position error (Δx, Δy, Δz)
- Velocity error (Δvx, Δvy, Δvz)
- PID action (4 RPM commands)

**Action Space** (4-dim):
- Residual corrections for each motor: [-1, +1]
- Scaled by `residual_scale` parameter

**Reward Function**:
```python
reward = -position_error² - velocity_error² - action_smoothness_penalty
```

#### 3. Residual Scaling

**Critical Parameter**: `residual_scale`

Controls how much influence RL has over PID:

```python
total_rpm[i] = pid_rpm[i] + residual_scale * rl_action[i]
```

**Phase 1**: `residual_scale = 200` RPM
- Strong RL corrections
- Good for simulation where control is precise

**Phase 2**: `residual_scale = 100` RPM
- Gentler RL corrections
- Necessary for real Tello hardware stability

---

## Two-Phase Development Approach

### Why Two Phases?

Sim-to-real transfer is challenging. A progressive approach reduces risk:

1. **Phase 1 (Simulation)**: Validate approach in controlled environment
2. **Phase 2 (Real Hardware)**: Deploy with hardware-specific adaptations

### Phase 1: Simulation Validation

**Goal**: Prove hybrid approach works better than PID baseline

**Environment**: `HybridAviary` (gym-pybullet-drones)
- Drone model: CF2X (Crazyflie 2.x)
- Physics: PyBullet at 240 Hz
- Trajectories: Circle, Figure8, Hover, Spiral, Waypoint

**Configuration**:
```yaml
residual_scale: 200           # Strong RL influence
domain_randomization:
  mass_variation: ±20%        # 0.8x - 1.2x
  inertia_variation: ±20%
  wind_force: 0.05N max
```

**Training**:
- 500K timesteps per trajectory
- Domain randomization enabled
- VecNormalize for observation normalization
- Models saved to `models/hybrid_robust/`

**Key Results**:
- ✅ Hybrid outperforms PID on all dynamic trajectories
- ✅ Strong robustness to domain randomization
- ✅ +13% to +50% improvement depending on trajectory

### Phase 2: Real Hardware Deployment

**Goal**: Deploy to DJI Tello, validate sim-to-real transfer

**Hardware**: DJI Tello
- Weight: 80g
- Control: Limited (lower authority than CF2X)
- Sensors: Optical flow (positioning), barometer (altitude)
- Constraints: Indoor only, ~10m range

**Configuration** (adapted from Phase 1):
```yaml
residual_scale: 100           # Reduced for stability (was 200)
domain_randomization:
  mass_variation: ±30%        # Increased robustness (was ±20%)
  inertia_variation: ±30%
  wind_force: 0.15N max       # Stronger perturbations (was 0.05N)
```

**Why Configuration Changed**:

| Change | Reason |
|--------|--------|
| **Residual scale ↓** | Tello has limited control authority; lower residual prevents oscillations discovered during circle tests |
| **DR variation ↑** | Real world has more uncertainty than simulation; need stronger robustness |
| **Wind force ↑** | Real indoor air currents stronger than expected |

**Training**:
- Retrained models with new configuration
- Only successful trajectories: Circle, Hover, Spiral
- Figure8 and Square too aggressive for Tello capabilities
- Models saved to `logs/hybrid_tello_drone/*/rl_only_*/`

**Key Results**:
- ✅ Successful deployment on 3/5 trajectories
- ✅ Improved tracking over PID baseline
- ✅ Robust to wind disturbances
- ⚠️ Hardware limits exclude some trajectories

---

## Domain Randomization Strategy

### Purpose

Train policies robust to:
- Model uncertainties (mass, inertia)
- External disturbances (wind)
- Sensor noise
- Sim-to-real transfer gaps

### Implementation

Randomization applied **at each episode reset**:

```python
def _randomize_dynamics(self):
    # Randomize mass
    mass_scale = np.random.uniform(0.8, 1.2)  # Phase 1: ±20%
    new_mass = self.original_mass * mass_scale
    
    # Randomize inertia
    inertia_scale = np.random.uniform(0.8, 1.2)
    new_inertia = [i * inertia_scale for i in self.original_inertia]
    
    # Apply to simulation
    p.changeDynamics(self.DRONE_IDS[0], -1, 
                    mass=new_mass,
                    localInertiaDiagonal=new_inertia)

def _apply_wind(self):
    # Random wind 10% of steps
    if np.random.rand() < 0.1:
        wind_force = np.random.uniform(-0.05, 0.05, 3)  # Phase 1
        p.applyExternalForce(self.DRONE_IDS[0], -1, 
                            wind_force, [0,0,0], p.LINK_FRAME)
```

### Phase Differences

| Aspect | Phase 1 | Phase 2 | Rationale |
|--------|---------|---------|-----------|
| **Mass** | 0.8x - 1.2x | 0.7x - 1.3x | Real world has more variation |
| **Inertia** | 0.8x - 1.2x | 0.7x - 1.3x | Payload uncertainty higher in reality |
| **Wind** | 0.05N | 0.15N | Indoor air currents underestimated |

**Critical Learning**: Models must be tested with the **same DR parameters they were trained with**. Testing Phase 1 models (trained with ±20%) using Phase 2 DR (±30%) caused -104% performance degradation!

---

## Training Methodology

### Hyperparameters

**PPO Configuration** (Stable-Baselines3):
```python
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    tensorboard_log="./logs/"
)
```

**Training Loop**:
```python
total_timesteps = 500_000
eval_freq = 10_000
save_freq = 50_000

model.learn(
    total_timesteps=total_timesteps,
    callback=eval_callback
)
```

### Observation Normalization

**VecNormalize** used to standardize observations:
- Computes running mean and std of observations
- Normalizes: `obs_normalized = (obs - mean) / (std + epsilon)`
- Crucial for RL stability with heterogeneous observations
- Saved alongside model for inference

### Training Infrastructure

- **Compute**: GPU-accelerated (CUDA 11.7+) for PyTorch
- **Duration**: ~2-3 hours per trajectory (500K timesteps)
- **Monitoring**: TensorBoard for real-time training metrics
- **Checkpointing**: Save every 50K timesteps + best model

---

## Evaluation Metrics

### Simulation Metrics

1. **Tracking Error**:
   ```python
   error = np.linalg.norm(current_position - target_position)
   mean_error = np.mean(errors_over_episode)
   ```

2. **Control Smoothness**:
   ```python
   action_changes = np.diff(actions, axis=0)
   smoothness = np.mean(np.var(action_changes, axis=0))
   ```
   Lower is better (less jitter).

3. **Episode Reward**:
   Cumulative reward over episode (higher is better).

4. **Success Rate**:
   Percentage of episodes completing trajectory.

### Real Drone Metrics

1. **Position Tracking**:
   - Mean absolute error per axis
   - 3D Euclidean distance from target

2. **Trajectory Completion**:
   - Did drone complete full trajectory?
   - Number of waypoints reached

3. **Stability**:
   - Oscillation amplitude
   - Recovery from perturbations

4. **Battery Usage**:
   - Flight time achieved
   - Energy efficiency

---

## Evolution of Methodology

### What Worked

✅ **Hybrid architecture**: Combining PID stability with RL optimization
✅ **Domain randomization**: Crucial for robustness
✅ **Progressive deployment**: Sim validation before hardware
✅ **VecNormalize**: Stabilized training significantly
✅ **Trajectory-specific PID tuning**: Better baseline performance

### What Failed

❌ **Behavioral Cloning + RL (BC+RL)**: 
- Attempted to pretrain with expert demonstrations
- RL failed to improve beyond BC baseline
- Discarded in favor of pure RL-only approach
- Deleted `bc_rl_*` training runs from repository

❌ **Fixed PID gains**: 
- Initially used same gains for all trajectories
- Poor performance on figure8 (too aggressive)
- Solution: Trajectory-specific tuning

❌ **Testing mismatches**:
- Initially tested Phase 1 models with Phase 2 configuration
- Catastrophic -104% performance on figure8
- Solution: Phase-specific test scripts

### Iterative Refinements

**Iteration 1**: PID only
- Baseline performance established
- Identified limitations on dynamic trajectories

**Iteration 2**: RL only
- Unstable, took too long to learn basics
- Abandoned for hybrid approach

**Iteration 3**: Hybrid RL-PID (residual_scale=200)
- ✅ Great simulation results
- ❌ Oscillations on real Tello (circle trajectory)

**Iteration 4**: Hybrid RL-PID (residual_scale=100, DR ±30%)
- ✅ Stable on real hardware
- ✅ Robust to perturbations
- ✅ Final successful approach

---

## Lessons Learned

### Technical Lessons

1. **Configuration consistency is critical**:
   - Models must be tested with training configuration
   - Document all hyperparameters carefully
   - Version control configurations alongside models

2. **Sim-to-real requires adaptation**:
   - Simulation is not reality; expect differences
   - Progressive deployment reduces risk
   - Hardware constraints may limit what's possible

3. **Domain randomization is essential**:
   - But don't over-randomize beyond training regime
   - Match DR to target deployment environment
   - Stronger DR for real hardware than simulation

4. **Residual learning is powerful**:
   - Easier to learn corrections than full policy
   - Baseline controller provides safety net
   - Scaling parameter crucial for stability

### Methodological Lessons

1. **Document as you go**:
   - Easy to forget why decisions were made
   - Configuration evolution tells important story
   - Future you will thank current you

2. **Test incrementally**:
   - Don't skip simulation validation
   - Start with simple tasks before complex
   - Validate each component before integration

3. **Expect iteration**:
   - First approach rarely works perfectly
   - Budget time for debugging and refinement
   - Failed experiments teach valuable lessons

4. **Hardware has the final say**:
   - Simulation results don't guarantee real success
   - Tello limits excluded some trajectories
   - Work within hardware capabilities, not against them

---

## Future Work

Potential directions for extension:

1. **More powerful hardware**: Test on larger drones with higher control authority
2. **Outdoor deployment**: GPS navigation, wind robustness
3. **Multi-agent**: Cooperative control of drone swarms
4. **Vision-based control**: Replace trajectory tracking with visual servoing
5. **Transfer learning**: Adapt policies across different drone platforms
6. **Safety guarantees**: Formal verification of learned policies

---

## References

- **gym-pybullet-drones**: https://github.com/utiasDSL/gym-pybullet-drones
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
- **PPO Paper**: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- **Domain Randomization**: Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World" (2017)
- **djitellopy**: https://djitellopy.readthedocs.io/

---

*This methodology evolved through experimentation, failure, and iteration. The final approach represents lessons learned from multiple failed attempts and the systematic debugging process documented in our implementation plan.*
