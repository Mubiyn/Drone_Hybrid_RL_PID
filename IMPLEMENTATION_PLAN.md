# Implementation Plan: Hybrid RL and Residual PID for Quadrotor Trajectory Tracking

## 0. Prerequisites
**Environment:**
*   All commands must be run within the `drone-rl-pid` conda environment.
*   Activate with: `conda activate drone-rl-pid`

## 1. Project Overview
**Goal:** Develop and compare three control approaches for a quadrotor:
1.  **Classical PID**: Standard cascade controller.
2.  **Pure RL (PPO)**: End-to-end Reinforcement Learning.
3.  **Hybrid (Residual)**: PID + RL, where the RL agent learns a residual correction to compensate for errors and disturbances.

**Key Feature:** Robustness will be achieved via **Domain Randomization** (varying mass, inertia, wind) and tested on a real DJI Tello drone.

**Tasks:** The controllers will be evaluated on 5 trajectory tasks:
1.  **Hover**: Maintain a fixed position.
2.  **Circle**: Track a circular path.
3.  **Figure8**: Track a figure-8 path.
4.  **Spiral**: Track an ascending spiral path.
5.  **Waypoint**: Navigate through a sequence of 3D points.

## 2. Architecture Strategy
To satisfy both the rigorous research requirements and the hardware limitations of the DJI Tello, we will implement two variations of the Hybrid controller:

### A. Research Architecture (Simulation)
*   **Control Type:** Low-level Motor RPM.
*   **Mechanism:** 
    $$RPM_{total} = RPM_{PID} + \lambda \cdot RPM_{RL}$$
*   **Why:** Allows the RL agent to compensate for complex dynamics like motor failure or unbalanced mass.

### B. Deployment Architecture (DJI Tello)
*   **Control Type:** High-level Velocity (Vx, Vy, Vz, Yaw_rate).
*   **Mechanism:** 
    $$Vel_{total} = Vel_{PID} + \lambda \cdot Vel_{RL}$$
*   **Why:** The Tello SDK does not allow direct motor RPM control.

## 3. Implementation Phases

### Phase 1: Simulation Environment & PID Baseline
**Objective:** Establish a working simulation and measure baseline PID performance.
*   **Tasks:**
    1.  Create `src/utils/trajectories.py`: Helper functions to generate target waypoints for all 5 tasks.
    2.  Create `src/envs/BaseTrackAviary.py`: A custom Gym environment inheriting from `BaseAviary` that implements the trajectory tracking logic.
    3.  Implement `src/controllers/PIDController.py`: A standard PID controller tuned for the simulation.
    4.  **Milestone:** Drone flies stable trajectories in simulation using PID.

### Phase 2: Pure RL (PPO) Implementation
**Objective:** Train a PPO agent to solve the tracking task from scratch.
*   **Tasks:**
    1.  Setup `src/training/train_ppo.py` using `stable-baselines3`.
    2.  Define observation space (Position Error, Velocity Error, Rotation Matrix).
    3.  Define reward function (penalize distance from target).
    4.  **Milestone:** PPO agent successfully tracks the trajectories.

### Phase 3: Hybrid (Residual) Implementation
**Objective:** Implement the core research contribution.
*   **Tasks:**
    1.  Create `src/envs/HybridAviary.py`: An environment that runs the PID controller internally and accepts "Residual RPM" actions from the agent.
    2.  Train the Hybrid agent.
    3.  **Milestone:** Hybrid agent tracks trajectories.

### Phase 4: Domain Randomization & Robustness
**Objective:** Make the controller robust to "Out-of-Distribution" dynamics.
*   **Tasks:**
    1.  Modify `HybridAviary.py` to include `randomize_dynamics()`:
        *   Random Mass ($\pm 20\%$)
        *   Random Inertia
        *   Wind Disturbances (External forces)
    2.  Retrain/Finetune the Hybrid agent on this randomized environment.
    3.  **Evaluation:** Compare PID vs. Hybrid under heavy wind/mass changes.

### Phase 5: Sim-to-Real Transfer (DJI Tello)
**Objective:** Deploy the robust logic to real hardware.
*   **Tasks:**
    1.  Create `src/real_drone/TelloWrapper.py`: Interface with `djitellopy`.
    2.  Adapt the Hybrid logic to "Velocity Control" mode.
    3.  **Demo:** Fly Tello with attached weights (mass disturbance) to demonstrate RL compensation.

## 4. Proposed File Structure
```text
src/
├── controllers/
│   ├── PIDController.py       # Classical PID
│   └── HybridController.py    # Wrapper for PID + Loaded PPO Model
├── envs/
│   ├── BaseTrackAviary.py     # Base class for tracking tasks
│   └── HybridAviary.py        # Env with internal PID & Residual Action
├── training/
│   ├── train_ppo.py           # Training script
│   └── configs.py             # Hyperparameters
├── testing/
│   ├── eval_simulation.py     # Compare PID/PPO/Hybrid in Sim
│   └── eval_robustness.py     # Test with wind/mass variations
├── real_drone/
│   └── tello_handler.py       # DJI Tello interface
└── utils/
    └── trajectories.py        # Trajectory generation logic
```
