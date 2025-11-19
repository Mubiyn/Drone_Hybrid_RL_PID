#!/usr/bin/env python3
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.utils.trajectories import get_trajectory
from src.utils.reward_funcs import get_reward_function


class DroneEnv(gym.Env):
    """Custom Gym environment for drone trajectory tasks with PPO training."""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, task_name, reward_fn=None, domain_randomization=True, gui=False):
        super().__init__()
        
        self.task_name = task_name
        self.domain_randomization = domain_randomization
        self.gui = gui
        self.TASK_IDS = {
                    'hover': 0,
                    'hover_extended': 1,
                    'circle': 2,
                    'figure8': 3,
                    'waypoint_delivery': 4,
                    'emergency_landing': 5
                }
        self.task_id = self.TASK_IDS.get(task_name, 0)

        
        # Select reward function
        if reward_fn is None:
            self.reward_fn = get_reward_function(task_name)
        else:
            self.reward_fn = reward_fn
        
        # Get trajectory
        self.trajectory = get_trajectory(task_name)
        self.trajectory_step = 0
        self.total_trajectory_steps = len(self.trajectory)
        
        # Waypoint bookkeeping (for waypoint_delivery)
        if self.task_name == "waypoint_delivery":
            # Must match your waypoint trajectory definition
            self.waypoints = np.array([
                [0, 0, 1],
                [2, 2, 1.5],
                [4, 0, 1],
                [2, -2, 1.5],
                [0, 0, 1],
            ])
            self.waypoint_index = 0
            self.total_waypoints = len(self.waypoints)
        else:
            self.waypoints = None
            self.waypoint_index = 0
            self.total_waypoints = 1
        
        # Ground-start initialization
        self.initial_pos = np.array([0.0, 0.0, 0.03])
        
        # Create environment
        self.env = CtrlAviary(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=np.array([self.initial_pos]),
            physics=Physics.PYB,
            gui=gui,
            record=False,
            pyb_freq=240,
            ctrl_freq=48
        )
        
        # Store hover RPM for action scaling
        self.hover_rpm = self.env.HOVER_RPM
        
        # Observation space: 20 elements
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(28,),
            dtype=np.float32
        )
        
        # Action space: 4D RPM commands normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )
        
        # Domain randomization parameters
        self.base_mass = self.env.M
        self.current_mass = self.base_mass
        self.wind_speed = 0.0
        self.motor_efficiency = np.ones(4)
        
        # Training parameters
        self.max_steps = min(500, self.total_trajectory_steps)  # tie to trajectory length
        self.step_count = 0
        
        # Wind direction state
        self.wind_direction = None
    def _build_observation(self, base_obs):
        current_pos = base_obs[0:3]
        target_pos = self.trajectory[min(self.trajectory_step, self.total_trajectory_steps - 1)]
        pos_error = target_pos - current_pos
        trajectory_progress = self.trajectory_step / max(1, self.total_trajectory_steps - 1)
        task_id_normalized = np.array([self.task_id], dtype=np.float32)

        aug_obs = np.concatenate([
            base_obs,                  # 20
            pos_error,                 # 3
            target_pos,                # 3
            np.array([trajectory_progress], dtype=np.float32),  # 1
            task_id_normalized
        ])
        return aug_obs.astype(np.float32)

    
    def reset(self, seed=None, options=None):
        """Reset environment and apply domain randomization."""
        super().reset(seed=seed)
        
        # Reset trajectory and step counter
        self.trajectory_step = 0
        self.step_count = 0
        
        # Reset waypoints
        if self.task_name == "waypoint_delivery" and self.waypoints is not None:
            self.waypoint_index = 0
        
        # Reset wind direction
        self.wind_direction = None
        
        # Reset PyBullet environment
        obs, info = self.env.reset(seed=seed)
        
        # Rebuild trajectory in case task-dependent params change later
        self.trajectory = get_trajectory(self.task_name)
        self.total_trajectory_steps = len(self.trajectory)
        self.max_steps = self.total_trajectory_steps
        
        # Apply domain randomization if enabled
        if self.domain_randomization:
            self._apply_domain_randomization()
        else:
            self.current_mass = self.base_mass
            self.wind_speed = 0.0
            self.motor_efficiency = np.ones(4)
            p.changeDynamics(self.env.DRONE_IDS[0], -1, mass=self.current_mass, 
                             physicsClientId=self.env.CLIENT)
        
        base_obs = obs[0].astype(np.float32)
        aug_obs = self._build_observation(base_obs)
        return aug_obs, info    
    
    def _apply_domain_randomization(self):
        """Apply domain randomization for robust training."""
        # Randomize mass (±20%)
        mass_factor = np.random.uniform(0.8, 1.2)
        self.current_mass = self.base_mass * mass_factor
        p.changeDynamics(self.env.DRONE_IDS[0], -1, mass=self.current_mass,
                         physicsClientId=self.env.CLIENT)
        
        # Randomize wind (0-1.5 m/s)
        self.wind_speed = np.random.uniform(0.0, 1.5)
        
        # Randomize motor efficiency (80-100%)
        self.motor_efficiency = np.random.uniform(0.8, 1.0, size=4)
    def step(self, action):
        """Execute one step in the environment."""
        self.step_count += 1

        # ---- 1) Optional warm-up: keep motors near hover for first few steps ----
        warmup_steps = 20
        if self.step_count <= warmup_steps:
            rpm_cmd = np.ones(4) * self.hover_rpm
        else:
            # ---- 2) Interpret action as [throttle, roll, pitch, yaw] ----
            # Clip for safety
            a = np.clip(action, -1.0, 1.0)

            # Small deviations around hover
            # Tune these scales if needed (they are intentionally small)
            throttle = self.hover_rpm + a[0] * 40.0   # ~±1% of hover
            roll     = a[1] * 30.0                    # differential roll term
            pitch    = a[2] * 30.0                    # differential pitch term
            yaw      = a[3] * 10.0                    # small yaw term

            # ---- 3) Motor mixing (X-configuration) ----
            # Order: [front-left, rear-left, rear-right, front-right]
            rpm_cmd = np.array([
                throttle - roll - pitch + yaw,   # motor 0
                throttle - roll + pitch - yaw,   # motor 1
                throttle + roll + pitch + yaw,   # motor 2
                throttle + roll - pitch - yaw    # motor 3
            ])

            # ---- 4) Global safety clamp around hover ----
            rpm_cmd = np.clip(
                rpm_cmd,
                0.8 * self.hover_rpm,
                1.3 * self.hover_rpm
            )

        # ---- 5) Apply motor efficiency ----
        action_modified = rpm_cmd * self.motor_efficiency

        # ---- 6) Apply wind disturbance (if any) ----
        if self.wind_speed > 0:
            if self.wind_direction is None:
                self.wind_direction = np.random.uniform(-np.pi, np.pi)
            wind_force = np.array([
                self.wind_speed * np.cos(self.wind_direction),
                self.wind_speed * np.sin(self.wind_direction),
                0.0
            ]) * self.current_mass
            p.applyExternalForce(
                self.env.DRONE_IDS[0], -1, wind_force, [0, 0, 0],
                p.WORLD_FRAME, physicsClientId=self.env.CLIENT
            )

        # ---- 7) Step environment ----
        obs, _, terminated, truncated, info = self.env.step(np.array([action_modified]))
        current_obs = obs[0]

        # Current state and target
        current_pos = current_obs[0:3]
        target_pos = self.trajectory[min(self.trajectory_step, self.total_trajectory_steps - 1)]
        rpy = current_obs[7:10]

        # Trajectory progress (0–1) for trajectory-style tasks
        trajectory_progress = self.trajectory_step / max(1, self.total_trajectory_steps - 1)

        # Waypoint tracking
        if self.task_name == "waypoint_delivery" and self.waypoints is not None:
            wp_pos = self.waypoints[self.waypoint_index]
            dist_to_wp = np.linalg.norm(current_pos - wp_pos)
            if dist_to_wp < 0.3 and self.waypoint_index < self.total_waypoints - 1:
                self.waypoint_index += 1

        # ---- 8) Compute reward with rich context ----
        reward = self.reward_fn(
            current_pos=current_pos,
            target_pos=target_pos,
            obs=current_obs,
            action=action,  # pass normalized action, not RPM
            step_count=self.step_count,
            trajectory_progress=trajectory_progress,
            waypoint_index=self.waypoint_index,
            total_waypoints=self.total_waypoints
        )

        # ---- 9) Update trajectory step ----
        self.trajectory_step += 1

        # ---- 10) Lenient termination conditions ----
        early_termination = False
        termination_penalty = 0.0

        if (abs(current_pos[0]) > 20.0 or abs(current_pos[1]) > 20.0 or
            current_pos[2] > 15.0 or current_pos[2] < 0.001):
            early_termination = True
            termination_penalty = -2.0
        elif abs(rpy[0]) > 1.48 or abs(rpy[1]) > 1.48:
            early_termination = True
            termination_penalty = -2.0

        if early_termination:
            reward += termination_penalty
            print(f"⚠️ Early termination at step {self.step_count}: "
                  f"pos=({current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}), "
                  f"rpy=({rpy[0]:.2f}, {rpy[1]:.2f})")

        done = (
            self.trajectory_step >= self.total_trajectory_steps or
            self.step_count >= self.max_steps or
            terminated or truncated or
            early_termination
        )

        aug_obs = self._build_observation(current_obs)
        return aug_obs, reward, done, False, info
            
    def render(self):
        pass
    
    def close(self):
        self.env.close()
