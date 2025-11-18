#!/usr/bin/env python3
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary

import sys
import os

# FIXED: More robust path handling
try:
    from src.utils.trajectories import get_trajectory
except ImportError:
    # Alternative import path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    sys.path.insert(0, project_root)
    from src.utils.trajectories import get_trajectory


class DroneEnv(gym.Env):
    """Custom Gym environment for drone trajectory tasks with PPO training."""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, task_name, reward_fn, domain_randomization=True, gui=False):
        super().__init__()
        
        self.task_name = task_name
        self.reward_fn = reward_fn
        self.domain_randomization = domain_randomization
        self.gui = gui
        
        # Get trajectory
        self.trajectory = get_trajectory(task_name)
        self.trajectory_step = 0
        
        # Ground-start initialization (matching PID baseline)
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
            shape=(20,),
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
        self.max_steps = 500
        self.step_count = 0
        
    def reset(self, seed=None, options=None):
        """Reset environment and apply domain randomization."""
        super().reset(seed=seed)
        
        # Reset trajectory and step counter
        self.trajectory_step = 0
        self.step_count = 0
        
        # Reset PyBullet environment
        obs, info = self.env.reset(seed=seed)
        
        # Apply domain randomization if enabled
        if self.domain_randomization:
            self._apply_domain_randomization()
        else:
            # Reset to nominal conditions
            self.current_mass = self.base_mass
            self.wind_speed = 0.0
            self.motor_efficiency = np.ones(4)
            p.changeDynamics(self.env.DRONE_IDS[0], -1, mass=self.current_mass, 
                           physicsClientId=self.env.CLIENT)
        
        return obs[0].astype(np.float32), info
    
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
        
        # Action scaling
        thrust_scale = 0.1
        action_denorm = self.hover_rpm * (1 + thrust_scale * action)
        action_denorm = np.clip(action_denorm, 0.1 * self.hover_rpm, 1.5 * self.hover_rpm)
        
        # Apply motor efficiency
        action_modified = action_denorm * self.motor_efficiency
        
        # Apply wind disturbance
        if self.wind_speed > 0:
            # Use consistent wind direction during episode
            if not hasattr(self, 'wind_direction'):
                self.wind_direction = np.random.uniform(-np.pi, np.pi)
            wind_force = np.array([
                self.wind_speed * np.cos(self.wind_direction),
                self.wind_speed * np.sin(self.wind_direction),
                0.0
            ]) * self.current_mass
            p.applyExternalForce(self.env.DRONE_IDS[0], -1, wind_force, [0, 0, 0],
                               p.WORLD_FRAME, physicsClientId=self.env.CLIENT)
        
        # Step environment
        obs, _, terminated, truncated, info = self.env.step(np.array([action_modified]))
        current_obs = obs[0]
        
        # Get current state and target
        current_pos = current_obs[0:3]
        target_pos = self.trajectory[min(self.trajectory_step, len(self.trajectory)-1)]
        rpy = current_obs[7:10]  # Roll, pitch, yaw
        
        # Compute reward
        reward = self.reward_fn(current_pos, target_pos, current_obs, action)
        
        # Update trajectory step
        self.trajectory_step += 1
        
        # Termination conditions
        early_termination = False
        termination_penalty = 0
        
        # RELAXED termination conditions:
        if (abs(current_pos[0]) > 10.0 or abs(current_pos[1]) > 10.0 or  # Increased from 5.0
            current_pos[2] > 8.0 or current_pos[2] < 0.005):  # More lenient
            early_termination = True
            termination_penalty = -5.0  # Reduced penalty
        
        # RELAXED tilt limits:
        elif abs(rpy[0]) > 1.3 or abs(rpy[1]) > 1.3:  # Increased from 1.05 (~75 degrees)
            early_termination = True
            termination_penalty = -5.0  # Reduced penalty            
        if early_termination:
            reward += termination_penalty
        
        # Check if episode is done
        done = (self.trajectory_step >= len(self.trajectory) or 
                self.step_count >= self.max_steps or 
                terminated or truncated or early_termination)
        
        return current_obs.astype(np.float32), reward, done, False, info
    
    def render(self):
        """Render is handled by PyBullet GUI."""
        pass
    
    def close(self):
        """Close the environment."""
        self.env.close()