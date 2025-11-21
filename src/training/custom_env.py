#!/usr/bin/env python3
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from src.utils.trajectories import get_trajectory


class DroneTaskEnv(gym.Env):
    """Custom Gym environment for drone trajectory tasks with PPO training."""

    metadata = {"render_modes": ["human"]}

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
            ctrl_freq=48,
        )

        # Store hover RPM for action scaling
        self.hover_rpm = self.env.HOVER_RPM

        # Observation space: 20D kinematic state
        # [pos(3), quat(4), rpy(3), vel(3), ang_vel(3), last_action(4)]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )

        # Action space: 4D RPM commands normalized to [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Domain randomization parameters
        self.base_mass = self.env.M
        self.current_mass = self.base_mass
        self.wind_speed = 0.0
        self.motor_efficiency = np.ones(4)
        self.wind_direction = 0.0

    def reset(self, seed=None, options=None):
        """Reset environment and apply domain randomization."""
        super().reset(seed=seed)

        # Reset trajectory
        self.trajectory_step = 0

        # Reset PyBullet environment
        obs, info = self.env.reset()

        # Apply domain randomization if enabled
        if self.domain_randomization:
            self._apply_domain_randomization()
        else:
            # Reset to nominal conditions
            self.current_mass = self.base_mass
            self.wind_speed = 0.0
            self.motor_efficiency = np.ones(4)
            self.wind_direction = 0.0
            p.changeDynamics(
                self.env.DRONE_IDS[0],
                -1,
                mass=self.current_mass,
                physicsClientId=self.env.CLIENT,
            )

        return obs[0].astype(np.float32), info

    def _apply_domain_randomization(self):
        """Apply domain randomization for robust training."""
        # Randomize mass (±30%)
        mass_factor = np.random.uniform(0.7, 1.3)
        self.current_mass = self.base_mass * mass_factor
        p.changeDynamics(
            self.env.DRONE_IDS[0],
            -1,
            mass=self.current_mass,
            physicsClientId=self.env.CLIENT,
        )
        self.env.M = self.current_mass
        self.env.GRAVITY = self.env.G * self.current_mass

        # Randomize wind (0-2 m/s)
        self.wind_speed = np.random.uniform(0.0, 2.0)
        self.wind_direction = np.random.uniform(-np.pi, np.pi)

        # Randomize motor efficiency (70-100%)
        self.motor_efficiency = np.random.uniform(0.7, 1.0, size=4)

    def step(self, action):
        """Execute one step in the environment."""
        # Denormalize action from [-1, 1] to RPM, symmetrically around HOVER_RPM.
        # This makes the action space more intuitive for the agent.
        # A positive action increases RPMs above hover, a negative action decreases them.
        # We use half of the available RPM range for scaling.
        action_range = (self.env.MAX_RPM - self.env.MIN_RPM) / 2
        action_denorm = self.hover_rpm + action * action_range
        action_denorm = np.clip(action_denorm, 0, self.env.MAX_RPM)

        # Apply motor efficiency (domain randomization)
        action_modified = action_denorm * self.motor_efficiency

        # Apply wind disturbance
        if self.wind_speed > 0:
            wind_force = np.array(
                [
                    self.wind_speed * np.cos(self.wind_direction),
                    self.wind_speed * np.sin(self.wind_direction),
                    0.0,
                ]
            )
            p.applyExternalForce(
                self.env.DRONE_IDS[0],
                -1,
                wind_force,
                [0, 0, 0],
                p.WORLD_FRAME,
                physicsClientId=self.env.CLIENT,
            )

        # Step environment
        obs, _, terminated, truncated, info = self.env.step(np.array([action_modified]))

        # Get current state and target
        current_pos = obs[0][0:3]
        target_pos = self.trajectory[
            min(self.trajectory_step, len(self.trajectory) - 1)
        ]
        rpy = obs[0][7:10]  # Roll, pitch, yaw

        # Compute reward
        reward = self.reward_fn(current_pos, target_pos, obs[0], action)

        # Update trajectory step
        self.trajectory_step += 1

        # Early termination conditions (prevent catastrophic failures)
        early_termination = False

        # Terminate if drone goes very far from origin (safety bounds)
        if abs(current_pos[0]) > 3.0 or abs(current_pos[1]) > 3.0:
            early_termination = True
            reward = -10.0  # Large penalty for going out of bounds

        # Terminate if drone flies dangerously high or crashes into ground
        if current_pos[2] > 3.0 or current_pos[2] < 0.015:
            early_termination = True
            reward = -10.0  # Large penalty for crash/flyaway

        # Terminate if drone tilts dangerously (>75 degrees - near flip)
        if abs(rpy[0]) > 1.3 or abs(rpy[1]) > 1.3:  # ~75 degrees
            early_termination = True
            reward = -10.0  # Large penalty for extreme tilt

        # Check if episode is done
        done = (
            self.trajectory_step >= len(self.trajectory)
            or terminated
            or truncated
            or early_termination
        )

        return obs[0].astype(np.float32), reward, done, False, info

    def render(self):
        """Render is handled by PyBullet GUI."""
        pass

    def close(self):
        """Close the environment."""
        self.env.close()
