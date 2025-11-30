import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
import torch

class WayPointNavigationEnv(gym.Env):
    def __init__(self, hover_policy_path, gui=False):
        super().__init__()

        # Load low-level hover policy
        from src.utils.RL import PPO
        self.hover_policy = PPO(
            state_dim=64, action_dim=72, lr_actor=0, lr_critic=0,
            gamma=0.99, K_epochs=1, eps_clip=0.2
        )
        self.hover_policy.load(hover_policy_path)

        # High level action: desired position offset (dx, dy, dz)
        self.action_space = spaces.Box(
            low=np.array([-0.5, -0.5, -0.3]),
            high=np.array([0.5, 0.5, 0.3]),
            dtype=np.float32
        )

        # Observation: 12D drone state + relative target position (3) = 15
        high = np.ones(15) * np.inf
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Create drone env
        self.env = HoverAviary(
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            gui=gui
        )

        # Define rectangle waypoints
        self.waypoints = [
            np.array([0.5, 0.5, 1.2]),
            np.array([0.5, -0.5, 1.2]),
            np.array([-0.5, -0.5, 1.2]),
            np.array([-0.5, 0.5, 1.2])
        ]

        self.current_wp = 0
        self.max_steps = 2000
        self.step_count = 0

    def _extract_state(self, obs20):
        pos = obs20[0:3]
        rpy = obs20[7:10]
        vel = obs20[10:13]
        ang = obs20[13:16]
        return np.concatenate([pos, vel, rpy, ang], axis=0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        raw, _ = self.env.reset()
        state = self._extract_state(raw[0])
        self.current_wp = 0
        self.step_count = 0

        target = self.waypoints[self.current_wp]
        rel = target - state[:3]

        obs = np.concatenate([state, rel])
        return obs, {}

    def step(self, action):
        self.step_count += 1

        # High-level action modifies desired target
        desired_offset = action  # small shift
        target = self.waypoints[self.current_wp] + desired_offset

        # Get current state
        raw_obs, _, _, _, _ = self.env.step(np.zeros((1,4)))
        state = self._extract_state(raw_obs[0])

        # Compute desired position correction
        pos_error = target - state[:3]

        # Feed corrected state into hover controller
        hover_input = state  # 12D state
        rpm_action = self.hover_policy.select_action(hover_input)
        rpm_action = np.expand_dims(rpm_action, axis=0)

        raw_obs, reward_base, terminated, truncated, _ = self.env.step(rpm_action)
        state = self._extract_state(raw_obs[0])

        # Compute reward for reaching waypoint
        dist = np.linalg.norm(self.waypoints[self.current_wp] - state[:3])
        reward = -dist  

        # Switch to next waypoint if close enough
        if dist < 0.15:
            self.current_wp += 1
            reward += 5.0

            if self.current_wp >= len(self.waypoints):
                # completed square
                reward += 10.0

        done = terminated or truncated or self.current_wp >= len(self.waypoints)
        obs = np.concatenate([state, self.waypoints[self.current_wp] - state[:3]])

        return obs, reward, done, False, {}
