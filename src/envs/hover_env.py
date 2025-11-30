import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p   # noqa: F401  # often needed implicitly by gym_pybullet_drones

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


class HoverEnv(gym.Env):
    """Minimal and stable Hover environment designed for PPO training."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, gui: bool = False):
        super().__init__()

        self.gui = gui

        # ---------- INTERNAL PYBULLET ENV ----------
        self.env = CtrlAviary(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=np.array([[0.0, 0.0, 0.05]]),
            physics=Physics.PYB,
            gui=gui,
            record=False,
            pyb_freq=240,   # physics Hz
            ctrl_freq=24,   # control Hz
        )

        # CF2X hover RPM
        self.hover_rpm = float(self.env.HOVER_RPM)

        # ---------- OBSERVATION SPACE ----------
        # [px, py, pz, vx, vy, vz, roll, pitch, yaw, wx, wy, wz] = 12D
        high = np.full(12, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32,
        )

        # ---------- ACTION SPACE ----------
        # 4 motors, normalized [-1, 1] → scaled around hover RPM
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # Episode length in control steps
        self.step_count = 0
        self.max_steps = 800

        # Hover target
        self.target_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # ==========================================================
    #                      UTILITIES
    # ==========================================================
    def _extract_state(self, raw_obs: np.ndarray) -> np.ndarray:
        """
        Convert underlying 20D state → 12D state:
        pos(3), vel(3), rpy(3), ang_vel(3)
        """
        # CtrlAviary obs layout (for CF2X) is usually:
        # [x, y, z, qw, qx, qy, qz, roll, pitch, yaw,
        #  vx, vy, vz, wx, wy, wz, ...]
        pos = raw_obs[0:3]
        rpy = raw_obs[7:10]
        vel = raw_obs[10:13]
        ang_vel = raw_obs[13:16]

        state = np.concatenate([pos, vel, rpy, ang_vel]).astype(np.float32)
        return state

    def _compute_reward_and_done(self, obs: np.ndarray):
        """
        Reward shaping & termination logic.
        """
        pos = obs[0:3]
        vel = obs[3:6]
        rpy = obs[6:9]
        ang_vel = obs[9:12]

        # Errors
        pos_error = pos - self.target_pos
        z_error = pos[2] - self.target_pos[2]
        xy_error_norm = np.linalg.norm(pos[0:2])
        vel_norm = np.linalg.norm(vel)
        tilt = np.abs(rpy[0]) + np.abs(rpy[1])  # roll + pitch magnitude

        # ----------------- REWARD -----------------
        reward = 0.0

        # Penalize vertical error
        reward += -2.0 * np.abs(z_error)

        # Penalize horizontal drift
        reward += -0.5 * xy_error_norm

        # Penalize velocity
        reward += -0.3 * vel_norm

        # Penalize tilt (roll, pitch)
        reward += -0.5 * tilt

        # Small penalty for high angular velocity
        reward += -0.05 * np.linalg.norm(ang_vel)

        # Bonus for being close to target altitude
        if np.abs(z_error) < 0.05:
            reward += 1.0
        if np.abs(z_error) < 0.02:
            reward += 2.0

        # "Survival" style small positive reward to encourage staying alive
        reward += 0.02

        # ----------------- TERMINATION -----------------
        terminated = False
        truncated = False

        # Crash or out-of-bounds (tune for your scene)
        if (
            pos[2] < 0.05              # almost on ground
            or pos[2] > 2.5            # way too high
            or np.abs(pos[0]) > 2.0    # far in x
            or np.abs(pos[1]) > 2.0    # far in y
            or tilt > 1.5              # too tilted (~86 deg total roll+pitch)
        ):
            terminated = True
            # Extra penalty on crash
            reward -= 5.0

        # Episode time limit
        if self.step_count >= self.max_steps:
            truncated = True

        return reward, terminated, truncated, {
            "z_error": float(z_error),
            "xy_error": float(xy_error_norm),
            "tilt": float(tilt),
            "vel_norm": float(vel_norm),
        }

    # ==========================================================
    #                        API
    # ==========================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        raw_obs, info = self.env.reset(seed=seed)
        # Single drone → index 0
        obs = self._extract_state(raw_obs[0])

        return obs, info

    def step(self, action):
        self.step_count += 1

        # Clip and cast action
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Map normalized action → RPM around hover
        BASE = self.hover_rpm            # ~18000 for CF2X
        RANGE = 4000.0                   # control authority

        rpm = BASE + action * RANGE
        # Safety clip
        rpm = np.clip(rpm, 0.0, float(self.env.MAX_RPM))

        # CtrlAviary expects an array of shape (num_drones, 4)
        rpm = np.expand_dims(rpm, axis=0)

        raw_obs, _, env_terminated, env_truncated, info = self.env.step(rpm)

        obs = self._extract_state(raw_obs[0])

        # Our own reward and done logic
        reward, terminated, truncated, extra_info = self._compute_reward_and_done(obs)

        # Combine with underlying env done flags if they are arrays
        if np.ndim(env_terminated) > 0:
            env_terminated = bool(env_terminated[0])
        if np.ndim(env_truncated) > 0:
            env_truncated = bool(env_truncated[0])

        terminated = bool(terminated or env_terminated)
        truncated = bool(truncated or env_truncated)

        # Merge info dicts
        info = {**info, **extra_info}

        return obs, reward, terminated, truncated, info

    def render(self):
        # CtrlAviary handles GUI if gui=True
        pass

    def close(self):
        self.env.close()
