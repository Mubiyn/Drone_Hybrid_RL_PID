import numpy as np
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class QuadcopterEnv(BaseRLAviary):

    metadata = {"render_modes": ["human", "rgb_array", "none"]}

    def __init__(self, tasks=None,task='hover',mode='test', custom_target=None, render_mode="none"):

        # Supported tasks
        self.tasks = tasks or ["circle", "figure8", "four_points", "goto"]
        self.custom_target = custom_target
        self.render_mode = render_mode

        if mode == 'train':
            self.task = np.random.choice(self.tasks)
        else:
            if task is None:
                raise ValueError("in test mode, provide a task")
            self.task =task

        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            neighbourhood_radius=np.inf,
            initial_xyzs=np.array([[0.0, 0.0, 0.05]]),
            initial_rpys=None,
            physics=Physics.PYB,
            pyb_freq=240,
            ctrl_freq=240,
            gui=(render_mode == "human"),
            record=False,
            obs=ObservationType.KIN,
            act=ActionType.PID,
        )

        # obs = pos(3) + vel(3) + rpy(3) + ang_vel(3) + rel_waypoint(3) + task_id(4) = 19
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(19,),
            dtype=np.float32
        )
        self.mode = mode
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        # Curriculum / episode settings
        self.phase = "hover"           # "hover" → then task name
        self.hover_steps = 0
        self.HOVER_THRESHOLD = 100     # how long to hover before switching
        self.max_steps = 2000
        self.current_steps = 0

        # Waypoints
        self.waypoints = self._build_waypoints(self.task)
        self.wp_idx = 0
        self.prev_dist = 0.0   # will be set in reset()


    # RESET
    def reset(self, seed=None, options=None):
        # 1) choose task & reset high-level state *before* parent reset
        if self.mode == 'train':
            self.task = np.random.choice(self.tasks)
        self.current_steps = 0
        self.phase = "hover"
        self.hover_steps = 0

        self.waypoints = self._build_waypoints(self.task)
        # safety: always ensure there's at least one waypoint
        if len(self.waypoints) == 0:
            self.waypoints = [np.array([1.0, 0.0, 1.0])]

        self.wp_idx = 0

        super().reset(seed=seed, options=options)

        # 3) init prev_dist
        pos = self._getDroneStateVector(0)[0:3]
        self.prev_dist = np.linalg.norm(pos - self.waypoints[0])

        return self._computeObs(), {}


    def _computeInfo(self):
        return {
            "waypoint_index": int(self.wp_idx),
            "phase": self.phase,
            "task": self.task
        }


    def step(self, action):
        state = self._getDroneStateVector(0)
        pos = state[0:3]

        # ---------------- HOVER PHASE ----------------
        if self.phase == "hover":
            target = np.array([0.0, 0.0, 1.0])  # fixed hover target
            obs, reward, terminated, truncated, info = super().step(target.reshape(1, 3))
            self.current_steps += 1
            return obs, reward, terminated, truncated, info

        # ---------------- TASK PHASE -----------------
        if isinstance(action, np.ndarray) and action.ndim == 2:
            delta = action[0]
        else:
            delta = action

        delta = np.clip(delta, -1.0, 1.0)

        # current waypoint
        wp = self.waypoints[self.wp_idx]
        rel = wp - pos       # vector toward waypoint

        target = pos + 0.5 * delta + 0.2 * rel

        # smooth altitude toward waypoint z
        target[2] = 0.8 * target[2] + 0.2 * wp[2]
        target = np.clip(target, [-3, -3, 0.5], [3, 3, 2.0])

        obs, reward, terminated, truncated, info = super().step(target.reshape(1, 3))
        self.current_steps += 1
        return obs, reward, terminated, truncated, info



    # WAYPOINTS
    def _build_waypoints(self, task):
        if task == "circle":
            r = 1.5
            angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
            return [np.array([r * np.cos(a), r * np.sin(a), 1.0]) for a in angles]

        if task == "figure8":
            r = 1.2
            angles = np.linspace(0, 2 * np.pi, 40)
            w1 = [np.array([-r * np.cos(a), r * np.sin(a), 1.0]) for a in angles]
            w2 = [np.array([ r * np.cos(a), r * np.sin(a), 1.0]) for a in angles]
            return w1 + w2

        if task == "four_points":
            return [
                np.array([ 0,  0, 1]),
                np.array([1,  1, 1]),
                np.array([-1, 1, 1]),
                np.array([ 0, 0, 1]),
               
            ]

        if task == "goto" and self.custom_target is not None:
            return [np.array([self.custom_target[0], self.custom_target[1], 1.0])]

        # fallback
        return [np.array([1.0, 0.0, 1.0])]


    # OBSERVATION
    def _computeObs(self):
        s = self._getDroneStateVector(0)
        pos, vel = s[0:3], s[10:13]
        rpy, ang_vel = s[7:10], s[13:16]

        # protect against out-of-range index
        idx = min(self.wp_idx, len(self.waypoints) - 1)
        wp = self.waypoints[idx]
        rel = wp - pos

        # one-hot task id
        task_id = np.zeros(len(self.tasks), dtype=np.float32)
        task_index = self.tasks.index(self.task)
        task_id[task_index] = 1.0

        return np.concatenate([pos, vel, rpy, ang_vel, rel, task_id]).astype(np.float32)


    # REWARD
    def _computeReward(self):
        s = self._getDroneStateVector(0)
        pos = s[0:3]
        vel = s[10:13]
        rpy = s[7:10]

        # ---------------- HOVER PHASE ----------------
        if self.phase == "hover":
            target = np.array([0.0, 0.0, 1.0])
            dist = np.linalg.norm(pos - target)

            reward = (
                -2.0 * dist
                -0.3 * np.linalg.norm(vel)
                -0.02 * (abs(rpy[0]) + abs(rpy[1]))
            )

            if dist < 0.25:
                self.hover_steps += 1
                reward += 2.0
            else:
                self.hover_steps = 0

            if self.hover_steps > self.HOVER_THRESHOLD:
                self.phase = self.task
                self.prev_dist = np.linalg.norm(pos - self.waypoints[self.wp_idx])

            if pos[2] < 0.05:
                reward -= 20.0

            return reward

        # ---------------- TASK PHASE -----------------
        wp = self.waypoints[self.wp_idx]
        rel = wp - pos
        dist = np.linalg.norm(rel)

        reward = 0.0

        reward -= 1.0 * dist

        # 2) Altitude and attitude
        reward -= 0.3 * abs(pos[2] - wp[2])
        reward -= 0.1 * (abs(rpy[0]) + abs(rpy[1]))

        # 3) Speed penalty (don’t go crazy fast)
        speed = np.linalg.norm(vel)
        reward -= 0.05 * speed

        # 4) Progress toward waypoint
        progress = self.prev_dist - dist     # >0 if getting closer
        reward += 10.0 * progress

        if progress < 0:
            reward -= 4.0    # moving away is bad

        vel_dir = vel / (speed + 1e-6)
        wp_dir = rel / (dist + 1e-6)
        alignment = np.dot(vel_dir, wp_dir)   # [-1, 1]
        reward += 3.0 * alignment

        if speed < 0.03 and dist > 0.4:
            reward -= 2.0

        self.prev_dist = dist

        # 7) Waypoint reached
        if dist < 0.25:
            reward += 15.0
            self.wp_idx += 1

            if self.wp_idx < len(self.waypoints):
                next_wp = self.waypoints[self.wp_idx]
                self.prev_dist = np.linalg.norm(pos - next_wp)

        # 8) Crash
        if pos[2] < 0.05:
            reward -= 10.0

        return reward

    # TERMINATION / TRUNCATION
    def _computeTerminated(self):
        z = self._getDroneStateVector(0)[2]

        # crash
        if z < 0.05:
            return True

        # in task phase: done when last waypoint reached
        if self.phase != "hover" and self.wp_idx >= len(self.waypoints):
            return True

        return False

    def _computeTruncated(self):
        return self.current_steps >= self.max_steps
