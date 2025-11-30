import numpy as np
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
import pybullet as p


class QuadcopterEnv(BaseRLAviary):

    metadata = {"render_modes": ["human", "rgb_array", "none"]}

    def __init__(self, task="circle", custom_target=None, render_mode="none"):

        self.task = task
        self.custom_target = custom_target
        self.render_mode = render_mode

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

        # Observation vector = pos(3)+vel(3)+rpy(3)+ang_vel(3) = 12
        # self.observation_space = spaces.Box(
        #     low=-np.inf, high=np.inf,
        #     shape=(12,), dtype=np.float32
        # )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(12,), dtype=np.float32
        )

        # Action is Δposition, clipped small to avoid instability
        # self.action_space = spaces.Box(
        #     low=np.array([-1, -1, -1], dtype=np.float32),
        #     high=np.array([1, 1, 1], dtype=np.float32),
        #     dtype=np.float32
        # )
        self.action_space = spaces.Box(
            low=np.array([-0.05, -0.05, -0.05], dtype=np.float32),
            high=np.array([ 0.05,  0.05,  0.05], dtype=np.float32)
        )


        # Curriculum parameters
        self.phase = "hover"
        self.hover_steps = 0
        self.HOVER_THRESHOLD = 800   # Switch to task after stable hover
        self.max_steps = 5000  # or 1000, pick a horizon you like
        self.current_steps = 0
        self.waypoints = self._build_waypoints()
        self.wp_idx = 0


    # -----------------------------------------------------------
    # Action conversion (Δposition → absolute position)
    # -----------------------------------------------------------
    def step(self, action):
        # action comes as (1,3) from SB3 when using VecEnv
        state = self._getDroneStateVector(0)
        pos = state[0:3]

      
        
        if isinstance(action, np.ndarray) and action.ndim == 2:
            delta = action[0]
        else:
            delta = action

        target = pos + delta
        # ---------------------------
        # PHASE 1: ABSOLUTE HOVERING
        # ---------------------------
        if self.phase == "hover":
            target = np.array([0, 0, 1.0])  # GO TO Z = 1.0 ALWAYS
            return super().step(target.reshape(1, -1))
        
        target = np.clip(target, [-3,-3,0.2], [3,3,2.0])
        
        obs, reward, terminated, truncated, info = super().step(target.reshape(1, -1))
        self.current_steps += 1
        return obs, reward, terminated, truncated, info

        #self.current_steps += 1

        #return super().step(target.reshape(1, -1))


    def _computeInfo(self):
        return {
            "waypoint_index": self.wp_idx,
            "phase": self.phase,
        }




    # -----------------------------------------------------------
    # Waypoints
    # -----------------------------------------------------------
    def _build_waypoints(self):
        if self.task == "circle":
            r = 1.5
            angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
            return [np.array([r * np.cos(a), r * np.sin(a), 1.0]) for a in angles]

        if self.task == "figure8":
            r = 1.2
            angles = np.linspace(0, 2 * np.pi, 40)
            w1 = [np.array([-r * np.cos(a), r * np.sin(a), 1.0]) for a in angles]
            w2 = [np.array([ r * np.cos(a), r * np.sin(a), 1.0]) for a in angles]
            return w1 + w2

        if self.task == "four_points":
            return [
                np.array([ 1,  1, 1]),
                np.array([-1,  1, 1]),
                np.array([-1, -1, 1]),
                np.array([ 1, -1, 1]),
                np.array([ 0,  0, 1]),
            ]

        if self.task == "goto" and self.custom_target is not None:
            return [np.array([self.custom_target[0], self.custom_target[1], 1.0])]

        return [np.array([1, 0, 1])]   # default WP


    # -----------------------------------------------------------
    # Observation
    # -----------------------------------------------------------
 
    def _computeObs(self):
        s = self._getDroneStateVector(0)
        pos = s[0:3]
        vel = s[10:13]
        rpy = s[7:10]
        ang_vel = s[13:16]

        # relative target
        if self.phase == "hover":
            target = np.array([0.0, 0.0, 1.0])
        else:
            target = self.waypoints[self.wp_idx]
        rel_wp = target - pos

        return np.concatenate([pos, vel, rpy, ang_vel]).astype(np.float32)



    # -----------------------------------------------------------
    # Reward with Hover Curriculum
    # -----------------------------------------------------------
    def _computeReward(self):
        s = self._getDroneStateVector(0)
        pos = s[0:3]
        vel = s[10:13]
        rpy = s[7:10]

        reward = 0.0

        # ================================
        # PHASE 1 — Hovering
        # ================================
        if self.phase == "hover":
            target = np.array([0, 0, 1.0])
            dist = np.linalg.norm(pos - target)

            # Hovering reward terms
            reward -= 2.0 * dist
            reward -= 0.3 * np.linalg.norm(vel)
            reward -= 0.1 * (abs(rpy[0]) + abs(rpy[1]))

            # Stability bonus
            if dist < 0.25:
                self.hover_steps += 1
                reward += 3.0
            else:
                self.hover_steps = 0

            # Transition to TASK after stable hovering
            if self.hover_steps > self.HOVER_THRESHOLD:
                if self.phase != self.task:
                    print("[ENV] Switching phase →", self.task)
                # initialize previous distance for progress reward
                self.phase = self.task
                wp = self.waypoints[self.wp_idx]
                self.prev_dist = np.linalg.norm(pos - wp)

            if pos[2] < 0.05:
                reward -= 20.0

            return reward


        # ================================
        # PHASE 2 — Task Waypoint Tracking
        # ================================
        wp = self.waypoints[self.wp_idx]
        dist = np.linalg.norm(pos - wp)
        

        # Distance reward
        reward -= 1.0*dist

        # Altitude control
        reward -= 0.5 * abs(pos[2] - 1.0)

        # Penalize tilt
        reward -= 0.5 * (abs(rpy[0]) + abs(rpy[1]))

        # Penalize speed
        reward -= 0.05 * np.linalg.norm(vel)

        # Progress reward
        if np.isnan(dist) or np.isinf(dist):
            return -5
        
        reward += 0.5 * (self.prev_dist - dist)   # progress only

        #progress = self.prev_dist - dist
        #reward += 3.0 * progress
        self.prev_dist = dist
        # Strong upward encouragement
        if pos[2] < 0.8:
            reward -= 3*(0.8 - pos[2])

        # Bonus when within hovering tolerance
        if dist < 0.15:
            self.hover_steps += 1
            reward += 3.0
        else:
            self.hover_steps = 0

        # Waypoint reached
        if dist < 0.25:
            reward += 5.0
            self.wp_idx += 1
            # reset prev_dist for new waypoint
            if self.wp_idx < len(self.waypoints):
                next_wp = self.waypoints[self.wp_idx]
                self.prev_dist = np.linalg.norm(pos - next_wp)

        if pos[2] < 0.05:
            reward -= 5.0

        return reward


    # -----------------------------------------------------------
    # Termination
    # -----------------------------------------------------------

    def _computeTerminated(self):
        z = self._getDroneStateVector(0)[2]

        # crash
        if z < 0.05:
            return True

        # In hover phase: NEVER terminate based on waypoint index
        if self.phase == "hover":
            return False

        # In task phase: terminate after last waypoint
        return self.wp_idx >= len(self.waypoints)



    def _computeTruncated(self):
        
        return self.current_steps >= self.max_steps



    # -----------------------------------------------------------
    # Reset
    # -----------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.current_steps = 0

        self.wp_idx = 0
        self.phase = "hover"
        self.hover_steps = 0

        # Initialize prev_dist for progress reward
        hover_target = np.array([0, 0, 1.0])
        pos = self._getDroneStateVector(0)[0:3]
        self.prev_dist = np.linalg.norm(pos - hover_target)

        return self._computeObs(), {}
