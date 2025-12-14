import numpy as np
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from src.utils.trajectories import TrajectoryGenerator

class BaseTrackAviary(BaseRLAviary):
    """
    Base environment for trajectory tracking tasks.
    """
    def __init__(self, 
                 trajectory_type='hover', 
                 drone_model=DroneModel.CF2X,
                 num_drones=1, 
                 physics=Physics.PYB, 
                 freq=240,
                 gui=False,
                 record=False,
                 obs=ObservationType.KIN,
                 act=ActionType.RPM):
        
        self.traj_gen = TrajectoryGenerator(trajectory_type=trajectory_type)
        
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         physics=physics,
                         pyb_freq=freq,
                         ctrl_freq=freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act)

    def _observationSpace(self):
        if self.OBS_TYPE == ObservationType.KIN:
            lo = -np.inf
            hi = np.inf
            obs_lower_bound = np.array([[lo]*18 for i in range(self.NUM_DRONES)])
            obs_upper_bound = np.array([[hi]*18 for i in range(self.NUM_DRONES)])
            return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
        else:
            return super()._observationSpace()

    def _computeObs(self):
        """
        Override to include target information and disturbance estimate.
        """
        if self.OBS_TYPE == ObservationType.KIN:
            obs_18 = np.zeros((self.NUM_DRONES, 18))
            
            t = self.step_counter / self.CTRL_FREQ
            target_pos, target_vel, _ = self.traj_gen.get_target(t)
            
            for i in range(self.NUM_DRONES):
                state = self._getDroneStateVector(i)
                base_obs = np.hstack([state[0:3], state[7:10], state[10:13], state[13:16]])
                pos_error = target_pos - state[0:3]
                vel_error = target_vel - state[10:13]
                obs_18[i, :] = np.hstack([base_obs, pos_error, vel_error])
                
            return obs_18.astype('float32')
        else:
            return super()._computeObs()

    def _preprocessAction(self, action):
        """
        Override to allow raw RPMs if values are large.
        """
        if self.ACT_TYPE == ActionType.RPM:
            # Check if action looks like raw RPM (e.g. > 100)
            if np.any(np.abs(action) > 100):
                # Assume raw RPM, clip to valid range
                rpm = np.clip(action, 0, self.MAX_RPM)
                # Update action buffer with normalized version for consistency
                # Normalized = (RPM / HOVER_RPM - 1) / 0.05
                normalized_action = (rpm / self.HOVER_RPM - 1) / 0.05
                self.action_buffer.append(normalized_action)
                return rpm
            else:
                # Standard normalized action
                return super()._preprocessAction(action)
        else:
            return super()._preprocessAction(action)

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        t = self.step_counter / self.CTRL_FREQ
        target_pos, target_vel, _ = self.traj_gen.get_target(t)
        
        # Position & velocity errors
        pos_error = np.linalg.norm(state[0:3] - target_pos)
        vel_error = np.linalg.norm(state[10:13] - target_vel)
        
        # Exponential position penalty (steeper gradient near target)
        pos_reward = -20.0 * (1.0 - np.exp(-5.0 * pos_error**2))
        
        # Velocity alignment (softer penalty)
        vel_reward = -2.0 * vel_error
        
        # Attitude stability (penalize tilting)
        r, p, y = state[7:10]
        att_reward = -1.0 * (r**2 + p**2)
        
        # Angular velocity smoothness
        ang_vel = np.linalg.norm(state[13:16])
        ang_reward = -0.5 * ang_vel
        
        # Action regularization (for Hybrid: discourage large corrections)
        action_penalty = 0.0
        if hasattr(self, 'last_rl_action') and self.last_rl_action is not None:
            action_penalty = -0.02 * np.sum(self.last_rl_action**2)
        
        # Combine
        reward = pos_reward + vel_reward + att_reward + ang_reward + action_penalty
        
        # Bonus for excellent tracking
        if pos_error < 0.1:
            reward += 10.0
        if pos_error < 0.05:
            reward += 20.0
            
        return reward

    def _computeTerminated(self):
        """
        Terminated if drone crashes or goes too far.
        """
        state = self._getDroneStateVector(0)
        if state[2] < 0.1: # Crashed on ground
            return True
        if state[2] > 5.0: # Too high
            return True
        return False
        
    def _computeTruncated(self):
        """
        Truncated if time limit reached.
        """
        if self.step_counter / self.CTRL_FREQ > self.traj_gen.duration:
            return True
        return False

    def _computeInfo(self):
        """
        Return extra info (e.g. current target error)
        """
        state = self._getDroneStateVector(0)
        t = self.step_counter / self.CTRL_FREQ
        target_pos, _, _ = self.traj_gen.get_target(t)
        dist = np.linalg.norm(state[0:3] - target_pos)
        return {"error": dist}
