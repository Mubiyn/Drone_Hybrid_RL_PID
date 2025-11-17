import numpy as np
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


class PIDController:
    def __init__(self, drone_model, config=None):
        self.ctrl = DSLPIDControl(drone_model=drone_model)
        self.config = config if config is not None else {}
        self.control_timestep = 1.0 / 48.0
        
    def compute_control(self, obs, target_pos, target_rpy=np.zeros(3)):
        pos = obs[0:3]
        quat = obs[3:7]
        rpy = obs[7:10]
        vel = obs[10:13]
        ang_vel = obs[13:16]
        
        action, _, _ = self.ctrl.computeControlFromState(
            control_timestep=self.control_timestep,
            state=np.hstack([pos, quat, rpy, vel, ang_vel]),
            target_pos=target_pos,
            target_rpy=target_rpy
        )
        
        return action
    
    def reset(self):
        self.ctrl.reset()
