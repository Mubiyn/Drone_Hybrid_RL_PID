import numpy as np
from scipy.spatial.transform import Rotation
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

class PIDController:
    """
    Wrapper for DSLPIDControl to work with our observation format.
    """
    def __init__(self, drone_model=DroneModel.CF2X, freq=240):
        self.ctrl = DSLPIDControl(drone_model=drone_model)
        self.dt = 1.0 / freq
        
    def compute_control(self, obs, target_pos, target_vel=None, target_yaw=0.0):
        """
        Compute RPMs based on current observation and target.
        
        Args:
            obs (np.ndarray): Observation vector (at least 12 elements).
                              [x,y,z, r,p,y, vx,vy,vz, wx,wy,wz, ...]
            target_pos (np.ndarray): Target position [x,y,z]
            target_vel (np.ndarray): Target velocity [vx,vy,vz]
            target_yaw (float): Target yaw angle
            
        Returns:
            rpm (np.ndarray): Motor RPMs
        """
        cur_pos = obs[0:3]
        cur_rpy = obs[3:6]
        cur_vel = obs[6:9]
        cur_ang_vel = obs[9:12]
        
        # Convert Euler to Quaternion
        # DSLPIDControl expects [x,y,z,w] format? Or [w,x,y,z]?
        # PyBullet uses [x,y,z,w]. Scipy uses [x,y,z,w].
        cur_quat = Rotation.from_euler('xyz', cur_rpy).as_quat()
        
        if target_vel is None:
            target_vel = np.zeros(3)
            
        rpm, _, _ = self.ctrl.computeControl(control_timestep=self.dt,
                                             cur_pos=cur_pos,
                                             cur_quat=cur_quat,
                                             cur_vel=cur_vel,
                                             cur_ang_vel=cur_ang_vel,
                                             target_pos=target_pos,
                                             target_rpy=np.array([0,0,target_yaw]),
                                             target_vel=target_vel)
        return rpm

class VelocityPIDController:
    """
    Simple Position -> Velocity PID for Tello-compatible control.
    """
    def __init__(self, kp=1.0, max_vel=1.0):
        self.kp = kp
        self.max_vel = max_vel
        
    def compute_control(self, obs, target_pos):
        """
        Compute Target Velocity based on position error.
        """
        cur_pos = obs[0:3]
        err = target_pos - cur_pos
        
        # P-Control
        target_vel = self.kp * err
        
        # Clip velocity
        norm = np.linalg.norm(target_vel)
        if norm > self.max_vel:
            target_vel = target_vel / norm * self.max_vel
            
        # Append yaw rate (0 for now)
        # Output: [vx, vy, vz, yaw_rate]
        return np.append(target_vel, 0.0)
