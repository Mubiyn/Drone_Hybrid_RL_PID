import time
import numpy as np
import matplotlib.pyplot as plt
from src.envs.BaseTrackAviary import BaseTrackAviary
from src.controllers.pid_controller import PIDController
from gym_pybullet_drones.utils.enums import DroneModel, Physics

def run_pid_baseline(trajectory_type='circle', duration=10.0, gui=False):
    print(f"Running PID Baseline for {trajectory_type}...")
    env = BaseTrackAviary(trajectory_type=trajectory_type, 
                          drone_model=DroneModel.CF2X,
                          physics=Physics.PYB,
                          freq=240,
                          gui=gui)
    
    ctrl = PIDController(drone_model=DroneModel.CF2X, freq=240)
    
    obs, _ = env.reset()
    
    pos_errors = []
    
    for i in range(int(duration * 240)):
        current_obs = obs[0]
        # State: 0-12
        # Target Pos: 12-15
        # Target Vel: 15-18
        
        target_pos = current_obs[12:15]
        target_vel = current_obs[15:18]
        
        # Compute control
        rpm = ctrl.compute_control(current_obs, target_pos, target_vel)
        
        # Step
        obs, reward, terminated, truncated, info = env.step(np.array([rpm]))
        
        # Log error
        pos_error = np.linalg.norm(current_obs[0:3] - target_pos)
        pos_errors.append(pos_error)
        
        if gui:
            time.sleep(1/240)
            
    env.close()
    
    mean_error = np.mean(pos_errors)
    print(f"Trajectory: {trajectory_type}")
    print(f"Mean Position Error: {mean_error:.4f} m")
    
    return pos_errors

if __name__ == "__main__":
    # Run without GUI for speed in this test
    run_pid_baseline(trajectory_type='hover', gui=False)
    run_pid_baseline(trajectory_type='circle', gui=False)
