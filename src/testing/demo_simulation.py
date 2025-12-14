import time
import os
from datetime import datetime
import numpy as np
import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.envs.BaseTrackAviary import BaseTrackAviary
from src.envs.HybridAviary import HybridAviary
from src.envs.RobustTrackAviary import RobustTrackAviary
from src.controllers.pid_controller import PIDController
from gym_pybullet_drones.utils.enums import DroneModel, Physics

def run_demo(controller_type, model_path=None, trajectory_type='circle', duration=15.0, record=True, domain_randomization=False):
    """
    Run a live demo of the controller.
    """
    print(f"Running Demo: {controller_type} | Trajectory: {trajectory_type} | DR: {domain_randomization}")
    
    # Prepare video recording path
    if record:
        video_dir = os.path.join("results", "videos", controller_type)
        os.makedirs(video_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
        video_name = f"{controller_type}_{trajectory_type}_{timestamp}.mp4"
        video_path = os.path.join(video_dir, video_name)

    # Setup Env with GUI
    def make_env():
        if controller_type == 'hybrid':
            return HybridAviary(trajectory_type=trajectory_type, 
                               gui=True, 
                               record=False, 
                               domain_randomization=domain_randomization)
        elif domain_randomization:
            return RobustTrackAviary(trajectory_type=trajectory_type, 
                                    gui=True, 
                                    record=False, 
                                    domain_randomization=True)
        else:
            return BaseTrackAviary(trajectory_type=trajectory_type, 
                                  gui=True, 
                                  record=False)
    
    # Wrap in VecEnv for model compatibility
    env = DummyVecEnv([make_env])
    
    # Load Model and VecNormalize
    model = None
    if controller_type in ['ppo', 'hybrid'] and model_path:
        try:
            model = PPO.load(model_path)
            print(f"Loaded model from {model_path}")
            
            # Load VecNormalize stats
            vec_norm_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
            if os.path.exists(vec_norm_path):
                print(f"Loading normalization stats from {vec_norm_path}")
                env = VecNormalize.load(vec_norm_path, env)
                env.training = False
                env.norm_reward = False
        except Exception as e:
            print(f"Warning: Could not load model: {e}")

    # PID Controller
    pid = None
    if controller_type == 'pid':
        pid = PIDController(freq=240)
        
    obs = env.reset()

    # Manually start recording on the actual environment
    if record:
        actual_env = env.envs[0] if hasattr(env, 'envs') else env.venv.envs[0]
        actual_env.RECORD = True
        actual_env.VIDEO_ID = p.startStateLogging(loggingType=p.STATE_LOGGING_VIDEO_MP4,
                                            fileName=video_path,
                                            physicsClientId=actual_env.CLIENT)
        print(f"Recording video to {video_path}")
    
    start_time = time.time()
    step = 0
    
    try:
        while step < int(duration * 240):
            # Handle observation shape: (1, 1, 18) or (1, 18)
            if obs.ndim == 3:
                current_obs = obs[0, 0]
            else:
                current_obs = obs[0]
            
            # Extract target from observation
            current_pos = current_obs[0:3]
            pos_error = current_obs[12:15]
            target_pos = current_pos + pos_error
            
            if controller_type == 'pid':
                current_vel = current_obs[6:9]
                vel_error = current_obs[15:18]
                target_vel = current_vel + vel_error
                # PID needs only state (first 12 dims)
                state_obs = current_obs[0:12]
                action = pid.compute_control(state_obs, target_pos, target_vel)
                action = np.array([[action]])  # (1, 1, 4) for VecEnv
            elif model:
                action, _ = model.predict(obs, deterministic=True)
                if action.ndim == 2:
                    action = action.reshape(1, 1, 4)
            else:
                action = [env.action_space.sample()]
                
            obs, _, _, _ = env.step(action)
            
            # Sync with real time
            time.sleep(1/240)
            step += 1
            
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        print("Demo finished.")

if __name__ == "__main__":
    # Example Usage:
    
    # 1. PID (Nominal)
    # run_demo('pid', trajectory_type='figure8', domain_randomization=False)
    
    # 2. PID (Robustness Fail)
    # run_demo('pid', trajectory_type='figure8', domain_randomization=True)
    
    # 3. Robust PPO (Update path!)
    # run_demo('ppo', model_path='logs/ppo_robust/figure8/LATEST/final_model.zip', trajectory_type='figure8', domain_randomization=True)
    
    # 4. Robust Hybrid (Update path!)
    # run_demo('hybrid', model_path='logs/hybrid_robust/figure8/LATEST/final_model.zip', trajectory_type='figure8', domain_randomization=True)
    
    # For now, just run PID circle
    run_demo('pid', trajectory_type='circle', duration=10.0)
