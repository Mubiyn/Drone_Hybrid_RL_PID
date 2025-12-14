import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.envs.BaseTrackAviary import BaseTrackAviary
from src.envs.RobustTrackAviary import RobustTrackAviary
from src.envs.HybridAviary import HybridAviary
from src.controllers.pid_controller import PIDController
from gym_pybullet_drones.utils.enums import DroneModel, Physics

# Configuration
TRAJECTORIES = ['hover', 'circle', 'figure8', 'spiral', 'waypoint']
MODELS_DIR = "models"
RESULTS_DIR = "results/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)

def eval_controller(controller_type, model_path=None, trajectory_type='circle', duration=15.0, robust=False):
    """
    Evaluate a controller on a specific trajectory.
    controller_type: 'pid', 'ppo', 'hybrid'
    robust: If True, enables wind and mass randomization.
    """
    print(f"Evaluating {controller_type.upper()} on {trajectory_type} (Robust: {robust})...")
    
    # Setup Env Factory
    def make_env():
        if controller_type == 'hybrid':
            return HybridAviary(trajectory_type=trajectory_type, gui=False, record=False, domain_randomization=robust)
        elif controller_type == 'ppo':
            if robust:
                return RobustTrackAviary(trajectory_type=trajectory_type, gui=False, record=False, domain_randomization=True)
            else:
                return BaseTrackAviary(trajectory_type=trajectory_type, gui=False, record=False)
        else: # PID
            if robust:
                return RobustTrackAviary(trajectory_type=trajectory_type, gui=False, record=False, domain_randomization=True)
            else:
                return BaseTrackAviary(trajectory_type=trajectory_type, gui=False, record=False)
    
    # Create Vectorized Env
    env = DummyVecEnv([make_env])
        
    # Load Model
    model = None
    if controller_type in ['ppo', 'hybrid']:
        if model_path and os.path.exists(model_path):
            try:
                model = PPO.load(model_path)
                print(f"Loaded model from {model_path}")
                
                # Load VecNormalize stats if available
                vec_norm_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
                if os.path.exists(vec_norm_path):
                    print(f"Loading normalization stats from {vec_norm_path}")
                    env = VecNormalize.load(vec_norm_path, env)
                    env.training = False
                    env.norm_reward = False
                else:
                    print("No normalization stats found. Using raw environment.")
                    
            except Exception as e:
                print(f"Failed to load model from {model_path}: {e}")
                env.close()
                return None, None
        else:
            print(f"Model path not found: {model_path}")
            env.close()
            return None, None

    # PID Controller (for 'pid' case)
    pid = None
    if controller_type == 'pid':
        pid = PIDController(freq=240)
        
    obs = env.reset()
    
    errors = []
    trajectory_trace = []
    target_trace = []
    
    # Run simulation
    steps = int(duration * 240)
    for i in range(steps):
        # obs shape can be (1, 18) or (1, 1, 18) depending on wrapping
        if obs.ndim == 3:
            current_obs = obs[0, 0]  # Extract from (1, 1, 18)
        else:
            current_obs = obs[0]  # Extract from (1, 18)
        
        # Obs structure: [x,y,z, r,p,y, vx,vy,vz, wx,wy,wz, pos_err(3), vel_err(3)]
        current_pos = current_obs[0:3]
        pos_error = current_obs[12:15]
        target_pos = current_pos + pos_error
        
        trajectory_trace.append(current_pos)
        target_trace.append(target_pos)
        errors.append(np.linalg.norm(pos_error))
        
        if controller_type == 'pid':
            current_vel = current_obs[6:9]
            vel_error = current_obs[15:18]
            target_vel = current_vel + vel_error
            # PID needs only the state (first 12 dims)
            state_obs = current_obs[0:12]
            action = pid.compute_control(state_obs, target_pos, target_vel)
            action = np.array([[action]]) # (1, 1, 4)
        elif model:
            action, _ = model.predict(obs, deterministic=True)
            # Ensure action is (1, 1, 4) if model outputs (1, 4)
            if action.ndim == 2:
                action = action.reshape(1, 1, 4)
        else:
            action = [env.action_space.sample()]
            
        obs, _, _, _ = env.step(action)
        
    env.close()
    return np.array(errors), np.array(trajectory_trace)

def run_comparison(robust=False):
    summary_data = []
    suffix = "_robust" if robust else ""

    for traj in TRAJECTORIES:
        print(f"\n{'='*40}")
        print(f"Processing Trajectory: {traj.upper()} (Robust: {robust})")
        print(f"{'='*40}")
        
        # Define model paths
        # Check for nominal or robust models based on evaluation mode?
        # Actually, we usually want to evaluate the ROBUST model in both conditions to see if it generalizes.
        # But if we trained a NOMINAL model, we might want to evaluate that too.
        # For simplicity, let's look for the model that matches the requested evaluation mode first, 
        # but fallback or allow specifying which model to load.
        
        # Current logic: 
        # If evaluating --robust, look for models/ppo_robust/...
        # If evaluating nominal, look for models/ppo_robust/... (Assuming we want to test the robust model's nominal performance)
        
        # BUT, now that we can train nominal models, we might want to compare:
        # 1. PID
        # 2. PPO (Robust)
        # 3. Hybrid (Robust)
        # 4. PPO (Nominal) - Optional
        # 5. Hybrid (Nominal) - Optional
        
        # Let's stick to the main comparison: Robust Models vs PID.
        # If you want to evaluate Nominal models, we can add logic.
        
        ppo_path = f"{MODELS_DIR}/ppo_robust/{traj}/final_model.zip"
        hybrid_path = f"{MODELS_DIR}/hybrid_robust/{traj}/final_model.zip"
        
        # Check if nominal models exist and add them?
        ppo_nom_path = f"{MODELS_DIR}/ppo_nominal/{traj}/final_model.zip"
        hybrid_nom_path = f"{MODELS_DIR}/hybrid_nominal/{traj}/final_model.zip"
        
        results = {}
        
        # 1. Evaluate PID
        pid_err, pid_trace = eval_controller('pid', trajectory_type=traj, robust=robust)
        if pid_err is not None:
            results['PID'] = pid_err
            summary_data.append({'Task': traj, 'Controller': 'PID', 'RMSE': np.sqrt(np.mean(pid_err**2)), 'Mean Error': np.mean(pid_err)})

        # 2. Evaluate PPO (Robust)
        ppo_err, ppo_trace = eval_controller('ppo', model_path=ppo_path, trajectory_type=traj, robust=robust)
        if ppo_err is not None:
            results['PPO (Robust)'] = ppo_err
            summary_data.append({'Task': traj, 'Controller': 'PPO (Robust)', 'RMSE': np.sqrt(np.mean(ppo_err**2)), 'Mean Error': np.mean(ppo_err)})

        # 3. Evaluate Hybrid (Robust)
        hybrid_err, hybrid_trace = eval_controller('hybrid', model_path=hybrid_path, trajectory_type=traj, robust=robust)
        if hybrid_err is not None:
            results['Hybrid (Robust)'] = hybrid_err
            summary_data.append({'Task': traj, 'Controller': 'Hybrid (Robust)', 'RMSE': np.sqrt(np.mean(hybrid_err**2)), 'Mean Error': np.mean(hybrid_err)})

        # 4. Evaluate PPO (Nominal) - If exists
        if os.path.exists(ppo_nom_path):
            ppo_nom_err, _ = eval_controller('ppo', model_path=ppo_nom_path, trajectory_type=traj, robust=robust)
            if ppo_nom_err is not None:
                results['PPO (Nominal)'] = ppo_nom_err
                summary_data.append({'Task': traj, 'Controller': 'PPO (Nominal)', 'RMSE': np.sqrt(np.mean(ppo_nom_err**2)), 'Mean Error': np.mean(ppo_nom_err)})

        # 5. Evaluate Hybrid (Nominal) - If exists
        if os.path.exists(hybrid_nom_path):
            hyb_nom_err, _ = eval_controller('hybrid', model_path=hybrid_nom_path, trajectory_type=traj, robust=robust)
            if hyb_nom_err is not None:
                results['Hybrid (Nominal)'] = hyb_nom_err
                summary_data.append({'Task': traj, 'Controller': 'Hybrid (Nominal)', 'RMSE': np.sqrt(np.mean(hyb_nom_err**2)), 'Mean Error': np.mean(hyb_nom_err)})

        
        # Plotting
        if results:
            plt.figure(figsize=(12, 6))
            for name, err in results.items():
                time_axis = np.linspace(0, len(err)/240, len(err))
                plt.plot(time_axis, err, label=f'{name} (Mean: {np.mean(err):.4f}m)')
            
            plt.title(f"Trajectory Tracking Error - {traj.capitalize()}{' (Robust)' if robust else ''}")
            plt.xlabel("Time (s)")
            plt.ylabel("Position Error (m)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            save_path = f"{RESULTS_DIR}/comparison_{traj}{suffix}.png"
            plt.savefig(save_path)
            print(f"Saved plot to {save_path}")
            plt.close()

    # Save Summary CSV
    if summary_data:
        df = pd.DataFrame(summary_data)
        csv_path = f"{RESULTS_DIR}/evaluation_summary{suffix}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nEvaluation Complete. Summary saved to {csv_path}")
        print("\nSummary Table:")
        print(df.pivot(index='Task', columns='Controller', values='RMSE'))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--robust", action="store_true", help="Enable robust evaluation (wind/mass noise)")
    args = parser.parse_args()
    
    run_comparison(robust=args.robust)
