import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from src.envs.HybridAviary import HybridAviary
from src.training.configs import PPO_CONFIG, TRAIN_CONFIG
from gym_pybullet_drones.utils.enums import DroneModel, Physics

def make_robust_hybrid_env(trajectory_type='hover', enable_dr=True):
    def _init():
        env = HybridAviary(trajectory_type=trajectory_type,
                           drone_model=DroneModel.CF2X,
                           physics=Physics.PYB,
                           freq=240,
                           gui=False,
                           record=False,
                           domain_randomization=enable_dr) # Enable/Disable DR
        return env
    return _init

def train_robust_hybrid(trajectory_type='hover', enable_dr=True):
    # Determine type string for paths
    type_str = "robust" if enable_dr else "nominal"
    
    # Create log dir
    log_dir = f"logs/hybrid_{type_str}/{trajectory_type}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create environment
    env = make_vec_env(make_robust_hybrid_env(trajectory_type, enable_dr), n_envs=TRAIN_CONFIG["num_envs"])
    # Critical: Don't normalize observations - PID needs raw physics values
    # Only normalize rewards to stabilize learning
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.)
    
    # Initialize PPO
    hybrid_config = PPO_CONFIG.copy()
    hybrid_config["learning_rate"] = 5e-4  # Slightly higher for faster adaptation
    
    model = PPO(env=env, tensorboard_log=log_dir, **hybrid_config)
    
    # Create separate eval environment (same config, no training)
    eval_env = make_vec_env(make_robust_hybrid_env(trajectory_type, enable_dr), n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True, clip_reward=10., training=False)
    # Train
    print(f"Starting Hybrid ({type_str.upper()}) training for {trajectory_type}...")
    try:
        model.learn(total_timesteps=TRAIN_CONFIG["total_timesteps"], callback=callbacks)
        save_path=log_dir, 
        name_prefix=f"hybrid_{type_str}_model"
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")
    finally:
        # Save final model
        model.save(f"{log_dir}/final_model")
        env.save(f"{log_dir}/vec_normalize.pkl")
        
        # Also save to models/ folder for easy access
        models_dir = f"models/hybrid_{type_str}/{trajectory_type}"
        os.makedirs(models_dir, exist_ok=True)
        model.save(f"{models_dir}/final_model")
        env.save(f"{models_dir}/vec_normalize.pkl")
        
        # Copy best_model if it exists (from EvalCallback)
        import shutil
        best_model_path = f"{log_dir}/best_model.zip"
        if os.path.exists(best_model_path):
            shutil.copy(best_model_path, f"{models_dir}/best_model.zip")
            print(f"✓ Best model saved to {models_dir}/best_model.zip")
        
        print(f"✓ Final model saved to {log_dir} and {models_dir}")
        print(f"  Use best_model.zip for deployment (highest eval reward)")
        print(f"  Use final_model.zip only if best_model doesn't exist")

    
    # CRITICAL: EvalCallback to save BEST model (not just final)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=10000,  # Evaluate every 10k steps
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Combine callbacks
    from stable_baselines3.common.callbacks import CallbackList
    callbacks = CallbackList([checkpoint_callback, eval_callback])
    
    # Train
    print(f"Starting Hybrid ({type_str.upper()}) training for {trajectory_type}...")
    try:
        model.learn(total_timesteps=TRAIN_CONFIG["total_timesteps"], callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")
    finally:
        # Save final model
        model.save(f"{log_dir}/final_model")
        env.save(f"{log_dir}/vec_normalize.pkl")
        
        # Also save to models/ folder for easy access
        models_dir = f"models/hybrid_{type_str}/{trajectory_type}"
        os.makedirs(models_dir, exist_ok=True)
        model.save(f"{models_dir}/final_model")
        env.save(f"{models_dir}/vec_normalize.pkl")
        
        # Copy best_model if it exists (from EvalCallback)
        import shutil
        best_model_path = f"{log_dir}/best_model.zip"
        if os.path.exists(best_model_path):
            shutil.copy(best_model_path, f"{models_dir}/best_model.zip")
            print(f"✓ Best model saved to {models_dir}/best_model.zip")
        
        print(f"✓ Final model saved to {log_dir} and {models_dir}")
        print(f"  Use best_model.zip for deployment (highest eval reward)")
        print(f"  Use final_model.zip only if best_model doesn't exist")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", type=str, default="all", help="Trajectory type or 'all'")
    parser.add_argument("--nominal", action="store_true", help="Disable Domain Randomization (Train Nominal Model)")
    args = parser.parse_args()

    enable_dr = not args.nominal
    trajectories = ['hover', 'circle', 'figure8', 'spiral', 'waypoint']

    if args.traj == 'all':
        for traj in trajectories:
            print(f"\n{'='*50}")
            print(f"STARTING HYBRID ({'ROBUST' if enable_dr else 'NOMINAL'}) TRAINING FOR: {traj.upper()}")
            print(f"{'='*50}\n")
            try:
                train_robust_hybrid(trajectory_type=traj, enable_dr=enable_dr)
            except Exception as e:
                print(f"Error training {traj}: {e}")
                continue
    else:
        # Train for specified trajectory
        train_robust_hybrid(trajectory_type=args.traj, enable_dr=enable_dr)
