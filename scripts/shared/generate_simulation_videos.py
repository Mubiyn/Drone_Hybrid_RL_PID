"""
Generate Simulation Videos for Phase 1 Results

This script generates demonstration videos for all tested trajectories
comparing PID baseline and Hybrid RL-PID controllers.

Videos are saved to: results/videos/{controller_type}/{trajectory}.mp4
"""

import os
import sys
import time
import numpy as np
import pybullet as p
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.envs.BaseTrackAviary import BaseTrackAviary
from src.envs.HybridAviary import HybridAviary
from src.controllers.pid_controller import PIDController


def generate_video(controller_type, trajectory_type, model_path=None, duration=15.0):
    """
    Generate a single video for a controller-trajectory combination.
    
    Args:
        controller_type: 'pid' or 'hybrid'
        trajectory_type: 'circle', 'figure8', 'hover', 'spiral', 'waypoint'
        model_path: Path to trained model (required for hybrid)
        duration: Video duration in seconds
    """
    print(f"\n{'='*60}")
    print(f"Generating: {controller_type.upper()} on {trajectory_type}")
    print(f"{'='*60}")
    
    # Setup video output path
    video_dir = os.path.join("results", "videos", controller_type)
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f"{trajectory_type}.mp4")
    
    # Create environment
    def make_env():
        if controller_type == 'hybrid':
            return HybridAviary(
                trajectory_type=trajectory_type,
                gui=True,
                record=False,
                domain_randomization=True  # Enable DR to match validated tests
            )
        else:  # PID
            from src.envs.RobustTrackAviary import RobustTrackAviary
            return RobustTrackAviary(
                trajectory_type=trajectory_type,
                gui=True,
                record=False,
                domain_randomization=True  # Enable DR to match validated tests
            )
    
    # Wrap in VecEnv
    env = DummyVecEnv([make_env])
    
    # Load model if hybrid
    model = None
    if controller_type == 'hybrid':
        if model_path is None or not os.path.exists(model_path):
            print(f"ERROR: Model path not found: {model_path}")
            env.close()
            return False
            
        try:
            model = PPO.load(model_path)
            print(f"✓ Loaded model: {model_path}")
            
            # Load VecNormalize stats
            vec_norm_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
            if os.path.exists(vec_norm_path):
                env = VecNormalize.load(vec_norm_path, env)
                env.training = False
                env.norm_reward = False
                print(f"✓ Loaded normalization stats")
        except Exception as e:
            print(f"ERROR loading model: {e}")
            env.close()
            return False
    
    # Setup PID controller
    pid = None
    if controller_type == 'pid':
        pid = PIDController(freq=240)
        print(f"✓ Initialized PID controller")
    
    # Reset environment
    obs = env.reset()
    
    # Start video recording
    try:
        actual_env = env.envs[0] if hasattr(env, 'envs') else env.venv.envs[0]
        actual_env.RECORD = True
        actual_env.VIDEO_ID = p.startStateLogging(
            loggingType=p.STATE_LOGGING_VIDEO_MP4,
            fileName=video_path,
            physicsClientId=actual_env.CLIENT
        )
        print(f"✓ Recording to: {video_path}")
    except Exception as e:
        print(f"ERROR starting recording: {e}")
        env.close()
        return False
    
    # Run simulation
    total_steps = int(duration * 240)
    step = 0
    
    try:
        print(f"Running simulation ({duration}s = {total_steps} steps)...")
        start_time = time.time()
        
        while step < total_steps:
            # Handle observation shape
            if obs.ndim == 3:
                current_obs = obs[0, 0]
            else:
                current_obs = obs[0]
            
            # Compute action
            if controller_type == 'pid':
                # Extract state and targets
                current_pos = current_obs[0:3]
                current_vel = current_obs[6:9]
                pos_error = current_obs[12:15]
                vel_error = current_obs[15:18]
                target_pos = current_pos + pos_error
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
            
            # Step environment
            obs, _, _, _ = env.step(action)
            
            # Sync with real time
            time.sleep(1/240)
            step += 1
            
            # Progress indicator
            if step % 240 == 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {step}/{total_steps} steps ({elapsed:.1f}s elapsed)")
        
        print(f"✓ Simulation complete!")
        return True
        
    except KeyboardInterrupt:
        print(f"\n⚠ Interrupted by user")
        return False
    except Exception as e:
        print(f"\nERROR during simulation: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        env.close()
        print(f"✓ Environment closed\n")


def main():
    """Generate all simulation videos for Phase 1 results"""
    
    print("\n" + "="*60)
    print("SIMULATION VIDEO GENERATION - Phase 1")
    print("="*60)
    
    # Define trajectories
    trajectories = ['circle', 'figure8', 'hover', 'spiral', 'waypoint']
    
    # Video duration (15 seconds should be enough to show trajectory)
    duration = 15.0
    
    # Find hybrid models
    models_base = "models"
    hybrid_models = {}
    
    print("\nSearching for trained models...")
    for traj in trajectories:
        # Look for hybrid_robust models (Phase 1 models with DR)
        model_path = os.path.join(models_base, "hybrid_robust", traj, "final_model.zip")
        if os.path.exists(model_path):
            hybrid_models[traj] = model_path
            print(f"  ✓ Found {traj}: {model_path}")
        else:
            print(f"  ✗ Missing {traj}: {model_path}")
    
    print(f"\nFound {len(hybrid_models)}/{len(trajectories)} hybrid models")
    
    # Generate videos
    results = {
        'pid': {},
        'hybrid': {}
    }
    
    print("\n" + "="*60)
    print("GENERATING PID VIDEOS")
    print("="*60)
    
    for traj in trajectories:
        success = generate_video('pid', traj, duration=duration)
        results['pid'][traj] = success
        time.sleep(2)  # Brief pause between videos
    
    print("\n" + "="*60)
    print("GENERATING HYBRID VIDEOS")
    print("="*60)
    
    for traj in trajectories:
        if traj in hybrid_models:
            success = generate_video('hybrid', traj, 
                                   model_path=hybrid_models[traj],
                                   duration=duration)
            results['hybrid'][traj] = success
        else:
            print(f"\nSkipping {traj} - no model found")
            results['hybrid'][traj] = False
        time.sleep(2)  # Brief pause between videos
    
    # Summary
    print("\n" + "="*60)
    print("VIDEO GENERATION SUMMARY")
    print("="*60)
    
    print("\nPID Videos:")
    for traj in trajectories:
        status = "✓" if results['pid'].get(traj, False) else "✗"
        print(f"  {status} {traj}")
    
    print("\nHybrid Videos:")
    for traj in trajectories:
        status = "✓" if results['hybrid'].get(traj, False) else "✗"
        print(f"  {status} {traj}")
    
    pid_success = sum(results['pid'].values())
    hybrid_success = sum(results['hybrid'].values())
    total = len(trajectories) * 2
    
    print(f"\nTotal: {pid_success + hybrid_success}/{total} videos generated")
    print(f"  PID: {pid_success}/{len(trajectories)}")
    print(f"  Hybrid: {hybrid_success}/{len(trajectories)}")
    
    print("\n✓ All done! Videos saved to results/videos/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
