# scripts/train_fresh_start.py
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
import sys
import os
import numpy as np

root = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
sys.path.insert(0, root)
from src.envs.multiple_task_env import DroneEnv
from src.utils.model_versions import ModelVersionManager
from src.utils.reward_funcs import get_reward_function

def make_env(task_name='hover', gui=False):
    return DroneEnv(
        task_name=task_name, 
        reward_fn=get_reward_function(task_name),
        domain_randomization=False,  # (or False if you want easier training)
        gui=gui
    )

def evaluate_model(model, task_name, episodes=3):
    """Evaluate model performance on a specific task"""
    total_reward = 0
    for _ in range(episodes):
        env = DroneEnv(task_name=task_name, reward_fn=get_reward_function(task_name), gui=False)
        obs, _ = env.reset()
        episode_reward = 0
        
        for step in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
            if done:
                break
                
        total_reward += episode_reward
        env.close()
    
    return total_reward / episodes

# --- Conservative Training Phases ---
training_phases = [
    ("hover", 3_000_000),      # More time on basic hover
    ("circle",3_000_000),
    # ("circle", 150_000),     # Then introduce movement
    # ("waypoint_delivery", 150_000),
    # ("figure8", 200_000),    # Most complex task last
    # ("emergency_landing", 200_000)
]

def train_fresh_start():
    # Initialize version manager
    version_manager = ModelVersionManager()
    version_manager.print_version_history()
    
    print("🔄 STARTING FRESH TRAINING")
    print("   Previous models will be preserved in version history")
    print("   Training with improved reward function and conservative settings")
    
    #env = DummyVecEnv([lambda: make_env('hover',gui=False)])
    model = None  # will be created in first phase

    
    best_combined_score = -float('inf')
    best_model = None
    
    for phase, (task_name, timesteps) in enumerate(training_phases):
        print(f"\n🎯 PHASE {phase + 1}/{len(training_phases)}: {task_name} for {timesteps:,} steps")
        env = DummyVecEnv([lambda tn=task_name: make_env(tn, gui=False)])

        #env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        if model is None:
            model = PPO(
                    "MlpPolicy",
                    env,
                    verbose=1,
                    learning_rate=3e-5,
                    n_steps=1024,
                    batch_size=64,
                    n_epochs=5,
                    gamma=0.998,
                    gae_lambda=0.90,
                    clip_range=0.1,
                    ent_coef=0.005,
                    vf_coef=0.5,
                    max_grad_norm=0.3,
                    tensorboard_log="./logs/ppo_drone/",
                    device="auto"
            )
        else:
            # Create environment for this phase
            model.set_env(env)
        model.learn(
            total_timesteps=timesteps, 
            reset_num_timesteps=False,
            progress_bar=True
        )
        
        # Evaluate performance
        print("📊 Evaluating performance...")
        task_scores = {}
        tasks_to_evaluate = ["hover", "circle"]  # Focus on key tasks
        
        for task in tasks_to_evaluate:
            try:
                score = evaluate_model(model, task)
                task_scores[task] = score
                print(f"   {task:18}: {score:8.2f}")
                
                # Quick performance check
                if task == "hover" and score > -100:
                    print("   ✅ Good hover performance!")
                elif task == "hover" and score > -200:
                    print("   ⚠️  Hover needs improvement")
                    
            except Exception as e:
                print(f"   {task:18}: Failed - {e}")
        
        # Calculate combined score
        hover_score = task_scores.get('hover', -1000)
        circle_score = task_scores.get('circle', -1000)
        combined_score = hover_score * 0.6 + circle_score * 0.4  # Weight hover more
        
        performance_metrics = {
            'combined_score': combined_score,
            'hover_score': hover_score,
            'circle_score': circle_score,
            'phase': phase + 1,
            'task_trained': task_name
        }
        
        print(f"   {'COMBINED SCORE':18}: {combined_score:8.2f}")
        
        # Save checkpoint if improved
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_model = model

            version_id = version_manager.save_best_model(
                model=model,
                performance_metrics=performance_metrics,
                task_scores=task_scores,
                notes=f"Fresh training - Phase {phase + 1}",
            )
            
            # Early success check
            if hover_score > -50:
                print("   🎉 EXCELLENT: Model learning well!")
            elif hover_score > -100:
                print("   ✅ GOOD: Steady progress!")
    
    # Final summary
    print("\n" + "=" * 60)
    print("FRESH TRAINING COMPLETE!")
    version_manager.print_version_history()
    
    best_version = version_manager.get_best_version()
    print(f"\nBEST MODEL: {best_version}")
    print(f"Combined Score: {best_combined_score:.2f}")
    
    # Save final model
    if best_model:
        best_model.save("models/PPO/ppo_drone_fresh_final")
        #env.save("models/PPO/vec_normalize_fresh.pkl")
        print("💾 Final model saved as: models/PPO/ppo_drone_fresh_final.zip")
    
    return best_model

if __name__ == "__main__":
    print("🚀 FRESH START TRAINING")
    print("=" * 60)
    print("Key improvements:")
    print("  • Strong focus on altitude control")
    print("  • Conservative learning parameters") 
    print("  • More time on basic hover")
    print("  • Better reward balancing")
    print("=" * 60)
    
    # Ensure model directory exists
    os.makedirs("models/PPO", exist_ok=True)
    
    # Train fresh
    model = train_fresh_start()
    
    if model:
        print("\n✅ Fresh training complete!")
        print("   Test your new model with:")
        print("   python scripts/test_drone.py --task=hover --gui=True")