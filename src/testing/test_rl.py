import os
import sys
import time
import numpy as np
from stable_baselines3 import PPO
import argparse
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add project root to path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, root)

from src.envs.multiple_task_env import DroneEnv
from src.utils.model_versions import ModelVersionManager
from src.utils.reward_funcs import get_reward_function

class DroneTester:
    def __init__(self, gui=True):
        self.gui = gui
        self.version_manager = ModelVersionManager()
        self.task_name = "hover"
    
    def load_best_model(self):
        """Load the best performing model"""
        model, version = self.version_manager.load_best_model()
        print(f"\n🎮 Testing: {model}, Version: {version}")
        print("=" * 50)
        if model:
            return model, version
        else:
            # Fallback to any available model
            try:
                model = PPO.load("models/PPO/ppo_drone_latest.zip")
                return model, "latest"
            except:
                try:
                    model = PPO.load("models/PPO/ppo_drone_final.zip")
                    return model, "final"
                except:
                    print("❌ No trained model found!")
                    return None, None
    
    def create_env(self, task_name, gui=None):
        """Create environment with optional VecNormalize wrapper"""
        if gui is None:
            gui = self.gui
            
        reward_function = get_reward_function(task_name)
        
        def make_env():
            return DroneEnv(
                task_name=task_name,
                reward_fn=reward_function,
                domain_randomization=False,
                gui=gui
            )
        
        dummy_env = DummyVecEnv([make_env])
        
        # Load VecNormalize if available
        vec_path = "models/PPO/vec_normalize_fresh.pkl"
        
        if os.path.exists(vec_path):
            print("📥 Loading VecNormalize stats...")
            env = VecNormalize.load(vec_path, dummy_env)
            env.training = False
            env.norm_reward = False
        else:
            print("⚠️ No VecNormalize file found. Running without normalization.")
            env = dummy_env
        
        return env
    
    def test_single_task(self, task_name="circle", max_steps=1000):
        """Test a single task with optional GUI"""
        self.task_name = task_name
        print(f"\n Testing: {task_name} {'(with GUI)' if self.gui else '(no GUI)'}")
        print("=" * 50)
        
        if self.gui:
            print("Controls:")
            print("  - Press 'Q' in PyBullet window to stop early")
            print("  - Close PyBullet window to exit")
            print("  - Watch the drone follow the trajectory!")
        print("=" * 50)
        
        # Load model
        model, version_name = self.load_best_model()
        if model is None:
            print("❌ Cannot test - no model available")
            return None, None
        
        print(f"🤖 Using model: {version_name}")
        
        # Create environment
        env = self.create_env(task_name, self.gui)
        
        # Reset environment - VecEnv returns only obs
        obs = env.reset()
        
        total_reward = 0
        episode_length = 0
        trajectory_data = []
        
        print(f"🚀 Starting {task_name} trajectory...")
        if self.gui:
            print("💡 Tip: The colored line shows the target trajectory")
        
        try:
            for step in range(max_steps):
                # Get action from trained policy
                action, _ = model.predict(obs, deterministic=True)
                
                # Take step - VecEnv returns 4 values: obs, reward, done, info
                obs, reward, done, info = env.step(action)
                
                # Extract scalar values from arrays
                reward_scalar = float(reward[0])
                done_scalar = bool(done[0])
                
                total_reward += reward_scalar
                episode_length += 1
                
                # Access the base environment through the wrapper
                base_env = env.envs[0]
                if hasattr(env, 'venv'):  # If VecNormalize wrapper
                    base_env = env.venv.envs[0]
                
                # Get current state
                current_pos = obs[0, 0:3]  # VecEnv adds batch dimension
                
                # Get target position from base environment
                traj_step = min(base_env.trajectory_step, len(base_env.trajectory) - 1)
                target_pos = base_env.trajectory[traj_step]
                pos_error = np.linalg.norm(current_pos - target_pos)
                
                # Store trajectory data
                trajectory_data.append({
                    'step': step,
                    'position': current_pos.copy(),
                    'target': target_pos.copy(),
                    'error': pos_error,
                    'reward': reward_scalar
                })
                
                # Print progress every 50 steps
                if step % 50 == 0:
                    print(f"   Step {step:3d}: pos_error={pos_error:.3f}, cumulative_reward={total_reward:.2f}")
                
                # Add delay for better visualization if GUI is enabled
                if self.gui:
                    time.sleep(1.0 / 48.0)  # Match control frequency
                
                # Check if episode is done
                if done_scalar:
                    if base_env.trajectory_step >= len(base_env.trajectory):
                        print(f"🎉 SUCCESS: Completed {task_name} trajectory!")
                    else:
                        print(f"⚠️  EARLY TERMINATION: Step {step}")
                    break
            else:
                print(f"⏰ MAX STEPS REACHED: {max_steps}")
                
        except KeyboardInterrupt:
            print("\n🛑 TEST STOPPED BY USER")
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            env.close()
        
        # Final statistics
        self._print_final_stats(trajectory_data, total_reward, task_name, episode_length)
        return trajectory_data, total_reward
    
    def _print_final_stats(self, trajectory_data, total_reward, task_name, episode_length):
        """Print final performance statistics"""
        if not trajectory_data:
            print("❌ No trajectory data collected")
            return
            
        errors = [data['error'] for data in trajectory_data]
        
        avg_error = np.mean(errors) if errors else float('inf')
        min_error = np.min(errors) if errors else float('inf')
        max_error = np.max(errors) if errors else float('inf')
        
        print("\n" + "=" * 60)
        print(f"📊 {task_name.upper()} TEST RESULTS")
        print("=" * 60)
        print(f"   Total Steps:        {episode_length}")
        print(f"   Final Reward:       {total_reward:.2f}")
        print(f"   Average Error:      {avg_error:.3f}")
        print(f"   Minimum Error:      {min_error:.3f}")
        print(f"   Maximum Error:      {max_error:.3f}")
        
        # Performance assessment
        if total_reward > -50:
            rating = "🏆 EXCELLENT: Model is performing well!"
        elif total_reward > -100:
            rating = "✅ GOOD: Model is learning!"
        elif total_reward > -200:
            rating = "⚠️ FAIR: Model needs more training"
        else:
            rating = "❌ POOR: Model struggling with the task"
            
        print(f"   Performance:        {rating}")
        print("=" * 60)
    
    def compare_with_random(self, task_name="circle", max_steps=200):
        """Compare trained model vs random actions"""
        self.task_name = task_name
        print(f"\n🔬 COMPARISON: Random actions vs Trained model ({task_name})")
        
        # Test random policy
        print("\n📊 Testing random actions...")
        env = self.create_env(task_name, gui=False)
        obs = env.reset()
        random_reward = 0
        
        for step in range(max_steps):
            action = np.random.uniform(-1, 1, (1, 4))  # VecEnv expects batch dimension
            obs, reward, done, info = env.step(action)
            random_reward += float(reward[0])
            if done[0]:
                break
        
        env.close()
        
        # Test trained policy
        print("📊 Testing trained model...")
        model, version_name = self.load_best_model()
        if model is None:
            print("❌ Cannot compare - no trained model available")
            return
        
        env = self.create_env(task_name, gui=False)
        obs = env.reset()
        trained_reward = 0
        
        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            trained_reward += float(reward[0])
            if done[0]:
                break
        
        env.close()
        
        # Results
        improvement = trained_reward - random_reward
        improvement_percent = (improvement / abs(random_reward)) * 100 if random_reward != 0 else 0
        
        print("\n" + "=" * 60)
        print("📊 COMPARISON RESULTS")
        print("=" * 60)
        print(f"   Random actions: {random_reward:8.2f}")
        print(f"   Trained model:  {trained_reward:8.2f}")
        print(f"   Improvement:    {improvement:8.2f} ({improvement_percent:+.1f}%)")
        
        if improvement > 0:
            print("   ✅ Trained model performs better!")
        else:
            print("   ❌ Random actions perform better - model needs more training")
        print("=" * 60)
    
    def interactive_test_menu(self):
        """Interactive menu to test different tasks"""
        tasks = {
            '1': ("hover", "Hover in place"),
            '2': ("hover_extended", "Extended hovering"),
            '3': ("circle", "Circular trajectory"),
            '4': ("figure8", "Figure-8 pattern"),
            '5': ("waypoint_delivery", "Waypoint navigation"),
            '6': ("emergency_landing", "Emergency landing")
        }
        
        while True:
            print("\n" + "=" * 60)
            print("🎮 DRONE FLIGHT SIMULATOR - GUI TEST")
            print("=" * 60)
            print(f"GUI Mode: {'ENABLED' if self.gui else 'DISABLED'}")
            print("Choose a task to test:")
            for key, (task, description) in tasks.items():
                print(f"   {key}. {task:20} - {description}")
            print("   0. Exit")
            print("=" * 60)
            
            choice = input("Enter your choice (0-6): ").strip()
            
            if choice == '0':
                print("👋 Goodbye!")
                break
            elif choice in tasks:
                task_name, description = tasks[choice]
                print(f"\n🛫 Launching: {description}")
                self.test_single_task(task_name)
            else:
                print("❌ Invalid choice. Please try again.")
    
    def demo_all_tasks(self):
        """Run a demo of all tasks sequentially"""
        tasks = ["hover", "circle", "figure8", "waypoint_delivery"]
        
        print("\n" + "=" * 60)
        print("🎬 DRONE FLIGHT DEMO - ALL TASKS")
        print("=" * 60)
        print(f"GUI Mode: {'ENABLED' if self.gui else 'DISABLED'}")
        print("This will run all tasks sequentially.")
        print("Each task will run for up to 400 steps.")
        if self.gui:
            print("Close PyBullet window at any time to stop.")
        print("=" * 60)
        
        input("Press Enter to start the demo...")
        
        for task in tasks:
            print(f"\n>>> Starting {task.upper()} demo...")
            self.test_single_task(task, max_steps=400)
            
            # Ask to continue
            if task != tasks[-1]:  # Not the last task
                cont = input("\nContinue to next task? (y/n): ").lower().strip()
                if cont != 'y':
                    print("Demo stopped.")
                    break
        else:
            print("\n🎉 Demo completed all tasks!")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test trained drone RL model')
    parser.add_argument('--gui', type=str, default='True', 
                       help='Enable GUI visualization (True/False)')
    parser.add_argument('--task', type=str, default='circle',
                       help='Task to test: hover, circle, figure8, waypoint_delivery, etc.')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Maximum steps to run the test')
    parser.add_argument('--compare', action='store_true',
                       help='Compare with random actions')
    parser.add_argument('--interactive', action='store_true',
                       help='Launch interactive menu')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo of all tasks')
    
    return parser.parse_args()


def main():
    """Main function with command line argument support"""
    args = parse_arguments()
    
    # Convert string to boolean for GUI
    gui_enabled = args.gui.lower() in ['true', '1', 'yes', 'y']
    
    tester = DroneTester(gui=gui_enabled)
    
    print("🚀 DRONE RL TESTER")
    print("=" * 50)
    print(f"GUI Mode: {'ENABLED' if gui_enabled else 'DISABLED'}")
    
    # Check if models exist
    model, version = tester.load_best_model()
    if model is None:
        print("❌ No trained models found!")
        print("   Please train a model first using:")
        print("   python scripts/train_with_versioning.py")
        return
    
    print(f"✅ Best model loaded: {version}")
    
    # Execute based on command line arguments
    if args.interactive:
        tester.interactive_test_menu()
    elif args.demo:
        tester.demo_all_tasks()
    elif args.compare:
        tester.test_single_task(args.task, args.max_steps)
        tester.compare_with_random(args.task)
    else:
        # Default: test the specified task
        tester.test_single_task(args.task, args.max_steps)


# Quick test functions for direct import
def quick_test(task_name="circle", gui=True):
    """Quick function to test a specific task"""
    tester = DroneTester(gui=gui)
    return tester.test_single_task(task_name)


def test_circle(gui=True):
    """Test circle trajectory"""
    return quick_test("circle", gui)


def test_hover(gui=True):
    """Test hover task"""
    return quick_test("hover", gui)


def test_figure8(gui=True):
    """Test figure-8 trajectory"""
    return quick_test("figure8", gui)


if __name__ == "__main__":
    main()