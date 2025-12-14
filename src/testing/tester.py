import time
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import pybullet as p
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, root)

#from stable_baselines3.ppo import PPO
from src.RL.models.PPO import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from src.envs.focal_env import QuadcopterEnv


def test_task(model_path, task="circle", max_steps=5000):

    print(f"[TEST] Loading model from: {model_path}")
    print(f"[TEST] Selected task: {task}")

    # Create environment for the SINGLE TASK
    env = DummyVecEnv([
        lambda: QuadcopterEnv(task=task, render_mode="human")
    ])

    # Load model
    model = PPO.load(path=model_path)#, env=env,verbose=1)

    # Reset env
    obs = env.reset()

    # Get ideal path for plotting
    ideal_path = np.array(env.envs[0].waypoints)
    drone_path = []
    
    # log_id = p.startStateLogging(
    #     p.STATE_LOGGING_VIDEO_MP4,
    #     f"{task}_flight.mp4"
    # )

    for step in range(max_steps):

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        pos = env.envs[0]._getDroneStateVector(0)[0:3]
        drone_path.append(pos)

        if done[0]:
            print(f"[TEST] Episode ended at step {step}")
            break

        time.sleep(1/80)
    
    #p.stopStateLogging(log_id)
    #print(f"[TEST] Recorded video saved as: {task}_flight.mp4")

    drone_path = np.array(drone_path)

    # -------------------- 2D XY Plot --------------------
    plt.figure(figsize=(7,7))
    plt.plot(ideal_path[:,0], ideal_path[:,1], 'r--', label="Ideal path")
    plt.plot(drone_path[:,0], drone_path[:,1], 'b-', label="Drone")
    plt.scatter(ideal_path[:,0], ideal_path[:,1], c='red', s=20)
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title(f"2D XY Path – {task}")
    plt.show()

    # -------------------- 3D Plot --------------------
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ideal_path[:,0], ideal_path[:,1], ideal_path[:,2], 'r--')
    ax.plot(drone_path[:,0], drone_path[:,1], drone_path[:,2], 'b-')
    ax.set_title(f"3D Trajectory – {task}")
    plt.show()

    print("[TEST] Finished.")
    
if __name__ == "__main__":

    test_task("models/new_circle_policy", task="circle", max_steps=7000)
