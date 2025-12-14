import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import time
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, root)
import torch

from src.RL.envs.Drone_env import QuadcopterEnv
from src.RL.models.PPO import PPO, PPOConfig


def test_model(model_path, task="circle", max_steps=5000, record_video=False):

    print(f"\n[TEST] Loading PPO model from: {model_path}")

    # --- create environment ---
    env = QuadcopterEnv(task=task, render_mode="human")
    obs, _ = env.reset()

    # --- load model ---
    cfg = PPOConfig(device="cuda" if torch.cuda.is_available() else "cpu")
    model = PPO(env, cfg)
    model.load(model_path)

    print("[TEST] Model loaded successfully!")

    drone_path = []

    # --- Optional: record video ---
    if record_video:
        log_id = p.startStateLogging(
            p.STATE_LOGGING_VIDEO_MP4,
            f"{task}_test_flight.mp4"
        )
        print("[TEST] Recording video...")

    # --- simulation loop ---
    for step in range(max_steps):

        action = model.predict(obs)       # deterministic action
        obs, reward, terminated, truncated, _ = env.step(action)

        # track drone position
        pos = env._getDroneStateVector(0)[0:3]
        drone_path.append(pos)

        if terminated or truncated:
            print(f"[TEST] Episode ended at step {step}")
            break
        
        #time.sleep(1/80)
    
    if record_video:
        p.stopStateLogging(log_id)
        print(f"[TEST] Video saved as {task}_test_flight.mp4")

    drone_path = np.array(drone_path)

    # --- Plot XY Path ---
    wp = np.array(env.waypoints)

    plt.figure(figsize=(6,6))
    plt.plot(wp[:,0], wp[:,1], 'r--', label="Waypoint Path")
    plt.plot(drone_path[:,0], drone_path[:,1], 'b-', label="Drone Path")
    plt.title(f"2D Path – {task}")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

    # --- Plot 3D Path ---
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(wp[:,0], wp[:,1], wp[:,2], 'r--')
    ax.plot(drone_path[:,0], drone_path[:,1], drone_path[:,2], 'b-')
    ax.set_title(f"3D Trajectory – {task}")
    plt.show()

    print("\n[TEST] Finished testing.")


if __name__ == "__main__":
    test_model("models/circle_model.zip", task="circle", max_steps=12000, record_video=True)
