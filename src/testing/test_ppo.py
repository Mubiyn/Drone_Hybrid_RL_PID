import time
import numpy as np
import sys, os
import matplotlib.pyplot as plt

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, root)

from src.envs.fly_env import QuadcopterEnv
from stable_baselines3.ppo import PPO
from stable_baselines3.common.env_util import DummyVecEnv


def test_flight_with_plot(model_path, task="circle", max_steps=10000):

    print(f"[TEST] Loading model from: {model_path}")

    # Create env FIRST
    env = DummyVecEnv([lambda: QuadcopterEnv(task, render_mode="human")])

    # Load model WITH env
    model = PPO.load(model_path)

    obs= env.reset()

    ideal_path = np.array(env.envs[0].waypoints)
    drone_path = []

    for step in range(max_steps):

        action, _ = model.predict(obs, deterministic=True)

        obs, rewards, dones, infos = env.step(action)

        pos = env.envs[0]._getDroneStateVector(0)[0:3]
        drone_path.append(pos)

        if dones[0]:
            print(f"[TEST] Episode ended at step {step}")
            break

        time.sleep(1/80)

    drone_path = np.array(drone_path)

    # ========================== PLOTS ==========================
    plt.figure(figsize=(8, 8))
    plt.plot(ideal_path[:, 0], ideal_path[:, 1], 'r--', label="Ideal Path")
    plt.plot(drone_path[:, 0], drone_path[:, 1], 'b-', label="Drone Path")
    plt.scatter(ideal_path[:, 0], ideal_path[:, 1], c='red', s=20)
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ideal_path[:, 0], ideal_path[:, 1], ideal_path[:, 2], 'r--')
    ax.plot(drone_path[:, 0], drone_path[:, 1], drone_path[:, 2], 'b-')
    plt.show()

    print("[TEST] Completed trajectory analysis.")


if __name__ == "__main__":
    test_flight_with_plot("models/circle_policy_new.zip", task="circle", max_steps=10000)
