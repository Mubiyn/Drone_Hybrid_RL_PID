
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import sys
import torch

import sys
import os
import numpy as np

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)

from src.utils.RL import PPO
from src.envs.current_env import UAVPointToPointEnv
from src.utils.sim_wrapper import HoverSimWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    sim = HoverSimWrapper(gui=False)
    env = UAVPointToPointEnv(sim=sim, gui=False)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ppo = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        update_epochs=5,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        batch_size=1024,
        minibatch_size=256,
    )

    total_steps = 1_000_000
    steps_collected = 0

    log_dir = "logs/uav_nav"
    os.makedirs(log_dir, exist_ok=True)
    model_path = os.path.join(log_dir, "ppo_uav_new.pt")

    episode = 0
    start_time = datetime.now().replace(microsecond=0)
    print("Training started at:", start_time)

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    rewards_history = []
    smoothed_history = []
    plot_line, = ax.plot([], [], label="Episode Reward")
    smooth_line, = ax.plot([], [], label="Smoothed (MA=20)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("UAV PPO Training Reward Curve")
    ax.legend()
    fig.show()

    def update_plot():
        plot_line.set_xdata(np.arange(len(rewards_history)))
        plot_line.set_ydata(rewards_history)

        if len(rewards_history) >= 20:
            smoothed = np.convolve(rewards_history, np.ones(20)/20, mode="valid")
            smoothed_history[:] = smoothed
            smooth_line.set_xdata(np.arange(len(smoothed_history)))
            smooth_line.set_ydata(smoothed_history)

        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

    while steps_collected < total_steps:
        obs, info = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action = ppo.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            reward = reward / 10.0  # reward normalization for stability

            ppo.store_reward_done(reward, done)
            ep_reward += reward
            steps_collected += 1
            obs = next_obs

            if len(ppo.rewards_buf) >= ppo.batch_size:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                    _, last_v = ppo.actor_critic.get_policy_value(obs_t)
                    last_value = float(last_v.cpu().item())

                ppo.update(last_value=last_value)
                print(f"[{steps_collected}] PPO update done")
                break  # IMPORTANT: end episode after update

            if steps_collected >= total_steps:
                break

        episode += 1
        print(f"Episode {episode} | reward = {ep_reward:.2f}")

        rewards_history.append(ep_reward)
        update_plot()

        if episode % 50 == 0:
            ppo.save(model_path)
            print(f"Checkpoint saved at {model_path}")

    env.close()
    end_time = datetime.now().replace(microsecond=0)
    print("Training finished at:", end_time)
    print("Total time:", end_time - start_time)
    ppo.save(model_path)
    print("Final model saved.")

if __name__ == "__main__":
    train()