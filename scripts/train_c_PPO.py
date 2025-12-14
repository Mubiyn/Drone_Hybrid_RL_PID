
import os, sys
from stable_baselines3.common.env_util import DummyVecEnv

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)

from src.RL.envs.Drone_env import QuadcopterEnv
from src.RL.models.PPO import PPO,PPOConfig
import torch
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

import numpy as np


class LivePlotter:
    def __init__(self, theme="dark", smooth_window=20):
        plt.ion()

        # ----------- THEME -----------
        if theme == "dark":
            plt.style.use("dark_background")
            bg = "#121212"
            grid_color = "#444"
        else:
            plt.style.use("seaborn-v0_8")
            bg = "white"
            grid_color = "#DDD"

        self.smooth_window = smooth_window

        self.fig, self.ax = plt.subplots(1, 2, figsize=(14, 5), dpi=120)
        self.fig.patch.set_facecolor(bg)

        self.rew_x, self.rew_raw, self.rew_smooth = [], [], []
        (self.rew_line_raw,) = self.ax[0].plot([], [], color="#55aaff", alpha=0.3, label="Raw Reward")
        (self.rew_line_smooth,) = self.ax[0].plot([], [], color="#00eaff", linewidth=2.5, label="Smoothed Reward")

        self.ax[0].set_title("Mean Episodic Reward", fontsize=14)
        self.ax[0].set_xlabel("Training Update", fontsize=12)
        self.ax[0].set_ylabel("Reward", fontsize=12)
        self.ax[0].grid(True, color=grid_color)
        self.ax[0].legend(fontsize=10)

        self.loss_x, self.pi_y, self.vf_y = [], [], []
        (self.pi_line,) = self.ax[1].plot([], [], linewidth=2, color="#ff6f69", label="Policy Loss")
        (self.vf_line,) = self.ax[1].plot([], [], linewidth=2, color="#ffcc5c", label="Value Loss")

        self.ax[1].set_title("Loss Curves", fontsize=14)
        self.ax[1].set_xlabel("Training Update", fontsize=12)
        self.ax[1].set_ylabel("Loss", fontsize=12)
        self.ax[1].grid(True, color=grid_color)
        self.ax[1].legend(fontsize=10)

        plt.tight_layout()

    
    def smooth(self, data):
        if len(data) < self.smooth_window:
            return data
        kernel = np.ones(self.smooth_window) / self.smooth_window
        return np.convolve(data, kernel, mode='same')

    #       CALLBACK UPDATE
    def __call__(self, update_idx, pi_loss, vf_loss, entropy, mean_reward):

        # REWARD PLOT 
        self.rew_x.append(update_idx)
        self.rew_raw.append(mean_reward)

        smooth_vals = self.smooth(self.rew_raw)
        self.rew_smooth = smooth_vals

        self.rew_line_raw.set_data(self.rew_x, self.rew_raw)
        self.rew_line_smooth.set_data(self.rew_x, smooth_vals)

        self.ax[0].relim()
        self.ax[0].autoscale_view()

        #  LOSS PLOT 
        self.loss_x.append(update_idx)
        self.pi_y.append(pi_loss)
        self.vf_y.append(vf_loss)

        self.pi_line.set_data(self.loss_x, self.pi_y)
        self.vf_line.set_data(self.loss_x, self.vf_y)

        self.ax[1].relim()
        self.ax[1].autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def save(self, filename="training_plot"):
            self.fig.savefig(f"results/{filename}.png", dpi=300, bbox_inches="tight")
            self.fig.savefig(f"results/{filename}.svg", dpi=300, bbox_inches="tight")
            print(f"[PLOT] Saved '{filename}.png' and '{filename}.svg'")

def make_env(task,**kwargs):
    return QuadcopterEnv(task=task,**kwargs)

if __name__ == "__main__":

    env = QuadcopterEnv(task="circle", render_mode="none")
    cfg = PPOConfig(
        total_timesteps=500_000,
    )

    plotter = LivePlotter()

    model = PPO(env=env, cfg=cfg, callback=plotter)
    model.train()

    model.save("models/circle_model.zip")
    plotter.save("training_plot")
