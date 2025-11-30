
import os, sys
from stable_baselines3.common.env_util import DummyVecEnv

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)

from src.envs.fly_env import QuadcopterEnv
#from src.controllers.PPO import PPO
from stable_baselines3.ppo import PPO

from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

import numpy as np

class LiveRewardPlot(BaseCallback):
    def __init__(self, verbose=0, update_freq=1000):
        super().__init__(verbose)
        self.update_freq = update_freq
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.xs, self.rs = [], []
        (self.line,) = self.ax.plot(self.xs, self.rs, 'b-')
        self.ax.set_title("Rolling Mean Episode Reward")
        self.ax.set_xlabel("Timesteps")
        self.ax.set_ylabel("Episode Reward")
        self.ax.grid(True)
        self.recent_rewards = []

    def _on_step(self) -> bool:
        # Accumulate step rewards (only for env 0, for simplicity)
        # But better: use env's info or monitor wrapper for *episode* rewards
        # Here: fallback to step rewards if you must, but warn.
        try:
            # Get rewards — may be None or not in locals
            rewards = self.locals.get("rewards")
            if rewards is not None and len(rewards) > 0:
                # Use first env's step reward (not ideal, but works for demo)
                r = float(rewards[0])
                self.recent_rewards.append(r)
        except Exception as e:
            if self.verbose:
                print(f"Warning in callback: {e}")

        # Update plot every N steps
        if self.n_calls % self.update_freq == 0 and self.recent_rewards:
            # Use *mean* of recent rewards (~last 100 steps) for smoother plot
            mean_r = np.mean(self.recent_rewards[-100:])
            self.xs.append(self.num_timesteps)
            self.rs.append(mean_r)

            self.line.set_xdata(self.xs)
            self.line.set_ydata(self.rs)
            self.ax.relim()
            self.ax.autoscale_view()
            try:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            except Exception:
                plt.close(self.fig)
                plt.ioff()
                return False  # stop callback if plot fails
        return True

    def _on_training_end(self) -> None:
        plt.ioff()
        plt.close(self.fig)


def live_plot():
    import matplotlib.pyplot as plt
    plt.ion()
    fig, ax = plt.subplots()
    xs, rs = [], []
    line, = ax.plot(xs, rs)
    ax.set_title("Reward During Training")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")

    def update(step, reward, loss):
        xs.append(step)
        rs.append(reward)
        line.set_xdata(xs)
        line.set_ydata(rs)
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.001)

    return update

def make_env(task,**kwargs):
    return QuadcopterEnv(task=task,**kwargs)

if __name__ == "__main__":

    #env = make_env(task="circle")
    #env = make_env(task="circle", render_mode="none")

    #env = DummyVecEnv([lambda: env])
    env = DummyVecEnv([lambda: make_env("circle", render_mode="none") for _ in range(8)])
    

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=512,
        n_epochs=20,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.4,
        max_grad_norm=0.8,
    )

    
    plot_cb = LiveRewardPlot()

    model.learn(total_timesteps=1_000_000, progress_bar=True, callback=plot_cb)

    model.save("models/circle_policy_new")
