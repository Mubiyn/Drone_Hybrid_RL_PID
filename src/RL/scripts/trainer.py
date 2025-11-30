import os, sys
from stable_baselines3.common.env_util import DummyVecEnv

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)

from src.envs.focal_env import QuadcopterEnv  
from stable_baselines3.ppo import PPO

from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import numpy as np


MODEL_PATH = os.path.join(root, "models", "multi_policy.zip")


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
        try:
            rewards = self.locals.get("rewards")
            if rewards is not None and len(rewards) > 0:
                r = float(rewards[0])
                self.recent_rewards.append(r)
        except Exception as e:
            if self.verbose:
                print(f"Warning in callback: {e}")

        if self.n_calls % self.update_freq == 0 and self.recent_rewards:
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
                return False
        return True

    def _on_training_end(self) -> None:
        plt.ioff()
        plt.close(self.fig)


def make_env(task, **kwargs):
    """Create a single-task env instance."""
    return QuadcopterEnv(task=task, **kwargs)


def trainer(task, model=None, total_timesteps=500_000):

    print(f"\n===== TRAINING ON TASK: {task} =====")
    env = DummyVecEnv([lambda: make_env(task, render_mode="none") for _ in range(8)])

    if model is None:
        if os.path.exists(MODEL_PATH):
            print(f"[INF] Loading existing model from {MODEL_PATH}")
            model = PPO.load(MODEL_PATH, env=env)
        else:
            print("[INFO] Creating NEW PPO model")
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
    else:
        model.set_env(env)

    plot_cb = LiveRewardPlot()
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=plot_cb)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    env.close()

    return model


if __name__ == "__main__":
    task_list = ["circle", "figure8", "four_points", "goto"]

    model = None
    for task in task_list:
        model = trainer(task=task, model=model, total_timesteps=1_000_000)

    print("\n[DONE] Trained multi_policy on all tasks sequentially.")
