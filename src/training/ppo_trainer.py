#!/usr/bin/env python3
import os
from typing import Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from src.training.reward_functions import get_reward_function
from src.training.custom_env import DroneTaskEnv


class PPOTrainer:
    def __init__(self, task_name: str, config: Optional[Dict[str, Any]] = None):
        self.task_name = task_name
        # Start with defaults, then update with user config
        self.config = self._default_config()
        if config:
            self.config.update(config)
        self.model: Optional[PPO] = None
        self.env: Optional[DummyVecEnv] = None

    def _default_config(self) -> Dict[str, Any]:
        return {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "total_timesteps": 100_000,
            "save_freq": 50_000,
            "eval_freq": 25_000,
            "n_eval_episodes": 5,
            "domain_randomization": True,
            "gui": False,
        }

    def create_env(self) -> DummyVecEnv:
        """Create training environment with task-specific reward function."""
        reward_fn = get_reward_function(self.task_name)

        env = DroneTaskEnv(
            task_name=self.task_name,
            reward_fn=reward_fn,
            domain_randomization=self.config["domain_randomization"],
            gui=self.config["gui"],
        )

        env = Monitor(env)
        self.env = DummyVecEnv([lambda: env])
        return self.env

    def create_model(self) -> PPO:
        """Create PPO model with configured hyperparameters."""
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=self.config["learning_rate"],
            n_steps=self.config["n_steps"],
            batch_size=self.config["batch_size"],
            n_epochs=self.config["n_epochs"],
            gamma=self.config["gamma"],
            gae_lambda=self.config["gae_lambda"],
            clip_range=self.config["clip_range"],
            ent_coef=self.config["ent_coef"],
            verbose=1,
            tensorboard_log=f"logs/ppo_{self.task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        return self.model

    def train(self, total_timesteps: Optional[int] = None) -> PPO:
        """Train the PPO model."""
        if self.env is None:
            self.create_env()
        if self.model is None:
            self.create_model()

        timesteps = total_timesteps or self.config["total_timesteps"]

        # Create callbacks
        checkpoint_dir = f"models/ppo/{self.task_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_callback = CheckpointCallback(
            save_freq=self.config["save_freq"],
            save_path=checkpoint_dir,
            name_prefix=f"ppo_{self.task_name}",
        )

        eval_env = DummyVecEnv(
            [
                lambda: Monitor(
                    DroneTaskEnv(
                        task_name=self.task_name,
                        reward_fn=get_reward_function(self.task_name),
                        domain_randomization=False,
                        gui=False,
                    )
                )
            ]
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=checkpoint_dir,
            log_path=checkpoint_dir,
            eval_freq=self.config["eval_freq"],
            n_eval_episodes=self.config["n_eval_episodes"],
            deterministic=True,
        )

        # Train
        print(f"\n{'='*60}")
        print(f"Training PPO on task: {self.task_name}")
        print(f"Total timesteps: {timesteps:,}")
        print(f"{'='*60}\n")

        self.model.learn(
            total_timesteps=timesteps, callback=[checkpoint_callback, eval_callback]
        )

        # Save final model
        final_path = os.path.join(checkpoint_dir, f"ppo_{self.task_name}_final.zip")
        self.model.save(final_path)
        print(f"\nFinal model saved to: {final_path}")

        return self.model

    def load_model(self, model_path: str) -> PPO:
        """Load a trained model."""
        self.model = PPO.load(model_path)
        return self.model

    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """Evaluate trained model."""
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")

        # Create a vectorized environment for evaluation, consistent with training
        eval_env = DummyVecEnv(
            [
                lambda: Monitor(
                    DroneTaskEnv(
                        task_name=self.task_name,
                        reward_fn=get_reward_function(self.task_name),
                        domain_randomization=False,
                        gui=False,
                    )
                )
            ]
        )

        # Use the built-in evaluation function for consistency
        episode_rewards, episode_lengths = evaluate_policy(  # type: ignore
            self.model,
            eval_env,
            n_eval_episodes=n_episodes,
            deterministic=True,
            return_episode_rewards=True,
        )
        eval_env.close()

        results = {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths),
        }

        print(f"\n{'='*60}")
        print(f"Evaluation Results ({n_episodes} episodes):")
        print(
            f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}"
        )
        print(
            f"Mean Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}"
        )
        print(f"{'='*60}\n")

        return results
