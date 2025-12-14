import os
import sys
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .actor_critic import ActorCritic
from .buffer import RolloutBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.4
    max_grad_norm: float = 0.8
    n_steps: int = 2048         # steps per environment (per rollout)
    batch_size: int = 256       # minibatch size from flattened rollouts
    n_epochs: int = 10
    device: str = "cpu"


class PPO:
    def __init__(self, env, cfg: PPOConfig, callback=None):
        self.env = env
        self.cfg = cfg
        self.callback = callback

        self.device = torch.device(cfg.device)

        # Robustly determine single-observation dimension even if env is vectorized.
        obs_space = getattr(env, "single_observation_space", None)
        if obs_space is None:
            obs_space = getattr(env, "observation_space", None)
        if obs_space is None:
            raise RuntimeError("Unable to determine observation space from env.")

        obs_shape = getattr(obs_space, "shape", None)
        if obs_shape is None:
            raise RuntimeError("observation_space has no shape attribute.")
        obs_dim = int(obs_shape[-1])

        # Robustly determine single-action dimension (handle VectorEnv batched spaces)
        act_space = getattr(env, "single_action_space", None)
        if act_space is None:
            act_space = getattr(env, "action_space", None)
        if act_space is None:
            raise RuntimeError("Unable to determine action space from env.")
        act_shape = getattr(act_space, "shape", None)
        if act_shape is None:
            raise RuntimeError("action_space has no shape attribute.")
        act_dim = int(act_shape[-1])

        # detect vectorized envs (number of parallel envs)
        self.n_envs = getattr(env, "num_envs", 1)
        if self.n_envs == 1 and hasattr(env, "envs"):
            try:
                self.n_envs = len(env.envs)
            except Exception:
                pass

        # Build model and optimizer
        self.model = ActorCritic(obs_dim, act_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate)

        # Buffer configured for n_steps per env and number of envs
        self.buffer = RolloutBuffer(cfg.n_steps, self.n_envs, obs_dim, act_dim, self.device)

    def select_action(self, obs):
        """
        Accepts obs shaped (n_envs, obs_dim) or (obs_dim,)
        Returns tuple of numpy arrays: actions (n_envs, act_dim), raw_actions, logp (n_envs,), values (n_envs,)
        """
        is_batched = True
        obs_arr = np.asarray(obs)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr[None, :]
            is_batched = False

        obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            dist = self.model.actor_dist(obs_t)
            raw_action = dist.rsample()  # reparameterized sample: shape (batch, act_dim)
            squashed = torch.tanh(raw_action)
            # log_prob correction for tanh
            logp = dist.log_prob(raw_action)
            # sum over action dims
            logp = logp - torch.log(1 - squashed.pow(2) + 1e-6)
            logp = logp.sum(-1)
            value = self.model.value(obs_t)

        actions_np = squashed.cpu().numpy()
        raw_np = raw_action.cpu().numpy()
        logp_np = logp.cpu().numpy()
        value_np = value.cpu().numpy()

        if not is_batched:
            return actions_np[0], raw_np[0], float(logp_np[0]), float(value_np[0])
        return actions_np, raw_np, logp_np, value_np

    def train(self, total_timesteps):
        cfg = self.cfg
        # reset env and initial obs
        res = self.env.reset()
        if isinstance(res, tuple):
            obs, _ = res
        else:
            obs = res

        total_steps = 0

        while total_steps < total_timesteps:
            self.buffer.reset()

            # COLLECT ROLLOUT: cfg.n_steps steps per env -> total samples = n_envs * n_steps
            for _ in range(cfg.n_steps):
                action, raw_action, logp, value = self.select_action(obs)
                res = self.env.step(action)
                # res can be tuple (obs, reward, terminated, truncated, info)
                if isinstance(res, tuple) and len(res) == 5:
                    next_obs, reward, terminated, truncated, _ = res
                else:
                    next_obs = res[0]
                    reward = res[1]
                    terminated = res[2]
                    truncated = res[3]

                reward_arr = np.asarray(reward)
                done_arr = np.asarray(terminated | truncated)

                # add data to buffer (batch add)
                self.buffer.add(obs, action, raw_action, logp, reward_arr, done_arr, value)

                obs = next_obs
                total_steps += max(1, self.n_envs)

                # If single env and done, reset it immediately
                if self.n_envs == 1 and np.any(done_arr):
                    res = self.env.reset()
                    if isinstance(res, tuple):
                        obs, _ = res
                    else:
                        obs = res

                if total_steps >= total_timesteps:
                    break

            # compute final value for each env given last obs
            obs_for_val = obs
            if isinstance(obs_for_val, np.ndarray) and obs_for_val.ndim == 1:
                obs_for_val = obs_for_val[None, :]

            with torch.no_grad():
                last_val = self.model.value(torch.tensor(obs_for_val, dtype=torch.float32, device=self.device)).cpu().numpy()
            if last_val.ndim > 1:
                last_val = last_val.reshape(-1)

            self.buffer.compute_returns_and_advantages(last_val, cfg.gamma, cfg.gae_lambda)

            # -------- UPDATE --------
            p_loss, v_loss, ent = self.update()

            if self.callback:
                mean_reward = float(np.mean(self.buffer.rewards))
                self.callback(total_steps, p_loss, v_loss, ent, mean_reward)

            print(f"[UPDATE] steps={total_steps} pi_loss={p_loss:.3f} vf_loss={v_loss:.3f} ent={ent:.5f}")

        print("TRAINING COMPLETE.")

    def update(self):
        cfg = self.cfg
        last_pi, last_vf, last_ent = 0.0, 0.0, 0.0

        total_samples = self.buffer.get_flat_size()
        batch_size = min(cfg.batch_size, total_samples)

        for _ in range(cfg.n_epochs):
            for obs_b, actions_b, raw_actions_b, old_logp_b, returns_b, adv_b in self.buffer.get_torch_batches(batch_size, self.device):
                dist = self.model.actor_dist(obs_b)
                values = self.model.value(obs_b)

                # recompute log_probs for raw actions
                new_logp = dist.log_prob(raw_actions_b)
                squashed = actions_b
                new_logp = new_logp - torch.log(1 - squashed.pow(2) + 1e-6)
                new_logp = new_logp.sum(-1)

                entropy = dist.entropy().sum(-1).mean()

                # ratio
                ratio = torch.exp(new_logp - old_logp_b)

                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.smooth_l1_loss(values, returns_b)

                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                last_pi = policy_loss.item()
                last_vf = value_loss.item()
                last_ent = entropy.item()

        return last_pi, last_vf, last_ent

    # ------------------------
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)

    def predict(self, obs, deterministic=True):
        """
        Deterministic policy for evaluation (mean action). Accepts obs shape (obs_dim,) or (n_envs, obs_dim).
        Returns action array of shape (act_dim,) or (n_envs, act_dim).
        """
        self.model.eval()
        obs_arr = np.asarray(obs)
        was_batched = True
        if obs_arr.ndim == 1:
            obs_arr = obs_arr[None, :]
            was_batched = False

        with torch.no_grad():
            obs_t = torch.tensor(obs_arr, dtype=torch.float32, device=self.device)
            dist = self.model.actor_dist(obs_t)
            if deterministic:
                raw = dist.mean
            else:
                raw = dist.rsample()
            action = torch.tanh(raw)

        action_np = action.cpu().numpy()
        if not was_batched:
            return action_np[0]
        return action_np