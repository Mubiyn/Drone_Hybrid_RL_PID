import os 
import sys
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# Import your env
# -------------------------------------------------
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)


# =======================
#  Actor-Critic Network
# =======================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()

        def mlp(in_dim, hidden_sizes, out_dim):
            layers = []
            last_dim = in_dim
            for h in hidden_sizes:
                layers.append(nn.Linear(last_dim, h))
                layers.append(nn.ReLU())
                last_dim = h
            layers.append(nn.Linear(last_dim, out_dim))
            return nn.Sequential(*layers)

        self.actor_mean = mlp(obs_dim, hidden_sizes, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))    # learnable std
        self.critic = mlp(obs_dim, hidden_sizes, 1)

    def actor_dist(self, obs):
        mean = self.actor_mean(obs)
        std = torch.exp(self.log_std)
        return Normal(mean, std)

    def value(self, obs):
        return self.critic(obs).squeeze(-1)



# =======================
#  PPO Config / Rollout Buffer
# =======================
@dataclass
class PPOConfig:
    total_timesteps: int = 500_000
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.4
    max_grad_norm: float = 0.8
    n_steps: int = 4096
    batch_size: int = 512
    n_epochs: int = 20
    device: str = "cpu"


class RolloutBuffer:
    def __init__(self, n_steps, obs_dim, act_dim, device="cpu"):
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        self.reset()

    def reset(self):
        self.observations = np.zeros((self.n_steps, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.n_steps, self.act_dim), dtype=np.float32)
        self.raw_actions = np.zeros((self.n_steps, self.act_dim), dtype=np.float32)
        self.log_probs = np.zeros((self.n_steps,), dtype=np.float32)
        self.rewards = np.zeros((self.n_steps,), dtype=np.float32)
        self.dones = np.zeros((self.n_steps,), dtype=np.float32)
        self.values = np.zeros((self.n_steps,), dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action, raw_action, log_prob, reward, done, value):
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.raw_actions[self.ptr] = raw_action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        adv = np.zeros_like(self.rewards)
        last_gae = 0

        for t in reversed(range(self.n_steps)):
            nonterminal = 1 - self.dones[t]
            next_value = last_value if t == self.n_steps - 1 else self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * nonterminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
            adv[t] = last_gae

        returns = adv + self.values

        # normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        self.advantages = adv.astype(np.float32)
        self.returns = returns.astype(np.float32)

    def get_torch_batches(self, batch_size, device):
        idxs = np.arange(self.n_steps)
        np.random.shuffle(idxs)

        for start in range(0, self.n_steps, batch_size):
            end = start + batch_size
            b = idxs[start:end]

            yield (
                torch.as_tensor(self.observations[b], device=device),
                torch.as_tensor(self.actions[b], device=device),
                torch.as_tensor(self.raw_actions[b], device=device),
                torch.as_tensor(self.log_probs[b], device=device),
                torch.as_tensor(self.returns[b], device=device),
                torch.as_tensor(self.advantages[b], device=device),
            )



# =======================
#  PPO Algorithm
# =======================
class PPO:
    def __init__(self, env, cfg: PPOConfig, callback=None):
        self.env = env
        self.cfg = cfg
        self.callback = callback

        self.device = torch.device(cfg.device)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        self.model = ActorCritic(obs_dim, act_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate)

        self.buffer = RolloutBuffer(cfg.n_steps, obs_dim, act_dim, self.device)

    # ------------------------
    # Action selection (SB3-style)
    # ------------------------
    def select_action(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            dist = self.model.actor_dist(obs_t)
            raw_action = dist.rsample()
            squashed = torch.tanh(raw_action)

            # log_prob correction
            logp = dist.log_prob(raw_action)
            logp = logp - torch.log(1 - squashed.pow(2) + 1e-6)
            logp = logp.sum(-1)

            value = self.model.value(obs_t)

        return (
            squashed.cpu().numpy()[0],
            raw_action.cpu().numpy()[0],
            logp.cpu().numpy()[0],
            value.cpu().numpy()[0],
        )

    # ------------------------
    # Training loop
    # ------------------------
    def train(self):
        cfg = self.cfg
        obs, _ = self.env.reset()
        total_steps = 0

        while total_steps < cfg.total_timesteps:
            self.buffer.reset()

            # -------- COLLECT ROLLOUT --------
            for _ in range(cfg.n_steps):
                action, raw_action, logp, value = self.select_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.buffer.add(obs, action, raw_action, logp, reward, done, value)

                obs = next_obs
                total_steps += 1

                if done:
                    obs, _ = self.env.reset()

                if total_steps >= cfg.total_timesteps:
                    break

            # compute final value
            with torch.no_grad():
                last_val = self.model.value(
                    torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                ).cpu().numpy()[0]

            self.buffer.compute_returns_and_advantages(last_val, cfg.gamma, cfg.gae_lambda)

            # -------- UPDATE --------
            p_loss, v_loss, ent = self.update()

            if self.callback:
                self.callback(total_steps, p_loss, v_loss, ent, np.mean(self.buffer.rewards))

            print(f"[UPDATE] steps={total_steps} pi_loss={p_loss:.3f} vf_loss={v_loss:.3f}")

        print("TRAINING COMPLETE.")

    # ------------------------
    # PPO Update
    # ------------------------
    def update(self):
        cfg = self.cfg
        last_pi, last_vf, last_ent = 0, 0, 0

        for _ in range(cfg.n_epochs):
            for obs, actions, raw_actions, old_logp, returns, adv in self.buffer.get_torch_batches(
                cfg.batch_size, self.device
            ):
                dist = self.model.actor_dist(obs)
                values = self.model.value(obs)

                # recompute log_probs
                new_logp = dist.log_prob(raw_actions)
                new_logp = new_logp - torch.log(1 - actions.pow(2) + 1e-6)
                new_logp = new_logp.sum(-1)

                entropy = dist.entropy().sum(-1).mean()

                # ratio
                ratio = torch.exp(new_logp - old_logp)

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.smooth_l1_loss(values, returns)

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
    def predict(self, obs):
        """Deterministic policy for evaluation (mean of Gaussian)."""
        self.model.eval()
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            dist = self.model.actor_dist(obs_t)

            # deterministic action = mean of Gaussian
            mean_action = torch.tanh(dist.mean)

        return mean_action.cpu().numpy()[0]
