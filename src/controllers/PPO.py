import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import os


class PPO(nn.Module):

    def __init__(
        self,
        observation_space,
        action_space,
        gamma=0.99,
        lam=0.95,
        learning_rate=3e-4,
        num_steps=2048,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        update_epochs=10,
        num_minibatches=4,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        # ==================================================
        # Dimensions
        # ==================================================
        self.obs_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]

        self.gamma = gamma
        self.lam = lam
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches

        hidden = 64

        # ==================================================
        # Policy network (Gaussian)
        # ==================================================
        self.actor_mean = nn.Sequential(
            nn.Linear(self.obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.act_dim)
        )

        # Log std is learned
        self.actor_logstd = nn.Parameter(torch.zeros(self.act_dim))

        # ==================================================
        # Value network
        # ==================================================
        self.critic = nn.Sequential(
            nn.Linear(self.obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        print(f"[PPO] Initialized continuous-action PPO (obs={self.obs_dim}, act={self.act_dim})")

    # ============================================================
    # Action sampling and value
    # ============================================================
    def get_action_and_value(self, obs):
        mean = self.actor_mean(obs)
        std = torch.exp(self.actor_logstd)
        dist = Normal(mean, std)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        value = self.critic(obs)

        return action, log_prob, value

    def evaluate_actions(self, obs, actions):
        mean = self.actor_mean(obs)
        std = torch.exp(self.actor_logstd)
        dist = Normal(mean, std)

        log_prob = dist.log_prob(actions).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        value = self.critic(obs).squeeze(-1)

        return log_prob, entropy, value

    # ============================================================
    # Predict (for evaluation)
    # ============================================================
    def predict(self, obs, deterministic=True):

        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)
        else:
            obs = obs.clone().detach().float()

        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

        device = next(self.parameters()).device
        obs = obs.to(device)

        with torch.no_grad():
            mean = self.actor_mean(obs)
            if deterministic:
                action = mean
            else:
                std = torch.exp(self.actor_logstd)
                dist = Normal(mean, std)
                action = dist.sample()

        return action.cpu().numpy(), None

    # ============================================================
    # Save & Load
    # ============================================================
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "ppo.pt"))
        print(f"[PPO] Saved → {path}/ppo.pt")

    def load(self, path):
        self.load_state_dict(torch.load(os.path.join(path, "ppo.pt")))
        print(f"[PPO] Loaded model from {path}")

    # ============================================================
    # Training Loop
    # ============================================================
    def learn(self, env, total_timesteps, plot_callback=None):

        print(f"[PPO] Training for {total_timesteps} timesteps...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        global_step = 0

        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).to(device)

        # ----------------------------
        # Main training loop
        # ----------------------------
        while global_step < total_timesteps:

            # rollout buffers
            obs_buffer = []
            act_buffer = []
            logp_buffer = []
            rew_buffer = []
            done_buffer = []
            val_buffer = []

            # ------------------------------------
            # RUN ENV FOR num_steps STEPS
            # ------------------------------------
            for _ in range(self.num_steps):

                obs_buffer.append(obs)

                with torch.no_grad():
                    action, logp, value = self.get_action_and_value(obs)

                act_buffer.append(action)
                logp_buffer.append(logp)
                val_buffer.append(value)

                # Action to env: (1, act_dim)
                act = action.cpu().numpy()
                if act.ndim == 1:
                    act = act.reshape(1, -1)

                next_obs, reward, terminated, truncated, _ = env.step(act)

                rew_buffer.append(torch.tensor(reward, dtype=torch.float32))
                done = torch.tensor(float(terminated or truncated), dtype=torch.float32)

                done_buffer.append(done)
                global_step += 1

                if terminated or truncated:
                    next_obs, _ = env.reset()

                obs = torch.tensor(next_obs, dtype=torch.float32).to(device)

            # Convert to tensors
            obs_tensor = torch.stack(obs_buffer).to(device)
            acts_tensor = torch.stack(act_buffer).to(device)
            logp_tensor = torch.stack(logp_buffer).to(device)
            rews_tensor = torch.stack(rew_buffer).to(device)
            dones_tensor = torch.stack(done_buffer).to(device)
            vals_tensor = torch.stack(val_buffer).squeeze(-1).to(device)

            # =========================================
            # Compute advantages (GAE)
            # =========================================
            with torch.no_grad():
                next_value = self.critic(obs).squeeze(-1)
                gae = 0
                advs = []

                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        next_val = next_value
                    else:
                        next_val = vals_tensor[t + 1]

                    delta = rews_tensor[t] + self.gamma * (1 - dones_tensor[t]) * next_val - vals_tensor[t]
                    gae = delta + self.gamma * self.lam * (1 - dones_tensor[t]) * gae
                    advs.insert(0, gae)

                adv_tensor = torch.stack(advs).to(device)
                ret_tensor = adv_tensor + vals_tensor

            # =========================================
            # PPO Optimization
            # =========================================
            batch_inds = torch.randperm(self.num_steps, device=device)
            minibatch_size = self.num_steps // self.num_minibatches

            for _ in range(self.update_epochs):
                for start in range(0, self.num_steps, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = batch_inds[start:end]

                    mb_obs = obs_tensor[mb_inds]
                    mb_actions = acts_tensor[mb_inds]
                    mb_adv = adv_tensor[mb_inds]
                    mb_returns = ret_tensor[mb_inds]
                    mb_logp_old = logp_tensor[mb_inds]

                    # normalize advantages
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    new_logp, entropy, new_values = self.evaluate_actions(mb_obs, mb_actions)

                    # policy ratio
                    ratio = torch.exp(new_logp - mb_logp_old)

                    # policy loss
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # value loss
                    value_loss = F.mse_loss(new_values, mb_returns)

                    # total loss
                    loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                    self.optimizer.step()

            # =============================
            # LIVE PLOT CALLBACK
            # =============================
            if plot_callback is not None:
                plot_callback(global_step, reward=rews_tensor.mean().item(), loss=loss.item())

        print("[PPO] Training complete.")
        return True
