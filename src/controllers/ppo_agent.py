# ppo_agent.py
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    """
    Shared MLP backbone with separate policy and value heads.
    Policy outputs mean; log_std is a learned parameter.
    Continuous actions in [-1, 1] via tanh squashing.
    """
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.policy_mean = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)

        # log_std is a learned parameter (one per action dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        raise NotImplementedError

    def get_policy_value(self, obs):
        """
        obs: [batch, state_dim]
        returns: dist (Normal), value [batch]
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        h = self.backbone(obs)
        mean = self.policy_mean(h)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)

        value = self.value_head(h).squeeze(-1)
        return dist, value

    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        dist, value = self.get_policy_value(obs_t)  # value shape [1]

        raw_action = dist.rsample()          # [1, action_dim]
        action = torch.tanh(raw_action)      # [-1,1]

        log_prob = dist.log_prob(raw_action).sum(-1)   # [1]
        log_prob -= torch.log(1 - action.pow(2) + 1e-8).sum(-1)

        # squeeze batch dim
        action = action.squeeze(0)           # [action_dim]
        log_prob = log_prob.squeeze(0)       # scalar tensor
        value = value.squeeze(0)             # scalar tensor

        return action.detach().cpu().numpy(), log_prob.detach(), value.detach()

    def act_deterministic(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        dist, value = self.get_policy_value(obs_t)

        mean = dist.mean          # [1, action_dim]
        action = torch.tanh(mean) # squash to [-1,1]
        action = action.squeeze(0)

        return action.detach().cpu().numpy(), value.detach()

    def evaluate_actions(self, obs, actions):
        """
        For PPO update:
          obs: [batch, state_dim]
          actions: [batch, action_dim] already tanh-squashed in [-1,1]
        Returns:
          log_probs, entropy, values
        """
        dist, values = self.get_policy_value(obs)

        # Inverse tanh to get raw_action from squashed action
        eps = 1e-6
        clipped_actions = torch.clamp(actions, -1 + eps, 1 - eps)
        raw_actions = 0.5 * torch.log((1 + clipped_actions) / (1 - clipped_actions))

        log_prob = dist.log_prob(raw_actions).sum(-1)
        log_prob -= torch.log(1 - clipped_actions.pow(2) + 1e-8).sum(-1)

        entropy = dist.entropy().sum(-1)

        return log_prob, entropy, values.squeeze(-1)


class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        update_epochs=10,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        batch_size=2048,
        minibatch_size=256,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.update_epochs = update_epochs
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size

        self.actor_critic = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)

        # rollout storage
        self.reset_buffer()

    def reset_buffer(self):
        self.obs_buf = []
        self.actions_buf = []
        self.logprobs_buf = []
        self.rewards_buf = []
        self.dones_buf = []
        self.values_buf = []

    def select_action(self, obs):
        """
        Obs is 1D np array.
        Returns action (np array in [-1,1] with shape (action_dim,)).
        """
        action, logprob, value = self.actor_critic.act(obs)

        self.obs_buf.append(obs.copy())
        self.actions_buf.append(action.copy())             # shape (action_dim,)
        self.logprobs_buf.append(float(logprob.cpu()))     # scalar
        self.values_buf.append(float(value.cpu()))         # scalar

        return action

    def store_reward_done(self, reward, done):
        self.rewards_buf.append(reward)
        self.dones_buf.append(done)

    def _compute_gae(self, last_value=0.0):
        """
        Compute GAE advantages and returns from the rollout.
        """
        rewards = np.array(self.rewards_buf, dtype=np.float32)
        dones = np.array(self.dones_buf, dtype=np.bool_)
        values = np.array(self.values_buf, dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(self, last_value=0.0):
        """
        Run PPO update after collecting batch_size steps.
        """
        # convert buffer to tensors
        obs = torch.as_tensor(np.array(self.obs_buf), dtype=torch.float32, device=device)
        actions = torch.as_tensor(np.array(self.actions_buf), dtype=torch.float32, device=device)
        old_logprobs = torch.as_tensor(np.array(self.logprobs_buf), dtype=torch.float32, device=device)
        values = torch.as_tensor(np.array(self.values_buf), dtype=torch.float32, device=device)

        advantages, returns = self._compute_gae(last_value=last_value)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=device)

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n_steps = obs.shape[0]
        idxs = np.arange(n_steps)

        for epoch in range(self.update_epochs):
            np.random.shuffle(idxs)

            for start in range(0, n_steps, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = idxs[start:end]

                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logprobs = old_logprobs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_values = values[mb_idx]

                logprobs, entropy, values_pred = self.actor_critic.evaluate_actions(mb_obs, mb_actions)

                # ratio = new / old
                ratios = torch.exp(logprobs - mb_old_logprobs)

                # surrogate objective
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_coef, 1 + self.clip_coef) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # value loss
                value_loss = (values_pred - mb_returns).pow(2).mean()

                # entropy loss
                entropy_loss = entropy.mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.reset_buffer()
    def select_action_eval(self, obs):
        action, value = self.actor_critic.act_deterministic(obs)
        return action

    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path, map_location=device))
        self.actor_critic.to(device)
