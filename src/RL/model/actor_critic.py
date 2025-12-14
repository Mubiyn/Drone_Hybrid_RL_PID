import torch
import torch.nn as nn
from torch.distributions.normal import Normal

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
        # learnable log_std per action
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = mlp(obs_dim, hidden_sizes, 1)

    def actor_dist(self, obs):
        """
        obs: (batch, obs_dim) or (obs_dim,) -> returns a Normal distribution over actions
        """
        mean = self.actor_mean(obs)
        std = torch.exp(self.log_std)
        # std shape broadcast to mean shape
        return Normal(mean, std)

    def value(self, obs):
        v = self.critic(obs)
        return v.squeeze(-1)