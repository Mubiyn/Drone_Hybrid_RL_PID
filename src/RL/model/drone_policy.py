import torch
import numpy as np
from .actor_critic import ActorCritic

class DronePolicy:
    def __init__(self, model_path, obs_dim=15, act_dim=3, device="cpu"):
        self.device = torch.device(device)
        self.model = ActorCritic(obs_dim, act_dim).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def act(self, obs: np.ndarray):
        """Deterministic action = tanh(mean)"""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            dist = self.model.actor_dist(obs_t)
            mean_action = torch.tanh(dist.mean)

        return mean_action.cpu().numpy()[0]
