import numpy as np
import torch

class RolloutBuffer:
    """
    Rollout buffer that supports vectorized envs.
    Stores arrays with shape (n_steps, n_envs, dim) and yields flattened minibatches.
    """

    def __init__(self, n_steps, n_envs, obs_dim, act_dim, device="cpu"):
        self.n_steps = int(n_steps)       # steps per environment
        self.n_envs = int(n_envs)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        self.reset()

    def reset(self):
        T, N = self.n_steps, self.n_envs
        self.observations = np.zeros((T, N, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((T, N, self.act_dim), dtype=np.float32)
        self.raw_actions = np.zeros((T, N, self.act_dim), dtype=np.float32)
        self.log_probs = np.zeros((T, N), dtype=np.float32)
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.dones = np.zeros((T, N), dtype=np.float32)
        self.values = np.zeros((T, N), dtype=np.float32)
        self.ptr = 0  # current time index (0..T-1)

    def add(self, obs, action, raw_action, log_prob, reward, done, value):
        """
        Add one time-step of data for all envs.
        All inputs can be batched with leading dim = n_envs, or single env scalars.
        Shapes expected:
          obs: (n_envs, obs_dim)
          action/raw_action: (n_envs, act_dim)
          log_prob/reward/done/value: (n_envs,)
        """
        if self.ptr >= self.n_steps:
            raise IndexError("RolloutBuffer overflow: tried to add more than n_steps")

        obs = np.asarray(obs)
        action = np.asarray(action)
        raw_action = np.asarray(raw_action)
        log_prob = np.asarray(log_prob)
        reward = np.asarray(reward)
        done = np.asarray(done)
        value = np.asarray(value)

        # ensure env dimension exists
        if obs.ndim == 1:
            obs = obs[None, :]
        if action.ndim == 1:
            action = action[None, :]
        if raw_action.ndim == 1:
            raw_action = raw_action[None, :]

        self.observations[self.ptr] = obs.astype(np.float32)
        self.actions[self.ptr] = action.astype(np.float32)
        self.raw_actions[self.ptr] = raw_action.astype(np.float32)
        self.log_probs[self.ptr] = log_prob.astype(np.float32)
        self.rewards[self.ptr] = reward.astype(np.float32)
        self.dones[self.ptr] = done.astype(np.float32)
        self.values[self.ptr] = value.astype(np.float32)
        self.ptr += 1

    def compute_returns_and_advantages(self, last_values, gamma, gae_lambda):
        """
        Compute GAE per environment across time.
        last_values should be shape (n_envs,) - the value estimate for the obs after last step.
        """
        T, N = self.n_steps, self.n_envs
        adv = np.zeros((T, N), dtype=np.float32)
        last_gae = np.zeros(N, dtype=np.float32)

        # ensure last_values shape
        last_values = np.asarray(last_values).astype(np.float32)
        assert last_values.shape[0] == N

        for t in reversed(range(T)):
            nonterminal = 1.0 - self.dones[t]
            next_values = last_values if t == T - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_values * nonterminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
            adv[t] = last_gae

        returns = adv + self.values

        # flatten advantages across time and envs for normalization
        flat_adv = adv.reshape(-1)
        adv_mean = flat_adv.mean()
        adv_std = flat_adv.std() + 1e-8
        adv = (adv - adv_mean) / adv_std

        self.advantages = adv.astype(np.float32)    # shape (T, N)
        self.returns = returns.astype(np.float32)   # shape (T, N)

    def get_torch_batches(self, batch_size, device):
        """
        Yield minibatches as torch tensors. The underlying storage is flattened across time and envs.
        Returns: obs, actions, raw_actions, old_logp, returns, adv  (all torch tensors on device)
        """
        T, N = self.n_steps, self.n_envs
        total = T * N
        idxs = np.arange(total)
        np.random.shuffle(idxs)

        # flatten arrays
        obs_flat = self.observations.reshape(total, self.obs_dim)
        acts_flat = self.actions.reshape(total, self.act_dim)
        raw_flat = self.raw_actions.reshape(total, self.act_dim)
        logp_flat = self.log_probs.reshape(total)
        ret_flat = self.returns.reshape(total)
        adv_flat = self.advantages.reshape(total)

        for start in range(0, total, batch_size):
            end = start + batch_size
            b = idxs[start:end]
            yield (
                torch.as_tensor(obs_flat[b], device=device, dtype=torch.float32),
                torch.as_tensor(acts_flat[b], device=device, dtype=torch.float32),
                torch.as_tensor(raw_flat[b], device=device, dtype=torch.float32),
                torch.as_tensor(logp_flat[b], device=device, dtype=torch.float32),
                torch.as_tensor(ret_flat[b], device=device, dtype=torch.float32),
                torch.as_tensor(adv_flat[b], device=device, dtype=torch.float32),
            )

    def get_flat_size(self):
        return self.n_steps * self.n_envs