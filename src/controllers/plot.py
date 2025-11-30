import matplotlib.pyplot as plt
from IPython.display import clear_output

class LivePlot:
    def __init__(self):
        self.rewards = []
        self.steps = []
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        self.loss_steps = []

    def update_rewards(self, reward, step):
        self.rewards.append(reward)
        self.steps.append(step)

    def update_losses(self, policy_loss, value_loss, entropy, update_step):
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropies.append(entropy)
        self.loss_steps.append(update_step)

    def draw(self):
        # Only draw if we have data
        if not self.steps and not self.loss_steps:
            return  # Skip drawing until first episode or update

        clear_output(wait=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

        # Plot rewards (only if we have any)
        if self.steps:
            ax1.plot(self.steps, self.rewards, 'o-', color='tab:blue', label='Episode Reward', markersize=4)
            ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax1.set_title(f"Episode Rewards (n={len(self.rewards)})")
            ax1.set_xlabel("Global Step")
            ax1.set_ylabel("Reward")
            ax1.grid(True)
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'No episodes yet', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title("Episode Rewards")

        # Plot losses
        if self.loss_steps:
            ax2.plot(self.loss_steps, self.policy_losses, label='Policy Loss', color='red')
            ax2.plot(self.loss_steps, self.value_losses, label='Value Loss', color='green')
            ax2.plot(self.loss_steps, self.entropies, label='Entropy', color='purple')
            ax2.set_title(f"PPO Losses & Entropy (updates={len(self.loss_steps)})")
            ax2.set_xlabel("Update Step")
            ax2.set_ylabel("Loss / Entropy")
            ax2.grid(True)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No updates yet', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title("PPO Losses & Entropy")

        plt.tight_layout()
        plt.show()