import sys,os
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)
from src.utils.RL import PPO
from src.envs.waypoint_env import WayPointNavigationEnv

def train_rectangle():
    env = WayPointNavigationEnv(
        hover_policy_path="logs/hover/27939_ppo_drone.pth",
        gui=False
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    ppo = PPO(
        state_dim=obs_dim,
        action_dim=act_dim,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        K_epochs=10,
        eps_clip=0.2,
        action_std=0.2
    )

    steps = 400000

    obs, _ = env.reset()
    for t in range(steps):
        action = ppo.select_action(obs)
        obs, reward, done, _, _ = env.step(action)

        ppo.buffer.rewards.append(reward)
        ppo.buffer.dones.append(done)

        if done:
            obs, _ = env.reset()

        if t % 2048 == 0:
            ppo.update()
            print("update")

    ppo.save("models/PPO/rectangle_nav.pth")
