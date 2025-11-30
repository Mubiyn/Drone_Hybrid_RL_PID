import os
import time
from datetime import datetime
import numpy as np
import sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from src.utils.RL import PPO


def train():
    # ---------- ENV ----------
    env = HoverAviary(
        obs=ObservationType('kin'),
        act=ActionType('rpm'),
        gui=False,              # GUI OFF for speed
        record=False
    )

    # get state_dim from real obs
    obs, info = env.reset(seed=42, options={})
    obs_flat = np.asarray(obs, dtype=np.float32).reshape(-1)
    state_dim = obs_flat.shape[0]
    action_dim = 4

    # ---------- PPO HYPERPARAMS ----------
    action_std = 0.6
    action_std_decay_rate = 0.02
    min_action_std = 0.1
    action_std_decay_freq = int(2.5e5)

    max_training_timesteps = int(3e5)   # start smaller
    K_epochs = 10                       # from 80 → 10
    eps_clip = 0.2
    gamma = 0.99
    lr_actor = 3e-4
    lr_critic = 1e-3

    log_dir = "logs/"
    run_num = "hover"
    os.makedirs(os.path.join(log_dir, str(run_num)), exist_ok=True)
    log_f_name = os.path.join(log_dir, f"PPO_log_{run_num}.csv")

    print("current logging run number for gym pybullet drone:", run_num)
    print("logging at:", log_f_name)
    log_f = open(log_f_name, "w+")
    log_f.write("episode,timestep,reward\n")

    print("EPISODE_LEN_SEC:", env.EPISODE_LEN_SEC)
    print("CTRL_FREQ:", env.CTRL_FREQ)
    steps_per_episode = env.EPISODE_LEN_SEC * env.CTRL_FREQ
    print("step per episode", steps_per_episode)

    update_timestep = steps_per_episode * 1   # update every episode
    print_freq = steps_per_episode * 5
    log_freq = steps_per_episode * 2
    save_model_freq = int(1e5)

    print_running_reward = 0.0
    print_running_episodes = 0

    log_running_reward = 0.0
    log_running_episodes = 0

    # ---------- PPO AGENT ----------
    ppo_agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        K_epochs=K_epochs,
        eps_clip=eps_clip,
        action_std_init=action_std,
        
    )

    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT):", start_time)

    time_step = 0
    i_episode = 0

    HOVER_RPM = env.HOVER_RPM
    RPM_RANGE = 500   # safe small control authority

    while time_step <= max_training_timesteps:
        obs, info = env.reset(seed=42, options={})
        current_ep_reward = 0.0

        for i in range(steps_per_episode):
            state = np.asarray(obs, dtype=np.float32).reshape(-1)

            # PPO outputs normalized action [-1,1]
            raw_action = ppo_agent.select_action(state)    # shape (4,)

            # map to RPM
            rpm = HOVER_RPM + raw_action * RPM_RANGE
            rpm = np.clip(rpm, 3000, env.MAX_RPM)
            rpm = np.expand_dims(rpm, axis=0)              # shape (1,4)

            obs, reward, terminated, truncated, info = env.step(rpm)
            done = terminated or truncated

            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            if time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            if log_running_episodes > 0 and time_step % log_freq == 0:
                log_avg_reward = log_running_reward / log_running_episodes
                log_f.write(f"{i_episode},{time_step},{log_avg_reward:.4f}\n")
                log_f.flush()
                log_running_reward = 0.0
                log_running_episodes = 0

            if print_running_episodes > 0 and time_step % print_freq == 0:
                print_avg_reward = print_running_reward / print_running_episodes
                print(f"Episode: {i_episode}\tTimestep: {time_step}\tAverage Reward: {print_avg_reward:.2f}")
                print_running_reward = 0.0
                print_running_episodes = 0

            if time_step % save_model_freq == 0:
                checkpoint_path = os.path.join(log_dir, str(run_num), f"{i_episode}_ppo_drone.pth")
                print("--------------------------------------------------------------------------------")
                print("saving model at:", checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time:", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------")

            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    end_time = datetime.now().replace(microsecond=0)
    print("================================================================================")
    print("Started training at (GMT):", start_time)
    print("Finished training at (GMT):", end_time)
    print("Total training time:", end_time - start_time)
    print("================================================================================")


if __name__ == "__main__":
    train()
