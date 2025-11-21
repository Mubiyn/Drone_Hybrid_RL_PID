#!/usr/bin/env python3
import numpy as np


def hover_reward(current_pos, target_pos, obs, action):
    """Reward function for hover task - POSITIVE ONLY rewards to avoid local optima."""
    # Position tracking error
    pos_error = np.linalg.norm(current_pos - target_pos)

    # Main position reward: exponential decay centered at target
    # At target (0m error): reward = 10.0
    # At 0.5m error: reward = 0.82
    # At 1.0m error (ground): reward = 0.067
    pos_reward = 10.0 * np.exp(-2.0 * pos_error)

    # Altitude progress bonus (encourage climbing from ground)
    # Linearly increases from 0 to 2 as altitude goes from 0 to target
    altitude_ratio = np.clip(current_pos[2] / target_pos[2], 0, 1.5)
    altitude_bonus = 2.0 * altitude_ratio

    # Stability bonus (low velocity when close to target)
    vel = obs[10:13]
    vel_magnitude = np.linalg.norm(vel)
    if pos_error < 0.3:  # Only reward stability when reasonably close
        stability_bonus = 1.0 * np.exp(-5.0 * vel_magnitude)
    else:
        stability_bonus = 0.0

    # Orientation bonus (level flight)
    rpy = obs[7:10]
    tilt = np.abs(rpy[0]) + np.abs(rpy[1])  # Total roll + pitch
    orientation_bonus = 1.0 * np.exp(-5.0 * tilt)

    # Total reward (all positive components)
    reward = pos_reward + altitude_bonus + stability_bonus + orientation_bonus

    # Large bonus for precision hovering at target
    if pos_error < 0.05 and vel_magnitude < 0.1:  # Within 5cm, nearly stationary
        reward += 10.0

    return reward


def waypoint_reward(current_pos, target_pos, obs, action):
    """Reward function for waypoint navigation."""
    # Position tracking error
    pos_error = np.linalg.norm(current_pos - target_pos)
    pos_reward = np.exp(-3.0 * pos_error)

    # Forward progress reward (heading towards target)
    vel = obs[10:13]
    direction_to_target = (target_pos - current_pos) / (pos_error + 1e-6)
    progress_reward = 0.5 * np.dot(vel, direction_to_target)

    # Orientation stability
    rpy = obs[7:10]
    orientation_penalty = -0.05 * (np.abs(rpy[0]) + np.abs(rpy[1]))

    # Control smoothness
    control_penalty = -0.005 * np.linalg.norm(action)

    reward = pos_reward + progress_reward + orientation_penalty + control_penalty

    # Bonus for reaching waypoint
    if pos_error < 0.1:  # Within 10cm
        reward += 2.0

    return reward


def trajectory_tracking_reward(current_pos, target_pos, obs, action):
    """General reward function for continuous trajectory tracking (figure-8, circle)."""
    # Tracking error
    pos_error = np.linalg.norm(current_pos - target_pos)
    tracking_reward = np.exp(-4.0 * pos_error)

    # Altitude progress bonus to encourage takeoff
    # This bonus encourages the drone to take off and stay airborne.
    # It's a simple step bonus for being above a minimum altitude.
    airborne_bonus = 2.0 if current_pos[2] > 0.2 else 0.0

    # Penalty for staying on the ground to discourage inaction
    ground_penalty = -0.5 if current_pos[2] < 0.1 else 0.0

    # Velocity alignment (want to move along trajectory)
    vel = obs[10:13]
    speed = np.linalg.norm(vel)
    # Only reward speed if the drone is airborne
    if current_pos[2] > 0.2:
        speed_reward = 0.2 * np.clip(
            speed, 0.0, 2.0
        )  # Encourage movement but not too fast
    else:
        speed_reward = 0.0

    # Smooth orientation
    rpy = obs[7:10]
    orientation_penalty = -0.08 * (np.abs(rpy[0]) + np.abs(rpy[1]))

    # Smooth control
    angular_vel = obs[13:16]
    angular_penalty = -0.05 * np.linalg.norm(angular_vel)
    control_penalty = -0.008 * np.linalg.norm(action)

    reward = (
        tracking_reward
        + airborne_bonus
        + ground_penalty
        + speed_reward
        + orientation_penalty
        + angular_penalty
        + control_penalty
    )

    # Bonus for precise tracking
    if pos_error < 0.08:
        reward += 1.5

    return reward


def emergency_landing_reward(current_pos, target_pos, obs, action):
    """Reward function for emergency landing."""
    # Descent progress (want to go down)
    height = current_pos[2]
    target_height = target_pos[2]
    height_error = abs(height - target_height)

    # Reward for descending
    if height > target_height:
        descent_reward = 1.0 - height / 2.0  # Higher reward as we descend
    else:
        descent_reward = 2.0  # Bonus for reaching target height

    # Vertical velocity control (not too fast)
    vel_z = obs[12]
    if vel_z < -0.5:  # Descending too fast
        vel_penalty = -2.0 * abs(vel_z + 0.5)
    else:
        vel_penalty = 0.0

    # Lateral stability (stay centered)
    lateral_error = np.linalg.norm(current_pos[0:2] - target_pos[0:2])
    lateral_penalty = -0.5 * lateral_error

    # Orientation control (stay level)
    rpy = obs[7:10]
    orientation_penalty = -0.2 * (np.abs(rpy[0]) + np.abs(rpy[1]))

    # Smooth landing (gentle touchdown)
    if height < 0.2 and abs(vel_z) < 0.3:
        gentle_landing_bonus = 3.0
    else:
        gentle_landing_bonus = 0.0

    reward = (
        descent_reward
        + vel_penalty
        + lateral_penalty
        + orientation_penalty
        + gentle_landing_bonus
    )

    return reward


def get_reward_function(task_name):
    """Get the appropriate reward function for a task."""
    reward_functions = {
        "hover": hover_reward,
        "hover_extended": hover_reward,
        "waypoint_delivery": waypoint_reward,
        "figure8": trajectory_tracking_reward,
        "circle": trajectory_tracking_reward,
        "emergency_landing": emergency_landing_reward,
    }

    if task_name not in reward_functions:
        raise ValueError(
            f"Unknown task: {task_name}. Available: {list(reward_functions.keys())}"
        )

    return reward_functions[task_name]
