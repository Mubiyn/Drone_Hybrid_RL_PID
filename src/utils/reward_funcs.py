import numpy as np


def hover_reward(current_pos, target_pos, obs, action, **kwargs):
    """Reward for stable hovering"""
    pos_error = np.linalg.norm(current_pos - target_pos)
    
    # Get orientation (roll, pitch, yaw)
    rpy = obs[7:10]
    orientation_penalty = np.sum(np.square(rpy))  # Penalty for tilting
    
    # Get velocities
    linear_vel = obs[10:13]
    angular_vel = obs[13:16]
    velocity_penalty = np.sum(np.square(linear_vel)) + np.sum(np.square(angular_vel))
    
    # Hover-specific rewards
    reward = -pos_error * 2.0                    # Position accuracy (higher weight)
    reward -= orientation_penalty * 0.5          # Stability
    reward -= velocity_penalty * 0.3             # Minimal movement
    reward -= np.sum(np.square(action)) * 0.1    # Smooth control
    
    # Bonus for maintaining position within small threshold
    if pos_error < 0.1:
        reward += 1.0
    if pos_error < 0.05:
        reward += 2.0
        
    return float(reward)

def trajectory_reward(current_pos, target_pos, obs, action, step_count, trajectory_progress=0.0, **kwargs):
    """Reward for following trajectories (circle, figure-8)"""
    pos_error = np.linalg.norm(current_pos - target_pos)
    
    # Get velocities and orientation
    linear_vel = obs[10:13]
    rpy = obs[7:10]
    
    # Progress-based reward (encourage moving along trajectory)
    progress_bonus = trajectory_progress * 0.1
    
    # Smoothness penalties
    orientation_penalty = np.sum(np.square(rpy[0:2]))  # Only roll/pitch
    jerk_penalty = np.sum(np.square(action)) * 0.05
    
    reward = -pos_error * 1.5                    # Position accuracy
    reward += progress_bonus                     # Encourage progression
    reward -= orientation_penalty * 0.3          # Reasonable tilt
    reward -= jerk_penalty                       # Smooth control
    
    # Speed matching bonus (ideal for trajectories)
    ideal_speed = 0.5  # m/s
    current_speed = np.linalg.norm(linear_vel)
    speed_error = abs(current_speed - ideal_speed)
    reward -= speed_error * 0.2
    
    return float(reward)

def waypoint_reward(current_pos, target_pos, obs, action, step_count,
                    waypoint_index=0, total_waypoints=1, **kwargs):
    """Reward for waypoint navigation"""
    pos_error = np.linalg.norm(current_pos - target_pos)
    
    # Waypoint completion bonus (significant reward for reaching waypoints)
    waypoint_threshold = 0.3
    waypoint_bonus = 0.0
    
    if pos_error < waypoint_threshold:
        waypoint_bonus = 10.0 * (waypoint_index + 1)  # More reward for later waypoints
    
    # Efficiency penalty (encourage direct paths)
    time_penalty = step_count * 0.01
    
    # Orientation penalty (less critical for waypoints)
    rpy = obs[7:10]
    orientation_penalty = np.sum(np.square(rpy)) * 0.1
    
    reward = -pos_error * 1.0                    # Position accuracy
    reward += waypoint_bonus                     # Waypoint completion
    reward -= time_penalty                       # Efficiency
    reward -= orientation_penalty                # Basic stability
    
    # Final destination bonus
    if waypoint_index == total_waypoints - 1 and pos_error < waypoint_threshold:
        reward += 20.0
        
    return float(reward)


def emergency_landing_reward(current_pos, target_pos, obs, action, step_count,**kwargs):
    """Reward for controlled emergency landing"""
    pos_error = np.linalg.norm(current_pos - target_pos)
    
    # Get velocities and orientation
    linear_vel = obs[10:13]
    rpy = obs[7:10]
    
    # Critical: vertical velocity control (should be downward but controlled)
    vertical_vel = linear_vel[2]
    ideal_descent_speed = -0.3  # m/s (controlled descent)
    descent_speed_error = abs(vertical_vel - ideal_descent_speed)
    
    # Critical: horizontal position stability
    horizontal_error = np.linalg.norm(current_pos[0:2] - target_pos[0:2])
    
    # Critical: landing attitude (should be upright)
    orientation_penalty = np.sum(np.square(rpy))
    
    # Time penalty (encourage timely landing)
    time_penalty = step_count * 0.02
    
    reward = -pos_error * 2.0                    # Overall position
    reward -= horizontal_error * 3.0             # Horizontal stability (critical!)
    reward -= descent_speed_error * 2.0          # Controlled descent speed
    reward -= orientation_penalty * 1.5          # Upright attitude
    reward -= time_penalty                       # Timeliness
    
    # Successful landing bonus
    if current_pos[2] < 0.15 and horizontal_error < 0.2 and abs(vertical_vel) < 0.5:
        reward += 30.0
    # Crash penalty
    elif abs(vertical_vel) > 1.5 or orientation_penalty > 2.0:
        reward -= 20.0
        
    return float(reward)

def agile_maneuver_reward(current_pos, target_pos, obs, action, step_count, trajectory_progress,**kwargs):
    """Reward for agile maneuvers (figure-8, complex paths)"""
    pos_error = np.linalg.norm(current_pos - target_pos)
    
    # Emphasize speed and progress for agile tasks
    linear_vel = obs[10:13]
    current_speed = np.linalg.norm(linear_vel)
    
    # Progress is key for agile maneuvers
    progress_bonus = trajectory_progress * 0.2
    
    # Allow more aggressive control but penalize extreme actions
    action_penalty = np.sum(np.square(action)) * 0.02
    
    # Orientation penalty (allow more tilt for agile maneuvers)
    rpy = obs[7:10]
    orientation_penalty = max(0, np.sum(np.square(rpy)) - 0.5) * 0.2  # Only penalize extreme tilts
    
    reward = -pos_error * 1.2                    # Position accuracy
    reward += progress_bonus                     # Emphasize progression
    reward += current_speed * 0.1                # Encourage movement
    reward -= orientation_penalty                # Moderate stability
    reward -= action_penalty                     # Reasonable control
    
    # Bonus for maintaining good speed
    if 0.3 < current_speed < 1.0:
        reward += 0.5
        
    return float(reward)

# --- Reward Function ---
def default_reward_func(current_pos, target_pos, obs, action, **kwargs):
    pos_error = np.linalg.norm(current_pos - target_pos)
    
    # Primary reward: position accuracy
    position_reward = -1.0 * pos_error
    
    # 🔧 FIX 5: STRONG tilt penalty to prevent excessive rolling
    rpy = obs[7:10]
    tilt_penalty = -0.8 * (abs(rpy[0]) + abs(rpy[1]))  # Increased penalty
    
    # Action penalty to encourage smooth control
    action_penalty = -0.01 * np.linalg.norm(action)
    
    # Gentle velocity penalty
    vel_penalty = -0.001 * (np.linalg.norm(obs[10:13]) + np.linalg.norm(obs[13:16]))
    
    # Big bonus for being close to target
    proximity_bonus = 0.0
    if pos_error < 0.1:
        proximity_bonus = 10.0
    elif pos_error < 0.3:
        proximity_bonus = 3.0
    elif pos_error < 0.5:
        proximity_bonus = 1.0
    
    # Bonus for maintaining altitude
    altitude_reward = -0.5 * abs(current_pos[2] - target_pos[2])
    
    total_reward = position_reward + tilt_penalty + action_penalty + vel_penalty + proximity_bonus + altitude_reward
    
    return float(total_reward)



reward_functions = {
            'hover': hover_reward,
            'hover_extended': hover_reward,
            'circle': trajectory_reward,
            'figure8': agile_maneuver_reward,
            'waypoint_delivery': waypoint_reward,
            'emergency_landing': emergency_landing_reward,
            'default': default_reward_func
    
        }

def get_reward_function(task_name):
    if task_name not in reward_functions.keys():
        return reward_functions['default']
    else:
        return reward_functions[task_name]