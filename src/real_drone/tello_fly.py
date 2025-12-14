from djitellopy import Tello
import time
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import torch
from scipy.spatial.transform import Rotation as R
import keyboard
from collections import deque

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root)

from scripts.mocap_client import MocapClient
from src.RL.model.drone_policy import DronePolicy
from src.envs.copi_env import QuadcopterEnv

class SafetySupervisor:
    """Emergency safety checks"""
    def __init__(self, bounds=np.array([[-2.0, 2.0], [-2.0, 2.0], [-0.3, 2.0]])):
        self.bounds = bounds
        self.emergency = False
        
    def check_position(self, pos):
        """Check if position is within safe bounds"""
        for i in range(3):
            if pos[i] < self.bounds[i, 0] or pos[i] > self.bounds[i, 1]:
                print(f"SAFETY: Position {pos} out of bounds on axis {i}")
                return False
        return True
        
    def check_velocity(self, vel, max_vel=2.0):
        """Check if velocity is within limits"""
        speed = np.linalg.norm(vel[:2])  # Only check horizontal velocity
        if speed > max_vel:
            print(f"SAFETY: Velocity {speed:.2f} m/s exceeds {max_vel} m/s")
            return False
        return True
        
    def emergency_stop(self, tello):
        """Execute emergency landing"""
        print("EMERGENCY STOP ACTIVATED")
        tello.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5)
        tello.land()
        self.emergency = True

class DeadReckoningEstimator:
    """Dead reckoning using Tello IMU and barometer"""
    def __init__(self, dt=1/30.0):
        self.dt = dt
        self.position = np.zeros(3)  # [x, y, z]
        self.velocity = np.zeros(3)
        self.yaw = 0.0  # degrees
        self.yaw_rad = 0.0  # radians
        
        # For barometer height estimation
        self.base_height = None
        self.height_filtered = 0.0
        self.height_alpha = 0.3  # Filter coefficient
        
        # For velocity estimation
        self.accel_bias = np.zeros(3)
        self.accel_queue = deque(maxlen=50)
        
        # Calibration flags
        self.calibrated = False
        self.gravity = 9.81
        
    def calibrate(self, tello, duration=2.0):
        """Calibrate accelerometer bias on ground"""
        print("Calibrating IMU...")
        accel_readings = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                # Get raw acceleration (in cm/s^2 from Tello)
                accel_x = tello.get_acceleration_x() / 100.0  # Convert to m/s^2
                accel_y = tello.get_acceleration_y() / 100.0
                accel_z = tello.get_acceleration_z() / 100.0
                accel_readings.append([accel_x, accel_y, accel_z])
            except:
                pass
            time.sleep(0.05)
        
        if accel_readings:
            accel_readings = np.array(accel_readings)
            self.accel_bias = np.mean(accel_readings, axis=0)
            # Z-axis should measure gravity when stationary
            self.accel_bias[2] -= self.gravity
            print(f"Calibrated accel bias: {self.accel_bias}")
            self.calibrated = True
            
    def update_from_tello(self, tello):
        """Update state from Tello sensors"""
        try:
            # Get IMU data
            roll = tello.get_roll()
            pitch = tello.get_pitch()
            self.yaw = tello.get_yaw()  # degrees
            self.yaw_rad = np.deg2rad(self.yaw)
            
            # Get acceleration (cm/s^2) and convert to m/s^2
            accel_x = tello.get_acceleration_x() / 100.0
            accel_y = tello.get_acceleration_y() / 100.0
            accel_z = tello.get_acceleration_z() / 100.0
            
            # Remove bias if calibrated
            if self.calibrated:
                accel_x -= self.accel_bias[0]
                accel_y -= self.accel_bias[1]
                accel_z -= self.accel_bias[2]
            
            # Get height from barometer (in cm, convert to m)
            raw_height = tello.get_height() / 100.0
            
            # Initialize base height on first reading
            if self.base_height is None:
                self.base_height = raw_height
                print(f"Base height set to: {self.base_height:.2f}m")
            
            # Filter height
            self.height_filtered = (self.height_alpha * raw_height + 
                                   (1 - self.height_alpha) * self.height_filtered)
            
            # Convert body-frame acceleration to world frame
            cos_yaw = np.cos(self.yaw_rad)
            sin_yaw = np.sin(self.yaw_rad)
            
            # Remove gravity from vertical acceleration
            # Assuming pitch/roll are small (<30 degrees)
            accel_z_world = accel_z - self.gravity
            
            # Transform to world frame
            accel_world_x = accel_x * cos_yaw - accel_y * sin_yaw
            accel_world_y = accel_x * sin_yaw + accel_y * cos_yaw
            
            # Update velocity with acceleration
            self.velocity[0] += accel_world_x * self.dt
            self.velocity[1] += accel_world_y * self.dt
            self.velocity[2] = (self.height_filtered - self.position[2]) / self.dt
            
            # Update position
            self.position[0] += self.velocity[0] * self.dt
            self.position[1] += self.velocity[1] * self.dt
            self.position[2] = self.height_filtered - self.base_height
            
            # Apply velocity damping to reduce drift
            self.velocity[:2] *= 0.95  # Damp horizontal velocity
            
            return True
            
        except Exception as e:
            print(f"Error reading Tello sensors: {e}")
            return False
    
    def reset_position(self, new_pos):
        """Reset position estimate (e.g., from mocap)"""
        self.position = np.array(new_pos)
        self.velocity = np.zeros(3)

def world_to_body_frame(action_world, yaw_rad):
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)
    forward_world, right_world, up = action_world
    forward_body = forward_world * cos_yaw + right_world * sin_yaw
    right_body = -forward_world * sin_yaw + right_world * cos_yaw
    
    return np.array([forward_body, right_body, up])

def initialize_drone_orientation(tello, dead_reckoner, target_yaw=0, tolerance=10):
    """Rotate drone to correct starting orientation using IMU"""
    print("Initializing drone orientation...")
    for _ in range(100):  # 10 second timeout
        dead_reckoner.update_from_tello(tello)
        current_yaw = dead_reckoner.yaw
        
        # Calculate yaw error
        yaw_error = target_yaw - current_yaw
        
        # Normalize to [-180, 180]
        while yaw_error > 180:
            yaw_error -= 360
        while yaw_error < -180:
            yaw_error += 360
        
        # Check if aligned
        if abs(yaw_error) < tolerance:
            print(f"Drone aligned: yaw = {current_yaw:.1f}°")
            tello.send_rc_control(0, 0, 0, 0)
            return True
        
        # Rotate toward target
        yaw_rate = int(np.clip(yaw_error * 0.3, -30, 30))
        tello.send_rc_control(0, 0, 0, yaw_rate)
        time.sleep(0.1)
    
    return False

def get_task_onehot(task, task_list=["hover","circle","figure8","waypoints","goto"]):
    oh = np.zeros(len(task_list), dtype=np.float32)
    if task in task_list:
        oh[task_list.index(task)] = 1.0
    return oh  

class WaypointFollower:
    def __init__(self, waypoints, reached_distance=0.25, hysteresis_distance=0.4):
        self.waypoints = waypoints
        self.reached_distance = reached_distance
        self.hysteresis_distance = hysteresis_distance
        self.current_idx = 0
        self.entered_target_zone = False
        self.time_in_zone = 0
        
    def update(self, position, velocity, dt):
        current_wp = self.waypoints[self.current_idx]
        distance = np.linalg.norm(position - current_wp)
        
        # Hysteresis-based waypoint progression
        if distance < self.reached_distance:
            if not self.entered_target_zone:
                self.entered_target_zone = True
                self.time_in_zone = 0
            else:
                self.time_in_zone += dt
                # Require 0.5 seconds in target zone before advancing
                if self.time_in_zone > 0.5 and self.current_idx < len(self.waypoints) - 1:
                    self.current_idx += 1
                    self.entered_target_zone = False
                    print(f"Waypoint {self.current_idx-1} reached, moving to {self.current_idx}")
        else:
            if distance > self.hysteresis_distance:
                self.entered_target_zone = False
                self.time_in_zone = 0
        
        return self.current_idx, current_wp

class PIDStabilizer:
    def __init__(self):
        # Conservative PID gains for dead reckoning
        self.pos_pid = {
            'kp': np.array([1.5, 1.5, 2.0]),    # Reduced for less aggressive control
            'ki': np.array([0.1, 0.1, 0.2]),    # Reduced integral
            'kd': np.array([0.8, 0.8, 1.2]),    # Reduced derivative
            'integral': np.zeros(3),
            'prev_error': np.zeros(3)
        }
        
        # Yaw stabilization
        self.yaw_pid = {'kp': 2.0, 'ki': 0.05, 'kd': 0.3, 'integral': 0, 'prev_error': 0}
        
        # Output limits
        self.max_xy_output = 60   # cm/s
        self.max_z_output = 40    # cm/s
        self.max_yaw_rate = 30    # deg/s
        
    def compute_control(self, current_pos, target_pos, current_yaw, target_yaw=None, dt=1/30.0):
        # Position control
        error = target_pos - current_pos
        
        # Apply deadzone for small errors
        deadzone = 0.08
        error[np.abs(error) < deadzone] = 0
        
        # Proportional
        p_term = self.pos_pid['kp'] * error
        
        # Integral (with anti-windup)
        self.pos_pid['integral'] += error * dt
        self.pos_pid['integral'] = np.clip(self.pos_pid['integral'], -0.5, 0.5)
        i_term = self.pos_pid['ki'] * self.pos_pid['integral']
        
        # Derivative
        d_error = (error - self.pos_pid['prev_error']) / dt
        d_term = self.pos_pid['kd'] * d_error
        self.pos_pid['prev_error'] = error
        
        # Combine
        control_world = p_term + i_term + d_term
        
        # Yaw control (maintain 0°)
        if target_yaw is None:
            target_yaw = 0.0
            
        yaw_error = target_yaw - current_yaw
        # Normalize yaw error
        while yaw_error > 180: yaw_error -= 360
        while yaw_error < -180: yaw_error += 360
        
        # Deadzone for yaw
        if abs(yaw_error) < 5:
            yaw_error = 0
        
        yaw_p = self.yaw_pid['kp'] * yaw_error
        self.yaw_pid['integral'] += yaw_error * dt
        self.yaw_pid['integral'] = np.clip(self.yaw_pid['integral'], -20, 20)
        yaw_i = self.yaw_pid['ki'] * self.yaw_pid['integral']
        yaw_d = self.yaw_pid['kd'] * (yaw_error - self.yaw_pid['prev_error']) / dt
        self.yaw_pid['prev_error'] = yaw_error
        
        yaw_rate = yaw_p + yaw_i + yaw_d
        yaw_rate = np.clip(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)
        
        return control_world, yaw_rate

class Normalizer:
    """Simple normalizer to match training conditions"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def normalize(self, obs):
        return (obs - self.mean) / (self.std + 1e-8)
    
def load_normalizer(norm_path):
    """Load normalization stats"""
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Normalizer file not found: {norm_path}")
    
    data = np.load(norm_path)
    mean = data['mean'].astype(np.float32)
    var = data['var'].astype(np.float32)
    std = np.sqrt(var + 1e-8)
    
    print(f"Loaded normalizer: mean shape {mean.shape}, std shape {std.shape}")
    return Normalizer(mean, std)

def main(policy, task="figure8", norm_path=None):
    print("Initializing real drone flight with Tello IMU...")
    
    if norm_path and os.path.exists(norm_path):
        normalizer = load_normalizer(norm_path)
        print("Normalizer loaded")
    else:
        print("WARNING: No normalizer found")
        normalizer = None
    
    # Initialize Tello
    tello = Tello()
    print("Connecting to Tello...")
    tello.connect()
    battery = tello.get_battery()
    print(f"Battery: {battery}%")
    
    if battery < 30:
        print("WARNING: Battery below 30%. Consider charging.")
        tello.end()
        return
    
    # Initialize dead reckoning estimator
    dead_reckoner = DeadReckoningEstimator(dt=1/30.0)
    
    # Initialize safety supervisor
    safety = SafetySupervisor()
    
    # Set up environment
    env = QuadcopterEnv(task=task)
    waypoints = np.array(env.waypoints, dtype=np.float32)
    print(f"Loaded {len(waypoints)} waypoints for task: {task}")
    
    print("Calibrating IMU on ground...")
    tello.streamon()
    time.sleep(2)  # Wait for sensors to stabilize
    
    # Calibrate IMU before takeoff
    dead_reckoner.calibrate(tello, duration=3.0)
    
    print("Taking off in 3 seconds...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    
    tello.takeoff()
    time.sleep(4)  # Allow drone to stabilize
    
    # Initialize orientation
    print("Aligning drone to starting orientation (0°)...")
    if not initialize_drone_orientation(tello, dead_reckoner, target_yaw=0):
        print("Failed to align. Landing...")
        safety.emergency_stop(tello)
        return
    
    # Reset position to zero after takeoff
    dead_reckoner.reset_position([0, 0, 0.8])  # Assume 0.8m altitude after takeoff
    
    # Initialize components
    drone_path = []
    dt = 1/30.0
    stabilizer = PIDStabilizer()
    wp_follower = WaypointFollower(waypoints, reached_distance=0.3)
    
    print("Starting control loop...")
    
    try:
        for step in range(1500):
            loop_start = time.time()
            
            # 1. Update dead reckoning from Tello sensors
            if not dead_reckoner.update_from_tello(tello):
                print("Failed to read Tello sensors, hovering...")
                tello.send_rc_control(0, 0, 0, 0)
                time.sleep(0.1)
                continue
            
            pos = dead_reckoner.position.copy()
            vel = dead_reckoner.velocity.copy()
            yaw = dead_reckoner.yaw
            yaw_rad = dead_reckoner.yaw_rad
            
            # Get roll and pitch from Tello
            roll = tello.get_roll()
            pitch = tello.get_pitch()
            rpy = np.array([roll, pitch, yaw])
            
            # 2. Safety checks
            safety_ok = safety.check_position(pos)
            if not safety_ok:
                print(f"Safety violation at position: {pos}")
                # Gradual slowdown
                for i in range(10):
                    scale = 1.0 - (i/10.0)
                    tello.send_rc_control(0, 0, int(-20*scale), 0)
                    time.sleep(0.05)
                tello.land()
                break
            
            if not safety.check_velocity(vel):
                print("Velocity safety violation - reducing speed...")
                vel *= 0.5  # Reduce velocity
            
            # Manual kill switch
            if keyboard.is_pressed('space'):
                print("Manual kill switch pressed - landing")
                safety.emergency_stop(tello)
                break
            
            # 3. Update waypoint following
            wp_idx, current_wp = wp_follower.update(pos, vel, dt)
            
            # 4. Get error vector
            error = current_wp - pos
            
            # 5. Build observation (matching your training)
            task_oh = get_task_onehot(task).astype(np.float32)
            
            # Use zero for angular velocity since we don't have gyro readings easily
            ang_vel = np.zeros(3)
            
            obs = np.concatenate([
                pos,           # 3: position (x, y, z)
                vel,           # 3: velocity (vx, vy, vz)
                rpy,           # 3: roll, pitch, yaw (degrees)
                ang_vel,       # 3: angular velocity (deg/s) - approximated as zero
                error,         # 3: error vector = target - current
                task_oh,       # 5: task one-hot encoding
            ]).astype(np.float32)
            
            drone_path.append(pos.copy())
            
            # 6. Normalize observation if needed
            if normalizer:
                if obs.shape[0] != normalizer.mean.shape[0]:
                    # Handle dimension mismatch
                    if normalizer.mean.shape[0] == 20 and obs.shape[0] == 21:
                        obs = obs[:20]
                    elif normalizer.mean.shape[0] == 21 and obs.shape[0] == 20:
                        obs = np.concatenate([obs, np.zeros(1)])
                obs = normalizer.normalize(obs)
            
            # 7. Get policy action
            action_world = policy.act(obs)
            action_world = np.clip(action_world, -1.5, 1.5)  # Conservative limits
            
            # 8. Get PID stabilization
            pid_action_world, yaw_rate = stabilizer.compute_control(
                pos, current_wp, yaw, dt=dt
            )
            
            # 9. Blend policy and PID (adjust ratio as needed)
            blend_ratio = 0.5  # 70% policy, 30% stabilization
            combined_action = (
                blend_ratio * action_world + 
                (1 - blend_ratio) * pid_action_world
            )
            
            # 10. Convert to body frame
            action_body = world_to_body_frame(combined_action, yaw_rad)
            
            # 11. Scale controls with adaptive gain
            distance_to_target = np.linalg.norm(error)
            
            if distance_to_target > 0.8:
                scale = 35  # Moderate gain for far targets
            elif distance_to_target > 0.3:
                scale = 25  # Reduced gain for medium distance
            else:
                scale = 15  # Minimal gain when close
            
            # Scale controls
            fb = int(np.clip(action_body[0] * scale, -50, 50))
            lr = int(np.clip(action_body[1] * scale, -50, 50))
            ud = int(np.clip(action_body[2] * scale, -30, 30))
            
            # Adjust yaw rate based on distance
            if distance_to_target > 0.5:
                yaw_rate = int(np.clip(yaw_rate, -20, 20))
            else:
                yaw_rate = 0
            
            # 12. Send control commands
            tello.send_rc_control(lr, fb, ud, yaw_rate)
            
            # 13. Logging
            if step % 30 == 0:
                print(f"\nStep {step}:")
                print(f"  Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] m")
                print(f"  Vel: [{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}] m/s")
                print(f"  Yaw: {yaw:.1f}°")
                print(f"  WP: {wp_idx}/{len(waypoints)}")
                print(f"  Dist to target: {distance_to_target:.2f} m")
                print(f"  Controls: fb={fb}, lr={lr}, ud={ud}, yaw={yaw_rate}")
            
            # 14. Maintain timing
            elapsed = time.time() - loop_start
            sleep_time = max(0.001, dt - elapsed)
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\nManual interrupt detected")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        try:
            tello.send_rc_control(0, 0, 0, 0)
            time.sleep(0.5)
            tello.land()
            time.sleep(2)
        except Exception as e:
            print(f"Error during landing: {e}")
        finally:
            tello.streamoff()
            tello.end()
        
        # Visualize results
        if len(drone_path) > 10:
            visualize_results(drone_path, waypoints, task)

def visualize_results(drone_path, waypoints, task):
    """Plot flight results"""
    drone_arr = np.array(drone_path)
    wp_arr = np.array(waypoints)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Top-down view
    axes[0].plot(wp_arr[:, 0], wp_arr[:, 1], 'r--', linewidth=2.5, label="Target Path")
    axes[0].plot(drone_arr[:, 0], drone_arr[:, 1], 'b-', linewidth=1.5, alpha=0.7, label="Drone Path")
    axes[0].scatter(drone_arr[0, 0], drone_arr[0, 1], c='g', s=100, label="Start")
    axes[0].scatter(drone_arr[-1, 0], drone_arr[-1, 1], c='orange', s=100, label="End")
    axes[0].scatter(wp_arr[:, 0], wp_arr[:, 1], c='r', s=50, alpha=0.5, marker='x', label="Waypoints")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    axes[0].set_title(f"{task} - Estimated Position (Dead Reckoning)")
    axes[0].legend()
    axes[0].grid(True)
    axes[0].axis('equal')
    
    # Altitude over time
    axes[1].plot(drone_arr[:, 2], 'b-', linewidth=1.5, alpha=0.7, label="Altitude")
    axes[1].axhline(y=wp_arr[0, 2], color='r', linestyle='--', label="Target Altitude")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Altitude (m)")
    axes[1].set_title("Altitude Profile (Barometer)")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    total_distance = np.sum(np.linalg.norm(np.diff(drone_arr, axis=0), axis=1))
    avg_altitude = np.mean(drone_arr[:, 2])
    alt_std = np.std(drone_arr[:, 2])
    
    print(f"\n📊 Flight Statistics (Dead Reckoning):")
    print(f"  Total estimated distance: {total_distance:.2f} m")
    print(f"  Average altitude: {avg_altitude:.2f} m (±{alt_std:.2f} m)")
    print(f"  Flight duration: {len(drone_arr)/30:.1f} seconds")
    print(f"  Note: Horizontal position is estimated and may drift over time")

if __name__ == "__main__":
    # IMPORTANT: Update these paths to match your setup
    NORM_PATH = "data/collected_observations/circle_tello_obs_stats.npz"
    POLICY_PATH = "data/agents/circle_tello_policy.pt"
    
    # Load policy
    policy = DronePolicy(
        POLICY_PATH,
        obs_dim=20,  # Verify this matches your actual observation dim
        act_dim=3,
        device="cpu"
    )
    
    # Run with Tello IMU
    main(policy, task="circle", norm_path=None)