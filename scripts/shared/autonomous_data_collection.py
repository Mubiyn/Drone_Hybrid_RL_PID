#!/usr/bin/env python3
"""
Autonomous Tello Data Collection with MoCap

Flies autonomous trajectories using PID control with MoCap ground truth.
Collects clean state-action data for offline RL fine-tuning.

CRITICAL: MoCap Markers Create Natural Domain Randomization
------------------------------------------------------------
The MoCap markers attached to the Tello change its dynamics:
- Added mass: +5-15g (total 85-95g vs bare 80g)
- Changed inertia: Asymmetric marker placement shifts CoM
- Increased drag: Markers add air resistance
- Observable drift: Heavier drone requires different control

Implication: Data collected WITH markers = training data for deployment WITH markers.
If markers are removed later, the fine-tuned model will NOT transfer to bare Tello.

Recommendation: Keep markers on permanently OR collect separate datasets for each config.

Usage:
    # Auto-tune PID gains (tests 20 combinations, ~3.5 min)
    python scripts/autonomous_data_collection.py --tune-pid --mocap
    
    # Collect data with tuned gains
    python scripts/autonomous_data_collection.py --trajectory circle --kp 0.6 --max-vel 0.7 --mocap --duration 60
"""

import argparse
import time
import numpy as np
import pickle
from datetime import datetime
from pathlib import Path
import sys

from djitellopy import Tello

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.controllers.pid_controller import VelocityPIDController


class TrajectoryGenerator:
    """Generates reference trajectories for autonomous flight"""
    
    @staticmethod
    def circle(t, center=(0, 0, 1.0), radius=0.032, period=20.0):
        """Circle trajectory (ultra-compact: 3.2cm radius)"""
        omega = 2 * np.pi / period
        x = center[0] + radius * np.cos(omega * t)
        y = center[1] + radius * np.sin(omega * t)
        z = center[2]
        
        # Velocity (derivative)
        vx = -radius * omega * np.sin(omega * t)
        vy = radius * omega * np.cos(omega * t)
        vz = 0.0
        
        return np.array([x, y, z]), np.array([vx, vy, vz])
    
    @staticmethod
    def figure8(t, center=(0, 0, 1.0), radius=0.032, period=30.0):
        """Figure-8 (lemniscate) trajectory (ultra-compact: 3.2cm radius)"""
        omega = 2 * np.pi / period
        scale = radius * 0.8
        
        x = center[0] + scale * np.sin(omega * t)
        y = center[1] + scale * np.sin(2 * omega * t) / 2
        z = center[2]
        
        vx = scale * omega * np.cos(omega * t)
        vy = scale * omega * np.cos(2 * omega * t)
        vz = 0.0
        
        return np.array([x, y, z]), np.array([vx, vy, vz])
    
    @staticmethod
    def spiral(t, center=(0, 0, 0.5), radius_max=0.04, height_max=0.06, period=40.0):
        """Upward spiral trajectory (ultra-compact: 4cm radius, 6cm height)"""
        omega = 2 * np.pi / period
        progress = (t % period) / period  # 0 to 1
        
        r = radius_max * progress
        x = center[0] + r * np.cos(omega * t)
        y = center[1] + r * np.sin(omega * t)
        z = center[2] + height_max * progress
        
        # Approximate velocity
        vx = (radius_max / period) * np.cos(omega * t) - r * omega * np.sin(omega * t)
        vy = (radius_max / period) * np.sin(omega * t) + r * omega * np.cos(omega * t)
        vz = height_max / period
        
        return np.array([x, y, z]), np.array([vx, vy, vz])
    
    @staticmethod
    def waypoint(t, waypoints, period_per_wp=8.0):
        """Waypoint trajectory with smooth transitions"""
        num_waypoints = len(waypoints)
        total_period = period_per_wp * num_waypoints
        
        # Loop through waypoints
        t_loop = t % total_period
        idx = int(t_loop / period_per_wp)
        t_segment = (t_loop % period_per_wp) / period_per_wp  # 0 to 1
        
        # Current and next waypoint
        wp_current = np.array(waypoints[idx % num_waypoints])
        wp_next = np.array(waypoints[(idx + 1) % num_waypoints])
        
        # Smooth interpolation (cosine easing)
        alpha = (1 - np.cos(np.pi * t_segment)) / 2
        pos = wp_current + alpha * (wp_next - wp_current)
        
        # Velocity (approximate derivative)
        vel = (wp_next - wp_current) / period_per_wp * np.pi / 2 * np.sin(np.pi * t_segment)
        
        return pos, vel
    
    @staticmethod
    def hover(t, position=(0, 0, 1.0)):
        """Stationary hover"""
        return np.array(position), np.zeros(3)


class AutonomousTelloController:
    """Autonomous Tello flight with PID control and data logging"""
    
    def __init__(self, trajectory_type='circle', kp=0.4, max_vel=0.5, use_mocap=False, tello_instance=None, open_loop=False):
        # Use provided Tello instance or create new one
        self.tello = tello_instance if tello_instance is not None else Tello()
        self.controller = VelocityPIDController(kp=kp, max_vel=max_vel)
        
        self.trajectory_type = trajectory_type
        self.use_mocap = use_mocap
        self.open_loop = open_loop
        
        # Data storage
        self.states = []
        self.actions = []
        self.timestamps = []
        self.battery_history = []
        
        # MoCap
        self.mocap_wrapper = None
        self.initial_position = None
        
        # Dead reckoning state (without MoCap)
        self.position_estimate = np.zeros(3)
        self.last_update_time = None
        
        # Trajectory
        self.trajectory_start_time = None
        self.learned_traj = None  # For learned trajectories
        self.approach_waypoints = None  # Approach phase (executed once)
        self.loop_waypoints = None  # Loop phase (repeats)
        self.approach_duration = 0.0
        self.loop_duration = 0.0
        self.waypoints = [
            (0.02, 0.02, 1.0),
            (0.02, -0.02, 1.008),
            (-0.02, -0.02, 1.0),
            (-0.02, 0.02, 0.992),
        ]
        
        print(f"\n Autonomous Tello Controller")
        print(f"   Trajectory: {trajectory_type}")
        print(f"   PID: kp={kp}, max_vel={max_vel} m/s")
        print(f"   MoCap: {'✓ ENABLED' if use_mocap else '✗ DISABLED (using dead reckoning)'}")
    
    def set_learned_trajectory(self, learned_traj):
        """Set learned trajectory from .pkl file"""
        self.learned_traj = learned_traj
        
        # Separate approach waypoints (from ground to trajectory start) from loop waypoints
        # Approach waypoints start at [0,0,0] and end at the trajectory height
        waypoints = learned_traj['waypoints']
        waypoint_times = learned_traj['waypoint_times']
        
        # Find where approach ends (when Z stops changing significantly)
        z_values = waypoints[:, 2]
        target_z = z_values[-1]  # Final height
        
        # Approach ends when we're within 5cm of target height
        approach_end_idx = 0
        for i in range(len(waypoints)):
            if abs(z_values[i] - target_z) < 0.05:
                approach_end_idx = i
                break
        
        if approach_end_idx > 0:
            # Split into approach and loop
            self.approach_waypoints = waypoints[:approach_end_idx+1]
            self.approach_times = waypoint_times[:approach_end_idx+1]
            self.approach_duration = waypoint_times[approach_end_idx]
            
            # Loop waypoints (rest of trajectory, shifted to start at t=0)
            self.loop_waypoints = waypoints[approach_end_idx:]
            self.loop_times = waypoint_times[approach_end_idx:] - waypoint_times[approach_end_idx]
            self.loop_duration = self.loop_times[-1]
            
            print(f"\n📍 Trajectory split:")
            print(f"   Approach: {len(self.approach_waypoints)} waypoints, {self.approach_duration:.1f}s")
            print(f"   Loop: {len(self.loop_waypoints)} waypoints, {self.loop_duration:.1f}s")
        else:
            # No approach phase, use entire trajectory as loop
            self.approach_waypoints = None
            self.loop_waypoints = waypoints
            self.loop_times = waypoint_times
            self.loop_duration = waypoint_times[-1]
            print(f"\n📍 No approach phase, using full trajectory as loop")
        self.waypoints = learned_traj['waypoints']
        print(f"   ✓ Loaded {len(self.waypoints)} waypoints")
    
    def connect(self):
        """Connect to Tello and MoCap"""
        print("\n📡 Connecting to Tello...")
        self.tello.connect()
        battery = self.tello.get_battery()
        print(f"   ✓ Battery: {battery}%")
        
        if battery < 20:
            print("   ✗ Battery too low! Charge before flying.")
            return False
        
        # Enable video stream for state feedback
        print("   Starting video stream...")
        self.tello.streamon()
        time.sleep(1)
        
        if self.use_mocap:
            print("\n📍 Connecting to MoCap...")
            try:
                from src.hardware.mocap_wrapper import MocapWrapper
                self.mocap_wrapper = MocapWrapper(
                    mode="multicast",
                    interface_ip="192.168.1.1",
                    mcast_addr="239.255.42.99",
                    data_port=1511
                )
                self.mocap_wrapper.start()
                time.sleep(2)  # Wait for data
                
                pos = self.mocap_wrapper.get_position()
                if pos is not None:
                    print(f"   ✓ MoCap tracking: {pos}")
                else:
                    print("   ✗ No MoCap data! Flying blind (using Tello sensors)")
                    self.use_mocap = False
            except Exception as e:
                print(f"   ✗ MoCap connection failed: {e}")
                self.use_mocap = False
        
        return True
    
    def get_state(self):
        """Get current 12D state: [x,y,z, roll,pitch,yaw, vx,vy,vz, wx,wy,wz]"""
        state = np.zeros(12)
        
        # Position [x, y, z]
        if self.use_mocap and self.mocap_wrapper and self.mocap_wrapper.is_tracked():
            pos = self.mocap_wrapper.get_position()
            if self.initial_position is None:
                self.initial_position = pos.copy()
            state[0:3] = pos - self.initial_position  # Relative to start
        else:
            # Dead reckoning from Tello sensors
            current_time = time.time()
            
            # Get velocity from optical flow
            try:
                vx = self.tello.get_speed_x() / 100.0  # dm/s to m/s
                vy = self.tello.get_speed_y() / 100.0
                vz = self.tello.get_speed_z() / 100.0
            except:
                vx = vy = vz = 0.0
            
            # Get height from barometer
            try:
                height_cm = self.tello.get_height()
                z = height_cm / 100.0 if height_cm > 0 else self.position_estimate[2]  # Keep last Z if sensor fails
            except:
                z = self.position_estimate[2]  # Keep last known Z
            
            # Integrate velocity for X, Y position
            if self.last_update_time is not None:
                dt = current_time - self.last_update_time
                if dt > 0 and dt < 1.0:  # Sanity check dt
                    self.position_estimate[0] += vx * dt
                    self.position_estimate[1] += vy * dt
                    # Only update Z if we got a valid reading, otherwise keep last value
                    if z > 0 or abs(z - self.position_estimate[2]) < 0.5:  # Valid reading or small change
                        self.position_estimate[2] = z
            else:
                if z > 0:
                    self.position_estimate[2] = z
            
            self.last_update_time = current_time
            state[0:3] = self.position_estimate
        
        # Orientation [roll, pitch, yaw]
        if self.use_mocap and self.mocap_wrapper and self.mocap_wrapper.is_tracked():
            quat = self.mocap_wrapper.get_quaternion()
            if quat is not None:
                # Quaternion to Euler angles
                qx, qy, qz, qw = quat
                
                # Roll (x-axis rotation)
                sinr_cosp = 2 * (qw * qx + qy * qz)
                cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
                roll = np.arctan2(sinr_cosp, cosr_cosp)
                
                # Pitch (y-axis rotation)
                sinp = 2 * (qw * qy - qz * qx)
                pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
                
                # Yaw (z-axis rotation)
                siny_cosp = 2 * (qw * qz + qx * qy)
                cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
                yaw = np.arctan2(siny_cosp, cosy_cosp)
                
                state[3:6] = [roll, pitch, yaw]
        else:
            # Tello IMU (pitch and roll only, no yaw from IMU directly)
            try:
                pitch = np.radians(self.tello.get_pitch())
                roll = np.radians(self.tello.get_roll())
                yaw = np.radians(self.tello.get_yaw())
                state[3:6] = [roll, pitch, yaw]
            except Exception as e:
                print(f"⚠️  Tello IMU error: {e}")
        
        # Velocity [vx, vy, vz] - from Tello optical flow
        try:
            vx = self.tello.get_speed_x() / 100.0  # dm/s to m/s
            vy = self.tello.get_speed_y() / 100.0
            vz = self.tello.get_speed_z() / 100.0
            state[6:9] = [vx, vy, vz]
        except Exception as e:
            print(f"⚠️  Tello velocity error: {e}")
        
        # Angular velocity [wx, wy, wz] - computed from orientation change
        # (Will be computed from derivatives in post-processing)
        state[9:12] = np.zeros(3)
        
        return state
    
    def get_target_position(self, t_elapsed):
        """Get target position and velocity from trajectory"""
        # Use learned trajectory waypoints if available
        if self.learned_traj is not None:
            # During approach phase (first time only)
            if self.approach_waypoints is not None and t_elapsed < self.approach_duration:
                waypoints = self.approach_waypoints
                waypoint_times = self.approach_times
                t_traj = t_elapsed
            else:
                # Loop phase (repeats continuously)
                waypoints = self.loop_waypoints
                waypoint_times = self.loop_times
                
                # Adjust time for approach offset and loop
                t_after_approach = t_elapsed - (self.approach_duration if self.approach_waypoints is not None else 0)
                t_traj = t_after_approach % self.loop_duration
            
            # Interpolate between waypoints
            idx = np.searchsorted(waypoint_times, t_traj)
            if idx == 0:
                target_pos = waypoints[0]
                target_vel = np.zeros(3)
            elif idx >= len(waypoints):
                target_pos = waypoints[-1]
                target_vel = np.zeros(3)
            else:
                alpha = (t_traj - waypoint_times[idx-1]) / (waypoint_times[idx] - waypoint_times[idx-1])
                target_pos = (1 - alpha) * waypoints[idx-1] + alpha * waypoints[idx]
                
                # Compute velocity from waypoint spacing
                # Use forward difference for better accuracy
                if idx < len(waypoints) - 1:
                    # Look ahead to next waypoint
                    dt = waypoint_times[idx+1] - waypoint_times[idx]
                    target_vel = (waypoints[idx+1] - waypoints[idx]) / dt
                else:
                    # Use backward difference at end
                    dt = waypoint_times[idx] - waypoint_times[idx-1]
                    target_vel = (waypoints[idx] - waypoints[idx-1]) / dt
            
            return target_pos, target_vel
        
        # Use generated trajectories
        if self.trajectory_type == 'circle':
            return TrajectoryGenerator.circle(t_elapsed)
        elif self.trajectory_type == 'figure8':
            return TrajectoryGenerator.figure8(t_elapsed)
        elif self.trajectory_type == 'spiral':
            return TrajectoryGenerator.spiral(t_elapsed)
        elif self.trajectory_type == 'waypoint':
            return TrajectoryGenerator.waypoint(t_elapsed, self.waypoints)
        elif self.trajectory_type == 'hover':
            return TrajectoryGenerator.hover(t_elapsed)
        else:
            return np.zeros(3), np.zeros(3)
    
    def fly_trajectory(self, duration=60.0, control_rate=20.0):
        """Execute autonomous flight along trajectory"""
        print(f"\n Starting autonomous flight")
        print(f"   Duration: {duration}s")
        print(f"   Control rate: {control_rate} Hz")
        print("\n   Press Ctrl+C to emergency stop\n")
        
        # Takeoff
        print("     Taking off...", flush=True)
        try:
            self.tello.takeoff()
            print("   ✓ Takeoff command sent", flush=True)
        except Exception as e:
            print(f"   ✗ Takeoff failed: {e}", flush=True)
            raise
        time.sleep(3)  # Stabilize
        print("   ✓ Stabilized", flush=True)
        
        # Get current position after takeoff to use as trajectory origin
        time.sleep(0.5)
        current_state = self.get_state()
        takeoff_position = current_state[0:3].copy()
        print(f"   📍 Takeoff position (relative): [{takeoff_position[0]:.3f}, {takeoff_position[1]:.3f}, {takeoff_position[2]:.3f}]", flush=True)
        print(f"   📍 Takeoff yaw: {np.degrees(current_state[5]):.1f}°", flush=True)
        
        # Print first trajectory target for debugging
        target_pos_test, _ = self.get_target_position(0.0)
        print(f"    First trajectory target: [{target_pos_test[0]:.3f}, {target_pos_test[1]:.3f}, {target_pos_test[2]:.3f}]", flush=True)
        print(f"    Initial position error: {np.linalg.norm(takeoff_position - target_pos_test):.3f}m", flush=True)
        
        self.trajectory_start_time = time.time()
        dt = 1.0 / control_rate
        max_iterations = int(duration * control_rate * 1.5)  # Safety: 1.5x expected iterations
        iteration = 0
        
        try:
            while True:
                loop_start = time.time()
                t_elapsed = loop_start - self.trajectory_start_time
                iteration += 1
                
                # Safety timeout (in case time check fails)
                if iteration > max_iterations:
                    print(f"\n   ⚠️  Safety timeout reached ({max_iterations} iterations)")
                    break
                
                if t_elapsed >= duration:
                    print(f"\n   ✓ Completed {duration:.1f}s flight")
                    break
                
                # Get current state
                state = self.get_state()
                
                # Get target from trajectory
                target_pos, target_vel = self.get_target_position(t_elapsed)
                
                # State is relative to initial_position (where drone was when MoCap connected)
                # Trajectories are designed around origin (0,0,0) with Z~1.0m
                # After takeoff, drone should be near (0,0,~0.9-1.0) in relative coords
                # So we can use trajectory targets directly - NO OFFSET NEEDED
                target_pos_adjusted = target_pos
                
                # Compute control action
                if self.open_loop:
                    # Open-loop: Use velocity from trajectory directly (no position feedback)
                    action = np.concatenate([target_vel, [0.0]])  # [vx, vy, vz, yaw_rate=0]
                else:
                    # Closed-loop: PID control with position feedback
                    action = self.controller.compute_control(state, target_pos_adjusted)
                
                # Transform velocity from world frame to body frame
                # action = [vx_world, vy_world, vz_world, yaw_rate]
                # Tello RC command expects body-frame velocities:
                # rc a b c d where a=left/right, b=forward/back, c=up/down, d=yaw
                yaw = state[5]  # Current yaw angle from state
                
                # Rotation from world to body frame (2D rotation in XY plane)
                vx_world = action[0]
                vy_world = action[1]
                vx_body = vx_world * np.cos(yaw) + vy_world * np.sin(yaw)   # forward/back
                vy_body = -vx_world * np.sin(yaw) + vy_world * np.cos(yaw)  # left/right
                
                # Convert to Tello RC control format: [-100, 100]
                # SDK: rc a b c d = rc left/right forward/back up/down yaw
                left_right_cmd = int(np.clip(vy_body * 100, -100, 100))  # m/s to cm/s
                fwd_back_cmd = int(np.clip(vx_body * 100, -100, 100))
                up_down_cmd = int(np.clip(action[2] * 100, -100, 100))
                yaw_cmd = int(np.clip(action[3] * 100, -100, 100))
                
                self.tello.send_rc_control(left_right_cmd, fwd_back_cmd, up_down_cmd, yaw_cmd)
                
                # Get battery (safely)
                try:
                    battery = self.tello.get_battery()
                    self.battery_history.append(battery)
                except:
                    battery = self.battery_history[-1] if self.battery_history else 0
                
                # Log data
                self.states.append(state.copy())
                self.actions.append(action.copy())
                
                # Status update every 1 second
                if len(self.states) % int(control_rate) == 0:
                    pos_error = np.linalg.norm(state[0:3] - target_pos_adjusted)
                    print(f"   t={t_elapsed:5.1f}s | Pos: [{state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}] | "
                          f"Target: [{target_pos_adjusted[0]:.2f}, {target_pos_adjusted[1]:.2f}, {target_pos_adjusted[2]:.2f}] | "
                          f"Cmd: [LR:{left_right_cmd}, FB:{fwd_back_cmd}, UD:{up_down_cmd}] | Error: {pos_error:.3f}m | Bat: {battery}%")
                elif iteration % 5 == 0:  # Dot every 5 iterations (~0.25s) to show progress
                    print(".", end="", flush=True)
                
                # Maintain control rate
                elapsed = time.time() - loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
        
        except KeyboardInterrupt:
            print("\n\n   ⚠️  EMERGENCY STOP")
        
        finally:
            # Stop and land
            print("\n   🛬 Landing...")
            self.tello.send_rc_control(0, 0, 0, 0)
            time.sleep(0.5)
            self.tello.land()
            time.sleep(2)
    
    def save_data(self, output_dir="data/tello_flights"):
        """Save collected flight data"""
        if len(self.states) == 0:
            print("   ✗ No data collected!")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"autonomous_{self.trajectory_type}_{timestamp}.pkl"
        filepath = output_path / filename
        
        data = {
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'timestamps': np.array(self.timestamps),
            'battery_history': self.battery_history,
            'trajectory_type': self.trajectory_type,
            'use_mocap': self.use_mocap,
            'controller': {
                'type': 'VelocityPID',
                'kp': self.controller.kp,
                'max_vel': self.controller.max_vel
            },
            'metadata': {
                'flight_mode': 'autonomous',
                'duration': self.timestamps[-1] if self.timestamps else 0,
                'samples': len(self.states),
                'battery_start': self.battery_history[0] if self.battery_history else 0,
                'battery_end': self.battery_history[-1] if self.battery_history else 0,
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"\n💾 Saved flight data:")
        print(f"   File: {filepath}")
        print(f"   Samples: {len(self.states)}")
        print(f"   Duration: {self.timestamps[-1]:.1f}s")
        print(f"   Trajectory: {self.trajectory_type}")
        print(f"   MoCap: {self.use_mocap}")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.use_mocap and self.mocap_wrapper:
                self.mocap_wrapper.stop()
        except:
            pass
        
        try:
            self.tello.streamoff()
        except:
            pass
        
        try:
            # Properly close the Tello connection
            self.tello.end()
            
            # Wait for UDP threads to actually close
            time.sleep(0.5)
            
            # Force cleanup of djitellopy global state (CRITICAL for re-connection)
            import djitellopy.tello as tello_module
            if hasattr(tello_module, 'drones'):
                # Remove this specific drone from the global registry
                host = self.tello.address[0]
                if host in tello_module.drones:
                    del tello_module.drones[host]
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")
            # Last resort: clear all drones
            try:
                import djitellopy.tello as tello_module
                if hasattr(tello_module, 'drones'):
                    tello_module.drones.clear()
            except:
                pass


def tune_pid_gains(args):
    """Interactive PID tuning mode"""
    print("\n🔧 PID TUNING MODE")
    print("=" * 60)
    print("Testing 20 PID configurations for comprehensive tuning.")
    print("=" * 60)
    
    # 20 tests: 5 kp values × 4 max_vel values
    kp_values = [0.3, 0.5, 0.7, 0.9, 1.1]
    max_vel_values = [0.4, 0.5, 0.7, 0.9]
    
    test_configs = [(kp, vel) for kp in kp_values for vel in max_vel_values]
    
    results = []
    
    # Create ONE Tello connection for all tests (CRITICAL: avoids djitellopy reconnection bug)
    from djitellopy import Tello
    tello = Tello()
    
    print("📡 Connecting to Tello (once for all tests)...")
    try:
        tello.connect()
        battery = tello.get_battery()
        print(f"   ✓ Battery: {battery}%")
        
        if battery < 10:
            print("   ✗ Battery too low! Charge before flying.")
            return
        
        tello.streamon()
        time.sleep(1)
    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
        return
    
    # Setup MoCap once
    mocap_wrapper = None
    use_mocap = args.mocap
    if use_mocap:
        print("\n📍 Connecting to MoCap...")
        try:
            from src.hardware.mocap_wrapper import MocapWrapper
            mocap_wrapper = MocapWrapper(
                mode="multicast",
                interface_ip="192.168.1.1",
                mcast_addr="239.255.42.99",
                data_port=1511
            )
            mocap_wrapper.start()
            time.sleep(2)
            
            pos = mocap_wrapper.get_position()
            if pos is not None:
                print(f"   ✓ MoCap tracking: {pos}")
            else:
                print("   ✗ No MoCap data! Flying blind (using Tello sensors)")
                use_mocap = False
        except Exception as e:
            print(f"   ✗ MoCap connection failed: {e}")
            use_mocap = False
    
    # Run tests with shared Tello connection
    for kp, max_vel in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing: kp={kp}, max_vel={max_vel}")
        print(f"{'='*60}")
        
        # Create controller with existing Tello instance
        controller = AutonomousTelloController(
            trajectory_type='hover',
            kp=kp,
            max_vel=max_vel,
            use_mocap=use_mocap,
            tello_instance=tello  # Pass shared Tello instance
        )
        
        # Inject the shared MoCap instance
        controller.mocap_wrapper = mocap_wrapper
        
        try:
                print("   [DEBUG] Starting fly_trajectory()...", flush=True)
                controller.fly_trajectory(duration=5.0, control_rate=20.0)
                print("   [DEBUG] Flight completed, saving data...", flush=True)
                controller.save_data()
                print("   [DEBUG] Data saved successfully", flush=True)
                
                # Compute tracking error
                states = np.array(controller.states)
                if len(states) == 0:
                    print("   ✗ No states collected!")
                    continue
                    
                target = np.array([0, 0, 1.0])
                errors = np.linalg.norm(states[:, 0:3] - target, axis=1)
                
                result = {
                    'kp': kp,
                    'max_vel': max_vel,
                    'mean_error': np.mean(errors),
                    'std_error': np.std(errors),
                    'max_error': np.max(errors),
                }
                results.append(result)
                print(f"   ✓ Result added to list", flush=True)
                
                print(f"\n   Tracking Error: {result['mean_error']:.3f} ± {result['std_error']:.3f} m (max: {result['max_error']:.3f})")
            
        except Exception as e:
            print(f"   ✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to land if crashed mid-flight
            try:
                tello.send_rc_control(0, 0, 0, 0)
                time.sleep(0.2)
                tello.land()
                time.sleep(2)
            except:
                pass
        
        # Don't call cleanup() - we reuse the Tello connection!
        # Just reset controller state
        controller.states = []
        controller.actions = []
        controller.timestamps = []
        controller.battery_history = []
        controller.trajectory_start_time = None  # Reset timer
        print("   ⏳ Waiting before next test...")
        time.sleep(2)
    
    # Final cleanup after ALL tests
    print("\n🔌 Closing connections...")
    try:
        tello.streamoff()
        tello.end()
    except:
        pass
    
    if mocap_wrapper:
        try:
            mocap_wrapper.stop()
        except:
            pass
    
    # Print summary
    print("\n\n" + "="*60)
    print("PID TUNING RESULTS")
    print("="*60)
    
    if not results:
        print(" No successful tests completed!")
        return
    
    print(f"{'kp':>6} {'max_vel':>8} {'Mean Error':>12} {'Std Error':>12} {'Max Error':>12}")
    print("-"*60)
    for r in sorted(results, key=lambda x: x['mean_error']):
        print(f"{r['kp']:6.1f} {r['max_vel']:8.1f} {r['mean_error']:12.3f} {r['std_error']:12.3f} {r['max_error']:12.3f}")
    
    # Best configuration
    best = min(results, key=lambda x: x['mean_error'])
    print(f"\n🏆 Best configuration: kp={best['kp']}, max_vel={best['max_vel']}")
    print(f"   Mean tracking error: {best['mean_error']:.3f} m")


def main():
    parser = argparse.ArgumentParser(description="Autonomous Tello data collection")
    parser.add_argument('--trajectory', type=str, default='circle',
                        choices=['circle', 'square', 'figure8', 'spiral', 'hover'],
                        help='Trajectory type')
    parser.add_argument('--trajectory-file', type=str, default=None,
                        help='Path to learned trajectory .pkl file (overrides --trajectory)')
    parser.add_argument('--duration', type=float, default=60.0,
                        help='Flight duration in seconds')
    parser.add_argument('--rate', type=float, default=20.0,
                        help='Control loop rate in Hz')
    parser.add_argument('--kp', type=float, default=0.7,
                        help='Position PID proportional gain')
    parser.add_argument('--max-vel', type=float, default=0.7,
                        help='Maximum velocity in m/s')
    parser.add_argument('--mocap', action='store_true',
                        help='Use MoCap for ground truth position')
    parser.add_argument('--open-loop', action='store_true',
                        help='Use open-loop velocity control (no position feedback)')
    parser.add_argument('--tune-pid', action='store_true',
                        help='Run PID tuning mode (tests multiple gain combinations)')
    parser.add_argument('--output-dir', type=str, default='data/tello_flights',
                        help='Output directory for flight data')
    
    args = parser.parse_args()
    
    if args.tune_pid:
        tune_pid_gains(args)
        return
    
    # Load learned trajectory if provided
    trajectory_type = args.trajectory
    if args.trajectory_file:
        print(f"Loading learned trajectory from: {args.trajectory_file}")
        with open(args.trajectory_file, 'rb') as f:
            learned_traj = pickle.load(f)
        trajectory_type = f"learned_{learned_traj['trajectory_label']}"
        print(f"  Trajectory: {learned_traj['trajectory_label']}")
        print(f"  Waypoints: {learned_traj['num_waypoints']}")
        print(f"  Duration: {learned_traj['duration']:.1f}s")
    else:
        learned_traj = None
    
    controller = AutonomousTelloController(
        trajectory_type=trajectory_type,
        kp=args.kp,
        max_vel=args.max_vel,
        use_mocap=args.mocap,
        open_loop=args.open_loop
    )
    
    # Set learned trajectory if loaded
    if learned_traj:
        controller.set_learned_trajectory(learned_traj)
    
    if not controller.connect():
        print("✗ Connection failed!")
        return
    
    try:
        controller.fly_trajectory(duration=args.duration, control_rate=args.rate)
        controller.save_data(output_dir=args.output_dir)
    except Exception as e:
        print(f"\n✗ Flight failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        controller.cleanup()
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()
