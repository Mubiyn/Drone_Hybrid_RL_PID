#!/usr/bin/env python3
"""
Manual Tello Control for Data Collection
Uses keyboard to fly Tello and records flight data for training.

Controls:
  W/S: Forward/Backward
  A/D: Left/Right
  Up/Down Arrow: Altitude
  Q/E: Rotate Left/Right
  T: Takeoff
  L: Land
  Space: Emergency Stop
  R: Start/Stop Recording
  1-4: Set Trajectory
  ESC: Quit

macOS Setup:
  This script requires Accessibility permissions for keyboard monitoring.
  Go to: System Settings → Privacy & Security → Accessibility
  Add Terminal (or your Python IDE) to the allowed apps.
"""

import time
import numpy as np
import pickle
import json
from datetime import datetime
from pathlib import Path
import warnings

try:
    from pynput import keyboard
except ImportError:
    print("Installing pynput for keyboard control...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'pynput'])
    from pynput import keyboard

from djitellopy import Tello


class TelloManualController:
    """Manual control of Tello with data logging for training dataset creation"""
    
    def __init__(self, data_dir="data/tello_flights"):
        self.tello = Tello()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Control state
        self.velocities = {'lr': 0, 'fb': 0, 'ud': 0, 'yaw': 0}
        self.is_flying = False
        self.recording = False
        self.running = True
        
        # Data buffers
        self.flight_data = []
        self.start_time = None
        self.episode_id = 0
        
        # Control parameters
        self.speed_increment = 20
        self.max_speed = 80
        
        # Motion capture (optional)
        self.mocap_client = None
        self.use_mocap = False
        self.initial_position = None  # For relative positioning with MoCap
        
        # Trajectory tracking
        self.target_trajectory = None
        self.trajectory_type = None
        
        # Previous state for computing derivatives
        self.prev_state = None
        self.prev_time = None
        
        print("=" * 60)
        print("Tello Manual Control - Data Collection")
        print("=" * 60)
        
    def connect(self, use_mocap=False):
        """Connect to Tello and optionally MoCap"""
        print(f"\n[DEBUG] connect() called with use_mocap={use_mocap}")
        print("\nConnecting to Tello...")
        self.tello.connect()
        battery = self.tello.get_battery()
        print(f"✓ Connected! Battery: {battery}%")
        
        if battery < 20:
            print("⚠️  WARNING: Low battery! Charge before flying.")
            return False
        elif battery < 30:
            print("⚠️  Battery is low (30-40%). Flight time will be limited.")
            
        # Pre-flight checks
        print("\nPre-flight checks:")
        try:
            # Enable SDK mode explicitly
            self.tello.send_command_without_return("command")
            time.sleep(0.5)
            print("  ✓ SDK mode enabled")
            
            # Check sensors
            temp = self.tello.get_temperature()
            print(f"  ✓ Temperature: {temp}°C")
            
            height = self.tello.get_height()
            print(f"  ✓ Height sensor: {height}cm")
            
            print("\n⚠️  IMPORTANT: Ensure drone is on FLAT SURFACE")
            print("  - Props are clear and spin freely")
            print("  - No obstructions within 3m radius")
            print("  - Battery charged (current: {battery}%)")
            
        except Exception as e:
            print(f"  ⚠️  Sensor check failed: {e}")
            
        # Start video stream
        self.tello.streamoff()
        time.sleep(0.5)
        self.tello.streamon()
        print("  ✓ Video stream started")
        
        # Optional MoCap
        print(f"[DEBUG] Checking MoCap: use_mocap={use_mocap}")
        if use_mocap:
            print("[DEBUG] Attempting to initialize MoCap...")
            try:
                from src.real_drone.mocap_wrapper import MocapWrapper
                
                print("[DEBUG] MocapWrapper imported successfully")
                self.mocap_client = MocapWrapper(
                    mode="multicast",
                    interface_ip="192.168.1.1",
                    mcast_addr="239.255.42.99",
                    data_port=1511
                )
                print("[DEBUG] MocapWrapper created, calling start()...")
                self.mocap_client.start()
                time.sleep(2)  # Allow connection to establish
                self.use_mocap = True
                print("✓ Motion Capture connected")
                print(f"[DEBUG] self.use_mocap is now: {self.use_mocap}")
            except Exception as e:
                print(f"⚠️  MoCap failed: {e}")
                import traceback
                traceback.print_exc()
                print("Continuing without MoCap...")
        else:
            print("[DEBUG] use_mocap is False, skipping MoCap initialization")
                
        return True
        
    def get_state(self):
        """Get 12-dim state: [x,y,z, r,p,y, vx,vy,vz, wx,wy,wz]"""
        current_time = time.time()
        
        # Position (WORLD FRAME)
        if self.use_mocap and self.mocap_client:
            pos = self.mocap_client.get_position()
            if pos is not None:
                # Track position relative to initial position (like autonomous mode)
                if self.initial_position is None:
                    self.initial_position = pos.copy()
                pos = pos - self.initial_position  # Relative position
            else:
                pos = np.array([0.0, 0.0, 1.0])
        else:
            # Without MoCap, use dead reckoning + height sensor
            try:
                height_cm = self.tello.get_height()
                z = height_cm / 100.0  # cm to meters
            except:
                z = 1.0  # Fallback
            
            # Integrate velocity for x, y position estimate
            if self.prev_state is not None and self.prev_time is not None:
                dt = current_time - self.prev_time
                if dt > 0 and dt < 1.0:  # Sanity check
                    # Get previous velocity in world frame
                    prev_vx, prev_vy = self.prev_state[6], self.prev_state[7]
                    # Simple integration (assumes velocity constant over dt)
                    dx = prev_vx * dt
                    dy = prev_vy * dt
                    x = self.prev_state[0] + dx
                    y = self.prev_state[1] + dy
                else:
                    x, y = 0.0, 0.0
            else:
                x, y = 0.0, 0.0
            
            pos = np.array([x, y, z])
            
        # Attitude - try MoCap first, then Tello IMU
        if self.use_mocap and self.mocap_client:
            quat = self.mocap_client.get_quaternion()
            if quat is not None:
                # Convert quaternion to Euler angles
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
            else:
                # Fallback to Tello IMU
                roll = np.radians(self.tello.get_roll())
                pitch = np.radians(self.tello.get_pitch())
                yaw = np.radians(self.tello.get_yaw())
        else:
            # Use Tello IMU
            try:
                pitch = np.radians(self.tello.get_pitch())
                roll = np.radians(self.tello.get_roll())
                yaw = np.radians(self.tello.get_yaw())
            except Exception as e:
                print(f"⚠️  Tello IMU error: {e}")
                pitch, roll, yaw = 0.0, 0.0, 0.0
            
        # Velocity (SDK: vgx, vgy, vgz in dm/s → convert to m/s)
        # Note: Tello velocities are in BODY FRAME (forward/left/up)
        try:
            vx = self.tello.get_speed_x() / 100.0  # cm/s to m/s
            vy = self.tello.get_speed_y() / 100.0
            vz = self.tello.get_speed_z() / 100.0
        except Exception as e:
            print(f"⚠️  Tello velocity error: {e}")
            vx, vy, vz = 0.0, 0.0, 0.0
            
        # Angular velocity (estimate)
        wx, wy, wz = 0.0, 0.0, 0.0
        if self.prev_state is not None and self.prev_time is not None:
            dt = time.time() - self.prev_time
            if dt > 0:
                prev_orient = self.prev_state[3:6]
                curr_orient = np.array([roll, pitch, yaw])
                angular_vel = (curr_orient - prev_orient) / dt
                wx, wy, wz = angular_vel
        
        state = np.array([
            pos[0], pos[1], pos[2],
            roll, pitch, yaw,
            vx, vy, vz,
            wx, wy, wz
        ], dtype=np.float32)
        
        return state
    
    def get_extended_observation(self, state, target_pos, target_vel):
        """Build 18-dim obs: [state(12), pos_error(3), vel_error(3)]"""
        current_pos = state[0:3]
        current_vel = state[6:9]
        
        pos_error = target_pos - current_pos
        vel_error = target_vel - current_vel
        
        return np.concatenate([state, pos_error, vel_error]).astype(np.float32)
    
    def compute_reward(self, state, target_pos, target_vel):
        """Compute reward matching training"""
        current_pos = state[0:3]
        current_vel = state[6:9]
        roll, pitch = state[3], state[4]
        
        pos_error = np.linalg.norm(current_pos - target_pos)
        pos_reward = -20.0 * (1.0 - np.exp(-5.0 * pos_error**2))
        
        vel_error = np.linalg.norm(current_vel - target_vel)
        vel_reward = -0.5 * vel_error
        
        attitude_penalty = -0.1 * (roll**2 + pitch**2)
        
        total_reward = pos_reward + vel_reward + attitude_penalty
        
        return total_reward, {
            'pos_error': pos_error,
            'vel_error': vel_error,
            'pos_reward': pos_reward,
            'vel_reward': vel_reward,
            'attitude_penalty': attitude_penalty
        }
    
    def set_trajectory(self, traj_type):
        """Set trajectory label for organizing data"""
        self.trajectory_type = traj_type
        print(f"\n📍 Label: {traj_type.upper()}")
        print("   Data will be saved with this label for organization")
        
    def toggle_recording(self):
        """Start/stop recording"""
        if not self.recording:
            self.recording = True
            self.start_time = time.time()
            self.flight_data = []
            self.episode_id += 1
            
            if self.trajectory_type:
                print(f"\n🔴 RECORDING (Episode {self.episode_id}, Label: {self.trajectory_type})")
            else:
                print(f"\n🔴 RECORDING (Episode {self.episode_id}, No label)")
                print("   TIP: Press 1-4 to label this flight")
        else:
            self.recording = False
            self.save_data()
            print("⏹️  STOPPED")
    
    def save_data(self):
        """Save comprehensive flight data"""
        if len(self.flight_data) == 0:
            print("⚠️  No data to save")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        traj_suffix = f"_{self.trajectory_type}" if self.trajectory_type else ""
        filename = self.data_dir / f"flight_{timestamp}{traj_suffix}.pkl"
        
        # Build dataset - simple states + actions for training
        data = {
            'states': np.array([d['state'] for d in self.flight_data]),
            'actions': np.array([d['action'] for d in self.flight_data]),
            'timestamps': np.array([d['timestamp'] for d in self.flight_data]),
            'battery_history': np.array([d.get('battery', 100) for d in self.flight_data]),
            'trajectory_label': self.trajectory_type,  # Optional label for organizing data
            'episode_id': self.episode_id,
            'use_mocap': self.use_mocap,
            'duration': self.flight_data[-1]['timestamp'] - self.flight_data[0]['timestamp'],
            'samples': len(self.flight_data)
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
            
        # Metadata
        metadata = {
            'timestamp': timestamp,
            'episode_id': self.episode_id,
            'trajectory_label': self.trajectory_type,
            'duration': float(data['duration']),
            'samples': int(data['samples']),
            'use_mocap': self.use_mocap,
            'sample_rate': data['samples'] / data['duration'],
            'battery_start': int(data['battery_history'][0]),
            'battery_end': int(data['battery_history'][-1]),
            'battery_used': int(data['battery_history'][0] - data['battery_history'][-1])
        }
        
        meta_file = self.data_dir / f"flight_{timestamp}{traj_suffix}_meta.json"
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"✓ Saved {len(self.flight_data)} samples to {filename}")
        print(f"  Episode: {self.episode_id}")
        print(f"  Label: {self.trajectory_type or 'Free flight'}")
        print(f"  Duration: {data['duration']:.1f}s @ {metadata['sample_rate']:.1f} Hz")
        print(f"  Battery: {metadata['battery_start']}% → {metadata['battery_end']}% ({metadata['battery_used']}% used)")
    
    def stop_movement(self):
        """Stop all movement"""
        self.velocities = {'lr': 0, 'fb': 0, 'ud': 0, 'yaw': 0}
        if self.is_flying:
            self.tello.send_rc_control(0, 0, 0, 0)
    
    def on_press(self, key):
        """Handle key presses"""
        try:
            # DEBUG: Print every key press to see if listener is working
            print(f"\n[DEBUG] Key pressed: {key}")
            
            if hasattr(key, 'char'):
                k = key.char.lower()
                print(f"[DEBUG] Char key: {k}")
                
                if k == 't' and not self.is_flying:
                    print("\n🚁 TAKING OFF...")
                    print("   Ensure drone is on flat surface!")
                    try:
                        # Try to enter command mode first
                        self.tello.send_command_without_return("command")
                        time.sleep(0.3)
                        
                        # Attempt takeoff
                        self.tello.takeoff()
                        self.is_flying = True
                        time.sleep(3)
                        print("✓ Airborne!")
                    except Exception as e:
                        print(f"✗ Takeoff failed: {e}")
                        print("\nTROUBLESHOOTING:")
                        print("  1. Ensure drone is on FLAT surface (not tilted)")
                        print("  2. Check props spin freely (no obstructions)")
                        print("  3. Battery should be >30% (current: {}%)".format(
                            self.tello.get_battery()))
                        print("  4. Try power cycling the drone")
                        print("  5. Ensure sufficient space for takeoff (3m radius)")
                        self.is_flying = False
                    
                elif k == 'l' and self.is_flying:
                    print("\n🛬 LANDING...")
                    self.stop_movement()
                    self.tello.land()
                    self.is_flying = False
                    print("✓ Landed")
                    
                elif k == 'r':
                    self.toggle_recording()
                    
                elif k == 'w':
                    self.velocities['fb'] = min(self.max_speed, self.velocities['fb'] + self.speed_increment)
                elif k == 's':
                    self.velocities['fb'] = max(-self.max_speed, self.velocities['fb'] - self.speed_increment)
                elif k == 'a':
                    self.velocities['lr'] = max(-self.max_speed, self.velocities['lr'] - self.speed_increment)
                elif k == 'd':
                    self.velocities['lr'] = min(self.max_speed, self.velocities['lr'] + self.speed_increment)
                elif k == 'q':
                    self.velocities['yaw'] = max(-self.max_speed, self.velocities['yaw'] - self.speed_increment)
                elif k == 'e':
                    self.velocities['yaw'] = min(self.max_speed, self.velocities['yaw'] + self.speed_increment)
                    
                elif k == '1':
                    self.set_trajectory('hover')
                elif k == '2':
                    self.set_trajectory('circle')
                elif k == '3':
                    self.set_trajectory('figure8')
                elif k == '4':
                    self.set_trajectory('spiral')
                    
            elif key == keyboard.Key.up:
                self.velocities['ud'] = min(self.max_speed, self.velocities['ud'] + self.speed_increment)
            elif key == keyboard.Key.down:
                self.velocities['ud'] = max(-self.max_speed, self.velocities['ud'] - self.speed_increment)
            elif key == keyboard.Key.space:
                print("\n⚠️  EMERGENCY STOP!")
                self.stop_movement()
            elif key == keyboard.Key.esc:
                print("\n👋 Shutting down...")
                self.running = False
                return False
                
        except AttributeError:
            pass
    
    def on_release(self, key):
        """Handle key release - STOP immediately (direct control)"""
        try:
            if hasattr(key, 'char'):
                k = key.char.lower()
                if k in ['w', 's']:
                    self.velocities['fb'] = 0
                elif k in ['a', 'd']:
                    self.velocities['lr'] = 0
                elif k in ['q', 'e']:
                    self.velocities['yaw'] = 0
            elif key in [keyboard.Key.up, keyboard.Key.down]:
                self.velocities['ud'] = 0
        except AttributeError:
            pass
    
    def run(self):
        """Main control loop"""
        print("\n" + "=" * 60)
        print("CONTROLS:")
        print("  W/S: Forward/Backward    A/D: Left/Right")
        print("  ↑/↓: Up/Down            Q/E: Rotate")
        print("  T: Takeoff              L: Land")
        print("  R: Start/Stop Recording  Space: Stop")
        print("  1-4: Set Trajectory     ESC: Quit")
        print("")
        print("KEY COMBINATIONS (for smooth patterns):")
        print("  W+D: Right curve (for circles)")
        print("  W+A: Left curve (for circles)")
        print("  W+D+↑: Right spiral (upward)")
        print("  W+A+↑: Left spiral (upward)")
        print("  S+A+↓: Descending left turn")
        print("  S+D+↓: Descending right turn")
        print("")
        print("TRAJECTORIES:")
        print("  1: Hover    2: Circle    3: Figure-8    4: Spiral")
        print("=" * 60)
        print("\nPress T to takeoff...")
        
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        
        last_print = time.time()
        control_freq = 20
        dt = 1.0 / control_freq
        
        try:
            while self.running:
                loop_start = time.time()
                
                if self.is_flying:
                    # Get current key states for combination detection
                    lr_cmd = self.velocities['lr']
                    fb_cmd = self.velocities['fb']
                    ud_cmd = self.velocities['ud']
                    yaw_cmd = self.velocities['yaw']
                    
                    # Smart damping for smooth curves
                    # When combining forward with left/right, reduce forward slightly
                    # to create natural circular motion
                    if fb_cmd != 0 and (lr_cmd != 0 or yaw_cmd != 0):
                        # Combination mode: reduce speed for smoother curves
                        fb_cmd = int(fb_cmd * 0.8)
                    
                    # When combining vertical with forward/turn, reduce vertical
                    if ud_cmd != 0 and (fb_cmd != 0 or lr_cmd != 0):
                        ud_cmd = int(ud_cmd * 0.7)
                    
                    self.tello.send_rc_control(lr_cmd, fb_cmd, ud_cmd, yaw_cmd)
                    
                    if self.recording:
                        current_time = time.time()
                        elapsed = current_time - self.start_time
                        state = self.get_state()
                        
                        # Action: actual RC commands sent
                        action = np.array([
                            lr_cmd / 100.0,
                            fb_cmd / 100.0,
                            ud_cmd / 100.0,
                            yaw_cmd / 100.0
                        ], dtype=np.float32)
                        
                        try:
                            battery = self.tello.get_battery()
                        except:
                            battery = 100
                        
                        self.flight_data.append({
                            'timestamp': elapsed,
                            'state': state,
                            'action': action,
                            'battery': battery
                        })
                        
                        self.prev_state = state
                        self.prev_time = current_time
                
                # Status print
                if time.time() - last_print > 1.0:
                    try:
                        battery = self.tello.get_battery()
                    except:
                        battery = 0
                    status = "🔴 REC" if self.recording else "⚪ IDLE"
                    flying = "✈️  FLY" if self.is_flying else "🛬 GND"
                    
                    # Show key combination hint
                    combo = ""
                    if self.velocities['fb'] > 0 and self.velocities['lr'] > 0:
                        combo = "  [Right curve]"
                    elif self.velocities['fb'] > 0 and self.velocities['lr'] < 0:
                        combo = "  [Left curve]"
                    elif self.velocities['fb'] > 0 and self.velocities['ud'] > 0:
                        combo = "  [Right spiral up]"
                    
                    print(f"\r{status} | {flying} | "
                          f"V:[{self.velocities['fb']:3d},{self.velocities['lr']:3d},"
                          f"{self.velocities['ud']:3d},{self.velocities['yaw']:3d}] | "
                          f"Bat:{battery}% | Samples:{len(self.flight_data) if self.recording else 0}{combo}",
                          end='', flush=True)
                    last_print = time.time()
                
                elapsed = time.time() - loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                    
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted")
            
        finally:
            print("\n\nCleaning up...")
            if self.recording:
                self.save_data()
            if self.is_flying:
                print("Landing...")
                self.stop_movement()
                self.tello.land()
            self.tello.streamoff()
            if self.mocap_client:
                self.mocap_client.stop()
            listener.stop()
            print("✓ Shutdown complete")


def main():
    import argparse
    import sys
    
    print(f"[DEBUG] sys.argv = {sys.argv}")
    
    parser = argparse.ArgumentParser(description="Manual Tello control for data collection")
    parser.add_argument('--mocap', action='store_true', help='Use motion capture')
    parser.add_argument('--data-dir', type=str, default='data/tello_flights',
                        help='Directory to save flight data')
    
    args = parser.parse_args()
    
    print(f"[DEBUG] Parsed arguments:")
    print(f"  args.mocap = {args.mocap}")
    print(f"  args.data_dir = {args.data_dir}")
    
    controller = TelloManualController(data_dir=args.data_dir)
    
    if controller.connect(use_mocap=args.mocap):
        controller.run()
    else:
        print("Connection failed.")


if __name__ == '__main__':
    main()
