import time
import numpy as np
from djitellopy import Tello

class TelloWrapper:
    """
    Wrapper for DJI Tello drone to interface with Hybrid RL Controller.
    Provides observation format matching simulation: [x,y,z, r,p,y, vx,vy,vz, wx,wy,wz]
    """
    def __init__(self, mocap_client=None, use_state_estimator=True):
        self.tello = Tello()
        self.tello.connect()
        battery = self.tello.get_battery()
        print(f"Tello Connected - Battery: {battery}%")
        
        if battery < 20:
            print("WARNING: Battery low! Consider charging before flight.")
        
        self.tello.streamoff()
        self.tello.streamon()
        
        # Motion Capture System (OptiTrack, Vicon, etc.)
        self.mocap_client = mocap_client
        self.use_mocap = mocap_client is not None
        
        # State variables
        self.pos = np.array([0.0, 0.0, 0.0])
        self.vel = np.zeros(3)
        self.last_pos = np.zeros(3)
        self.yaw = 0.0
        
        # For velocity estimation
        self.last_time = time.time()
        self.use_state_estimator = use_state_estimator
        
        print(f"Using MoCap: {self.use_mocap}")
        print(f"State Estimator: {self.use_state_estimator}")
        
    def takeoff(self):
        self.tello.takeoff()
        # If MoCap is available, use it to set initial height
        if self.mocap_client:
            self.pos = self.mocap_client.get_position()
        else:
            self.pos[2] = 1.0 # Assume 1m height after takeoff
        
    def land(self):
        self.tello.land()
        
    def get_obs(self):
        """
        Get drone observation matching simulation format:
        [x, y, z, r, p, y, vx, vy, vz, wx, wy, wz] (12 dimensions)
        """
        current_time = time.time()
        dt = current_time - self.last_time
        
        # === POSITION ===
        if self.use_mocap:
            # Use Motion Capture for ground truth position
            new_pos = self.mocap_client.get_position()
            if new_pos is not None:
                self.pos = new_pos
        else:
            # Use Tello's height sensor + dead reckoning (less accurate)
            try:
                height_cm = self.tello.get_height()
                self.pos[2] = height_cm / 100.0  # cm to meters
            except:
                pass  # Keep last known height
        
        # === ATTITUDE (Roll, Pitch, Yaw) ===
        try:
            pitch_deg = self.tello.get_pitch()
            roll_deg = self.tello.get_roll()
            yaw_deg = self.tello.get_yaw()
            
            r = np.radians(roll_deg)
            p = np.radians(pitch_deg)
            
            # Use MoCap yaw if available (more accurate)
            if self.use_mocap:
                y = self.mocap_client.get_yaw()
            else:
                y = np.radians(yaw_deg)
        except:
            r, p, y = 0.0, 0.0, 0.0
        
        # === VELOCITY ===
        if self.use_state_estimator and dt > 0:
            # Estimate velocity from position change (works best with MoCap)
            self.vel = (self.pos - self.last_pos) / dt
            self.last_pos = self.pos.copy()
        else:
            # Use Tello's velocity estimates (dm/s -> m/s)
            try:
                vx = self.tello.get_speed_x() / 10.0
                vy = self.tello.get_speed_y() / 10.0  
                vz = self.tello.get_speed_z() / 10.0
                self.vel = np.array([vx, vy, vz])
            except:
                pass  # Keep last velocity
        
        # === ANGULAR VELOCITY ===
        # Tello doesn't provide gyro data directly
        # Could estimate from attitude changes, but risky
        wx, wy, wz = 0.0, 0.0, 0.0
        
        self.last_time = current_time
        
        obs = np.array([
            self.pos[0], self.pos[1], self.pos[2],  # Position
            r, p, y,                                  # Attitude
            self.vel[0], self.vel[1], self.vel[2],   # Velocity
            wx, wy, wz                                # Angular velocity
        ], dtype=np.float32)
        
        return obs
        
    def step(self, action_vel):
        """
        Send velocity command to Tello.
        
        Args:
            action_vel: [vx, vy, vz, yaw_rate] in (m/s, m/s, m/s, rad/s)
        
        Tello RC Control expects:
            - left_right: -100 to 100
            - forward_back: -100 to 100  
            - up_down: -100 to 100
            - yaw: -100 to 100 (degrees/s)
        
        Tello max speed is ~8 m/s but safer to limit to 1-2 m/s
        """
        # Safety limits (m/s) - Conservative for first flights
        MAX_VEL = 0.8  # Reduced from 1.5 m/s
        vx = np.clip(action_vel[0], -MAX_VEL, MAX_VEL)
        vy = np.clip(action_vel[1], -MAX_VEL, MAX_VEL)
        vz = np.clip(action_vel[2], -MAX_VEL, MAX_VEL)
        yaw_rate = np.clip(action_vel[3], -np.pi/2, np.pi/2)  # Reduced yaw rate
        
        # Convert to Tello units
        # Tello velocity range: -100 to 100
        # More conservative scaling for safety
        scale = 60.0  # 1 m/s = 60 units (gentler than before)
        
        fb = int(vx * scale)  # Forward/backward (x-axis)
        lr = int(vy * scale)  # Left/right (y-axis)
        ud = int(vz * scale)  # Up/down (z-axis)
        yaw_cmd = int(np.degrees(yaw_rate))  # Yaw rate in deg/s
        
        # Final clipping to Tello range (reduced max to 80 for extra safety)
        fb = int(np.clip(fb, -80, 80))
        lr = int(np.clip(lr, -80, 80))
        ud = int(np.clip(ud, -80, 80))
        yaw_cmd = int(np.clip(yaw_cmd, -80, 80))
        
        self.tello.send_rc_control(lr, fb, ud, yaw_cmd)
        
    def emergency_stop(self):
        """Emergency stop - kills motors immediately"""
        print("EMERGENCY STOP!")
        self.tello.emergency()
    
    def get_battery(self):
        """Get current battery percentage"""
        try:
            return self.tello.get_battery()
        except:
            return -1
    
    def get_status(self):
        """Get detailed status info"""
        try:
            return {
                'battery': self.tello.get_battery(),
                'height': self.tello.get_height(),
                'temperature': self.tello.get_temperature(),
                'flight_time': self.tello.get_flight_time(),
            }
        except Exception as e:
            print(f"Error getting status: {e}")
            return {}
    
    def close(self):
        """Cleanup and disconnect"""
        try:
            self.tello.streamoff()
        except:
            pass
        self.tello.end()
        print("Tello disconnected")
