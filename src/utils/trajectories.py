import numpy as np

class TrajectoryGenerator:
    """
    Generates target trajectories for drone tracking tasks.
    Supported types: 'hover', 'circle', 'figure8', 'spiral', 'waypoint', 'square'
    """
    def __init__(self, trajectory_type='hover', duration=10.0, radius=1.0, height=1.0):
        self.trajectory_type = trajectory_type
        self.duration = duration
        self.radius = radius
        self.height = height
        self.center = np.array([0, 0, height])
        
        # Waypoints for 'waypoint' task
        self.waypoints = np.array([
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
            [0, 0, 1]
        ])

    def get_target(self, t):
        """
        Returns the target state at time t.
        
        Args:
            t (float): Current time in seconds.
            
        Returns:
            pos (np.ndarray): Target position [x, y, z]
            vel (np.ndarray): Target velocity [vx, vy, vz]
            yaw (float): Target yaw angle
        """
        if self.trajectory_type == 'hover':
            return self._hover(t)
        elif self.trajectory_type == 'circle':
            return self._circle(t)
        elif self.trajectory_type == 'figure8':
            return self._figure8(t)
        elif self.trajectory_type == 'spiral':
            return self._spiral(t)
        elif self.trajectory_type == 'waypoint':
            return self._waypoint(t)
        elif self.trajectory_type == 'square':
            return self._square(t)
        else:
            raise ValueError(f"Unknown trajectory type: {self.trajectory_type}")

    def _hover(self, t):
        pos = self.center
        vel = np.zeros(3)
        yaw = 0.0
        return pos, vel, yaw

    def _circle(self, t):
        # Circular path in XY plane
        omega = 2 * np.pi / self.duration  # Angular velocity
        angle = omega * t
        
        x = self.radius * np.cos(angle)
        y = self.radius * np.sin(angle)
        z = self.height
        
        vx = -self.radius * omega * np.sin(angle)
        vy = self.radius * omega * np.cos(angle)
        vz = 0
        
        pos = np.array([x, y, z])
        vel = np.array([vx, vy, vz])
        yaw = angle # Face direction of motion? Or 0? Let's keep 0 for simplicity or tangent.
        # For simple tracking, yaw=0 is easier. 
        # If we want it to face forward: yaw = angle + np.pi/2
        yaw = 0.0 
        
        return pos, vel, yaw

    def _figure8(self, t):
        # Lemniscate of Bernoulli or similar
        omega = 2 * np.pi / self.duration
        angle = omega * t
        
        x = self.radius * np.sin(angle)
        y = self.radius * np.sin(angle) * np.cos(angle)
        z = self.height
        
        # Derivatives
        vx = self.radius * omega * np.cos(angle)
        vy = self.radius * omega * (np.cos(angle)**2 - np.sin(angle)**2)
        vz = 0
        
        pos = np.array([x, y, z])
        vel = np.array([vx, vy, vz])
        yaw = 0.0
        return pos, vel, yaw

    def _spiral(self, t):
        # Ascending spiral
        omega = 2 * np.pi / self.duration
        angle = omega * t
        climb_rate = 0.1 # m/s
        
        x = self.radius * np.cos(angle)
        y = self.radius * np.sin(angle)
        z = self.height + climb_rate * t
        
        vx = -self.radius * omega * np.sin(angle)
        vy = self.radius * omega * np.cos(angle)
        vz = climb_rate
        
        pos = np.array([x, y, z])
        vel = np.array([vx, vy, vz])
        yaw = 0.0
        return pos, vel, yaw

    def _waypoint(self, t):
        # Simple linear interpolation between waypoints
        # This is a simplified version. 
        # Assuming total duration is split equally among segments.
        num_segments = len(self.waypoints) - 1
        segment_duration = self.duration / num_segments
        
        segment_idx = int(t // segment_duration)
        if segment_idx >= num_segments:
            segment_idx = num_segments - 1
            local_t = segment_duration
        else:
            local_t = t % segment_duration
            
        p_start = self.waypoints[segment_idx]
        p_end = self.waypoints[segment_idx + 1]
        
        alpha = local_t / segment_duration
        pos = (1 - alpha) * p_start + alpha * p_end
        
        vel = (p_end - p_start) / segment_duration
        yaw = 0.0
        
        return pos, vel, yaw

    def _square(self, t):
        """Square path in XY plane"""
        # Square with side length 2*radius, centered at origin
        side_length = 2 * self.radius
        perimeter = 4 * side_length
        speed = perimeter / self.duration
        
        # Distance traveled
        dist = (speed * t) % perimeter
        
        # Determine which side we're on
        if dist < side_length:  # Bottom side (moving right)
            x = -self.radius + dist
            y = -self.radius
            vx = speed
            vy = 0
        elif dist < 2 * side_length:  # Right side (moving up)
            x = self.radius
            y = -self.radius + (dist - side_length)
            vx = 0
            vy = speed
        elif dist < 3 * side_length:  # Top side (moving left)
            x = self.radius - (dist - 2 * side_length)
            y = self.radius
            vx = -speed
            vy = 0
        else:  # Left side (moving down)
            x = -self.radius
            y = self.radius - (dist - 3 * side_length)
            vx = 0
            vy = -speed
        
        pos = np.array([x, y, self.height])
        vel = np.array([vx, vy, 0])
        yaw = 0.0
        
        return pos, vel, yaw
