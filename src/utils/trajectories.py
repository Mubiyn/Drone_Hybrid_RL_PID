import numpy as np


class TrajectoryGenerator:
    
    @staticmethod
    def hover(position, duration, freq):
        num_steps = int(duration * freq)
        return np.tile(position, (num_steps, 1))
    
    @staticmethod
    def waypoints(points, duration_per_segment, freq):
        trajectory = []
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]
            steps = int(duration_per_segment * freq)
            segment = np.linspace(start, end, steps)
            trajectory.append(segment)
        return np.vstack(trajectory)
    
    @staticmethod
    def figure8(center, radius, height, period, freq, duration):
        num_steps = int(duration * freq)
        t = np.linspace(0, duration, num_steps)
        
        x = center[0] + radius * np.sin(2 * np.pi * t / period)
        y = center[1] + radius * np.sin(4 * np.pi * t / period) / 2
        z = np.full_like(x, center[2] + height)
        
        return np.column_stack([x, y, z])
    
    @staticmethod
    def circle(center, radius, period, freq, duration):
        num_steps = int(duration * freq)
        t = np.linspace(0, duration, num_steps)
        
        x = center[0] + radius * np.cos(2 * np.pi * t / period)
        y = center[1] + radius * np.sin(2 * np.pi * t / period)
        z = np.full_like(x, center[2])
        
        return np.column_stack([x, y, z])
    
    @staticmethod
    def emergency_landing(start_height, target_height, duration, freq):
        num_steps = int(duration * freq)
        z = np.linspace(start_height, target_height, num_steps)
        x = np.zeros_like(z)
        y = np.zeros_like(z)
        return np.column_stack([x, y, z])


TASK_TRAJECTORIES = {
    'hover': {
        'generator': TrajectoryGenerator.hover,
        'params': {'position': [0, 0, 1.0], 'duration': 10, 'freq': 48}
    },
    'hover_extended': {
        'generator': TrajectoryGenerator.hover,
        'params': {'position': [0, 0, 1.0], 'duration': 30, 'freq': 48}
    },
    'waypoint_delivery': {
        'generator': TrajectoryGenerator.waypoints,
        'params': {
            'points': [[0,0,1], [2,2,1.5], [4,0,1], [2,-2,1.5], [0,0,1]],
            'duration_per_segment': 3,
            'freq': 48
        }
    },
    'figure8': {
        'generator': TrajectoryGenerator.figure8,
        'params': {
            'center': [0, 0, 0],
            'radius': 1.0,
            'height': 1.0,
            'period': 8,
            'freq': 48,
            'duration': 16
        }
    },
    'circle': {
        'generator': TrajectoryGenerator.circle,
        'params': {
            'center': [0, 0, 1],
            'radius': 1.5,
            'period': 10,
            'freq': 48,
            'duration': 20
        }
    },
    'emergency_landing': {
        'generator': TrajectoryGenerator.emergency_landing,
        'params': {
            'start_height': 2.0,
            'target_height': 0.1,
            'duration': 10,
            'freq': 48
        }
    }
}


def get_trajectory(task_name):
    if task_name not in TASK_TRAJECTORIES:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_TRAJECTORIES.keys())}")
    
    task = TASK_TRAJECTORIES[task_name]
    return task['generator'](**task['params'])
