import numpy as np


def compute_rmse(desired, actual):
    return np.sqrt(np.mean(np.sum((desired - actual)**2, axis=1)))


def compute_max_error(desired, actual):
    return np.max(np.linalg.norm(desired - actual, axis=1))


def compute_settling_time(positions, target, threshold=0.05, freq=48):
    errors = np.linalg.norm(positions - target, axis=1)
    
    settling_indices = np.where(errors < threshold)[0]
    if len(settling_indices) == 0:
        return None
    
    first_settle = settling_indices[0]
    
    for i in range(first_settle, len(errors)):
        if errors[i] > threshold:
            return None
    
    return first_settle / freq


def compute_control_effort(actions):
    return {
        'mean': np.mean(np.abs(actions)),
        'max': np.max(np.abs(actions)),
        'std': np.std(actions),
        'saturation_pct': np.mean(np.max(np.abs(actions), axis=1) > 0.95) * 100
    }


def compute_success_rate(errors, threshold=0.1):
    return np.mean(errors < threshold) * 100


def evaluate_trajectory(positions, targets, actions=None):
    results = {
        'rmse': compute_rmse(targets, positions),
        'max_error': compute_max_error(targets, positions),
        'mean_error': np.mean(np.linalg.norm(targets - positions, axis=1)),
        'final_error': np.linalg.norm(targets[-1] - positions[-1])
    }
    
    if len(targets) == 1 or np.allclose(targets[0], targets[-1]):
        settling = compute_settling_time(positions, targets[0])
        if settling is not None:
            results['settling_time'] = settling
    
    if actions is not None:
        results['control_effort'] = compute_control_effort(actions)
    
    return results
