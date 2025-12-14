#!/usr/bin/env python3
"""
Phase 1 Simulation Perturbation Testing

Tests models/hybrid_robust/ models against domain randomization in gym-pybullet-drones.
Compares Hybrid RL vs PID baseline under various perturbations.

Perturbation types:
- none: No domain randomization (baseline)
- wind: Mass ±30%, inertia ±30%, wind forces 0.15N
"""

import sys
import pickle
import numpy as np
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from src.envs.HybridAviary import HybridAviary
from src.controllers.pid_controller import PIDController
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


def test_pid_baseline(trajectory_file, n_episodes=5, domain_randomization=False):
    """Test PID baseline controller"""
    
    with open(trajectory_file, 'rb') as f:
        traj = pickle.load(f)
    trajectory_label = traj.get('trajectory_label', 'unknown')
    
    # Tuned gains (same as HybridAviary)
    TUNED_GAINS = {
        'circle': {'kp': 0.8, 'max_vel': 0.9},
        'square': {'kp': 0.7, 'max_vel': 0.8},
        'figure8': {'kp': 0.4, 'max_vel': 0.5},
        'spiral': {'kp': 0.8, 'max_vel': 0.9},
        'hover': {'kp': 0.6, 'max_vel': 0.7},
        'waypoint': {'kp': 0.7, 'max_vel': 0.8}
    }
    
    gains = TUNED_GAINS.get(trajectory_label, {'kp': 1.0, 'max_vel': 1.0})
    
    # Create environment (dummy for PID testing, we'll run PID ourselves)
    # We need the environment just to get the trajectory and physics
    from src.envs.BaseTrackAviary import BaseTrackAviary
    
    env = BaseTrackAviary(
        trajectory_type=trajectory_label,
        drone_model=DroneModel.CF2X,
        physics=Physics.PYB,
        freq=240,
        gui=False,
        act=ActionType.RPM
    )
    
    # Initialize PID controller
    pid = PIDController(drone_model=DroneModel.CF2X, freq=240)
    
    # Original Phase 1 DR parameters (not Phase 2's increased values)
    if domain_randomization:
        import pybullet as p
        original_mass = 0.027
        original_inertia = [1.4e-5, 1.4e-5, 2.17e-5]
    
    episode_rewards = []
    episode_lengths = []
    tracking_errors = []
    control_smoothness = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        
        # Apply randomization after reset
        if domain_randomization:
            # Use ORIGINAL Phase 1 parameters: ±20% mass/inertia, not Phase 2's ±30%
            mass_scale = np.random.uniform(0.8, 1.2)
            new_mass = original_mass * mass_scale
            inertia_scale = np.random.uniform(0.8, 1.2)
            new_inertia = [i * inertia_scale for i in original_inertia]
            
            import pybullet as p
            for i in range(env.NUM_DRONES):
                p.changeDynamics(env.DRONE_IDS[i], -1, mass=new_mass, 
                               localInertiaDiagonal=new_inertia, 
                               physicsClientId=env.CLIENT)
        
        done = False
        episode_reward = 0
        step_count = 0
        errors = []
        actions = []
        
        while not done:
            # Extract target from observation
            drone_obs = obs[0]
            current_pos = drone_obs[0:3]
            current_vel = drone_obs[6:9]
            pos_error = drone_obs[12:15]
            vel_error = drone_obs[15:18]
            
            target_pos = current_pos + pos_error
            target_vel = current_vel + vel_error
            
            # Compute PID action
            action = pid.compute_control(drone_obs, target_pos, target_vel)
            actions.append(action)
            
            # Apply wind if domain randomization
            if domain_randomization and np.random.rand() < 0.1:  # 10% of steps
                # Use ORIGINAL Phase 1 wind: 0.05N, not Phase 2's 0.15N
                wind_force = np.random.uniform(-0.05, 0.05, 3)
                for i in range(env.NUM_DRONES):
                    p.applyExternalForce(env.DRONE_IDS[i], -1, wind_force, 
                                       [0,0,0], p.LINK_FRAME, 
                                       physicsClientId=env.CLIENT)
            
            # Step
            obs, reward, terminated, truncated, info = env.step(np.array([action]))
            done = terminated or truncated
            
            episode_reward += reward
            step_count += 1
            
            # Track error
            pos_error_mag = np.linalg.norm(obs[0][12:15])
            errors.append(pos_error_mag)
        
        # Compute control smoothness (variance of action changes)
        if len(actions) > 1:
            action_changes = np.diff(np.array(actions), axis=0)
            smoothness = np.mean(np.var(action_changes, axis=0))
        else:
            smoothness = 0.0
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        tracking_errors.append(np.mean(errors))
        control_smoothness.append(smoothness)
    
    env.close()
    
    results = {
        'controller': 'PID',
        'trajectory': trajectory_label,
        'n_episodes': n_episodes,
        'domain_randomization': domain_randomization,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'mean_tracking_error': float(np.mean(tracking_errors)),
        'std_tracking_error': float(np.std(tracking_errors)),
        'mean_smoothness': float(np.mean(control_smoothness)),
        'std_smoothness': float(np.std(control_smoothness))
    }
    
    return results


def test_hybrid_model(model_path, vec_normalize_path, trajectory_file, 
                     n_episodes=5, domain_randomization=False):
    """Test Hybrid RL model"""
    
    with open(trajectory_file, 'rb') as f:
        traj = pickle.load(f)
    trajectory_label = traj.get('trajectory_label', 'unknown')
    
    # Load model
    model = PPO.load(model_path)
    
    # Load VecNormalize stats if they exist
    vec_normalize = None
    if vec_normalize_path.exists():
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        # Create dummy env for VecNormalize
        # NOTE: hybrid_robust models were trained with:
        # - residual_scale=200 (not current 100)
        # - DR params: ±20% mass/inertia, 0.05N wind (not current ±30%, 0.15N)
        def make_env():
            env = HybridAviary(
                trajectory_type=trajectory_label,
                drone_model=DroneModel.CF2X,
                physics=Physics.PYB,
                freq=240,
                gui=False,
                domain_randomization=domain_randomization
            )
            # Restore original Phase 1 training config
            env.residual_scale = 200.0
            # Override DR parameters if domain_randomization is enabled
            if domain_randomization:
                # Save original randomization method
                original_randomize = env._randomize_dynamics
                original_apply_wind = env._apply_wind
                
                # Override with Phase 1 parameters
                def phase1_randomize():
                    import pybullet as p
                    mass_scale = np.random.uniform(0.8, 1.2)  # ±20%
                    new_mass = env.original_mass * mass_scale
                    inertia_scale = np.random.uniform(0.8, 1.2)  # ±20%
                    new_inertia = [i * inertia_scale for i in env.original_inertia]
                    for i in range(env.NUM_DRONES):
                        p.changeDynamics(env.DRONE_IDS[i], -1, mass=new_mass,
                                       localInertiaDiagonal=new_inertia,
                                       physicsClientId=env.CLIENT)
                
                def phase1_wind():
                    import pybullet as p
                    wind_force = np.random.uniform(-0.05, 0.05, 3)  # 0.05N max
                    for i in range(env.NUM_DRONES):
                        p.applyExternalForce(env.DRONE_IDS[i], -1, wind_force,
                                           [0,0,0], p.LINK_FRAME,
                                           physicsClientId=env.CLIENT)
                
                env._randomize_dynamics = phase1_randomize
                env._apply_wind = phase1_wind
            return env
        
        dummy_env = DummyVecEnv([make_env])
        vec_normalize = VecNormalize.load(vec_normalize_path, dummy_env)
        vec_normalize.training = False
        vec_normalize.norm_reward = False
        env = vec_normalize
    else:
        # No normalization
        # NOTE: hybrid_robust models were trained with Phase 1 config
        env = HybridAviary(
            trajectory_type=trajectory_label,
            drone_model=DroneModel.CF2X,
            physics=Physics.PYB,
            freq=240,
            gui=False,
            domain_randomization=domain_randomization
        )
        env.residual_scale = 200.0  # Restore original training config
        
        # Override DR parameters if domain_randomization is enabled
        if domain_randomization:
            def phase1_randomize():
                import pybullet as p
                mass_scale = np.random.uniform(0.8, 1.2)  # ±20%
                new_mass = env.original_mass * mass_scale
                inertia_scale = np.random.uniform(0.8, 1.2)  # ±20%
                new_inertia = [i * inertia_scale for i in env.original_inertia]
                for i in range(env.NUM_DRONES):
                    p.changeDynamics(env.DRONE_IDS[i], -1, mass=new_mass,
                                   localInertiaDiagonal=new_inertia,
                                   physicsClientId=env.CLIENT)
            
            def phase1_wind():
                import pybullet as p
                wind_force = np.random.uniform(-0.05, 0.05, 3)  # 0.05N max
                for i in range(env.NUM_DRONES):
                    p.applyExternalForce(env.DRONE_IDS[i], -1, wind_force,
                                       [0,0,0], p.LINK_FRAME,
                                       physicsClientId=env.CLIENT)
            
            env._randomize_dynamics = phase1_randomize
            env._apply_wind = phase1_wind
    
    episode_rewards = []
    episode_lengths = []
    tracking_errors = []
    control_smoothness = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        errors = []
        rl_actions = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            rl_actions.append(action.flatten() if hasattr(action, 'flatten') else action)
            
            obs, reward, done_array, info = env.step(action)
            
            # Handle VecEnv vs regular env
            if isinstance(done_array, np.ndarray):
                done = done_array[0]
                reward_val = reward[0] if isinstance(reward, np.ndarray) else reward
                obs_array = obs[0] if len(obs.shape) > 1 else obs
            else:
                done = done_array
                reward_val = reward
                obs_array = obs
            
            episode_reward += reward_val
            step_count += 1
            
            # Track error - handle both VecEnv and regular env obs shapes
            if len(obs_array.shape) > 1:
                pos_error_mag = np.linalg.norm(obs_array[0][12:15])
            else:
                pos_error_mag = np.linalg.norm(obs_array[12:15])
            errors.append(pos_error_mag)
        
        # Compute control smoothness
        if len(rl_actions) > 1:
            action_changes = np.diff(np.array(rl_actions), axis=0)
            smoothness = np.mean(np.var(action_changes, axis=0))
        else:
            smoothness = 0.0
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        tracking_errors.append(np.mean(errors))
        control_smoothness.append(smoothness)
    
    env.close()
    
    results = {
        'controller': 'Hybrid',
        'trajectory': trajectory_label,
        'model_path': str(model_path),
        'n_episodes': n_episodes,
        'domain_randomization': domain_randomization,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'mean_tracking_error': float(np.mean(tracking_errors)),
        'std_tracking_error': float(np.std(tracking_errors)),
        'mean_smoothness': float(np.mean(control_smoothness)),
        'std_smoothness': float(np.std(control_smoothness))
    }
    
    return results


def main():
    models_dir = Path('models/hybrid_robust')
    traj_dir = Path('data/expert_trajectories')
    output_dir = Path('results/phase1_simulation/perturbation_tests')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("PHASE 1: SIMULATION PERTURBATION TESTING")
    print(f"{'='*70}\n")
    print(f"Models: {models_dir}")
    print(f"Output: {output_dir}\n")
    
    # Find all hybrid_robust models
    trajectories = ['circle', 'figure8', 'hover', 'spiral', 'waypoint']
    
    all_results = []
    
    for traj_type in trajectories:
        model_dir = models_dir / traj_type
        model_file = model_dir / 'final_model.zip'
        vec_normalize_file = model_dir / 'vec_normalize.pkl'
        traj_file = traj_dir / f'perfect_{traj_type}_trajectory.pkl'
        
        if not model_file.exists():
            print(f"⚠️  Skipping {traj_type}: model not found")
            continue
        
        if not traj_file.exists():
            print(f"⚠️  Skipping {traj_type}: trajectory file not found")
            continue
        
        print(f"\n{'='*70}")
        print(f"Testing: {traj_type.upper()}")
        print(f"{'='*70}\n")
        
        # Test 1: PID baseline without perturbation
        print(f"\n[1/4] Testing PID baseline (no perturbation)...")
        pid_baseline = test_pid_baseline(traj_file, n_episodes=5, domain_randomization=False)
        all_results.append(pid_baseline)
        print(f"✓ PID Baseline: Error={pid_baseline['mean_tracking_error']:.4f}m, "
              f"Smoothness={pid_baseline['mean_smoothness']:.2f}")
        
        # Test 2: PID with perturbation
        print(f"\n[2/4] Testing PID with domain randomization...")
        pid_perturb = test_pid_baseline(traj_file, n_episodes=5, domain_randomization=True)
        all_results.append(pid_perturb)
        print(f"✓ PID + DR: Error={pid_perturb['mean_tracking_error']:.4f}m, "
              f"Smoothness={pid_perturb['mean_smoothness']:.2f}")
        
        # Test 3: Hybrid baseline without perturbation
        print(f"\n[3/4] Testing Hybrid baseline (no perturbation)...")
        hybrid_baseline = test_hybrid_model(model_file, vec_normalize_file, traj_file, 
                                           n_episodes=5, domain_randomization=False)
        all_results.append(hybrid_baseline)
        print(f"✓ Hybrid Baseline: Error={hybrid_baseline['mean_tracking_error']:.4f}m, "
              f"Smoothness={hybrid_baseline['mean_smoothness']:.2f}")
        
        # Test 4: Hybrid with perturbation
        print(f"\n[4/4] Testing Hybrid with domain randomization...")
        hybrid_perturb = test_hybrid_model(model_file, vec_normalize_file, traj_file, 
                                          n_episodes=5, domain_randomization=True)
        all_results.append(hybrid_perturb)
        print(f"✓ Hybrid + DR: Error={hybrid_perturb['mean_tracking_error']:.4f}m, "
              f"Smoothness={hybrid_perturb['mean_smoothness']:.2f}")
        
        # Quick comparison
        print(f"\n{'─'*70}")
        print(f"Summary for {traj_type.upper()}:")
        print(f"{'─'*70}")
        print(f"                    Tracking Error    Smoothness")
        print(f"PID Baseline:       {pid_baseline['mean_tracking_error']:.4f}m          {pid_baseline['mean_smoothness']:.2f}")
        print(f"PID + DR:           {pid_perturb['mean_tracking_error']:.4f}m          {pid_perturb['mean_smoothness']:.2f}")
        print(f"Hybrid Baseline:    {hybrid_baseline['mean_tracking_error']:.4f}m          {hybrid_baseline['mean_smoothness']:.2f}")
        print(f"Hybrid + DR:        {hybrid_perturb['mean_tracking_error']:.4f}m          {hybrid_perturb['mean_smoothness']:.2f}")
        
        # Calculate improvements
        baseline_improvement = ((pid_baseline['mean_tracking_error'] - hybrid_baseline['mean_tracking_error']) 
                               / pid_baseline['mean_tracking_error'] * 100)
        dr_improvement = ((pid_perturb['mean_tracking_error'] - hybrid_perturb['mean_tracking_error']) 
                         / pid_perturb['mean_tracking_error'] * 100)
        
        print(f"\nImprovement:")
        print(f"  Baseline: {baseline_improvement:+.1f}%")
        print(f"  With DR:  {dr_improvement:+.1f}%")
        print(f"{'─'*70}\n")
    
    # Save all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f'perturbation_test_results_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'test_description': 'Phase 1 simulation perturbation testing',
            'models_tested': str(models_dir),
            'episodes_per_test': 5,
            'results': all_results
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✓ Results saved to: {output_file}")
    print(f"{'='*70}\n")
    
    # Print overall summary
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}\n")
    
    # Group by trajectory
    trajectories_tested = sorted(set(r['trajectory'] for r in all_results))
    
    for traj in trajectories_tested:
        traj_results = [r for r in all_results if r['trajectory'] == traj]
        pid_base = next(r for r in traj_results if r['controller'] == 'PID' and not r['domain_randomization'])
        pid_dr = next(r for r in traj_results if r['controller'] == 'PID' and r['domain_randomization'])
        hybrid_base = next(r for r in traj_results if r['controller'] == 'Hybrid' and not r['domain_randomization'])
        hybrid_dr = next(r for r in traj_results if r['controller'] == 'Hybrid' and r['domain_randomization'])
        
        base_imp = ((pid_base['mean_tracking_error'] - hybrid_base['mean_tracking_error']) 
                   / pid_base['mean_tracking_error'] * 100)
        dr_imp = ((pid_dr['mean_tracking_error'] - hybrid_dr['mean_tracking_error']) 
                 / pid_dr['mean_tracking_error'] * 100)
        
        print(f"{traj.upper():<10} | Baseline: {base_imp:+6.1f}% | With DR: {dr_imp:+6.1f}%")
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
