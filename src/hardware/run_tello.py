import time
import numpy as np
import argparse
import pickle
import torch
import torch.nn as nn
from pathlib import Path
from src.hardware.TelloWrapper import TelloWrapper
from src.controllers.pid_controller import VelocityPIDController
from src.utils.trajectories import TrajectoryGenerator
from stable_baselines3 import PPO


class BehaviorCloningPolicy(nn.Module):
    """Neural network policy for imitating manual flight (matches training script)"""
    
    def __init__(self, state_dim=12, action_dim=4, hidden_sizes=[256, 256]):
        super().__init__()
        
        layers = []
        in_dim = state_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_size))
            in_dim = hidden_size
        
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
        
        self.state_mean = None
        self.state_std = None
    
    def forward(self, state):
        if self.state_mean is not None:
            state = (state - self.state_mean) / (self.state_std + 1e-8)
        return self.network(state)
    
    def set_normalization(self, mean, std):
        """Set state normalization parameters"""
        self.state_mean = torch.FloatTensor(mean)
        self.state_std = torch.FloatTensor(std)


def run_tello_trajectory(controller_type='pid', model_path=None, trajectory_type='hover', 
                         trajectory_file=None, duration=10.0, use_mocap=False):
    """
    Run Tello drone with PID or Hybrid RL controller.
    
    Args:
        controller_type: 'pid' or 'hybrid' or 'manual_bc' (behavioral cloning)
        model_path: Path to trained model (PPO for hybrid, .pth for manual_bc)
        trajectory_type: 'hover', 'circle', etc. (ignored if trajectory_file provided)
        trajectory_file: Path to learned trajectory .pkl file (from train_pid_from_manual.py)
        duration: Flight duration in seconds
        use_mocap: Whether to use motion capture system
    """
    print("=" * 60)
    print(f"Tello Real Drone Test")
    print(f"Controller: {controller_type.upper()}")
    if trajectory_file:
        print(f"Trajectory: {trajectory_file}")
    else:
        print(f"Trajectory: {trajectory_type}")
    print(f"Duration: {duration}s")
    print("=" * 60)
    
    # Initialize MoCap if requested
    mocap_client = None
    if use_mocap:
        try:
            from src.hardware.mocap_client import NatNetClient
            mocap_client = NatNetClient()
            mocap_client.start()
            print("Motion Capture system connected")
            time.sleep(1)
        except Exception as e:
            print(f"Failed to connect to MoCap: {e}")
            print("Continuing without MoCap...")
            mocap_client = None
    
    # Initialize Tello
    tello = TelloWrapper(mocap_client=mocap_client)
    
    # Initialize Controllers
    # Conservative PID gains for real Tello (much gentler than simulation)
    pid = VelocityPIDController(kp=0.4, max_vel=0.5)  # Reduced from 0.8
    
    # Load model if needed
    rl_model = None
    bc_model = None
    
    if controller_type == 'hybrid':
        if model_path is None:
            print("ERROR: Hybrid controller requires model_path")
            return
        try:
            rl_model = PPO.load(model_path)
            print(f"Loaded RL model from {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            return
    
    elif controller_type == 'manual_bc':
        if model_path is None:
            print("ERROR: Manual BC controller requires model_path")
            return
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dim = checkpoint['state_dim']
            action_dim = checkpoint['action_dim']
            hidden_sizes = checkpoint['hidden_sizes']
            
            bc_model = BehaviorCloningPolicy(state_dim, action_dim, hidden_sizes)
            bc_model.load_state_dict(checkpoint['model_state_dict'])
            bc_model.set_normalization(checkpoint['state_mean'], checkpoint['state_std'])
            bc_model.eval()
            print(f"Loaded BC model from {model_path}")
            print(f"  State dim: {state_dim}, Action dim: {action_dim}")
        except Exception as e:
            print(f"Failed to load BC model: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Initialize trajectory
    if trajectory_file:
        try:
            with open(trajectory_file, 'rb') as f:
                learned_traj = pickle.load(f)
            waypoints = learned_traj['waypoints']
            waypoint_times = learned_traj['waypoint_times']
            print(f"Loaded learned trajectory: {learned_traj['trajectory_label']}")
            print(f"  Waypoints: {len(waypoints)}")
            print(f"  Duration: {learned_traj['duration']:.1f}s")
            traj = None  # Will interpolate waypoints manually
        except Exception as e:
            print(f"Failed to load trajectory file: {e}")
            return
    else:
        # Use generated trajectory
        traj = TrajectoryGenerator(trajectory_type=trajectory_type, 
                                   radius=0.5,  # Reduced from 0.8m for safety
                                   height=1.0,
                                   duration=duration)  # Match flight duration
        waypoints = None
        waypoint_times = None
    
    # Safety check
    print(f"\nBattery: {tello.get_battery()}%")
    print("Ready to takeoff. Press Enter to start (Ctrl+C to abort)...")
    try:
        input()
    except KeyboardInterrupt:
        print("Aborted")
        return
    
    print("\nTaking off...")
    tello.takeoff()
    time.sleep(3)  # Stabilize
    
    # Set initial position if using MoCap
    if mocap_client:
        tello.pos = mocap_client.get_position()
        print(f"Initial position: {tello.pos}")
    
    start_time = time.time()
    last_print = start_time
    control_freq = 20  # Hz
    dt = 1.0 / control_freq
    
    errors = []
    
    try:
        while time.time() - start_time < duration:
            loop_start = time.time()
            t = time.time() - start_time
            
            # Get current observation (12-dim state)
            obs = tello.get_obs()
            current_pos = obs[0:3]
            
            # Get target from trajectory
            if waypoints is not None:
                # Interpolate learned trajectory waypoints
                t_normalized = t / waypoint_times[-1]  # Normalize to [0, 1]
                t_traj = t_normalized * waypoint_times[-1]
                
                idx = np.searchsorted(waypoint_times, t_traj)
                if idx == 0:
                    target_pos = waypoints[0]
                elif idx >= len(waypoints):
                    target_pos = waypoints[-1]
                else:
                    alpha = (t_traj - waypoint_times[idx-1]) / (waypoint_times[idx] - waypoint_times[idx-1])
                    target_pos = (1 - alpha) * waypoints[idx-1] + alpha * waypoints[idx]
                target_vel = np.zeros(3)
            else:
                target_pos, target_vel, _ = traj.get_target(t)
            
            # Compute position error
            pos_error = np.linalg.norm(current_pos - target_pos)
            errors.append(pos_error)
            
            # === CONTROL COMPUTATION ===
            if controller_type == 'pid':
                # Pure PID control
                action_vel = pid.compute_control(obs, target_pos)
            
            elif controller_type == 'manual_bc':
                # Behavioral cloning: Direct imitation of manual actions
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)  # Add batch dim
                    action_tensor = bc_model(obs_tensor)
                    action_vel = action_tensor.squeeze(0).numpy()  # [lr, fb, ud, yaw]
                
            elif controller_type == 'hybrid':
                # Hybrid: PID + RL residual
                # Note: Trained model outputs RPM residuals for simulation
                # For real drone, we interpret this as velocity corrections
                
                # Build full observation (18-dim: state + pos_error + vel_error)
                current_vel = obs[6:9]
                pos_err_vec = target_pos - current_pos
                vel_err_vec = target_vel - current_vel
                
                full_obs = np.concatenate([
                    obs,           # 12-dim state
                    pos_err_vec,   # 3-dim position error
                    vel_err_vec    # 3-dim velocity error
                ])  # Total: 18 dims
                
                # Reshape for model (expects batch)
                full_obs_batch = full_obs.reshape(1, -1)
                
                # Get RL action (4-dim output, trained for RPM residuals)
                rl_action, _ = rl_model.predict(full_obs_batch, deterministic=True)
                rl_action = rl_action[0]  # Remove batch dim
                
                # PID baseline velocity command
                pid_vel = pid.compute_control(obs, target_pos)
                
                # Convert RL action to velocity residual
                # Trained model outputs normalized actions [-1, 1]
                # Scale conservatively for velocity corrections
                residual_scale = 0.1  # Reduced from 0.2 - very conservative for first flights
                
                # RL action has 4 dims [vx_res, vy_res, vz_res, yaw_res]
                # Add only xyz residuals to PID velocity, keep PID's yaw
                action_vel = pid_vel.copy()
                action_vel[0:3] += rl_action[0:3] * residual_scale
            
            # Send command
            tello.step(action_vel)
            
            # Print status every second
            if time.time() - last_print > 1.0:
                print(f"t={t:.1f}s | Pos: [{current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}] | "
                      f"Target: [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}] | "
                      f"Error: {pos_error:.3f}m | Battery: {tello.get_battery()}%")
                last_print = time.time()
            
            # Maintain control frequency
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
                
    except KeyboardInterrupt:
        print("\nStopping by user...")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Land safely
        print("\nLanding...")
        tello.land()
        time.sleep(2)
        
        # Cleanup
        if mocap_client:
            mocap_client.stop()
        tello.close()
        
        # Print statistics
        if errors:
            print("\n" + "=" * 60)
            print("Flight Statistics:")
            print(f"  Mean Error: {np.mean(errors):.3f}m")
            print(f"  Max Error: {np.max(errors):.3f}m")
            print(f"  RMSE: {np.sqrt(np.mean(np.array(errors)**2)):.3f}m")
            print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tello Real Drone Controller")
    parser.add_argument('--controller', type=str, default='pid', 
                       choices=['pid', 'hybrid', 'manual_bc'],
                       help='Controller type: pid (PID only), hybrid (PID+RL), manual_bc (behavioral cloning)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model (PPO .zip for hybrid, .pth for manual_bc)')
    parser.add_argument('--traj', type=str, default='hover',
                       help='Trajectory type (hover, circle, figure8, etc.)')
    parser.add_argument('--trajectory-file', type=str, default=None,
                       help='Path to learned trajectory .pkl file (from train_pid_from_manual.py)')
    parser.add_argument('--duration', type=float, default=15.0,
                        help='Flight duration in seconds (15s = 1 full circle/figure8)')
    parser.add_argument('--mocap', action='store_true',
                       help='Use motion capture system')
    
    args = parser.parse_args()
    
    # Auto-select model path if needed
    if args.controller == 'hybrid' and args.model is None:
        args.model = f"models/hybrid_robust/{args.traj}/final_model.zip"
        print(f"Using default hybrid model: {args.model}")
    elif args.controller == 'manual_bc' and args.model is None:
        args.model = "models/manual_bc/best_model.pth"
        print(f"Using default BC model: {args.model}")
    
    run_tello_trajectory(
        controller_type=args.controller,
        model_path=args.model,
        trajectory_type=args.traj,
        trajectory_file=args.trajectory_file,
        duration=args.duration,
        use_mocap=args.mocap
    )
