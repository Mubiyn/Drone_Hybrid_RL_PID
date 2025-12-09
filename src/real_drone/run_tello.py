import time
import numpy as np
import argparse
from src.real_drone.TelloWrapper import TelloWrapper
from src.controllers.pid_controller import VelocityPIDController
from src.utils.trajectories import TrajectoryGenerator
from stable_baselines3 import PPO

def run_tello_trajectory(controller_type='pid', model_path=None, trajectory_type='hover', 
                         duration=10.0, use_mocap=False):
    """
    Run Tello drone with PID or Hybrid RL controller.
    
    Args:
        controller_type: 'pid' or 'hybrid'
        model_path: Path to trained PPO model (for hybrid)
        trajectory_type: 'hover', 'circle', etc.
        duration: Flight duration in seconds
        use_mocap: Whether to use motion capture system
    """
    print("=" * 60)
    print(f"Tello Real Drone Test")
    print(f"Controller: {controller_type.upper()}")
    print(f"Trajectory: {trajectory_type}")
    print(f"Duration: {duration}s")
    print("=" * 60)
    
    # Initialize MoCap if requested
    mocap_client = None
    if use_mocap:
        try:
            from src.real_drone.mocap_client import NatNetClient
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
    
    # Load Hybrid RL model if needed
    rl_model = None
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
    
    # Initialize trajectory
    # Smaller, slower trajectory for real drone
    traj = TrajectoryGenerator(trajectory_type=trajectory_type, 
                               radius=0.5,  # Reduced from 0.8m for safety
                               height=1.0,
                               duration=15.0)  # Slower trajectory
    
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
            target_pos, target_vel, _ = traj.get_target(t)
            
            # Compute position error
            pos_error = np.linalg.norm(current_pos - target_pos)
            errors.append(pos_error)
            
            # === CONTROL COMPUTATION ===
            if controller_type == 'pid':
                # Pure PID control
                action_vel = pid.compute_control(obs, target_pos)
                
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
                       choices=['pid', 'hybrid'],
                       help='Controller type')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model (for hybrid)')
    parser.add_argument('--traj', type=str, default='hover',
                       help='Trajectory type')
    parser.add_argument('--duration', type=float, default=10.0,
                       help='Flight duration in seconds')
    parser.add_argument('--mocap', action='store_true',
                       help='Use motion capture system')
    
    args = parser.parse_args()
    
    # Auto-select model path if hybrid controller
    if args.controller == 'hybrid' and args.model is None:
        args.model = f"models/hybrid_robust/{args.traj}/final_model.zip"
        print(f"Using default model: {args.model}")
    
    run_tello_trajectory(
        controller_type=args.controller,
        model_path=args.model,
        trajectory_type=args.traj,
        duration=args.duration,
        use_mocap=args.mocap
    )
