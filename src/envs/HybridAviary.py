import numpy as np
import pybullet as p
from gymnasium import spaces
from src.envs.BaseTrackAviary import BaseTrackAviary
from src.controllers.pid_controller import PIDController, VelocityPIDController
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class HybridAviary(BaseTrackAviary):
    """
    Hybrid Environment: PID + Residual RL.
    The RL agent outputs a residual correction to the PID's RPM command.
    """
    def __init__(self, 
                 trajectory_type='hover', 
                 drone_model=DroneModel.CF2X,
                 num_drones=1, 
                 physics=Physics.PYB, 
                 freq=240,
                 gui=False,
                 record=False,
                 obs=ObservationType.KIN,
                 act=ActionType.RPM,
                 domain_randomization=False):
        
        super().__init__(trajectory_type=trajectory_type,
                         drone_model=drone_model,
                         num_drones=num_drones,
                         physics=physics,
                         freq=freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act)
        
        # Initialize internal PID controller
        if self.ACT_TYPE == ActionType.RPM:
            self.pid_controller = PIDController(drone_model=drone_model, freq=freq)
            self.residual_scale = 200.0 # RPM (Smaller corrections for stability)
        elif self.ACT_TYPE == ActionType.VEL:
            self.pid_controller = VelocityPIDController(kp=1.0, max_vel=1.0)
            self.residual_scale = 0.5 # m/s
        
        # Domain Randomization
        self.domain_randomization = domain_randomization
        self.original_mass = 0.027 # CF2X default
        self.original_inertia = [1.4e-5, 1.4e-5, 2.17e-5] # CF2X default
        
        # Track last action for regularization
        self.last_rl_action = None

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self.domain_randomization:
            self._randomize_dynamics()
        self.last_rl_action = np.zeros(4)
        return obs, info

    def _randomize_dynamics(self):
        """
        Randomize mass and inertia.
        """
        # Mass +/- 30% (Increased from 20%)
        mass_scale = np.random.uniform(0.7, 1.3)
        new_mass = self.original_mass * mass_scale
        
        # Inertia +/- 30% (Increased from 20%)
        inertia_scale = np.random.uniform(0.7, 1.3)
        new_inertia = [i * inertia_scale for i in self.original_inertia]
        
        for i in range(self.NUM_DRONES):
            p.changeDynamics(self.DRONE_IDS[i], -1, mass=new_mass, localInertiaDiagonal=new_inertia, physicsClientId=self.CLIENT)

    def step(self, action):
        """
        Override step to combine PID and RL actions.
        """
        if self.domain_randomization:
            self._apply_wind()
            
        # 1. Get current state for PID
        # We need to construct the observation vector expected by PIDController
        # BaseTrackAviary._computeObs returns the full observation (State + Target)
        # We can just use that.
        
        # But _computeObs is called at the end of step().
        # We need the *current* observation before the step.
        # We can call _computeObs() again, or use the state directly.
        # Calling _computeObs() is safer to ensure consistency.
        
        full_obs = self._computeObs()
        
        combined_action_list = []
        
        for i in range(self.NUM_DRONES):
            # Extract single drone obs
            # full_obs is (NUM_DRONES, 21)
            # Obs structure: [x,y,z, r,p,y, vx,vy,vz, wx,wy,wz, pos_err, vel_err, disturbance_est]
            drone_obs = full_obs[i]
            
            # Reconstruct target from error (first 18 elements same as before)
            current_pos = drone_obs[0:3]
            current_vel = drone_obs[6:9]
            pos_error = drone_obs[12:15]
            vel_error = drone_obs[15:18]
            # disturbance_est = drone_obs[18:21]  # Available for RL agent
            
            target_pos = current_pos + pos_error
            target_vel = current_vel + vel_error
            
            if self.ACT_TYPE == ActionType.RPM:
                # Compute PID action (Raw RPM)
                base_action = self.pid_controller.compute_control(drone_obs, target_pos, target_vel)
            elif self.ACT_TYPE == ActionType.VEL:
                # Compute PID action (Target Velocity)
                base_action = self.pid_controller.compute_control(drone_obs, target_pos)
            
            # Compute Residual
            # action is (NUM_DRONES, 4)
            residual = action[i] * self.residual_scale
            
            # Track RL action for reward computation
            if i == 0:  # Only for first drone
                self.last_rl_action = action[i].copy()
            
            # Combine
            combined_action = base_action + residual
            combined_action_list.append(combined_action)
            
        combined_action_array = np.array(combined_action_list)
        
        # Pass combined action to parent step
        return super().step(combined_action_array)

    def _apply_wind(self):
        """
        Apply random external force (wind).
        """
        # Random wind force up to 0.15 N (approx 15g force) - Increased from 0.05
        # We can vary this per step or per episode. 
        # Let's vary per step for turbulence effect.
        wind_force = np.random.uniform(-0.15, 0.15, 3)
        for i in range(self.NUM_DRONES):
            p.applyExternalForce(self.DRONE_IDS[i], -1, wind_force, [0,0,0], p.LINK_FRAME, physicsClientId=self.CLIENT)
