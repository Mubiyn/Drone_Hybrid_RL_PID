import numpy as np
import pybullet as p
from src.envs.BaseTrackAviary import BaseTrackAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

class RobustTrackAviary(BaseTrackAviary):
    """
    BaseTrackAviary with Domain Randomization enabled.
    Adds wind and mass/inertia randomization to test robustness.
    """
    def __init__(self, 
                 trajectory_type='hover', 
                 drone_model=DroneModel.CF2X,
                 num_drones=1, 
                 physics=Physics.PYB, 
                 freq=240,
                 gui=False,
                 record=False,
                 domain_randomization=True):
        
        super().__init__(trajectory_type=trajectory_type,
                         drone_model=drone_model,
                         num_drones=num_drones,
                         physics=physics,
                         freq=freq,
                         gui=gui,
                         record=record)
        
        self.domain_randomization = domain_randomization
        self.original_mass = 0.027
        self.original_inertia = [1.4e-5, 1.4e-5, 2.17e-5]

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self.domain_randomization:
            self._randomize_dynamics()
            # Initialize constant wind bias for this episode (Steady Wind)
            # Using 0.15N as determined by stress test to be breaking point
            self.wind_bias = np.random.uniform(-0.15, 0.15, 3)
        else:
            self.wind_bias = np.zeros(3)
        return obs, info

    def _randomize_dynamics(self):
        import numpy as np
        import pybullet as p
        
        # Mass +/- 30% (Increased from 20%)
        mass_scale = np.random.uniform(0.7, 1.3)
        new_mass = self.original_mass * mass_scale
        
        # Inertia +/- 30% (Increased from 20%)
        inertia_scale = np.random.uniform(0.7, 1.3)
        new_inertia = [i * inertia_scale for i in self.original_inertia]
        
        for i in range(self.NUM_DRONES):
            p.changeDynamics(self.DRONE_IDS[i], -1, mass=new_mass, localInertiaDiagonal=new_inertia, physicsClientId=self.CLIENT)

    def step(self, action):
        if self.domain_randomization:
            self._apply_wind()
        return super().step(action)

    def _apply_wind(self):
        import numpy as np
        import pybullet as p
        # Apply constant wind bias + random turbulence
        turbulence = np.random.uniform(-0.05, 0.05, 3)
        total_wind = self.wind_bias + turbulence
        
        for i in range(self.NUM_DRONES):
            p.applyExternalForce(self.DRONE_IDS[i], -1, total_wind, [0,0,0], p.LINK_FRAME, physicsClientId=self.CLIENT)


