# hover_sim_wrapper.py
import numpy as np
import time
import pybullet as p
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from scipy.spatial.transform import Rotation as R

class HoverSimWrapper:
    def __init__(self, gui=False):
        self.gui = gui
        self.env = HoverAviary(gui=gui, obs=ObservationType.KIN, act=ActionType.RPM, record=False)
        self.client = self.env.CLIENT
        self.drone_id = 0
        self.drone_uid = self.env.DRONE_IDS[self.drone_id]
        dyn_info = p.getDynamicsInfo(self.drone_uid, -1, physicsClientId=self.client)
        self.mass = dyn_info[0]
        self.CTRL_FREQ = self.env.CTRL_FREQ
        self.physics_dt = 1.0 / self.CTRL_FREQ

    def get_state(self):
        pos, quat = p.getBasePositionAndOrientation(self.drone_uid, physicsClientId=self.client)
        lin_vel, ang_vel = p.getBaseVelocity(self.drone_uid, physicsClientId=self.client)
        return np.array(pos), np.array(quat), np.array(lin_vel), np.array(ang_vel)

    def apply_external_force_and_torque(self, force_world, torque_body):
        pos, quat = p.getBasePositionAndOrientation(self.drone_uid, physicsClientId=self.client)
        rot = R.from_quat(quat).as_matrix()
        torque_world = rot @ np.array(torque_body, dtype=np.float32)

        p.applyExternalForce(self.drone_uid, -1, force_world.tolist(), [0, 0, 0], flags=p.WORLD_FRAME, physicsClientId=self.client)
        p.applyExternalTorque(self.drone_uid, -1, torque_world.tolist(), flags=p.WORLD_FRAME, physicsClientId=self.client)

    def step_simulation(self, dt):
        steps_needed = max(1, int(dt / self.physics_dt))
        for _ in range(steps_needed):
            p.stepSimulation(physicsClientId=self.client)
            if self.gui:
                time.sleep(self.physics_dt)

    def reset_to(self, pos, quat_xyzw, seed=None):
        self.env.reset(seed=seed)
        p.resetBasePositionAndOrientation(self.drone_uid, posObj=pos.tolist(), ornObj=quat_xyzw.tolist(), physicsClientId=self.client)
        p.resetBaseVelocity(self.drone_uid, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)

    def close(self):
        self.env.close()
