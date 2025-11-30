import numpy as np
import time
import pybullet as p
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from scipy.spatial.transform import Rotation as R


class HoverSimWrapper:
    def __init__(self, gui: bool = False):
        self.gui = gui

        # Create environment WITHOUT freq argument
        self.env = HoverAviary(
            gui=gui,
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            record=False
        )

        self.client = self.env.CLIENT
        self.drone_id = 0
        self.drone_uid = self.env.DRONE_IDS[self.drone_id]

        # -------------------------------
        # ✔ MANUALLY OVERRIDE FREQUENCY
        # -------------------------------
        TARGET_FREQ = 60      # Reduce CPU load
        self.env.CTRL_FREQ = TARGET_FREQ
        self.CTRL_FREQ = TARGET_FREQ
        self.physics_dt = 1.0 / TARGET_FREQ

        # ------------------------------------
        # LOAD DRONE PHYSICAL INFORMATION
        # ------------------------------------
        dyn_info = p.getDynamicsInfo(self.drone_uid, -1, physicsClientId=self.client)
        self.mass = dyn_info[0]

        # Some versions expose MAX_RPM/HOVER_RPM; if not, fallback values
        self.MAX_RPM = float(getattr(self.env, "MAX_RPM", 20000.0))
        self.HOVER_RPM = float(getattr(self.env, "HOVER_RPM", 15000.0))

        # Range around hover
        self.RPM_RANGE = self.MAX_RPM - self.HOVER_RPM

    # ----------------------- STATE ----------------------- #
    def get_state(self):
        pos, quat = p.getBasePositionAndOrientation(self.drone_uid, physicsClientId=self.client)
        lin_vel, ang_vel = p.getBaseVelocity(self.drone_uid, physicsClientId=self.client)
        return np.array(pos), np.array(quat), np.array(lin_vel), np.array(ang_vel)

    # ----------------------- ACTION ---------------------- #
    def step_with_rpm_action(self, action_norm: np.ndarray):
        """Normalize [-1,1] actions to actual RPM."""
        action_norm = np.asarray(action_norm, dtype=np.float32)
        action_norm = np.clip(action_norm, -1.0, 1.0)

        rpm = self.HOVER_RPM + action_norm * self.RPM_RANGE
        rpm = np.clip(rpm, 0.0, self.MAX_RPM)

        rpm = rpm.reshape(1, 4).astype(np.float32)

        # HoverAviary expects shape (NUM_DRONES, 4)
        self.env.step(rpm)

        if self.gui:
            time.sleep(self.physics_dt)

    # ----------------------- RESET ----------------------- #
    def reset_to(self, pos, quat_xyzw, seed=None):
        self.env.reset(seed=seed)

        p.resetBasePositionAndOrientation(
            self.drone_uid,
            posObj=pos.tolist(),
            ornObj=quat_xyzw.tolist(),
            physicsClientId=self.client,
        )

        p.resetBaseVelocity(
            self.drone_uid,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            physicsClientId=self.client,
        )

    def close(self):
        self.env.close()
