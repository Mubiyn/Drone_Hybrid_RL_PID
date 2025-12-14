import time
import argparse
import math
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

plt.ion()  # interactive plotting


class Live3DPlot:
    def __init__(self, ref_points=None):
        self.fig = plt.figure(figsize=(12, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.x_actual, self.y_actual, self.z_actual = [], [], []

        # Optional reference path (e.g. line / waypoints / circle)
        if ref_points is not None and len(ref_points) > 1:
            ref_points = np.array(ref_points)
            self.ax.plot(ref_points[:, 0], ref_points[:, 1], ref_points[:, 2],
                         'r--', linewidth=2, alpha=0.7, label='Reference Path')

        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_zlabel('Z [m]')
        self.ax.set_title('Live 3D Drone Trajectory Tracking')

        # Some reasonable default limits
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([0, 2])

        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

        self.actual_line, = self.ax.plot([], [], [], 'b-', linewidth=2, label='Actual Path')
        self.actual_point, = self.ax.plot([], [], [], 'bo', markersize=8, label='Current Position')

        plt.tight_layout()

    def update_plot(self, actual_pos, target_pos):
        self.x_actual.append(actual_pos[0])
        self.y_actual.append(actual_pos[1])
        self.z_actual.append(actual_pos[2])

        self.actual_line.set_data(self.x_actual, self.y_actual)
        self.actual_line.set_3d_properties(self.z_actual)

        self.actual_point.set_data([actual_pos[0]], [actual_pos[1]])
        self.actual_point.set_3d_properties([actual_pos[2]])

        error = np.linalg.norm(actual_pos - target_pos)
        self.ax.set_title(
            f'Live 3D Tracking | Pos: ({actual_pos[0]:.2f}, {actual_pos[1]:.2f}, {actual_pos[2]:.2f}) | '
            f'Error: {error:.3f}m'
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        plt.close(self.fig)


def tune_pid_for_generic_motion(pid_controller):
    """
    Generic PID tuning for position tracking (not only circular).
    You can tweak these as needed.
    """
    # Position gains [x, y, z]
    pid_controller.P_COEFF_FOR = np.array([0.25, 0.25, 1.0])
    pid_controller.I_COEFF_FOR = np.array([0.008, 0.008, 0.008])
    pid_controller.D_COEFF_FOR = np.array([0.15, 0.15, 0.4])

    # Attitude gains [roll, pitch, yaw]
    pid_controller.P_COEFF_TOR = np.array([50000., 50000., 4000.])
    pid_controller.I_COEFF_TOR = np.array([0.0, 0.0, 80.])
    pid_controller.D_COEFF_TOR = np.array([15000., 15000., 8000.])

    print("✓ Generic PID position P:", pid_controller.P_COEFF_FOR)
    print("✓ Generic PID position I:", pid_controller.I_COEFF_FOR)
    print("✓ Generic PID position D:", pid_controller.D_COEFF_FOR)
    print("✓ Generic PID attitude P:", pid_controller.P_COEFF_TOR)
    print("✓ Generic PID attitude I:", pid_controller.I_COEFF_TOR)
    print("✓ Generic PID attitude D:", pid_controller.D_COEFF_TOR)


def trajectory_circle(t, center, radius, height, period):
    theta = (t / period) * 2.0 * math.pi
    pos = np.array([
        center[0] + radius * math.cos(theta),
        center[1] + radius * math.sin(theta),
        height
    ])

    # Yaw along velocity direction
    omega = 2 * math.pi / period
    vel_x = -radius * omega * math.sin(theta)
    vel_y = radius * omega * math.cos(theta)
    yaw = math.atan2(vel_y, vel_x) if (abs(vel_x) > 1e-3 or abs(vel_y) > 1e-3) else 0.0
    rpy = np.array([0.0, 0.0, yaw])
    return pos, rpy


def trajectory_line(t, start_pos, end_pos, total_time, takeoff_time=2.0, hover_end=True):
    """
    Take off vertically at start_pos, then move in a straight line to end_pos.
    - t < takeoff_time: go from z=0.1 to start_pos[2]
    - takeoff_time <= t <= takeoff_time + total_time: move along line
    - > that: hover at end_pos
    """
    start_pos = np.array(start_pos, dtype=float)
    end_pos = np.array(end_pos, dtype=float)

    # Phase 1: vertical takeoff
    if t < takeoff_time:
        alpha = t / takeoff_time
        pos = start_pos.copy()
        pos[2] = 0.1 + alpha * (start_pos[2] - 0.1)
        yaw = 0.0
        return pos, np.array([0.0, 0.0, yaw])

    # Phase 2: line motion
    tau = t - takeoff_time
    if tau < total_time:
        alpha = tau / total_time  # 0 -> 1
        pos = (1 - alpha) * start_pos + alpha * end_pos
        vel = (end_pos - start_pos) / total_time
        yaw = math.atan2(vel[1], vel[0]) if (abs(vel[0]) > 1e-3 or abs(vel[1]) > 1e-3) else 0.0
        return pos, np.array([0.0, 0.0, yaw])

    # Phase 3: hover at end
    pos = end_pos.copy()
    yaw = 0.0
    if hover_end:
        return pos, np.array([0.0, 0.0, yaw])
    else:
        # If you ever want, you can make it return to start or continue somewhere else
        return pos, np.array([0.0, 0.0, yaw])


def trajectory_waypoints(t, waypoints, segment_time=4.0, takeoff_height=0.5):
    """
    Move through a list of waypoints in 3D, one segment per 'segment_time'.
    - Smooth linear interpolation between waypoints.
    - Simple vertical takeoff before starting path.
    """
    waypoints = np.array(waypoints, dtype=float)
    num_pts = waypoints.shape[0]

    # Takeoff to first waypoint height
    if t < 2.0:
        alpha = t / 2.0
        pos = np.array([waypoints[0, 0], waypoints[0, 1], 0.1 + alpha * (takeoff_height - 0.1)])
        yaw = 0.0
        return pos, np.array([0.0, 0.0, yaw])

    t_path = t - 2.0
    total_path_time = segment_time * (num_pts - 1)

    if t_path >= total_path_time:
        # Hover at last waypoint
        pos = waypoints[-1]
        yaw = 0.0
        return pos, np.array([0.0, 0.0, yaw])

    seg_idx = int(t_path // segment_time)
    alpha = (t_path - seg_idx * segment_time) / segment_time

    p0 = waypoints[seg_idx]
    p1 = waypoints[seg_idx + 1]

    pos = (1 - alpha) * p0 + alpha * p1
    vel = (p1 - p0) / segment_time
    yaw = math.atan2(vel[1], vel[0]) if (abs(vel[0]) > 1e-3 or abs(vel[1]) > 1e-3) else 0.0

    return pos, np.array([0.0, 0.0, yaw])


def get_target(t, traj_type, params):
    """
    Unified interface for getting target_pos, target_rpy.
    traj_type: 'circle', 'line', 'waypoints'
    params: dict of parameters for each type.
    """
    if traj_type == "circle":
        return trajectory_circle(
            t,
            params["center"],
            params["radius"],
            params["height"],
            params["period"]
        )
    elif traj_type == "line":
        return trajectory_line(
            t,
            params["start_pos"],
            params["end_pos"],
            params["line_time"],
            takeoff_time=params.get("takeoff_time", 2.0),
            hover_end=True
        )
    elif traj_type == "waypoints":
        return trajectory_waypoints(
            t,
            params["waypoints"],
            segment_time=params.get("segment_time", 4.0),
            takeoff_height=params.get("takeoff_height", 0.5)
        )
    else:
        # Fallback: hover at origin at given height
        pos = np.array([0.0, 0.0, params.get("height", 0.5)])
        rpy = np.array([0.0, 0.0, 0.0])
        return pos, rpy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generic PID trajectory tracking (circle / line / waypoints)")
    parser.add_argument('--physics', default="pyb", type=Physics, choices=Physics)
    parser.add_argument('--gui', default=True, type=str2bool)
    parser.add_argument('--record_video', default=False, type=str2bool)
    parser.add_argument('--plot', default=True, type=str2bool)
    parser.add_argument('--live_plot', default=True, type=str2bool)
    parser.add_argument('--simulation_freq_hz', default=240, type=int)
    parser.add_argument('--control_freq_hz', default=48, type=int)
    parser.add_argument('--duration_sec', default=25, type=int)

    # trajectory selection
    parser.add_argument('--traj_type', default="line", type=str,
                        choices=["circle", "line", "waypoints"],
                        help="Trajectory type: circle | line | waypoints")

    # circle params (if you still want it)
    parser.add_argument('--radius', default=0.5, type=float)
    parser.add_argument('--period', default=15.0, type=float)

    ARGS = parser.parse_args()

    INIT_XYZS = np.array([[0.0, 0.0, 0.1]])
    INIT_RPYS = np.array([[0.0, 0.0, 0.0]])
    NUM_DRONES = 1
    AGGR_PHY_STEPS = 1

    # Define trajectory parameters
    if ARGS.traj_type == "circle":
        traj_params = {
            "center": (0.0, -0.3),
            "radius": ARGS.radius,
            "height": 0.5,
            "period": ARGS.period
        }
        ref_points = [
            [traj_params["center"][0] + traj_params["radius"] * math.cos(th),
             traj_params["center"][1] + traj_params["radius"] * math.sin(th),
             traj_params["height"]]
            for th in np.linspace(0, 2 * math.pi, 100)
        ]

    elif ARGS.traj_type == "line":
        start_pos = np.array([0.0, 0.0, 0.7])
        end_pos = np.array([1.0, 1.0, 1.0])  # change this to any 3D point
        traj_params = {
            "start_pos": start_pos,
            "end_pos": end_pos,
            "line_time": 10.0,
            "takeoff_time": 2.0
        }
        ref_points = [start_pos, end_pos]

    elif ARGS.traj_type == "waypoints":

        # ============================
        # FULL MISSION WAYPOINTS
        # Takeoff → Straight → Square → ZigZag → Stairs → Spiral → Return → Land
        # ============================



        waypoints = [
            # --- TAKEOFF ---
            [0.0, 0.0, 0.7],  

            # --- STRAIGHT LINE ---
            [1.0, 0.0, 0.7],

            # --- SQUARE PATH ---
            [1.0, 1.0, 0.7],
            [0.0, 1.0, 0.7],
            [0.0, 0.0, 0.7],

            # # --- ZIG-ZAG ---
            # [0.5, 0.2, 0.8],
            # [1.0, -0.2, 0.9],
            # [1.5, 0.2, 1.0],
            # [2.0, -0.2, 1.1],

            # --- STAIRS (Step Up Path) ---
            [2.5, -0.2, 1.3],
            [3.0, -0.2, 1.5],
            [3.5, -0.2, 1.7],
            
            [0, 2.0, 1.9],

            # # --- SPIRAL ---
            # [3.3, 0.0, 1.9],
            # [3.0, 0.3, 2.1],
            # [2.7, 0.0, 2.3],
            # [3.0, -0.3, 2.5],
            # [3.3, 0.0, 2.7],

            # --- RETURN HOME ---
            [0.0, 0.0, 0.7],

            # --- LAND ---
            [0.0, 0.0, 0.3],
            [0.0, 0.0, 0.1]
        ]

        # Trajectory parameters
        traj_params = {
            "waypoints": waypoints,
            "segment_time": 4.0,      # how long drone spends on each segment
            "takeoff_height": 0.7     # height to rise before starting mission
        }

        num_segments = len(waypoints) - 1
        mission_time = 2.0 + num_segments * traj_params["segment_time"] + 3.0  # takeoff + segments + landing

        ARGS.duration_sec = mission_time
        print(f"Auto duration: {mission_time:.1f} sec (based on {num_segments} segments)")

        ref_points = waypoints


    print("\n" + "=" * 60)
    print("GENERIC TRAJECTORY TRACKING")
    print("=" * 60)
    print(f"Trajectory type: {ARGS.traj_type}")
    print(f"Duration: {ARGS.duration_sec} s")

    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=NUM_DRONES,
        initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
        physics=ARGS.physics,
        neighbourhood_radius=10,
        freq=ARGS.simulation_freq_hz,
        aggregate_phy_steps=AGGR_PHY_STEPS,
        gui=ARGS.gui,
        record=ARGS.record_video,
        obstacles=False,
        user_debug_gui=False
    )

    PYB_CLIENT = env.getPyBulletClient()

    logger = Logger(
        logging_freq_hz=int(ARGS.simulation_freq_hz / AGGR_PHY_STEPS),
        num_drones=NUM_DRONES
    )

    ctrl = [DSLPIDControl(env) for _ in range(NUM_DRONES)]
    tune_pid_for_generic_motion(ctrl[0])

    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ / ARGS.control_freq_hz))
    action = {"0": np.array([0.0, 0.0, 0.0, 0.0])}

    START = time.time()

    if ARGS.live_plot and ARGS.plot:
        live_plot = Live3DPlot(ref_points)
        PLOT_UPDATE_FREQ = 10
    else:
        live_plot = None

    print("\nStarting simulation...")
    try:
        for i in range(0, int(ARGS.duration_sec * env.SIM_FREQ), AGGR_PHY_STEPS):
            current_time = i / env.SIM_FREQ
            obs, reward, done, info = env.step(action)

            if i % CTRL_EVERY_N_STEPS == 0:
                target_pos, target_rpy = get_target(current_time, ARGS.traj_type, traj_params)

                state = obs["0"]["state"]
                rpm, _, _ = ctrl[0].computeControlFromState(
                    control_timestep=CTRL_EVERY_N_STEPS * env.TIMESTEP,
                    state=state,
                    target_pos=target_pos,
                    target_rpy=target_rpy
                )
                action["0"] = rpm

            # log
            current_pos = obs["0"]["state"][0:3]
            logger.log(
                drone=0,
                timestamp=current_time,
                state=obs["0"]["state"],
                control=np.hstack([current_pos, INIT_RPYS[0, :], np.zeros(6)])
            )

            # live plot
            if live_plot is not None and i % PLOT_UPDATE_FREQ == 0:
                live_plot.update_plot(current_pos, target_pos)

            if i % env.SIM_FREQ == 0:
                env.render()
                error = np.linalg.norm(current_pos - target_pos)
                print(f"t={current_time:5.1f}s | Error={error:.3f}m | Pos=({current_pos[0]:.2f}, "
                      f"{current_pos[1]:.2f}, {current_pos[2]:.2f})")

            if ARGS.gui:
                sync(i, START, env.TIMESTEP)

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        env.close()
        logger.save()

        if live_plot is not None:
            print("Keeping live plot open for inspection...")
            plt.ioff()
            plt.show()
            live_plot.close()

    if ARGS.plot:
        logger.plot()
