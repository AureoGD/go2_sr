import os
import numpy as np
import mujoco
import mujoco.viewer

from sim.go2_sim import Go2Sim
from control.self_righting.time_based_solution.time_based_scheduler import SchedulerTB
from tpe.simulation.scenario import TimeSchedulingScenario


# ======================================================
# LOGGER
# ======================================================
class EpisodeLogger:

    def __init__(self):
        self.reset()

    def reset(self):
        self.data = {
            "robot": {
                "pos": [],
                "vel": [],
                "omega": [],
                "rpy": [],
                "quat": [],
                "q": [],
                "dq": []
            },
            "low_level": {
                "tau": [],
                "qr": [],
                "dqr": []
            },
            "task": {
                "controller_index": [],
                "action_group": [],
                "phase_progress": []
            }
        }

    def log(self, state, controller):

        # ------------------------
        # ROBOT STATE
        # ------------------------
        rs = state.robot

        self.data["robot"]["pos"].append(rs.r_pos.copy())
        self.data["robot"]["vel"].append(rs.r_vel.copy())
        self.data["robot"]["omega"].append(rs.omega.copy())
        self.data["robot"]["rpy"].append(rs.rpy.copy())

        if hasattr(rs, "epsilon"):
            self.data["robot"]["quat"].append(rs.epsilon.copy())

        self.data["robot"]["q"].append(rs.q.copy())
        self.data["robot"]["dq"].append(rs.dq.copy())

        # ------------------------
        # LOW LEVEL STATE
        # ------------------------
        if hasattr(state, "low_level"):
            ls = state.low_level

            if hasattr(ls, "tau"):
                self.data["low_level"]["tau"].append(ls.tau.copy())

            if hasattr(ls, "qr"):
                self.data["low_level"]["qr"].append(ls.qr.copy())

            if hasattr(ls, "dqr"):
                self.data["low_level"]["dqr"].append(ls.dqr.copy())

        # ------------------------
        # TASK STATE (SchedulerTB)
        # ------------------------
        ts = controller.task_state

        self.data["task"]["controller_index"].append(getattr(ts, "controller_index", 0))

        self.data["task"]["action_group"].append(getattr(ts, "action_group", 0))

        self.data["task"]["phase_progress"].append(getattr(ts, "controller_evolution", 0.0))

    def save(self, path):

        flat_data = {}

        for group, values in self.data.items():
            for key, val in values.items():
                if len(val) > 0:
                    flat_data[f"{group}_{key}"] = np.array(val)

        np.savez(path, **flat_data)


# ======================================================
# SIMULATION
# ======================================================
def run_simulation(num_episodes=100, start_ep=0, render=False, save_dir="tpe/data/raw"):

    os.makedirs(save_dir, exist_ok=True)

    # ------------------------
    # MuJoCo setup
    # ------------------------
    urdf_path = "sim/assets/unitree_go2/go2.urdf"
    scene_path = "sim/assets/unitree_go2/scene.xml"

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    viewer = None
    if render:
        viewer = mujoco.viewer.launch_passive(model, data)

    # ------------------------
    # scenario
    # ------------------------
    scenario = TimeSchedulingScenario()

    # ------------------------
    # controller (stochastic)
    # ------------------------
    controller = SchedulerTB(stochastic=True)

    # ------------------------
    # sim
    # ------------------------
    sim = Go2Sim(urdf_path=urdf_path, mj_model=model, mj_data=data, controller=controller, viewer=viewer)

    for ep in range(start_ep, num_episodes):

        print(f"\nEpisode {ep}")

        # ------------------------
        # reset scenario
        # ------------------------
        scenario.reset()

        q0, rpy, b0 = scenario.get_initial_state()
        total_steps = scenario.get_total_steps()

        sim.reset_robot_pose(b0=b0, q0=q0, r0=rpy)

        # ------------------------
        # logger
        # ------------------------
        logger = EpisodeLogger()

        # ------------------------
        # loop
        # ------------------------
        for _ in range(total_steps):

            action = scenario.get_action()

            sim.simulation_loop(action=action)

            state = sim.state

            logger.log(state, controller)

        # ------------------------
        # save
        # ------------------------
        save_path = os.path.join(save_dir, f"episode_{ep:05d}.npz")

        logger.save(save_path)

        print(f"Saved: {save_path}")


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":

    run_simulation(num_episodes=2500, start_ep=820, render=True)
