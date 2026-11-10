import argparse
import numpy as np
import mujoco
import mujoco.viewer

from pathlib import Path
import json

from sim.go2_sim import Go2Sim
from sim.engine.pinocchio_engine import PinocchioEngine
import pinocchio as pin

from control.self_righting.rgc_mpc_solution.rgc_scheduler import SchedulerRGCMPC
from fsm.self_righting_fsm import RobotStatus, SelfRightingFSM

np.set_printoptions(linewidth=200)

SCRIPT_DIR = Path(__file__).resolve().parent.parent

IC_FILE = SCRIPT_DIR / "evaluation" / "ics" / "ics_v2.json"

IC_ID = None
SLOPE = 4
SLOP_ANG_INC = 5


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--out",
                        type=str,
                        default="rgc_data",
                        help="output name (saved to evaluation/results/<out>.npz)")
    args = parser.parse_args()

    root_joint = pin.JointModelFreeFlyer()
    pin_model = pin.buildModelFromUrdf("sim/assets/unitree_go2/go2.urdf", root_joint)
    pin_engine = PinocchioEngine(pin_model)

    model = mujoco.MjModel.from_xml_path("sim/assets/unitree_go2/scene.xml")
    data = mujoco.MjData(model)
    viewer = mujoco.viewer.launch_passive(model, data)

    conf_controller = {"pin_engine": pin_engine}

    # FSM
    control = SchedulerRGCMPC(**conf_controller)

    sim = Go2Sim(mj_model=model, mj_data=data, controller=control, pin_engine=pin_engine, viewer=viewer)

    if IC_ID is None:
        b0 = [0, 0, 0.3]
        r0 = [np.pi, 0, 0]
        q0 = [0.7, 1.4, -2.6, -0.7, 1.4, -2.6, 0.7, 1.4, -2.6, -0.7, 1.4, -2.6]  #go safe final posture
    else:
        with open(IC_FILE, "r") as f:
            data = json.load(f)
        ic = data["ics"][IC_ID]

        slope_angle = SLOPE * SLOP_ANG_INC
        z = -0.05 + 3 * np.sin(np.deg2rad(slope_angle)) / 2
        b0 = [0, 3 * SLOPE, 0.3 + z]

        yaw = ic["yaw"]

        q0 = ic["q0"]

        r0 = [np.pi, 0.0, yaw]

    sim.reset_robot_pose(b0=b0, r0=r0, q0=q0)

    robot_status = RobotStatus(sim.state, sim.controller.task_state, sim.torque_limits, sim.joint_limits)
    fsm = SelfRightingFSM('CW')
    last_action = -1
    iter_k = 0
    tick = 0
    last_tick = 0
    while not fsm.stand_flag:
        if tick < 100:
            action = 0
        else:
            action = fsm.update(robot_status)
        if last_action != action:
            # print((tick - last_tick) * 0.01)
            last_tick = tick
            last_action = action

        sim.simulation_loop(action)

        tick += 1

        if fsm.stand_flag:
            print(f"Finish stand-up. T: {tick*0.01}")


if __name__ == "__main__":
    main()
