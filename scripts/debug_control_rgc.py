import argparse
import json
import torch
import time
import random

import numpy as np
import mujoco
import mujoco.viewer

from pathlib import Path

from sim.go2_sim import Go2Sim
from sim.engine.pinocchio_engine import PinocchioEngine
import pinocchio as pin

from control.self_righting.rgc_mpc_solution.rgc_scheduler import SchedulerRGCMPC
from fsm.self_righting_fsm import RobotStatus, SelfRightingFSM

np.set_printoptions(linewidth=200)


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

    b0 = [0, 0, 0.85]
    r0 = [np.pi, 0, np.pi * 0 / 180]
    q0 = [-0.045, 1.26, -2.8, 0.5, 1.26, -2.8, -0.31, 1.295, -2.8, 0.31, 1.295, -2.8]
    # q0 = [0.0, 1.4, -2.7, 0.0, 1.4, -2.7, 0.0, 1.4, -2.7, 0.0, 1.4, -2.7]
    # q0 = [0.7, 1.4, -2.6, -0.7, 1.4, -2.6, 0.7, 1.4, -2.6, -0.7, 1.4, -2.6]  #go safe final posture

    sim.reset_robot_pose(b0=b0, r0=r0, q0=q0)

    robot_status = RobotStatus(sim.state, sim.controller.task_state, sim.torque_limits, sim.joint_limits)
    fsm = SelfRightingFSM(default='CW')
    last_action = -1
    iter_k = 0

    for tick in range(1500):
        if tick < 50:
            action = 0
        else:
            action = fsm.update(robot_status)
        if last_action != action:
            last_action = action
        sim.simulation_loop(action)


if __name__ == "__main__":
    main()
