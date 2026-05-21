import argparse
import json
import torch
import time
import random

import numpy as np
import mujoco
import mujoco.viewer

from sim.go2_sim import Go2Sim
from sim.engine.pinocchio_engine import PinocchioEngine
import pinocchio as pin

from control.self_righting.rgc_mpc_solution.rgc_scheduler import SchedulerRGCMPC


def dummy(tick):
    if tick < 50:
        action = 0
    elif tick < 150:
        action = 1
    elif tick < 500:
        action = 2
    elif tick < 600:
        action = 3
    else:
        action = 4

    return action


def main():

    root_joint = pin.JointModelFreeFlyer()
    pin_model = pin.buildModelFromUrdf("sim/assets/unitree_go2/go2.urdf", root_joint)
    pin_engine = PinocchioEngine(pin_model)

    model = mujoco.MjModel.from_xml_path("sim/assets/unitree_go2/scene.xml")
    data = mujoco.MjData(model)
    viewer = mujoco.viewer.launch_passive(model, data)

    conf_controller = {"pin_engine": pin_engine}

    control = SchedulerRGCMPC(**conf_controller)

    sim = Go2Sim(mj_model=model, mj_data=data, controller=control, pin_engine=pin_engine, viewer=viewer)

    b0 = [0, 0, 0.13]
    # r0 = [0, 0, 0]
    # q0 = [-0.0, 1.40, -2.72, -0.0, 1.40, -2.72, -0.0, 1.40, -2.72, -0.00, 1.4, -2.72]

    r0 = [np.pi, 0, 0]
    q0 = [-0.3, 1.0, -1.72, -0.5, 1.0, -1.72, 0.5, 1.40, -1.7, -0.5, 0.8, -1.5]

    sim.reset_robot_pose(b0=b0, r0=r0, q0=q0)

    for tick in range(1500):
        action = dummy(tick)
        sim.simulation_loop(action)


if __name__ == "__main__":
    main()
