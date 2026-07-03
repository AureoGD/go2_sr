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
from control.self_righting.time_based_solution.time_based_scheduler import SchedulerTB
from control.self_righting.unitree_solution.unitree_solution import UnitreeSelfRighting

from fsm.self_righting_fsm import RobotStatus, SelfRightingFSM
from fsm.self_righting_tsm import SelfRightingTSM

mode = 'fsm'

np.set_printoptions(linewidth=200)


def main():

    root_joint = pin.JointModelFreeFlyer()
    pin_model = pin.buildModelFromUrdf("sim/assets/unitree_go2/go2.urdf", root_joint)
    pin_engine = PinocchioEngine(pin_model)

    model = mujoco.MjModel.from_xml_path("sim/assets/unitree_go2/scene.xml")
    data = mujoco.MjData(model)
    viewer = mujoco.viewer.launch_passive(model, data)

    conf_controller = {"pin_engine": pin_engine}

    # FSM
    control = SchedulerRGCMPC(**conf_controller)

    # TSM
    # control = SchedulerTB(**conf_controller)

    # Unitree
    # control = UnitreeSelfRighting(**conf_controller)

    sim = Go2Sim(mj_model=model, mj_data=data, controller=control, pin_engine=pin_engine, viewer=viewer)

    b0 = [2.5, 0, 0.5]
    r0 = [np.pi, 0, -np.pi * 125 / 180]
    q0 = [-0.3, 1.0, -1.72, -0.5, 1.0, -1.72, 0.5, 1.40, -1.7, -0.5, 0.8, -1.5]

    sim.reset_robot_pose(b0=b0, r0=r0, q0=q0)

    robot_status = RobotStatus(sim.state, sim.controller.task_state, sim.torque_limits, sim.joint_limits)
    fsm = SelfRightingFSM(default='CCW')
    last_action = -1

    for tick in range(2000):
        if tick < 50:
            action = 0
        else:
            # action = tsm.step(sim.state.robot.rpy)
            action = fsm.update(robot_status)
        if last_action != action:
            last_action = action
        sim.simulation_loop(action)


if __name__ == "__main__":
    main()
