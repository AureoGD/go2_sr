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


def dummy_time_base(tick):
    if tick < 50:
        action = 0
    elif tick < 150:
        action = 1
    elif tick < 450:
        action = 2
    elif tick < 800:
        action = 3
    elif tick < 1050:
        action = 4
    elif tick < 1250:
        action = 5
    else:
        action = 6
    return action


def main():

    root_joint = pin.JointModelFreeFlyer()
    pin_model = pin.buildModelFromUrdf("sim/assets/unitree_go2/go2.urdf", root_joint)
    pin_engine = PinocchioEngine(pin_model)

    model = mujoco.MjModel.from_xml_path("sim/assets/unitree_go2/scene.xml")
    data = mujoco.MjData(model)
    viewer = mujoco.viewer.launch_passive(model, data)

    conf_controller = {"pin_engine": pin_engine}

    # if mode == 'time':
    #     control = SchedulerTB(**conf_controller)
    # else:
    #     control = SchedulerRGCMPC(**conf_controller)

    control = SchedulerTB(**conf_controller)
    # control = UnitreeSelfRighting(**conf_controller)

    sim = Go2Sim(mj_model=model, mj_data=data, controller=control, pin_engine=pin_engine, viewer=viewer)
    tsm = SelfRightingTSM()
    b0 = [2.5, 0, 1]
    # r0 = [0, 0, 0]
    # q0 = [-0.0, 1.40, -2.72, -0.0, 1.40, -2.72, -0.0, 1.40, -2.72, -0.00, 1.4, -2.72]

    r0 = [np.pi, 0, 0.1]
    q0 = [-0.3, 1.0, -1.72, -0.5, 1.0, -1.72, 0.5, 1.40, -1.7, -0.5, 0.8, -1.5]

    sim.reset_robot_pose(b0=b0, r0=r0, q0=q0)

    # robot_status = RobotStatus(sim.state, sim.controller.task_state, sim.torque_limits, sim.joint_limits)
    # fsm = SelfRightingFSM(default='CCW')
    last_action = -1
    
    for tick in range(2000):
        if tick<50:
            action = 0
        else:
            action = tsm.step(sim.state.robot.rpy)
        # if mode == 'time':
        #     action = dummy_time_base(tick)
        # elif mode == 'rgc':
        #     action = dummy_rgc(tick)
        # else:
        #     action = fsm.update(robot_status)
        # if last_action != action:
        #     last_action = action
        #     print(action)
        sim.simulation_loop(action)


if __name__ == "__main__":
    main()
