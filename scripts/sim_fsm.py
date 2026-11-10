import numpy as np
import mujoco
import mujoco.viewer

from sim.go2_sim import Go2Sim
from sim.engine.pinocchio_engine import PinocchioEngine
import pinocchio as pin

from control.self_righting.rgc_mpc_solution.rgc_scheduler import SchedulerRGCMPC
from fsm.self_righting_fsm import RobotStatus, SelfRightingFSM
from sim.utils.logging import log_to_arrays
import copy


class SimFSM():

    def __init__(self, log=False):
        self.max_tick = 750

        root_joint = pin.JointModelFreeFlyer()
        pin_model = pin.buildModelFromUrdf("sim/assets/unitree_go2/go2.urdf", root_joint)
        pin_engine = PinocchioEngine(pin_model)

        model = mujoco.MjModel.from_xml_path("sim/assets/unitree_go2/scene.xml")
        data = mujoco.MjData(model)
        viewer = mujoco.viewer.launch_passive(model, data)

        conf_controller = {"pin_engine": pin_engine}

        control = SchedulerRGCMPC(**conf_controller)

        self.sim = Go2Sim(mj_model=model,
                          mj_data=data,
                          controller=control,
                          pin_engine=pin_engine,
                          viewer=viewer,
                          log_ep=log)

        self.robot_status = RobotStatus(self.sim.state, self.sim.controller.task_state, self.sim.torque_limits,
                                        self.sim.joint_limits)
        self.fsm = SelfRightingFSM('CW')

        self.ep_log = log

        b0 = [2.5, 0, 1]
        r0 = [np.pi, 0, 0.1]
        q0 = [-0.3, 1.0, -1.72, -0.5, 1.0, -1.72, 0.5, 1.40, -1.7, -0.5, 0.8, -1.5]
        self.sim_reset(r0, b0, q0)
        self.action_list = []
        self.control_task = []

    def sim_reset(self, r0, b0, q0):
        self.fsm.reset()
        self.sim.reset_robot_pose(b0=b0, r0=r0, q0=q0)

    def run_sim(self):
        tick = 0
        # while tick < self.max_tick and not self.fsm.stand_flag:
        while tick < self.max_tick:
            if tick < 100:
                action = 0
            else:
                action = self.fsm.update(self.robot_status)
            self.action_list.append(action)
            self.sim.simulation_loop(action)
            self.control_task.append(copy.deepcopy(self.sim.controller.task_state))
            tick += 1

        if self.ep_log:
            return self.task_success(), log_to_arrays(self.sim.log), self.action_list, log_to_arrays(self.control_task) 
        else:
            return self.task_success(), None, None

    def task_success(self):
        flag_succes = False

        if self.fsm.stand_flag:
            rs = self.sim.rs
            flag_succes = self.fsm.stand_flag and bool(np.all(rs.foot_touching) == 1)

        return flag_succes
