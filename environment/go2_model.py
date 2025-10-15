import os
import time
import numpy as np
import pybullet_data
import pybullet as p
import pinocchio as pin
from robot_states import RobotStates
from sr_strategies.rgc.controller_screduller import ControlScheduller


class Go2ModelSim():

    def __init__(self, task_control=None, render=False):
        # system dynamics and control sample time
        self._is_render = render
        self.pin_model = None
        self.data = None

        self.task_control = task_control

        self.links_ids = []
        self.foot_ids = []
        self.joint_idx_list = []

        self.legs = ['FL', 'FR', 'RL', 'RR']
        self.links = ['hip', 'thigh', 'calf']

        self.dyn_dt = 0.001
        self.con_dt = 0.01

        self.physics_client = p.connect(p.GUI if self._is_render else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self._load_scnerio()
        self.model = self._load_robot()

        self._reset_robot_pose()

        self.kp = 50
        self.kd = 2.5

        self.base_joint_idx = None
        self.q = np.zeros((12, 1), dtype=np.float64)
        self.qr = np.zeros((12, 1), dtype=np.float64)
        self.dq = np.zeros((12, 1), dtype=np.float64)
        self.dqr = np.zeros((12, 1), dtype=np.float64)
        self.tau_max = np.array([23.7, 23.7, 45.43, 23.7, 23.7, 45.43, 23.7, 23.7, 45.43, 23.7, 23.7,
                                 45.43]).reshape(12, 1)

        self.base_pos = None
        self.base_orn = None
        self.interations = 0
        self.robot_states = RobotStates()

        config = {
            'model': self.pin_model,
            'data': self.data,
            'links': self.links,
            'legs': self.legs,
            'links_ids': self.links_ids,
            'foot_ids': self.foot_ids,
            'kp': self.kp,
            'kd': self.kd,
            'robot_states': self.robot_states
        }

        self.task_control = ControlScheduller(**config)

        self._update_robot_sim_states()

    def _physics(self, tau):
        p.setJointMotorControlArray(self.model, self.joint_idx_list, p.TORQUE_CONTROL, forces=tau)
        p.stepSimulation()

    def _low_level_control(self):
        self._update_robot_sim_states()
        tau = self.kp * (self.qr - self.q) + self.kd * (self.dqr - self.dq)
        tau_g = self._comp_tau_g()
        return np.clip(tau + tau_g, -self.tau_max, self.tau_max)

    def _comp_tau_g(self):
        q = np.vstack((self.base_pos, self.base_orn, self.q))
        pin.forwardKinematics(self.pin_model, self.data, q)
        J_com = pin.jacobianCenterOfMass(self.pin_model, self.data, q)
        tau_g = np.matmul(J_com.T, np.array([0, 0, -9.81]))
        return tau_g[6:].reshape(12, 1)

    def _task_control(self, mode=None):
        # self._test_kinematics()
        # update robot states if needed
        # update the qr
        # for sm implementation
        # self.qr = self.task_control.update()
        # for sr implementation
        self.robot_states.qr = self.qr
        self.qr = self.task_control.update(mode)

    def _update_robot_sim_states(self):
        # joint positions and velocities
        for idx in range(len(self.joint_idx_list)):
            self.q[idx], self.dq[idx], _, _ = p.getJointState(self.model, self.joint_idx_list[idx])

        # Base pose derivatives
        base_lin_vel_tuple, base_ang_vel_tuple = p.getBaseVelocity(self.model)
        self.base_lin_vel = np.array(base_lin_vel_tuple).reshape(3, 1)
        self.base_ang_vel = np.array(base_ang_vel_tuple).reshape(3, 1)
        # Base pose
        base_pos_tuple, base_ori_tuple = p.getBasePositionAndOrientation(self.model)
        self.base_pos = np.array(base_pos_tuple).reshape(3, 1)
        self.base_orn = np.array(base_ori_tuple).reshape(4, 1)

        self.robot_states.b_pos = self.base_pos
        self.robot_states.epsilon = self.base_orn
        self.robot_states.b_vel = self.base_lin_vel
        self.robot_states.omega = self.base_ang_vel
        self.robot_states.q = self.q
        self.robot_states.dq = self.dq

    def control_loop(self, mode):
        if self.interations > 100:
            self._task_control(mode=mode)
        for _ in range(int(self.con_dt / self.dyn_dt)):
            tau = self._low_level_control()
            self._physics(tau=tau)
        if self._is_render:
            time.sleep(self.con_dt)
        self.interations += 1

    def _load_scnerio(self, scenario_path=None):
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.dyn_dt)
        if scenario_path is None:
            plane = p.loadURDF("plane.urdf", [0, 0, 0], [0, 0, 0, 1])
        else:
            plane = p.loadURDF(scenario_path, [0, 0, 0], [0, 0, 0, 1])
        p.changeDynamics(plane, -1, lateralFriction=0.9)

    def _load_robot(self):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "go2.urdf")
        model = p.loadURDF(model_path, [0, 0, 0.0], [0, 0, 0, 1])
        root_joint = pin.JointModelFreeFlyer()
        self.pin_model = pin.buildModelFromUrdf(model_path, root_joint)
        self.data = self.pin_model.createData()
        joint_name_to_id = {}
        num_joints = p.getNumJoints(model)
        for i in range(num_joints):
            joint_info = p.getJointInfo(model, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_id = joint_info[0]
            joint_name_to_id[joint_name] = joint_id

        for leg in self.legs:
            for l_name in self.links:
                self.joint_idx_list.append(joint_name_to_id[f'{leg}_{l_name}_joint'])
                self.links_ids.append(self.pin_model.getFrameId(f'{leg}_{l_name}'))
            self.foot_ids.append(self.pin_model.getFrameId(f'{leg}_foot'))

        for joint in self.joint_idx_list:
            p.setJointMotorControl2(model, joint, p.VELOCITY_CONTROL, force=0)

        return model

    def _reset_robot_pose(self, q0=None, b0=None, r0=None):
        if q0 is None:
            q0 = (0.0, 1.36, -2.65, 0.0, 1.36, -2.65, 0, 1.36, -2.65, 0, 1.36, -2.65)
            b0 = [0, 0, 0.15]
            r0 = [0, 0, 0]

        quat = p.getQuaternionFromEuler(r0)

        for joint, joint_pos in zip(self.joint_idx_list, q0):
            p.resetJointState(self.model, joint, joint_pos)

        p.resetBasePositionAndOrientation(self.model, b0, quat)
