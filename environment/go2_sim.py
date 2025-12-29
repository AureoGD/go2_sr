import os
import time
import numpy as np
import mujoco
import mujoco.viewer
import math
import pinocchio as pin
from environment.robot_states import RobotStates
from scipy.spatial.transform import Rotation
from environment.strategies.unitree_self_righting import UnitreeSelfRighting
from environment.strategies.scheduler_rgc_mpc import SchedulerRGCMPC
from environment.strategies.rgc_mpc.base_controller import BaseRGC
import itertools
import signal

# ============================================================
# NUMERICAL SAFETY (NO LOGIC CHANGE)
# ============================================================
np.seterr(all="raise")
np.set_printoptions(linewidth=1000, precision=3)


class Go2ModelSimMuJoCo():

    def __init__(self, task_control=None, render=False, strategy_name='unitree', **kwargs):
        self._is_render = render
        self.pin_model = None
        self.pin_data = None
        self.geo_data = None
        self.geo_model = None
        self.task_control = task_control

        self.links_ids = []
        self.foot_ids = []
        self.joint_idx_list = []

        self.legs = ['FR', 'FL', 'RR', 'RL']
        self.links = ['hip', 'thigh', 'calf']

        self.dyn_dt = 0.001
        self.con_dt = 0.01

        # ====================================================
        # LOAD MUJOCO MODEL (UNCHANGED)
        # ====================================================
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "assets/unitree_go2/scene.xml")

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL

        # ====================================================
        # LOAD PINOCCHIO MODEL (UNCHANGED)
        # ====================================================
        urdf_path = os.path.join(current_dir, "assets/unitree_go2/go2.urdf")
        if os.path.exists(urdf_path):
            root_joint = pin.JointModelFreeFlyer()
            self.pin_model = pin.buildModelFromUrdf(urdf_path, root_joint)
            self.pin_data = self.pin_model.createData()
            self.geo_model = pin.buildGeomFromUrdf(self.pin_model, urdf_path, pin.GeometryType.COLLISION)
            self.geom_data = pin.GeometryData(self.geo_model)
        else:
            self.geom_data = None

        # ====================================================
        # ORIGINAL JOINT MAPPING (UNCHANGED)
        # ====================================================
        self._setup_joint_mapping()

        self.model.opt.timestep = self.dyn_dt

        self.viewer = None
        if self._is_render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # ====================================================
        # CONTROL GAINS (UNCHANGED)
        # ====================================================
        self.kp = 50
        self.kd = 2.5
        self.KP = self.kp * np.eye(12)
        self.KD = self.kd * np.eye(12)

        # ====================================================
        # STATE VARIABLES (UNCHANGED)
        # ====================================================
        self.q = np.zeros(12)
        self.qr = np.zeros(12)
        self.dq = np.zeros(12)
        self.dqr = np.zeros(12)
        self.delta_qr = np.zeros(12)

        self.tau_max = np.array([23.7, 23.7, 45.43, 23.7, 23.7, 45.43, 23.7, 23.7, 45.43, 23.7, 23.7, 45.43])

        self.robot_states = RobotStates()

        # ====================================================
        # STRATEGY INIT (UNCHANGED)
        # ====================================================
        config = {
            'model': self.pin_model,
            'data': self.pin_data,
            'geo_model': self.geo_model,
            'geo_data': self.geo_data,
            'links': self.links,
            'legs': self.legs,
            'links_ids': self.links_ids,
            'foot_ids': self.foot_ids,
            'kp': self.kp,
            'kd': self.kd,
            'robot_states': self.robot_states
        }

        self.base_rgc = BaseRGC(**config)

        if self.task_control is None:
            if strategy_name == 'rgc':
                self.task_control = SchedulerRGCMPC(**config)
            else:
                self.task_control = UnitreeSelfRighting(**config)

        # ====================================================
        # TIMEOUT HANDLER (NEW)
        # ====================================================
        signal.signal(signal.SIGALRM, self._timeout_handler)

        self._update_robot_sim_states()

    # ========================================================
    # SAFETY HELPERS (NEW)
    # ========================================================
    def _assert_finite(self, name, x):
        if not np.all(np.isfinite(x)):
            raise FloatingPointError(f"[NaN/Inf DETECTED] {name}: {x}")

    def _timeout_handler(self, signum, frame):
        raise TimeoutError("MuJoCo mj_step timeout")

    # ========================================================
    # JOINT MAPPING (ORIGINAL — UNTOUCHED)
    # ========================================================
    def _setup_joint_mapping(self):
        self.joint_names = []
        self.actuator_names = []

        for leg in self.legs:
            for link in self.links:
                self.joint_names.append(f"{leg}_{link}_joint")
                self.actuator_names.append(f"{leg}_{link}")

        self.joint_idx_list = []
        self.joint_qpos_addr = []

        for name in self.joint_names:
            joint_id = self.model.joint(name).id
            self.joint_idx_list.append(joint_id)
            self.joint_qpos_addr.append(self.model.jnt_qposadr[joint_id])

        self.actuator_idx_list = []
        for name in self.actuator_names:
            self.actuator_idx_list.append(self.model.actuator(name).id)

        if self.pin_model is not None:
            for leg in self.legs:
                for l_name in self.links:
                    try:
                        self.links_ids.append(self.pin_model.getFrameId(f'{leg}_{l_name}'))
                    except:
                        pass
                try:
                    self.foot_ids.append(self.pin_model.getFrameId(f'{leg}_foot'))
                except:
                    pass

    # ========================================================
    # PHYSICS STEP (INSTRUMENTED)
    # ========================================================
    def _physics(self, tau):
        self._assert_finite("tau", tau)
        self._assert_finite("qpos(before)", self.data.qpos)
        self._assert_finite("qvel(before)", self.data.qvel)

        for i, actuator_idx in enumerate(self.actuator_idx_list):
            self.data.ctrl[actuator_idx] = tau[i]

        mujoco.mj_step(self.model, self.data)

        self._assert_finite("qpos(after)", self.data.qpos)
        self._assert_finite("qvel(after)", self.data.qvel)

    # ========================================================
    # LOW-LEVEL CONTROL (INSTRUMENTED)
    # ========================================================
    def _low_level_control(self):
        self._update_robot_sim_states()

        self._assert_finite("q", self.q)
        self._assert_finite("dq", self.dq)
        self._assert_finite("qr", self.qr)

        tau = self.KP @ (self.qr - self.q) + self.KD @ (self.dqr - self.dq)
        self._assert_finite("tau_pd", tau)

        self.robot_states.tau_pd = tau.reshape(12, 1)

        tau_g = self._comp_tau_g() if self.pin_model is not None else np.zeros(12)
        self._assert_finite("tau_g", tau_g)

        return np.clip(tau + tau_g, -self.tau_max, self.tau_max)

    # ========================================================
    # GRAVITY (INSTRUMENTED)
    # ========================================================
    def _comp_tau_g(self):
        q_full = np.vstack((self.robot_states.b_pos, self.robot_states.epsilon, self.robot_states.q[3:6],
                            self.robot_states.q[0:3], self.robot_states.q[9:12], self.robot_states.q[6:9]))

        tau_g = pin.computeGeneralizedGravity(self.pin_model, self.pin_data, q_full)[6:]
        self._assert_finite("pin.tau_g", tau_g)

        tau_g = np.vstack((tau_g[3:6], tau_g[0:3], tau_g[9:12], tau_g[6:9])).reshape(12)
        self.robot_states.tau_g = tau_g.reshape(12, 1)
        return tau_g

    # ========================================================
    # STATE UPDATE (INSTRUMENTED)
    # ========================================================
    def _update_robot_sim_states(self):
        for i, qpos_addr in enumerate(self.joint_qpos_addr):
            self.q[i] = self.data.qpos[qpos_addr]
            qvel_addr = self.model.jnt_dofadr[self.joint_idx_list[i]]
            self.dq[i] = self.data.qvel[qvel_addr]

        self.robot_states.q = self.q.reshape(12, 1)
        self.robot_states.dq = self.dq.reshape(12, 1)
        self.robot_states.b_pos = self.data.qpos[0:3].reshape(3, 1)
        self.robot_states.epsilon = np.array(
            (self.data.qpos[4], self.data.qpos[5], self.data.qpos[6], self.data.qpos[3])).reshape(4, 1)
        self.robot_states.omega = self.data.qvel[3:6].reshape(3, 1)
        self.robot_states.rpy = self._quat_to_euler(self.robot_states.epsilon).reshape(3, 1)

        self._assert_finite("robot_states.q", self.robot_states.q)
        self._assert_finite("robot_states.dq", self.robot_states.dq)

        self.base_rgc.com_quatities()

    # ========================================================
    # MAIN LOOP (TIMEOUT-PROTECTED)
    # ========================================================
    def control_loop(self, mode):
        if mode != -1:
            self._task_control(mode)

            if self.robot_states.critical_mpc_fail:
                return

        for _ in range(int(self.con_dt / self.dyn_dt)):
            signal.alarm(1)
            try:
                tau = self._low_level_control()
                self._physics(tau)
            finally:
                signal.alarm(0)

        if self._is_render and self.viewer:
            self.viewer.sync()
            time.sleep(self.con_dt)

    # ========================================================
    # TASK CONTROL (UNCHANGED)
    # ========================================================
    def _task_control(self, mode):
        self.qr += self.delta_qr
        self.robot_states.qr = self.qr.reshape(12, 1)
        self.delta_qr, self.KP, self.KD = self.task_control.update(mode)

    def reset_robot_pose(self, q0=None, b0=None, r0=None):
        """Reset robot to initial pose (ORIGINAL LOGIC)"""

        if q0 is None:
            q0 = [0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7]

        if b0 is None:
            b0 = [0, 0, 0.085]

        if r0 is None:
            r0 = [np.pi, 0, 0]

        # --- OPTIONAL SAFETY CHECKS (DO NOT CHANGE STATE) ---
        # Uncomment if you want early failure on bad resets
        self._assert_finite("reset.q0", np.array(q0))
        self._assert_finite("reset.b0", np.array(b0))
        self._assert_finite("reset.r0", np.array(r0))

        # Base position
        self.data.qpos[0:3] = b0

        # Base orientation (YOUR quaternion convention)
        quat = self._euler_to_quat(r0)
        self.data.qpos[3:7] = quat

        # Joint positions (YOUR mapping)
        for i, qpos_addr in enumerate(self.joint_qpos_addr):
            if qpos_addr != -1 and qpos_addr < self.model.nq:
                self.data.qpos[qpos_addr] = q0[i]

        # Zero velocities
        self.data.qvel[:] = 0.0
        self.qr = np.array(q0).reshape(12)

        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)

        # Let contacts settle (YOUR logic)
        render_aux = self._is_render
        self._is_render = False
        for _ in range(100):
            self._physics(tau=np.zeros(12))
        self._is_render = render_aux

        self._update_robot_sim_states()
        self.iterations = 0

    def _euler_to_quat(self, euler):
        """Convert Euler angles to quaternion in MuJoCo's [w, x, y, z] order"""
        r = Rotation.from_euler('xyz', euler)
        quat_xyzw = r.as_quat()
        quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        return quat_wxyz

    def _quat_to_euler(self, q):
        """
        Converts a quaternion [x, y, z, w] to RPY [roll, pitch, yaw].
        """
        x, y, z, w = q

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.sign(sinp) * np.pi / 2  # Gimbal lock fallback
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    # ========================================================
    # RESET / CLOSE (UNCHANGED)
    # ========================================================
    def close(self):
        if self.viewer:
            self.viewer.close()
