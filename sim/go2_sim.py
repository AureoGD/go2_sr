import numpy as np
import mujoco
import pinocchio as pin

from sim.state import SystemState
from sim.engine.pinocchio_engine import PinocchioEngine
from sim.utils.transforms import euler_to_quat, quat_to_euler


class Go2Sim:

    # ======================================================
    # INIT
    # ======================================================
    def __init__(self, urdf_path, mj_model, mj_data, controller=None, con_dt=0.01, dyn_dt=0.001, viewer=None):

        # -------------------------------
        # MuJoCo
        # -------------------------------
        self.mj_model = mj_model
        self.mj_data = mj_data

        # -------------------------------
        # STATE
        # -------------------------------
        self.state = SystemState()
        self.robot_state = self.state.robot
        self.low_level_state = self.state.low_level

        # -------------------------------
        # PINOCCHIO
        # -------------------------------
        root_joint = pin.JointModelFreeFlyer()
        self.pin_model = pin.buildModelFromUrdf(urdf_path, root_joint)
        self.pin_engine = PinocchioEngine(self.pin_model)

        # -------------------------------
        # CONTROLLER
        # -------------------------------
        self.controller = controller

        # -------------------------------
        # TIMING
        # -------------------------------
        self.con_dt = con_dt
        self.dyn_dt = dyn_dt
        self.n_substeps = int(self.con_dt / self.dyn_dt)

        # -------------------------------
        # LOW-LEVEL GAINS
        # -------------------------------
        self.KP = np.eye(12) * 50.0
        self.KD = np.eye(12) * 2.0

        # ------------------------------
        # JOINT AND TORQUE LIMITS
        # ------------------------------
        self.joint_limits = self.pin_engine.get_joint_limits()
        self.torque_limits = self.pin_engine.get_torque_limits()

        # -------------------------------
        # VIEWER
        # -------------------------------
        self.viewer = viewer
        self._is_render = viewer is not None

        # -------------------------------
        # INTERNAL
        # -------------------------------
        self.iterations = 0

    # ======================================================
    # MAIN LOOP
    # ======================================================
    def simulation_loop(self, action):

        # --------------------------------------
        # 1. Atualizar estado
        # --------------------------------------
        self._update_robot_state_from_mujoco()

        self.pin_engine.update(self.robot_state)
        self._com_quantities()

        # --------------------------------------
        # 2. HIGH-LEVEL CONTROLLER (100 Hz)
        # --------------------------------------
        if self.controller is not None:
            self.controller.before_step(self.state, action)

        # --------------------------------------
        # 3. LOW-LEVEL LOOP (1 kHz)
        # --------------------------------------
        for _ in range(self.n_substeps):

            tau = self._low_level_control()

            self._physics(tau)

            self._update_robot_state_from_mujoco()

            self.pin_engine.update(self.robot_state)

            self._com_quantities()

        # --------------------------------------
        # 4. AFTER STEP
        # --------------------------------------
        if self.controller is not None:

            self.controller.after_step(self.state)

        # --------------------------------------
        # 5. RENDER
        # --------------------------------------
        if self._is_render:
            self._render()

        self.iterations += 1

    # ======================================================
    # CoM quantities
    # ======================================================
    def _com_quantities(self):
        self.robot_state.r_pos = self.pin_engine.com()
        self.robot_state.r_vel = self.pin_engine.vcom()

    # ======================================================
    # LOW-LEVEL CONTROL (PD + gravity)
    # ======================================================
    def _low_level_control(self):

        q = self.robot_state.q
        dq = self.robot_state.dq
        qr = self.low_level_state.qr

        KP = self.low_level_state.Kp
        KD = self.low_level_state.Kd

        q_error = qr - q
        dq_error = -dq

        tau_pd = KP * q_error + KD * dq_error
        tau_g = self.pin_engine.gravity()

        tau = tau_pd + tau_g
        self.low_level_state.tau_pd = tau_pd
        self.low_level_state.tau_g = tau_g
        self.low_level_state.tau = tau

        return np.clip(tau_pd + tau_g, -self.torque_limits, self.torque_limits)

    # ======================================================
    # PHYSICS
    # ======================================================
    def _physics(self, tau):
        self.mj_data.ctrl[:] = tau
        mujoco.mj_step(self.mj_model, self.mj_data)

    # ======================================================
    # UPDATE STATE FROM MUJOCO
    # ======================================================
    def _update_robot_state_from_mujoco(self):

        # base
        self.robot_state.b_pos = self.mj_data.qpos[0:3].copy()

        # MuJoCo (wxyz) → Pinocchio (xyzw)
        quat_wxyz = self.mj_data.qpos[3:7]
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        self.robot_state.epsilon = quat_xyzw

        self.robot_state.rpy = quat_to_euler(quat=quat_xyzw, order='xyzw')

        # joints (Unitree order)
        self.robot_state.q = self.mj_data.qpos[7:].copy()

        # velocities
        self.robot_state.b_vel = self.mj_data.qvel[0:3].copy()
        self.robot_state.omega = self.mj_data.qvel[3:6].copy()
        self.robot_state.dq = self.mj_data.qvel[6:].copy()

    # ======================================================
    # RENDER
    # ======================================================
    def _render(self):

        if self.viewer is None:
            return

        base_pos = self.robot_state.b_pos
        self.viewer.cam.lookat[:] = base_pos
        self.viewer.sync()

    # ======================================================
    # RESET ROBOT POSE (COM SETTLING)
    # ======================================================
    def reset_robot_pose(self, q0=None, b0=None, r0=None):

        if q0 is None:
            q0 = [0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7]

        if b0 is None:
            b0 = [0, 0, 0.085]

        if r0 is None:
            r0 = [np.pi, 0, 0]

        q0 = np.array(q0)
        b0 = np.array(b0)
        r0 = np.array(r0)

        self.mj_data.qpos[0:3] = b0

        quat_xyzw = euler_to_quat(r0, 'xyzw')

        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        self.mj_data.qpos[3:7] = quat_wxyz

        self.mj_data.qpos[7:] = q0

        self.mj_data.qvel[:] = 0.0

        self.low_level_state.qr = q0.copy()

        mujoco.mj_forward(self.mj_model, self.mj_data)

        render_aux = self._is_render
        self._is_render = False

        self._update_robot_state_from_mujoco()

        self.pin_engine.update(self.robot_state)

        for _ in range(100):

            tau = self._low_level_control()

            self._physics(tau)

            self._update_robot_state_from_mujoco()

            self.pin_engine.update(self.robot_state)

        self._is_render = render_aux

        self._update_robot_state_from_mujoco()

        self.pin_engine.update(self.robot_state)

        if self.controller is not None:
            if hasattr(self.controller, "reset"):
                self.controller.reset()

        self.iterations = 0
