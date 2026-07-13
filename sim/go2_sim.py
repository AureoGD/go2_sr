import numpy as np
import mujoco

from sim.state import SystemState
from sim.engine.pinocchio_engine import PinocchioEngine
from sim.utils.transforms import euler_to_quat, quat_to_euler
from sim.debug.debug_visualization import DebugVisualizer

import copy

class Go2Sim:

    # ======================================================
    # INIT
    # ======================================================
    def __init__(self, mj_model, mj_data, controller=None, pin_engine=None, con_dt=0.01, dyn_dt=0.001, viewer=None, log_ep=False):

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
        self.debug_state = self.state.debug

        # -------------------------------
        # PINOCCHIO
        # -------------------------------
        self.pin_engine = pin_engine

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

        # self.debug_viz = None

        self.debug_viz = DebugVisualizer(self.viewer)

        # -------------------------------
        # INTERNAL
        # -------------------------------
        self.iterations = 0

        self.log_ep = log_ep
        self.log = []

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

            if self.log_ep:

                self.pin_engine.update(self.robot_state)

                self._com_quantities()

                self.log.append(copy.deepcopy(self.state))

            # self._feet_quatities()

        self._feet_quatities()
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

    def _feet_quatities(self):
        foot_map = {'FR': 0, 'FL': 1, 'RR': 2, 'RL': 3}

        self.robot_state.foot_touching[:] = 0

        for i in range(self.mj_data.ncon):

            contact = self.mj_data.contact[i]

            g1 = self.mj_model.geom(contact.geom1).name
            g2 = self.mj_model.geom(contact.geom2).name

            if g1 in foot_map:
                self.robot_state.foot_touching[foot_map[g1]] = 1

            if g2 in foot_map:
                self.robot_state.foot_touching[foot_map[g2]] = 1

    # ======================================================
    # LOW-LEVEL CONTROL (PD + gravity)
    # ======================================================
    def _low_level_control(self):

        q = self.robot_state.q
        dq = self.robot_state.dq
        qr = self.low_level_state.qr

        if not np.isfinite(q).all():
            raise RuntimeError("q inválido antes do gravity")

        if not np.isfinite(dq).all():
            raise RuntimeError("dq inválido antes do gravity")

        KP = self.low_level_state.Kp
        KD = self.low_level_state.Kd

        q_error = qr - q
        dq_error = -dq

        tau_pd = KP * q_error + KD * dq_error
        tau_g = self.pin_engine.gravity()

        if self.controller.comp_grav:
            tau = tau_pd+tau_g
        else:
            tau = tau_pd
        
        self.low_level_state.tau_pd = tau_pd
        self.low_level_state.tau_g = tau_g
        self.low_level_state.tau = tau

        return np.clip(tau, -self.torque_limits, self.torque_limits)

    # ======================================================
    # PHYSICS
    # ======================================================
    def _physics(self, tau):
        if not np.isfinite(tau).all():
            raise RuntimeError("Torque inválido (NaN/Inf)")

        self.mj_data.ctrl[:] = tau

        mujoco.mj_step(self.mj_model, self.mj_data)

        if not np.isfinite(self.mj_data.qpos).all():
            raise RuntimeError("Estado inválido após mj_step (qpos)")

        if not np.isfinite(self.mj_data.qvel).all():
            raise RuntimeError("Estado inválido após mj_step (qvel)")

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

        # ---------------------------
        # ADD THIS BLOCK
        # ---------------------------
        if self.debug_viz is not None:

            self.debug_viz.render(self.debug_state.sw_foot_data, self.debug_state.plane_pos, self.robot_state.rpy)

        self.viewer.sync()

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

        self.log.clear()
