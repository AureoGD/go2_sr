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
    def __init__(self,
                 mj_model,
                 mj_data,
                 controller=None,
                 pin_engine=None,
                 con_dt=0.01,
                 dyn_dt=0.001,
                 viewer=None,
                 log_ep=False):

        # -------------------------------
        # MuJoCo
        # -------------------------------
        self.mj_model = mj_model
        self.mj_data = mj_data

        self.body_id = self.mj_model.body("base").id

        def geom_ids_for_body(model, body_name):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            return np.where(model.geom_bodyid == body_id)[0]

        self.geom_ids_fr_thigh = geom_ids_for_body(self.mj_model, "FR_hip")
        self.geom_ids_rr_thigh = geom_ids_for_body(self.mj_model, "RR_hip")
        self.geom_ids_fr_foot = geom_ids_for_body(self.mj_model, "RL_foot")

        # -------------------------------
        # STATE
        # -------------------------------
        self.state = SystemState()
        self.rs = self.state.robot
        self.cs = self.state.low_level
        self.dg = self.state.debug

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

        self.pin_engine.update(self.rs)
        self._com_quantities()

        # only for test the force log
        self._feet_quatities()

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

                self.pin_engine.update(self.rs)

                self._com_quantities()

                self.log.append(copy.deepcopy(self.state))

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
        self.rs.r_pos = self.pin_engine.com()
        self.rs.r_vel = self.pin_engine.vcom()

    def _feet_quatities(self):
        foot_map = {'FR': 0, 'FL': 1, 'RR': 2, 'RL': 3}

        self.rs.foot_touching[:] = 0
        self.rs.force_mj[:] = 0.0

        f = np.zeros(6)

        for i in range(self.mj_data.ncon):

            contact = self.mj_data.contact[i]
            g1 = self.mj_model.geom(contact.geom1).name
            g2 = self.mj_model.geom(contact.geom2).name

            foot = g1 if g1 in foot_map else (g2 if g2 in foot_map else None)
            if foot is None:
                continue

            idx = foot_map[foot]
            self.rs.foot_touching[idx] = 1

            mujoco.mj_contactForce(self.mj_model, self.mj_data, i, f)
            R = contact.frame.reshape(3, 3)
            f_world = R.T @ f[:3]
            if g1 in foot_map:
                f_world = -f_world
            self.rs.force_mj[idx] += f_world

    # ======================================================
    # LOW-LEVEL CONTROL (PD + gravity)
    # ======================================================
    def _low_level_control(self):

        q = self.rs.q
        dq = self.rs.dq
        qr = self.cs.qr

        if not np.isfinite(q).all():
            raise RuntimeError("q inválido antes do gravity")

        if not np.isfinite(dq).all():
            raise RuntimeError("dq inválido antes do gravity")

        KP = self.cs.Kp
        KD = self.cs.Kd

        q_error = qr - q
        dq_error = -dq

        tau_pd = KP * q_error + KD * dq_error

        tau_g = self.pin_engine.gravity()

        if self.controller.comp_grav:
            tau = tau_pd + tau_g
        else:
            tau = tau_pd

        self.cs.tau_pd = tau_pd
        self.cs.tau_g = tau_g
        self.cs.tau = tau

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
        self.rs.b_pos = self.mj_data.qpos[0:3].copy()

        # MuJoCo (wxyz) → Pinocchio (xyzw)
        quat_wxyz = self.mj_data.qpos[3:7]
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        self.rs.epsilon = quat_xyzw

        self.rs.rpy = quat_to_euler(quat=quat_xyzw, order='xyzw')

        # joints (Unitree order)
        self.rs.q = self.mj_data.qpos[7:].copy()

        # velocities
        R_wb = self.mj_data.xmat[self.body_id].reshape(3, 3)

        self.rs.b_vel = self.mj_data.qvel[0:3].copy()  # world frame
        self.rs.b_vel_b = R_wb.T @ self.rs.b_vel  # body frame

        self.rs.omega_b = self.mj_data.qvel[3:6].copy()  # body frame
        self.rs.omega = R_wb @ self.rs.omega_b  # world frame

        self.rs.dq = self.mj_data.qvel[6:].copy()

        mujoco.mj_rnePostConstraint(self.mj_model, self.mj_data)
        self.rs.f_com_total = np.sum(self.mj_data.cfrc_ext[:, 3:6], axis=0)
        # self.rs.f_com_total[2] -= self.mj_model.body_mass.sum() * 9.81

    # ======================================================
    # RENDER
    # ======================================================
    def _render(self):

        if self.viewer is None:
            return

        base_pos = self.rs.b_pos
        self.viewer.cam.lookat[:] = base_pos

        if self.debug_viz is not None:
            self.debug_viz.render(self.dg.sw_foot_data)

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

        self.cs.qr = q0.copy()

        mujoco.mj_forward(self.mj_model, self.mj_data)

        render_aux = self._is_render
        self._is_render = False

        self._update_robot_state_from_mujoco()

        self.pin_engine.update(self.rs)

        for _ in range(100):

            tau = self._low_level_control()

            self._physics(tau)

            self._update_robot_state_from_mujoco()

            self.pin_engine.update(self.rs)

        self._is_render = render_aux

        self._update_robot_state_from_mujoco()

        self.pin_engine.update(self.rs)

        if self.controller is not None:
            if hasattr(self.controller, "reset"):
                self.controller.reset()

        self.iterations = 0

        self.log.clear()
