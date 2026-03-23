import numpy as np
import pinocchio as pin
from scipy.linalg import block_diag
from environment.strategies.rgc_mpc.base_controller import BaseRGC
from environment.strategies.rgc_mpc.poligon_constraint import SupportPolygonConstraint


class RollCW(BaseRGC):

    TASK_NAME = "roll_cw"
    TASK_LEVEL = 3

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not self.runtime:
            return

        self.N = 20
        self.M = 10
        self.ts = 0.01
        self.ws = 30
        self.convergence_threshold = 0.35

        self._update_detector()

        self.nx = 23
        self.nu = 12
        self.ny = 16
        self.nc = 18

        self.A = np.zeros((self.nx, self.nx), dtype=np.float32)
        self.B = np.zeros((self.nx, self.nu), dtype=np.float32)

        self.Aa = np.zeros((self.nx + self.nu, self.nx + self.nu), dtype=np.float32)
        self.Ba = np.zeros((self.nx + self.nu, self.nu), dtype=np.float32)
        self.Ca = np.zeros((self.ny, self.nx + self.nu), dtype=np.float32)

        self.C_cons = np.zeros((self.nc, self.nx + self.nu), dtype=np.float32)

        self.Aa[self.nx:, self.nx:] = np.identity(self.nu)
        self.Ba[self.nx:, :] = np.identity(self.nu)

        # Body orientation
        self.Ca[:12, 3:15] = np.eye(12)
        self.Ca[12:, 18:22] = np.eye(4)
        self.C_cons[0:12, 23:] = np.identity(12)

        self.Is = np.concatenate((np.identity(3), np.identity(3), np.identity(3), np.identity(3)), axis=1)

        Qfl = np.diag(np.array([1, 1, 1]))
        Qrl = np.diag(np.array([1, 1, 1]))
        Qeps = 0.1 * np.diag(np.array([1, 1, 1, 1]))

        Q = block_diag(Qrl, Qfl, Qrl, Qfl, Qeps)
        self.Q = block_diag(*[Q] * self.N)

        # References
        qr = np.array([0.4, 1.5, -2.0, -0.6, 1.3, -2.6, 0.4, 1.5, -2.0, 0.4, 3.75, -1.5]).reshape(12, 1)
        qeps = np.array([0, 0, 0, 1]).reshape(4, 1)
        ref = np.vstack((qr, qeps))
        self.ref = np.tile(ref, (self.N, 1))

        # Update control action weight matrix
        Rdqrfr = 750 * np.diag(np.array([1, 10, 10]))
        Rdqrfl = np.diag(np.array([1, 1, 1]))
        Rdqrr = 750 * np.diag(np.array([1, 10, 10]))
        Rdqrl = 75 * np.diag(np.array([1, 1, 1]))

        R = block_diag(Rdqrfr, Rdqrfl, Rdqrr, Rdqrl)
        self.R = block_diag(*[R] * self.M)

        qr_l = np.array([
            -1.0472, -1.5708, -2.7227, -1.0472, -1.5708, -2.7227, -1.0472, -0.5236, -2.7227, -1.0472, -0.5236, -2.7227
        ])
        qr_u = np.array(
            [1.0472, 3.4907, -0.83776, 1.0472, 3.4907, -0.83776, 1.0472, 4.5379, -0.83776, 1.0472, 4.5379, -0.83776])

        self.qr_l = qr_l.reshape(12, 1)
        self.qr_u = qr_u.reshape(12, 1)

        self.com_const = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]).reshape(6, 1)

        self.Jinv = np.zeros((12, 12), dtype=np.float32)

        # (FR=3, FL=0, RR=9, RL=6)
        # Contact at front-right foot, rear-right foot and rear-left foot
        self.leg_idx = [3, 9]

        self.contact_ids = [
            self.model.getFrameId('FR_thigh_joint'),
            self.model.getFrameId('RR_thigh_joint'),
        ]

        self.contacts = np.zeros((5, 3), dtype=np.float32)

        self.first_int = True

        self.active = 1
        self.safe_side = False

        # self.polig_const = SupportPolygonConstraint()

    def update_model(self):

        q, dq = self.ordering_joints()

        # 1. COMPUTE ALL KINEMATICS (Positions and Velocities)
        # Do this once at the beginning.
        pin.forwardKinematics(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data)

        # 2. COMPUTE JOINT-SPACE MATRICES
        pin.crba(self.model, self.data, q)
        pin.computeCoriolisMatrix(self.model, self.data, q, dq)

        # 3. COMPUTE ALL CENTROIDAL QUANTITIES
        pin.ccrba(self.model, self.data, q, dq)

        J_com_full = pin.jacobianCenterOfMass(self.model, self.data, q, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:, 6:]

        J_com = np.hstack([
            J_com_full[:, 3:6],  # front-right leg
            J_com_full[:, 0:3],  # front-left
            J_com_full[:, 9:12],  # rear-right leg
            J_com_full[:, 6:9]  # rear-left leg
        ])

        r = self.data.com[0]
        dr = self.data.vcom[0]

        # Save states
        self.robot_states.r_vel = dr.reshape(3, 1)
        self.robot_states.r_pos = r.reshape(3, 1)

        # Update angular velocity to quaternions matrix

        x, y, z, w = self.robot_states.epsilon
        T = 0.5 * np.array([[w, z, -y], [-z, w, x], [y, -x, w], [-x, -y, -z]])

        pivot_fr = self.data.oMf[self.model.getFrameId('FR_hip_joint')].translation
        pivot_rr = self.data.oMf[self.model.getFrameId('RR_hip_joint')].translation
        mean_pivot = (pivot_fr + pivot_rr) / 2

        Jc = np.zeros((12, 12))
        Jc[0:3, 0:3] = pin.computeFrameJacobian(self.model, self.data, q, self.model.getFrameId('FR_foot'),
                                                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[0:3, 9:12]
        foot_fr = self.data.oMf[self.model.getFrameId('FR_foot')].translation
        cross_fr = self.skew_symmetric_matrix(foot_fr - pivot_fr)

        Jc[3:6:, 3:6] = pin.computeFrameJacobian(self.model, self.data, q, self.model.getFrameId('FL_foot'),
                                                 pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[0:3, 6:9]

        Jc[6:9, 6:9] = pin.computeFrameJacobian(self.model, self.data, q, self.model.getFrameId('RR_foot'),
                                                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[0:3, 15:18]
        foot_rr = self.data.oMf[self.model.getFrameId('RR_foot')].translation
        cross_rr = self.skew_symmetric_matrix(foot_rr - pivot_rr)

        Jc[9:12, 9:12] = pin.computeFrameJacobian(self.model, self.data, q, self.model.getFrameId('RL_foot'),
                                                  pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[0:3, 12:15]
        foot_rl = self.data.oMf[self.model.getFrameId('RL_foot')].translation
        cross_rl = self.skew_symmetric_matrix(foot_rl - mean_pivot)

        self.contacts[0, :] = pivot_rr
        self.contacts[1, :] = foot_rr
        self.contacts[2, :] = foot_fr
        self.contacts[3, :] = pivot_fr
        self.contacts[4, :] = foot_rl

        cross_fl = np.zeros((3, 3))  # only to validate the rotation dynamics
        Sa = np.vstack((cross_fr, cross_fl, cross_rr, cross_rl))

        gamma = Jc
        gamma[3:6, :] = np.hstack((np.zeros((3, 3)), (np.eye(3)), np.zeros((3, 6))))

        gamma_a_star = np.linalg.inv(gamma) @ Sa

        self.Jinv = np.linalg.inv(Jc.T)

        I_com = self.data.Ig.inertia
        mass = self.data.mass[0]

        c_pivot = (pivot_fr + pivot_rr) / 2
        lever = self.data.com[0] - c_pivot
        S = self.skew_symmetric_matrix(lever)

        I_pivot = I_com + mass * (S.T @ S)
        I_inv = np.linalg.inv(I_pivot)

        comp_grav = S @ np.array([0, 0, mass])
        term_grav = I_inv @ comp_grav

        k3 = self.kp * I_inv @ gamma_a_star.T
        k4 = self.kd * I_inv @ gamma_a_star.T

        self.A[0:3, 0:3] = -k4 @ gamma_a_star
        self.A[0:3, 3:15] = -k3
        self.A[0:3, 22] = term_grav

        self.A[3:15, 0:3] = gamma_a_star

        self.A[15:18, 0:3] = -S + J_com @ gamma_a_star

        self.A[18:22, 0:3] = T.reshape(4, 3)

        self.B[0:3, :] = k3

        self.Aa[0:23, 0:23] = np.identity(self.nx) + self.ts * self.A
        self.Aa[0:23, 23:] = self.ts * self.B

        self.Ba[0:23, :] = self.ts * self.B

        self.x = np.vstack((self.robot_states.omega, self.robot_states.q, self.robot_states.r_pos,
                            self.robot_states.epsilon, -9.81, self.robot_states.qr))

    def define_constraints_matrices(self):

        if self.first_int:
            current_yaw = self.robot_states.r_pos[2, 0]
            quat_ref, R_ref = self.eps_reference(current_yaw=current_yaw)
            self.ref.reshape(self.N, self.ny)[:, 12:] = quat_ref

            l = np.vstack((self.qr_l, -self.com_const))
            u = np.vstack((self.qr_u, self.com_const))
            self.l = np.tile(l, (self.N, 1))
            self.u = np.tile(u, (self.N, 1))

            self.first_int = False

        center, r, A_hex, b_hex = self.center_optimizer.solve(self.contacts[1:, :])
        self.C_cons[12:, 15:17] = A_hex
        self.u.reshape(self.N, self.nc)[:, 12:] = b_hex

        # Apol, bpol, _ = self.polig_const.solve(self.contacts[1:, :])
        # self.C_cons[12:, 15:17] = Apol
        # self.u.reshape(self.N, self.nc)[:, 12:] = bpol

        Phi_cons = np.zeros((self.nc * self.N, self.nx + self.nu))
        aux_cons = np.zeros((self.nc, self.nu))

        Phi_cons[0:self.nc, :] = self.C_cons @ self.Aa
        aux_cons = self.C_cons @ self.Ba

        return aux_cons, Phi_cons
