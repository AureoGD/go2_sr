import numpy as np
import pinocchio as pin
from scipy.linalg import block_diag
from environment.strategies.rgc_mpc.base_controller import BaseRGC


class StandUpPhase(BaseRGC):
    TASK_NAME = "stand"
    TASK_LEVEL = 5

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not self.runtime:
            return

        self.N = 20
        self.M = 10
        self.ts = 0.01
        self.convergence_threshold = 0.23
        self._update_detector()

        self.nx = 26
        self.nu = 12
        self.ny = 8
        self.nc = 22  # rx, ry, GRF

        self.A = np.zeros((self.nx, self.nx), dtype=np.float32)
        self.B = np.zeros((self.nx, self.nu), dtype=np.float32)

        self.Aa = np.zeros((self.nx + self.nu, self.nx + self.nu), dtype=np.float32)
        self.Ba = np.zeros((self.nx + self.nu, self.nu), dtype=np.float32)
        self.Ca = np.zeros((self.ny, self.nx + self.nu), dtype=np.float32)

        self.C_cons = np.zeros((self.nc, self.nx + self.nu), dtype=np.float32)

        self.A[18:21, 0:3] = np.identity(3)
        self.A[2, 25] = 1

        self.Aa[26:, 26:] = np.identity(self.nu)
        self.Ba[26:, :] = np.identity(self.nu)

        # CoM z position and body orientatio
        self.Ca[0, 20] = 1
        self.Ca[1:5, 21:25] = np.identity(4)
        self.Ca[5:, 0:3] = np.identity(3)

        # rx and rz
        self.C_cons[0:2, 18:20] = np.identity(2)

        self.contacts = np.zeros((4, 3), dtype=np.float32)

        self.Is = np.concatenate((np.identity(3), np.identity(3), np.identity(3), np.identity(3)), axis=1)

        self.L = np.zeros((12, 38), dtype=np.float32)
        self.L[:, 6:18] = -self.kp * np.identity(12)
        self.L[:, 26:] = self.kp * np.identity(12)

        Qrz = np.array([10])
        Qeps = 1.5 * np.eye(4)
        Qdr = 1 * np.eye(3)
        Q = block_diag(Qrz, Qeps, Qdr)

        self.Q = block_diag(*[Q] * self.N)

        self.Q = block_diag(*[Q] * self.N)

        # Update control action weight matrix
        dqrWeight = np.array([1, 1, 1])
        Rdqr = np.diag(dqrWeight)
        R = block_diag(Rdqr, Rdqr, Rdqr, Rdqr)
        self.R = block_diag(*[R] * self.M)

        # rzRef = np.array([[0.25]]).reshape(1, 1)
        # epsRef = np.array([0, 0, 0, 1]).reshape(4, 1)
        # ref = np.vstack((rzRef, epsRef))
        # self.ref = np.tile(ref, (self.N, 1))

        # GRF vector:
        # Constraints for one foot
        foot_l = np.array([-np.inf, -np.inf, 0, 0, 30])
        foot_u = np.array([0, 0, np.inf, np.inf, 100])

        # Stack for all 4 feet
        self.f_l = np.tile(foot_l.reshape(-1, 1), (4, 1))  # Shape: (20, 1)
        self.f_u = np.tile(foot_u.reshape(-1, 1), (4, 1))  # Shape: (20, 1)

        self.Jinv = None

        # FR, FL, RR, RL
        self.leg_idx = [3, 0, 9, 6]

        self.contact_ids = [
            self.model.getFrameId('FR_foot'),
            self.model.getFrameId('FL_foot'),
            self.model.getFrameId('RR_foot'),
            self.model.getFrameId('RL_foot'),
        ]

        self.first_int = True

        self.min_obj_val = 0.0004

    def update_model(self):

        q, dq = self.ordering_joints()

        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pin.ccrba(self.model, self.data, q, dq)
        r = pin.centerOfMass(self.model, self.data, q)

        #  Evaluate the CoM velocity
        centroidal_momentum = pin.computeCentroidalMomentum(self.model, self.data, q, dq)
        dr = centroidal_momentum.linear / self.data.mass[0]

        # Save states
        self.robot_states.r_vel = dr.reshape(3, 1)
        self.robot_states.r_pos = r.reshape(3, 1)

        x, y, z, w = self.robot_states.epsilon

        T = 0.5 * np.array([[w, z, -y], [-z, w, x], [y, -x, w], [-x, -y, -z]])

        I = self.data.Ig.inertia
        Iinv = np.linalg.inv(I)

        J_com_full = pin.jacobianCenterOfMass(self.model, self.data, q, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:, 6:]

        J_com = np.hstack([
            J_com_full[:, self.leg_idx[0]:self.leg_idx[0] + 3],  # front-right leg
            J_com_full[:, self.leg_idx[1]:self.leg_idx[1] + 3],  # front-left
            J_com_full[:, self.leg_idx[2]:self.leg_idx[2] + 3],  # rear-right leg
            J_com_full[:, self.leg_idx[3]:self.leg_idx[3] + 3]  # rear-left leg
        ])

        J_com_stacked = np.vstack([J_com, J_com, J_com, J_com])

        Jc = np.zeros((12, 12), dtype=np.float32)
        gamma = np.zeros((12, 12), dtype=np.float32)
        Sa = np.zeros((12, 3), dtype=np.float32)

        for i in range(len(self.contact_ids)):
            contact_id = self.contact_ids[i]
            leg_q_start_idx = self.leg_idx[i]

            Jc_full = pin.computeFrameJacobian(self.model, self.data, q, contact_id,
                                               pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[0:3, 6 + leg_q_start_idx:9 +
                                                                                       leg_q_start_idx]

            Jc[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = Jc_full

            contact_pos = self.data.oMf[contact_id].translation
            Sa[i * 3:(i + 1) * 3, :] = self.skew_symmetric_matrix(contact_pos - r)

            # Store contact position
            self.contacts[i, :] = contact_pos

        Gamma = J_com_stacked - Jc

        gamma_inv = np.linalg.inv(Gamma)

        I_stack = np.vstack([np.eye(3), np.eye(3), np.eye(3), np.eye(3)])  # 9x3 part

        gamma_l_star = gamma_inv @ I_stack
        gamma_a_star = gamma_inv @ Sa

        self.Jinv = (np.linalg.inv(Jc.T))

        k1 = (self.kp / self.total_mass) * self.Is @ self.Jinv
        k2 = (self.kd / self.total_mass) * self.Is @ self.Jinv
        k3 = self.kp * Iinv @ -Sa.T @ self.Jinv
        k4 = self.kd * Iinv @ -Sa.T @ self.Jinv

        self.A[0:3, 0:3] = k2 @ gamma_l_star
        self.A[0:3, 3:6] = -k2 @ gamma_a_star
        self.A[0:3, 6:18] = k1

        self.A[3:6, 0:3] = k4 @ gamma_l_star
        self.A[3:6, 3:6] = -k4 @ gamma_a_star
        self.A[3:6, 6:18] = k3

        self.A[6:18, 0:3] = gamma_l_star
        self.A[6:18, 3:6] = -gamma_a_star

        self.A[21:25, 3:6] = T.reshape(4, 3)

        self.B[0:3, 0:12] = -k1
        self.B[3:6, 0:12] = -k3

        self.Aa[0:26, 0:26] = np.identity(self.nx) + self.ts * self.A
        self.Aa[0:26, 26:] = self.ts * self.B

        self.Ba[0:26, :] = self.ts * self.B

        # dr, omega, q, r, eps, qr, g
        self.x = np.vstack((self.robot_states.r_vel, self.robot_states.omega, self.robot_states.q,
                            self.robot_states.r_pos, self.robot_states.epsilon, -9.81, self.robot_states.qr))

        self.L[:, 0:3] = -self.kd * gamma_l_star
        self.L[:, 3:6] = self.kd * gamma_a_star

        # self.define_constraints_matrices(Jinv)s

    def define_constraints_matrices(self):

        Phi_cons = np.zeros((self.nc * self.N, self.nx + self.nu))
        aux_cons = np.zeros((self.nc, self.nu))

        n_fr, t1_fr, t2_fr = self.cont_surfaces(self.contacts[0, :], self.contacts[1, :], self.contacts[2, :])
        n_fl, t1_fl, t2_fl = self.cont_surfaces(self.contacts[1, :], self.contacts[2, :], self.contacts[3, :])
        n_rr, t1_rr, t2_rr = self.cont_surfaces(self.contacts[2, :], self.contacts[3, :], self.contacts[0, :])
        n_rl, t1_rl, t2_rl = self.cont_surfaces(self.contacts[3, :], self.contacts[0, :], self.contacts[1, :])

        mu = 0.9 / np.sqrt(2)

        Cf_fr = self.cf_matrix(n_fr, t1_fr, t2_fr, mu)
        Cf_fl = self.cf_matrix(n_fl, t1_fl, t2_fl, mu)
        Cf_rr = self.cf_matrix(n_rr, t1_rr, t2_rr, mu)
        Cf_rl = self.cf_matrix(n_rl, t1_rl, t2_rl, mu)

        Cf = block_diag(Cf_fl, Cf_fr, Cf_rr, Cf_rl)
        Fc_mtx = -Cf @ self.Jinv
        aux_cons[0:2, :] = self.C_cons[0:2, :] @ self.Ba
        aux_cons[2:, :] = self.kp * Fc_mtx
        self.C_cons[2:, :] = Fc_mtx @ self.L
        Phi_cons[0:self.nc, :] = self.C_cons @ self.Aa

        if self.first_int:
            yaw = self.robot_states.rpy[2, 0]

            center, radius = self.center_optimizer.solve(self.contacts)

            epsRef, _ = self.eps_reference(
                current_yaw=yaw,
                desired_yaw=None  # Keep current yaw
            )
            epsRef = epsRef.reshape(4, 1)
            rzRef = self.robot_states.r_pos[2] + np.array([[0.2]]).reshape(1, 1)
            dr_ref = np.zeros((3, 1))
            ref = np.vstack((rzRef, epsRef, dr_ref))
            self.ref = np.tile(ref, (self.N, 1))

            l, u = self.center_of_mass_constraint()
            l = np.vstack((l, self.f_l))
            u = np.vstack((u, self.f_u))
            self.l = np.tile(l, (self.N, 1))
            self.u = np.tile(u, (self.N, 1))
            self.first_int = False

        return aux_cons, Phi_cons

    def center_of_mass_constraint(self):
        l = (self.robot_states.r_pos[0:2, 0]).reshape(2, 1) - 0.1 * np.ones((2, 1))
        u = (self.robot_states.r_pos[0:2, 0]).reshape(2, 1) + 0.1 * np.ones((2, 1))

        return l, u
