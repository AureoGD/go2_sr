import numpy as np
import pinocchio as pin
from scipy.linalg import block_diag
from control.self_righting.rgc_mpc_solution.rgc_base_controller import BaseRGCController
from control.self_righting.rgc_mpc_solution.utils.epsilon_reference import eps_reference
from control.self_righting.rgc_mpc_solution.constraints.pyramid_friction import pyramid_friction
from control.self_righting.rgc_mpc_solution.constraints.chebyshev_center import ChebyshevCenterSolver


class ProneCW(BaseRGCController):

    def __init__(self, robot_states, **kwargs):
        super().__init__(robot_states, **kwargs)

        self.action_group = 5

        # Predic and control horizons and sampe time
        self.N = 20
        self.M = 10
        self.ts = 0.01

        # Number of states, inputs, outputs and constarints
        self.nx = 26
        self.nu = 12
        self.ny = 8
        self.nc = 28

        # Dynamic matrices
        self.A = np.zeros((self.nx, self.nx), dtype=np.float32)
        self.B = np.zeros((self.nx, self.nu), dtype=np.float32)

        self.A[18:21, 0:3] = np.identity(3)
        self.A[2, 25] = 1

        # Aumented matrices
        self.Aa = np.zeros((self.nx + self.nu, self.nx + self.nu), dtype=np.float32)
        self.Ba = np.zeros((self.nx + self.nu, self.nu), dtype=np.float32)
        self.Ca = np.zeros((self.ny, self.nx + self.nu), dtype=np.float32)

        # Constraint matrix
        self.Cc = np.zeros((self.nc, self.nx + self.nu), dtype=np.float32)

        # Initialize constans
        self.Aa[self.nx:, self.nx:] = np.identity(self.nu)
        self.Ba[self.nx:, :] = np.identity(self.nu)

        self.Ca[0:2, 7:9] = np.eye(2)  # joint pos
        self.Ca[2:4, 13:15] = np.eye(2)  # joint pos
        self.Ca[4:, 21:25] = np.eye(4)  # body orientation

        self.Cc[0:12, 26:] = np.identity(12)  # qr constraints

        self.contacts = np.zeros((4, 3), dtype=np.float32)

        Qq = 1 * np.diag([1, 1])
        Qeps = 2 * np.diag(np.array([1, 1, 1, 1]))  # Quaternions

        Q = block_diag(Qq, Qq, Qeps)

        self.Q = block_diag(*[Q] * self.N)

        # References
        self.qr = np.array([1.4, -2.7, 1.4, -2.7]).reshape(4, 1)

        # Update control action weight matrix
        Rdqfr = 300 * np.diag(np.array([1, 1, 1]))
        Rdqfl = 1 * np.diag(np.array([1, 1, 1]))
        Rdqrr = 300 * np.diag(np.array([1, 1, 1]))
        Rdqrl = 1 * np.diag(np.array([1, 1, 1]))

        R = block_diag(Rdqfr, Rdqfl, Rdqrr, Rdqrl)
        self.R = block_diag(*[R] * self.M)

        self.cheby_center_solver = ChebyshevCenterSolver()
        self.com_const = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]).reshape(6, 1)

        # ----------------------------------------
        # Constraints
        # ----------------------------------------
        foot_l = np.array([-np.inf, -np.inf, 0, 0, 10])
        foot_u = np.array([0, 0, np.inf, np.inf, 100])

        # Stack for all 4 feet
        self.f_l = np.tile(foot_l.reshape(-1, 1), (2, 1))
        self.f_u = np.tile(foot_u.reshape(-1, 1), (2, 1))

        self.L = np.zeros((12, self.nx + self.nu), dtype=np.float32)
        self.L[:, 6:18] = -self.kp * np.identity(12)
        self.L[:, 26:] = self.kp * np.identity(12)

        self.Jinv = np.zeros((12, 12), dtype=np.float32)

        self.first_int = True

        self.Kp_vec = np.ones(12) * self.kp
        self.Kd_vec = np.ones(12) * self.kd

        self.Is = np.concatenate((np.identity(3), np.identity(3), np.identity(3), np.identity(3)), axis=1)

        self.contacts_cons = np.zeros((4, 3), dtype=np.float32)

    def update_model(self):
        r = self.rs.r_pos

        x, y, z, w = self.rs.epsilon

        T = 0.5 * np.array([[w, z, -y], [-z, w, x], [y, -x, w], [-x, -y, -z]])

        I = self.pin_engine.centroidal_inertia()
        Iinv = np.linalg.inv(I)

        J_com = self.pin_engine.com_jacobian()

        J_com_stacked = np.vstack([J_com, J_com, J_com, J_com])

        Jc = np.zeros((12, 12), dtype=np.float32)
        Sa = np.zeros((12, 3), dtype=np.float32)

        pivot_link = {
            "FR": 'foot',
            "FL": 'foot',
            "RR": 'foot',
            "RL": 'foot',
        }

        cons_link = {
            "FR": 'hip',
            "FL": 'foot',
            "RR": 'hip',
            "RL": 'foot',
        }

        for i, leg in enumerate(self.leg_names):

            s = slice(3 * i, 3 * (i + 1))

            Jc[s, s] = self.pin_engine.linear_leg_jacobian(leg, pivot_link[leg])

            contact_pos = self.pin_engine.frame_pos(leg, pivot_link[leg])

            Sa[i * 3:(i + 1) * 3, :] = self.skew_symmetric_matrix(contact_pos - r.flatten())

            self.contacts[i, :] = contact_pos

            self.contacts_cons[i, :] = self.pin_engine.frame_pos(leg, cons_link[leg])

        Gamma = J_com_stacked - Jc

        gamma_inv = np.linalg.pinv(Gamma)

        gamma_l_star = gamma_inv @ self.Is.T
        gamma_a_star = gamma_inv @ Sa

        self.Jinv = (np.linalg.pinv(Jc.T))

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

        # dr, omega, q, r, eps, g, qr
        self.x = np.vstack(
            (self.rs.r_vel.reshape(-1, 1), self.rs.omega.reshape(-1, 1), self.rs.q.reshape(-1, 1),
             self.rs.r_pos.reshape(-1, 1), self.rs.epsilon.reshape(-1,
                                                                   1), np.array([[-9.81]]), self.cs.qr.reshape(-1, 1)))

        self.L[:, 0:3] = -self.kd * gamma_l_star
        self.L[:, 3:6] = self.kd * gamma_a_star

    def build_constraint_matrices(self):

        if self.first_int:
            c, r, A_hex, b_hex = self.cheby_center_solver.solve(self.contacts_cons[:, :])
            self.Cc[22:, 18:20] = A_hex

            l = np.vstack((self.q_min.reshape(-1, 1), self.f_l.reshape(-1, 1), -self.com_const.reshape(-1, 1)))
            u = np.vstack((self.q_max.reshape(-1, 1), self.f_u.reshape(-1, 1), b_hex.reshape(-1, 1)))

            self.l = np.tile(l, (self.N, 1))
            self.u = np.tile(u, (self.N, 1))
            self.first_int = False

        Phi_cons = np.zeros((self.nc * self.N, self.nx + self.nu))
        aux_cons = np.zeros((self.nc, self.nu))

        pyramid_fric_matrix = pyramid_friction(self.contacts, 0.7 / np.sqrt(2))
        aux = -pyramid_fric_matrix @ self.Jinv @ self.L
        self.Cc[12:17, :] = aux[5:10, :]
        self.Cc[17:22, :] = aux[15:20, :]

        Phi_cons[0:self.nc, :] = self.Cc @ self.Aa
        aux_cons = self.Cc @ self.Ba

        return aux_cons, Phi_cons

    def build_reference(self):
        if self.first_int:
            yaw = self.rs.rpy[2]
            epsRef, _ = eps_reference(current_yaw=yaw, desired_yaw=None)
            epsRef = epsRef.reshape(4, 1)

            self.qr[0:2] = self.rs.q[1:3].reshape(2, 1)
            self.qr[2:] = self.rs.q[7:9].reshape(2, 1)

            ref = np.vstack((self.qr.reshape(-1, 1), epsRef))

            self.ref = np.tile(ref, (self.N, 1))
