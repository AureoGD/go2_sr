import numpy as np
from scipy.linalg import block_diag
from control.self_righting.rgc_mpc_solution.rgc_base_controller import BaseRGCController
from control.self_righting.rgc_mpc_solution.utils.epsilon_reference import eps_reference
from control.self_righting.rgc_mpc_solution.utils.geometry import plane_normal
from control.self_righting.rgc_mpc_solution.constraints.support_polytope import support_polytope
from control.self_righting.rgc_mpc_solution.constraints.pyramid_friction import pyramid_friction
from control.self_righting.rgc_mpc_solution.utils.build_gamma_star import GammaBuilder, CONFIGS


class SettleCW(BaseRGCController):

    def __init__(self, robot_states, **kwargs):
        super().__init__(robot_states, **kwargs)

        self.phase = 5

        # ----------------------------------------
        # Low-level mode controller gains
        # ----------------------------------------

        self.Kp_vec = np.ones(12) * self.kp
        self.Kd_vec = np.ones(12) * self.kd

        # Predic and control horizons and sampe time
        self.N = 20
        self.M = 20
        self.ts = 0.01

        # Number of states, inputs, outputs and constarints

        self.nx = 26  # CoM lin vel (3, 1), CoM ang vel (3, 1), joint pos. (12, 1), CoM pos (3, 1), epsilon (4, 1), gravity (1, 1)
        self.nu = 12  # delta qr (12, 1)
        self.ny = 9  # CoM position (3, 1), body orientation (4, 1), CoM linear vel. (3, 1), and CoM ang. vel. (3, 1)

        # Dynamic matrices
        self.A = np.zeros((self.nx, self.nx), dtype=np.float32)
        self.B = np.zeros((self.nx, self.nu), dtype=np.float32)

        # Aumented matrices
        self.Aa = np.zeros((self.nx + self.nu, self.nx + self.nu), dtype=np.float32)
        self.Ba = np.zeros((self.nx + self.nu, self.nu), dtype=np.float32)
        self.Cy = np.zeros((self.ny, self.nx + self.nu), dtype=np.float32)

        self.damping = 2
        self.Lambda = np.diag(self.Kp_vec / (self.Kd_vec + self.damping))
        self.Lambda[0, 0] = 0
        self.Lambda[3:6, 3:6] = 0
        self.Lambda[6, 6] = 0
        self.Lambda[9:, 9:] = 0

        # Initialize constans
        self.A[2, 25] = 1
        self.A[6:18, 6:18] = -self.Lambda
        self.A[18:21, 0:3] = np.identity(3)

        self.B[6:18, :] = self.Lambda

        self.Aa[26:, 26:] = np.identity(self.nu)
        self.Ba[26:, :] = np.identity(self.nu)

        self.Is = np.concatenate((np.identity(3), np.identity(3), np.identity(3), np.identity(3)), axis=1)

        # OUTPUT MATRIX:
        self.Cy[0, 20] = 1  # z CoM
        self.Cy[1:5, 21:25] = np.eye(4)
        self.Cy[5, 7] = 1  # FR
        self.Cy[6, 13] = 1  # RR
        self.Cy[7, 14] = 1
        self.Cy[8, 8] = 1

        # Constraints
        self.i_qr = slice(0, 12)
        self.i_com = slice(self.i_qr.stop, self.i_qr.stop + 4)
        self.i_xi = slice(self.i_com.stop, self.i_com.stop + 4)
        self.nch = self.i_xi.stop

        self.Cch = np.zeros((self.nch, self.nx + self.nu), dtype=np.float32)

        self.Cch[self.i_qr, self.nx:] = np.identity(12)

        self.gamma_builder = GammaBuilder(self.pin_engine, CONFIGS["settle_cw"], self.Kp_vec, self.Kd_vec, 2)

        # ----------------------------------------
        # Weights
        # ----------------------------------------
        Qeps = 1.75 * np.diag(np.array([1, 1, 1, 1]))
        Qq = 0.2 * np.diag(np.array([1, 1, 1, 1]))
        Qz = 0.75 * np.diag(np.array([1]))

        Q = block_diag(Qz, Qeps, Qq)
        self.Q = block_diag(*[Q] * self.N)

        # Update control action weight matrix
        Rp = np.diag(np.array([1, 20, 20]))
        Rs = np.diag(np.array([1, 1, 1]))

        R = block_diag(Rp, Rs, Rp, Rs)
        self.R = block_diag(*[R] * self.M)

        self.R[(self.M - 1) * self.nu:self.M * self.nu, (self.M - 1) * self.nu:self.M * self.nu] *= 10

        # ----------------------------------------
        # Constraints
        # ----------------------------------------
        foot_l1 = np.array([-np.inf, -np.inf, 0, 0, 30])
        foot_u1 = np.array([0, 0, np.inf, np.inf, 75])

        foot_l2 = np.array([-np.inf, -np.inf, -np.inf, -np.inf, 30])
        foot_u2 = np.array([np.inf, np.inf, np.inf, np.inf, 75])

        self.f_l = np.vstack(
            (foot_l2.reshape(-1, 1), foot_l1.reshape(-1, 1), foot_l2.reshape(-1, 1), foot_l1.reshape(-1, 1)))
        self.f_u = np.vstack(
            (foot_u2.reshape(-1, 1), foot_u1.reshape(-1, 1), foot_u2.reshape(-1, 1), foot_u1.reshape(-1, 1)))

        self.L = np.zeros((12, self.nx + self.nu), dtype=np.float32)
        self.L[:, 6:18] = -self.kp * np.identity(12)
        self.L[:, 26:] = self.kp * np.identity(12)

        self.Jinv = None
        self.contacts = np.zeros((4, 3), dtype=np.float32)

        self.first_int = True

        self.ncs = 20
        self.Ccs = np.zeros((self.ncs, self.nx + self.nu), dtype=np.float32)
        w_grf = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        wcs = np.vstack((0.01 * w_grf, 1 * w_grf, 0.01 * w_grf, 1 * w_grf))

        self.wcs = np.tile(wcs, (self.N, 1))

    def update_model(self):

        r = self.rs.r_pos

        x, y, z, w = self.rs.epsilon

        T = 0.5 * np.array([[w, z, -y], [-z, w, x], [y, -x, w], [-x, -y, -z]])

        I = self.pin_engine.centroidal_inertia()
        Iinv = np.linalg.inv(I)

        gamma_l_star, gamma_a_star, _, Sa, Jc = self.gamma_builder.build(r, use_gamma_e=False)

        self.contacts[0, :] = self.pin_engine.frame_pos("FR", "thigh")
        self.contacts[1, :] = self.pin_engine.frame_pos("FL", "foot")
        self.contacts[2, :] = self.pin_engine.frame_pos("RR", "thigh")
        self.contacts[3, :] = self.pin_engine.frame_pos("RL", "foot")

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

        # dr, omega, q, r, eps, g, qr
        self.x = np.vstack(
            (self.rs.r_vel.reshape(-1, 1), self.rs.omega.reshape(-1, 1), self.rs.q.reshape(-1, 1),
             self.rs.r_pos.reshape(-1, 1), self.rs.epsilon.reshape(-1,
                                                                   1), np.array([[-9.81]]), self.cs.qr.reshape(-1, 1)))

        self.L[:, 0:3] = -self.kd * gamma_l_star
        self.L[:, 3:6] = self.kd * gamma_a_star

    def build_hard_constraint_matrices(self):

        Phi_cons = np.zeros((self.nch * self.N, self.nx + self.nu))
        aux_cons = np.zeros((self.nch, self.nu))

        A, bu, bl = support_polytope(self.contacts)
        self.Cch[self.i_com, 18:20] = A

        p_axis = np.sum(self.contacts[:, 2]) / 4
        z = float(self.rs.r_pos[2] - p_axis)
        omega_0 = np.sqrt(9.81 / max(z, 1e-6))
        # self.Cch[self.i_xi, 0:2] = A @ self.A[15:17, 0:2] / omega_0
        self.Cch[self.i_xi, 0:2] = A / omega_0
        self.Cch[self.i_xi, 18:20] = A

        l = np.vstack((self.q_min.reshape(-1, 1), bl.reshape(-1, 1), bl.reshape(-1, 1)))
        u = np.vstack((self.q_max.reshape(-1, 1), bu.reshape(-1, 1), bu.reshape(-1, 1)))

        self.lch = np.tile(l, (self.N, 1))
        self.uch = np.tile(u, (self.N, 1))

        Phi_cons[0:self.nch, :] = self.Cch @ self.Aa
        aux_cons = self.Cch @ self.Ba

        return aux_cons, Phi_cons

    def build_soft_constraint_matrices(self):
        pyramid_fric_matrix, n_l, t1_l, t2_l = pyramid_friction(self.contacts, 0.7 / np.sqrt(2))

        self.Ccs[:, :] = -pyramid_fric_matrix @ self.Jinv @ self.L

        l = np.vstack((self.f_l.reshape(-1, 1)))
        u = np.vstack((self.f_u.reshape(-1, 1)))
        self.lcs = np.tile(l, (self.N, 1))
        self.ucs = np.tile(u, (self.N, 1))

        Phi_cs = np.zeros((self.ncs * self.N, self.nx + self.nu))
        Phi_cs[0:self.ncs, :] = self.Ccs @ self.Aa
        aux_cs = self.Ccs @ self.Ba

        return aux_cs, Phi_cs

    def build_reference(self):
        if self.first_int:
            yaw = self.rs.rpy[2]
            n, _ = plane_normal(self.contacts)
            self.task_state.plane_normal = n
            epsRef, _ = eps_reference(current_yaw=yaw, plane_normal=n, current_epsilon=self.rs.epsilon)
            ref = np.vstack((np.array([0]), epsRef.reshape(-1, 1), np.array([2.6]), np.array([2.6]), np.array([-2.6]),
                             np.array([-2.6])))

            self.ref = np.tile(ref, (self.N, 1))
