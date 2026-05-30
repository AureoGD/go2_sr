import numpy as np
import pinocchio as pin
from scipy.linalg import block_diag
from control.self_righting.rgc_mpc_solution.rgc_base_controller import BaseRGCController
from control.self_righting.rgc_mpc_solution.utils.epsilon_reference import eps_reference
from control.self_righting.rgc_mpc_solution.constraints.pyramid_friction import pyramid_friction


class ProneCW(BaseRGCController):

    def __init__(self, robot_states, **kwargs):
        super().__init__(robot_states, **kwargs)

        self.action_group = 5

        # Predic and control horizons and sampe time
        self.N = 20
        self.M = 10
        self.ts = 0.01

        # Number of states, inputs, outputs and constarints
        self.nx = 23
        self.nu = 12
        self.ny = 12
        self.nc = 22

        # Dynamic matrices
        self.A = np.zeros((self.nx, self.nx), dtype=np.float32)
        self.B = np.zeros((self.nx, self.nu), dtype=np.float32)

        # Aumented matrices
        self.Aa = np.zeros((self.nx + self.nu, self.nx + self.nu), dtype=np.float32)
        self.Ba = np.zeros((self.nx + self.nu, self.nu), dtype=np.float32)
        self.Ca = np.zeros((self.ny, self.nx + self.nu), dtype=np.float32)

        # Constraint matrix
        self.Cc = np.zeros((self.nc, self.nx + self.nu), dtype=np.float32)

        # Initialize constans
        self.Aa[self.nx:, self.nx:] = np.identity(self.nu)
        self.Ba[self.nx:, :] = np.identity(self.nu)

        self.Ca[:12, 3:15] = np.eye(12)  # joint pos

        # self.Ca[0:, 18:22] = np.eye(4)  # body orientation

        self.Cc[0:12, 23:] = np.identity(12)  # qr constraints

        self.contacts = np.zeros((4, 3), dtype=np.float32)

        Qq = 1 * np.diag([1, 1, 1])
        # Qeps = 0.01 * np.diag(np.array([1, 1, 1, 1]))  # Quaternions

        Q = block_diag(Qq, Qq, Qq, Qq)

        self.Q = block_diag(*[Q] * self.N)

        # References
        self.qr = np.array([0.00, 1.4, -2.7, -0.00, 1.4, -2.7, 0.00, 1.40, -2.7, -0.0, 1.4, -2.7]).reshape(12, 1)

        # Update control action weight matrix
        Rdqfr = 10 * np.diag(np.array([1, 1, 1]))
        Rdqfl = 10 * np.diag(np.array([1, 1, 1]))
        Rdqrr = 10 * np.diag(np.array([1, 1, 1]))
        Rdqrl = 10 * np.diag(np.array([1, 1, 1]))

        R = block_diag(Rdqfr, Rdqfl, Rdqrr, Rdqrl)
        self.R = block_diag(*[R] * self.M)

        # ----------------------------------------
        # Constraints
        # ----------------------------------------
        foot_l = np.array([-np.inf, -np.inf, 0, 0, 30])
        foot_u = np.array([0, 0, np.inf, np.inf, 100])

        # Stack for all 4 feet
        self.f_l = np.tile(foot_l.reshape(-1, 1), (2, 1))
        self.f_u = np.tile(foot_u.reshape(-1, 1), (2, 1))

        self.L = np.zeros((6, self.nx + self.nu), dtype=np.float32)
        self.L[0:3, 6:9] = -self.kp * np.identity(3)
        self.L[3:6, 12:15] = -self.kp * np.identity(3)

        self.L[0:3, 26:29] = self.kp * np.identity(3)
        self.L[3:6, 32:35] = self.kp * np.identity(3)

        self.Jinv = np.zeros((12, 12), dtype=np.float32)

        self.first_int = True

        self.Kp_vec = np.ones(12) * self.kp
        self.Kd_vec = np.ones(12) * self.kd

    def update_model(self):

        J_com = self.pin_engine.com_jacobian()

        x, y, z, w = self.rs.epsilon
        T = 0.5 * np.array([[w, z, -y], [-z, w, x], [y, -x, w], [-x, -y, -z]])

        pivot_fr = self.pin_engine.frame_pos("FR", "hip")
        pivot_rr = self.pin_engine.frame_pos("RR", "hip")

        mean_pivot = 0.5 * (pivot_fr + pivot_rr)

        Jc = np.zeros((12, 12))
        Sa = np.zeros((12, 3))

        pivot_map_pos = {
            "FR": pivot_fr,
            "FL": pivot_fr,  # <-swing leg
            "RR": pivot_fr,
            "RL": pivot_fr,  # <-swing leg
        }

        foot_positions = {}
        for i, leg in enumerate(self.leg_names):
            s = slice(3 * i, 3 * (i + 1))

            Jc[s, s] = self.pin_engine.linear_leg_jacobian(leg, "foot")

            foot = self.pin_engine.frame_pos(leg, "foot")

            foot_positions[leg] = foot

            pivot = pivot_map_pos[leg]

            if pivot is not None:
                Sa[3 * i:3 * (i + 1), :] = self.skew_symmetric_matrix(foot - pivot)

        self.contacts[0, :] = foot_positions['FR']
        self.contacts[1, :] = foot_positions['FL']
        self.contacts[2, :] = foot_positions['RR']
        self.contacts[3, :] = foot_positions['RL']

        gamma = Jc.copy()
        inv_gamma = np.linalg.inv(gamma)

        gamma_a_star = inv_gamma @ Sa

        self.Jinv = np.linalg.inv(gamma.T)

        I_com = self.pin_engine.centroidal_inertia()
        mass = self.total_mass

        r = self.rs.r_pos
        lever = r.flatten() - mean_pivot
        S = self.skew_symmetric_matrix(lever)

        I_pivot = I_com + mass * (S.T @ S)
        I_inv = np.linalg.inv(I_pivot)

        comp_grav = S @ np.array([0, 0, mass])
        term_grav = I_inv @ comp_grav

        k1 = self.kp
        k2 = self.kd * gamma_a_star

        k3 = I_inv @ Sa.T @ self.Jinv

        self.A[0:3, 0:3] = -k3 @ k2
        self.A[0:3, 3:15] = -k3 * k1
        self.A[0:3, 22] = term_grav

        self.A[3:15, 0:3] = gamma_a_star

        self.A[15:18, 0:3] = J_com @ gamma_a_star

        self.A[18:22, 0:3] = T.reshape(4, 3)

        self.B[0:3, :] = k3 * k1

        self.Aa[0:self.nx, 0:self.nx] = np.identity(self.nx) + self.ts * self.A
        self.Aa[0:self.nx, self.nx:] = self.ts * self.B

        self.Ba[0:self.nx, :] = self.ts * self.B

        self.x = np.vstack((self.rs.omega.reshape(-1, 1), self.rs.q.reshape(-1, 1), self.rs.r_pos.reshape(-1, 1),
                            self.rs.epsilon.reshape(-1, 1), -9.81, self.cs.qr.reshape(-1, 1)))

        self.L[0:3, 0:3] = self.kd * gamma_a_star[3:6, :]
        self.L[3:6, 0:3] = self.kd * gamma_a_star[9:12, :]

    def build_constraint_matrices(self):

        if self.first_int:
            l = np.vstack((self.q_min.reshape(-1, 1), self.f_l.reshape(-1, 1)))
            u = np.vstack((self.q_max.reshape(-1, 1), self.f_u.reshape(-1, 1)))

            self.l = np.tile(l, (self.N, 1))
            self.u = np.tile(u, (self.N, 1))
            self.first_int = False

        Phi_cons = np.zeros((self.nc * self.N, self.nx + self.nu))
        aux_cons = np.zeros((self.nc, self.nu))

        pyramid_fric_matrix = pyramid_friction(self.contacts, 0.7 / np.sqrt(2))
        self.Cc[12:17, :] = -pyramid_fric_matrix[5:10, 3:6] @ self.Jinv[3:6, 3:6] @ self.L[0:3, :]
        self.Cc[17:, :] = -pyramid_fric_matrix[15:20, 9:12] @ self.Jinv[9:12, 9:12] @ self.L[3:6, :]

        Phi_cons[0:self.nc, :] = self.Cc @ self.Aa
        aux_cons = self.Cc @ self.Ba

        return aux_cons, Phi_cons

    def build_reference(self):
        if self.first_int:
            # yaw = self.rs.rpy[2]
            # epsRef, _ = eps_reference(current_yaw=yaw, desired_yaw=None)
            # epsRef = epsRef.reshape(4, 1)

            ref = np.vstack((self.qr.reshape(-1, 1)))
            self.ref = np.tile(ref, (self.N, 1))
