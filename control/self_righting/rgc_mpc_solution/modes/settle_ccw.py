import numpy as np
import pinocchio as pin
from scipy.linalg import block_diag
from control.self_righting.rgc_mpc_solution.rgc_base_controller import BaseRGCController
from control.self_righting.rgc_mpc_solution.utils.epsilon_reference import eps_reference
from control.self_righting.rgc_mpc_solution.constraints.chebyshev_center import ChebyshevCenterSolver
from control.self_righting.rgc_mpc_solution.constraints.pyramid_friction import pyramid_friction


class SettleCCW(BaseRGCController):

    def __init__(self, robot_states, **kwargs):
        super().__init__(robot_states, **kwargs)

        self.phase = 5

        # Predic and control horizons and sampe time
        self.N = 20
        self.M = 10
        self.ts = 0.01

        # Number of states, inputs, outputs and constarints
        self.nx = 23  # CoM ang vel (3, 1), joint pos. (12, 1), CoM pos (3, 1), epsilon (4, 1), gravity (1, 1)
        self.nu = 12  # delta qr (12, 1)
        self.ny = 4  # joint pos (12, 1), body orientation (4, 1)
        self.nc = 28  # qr (12, 1), CoM projection (6, 1)

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

        # Body orientation
        self.Ca[:4, 18:22] = np.eye(4)
        
        self.Cc[0:12, 23:] = np.identity(12)

        self.Is = np.concatenate((np.identity(3), np.identity(3), np.identity(3), np.identity(3)), axis=1)

        Qeps = 0.2 * np.diag(np.array([1, 1, 1, 1]))

        Q = block_diag(Qeps)
        self.Q = block_diag(*[Q] * self.N)

        # Update control action weight matrix
        Rdqrfr = np.diag(np.array([1, 1, 1]))
        Rdqrfl = np.diag(np.array([1, 1, 1]))
        Rdqrr = np.diag(np.array([1, 1, 1]))
        Rdqrl = np.diag(np.array([1, 1, 1]))

        R = block_diag(Rdqrfr, Rdqrfl, Rdqrr, Rdqrl)
        self.R = block_diag(*[R] * self.M)

        self.com_const = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]).reshape(6, 1)

        self.Jinv = np.zeros((12, 12), dtype=np.float32)

        self.contacts = np.zeros((5, 3), dtype=np.float32)

        self.first_int = True

        self.cheby_center_solver = ChebyshevCenterSolver()

        # ----------------------------------------
        # Low-level mode controller gains
        # ----------------------------------------

        self.Kp_vec = np.ones(12) * self.kp
        self.Kd_vec = np.ones(12) * self.kd

        self.Kp_mtx = np.diag(self.Kp_vec)
        self.Kd_mtx = np.diag(self.Kd_vec)

        # ----------------------------------------
        # Constraints
        # ----------------------------------------
        foot_l = np.array([-np.inf, -np.inf, 0, 0, 10])
        foot_u = np.array([0, 0, np.inf, np.inf, 100])

        # Stack for all 2 feet
        self.f_l = np.tile(foot_l.reshape(-1, 1), (2, 1))
        self.f_u = np.tile(foot_u.reshape(-1, 1), (2, 1))

        self.L = np.zeros((6, self.nx + self.nu), dtype=np.float32)
        self.L[0:3, 3:6] = -self.kp * np.identity(3)
        self.L[0:3, 23:26] = self.kp * np.identity(3)

        self.L[3:6, 9:12] = -self.kp * np.identity(3)
        self.L[3:6, 29:32] = self.kp * np.identity(3)

    def update_model(self):
        x, y, z, w = self.rs.epsilon

        T = 0.5 * np.array([[w, z, -y], [-z, w, x], [y, -x, w], [-x, -y, -z]])

        J_com = self.pin_engine.com_jacobian()

        pivot_f = self.pin_engine.frame_pos("FL", "thigh")
        pivot_r = self.pin_engine.frame_pos("RL", "thigh")

        Jc = np.zeros((12, 12))
        Sa = np.zeros((12, 3))

        pivot_map = {
            "FR": pivot_f,
            "FL": pivot_f,
            "RR": pivot_r,
            "RL": pivot_r,
        }

        foot_positions = {}
        for i, leg in enumerate(self.leg_names):
            s = slice(3 * i, 3 * (i + 1))

            Jc[s, s] = self.pin_engine.linear_leg_jacobian(leg, "foot")

            foot = self.pin_engine.frame_pos(leg, "foot")

            foot_positions[leg] = foot

            pivot = pivot_map[leg]

            if pivot is not None:
                Sa[3 * i:3 * (i + 1), :] = self.skew_symmetric_matrix(foot - pivot)

        self.contacts[0, :] = self.pin_engine.frame_pos('FR', "foot")
        self.contacts[1, :] = self.pin_engine.frame_pos('RR', "foot")
        self.contacts[2, :] = pivot_f
        self.contacts[3, :] = pivot_r

        gamma = Jc

        gamma_a_star = np.linalg.inv(gamma) @ Sa

        self.Jinv = np.linalg.inv(Jc.T)

        I_com = self.pin_engine.centroidal_inertia()
        mass = self.total_mass

        c_pivot = (pivot_f + pivot_r) / 2
        r = self.rs.r_pos
        lever = r.flatten() - c_pivot
        S = self.skew_symmetric_matrix(lever)

        I_pivot = I_com + mass * (S.T @ S)
        I_inv = np.linalg.inv(I_pivot)

        comp_grav = S @ np.array([0, 0, mass])
        term_grav = I_inv @ comp_grav

        k3 = I_inv @ gamma_a_star.T @ self.Kp_mtx
        k4 = I_inv @ gamma_a_star.T @ self.Kd_mtx

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

        self.x = np.vstack((self.rs.omega.reshape(-1, 1), self.rs.q.reshape(-1, 1), self.rs.r_pos.reshape(-1, 1),
                            self.rs.epsilon.reshape(-1, 1), np.array([[-9.81]]), self.cs.qr.reshape(-1, 1)))

        self.L[0:3, 0:3] = -self.kd * gamma_a_star[0:3, :]
        self.L[3:6, 0:3] = -self.kd * gamma_a_star[6:9, :]

    def build_output_constraint_matrices(self):

        Phi_cons = np.zeros((self.nc * self.N, self.nx + self.nu))
        aux_cons = np.zeros((self.nc, self.nu))

        _, _, A_hex, b_hex = self.cheby_center_solver.solve(self.contacts[:, :])
        self.Cc[22:, 15:17] = A_hex

        pyramid_fric_matrix = pyramid_friction(self.contacts[0:3, :], 0.7 / np.sqrt(2))
        J = np.zeros((6, 6))
        J[0:3, 0:3] = self.Jinv[0:3, 0:3]
        J[3:6, 3:6] = self.Jinv[6:9, 6:9]
        self.Cc[12:22, :] = -pyramid_fric_matrix[0:10, 0:6] @ J @ self.L

        Phi_cons[0:self.nc, :] = self.Cc @ self.Aa
        aux_cons = self.Cc @ self.Ba

        if self.first_int:

            l = np.vstack((self.q_min.reshape(-1, 1), self.f_l.reshape(-1, 1), -self.com_const.reshape(-1, 1)))
            u = np.vstack((self.q_max.reshape(-1, 1), self.f_u.reshape(-1, 1), b_hex.reshape(-1, 1)))

            self.l = np.tile(l, (self.N, 1))
            self.u = np.tile(u, (self.N, 1))
            self.first_int = False

        return aux_cons, Phi_cons

    def build_reference(self):
        if self.first_int:
            yaw = self.rs.rpy[2]
            epsRef, _ = eps_reference(current_yaw=yaw, desired_yaw=None, current_epsilon=self.rs.epsilon)
            self.ref = np.tile(epsRef.reshape(-1, 1), (self.N, 1))



