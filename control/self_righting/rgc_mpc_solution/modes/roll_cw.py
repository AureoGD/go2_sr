import numpy as np
import pinocchio as pin
from scipy.linalg import block_diag
from control.self_righting.rgc_mpc_solution.rgc_base_controller import BaseRGCController
from control.self_righting.rgc_mpc_solution.utils.epsilon_reference import eps_reference
from control.self_righting.rgc_mpc_solution.constraints.chebyshev_center import ChebyshevCenterSolver


class RollCW(BaseRGCController):

    def __init__(self, robot_states, **kwargs):
        super().__init__(robot_states, **kwargs)

        self.phase = 3

        # Predic and control horizons and sampe time
        self.N = 20
        self.M = 10
        self.ts = 0.01

        # Number of states, inputs, outputs and constarints
        self.nx = 23  # CoM ang vel (3, 1), joint pos. (12, 1), CoM pos (3, 1), epsilon (4, 1), gravity (1, 1)
        self.nu = 12  # delta qr (12, 1)
        self.ny = 16  # joint pos (12, 1), body orientation (4, 1)
        self.nc = 30  # qr (12, 1), CoM projection (6, 1) + dqr (12, 1)

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
        self.Ca[:12, 3:15] = np.eye(12)
        self.Ca[12:, 18:22] = np.eye(4)

        self.Cc[0:12, 23:] = np.identity(12)

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
        self.Kd_vec[3:6] = self.kd / 10.0

        self.Kp_mtx = np.diag(self.Kp_vec)
        self.Kd_mtx = np.diag(self.Kd_vec)

        self.Cc[18:, 3:15] = -self.Kp_mtx
        self.Cc[18:, 23:] = self.Kp_mtx

    def update_model(self):
        x, y, z, w = self.rs.epsilon

        T = 0.5 * np.array([[w, z, -y], [-z, w, x], [y, -x, w], [-x, -y, -z]])

        J_com = self.pin_engine.com_jacobian()

        pivot_fr = self.pin_engine.frame_pos("FR", "hip")
        pivot_rr = self.pin_engine.frame_pos("RR", "hip")

        mean_pivot = 0.5 * (pivot_fr + pivot_rr)

        Jc = np.zeros((12, 12))
        Sa = np.zeros((12, 3))

        pivot_map = {
            "FR": pivot_fr,
            "FL": None,  # <-inative
            "RR": pivot_rr,
            "RL": mean_pivot,
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

        self.contacts[0, :] = pivot_rr
        self.contacts[1, :] = foot_positions['RR']
        self.contacts[2, :] = foot_positions['FR']
        self.contacts[3, :] = pivot_fr
        self.contacts[4, :] = foot_positions['RL']

        gamma = Jc
        gamma[3:6, :] = np.hstack((np.zeros((3, 3)), (np.eye(3)), np.zeros((3, 6))))

        gamma_a_star = np.linalg.inv(gamma) @ Sa

        self.Jinv = np.linalg.inv(Jc.T)

        I_com = self.pin_engine.centroidal_inertia()
        mass = self.total_mass

        c_pivot = (pivot_fr + pivot_rr) / 2
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

        self.Cc[18:, 0:3] = -self.Kd_mtx @ gamma_a_star

    def build_output_constraint_matrices(self):

        Phi_cons = np.zeros((self.nc * self.N, self.nx + self.nu))
        aux_cons = np.zeros((self.nc, self.nu))

        if self.first_int:
            _, _, A_hex, b_hex = self.cheby_center_solver.solve(self.contacts[:, :])
            self.Cc[12:18, 15:17] = A_hex
            l = np.vstack((self.q_min.reshape(-1, 1), -self.com_const.reshape(-1, 1), self.tau_min.reshape(-1, 1)))
            u = np.vstack((self.q_max.reshape(-1, 1), b_hex.reshape(-1, 1), self.tau_max.reshape(-1, 1)))
            self.l = np.tile(l, (self.N, 1))
            self.u = np.tile(u, (self.N, 1))

            self.first_int = False

        Phi_cons[0:self.nc, :] = self.Cc @ self.Aa
        aux_cons = self.Cc @ self.Ba

        return aux_cons, Phi_cons

    # def build_input_constraint_matrices(self):
    #     if self.G_cu is None:
    #         G_cu = np.eye(12)
    #         self.G_cu = block_diag(*[G_cu] * self.M)
    #         l = (self.tau_min / self.Kp_vec).reshape(12, 1)
    #         u = (self.tau_max / self.Kp_vec).reshape(12, 1)
    #         self.lu = np.tile(l, (self.M, 1))
    #         self.uu = np.tile(u, (self.M, 1))

    #     return self.G_cu, self.Phi_cu

    def build_reference(self):
        if self.first_int:
            yaw = self.rs.rpy[2]
            epsRef, _ = eps_reference(current_yaw=yaw, desired_yaw=None)

            self.ref.reshape(self.N, self.ny)[:, 12:] = epsRef.reshape(1, 4)
