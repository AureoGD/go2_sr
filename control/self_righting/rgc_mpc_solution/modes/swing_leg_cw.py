import numpy as np
import pinocchio as pin
from scipy.linalg import block_diag
from control.self_righting.rgc_mpc_solution.utils.swing_planner import SwingFootPlanner

from control.self_righting.rgc_mpc_solution.utils.geometry import plane_normal
from control.self_righting.rgc_mpc_solution.utils.epsilon_reference import eps_reference

from control.self_righting.rgc_mpc_solution.rgc_base_controller import BaseRGCController
from control.self_righting.rgc_mpc_solution.constraints.chebyshev_center import ChebyshevCenterSolver


class SwingLegCW(BaseRGCController):

    def __init__(self, robot_states, **kwargs):
        super().__init__(robot_states, **kwargs)

        self.phase = 4

        # Predic and control horizons and sampe time
        self.N = 20
        self.M = 10
        self.ts = 0.01

        # Number of states, inputs, outputs and constarints
        self.nx = 29
        self.nu = 12  # delta qr (12, 1)
        self.ny = 16  # FR q (3, 1), FL q (3, 1), epsilon (4, 1), FL foot (3, 1), RL foot (3, 1)
        self.nc = 30  # qr (12, 1), CoM projection (6, 1), dqr (12, 1)

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

        self.Ca[:3, 3:6] = np.eye(3)  # FR joints
        self.Ca[3:6, 9:12] = np.eye(3)  # RR joints
        self.Ca[6:10, 18:22] = np.eye(4)  # Quaternions
        self.Ca[10:13, 23:26] = np.eye(3)  # FL foot
        self.Ca[13:16, 26:29] = np.eye(3)  # RL foot

        self.Cc[0:12, self.nx:] = np.identity(12)

        # ----------------------------------------
        # Weights
        # ----------------------------------------

        Qr = 10 * np.diag(np.array([1, 1, 1]))  # FR and RR joints
        Qeps = 0.1 * np.diag(np.array([1, 1, 1, 1]))  # Quaternions
        Qposfl = 1.5 * np.diag(np.array([6, 6, 4]))  # FL foot (P1)
        Qposrl = 1.5 * np.diag(np.array([6, 6, 4]))  # RL foot (P2)

        Q = block_diag(Qr, Qr, Qeps, Qposfl, Qposrl)
        self.Q = block_diag(*[Q] * self.N)

        # Update control action weight matrix
        Rdqrfr = 1000 * np.diag(np.array([1.3, 1, 1])) # original 1 1 1
        Rdqrfl = 40 * np.diag(np.array([1, 1, 1]))
        Rdqrr = 1000 * np.diag(np.array([1.3, 1, 1]))
        Rdqrl = 40 * np.diag(np.array([1, 1, 1]))

        R = block_diag(Rdqrfr, Rdqrfl, Rdqrr, Rdqrl)
        self.R = block_diag(*[R] * self.M)

        # reference
        self.qr = np.array([0.4, 1.5, -2.0, 0.4, 1.5, -2.0]).reshape(6, 1)

        self.Jinv = np.zeros((12, 12), dtype=np.float32)

        self.Is = np.concatenate((np.identity(3), np.identity(3), np.identity(3), np.identity(3)), axis=1)

        self.contacts = np.zeros((4, 3), dtype=np.float32)

        self.first_int = True

        self.leg_path = SwingFootPlanner(N=self.N, dt=self.ts)

        self.cheby_center_solver = ChebyshevCenterSolver()
        self.com_const = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]).reshape(6, 1)

        # ----------------------------------------
        # Low-level mode controller gains
        # ----------------------------------------

        self.Kp_vec = np.ones(12) * self.kp
        self.Kd_vec = np.ones(12) * self.kd
        self.Kd_vec[3:6] = self.kd / 10.0
        self.Kd_vec[9:12] = self.kd / 10.0

        self.Kp_mtx = np.diag(self.Kp_vec)
        self.Kd_mtx = np.diag(self.Kd_vec)

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
            "FL": None,  # <-swing leg
            "RR": pivot_rr,
            "RL": None,  # <-swing leg
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
        self.contacts[1, :] = self.pin_engine.frame_pos('FR', 'thigh')
        self.contacts[2, :] = self.pin_engine.frame_pos('RR', 'thigh')
        self.contacts[3, :] = foot_positions['RR']

        M = self.pin_engine.actuated_mass_matrix()

        M_diag = np.maximum(np.diag(M), 1e-6)

        lambda_vec = np.sqrt(self.Kp_vec / M_diag)
        Lambda = np.diag(lambda_vec)

        Lambda_swing = np.zeros((12, 12))

        Lambda_swing[3:6, 3:6] = Lambda[3:6, 3:6]
        Lambda_swing[9:12, 9:12] = Lambda[9:12, 9:12]

        gamma = Jc.copy()
        gamma[3:6, 3:6] = np.eye(3)
        gamma[9:12, 9:12] = np.eye(3)

        inv_gamma = np.linalg.inv(gamma)

        gamma_a_star = inv_gamma @ Sa
        gamma_q_star = inv_gamma @ Lambda_swing

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

        k1 = self.Kp_mtx - self.Kd_mtx @ gamma_q_star
        k2 = self.Kd_mtx @ gamma_a_star

        k3 = I_inv @ Sa.T @ self.Jinv

        J_fl = self.pin_engine.linear_leg_jacobian('FL', 'foot')

        J_rl = self.pin_engine.linear_leg_jacobian('RL', 'foot')

        foot_fl = self.pin_engine.frame_pos('FL', 'foot')
        foot_rl = self.pin_engine.frame_pos('RL', 'foot')

        self.A[0:3, 0:3] = -k3 @ k2
        self.A[0:3, 3:15] = -k3 @ k1
        self.A[0:3, 22] = term_grav

        self.A[3:15, 0:3] = gamma_a_star
        self.A[3:15, 3:15] = -gamma_q_star

        self.A[15:18, 0:3] = J_com @ gamma_a_star
        self.A[15:18, 3:15] = -J_com @ gamma_q_star

        self.A[18:22, 0:3] = T.reshape(4, 3)

        self.A[23:26, 0:3] = self.skew_symmetric_matrix(foot_fl - mean_pivot)
        self.A[23:26, 6:9] = -J_fl @ Lambda[3:6, 3:6]

        self.A[26:29, 12:15] = -J_rl @ Lambda[9:12, 9:12]
        self.A[26:29, 0:3] = self.skew_symmetric_matrix(foot_rl - mean_pivot)

        self.B[0:3, :] = k3 @ k1

        self.B[3:15, :] = gamma_q_star

        self.B[15:18, :] = J_com @ gamma_q_star

        self.B[23:26, 3:6] = J_fl @ Lambda[3:6, 3:6]
        self.B[26:29, 9:12] = J_rl @ Lambda[9:12, 9:12]

        self.Aa[0:self.nx, 0:self.nx] = np.identity(self.nx) + self.ts * self.A
        self.Aa[0:self.nx, self.nx:] = self.ts * self.B

        self.Ba[0:self.nx, :] = self.ts * self.B

        self.x = np.vstack((self.rs.omega.reshape(-1, 1), self.rs.q.reshape(-1, 1), self.rs.r_pos.reshape(-1, 1),
                            self.rs.epsilon.reshape(-1, 1), -9.81, foot_fl.reshape(-1, 1), foot_rl.reshape(-1, 1),
                            self.cs.qr.reshape(-1, 1)))

        self.Cc[18:, 0:3] = -self.Kd_mtx @ gamma_a_star
        aux = self.Kp_mtx - self.Kd_mtx @ gamma_q_star
        self.Cc[18:, 3:15] = -aux
        self.Cc[18:, 29:] = aux

    def build_output_constraint_matrices(self):

        if self.first_int:

            _, _, A_hex, b_hex = self.cheby_center_solver.solve(
                np.vstack((self.contacts[:, :], self.x[26:29].reshape(1, 3))))
            self.Cc[12:18, 15:17] = A_hex

            l = np.vstack((self.q_min.reshape(-1, 1), -self.com_const.reshape(-1, 1), self.tau_min.reshape(-1, 1)))
            u = np.vstack((self.q_max.reshape(-1, 1), b_hex.reshape(-1, 1), self.tau_max.reshape(-1, 1)))
            self.l = np.tile(l, (self.N, 1))
            self.u = np.tile(u, (self.N, 1))

            self.first_int = False

        Phi_cons = np.zeros((self.nc * self.N, self.nx + self.nu))
        aux_cons = np.zeros((self.nc, self.nu))

        Phi_cons[0:self.nc, :] = self.Cc @ self.Aa
        aux_cons = self.Cc @ self.Ba

        return aux_cons, Phi_cons

    def build_reference(self):

        sw_foot_front = self.x[23:26]
        sw_foot_rear = self.x[26:29]

        if self.first_int:
            self.leg_path.update_geometry(sw_foot_front, sw_foot_rear, self.contacts)
            yaw = self.rs.rpy[2]
            epsRef, _ = eps_reference(current_yaw=yaw, desired_yaw=None)
            epsRef = epsRef.reshape(4, 1)
            ref = np.vstack((self.qr.reshape(-1, 1), epsRef, np.zeros((3, 1)), np.zeros((3, 1))))
            self.ref = np.tile(ref, (self.N, 1))  # (N*ny, 1)

        fl_ref = self.leg_path.get_front_ref(sw_foot_front)  # (N,3)
        rl_ref = self.leg_path.get_rear_ref(sw_foot_rear)  # (N,3)

        self.task_state.swing_foot_error[0:3] = (self.leg_path.ref_front.reshape(3, 1) - sw_foot_front).reshape(3,)
        self.task_state.swing_foot_error[3:6] = (self.leg_path.ref_rear.reshape(3, 1) - sw_foot_rear).reshape(3,)

        self.ref.reshape(self.N, self.ny)[:, 10:13] = fl_ref
        self.ref.reshape(self.N, self.ny)[:, 13:] = rl_ref
