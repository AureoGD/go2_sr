import numpy as np
import pinocchio as pin
from scipy.linalg import block_diag
from control.self_righting.rgc_mpc_solution.utils.swing_planner import SwingFootPlanner

from control.self_righting.rgc_mpc_solution.utils.geometry import plane_normal
from control.self_righting.rgc_mpc_solution.utils.epsilon_reference import eps_reference

from control.self_righting.rgc_mpc_solution.rgc_base_controller import BaseRGCController
from control.self_righting.rgc_mpc_solution.constraints.chebyshev_center import ChebyshevCenterSolver
from control.self_righting.rgc_mpc_solution.constraints.capture_point import CapturePointConstraint
from control.self_righting.rgc_mpc_solution.utils.build_gamma_star import GammaBuilder, CONFIGS


class SwingLegCCW(BaseRGCController):

    def __init__(self, robot_states, **kwargs):
        super().__init__(robot_states, **kwargs)

        self.phase = 4

        # Predic and control horizons and sampe time
        self.N = 20
        self.M = 10
        self.ts = 0.01

        # Number of states, inputs, outputs and constarints
        self.nx = 32
        self.nu = 12  # delta qr (12, 1)
        self.ny = 12  # epsilon (4, 1), FL foot (3, 1), RL foot (3, 1)

        # Constraints slices
        self.i_qr = slice(0, 12)
        self.i_tau = slice(self.i_qr.stop, self.i_qr.stop + 12)
        self.i_com = slice(self.i_tau.stop, self.i_tau.stop + 5)
        self.nch = self.i_tau.stop

        # ----------------------------------------
        # Low-level mode controller gains
        # ----------------------------------------

        self.Kp_vec = np.ones(12) * self.kp
        self.Kd_vec = np.ones(12) * self.kd
        self.Kd_vec[0:3] = self.kd / 10.0
        self.Kd_vec[6:9] = self.kd / 10.0

        self.Kp_mtx = np.diag(self.Kp_vec)
        self.Kd_mtx = np.diag(self.Kd_vec)

        # Dynamic matrices
        self.A = np.zeros((self.nx, self.nx), dtype=np.float32)
        self.B = np.zeros((self.nx, self.nu), dtype=np.float32)

        self.damping = 2
        self.Lambda = np.diag(self.Kp_vec / (self.Kd_vec + self.damping))
        self.Lambda[3:6, 3:6] = 0
        self.Lambda[9:12, 9:12] = 0

        self.A[6:18, 6:18] = -self.Lambda
        self.A[18:21, 0:3] = np.eye(3)
        self.A[2, 25] = 1

        self.B[6:18, :] = self.Lambda

        # Aumented matrices
        self.Aa = np.zeros((self.nx + self.nu, self.nx + self.nu), dtype=np.float32)
        self.Ba = np.zeros((self.nx + self.nu, self.nu), dtype=np.float32)
        self.Cy = np.zeros((self.ny, self.nx + self.nu), dtype=np.float32)

        # Initialize constans
        self.Aa[self.nx:, self.nx:] = np.identity(self.nu)
        self.Ba[self.nx:, :] = np.identity(self.nu)

        # Constraint matrix
        self.Cch = np.zeros((self.nch, self.nx + self.nu), dtype=np.float32)

        self.Cy[0:4, 21:25] = np.eye(4)  # Quaternions
        self.Cy[4, 9] = 1
        self.Cy[5, 15] = 1
        self.Cy[6:, 26:32] = np.eye(6)  # FL foot & RL foot

        self.gamma_builder = GammaBuilder(self.pin_engine, CONFIGS["swing_ccw"], self.Kp_vec, self.Kd_vec, 2)

        # ----------------------------------------
        # Weights
        # ----------------------------------------
        Qeps = 0.5 * np.diag(np.array([1, 1, 1, 1]))  # Quaternions
        Q_pf = 2.5 * np.diag(np.array([1, 1, 1]))  # FL foot (P1)
        Q_pr = 2.5 * np.diag(np.array([1, 1, 1]))  # RL foot (P2)
        Q_q = np.diag(np.array([1]))

        Q = block_diag(Qeps, Q_q, Q_q, Q_pf, Q_pr)
        self.Q = block_diag(*[Q] * self.N)

        # Update control action weight matrix
        R_p = 250 * np.diag(np.array([0.1, 1, 1]))  # original 1 1 1
        R_s = 10 * np.diag(np.array([1, 1, 1]))

        R = block_diag(R_s, R_p, R_s, R_p)
        self.R = block_diag(*[R] * self.M)

        self.q_ref = np.array(([-0.6]))

        self.Jinv = np.zeros((12, 12), dtype=np.float32)

        self.contacts = np.zeros((4, 3), dtype=np.float32)

        self.first_int = True

        self.leg_path = SwingFootPlanner(N=self.N, dt=self.ts)

        self.com_const = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]).reshape(6, 1)

        # ----------------------------------------
        # Constraints
        # ----------------------------------------
        self.capture_point = CapturePointConstraint('CCW')

        # q_r constraint
        self.Cch[self.i_qr, self.nx:] = np.identity(12)
        # tau constraint
        Kpe = self.Kp_mtx - self.Kd_mtx @ self.Lambda
        self.Cch[self.i_tau, 6:18] = -Kpe
        self.Cch[self.i_tau, self.nx:] = Kpe

        # --- soft: RL normal contact force, priced not enforced ---
        self.ncs = 1
        self.Ccs = np.zeros((self.ncs, self.nx + self.nu), dtype=np.float32)

        self.wcs = 10 * np.ones(self.ncs * self.N)

        self.Is = np.concatenate((np.zeros((3, 3)), np.identity(3), np.zeros((3, 3)), np.identity(3)), axis=1)

    def update_model(self):

        x, y, z, w = self.rs.epsilon
        T = 0.5 * np.array([[w, z, -y], [-z, w, x], [y, -x, w], [-x, -y, -z]])

        self.contacts[0, :] = self.pin_engine.frame_pos("FL", "foot")
        self.contacts[1, :] = self.pin_engine.frame_pos('FL', 'thigh')
        self.contacts[2, :] = self.pin_engine.frame_pos('RL', 'thigh')
        self.contacts[3, :] = self.pin_engine.frame_pos("RL", "foot")

        r = self.rs.r_pos.flatten()
        gl, ga, _, Sa, Jc = self.gamma_builder.build(r, use_gamma_e=False)

        self.Jinv = np.linalg.pinv(Jc.T)

        I = self.pin_engine.centroidal_inertia()
        Iinv = np.linalg.inv(I)

        k1 = (1 / self.total_mass) * self.Is @ self.Kp_mtx @ self.Jinv
        k2 = (self.kd / self.total_mass) * self.Is @ self.Kd_mtx @ self.Jinv
        k3 = Iinv @ -Sa.T @ self.Kp_mtx @ self.Jinv
        k4 = Iinv @ -Sa.T @ self.Kd_mtx @ self.Jinv

        foot_f = self.pin_engine.frame_pos('FR', 'foot')
        foot_r = self.pin_engine.frame_pos('RR', 'foot')

        J_f = self.pin_engine.linear_leg_jacobian('FR', 'foot')
        J_r = self.pin_engine.linear_leg_jacobian('RR', 'foot')

        self.A[0:3, 0:3] = k2 @ gl
        self.A[0:3, 3:6] = -k2 @ ga
        self.A[0:3, 6:18] = k1

        self.A[3:6, 0:3] = k4 @ gl
        self.A[3:6, 3:6] = -k4 @ ga
        self.A[3:6, 6:18] = k3

        self.A[6:18, 0:3] = gl
        self.A[6:18, 3:6] = -ga

        self.A[21:25, 3:6] = T.reshape(4, 3)

        self.A[26:29, 0:3] = -Jc[3:6, 3:6] @ gl[3:6, :]
        lf = -self.skew_symmetric_matrix(foot_f - self.contacts[1, :]) + Jc[3:6, 3:6] @ ga[3:6, :]
        self.A[26:29, 3:6] = lf
        self.A[26:29, 6:9] = -J_f @ self.Lambda[0:3, 0:3]

        self.A[29:32, 0:3] = -Jc[9:12, 9:12] @ gl[9:12, :]
        lr = -self.skew_symmetric_matrix(foot_r - self.contacts[2, :]) + Jc[9:12, 9:12] @ ga[9:12, :]
        self.A[29:32, 3:6] = lr
        self.A[29:32, 12:15] = -J_r @ self.Lambda[6:9, 6:9]

        self.B[0:3, 0:12] = -k1
        self.B[3:6, 0:12] = -k3
        self.B[26:29, 0:3] = J_f @ self.Lambda[0:3, 0:3]
        self.B[29:32, 6:9] = J_r @ self.Lambda[6:9, 6:9]

        self.Aa[0:self.nx, 0:self.nx] = np.identity(self.nx) + self.ts * self.A
        self.Aa[0:self.nx, self.nx:] = self.ts * self.B

        self.x = np.vstack((self.rs.r_vel.reshape(-1, 1), self.rs.omega.reshape(-1, 1), self.rs.q.reshape(-1, 1),
                            self.rs.r_pos.reshape(-1, 1), self.rs.epsilon.reshape(-1, 1), -9.81, foot_f.reshape(-1, 1),
                            foot_r.reshape(-1, 1), self.cs.qr.reshape(-1, 1)))

        self.Cch[self.i_tau, 0:3] = -self.Kd_mtx @ gl
        self.Cch[self.i_tau, 3:6] = self.Kd_mtx @ ga

    def build_hard_constraint_matrices(self):

        if self.first_int:
            l = np.vstack((self.q_min.reshape(-1, 1), self.tau_min.reshape(-1, 1)))
            u = np.vstack((self.q_max.reshape(-1, 1), self.tau_max.reshape(-1, 1)))

            self.lch = np.tile(l, (self.N, 1))
            self.uch = np.tile(u, (self.N, 1))

            self.first_int = False

        Phi_cons = np.zeros((self.nch * self.N, self.nx + self.nu))
        aux_cons = np.zeros((self.nch, self.nu))

        Phi_cons[0:self.nch, :] = self.Cch @ self.Aa
        aux_cons = self.Cch @ self.Ba

        return aux_cons, Phi_cons

    def build_soft_constraint_matrices(self):
        # Contact 1: FL thigh
        # Contcat 2: RL thigh

        r = self.x[18:21].reshape(3,)
        dr = self.x[0:3].reshape(3,)

        self.capture_point.update_geometry(self.contacts[1, :], self.contacts[2, :], r)
        coeff_rcom, coeff_drcom, l_csi = self.capture_point.constraint_row()

        xi = self.capture_point.capture_point(r, dr)
        xi = xi.reshape(3,)
        cv = self.capture_point.constraint_value(xi)
        self.capture_point.update_crossings(xi, r)

        self.Ccs[0, 0:3] = coeff_drcom
        self.Ccs[0, 18:21] = coeff_rcom

        self.lcs = np.tile(l_csi, (self.N, 1))
        self.ucs = np.tile(np.inf, (self.N, 1))

        Phi_cs = np.zeros((self.ncs * self.N, self.nx + self.nu))
        Phi_cs[0:self.ncs, :] = self.Ccs @ self.Aa
        aux_cs = self.Ccs @ self.Ba

        return aux_cs, Phi_cs

    def build_reference(self):

        sw_foot_front = self.x[26:29]
        sw_foot_rear = self.x[29:32]

        if self.first_int:
            self.leg_path.update_geometry(sw_foot_front, sw_foot_rear, self.contacts)
            ref = np.vstack((np.zeros((4, 1)), self.q_ref.reshape(1, 1), self.q_ref.reshape(1, 1), np.zeros(
                (3, 1)), np.zeros((3, 1))))
            self.ref = np.tile(ref, (self.N, 1))
        else:
            self.leg_path.update_endpoints(self.contacts)

        ref_front, err_front = self.leg_path.get_front_ref(sw_foot_front)
        ref_rear, err_rear = self.leg_path.get_rear_ref(sw_foot_rear)
        self.task_state.swing_foot_error[0:3] = err_front
        self.task_state.swing_foot_error[3:6] = err_rear

        yaw = self.rs.rpy[2]
        epsRef, _ = eps_reference(current_yaw=yaw, desired_yaw=None, current_epsilon=self.rs.epsilon)
        self.dg.eps_ref = epsRef.copy()
        epsRef = epsRef.reshape(4, 1)

        self.ref.reshape(self.N, self.ny)[:, 6:9] = ref_front
        self.ref.reshape(self.N, self.ny)[:, 9:] = ref_rear
        self.ref.reshape(self.N, self.ny)[:, 0:4] = epsRef.reshape(4,)
