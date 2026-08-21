import numpy as np
import pinocchio as pin
from scipy.linalg import block_diag
from control.self_righting.rgc_mpc_solution.rgc_base_controller import BaseRGCController
from control.self_righting.rgc_mpc_solution.utils.epsilon_reference import eps_reference
from control.self_righting.rgc_mpc_solution.constraints.chebyshev_center import ChebyshevCenterSolver
from control.self_righting.rgc_mpc_solution.constraints.pyramid_friction import pyramid_friction
from control.self_righting.rgc_mpc_solution.utils.build_gamma_star import GammaBuilder, CONFIGS
from control.self_righting.rgc_mpc_solution.constraints.capture_point import CapturePointConstraint
from control.self_righting.rgc_mpc_solution.constraints.support_polytope import support_polytope


class RollCCW(BaseRGCController):

    def __init__(self, robot_states, **kwargs):
        super().__init__(robot_states, **kwargs)

        self.phase = 3

        # Predic and control horizons and sampe time
        self.N = 20
        self.M = 10
        self.ts = 0.01

        # Number of states, inputs, outputs and constarints
        self.nx = 26  # CoM lin vel (3, 1), CoM ang vel (3, 1), joint pos. (12, 1), CoM pos (3, 1), epsilon (4, 1), gravity (1, 1)
        self.nu = 12  # delta qr (12, 1)
        self.ny = 7  # joint pos (12, 1), body orientation (4, 1)

        # Constraints slices
        self.i_qr = slice(0, 12)
        self.i_tau = slice(self.i_qr.stop, self.i_qr.stop + 12)
        self.i_com = slice(self.i_tau.stop, self.i_tau.stop + 5)
        self.nch = self.i_com.stop

        # Dynamic matrices
        self.A = np.zeros((self.nx, self.nx), dtype=np.float32)
        self.B = np.zeros((self.nx, self.nu), dtype=np.float32)

        self.A[18:21, 0:3] = np.eye(3)
        self.A[2, 25] = 1

        # Aumented matrices
        self.Aa = np.zeros((self.nx + self.nu, self.nx + self.nu), dtype=np.float32)
        self.Ba = np.zeros((self.nx + self.nu, self.nu), dtype=np.float32)
        self.Cy = np.zeros((self.ny, self.nx + self.nu), dtype=np.float32)

        # Constraint matrix
        self.Cch = np.zeros((self.nch, self.nx + self.nu), dtype=np.float32)

        # Initialize constans
        self.Aa[self.nx:, self.nx:] = np.identity(self.nu)
        self.Ba[self.nx:, :] = np.identity(self.nu)

        self.Cy[:4, 21:25] = np.eye(4)  # orientation
        self.Cy[4:7, 12:15] = np.eye(3)  # RR leg

        self.gamma_builder = GammaBuilder(self.pin_engine, CONFIGS["roll_ccw"], self.Kp_vec, self.Kd_vec, 2)

        Qq = np.diag(np.array([1, 1, 1]))
        Qeps = 4 * np.diag(np.array([1, 1, 1, 1]))

        Q = block_diag(Qeps, Qq)
        self.Q = block_diag(*[Q] * self.N)

        # References
        qr = np.array([-0.6, 3.75, -1.5]).reshape(3, 1)
        qeps = np.array([0, 0, 0, 1]).reshape(4, 1)

        ref = np.vstack((qeps, qr))

        self.ref = np.tile(ref, (self.N, 1))

        # Control action weight matrix
        R_pivot = np.diag(np.array([15, 20, 20]))
        R_fixed = np.diag(np.array([1, 1, 1]))
        R_rear = np.diag(np.array([1, 1, 1]))

        R = block_diag(R_fixed, R_pivot, R_rear, R_pivot)
        self.R = block_diag(*[R] * self.M)

        self.com_const = np.array([np.inf, np.inf, np.inf, np.inf, np.inf]).reshape(5, 1)

        self.Jinv = np.zeros((12, 12), dtype=np.float32)

        self.contacts = np.zeros((5, 3), dtype=np.float32)

        self.first_int = True

        # ----------------------------------------
        # Low-level mode controller gains
        # ----------------------------------------

        self.Kp_vec = np.ones(12) * self.kp
        self.Kd_vec = np.ones(12) * self.kd
        self.Kd_vec[0:3] = self.kd / 10.0

        self.Kp_mtx = np.diag(self.Kp_vec)
        self.Kd_mtx = np.diag(self.Kd_vec)

        # ----------------------------------------
        # Constraints
        # ----------------------------------------
        # q_r constraint
        self.Cch[self.i_qr, self.nx:] = np.identity(12)
        # tau constraint
        self.Cch[self.i_tau, 6:18] = -self.Kp_mtx
        self.Cch[self.i_tau, self.nx:] = self.Kp_mtx

        # --- soft: RL normal contact force, priced not enforced ---

        self.capture_point = CapturePointConstraint('CCW')

        self.ncs = 8
        self.Ccs = np.zeros((self.ncs, self.nx + self.nu), dtype=np.float32)

        foot_l = np.array([-np.inf, -np.inf, 0, 0, 10])
        foot_u = np.array([0, 0, np.inf, np.inf, 150])

        self.f_l = np.tile(foot_l.reshape(-1, 1), (1, 1))
        self.f_u = np.tile(foot_u.reshape(-1, 1), (1, 1))

        w_grf = np.array([1., 1., 1., 1., 1.])
        w_nf = np.array([1., 1.])
        w_pc = np.array([5])

        wcs = np.vstack((w_grf.reshape(-1, 1), w_nf.reshape(-1, 1), w_pc.reshape(-1, 1)))

        self.wcs = np.tile(wcs, (self.N, 1))

        self.Is = np.concatenate((np.zeros((3, 3)), np.identity(3), np.identity(3), np.identity(3)), axis=1)

        self.last_pk = None

    def update_model(self):
        x, y, z, w = self.rs.epsilon

        T = 0.5 * np.array([[w, z, -y], [-z, w, x], [y, -x, w], [-x, -y, -z]])

        self.contacts[0, :] = self.pin_engine.frame_pos("RL", "thigh")
        self.contacts[1, :] = self.pin_engine.frame_pos("RL", "foot")
        self.contacts[2, :] = self.pin_engine.frame_pos("FL", "foot")
        self.contacts[3, :] = self.pin_engine.frame_pos("FL", "thigh")
        self.contacts[4, :] = self.pin_engine.frame_pos("RR", "foot")

        r = self.rs.r_pos.flatten()

        gl, ga, _, Sa, Jc = self.gamma_builder.build(r, use_gamma_e=False)

        self.Jinv = np.linalg.pinv(Jc).T

        I = self.pin_engine.centroidal_inertia()
        Iinv = np.linalg.inv(I)

        k1 = (self.kp / self.total_mass) * self.Is @ self.Jinv
        k2 = (self.kd / self.total_mass) * self.Is @ self.Jinv
        k3 = self.kp * Iinv @ -Sa.T @ self.Jinv
        k4 = self.kd * Iinv @ -Sa.T @ self.Jinv

        self.A[0:3, 0:3] = k2 @ gl
        self.A[0:3, 3:6] = -k2 @ ga
        self.A[0:3, 6:18] = k1

        self.A[3:6, 0:3] = k4 @ gl
        self.A[3:6, 3:6] = -k4 @ ga
        self.A[3:6, 6:18] = k3

        self.A[6:18, 0:3] = gl
        self.A[6:18, 3:6] = -ga

        self.A[21:25, 3:6] = T.reshape(4, 3)

        self.B[0:3, 0:12] = -k1
        self.B[3:6, 0:12] = -k3

        self.Aa[0:self.nx, 0:self.nx] = np.identity(self.nx) + self.ts * self.A
        self.Aa[0:self.nx, self.nx:] = self.ts * self.B

        # dr, omega, q, r, eps, qr, g
        self.x = np.vstack(
            (self.rs.r_vel.reshape(-1, 1), self.rs.omega.reshape(-1, 1), self.rs.q.reshape(-1, 1), r.reshape(-1, 1),
             self.rs.epsilon.reshape(-1, 1), np.array([[-9.81]]), self.cs.qr.reshape(-1, 1)))

        self.Cch[self.i_tau, 0:3] = -self.Kd_mtx @ gl
        self.Cch[self.i_tau, 3:6] = self.Kd_mtx @ ga

    def build_hard_constraint_matrices(self):

        Phi_cons = np.zeros((self.nch * self.N, self.nx + self.nu))
        aux_cons = np.zeros((self.nch, self.nu))

        A, b = support_polytope(self.contacts)
        self.Cch[self.i_com, 18:20] = A
        l = np.vstack((self.q_min.reshape(-1, 1), self.tau_min.reshape(-1, 1), -self.com_const.reshape(-1, 1)))
        u = np.vstack((self.q_max.reshape(-1, 1), self.tau_max.reshape(-1, 1), b.reshape(-1, 1)))

        self.lch = np.tile(l, (self.N, 1))
        self.uch = np.tile(u, (self.N, 1))
        self.first_int = False

        Phi_cons[0:self.nch, :] = self.Cch @ self.Aa
        aux_cons = self.Cch @ self.Ba

        return aux_cons, Phi_cons

    def build_soft_constraint_matrices(self):
        # Contact 0: RL thigh
        # Contcat 3: FL thigh
        # Contcat 4: RR foot

        fric_cons_contacts = np.vstack((self.contacts[4, :], self.contacts[3, :], self.contacts[0, :]))

        pyramid_fric_matrix, n_l, t1_l, t2_l = pyramid_friction(fric_cons_contacts, 0.7 / np.sqrt(2))

        self.Ccs[0:5, :] = -pyramid_fric_matrix[0:5, 0:3] @ self.Jinv[6:9, 6:9] @ self.Cch[18:21, :]
        self.Ccs[5, :] = -n_l[1] @ (self.Jinv[3:6, 3:6] @ self.Cch[15:18, :])
        self.Ccs[6, :] = -n_l[2] @ (self.Jinv[9:12, 9:12] @ self.Cch[21:24, :])

        r = self.x[18:21].reshape(3,)
        dr = self.x[0:3].reshape(3,)

        self.capture_point.update_geometry(self.contacts[3, :], self.contacts[0, :], r)
        coeff_rcom, coeff_drcom, l_csi = self.capture_point.constraint_row()

        xi = self.capture_point.capture_point(r, dr)
        xi = xi.reshape(3,)
        cv = self.capture_point.constraint_value(xi)
        self.capture_point.update_crossings(xi, r)
        self.dg.cv = cv

        self.Ccs[7, 0:3] = coeff_drcom
        self.Ccs[7, 18:21] = coeff_rcom

        l = np.vstack((self.f_l.reshape(-1, 1), 30, 15, l_csi))
        u = np.vstack((self.f_u.reshape(-1, 1), 150, 150, np.inf))
        self.lcs = np.tile(l, (self.N, 1))
        self.ucs = np.tile(u, (self.N, 1))

        Phi_cs = np.zeros((self.ncs * self.N, self.nx + self.nu))
        Phi_cs[0:self.ncs, :] = self.Ccs @ self.Aa
        aux_cs = self.Ccs @ self.Ba

        self.task_state.cp_trig_signal = self.capture_point.cp_crossed
        self.task_state.com_trig_signal = self.capture_point.com_crossed

        return aux_cs, Phi_cs

    def build_reference(self):
        # if self.first_int:
        yaw = self.rs.rpy[2]
        epsRef, _ = eps_reference(current_yaw=yaw, desired_yaw=None, current_epsilon=self.rs.epsilon)
        self.dg.eps_ref = epsRef.copy()
        self.ref.reshape(self.N, self.ny)[:, 0:4] = epsRef.reshape(1, 4)
