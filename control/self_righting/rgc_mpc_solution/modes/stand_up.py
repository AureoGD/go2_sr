import numpy as np
import pinocchio as pin
from scipy.linalg import block_diag
from control.self_righting.rgc_mpc_solution.rgc_base_controller import BaseRGCController

from control.self_righting.rgc_mpc_solution.constraints.pyramid_friction import pyramid_friction
from control.self_righting.rgc_mpc_solution.constraints.chebyshev_center import ChebyshevCenterSolver

from control.self_righting.rgc_mpc_solution.utils.epsilon_reference import eps_reference


class StandUp(BaseRGCController):

    def __init__(self, robot_states, **kwargs):
        super().__init__(robot_states, **kwargs)

        self.action_group = 6

        # Predic and control horizons and sampe time
        self.N = 20
        self.M = 10
        self.ts = 0.01

        # Number of states, inputs, outputs and constarints

        self.nx = 26  # CoM lin vel (3, 1), CoM ang vel (3, 1), joint pos. (12, 1), CoM pos (3, 1), epsilon (4, 1), gravity (1, 1)
        self.nu = 12  # delta qr (12, 1)
        self.ny = 11  # CoM z position (1, 1), body orientation (4, 1), CoM linear vel. (3, 1), and CoM ang. vel. (3, 1)
        self.nc = 38  # ground reaction forces (20, 1), qr (12, 1), CoM projection (6, 1)

        # Dynamic matrices
        self.A = np.zeros((self.nx, self.nx), dtype=np.float32)
        self.B = np.zeros((self.nx, self.nu), dtype=np.float32)

        # Aumented matrices
        self.Aa = np.zeros((self.nx + self.nu, self.nx + self.nu), dtype=np.float32)
        self.Ba = np.zeros((self.nx + self.nu, self.nu), dtype=np.float32)
        self.Ca = np.zeros((self.ny, self.nx + self.nu), dtype=np.float32)

        # Aumented matrices
        self.Cc = np.zeros((self.nc, self.nx + self.nu), dtype=np.float32)

        # Initialize constans
        self.A[18:21, 0:3] = np.identity(3)
        self.A[2, 25] = 1

        self.Aa[26:, 26:] = np.identity(self.nu)
        self.Ba[26:, :] = np.identity(self.nu)

        # OUTPUT MATRIX:
        self.Ca[0, 20] = 1  # z com pos
        self.Ca[1:5, 21:25] = np.identity(4)  # epsilon
        self.Ca[5:8, 0:3] = np.identity(3)  # com lin vel
        self.Ca[8:, 3:6] = np.identity(3)  # com ang vel

        # Joint position constrain
        self.Cc[26:, 26:] = np.identity(12)

        # ----------------------------------------
        # Weights
        # ----------------------------------------

        # Output weight matrix
        Q_rz = np.array([5])
        Q_eps = 7.5 * np.eye(4)
        Q_dr = 1 * np.eye(3)
        Q_omega = 1 * np.eye(3)
        Q = block_diag(Q_rz, Q_eps, Q_dr, Q_omega)

        self.Q = block_diag(*[Q] * self.N)

        # Input weight matrix
        dqrWeight = np.array([1, 1, 1])
        Rdqr = np.diag(dqrWeight)
        R = 0.75 * block_diag(Rdqr, Rdqr, Rdqr, Rdqr)

        self.R = block_diag(*[R] * self.M)

        # ----------------------------------------
        # Constraints
        # ----------------------------------------
        foot_l = np.array([-np.inf, -np.inf, 0, 0, 30])
        foot_u = np.array([0, 0, np.inf, np.inf, 100])

        # Stack for all 4 feet
        self.f_l = np.tile(foot_l.reshape(-1, 1), (4, 1))
        self.f_u = np.tile(foot_u.reshape(-1, 1), (4, 1))

        self.com_const = -np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]).reshape(6, 1)

        # ----------------------------------------
        # Controller specific variables and objects
        # ----------------------------------------
        self.z_ref = np.array([[0.2]]).reshape(1, 1)

        self.L = np.zeros((12, 38), dtype=np.float32)
        self.L[:, 6:18] = -self.kp * np.identity(12)
        self.L[:, 26:] = self.kp * np.identity(12)

        self.Jinv = None
        self.contacts = np.zeros((4, 3), dtype=np.float32)
        self.Is = np.concatenate((np.identity(3), np.identity(3), np.identity(3), np.identity(3)), axis=1)

        self.cheby_center_solver = ChebyshevCenterSolver()

        self.first_cont_interation = True

        # ----------------------------------------
        # Low-level mode controller gains
        # ----------------------------------------

        self.Kp_vec = np.ones(12) * self.kp
        self.Kd_vec = np.ones(12) * self.kd

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

        for i, leg in enumerate(self.leg_names):

            Jc_full = self.pin_engine.frame_jacobian(leg, "foot")[0:3, i * 3:(i + 1) * 3]

            Jc[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = Jc_full

            contact_pos = self.pin_engine.frame_pos(leg, "foot")
            Sa[i * 3:(i + 1) * 3, :] = self.skew_symmetric_matrix(contact_pos - r.flatten())

            self.contacts[i, :] = contact_pos

        Gamma = J_com_stacked - Jc

        gamma_inv = np.linalg.inv(Gamma)

        gamma_l_star = gamma_inv @ self.Is.T
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
        self.x = np.vstack(
            (self.rs.r_vel.reshape(-1, 1), self.rs.omega.reshape(-1, 1), self.rs.q.reshape(-1, 1),
             self.rs.r_pos.reshape(-1, 1), self.rs.epsilon.reshape(-1,
                                                                   1), np.array([[-9.81]]), self.cs.qr.reshape(-1, 1)))

        self.L[:, 0:3] = -self.kd * gamma_l_star
        self.L[:, 3:6] = self.kd * gamma_a_star

    def build_constraint_matrices(self):

        Phi_cons = np.zeros((self.nc * self.N, self.nx + self.nu))
        aux_cons = np.zeros((self.nc, self.nu))

        pyramid_fric_matrix = pyramid_friction(self.contacts, 0.7 / np.sqrt(2))
        self.Cc[:20, :] = -pyramid_fric_matrix @ self.Jinv @ self.L

        if self.first_cont_interation:

            _, _, A_hex, b_hex = self.cheby_center_solver.solve(self.contacts[:, :])
            self.Cc[20:26, 18:20] = A_hex

            l = np.vstack((self.f_l.reshape(-1, 1), self.com_const.reshape(-1, 1), self.q_min.reshape(-1, 1)))
            u = np.vstack((self.f_u.reshape(-1, 1), b_hex.reshape(-1, 1), self.q_max.reshape(-1, 1)))

            self.l = np.tile(l, (self.N, 1))
            self.u = np.tile(u, (self.N, 1))

            self.first_cont_interation = False

        Phi_cons[0:self.nc, :] = self.Cc @ self.Aa
        aux_cons = self.Cc @ self.Ba

        return aux_cons, Phi_cons

    def build_reference(self):

        if self.first_cont_interation:

            # update z ref
            z = self.rs.r_pos[2]
            rzRef = z + self.z_ref

            # keep the currently yaw
            yaw = self.rs.rpy[2]
            epsRef, _ = eps_reference(current_yaw=yaw, desired_yaw=None)
            epsRef = epsRef.reshape(4, 1)

            ref = np.vstack((rzRef, epsRef, np.zeros((3, 1)), np.zeros((3, 1))))

            self.ref = np.tile(ref, (self.N, 1))
