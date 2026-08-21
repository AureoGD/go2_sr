import numpy as np
import pinocchio as pin
from scipy.linalg import block_diag
from control.self_righting.rgc_mpc_solution.rgc_base_controller import BaseRGCController
from control.self_righting.rgc_mpc_solution.constraints.self_collision import self_collision_constraints


class GoSafe(BaseRGCController):

    def __init__(self, robot_states, **kwargs):
        super().__init__(robot_states, **kwargs)

        self.phase = 1

        # Predic and control horizons and sampe time
        self.N = 20
        self.M = 10
        self.ts = 0.01

        # Number of states, inputs, outputs and constarints
        self.nx = 12  # q (12, 1)
        self.nu = 12  # delta qr (12, 1)
        self.ny = 12  # qr (12, 1)

        # Constraints slices
        self.i_qr = slice(0, 12)
        self.i_tau = slice(self.i_qr.stop, self.i_qr.stop + 12)
        self.nch = self.i_tau.stop

        # ----------------------------------------
        # Low-level mode controller gains
        # ----------------------------------------

        self.Kp_vec = np.ones(12) * self.kp / 2
        self.Kd_vec = np.ones(12) * self.kd / 10

        # ----------------------------------------
        # Matrices creation
        # ----------------------------------------
        self.A = np.zeros((self.nx, self.nx), dtype=np.float32)
        self.B = np.zeros((self.nx, self.nu), dtype=np.float32)

        # Aumented matrices
        self.Aa = np.zeros((self.nx + self.nu, self.nx + self.nu), dtype=np.float32)
        self.Ba = np.zeros((self.nx + self.nu, self.nu), dtype=np.float32)
        self.Cy = np.zeros((self.ny, self.nx + self.nu), dtype=np.float32)

        # Hard constraint matrix
        self.Cch = np.zeros((self.nch, self.nx + self.nu), dtype=np.float32)

        # ----------------------------------------
        # Matrices inicialization
        # ----------------------------------------

        self.damping = 2
        self.Lambda = np.diag(self.Kp_vec / (self.Kd_vec + self.damping))

        self.Aa[0:12, 0:12] = np.eye(12) + self.ts * -self.Lambda
        self.Aa[0:12, self.nx:] = self.ts * self.Lambda
        # self.Aa[12:24, 0:12] = np.eye(12)
        self.Aa[self.nx:, self.nx:] = np.identity(self.nu)
        self.Ba[self.nx:, :] = np.identity(self.nu)

        # Output matrix
        self.Cy[:, 0:12] = np.identity(12)
        # self.q_ant = np.zeros(12)

        # Constraint matrix
        self.Cch[self.i_qr, 12:] = np.identity(12)
        Ke = np.diag(self.Kp_vec)-np.diag(self.Kd_vec)@self.Lambda
        self.Cch[self.i_tau, :12] = -Ke
        self.Cch[self.i_tau, 12:] = Ke

        # ----------------------------------------
        # Weights
        # ----------------------------------------

        Qq = 1 * np.eye(12)
        self.Q = block_diag(*[Qq] * self.N)

        dqrWfr = np.diag(1 * np.array([1, 1, 1]))
        dqrWfl = np.diag(1 * np.array([1, 1, 1]))
        dqrWrr = np.diag(1 * np.array([1, 1, 1]))
        dqrWrl = np.diag(1 * np.array([1, 1, 1]))

        R = 100 * block_diag(dqrWfr, dqrWfl, dqrWrr, dqrWrl)
        self.R = block_diag(*[R] * self.M)

        # ----------------------------------------
        # Reference
        # ----------------------------------------
        qr = np.array([[0.7, 1.4, -2.6, -0.7, 1.4, -2.6, 0.7, 1.4, -2.6, -0.7, 1.4, -2.6]]).transpose()
        self.ref = np.tile(qr, (self.N, 1))

        # ----------------------------------------
        # Controller specific variables and objects
        # ----------------------------------------
        self.ncs = 6
        self.wcs_collision = 1
        self.radius = 0.025
        self.d_safe = 0.05
        self.collision_pairs = [
            ("FR", "FL"),
            ("FR", "RR"),
            ("FR", "RL"),
            ("FL", "RR"),
            ("FL", "RL"),
            ("RR", "RL"),
        ]

        self.first_int = True

        self.inf_vec = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

    def update_model(self):
        self.x = np.vstack((self.rs.q.reshape(-1, 1), self.cs.qr.reshape(-1, 1)))

    def build_hard_constraint_matrices(self):

        Phi_cons = np.zeros((self.nch * self.N, self.nx + self.nu))
        aux_cons = np.zeros((self.nch, self.nu))

        Phi_cons[:self.nch, :] = self.Cch @ self.Aa
        aux_cons = self.Cch @ self.Ba

        l = np.vstack((self.q_min.reshape(-1, 1), self.tau_min.reshape(-1, 1)))
        u = np.vstack((self.q_max.reshape(-1, 1), self.tau_max.reshape(-1, 1)))

        self.lch = np.tile(l, (self.N, 1))
        self.uch = np.tile(u, (self.N, 1))

        return aux_cons, Phi_cons

    def build_soft_constraint_matrices(self):

        Phi_cons = np.zeros((self.ncs * self.N, self.nx + self.nu))
        aux_cons = np.zeros((self.ncs, self.nu))

        Jrow, dist = self_collision_constraints(self.pin_engine, self.collision_pairs, self.radius, self.d_safe)

        Ccs = np.zeros((self.ncs, self.nx + self.nu))
        Ccs[:, 0:12] = Jrow

        q0 = self.x[0:12]
        nJq0 = (Jrow @ q0).reshape(-1, 1)
        self.lcs = np.tile(dist + nJq0, (self.N, 1))
        self.ucs = np.tile(np.full((self.ncs, 1), np.inf), (self.N, 1))

        self.wcs = self.wcs_collision * np.ones(self.ncs * self.N)

        Phi_cons[:self.ncs, :] = Ccs @ self.Aa
        aux_cons = Ccs @ self.Ba
        return aux_cons, Phi_cons

    def build_reference(self):
        pass
