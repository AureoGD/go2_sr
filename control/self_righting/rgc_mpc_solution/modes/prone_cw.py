import numpy as np
from scipy.linalg import block_diag
from control.self_righting.rgc_mpc_solution.rgc_base_controller import BaseRGCController


class ProneCW(BaseRGCController):

    def __init__(self, robot_states, **kwargs):
        super().__init__(robot_states, **kwargs)

        self.phase = 6

        # Predic and control horizons and sampe time
        self.N = 20
        self.M = 20
        self.ts = 0.01

        # Number of states, inputs, outputs and constarints
        self.nx = 12  # delta q (12, 1)
        self.nu = 12  # delta qr (12, 1)
        self.ny = 12  # qr (12, 1)

        # Constraints slices
        self.i_qr = slice(0, 12)
        self.i_tau = slice(self.i_qr.stop, self.i_qr.stop + 12)
        self.nch = self.i_tau.stop

        # ----------------------------------------
        # Low-level mode controller gains
        # ----------------------------------------

        self.Kp_vec = np.ones(12) * self.kp
        self.Kd_vec = np.ones(12) * self.kd

        # Dynamic matrices
        self.A = np.zeros((self.nx, self.nx), dtype=np.float32)
        self.B = np.zeros((self.nx, self.nu), dtype=np.float32)

        # Aumented matrices
        self.Aa = np.zeros((self.nx + self.nu, self.nx + self.nu), dtype=np.float32)
        self.Ba = np.zeros((self.nx + self.nu, self.nu), dtype=np.float32)
        self.Cy = np.zeros((self.ny, self.nx + self.nu), dtype=np.float32)

        # Constraint matrix
        self.Cch = np.zeros((self.nch, self.nx + self.nu), dtype=np.float32)

        # ----------------------------------------
        # Matrices inicialization
        # ----------------------------------------

        self.damping = 2
        self.Lambda = np.diag(self.Kp_vec / (self.Kd_vec + self.damping))

        self.Aa[0:12, 0:12] = np.eye(12) + self.ts * -self.Lambda
        self.Aa[0:12, self.nx:] = self.ts * self.Lambda
        self.Aa[self.nx:, self.nx:] = np.identity(self.nu)
        self.Ba[self.nx:, :] = np.identity(self.nu)

        # Output matrix
        self.Cy[:, 0:12] = np.identity(12)

        # Constraint matrix
        self.Cch[self.i_qr, 12:] = np.identity(12)
        Ke = np.diag(self.Kp_vec) - np.diag(self.Kd_vec) @ self.Lambda
        self.Cch[self.i_tau, :12] = -Ke
        self.Cch[self.i_tau, 12:] = Ke

        # ----------------------------------------
        # Weights
        # ----------------------------------------

        Qq = 1.25 * np.eye(12)
        self.Q = block_diag(*[Qq] * self.N)

        # Update control action weight matrix
        Rdqfr = 200 * np.diag(np.array([1, 1, 1]))
        Rdqfl = 50 * np.diag(np.array([1, 1, 1]))
        Rdqrr = 100 * np.diag(np.array([1, 1, 1]))
        Rdqrl = 50 * np.diag(np.array([1, 1, 1]))

        R = block_diag(Rdqfr, Rdqfl, Rdqrr, Rdqrl)
        self.R = 0.75 * block_diag(*[R] * self.M)

        # ----------------------------------------
        # Reference
        # ----------------------------------------
        self.qr1 = np.array([[-0.5, 1.4, -2.7, 0.5, 1.4, -2.7, -0.5, 1.4, -2.7, 0.5, 1.4, -2.7]]).transpose()
        self.qr2 = np.array([[0.0, 1.4, -2.7, 0.0, 1.4, -2.7, 0.0, 1.4, -2.7, 0.0, 1.4, -2.7]]).transpose()

        self.first_int = True

        self.qr_start = None

        self.stg1 = False
        self.stg2 = False
        self.stg3 = False
        self.stg4 = False

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

    def build_reference(self):

        if self.first_int:
            self.qr_start = self.rs.q.copy()
            self.first_int = False

        if not self.stg1:
            ref = self.qr_start.copy()
            ref[5] = -2.6  # calf FL
            ref[11] = -2.6  # calf RL
            self.ref = np.tile(ref.reshape(-1, 1), (self.N, 1))
            self.stg1 = True
        if self.stg1 and not self.stg2:
            if np.linalg.norm(self.ref[5] - self.rs.q[5]) < 0.20 and np.linalg.norm(self.ref[11] -
                                                                                    self.rs.q[11]) < 0.20:
                ref = self.ref[0:12].copy()
                ref[0] = 0
                ref[6] = 0
                self.ref = np.tile(ref.reshape(-1, 1), (self.N, 1))
                self.stg2 = True
        if self.stg2 and not self.stg3:
            if np.linalg.norm(self.ref[0] - self.rs.q[0]) < 0.20 and np.linalg.norm(self.ref[6] - self.rs.q[6]) < 0.20:
                self.ref = np.tile(self.qr1.reshape(-1, 1), (self.N, 1))
                self.stg3 = True
        if self.stg3 and not self.stg4:
            if np.linalg.norm(self.qr1 - self.rs.q.reshape(-1, 1)) < 0.20:
                self.ref = np.tile(self.qr2.reshape(-1, 1), (self.N, 1))
                self.stg4 = True

        self.task_state.prone_final_stage = self.stg4
