import numpy as np
import pinocchio as pin
from scipy.linalg import block_diag
from control.self_righting.rgc_mpc_solution.rgc_base_controller import BaseRGCController
from control.self_righting.rgc_mpc_solution.constraints.self_collision import self_collision_constraints


class ProneCCW(BaseRGCController):

    def __init__(self, robot_states, **kwargs):
        super().__init__(robot_states, **kwargs)

        self.phase = 6

        # Predic and control horizons and sampe time
        self.N = 20
        self.M = 10
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

        self.Kp_vec = np.ones(12) * self.kp * 0.15
        self.Kd_vec = np.ones(12) * self.kd * 0.15

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

        Qq = 1 * np.eye(12)
        self.Q = block_diag(*[Qq] * self.N)

        # Update control action weight matrix
        Rdqfr = 50 * np.diag(np.array([1, 1, 1]))
        Rdqfl = 50 * np.diag(np.array([1, 1, 1]))
        Rdqrr = 50 * np.diag(np.array([1, 1, 1]))
        Rdqrl = 50 * np.diag(np.array([1, 1, 1]))

        R = block_diag(Rdqfr, Rdqfl, Rdqrr, Rdqrl)
        self.R = block_diag(*[R] * self.M)

        # ----------------------------------------
        # Reference
        # ----------------------------------------
        self.qr1 = np.array([[-0.5, 1.4, -2.7, 0.5, 1.4, -2.7, -0.5, 1.4, -2.7, 0.5, 1.4, -2.7]]).transpose()
        self.qr2 = np.array([[0.0, 1.4, -2.7, 0.0, 1.4, -2.7, 0.0, 1.4, -2.7, 0.0, 1.4, -2.7]]).transpose()

        self.qr_f1 = np.array([[2.7, -2.7]]).transpose()
        self.qr_r1 = np.array([[2.7, -2.7]]).transpose()

        self.first_int = True

        self.qr_start = None
        self.norm_er = 0
        self.norm_ef = 0
        self.norm_eo = 0

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

    #TODO: implement in a smart way
    def build_reference(self):

        if self.first_int:
            self.qr_start = self.rs.q.copy()
            self.norm_ef, self.norm_er = self.eval_norms()
            self.first_int = False

        norm_ef, norm_er = self.eval_norms()

        eps = 1e-6
        percent_f = norm_ef / max(self.norm_ef, eps)
        percent_r = norm_er / max(self.norm_er, eps)

        if percent_f > 0.25 and not self.stg2:
            if not self.stg1:
                ref = self.qr_start.copy()
                ref[4:6] = self.qr_f1.reshape(2,)  # FL
                ref[2] = -2.6  # FR
                ref[8] = -2.6  #RR
                self.ref = np.tile(ref.reshape(-1, 1), (self.N, 1))
                self.stg1 = True
        elif percent_r > 0.25 and not self.stg3:
            if not self.stg2:
                ref = self.qr_start.copy()
                ref[3] = 0  # FL
                ref[4:6] = self.qr_f1.reshape(2,)  # FL
                ref[2] = -2.6  # FR
                ref[10:12] = self.qr_r1.reshape(2,)  # RL
                ref[8] = -2.6  # RR
                self.ref = np.tile(ref.reshape(-1, 1), (self.N, 1))
                self.stg2 = True
        elif np.linalg.norm(self.qr1 - self.cs.qr.reshape(-1, 1)) > 0.2 and not self.stg4:
            if not self.stg3:
                self.ref = np.tile(self.qr1.reshape(-1, 1), (self.N, 1))
                self.stg3 = True
        else:
            self.stg4 = True
            self.ref = np.tile(self.qr2.reshape(-1, 1), (self.N, 1))
        self.task_state.prone_final_stage = self.stg4

    def eval_norms(self):
        error_fr = self.rs.q[4:6].reshape(2, 1) - self.qr_f1
        error_rr = self.rs.q[10:12].reshape(2, 1) - self.qr_r1
        norm_ef = np.linalg.norm(error_fr)
        norm_er = np.linalg.norm(error_rr)

        return norm_ef, norm_er
