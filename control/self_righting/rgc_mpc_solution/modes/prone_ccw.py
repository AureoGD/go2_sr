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
        self.nc = 24  # qr (12,1), legs links dist (6, 1)

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

        # Output matrix
        self.Ca[:, 0:12] = np.identity(12)  # joint pos

        # Constraint matrix
        self.Cc[0:12, 12:] = np.identity(12)

        # ----------------------------------------
        # Weights
        # ----------------------------------------

        Qq = 1 * np.eye(12)
        self.Q = block_diag(*[Qq] * self.N)

        # Update control action weight matrix
        Rdqfr = 10 * np.diag(np.array([1, 1, 1]))
        Rdqfl = 10 * np.diag(np.array([1, 1, 1]))
        Rdqrr = 10 * np.diag(np.array([1, 1, 1]))
        Rdqrl = 10 * np.diag(np.array([1, 1, 1]))

        R = block_diag(Rdqfr, Rdqfl, Rdqrr, Rdqrl)
        self.R = block_diag(*[R] * self.M)

        # ----------------------------------------
        # Reference
        # ----------------------------------------

        self.qr2 = np.array([[0.0, 1.4, -2.7, 0.0, 1.4, -2.7, 0.0, 1.4, -2.7, 0.0, 1.4, -2.7]]).transpose()

        self.qr_f1 = np.array([[2.7, -2.7]]).transpose()
        self.qr_r1 = np.array([[2.7, -2.7]]).transpose()

        M = np.diag([0.02, 0.011, 0.005, 0.011, 0.011, 0.005, 0.011, 0.011, 0.005, 0.011, 0.011, 0.005])

        M_diag = np.diag(M)
        Kp_diag = self.kp

        self.Iu = np.eye(12)

        self.lambda_vec = np.sqrt(Kp_diag / M_diag)

        self.alpha = self.Iu - self.ts * np.diag(self.lambda_vec)

        self.first_int = True

        self.inf_vec = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

        self.qr_start = None
        self.norm_er = 0
        self.norm_ef = 0
        self.norm_eo = 0
        self.final_ref = False

        # ----------------------------------------
        # Low-level mode controller gains
        # ----------------------------------------

        self.Kp_vec = np.ones(12) * self.kp * 0.25
        self.Kd_vec = np.ones(12) * self.kd * 0.25

        self.kp_mtx = np.diag(self.Kp_vec)
        self.kd_mtx = np.diag(self.Kd_vec)

    def update_model(self):
        M = self.pin_engine.actuated_mass_matrix()

        M_diag = np.maximum(np.diag(M), 1e-6)

        self.lambda_vec = np.sqrt(self.kp / M_diag)

        self.alpha = self.Iu - self.ts * np.diag(self.lambda_vec)

        self.Aa[0:12, 0:12] = self.alpha
        self.Aa[0:12, 12:] = self.Iu - self.alpha

        self.x = np.vstack((self.rs.q.reshape(-1, 1), self.cs.qr.reshape(-1, 1)))

    def build_output_constraint_matrices(self):

        Phi_cons = np.zeros((self.nc * self.N, self.nx + self.nu))
        aux_cons = np.zeros((self.nc, self.nu))

        tau_c = self.kp_mtx - np.diag(self.lambda_vec) @ self.kd_mtx
        self.Cc[12:, 0:12] = -tau_c
        self.Cc[12:, 12:] = tau_c

        Phi_cons[:self.nc, :] = self.Cc @ self.Aa
        aux_cons = self.Cc @ self.Ba

        if self.first_int:
            l = np.vstack((self.q_min.reshape(-1, 1), self.tau_min.reshape(-1, 1)))
            u = np.vstack((self.q_max.reshape(-1, 1), self.tau_max.reshape(-1, 1)))

            self.l = np.tile(l, (self.N, 1))
            self.u = np.tile(u, (self.N, 1))

            self.first_int = False

        return aux_cons, Phi_cons

    #TODO: implement in a smart way
    def build_reference(self):

        if self.first_int:
            self.qr_start = self.rs.q.copy()
            self.norm_ef, self.norm_er = self.eval_norms()

        norm_ef, norm_er = self.eval_norms()

        eps = 1e-6
        percent_f = norm_ef / max(self.norm_ef, eps)
        percent_r = norm_er / max(self.norm_er, eps)

        if percent_f > 0.25 and not self.final_ref:
            ref = self.qr_start.copy()
            ref[2] = -2.6 # FR
            ref[4:6] = self.qr_f1.reshape(2,) # FL 
            ref[8] = -2.6 #RR
            self.ref = np.tile(ref.reshape(-1, 1), (self.N, 1))
        elif percent_r > 0.25 and not self.final_ref:
            ref = self.qr_start.copy()
            ref[3] = 0  # FL 
            ref[4:6] = self.qr_f1.reshape(2,) # FL 
            ref[2] = -2.6 # FR
            ref[10:12] = self.qr_r1.reshape(2,) # RL
            ref[8] = -2.6 # RR
            self.ref = np.tile(ref.reshape(-1, 1), (self.N, 1))
        else:
            self.final_ref = True
            self.ref = np.tile(self.qr2.reshape(-1, 1), (self.N, 1))

    def eval_norms(self):
        error_fr = self.rs.q[4:6].reshape(2, 1) - self.qr_f1
        error_rr = self.rs.q[10:12].reshape(2, 1) - self.qr_r1
        norm_ef = np.linalg.norm(error_fr)
        norm_er = np.linalg.norm(error_rr)

        return norm_ef, norm_er
