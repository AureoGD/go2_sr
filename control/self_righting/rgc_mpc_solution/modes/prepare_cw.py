import numpy as np
import pinocchio as pin
from scipy.linalg import block_diag

from control.self_righting.rgc_mpc_solution.rgc_base_controller import BaseRGCController
from control.self_righting.rgc_mpc_solution.constraints.plane_colision import PlaneConstraint


class PrepareCW(BaseRGCController):

    TASK_NAME = "prepare_cw"
    TASK_LEVEL = 2

    def __init__(self, robot_states, **kwargs):
        super().__init__(robot_states, **kwargs)

        self.phase = 2

        # Predic and control horizons and sampe time
        self.N = 20
        self.M = 10
        self.ts = 0.01

        # Number of states, inputs, outputs and constarints
        self.nx = 18  # joint pos (12, 1), rl knee pos (3, 1), rl foot pos (3, 1)
        self.nu = 12  # delta qr (12, 1)
        self.ny = 12  # joint pos (12, 1)
        self.nc = 25  # qr (12, 1), knee contact (1,1), (12,1)

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
        self.Aa[12:15, 12:15] = np.identity(3)
        self.Aa[15:18, 12:15] = np.identity(3)

        self.Ba[self.nx:, :] = np.identity(self.nu)

        # Joint position
        self.Ca[0:12, 0:12] = np.identity(12)

        # Joint reference
        self.Cc[0:12, 18:] = np.identity(12)

        M = np.diag([0.02, 0.011, 0.005, 0.011, 0.011, 0.005, 0.011, 0.011, 0.005, 0.011, 0.011, 0.005])

        M_diag = np.diag(M)
        Kp_diag = self.kp

        self.lambda_vec = np.sqrt(Kp_diag / M_diag)

        self.alpha = np.eye(12) - self.ts * np.diag(self.lambda_vec)

        Qq = np.array([0.01, 0.01, 0.01])
        Qq = np.diag(Qq)

        Qf = np.array([1, 0.05, 0.05])
        Qf = np.diag(Qf)

        Q = block_diag(Qq, Qq, Qq, Qf)
        self.Q = block_diag(*[Q] * self.N)

        Rdqr = np.array([10, 10, 10])
        Rdqr = np.diag(Rdqr)
        Rdqrf = np.array([0.1, 20, 5])
        Rdqrf = np.diag(Rdqrf)
        R = block_diag(Rdqr, Rdqr, Rdqr, Rdqrf)
        self.R = block_diag(*[R] * self.M)

        qr = np.array([[-0.6, 1.5, -2.0, -0.8, 1.0, -2.6, -0.6, 1.5, -2.0, -1.025, 4.15, -2.2]]).transpose()
        ref = np.vstack((qr))
        self.ref = np.tile(ref, (self.N, 1))

        self.p_offset_local = np.array([0.0, 0.0, 0.06755])
        self.d_safe = 0.05
        R_b_plane = self.roty(-5)
        self.n_local = R_b_plane @ np.array([0.0, 0.0, 1.0])

        self.colision_cons = PlaneConstraint(self.n_local, self.p_offset_local, self.d_safe, np.zeros((3, 1)))

        self.first_int = True

        self.Iu = np.eye(12)

        # ----------------------------------------
        # Low-level mode controller gains
        # ----------------------------------------

        self.Kp_vec = np.ones(12) * self.kp
        self.Kd_vec = np.ones(12) * self.kd / 10

        self.kp_mtx = np.diag(self.Kp_vec)
        self.kd_mtx = np.diag(self.Kd_vec)

    def update_model(self):
        M = self.pin_engine.actuated_mass_matrix()

        M_diag = np.maximum(np.diag(M), 1e-6)

        self.lambda_vec = np.sqrt(self.kp / M_diag)

        self.alpha = self.Iu - self.ts * np.diag(self.lambda_vec)

        Jk = self.pin_engine.linear_leg_jacobian('RL', 'calf')
        Jf = self.pin_engine.linear_leg_jacobian('RL', 'foot')

        self.Aa[0:12, 0:12] = self.alpha
        self.Aa[0:12, 18:] = self.Iu - self.alpha

        self.Aa[12:15, 9:12] = -self.ts * Jk
        self.Aa[12:15, 27:] = self.ts * Jk

        self.Aa[15:18, 9:12] = -self.ts * Jf
        self.Aa[15:18, 27:] = self.ts * Jf

        rl_foot = self.pin_engine.frame_pos('RL', 'foot')
        rl_knee = self.pin_engine.frame_pos('RL', 'calf')

        self.x = np.vstack(
            (self.rs.q.reshape(-1, 1), rl_knee.reshape(-1, 1), rl_foot.reshape(-1, 1), self.cs.qr.reshape(-1, 1)))

    def build_output_constraint_matrices(self):
        Phi_cons = np.zeros((self.nc * self.N, self.nx + self.nu))
        aux_cons = np.zeros((self.nc, self.nu))

        if self.first_int:
            Rb = self.pin_engine.get_base_rot_mtx()
            n_w, lb, ub, p_offset_w = self.colision_cons.update(p_base=self.rs.b_pos, R_b=Rb, p=self.x[12:15])
            self.dg.plane_pos = p_offset_w
            self.Cc[12, 12:15] = n_w
            tau_c = self.kp_mtx - np.diag(self.lambda_vec) @ self.kd_mtx
            self.Cc[13:, 0:12] = -tau_c
            self.Cc[13:, 18:] = tau_c

            margin = float(n_w @ self.x[12:15]) - lb
            slack = max(0.0, -margin + 1e-4)

            l = np.vstack((self.q_min.reshape(-1, 1), lb - slack, self.tau_min.reshape(-1, 1)))
            u = np.vstack((self.q_max.reshape(-1, 1), ub, self.tau_max.reshape(-1, 1)))

            self.l = np.tile(l, (self.N, 1))
            self.u = np.tile(u, (self.N, 1))

        Phi_cons[:self.nc, :] = self.Cc @ self.Aa
        aux_cons = self.Cc @ self.Ba

        return aux_cons, Phi_cons

    def roty(self, theta):
        theta = np.pi * theta / 180
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def build_reference(self):
        pass
