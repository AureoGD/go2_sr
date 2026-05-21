import numpy as np
import pinocchio as pin
from scipy.linalg import block_diag
from control.self_righting.rgc_mpc_solution.rgc_base_controller import BaseRGCController
from control.self_righting.rgc_mpc_solution.constraints.self_collision import self_collision_constraints


class GoSafe(BaseRGCController):

    def __init__(self, robot_states, **kwargs):
        super().__init__(robot_states, **kwargs)

        self.action_group = 1

        # Predic and control horizons and sampe time
        self.N = 20
        self.M = 10
        self.ts = 0.01

        # Number of states, inputs, outputs and constarints
        self.nx = 12  # delta q (12, 1)
        self.nu = 12  # delta qr (12, 1)
        self.ny = 12  # qr (12, 1)
        self.nc = 18  # qr (12,1), legs links dist (6, 1)

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

        dqrWeight = 100 * np.array([1, 1, 1])
        Rdqr = np.diag(dqrWeight)
        R = block_diag(Rdqr, Rdqr, Rdqr, Rdqr)
        self.R = block_diag(*[R] * self.M)

        # ----------------------------------------
        # Reference
        # ----------------------------------------

        qr = np.array([[0.7, 1.4, -2.6, -0.7, 1.4, -2.6, 0.7, 1.4, -2.6, -0.7, 1.4, -2.6]]).transpose()
        self.ref = np.tile(qr, (self.N, 1))

        # ----------------------------------------
        # Controller specific variables and objects
        # ----------------------------------------

        self.radius = 0.010
        self.d_safe = 0.005

        self.collision_pairs = [
            ("FR", "FL"),
            ("FR", "RR"),
            ("FR", "RL"),
            ("FL", "RR"),
            ("FL", "RL"),
            ("RR", "RL"),
        ]

        M = np.diag([0.02, 0.011, 0.005, 0.011, 0.011, 0.005, 0.011, 0.011, 0.005, 0.011, 0.011, 0.005])

        M_diag = np.diag(M)
        Kp_diag = self.kp

        self.Iu = np.eye(12)

        self.lambda_vec = np.sqrt(Kp_diag / M_diag)

        self.alpha = self.Iu - self.ts * np.diag(self.lambda_vec)

        self.first_int = True

        self.inf_vec = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

        # ----------------------------------------
        # Low-level mode controller gains
        # ----------------------------------------

        self.Kp_vec = np.ones(12) * self.kp / 2
        self.Kd_vec = np.ones(12) * self.kd / 10

    def update_model(self):
        M = self.pin_engine.actuated_mass_matrix()

        M_diag = np.maximum(np.diag(M), 1e-6)

        self.lambda_vec = np.sqrt(self.kp / M_diag)

        self.alpha = self.Iu - self.ts * np.diag(self.lambda_vec)

        self.Aa[0:12, 0:12] = self.alpha
        self.Aa[0:12, 12:] = self.Iu - self.alpha

        self.x = np.vstack((self.rs.q.reshape(-1, 1), self.cs.qr.reshape(-1, 1)))

    def build_constraint_matrices(self):

        J, dist = self_collision_constraints(pin_engine=self.pin_engine,
                                             pairs=self.collision_pairs,
                                             radius=self.radius,
                                             d_safe=self.d_safe)

        Phi_cons = np.zeros((self.nc * self.N, self.nx + self.nu))
        aux_cons = np.zeros((self.nc, self.nu))

        self.Cc[12:, 0:12] = self.ts * J @ (-self.lambda_vec * self.Iu).reshape(12, 12)
        self.Cc[12:, 12:] = self.ts * J @ (self.lambda_vec * self.Iu).reshape(12, 12)

        Phi_cons[:self.nc, :] = self.Cc @ self.Aa
        aux_cons = self.Cc @ self.Ba

        if self.first_int:
            l = np.vstack((self.q_min.reshape(-1, 1), dist.reshape(-1, 1)))
            u = np.vstack((self.q_max.reshape(-1, 1), self.inf_vec.reshape(-1, 1)))

            self.l = np.tile(l, (self.N, 1))
            self.u = np.tile(u, (self.N, 1))

        return aux_cons, Phi_cons

    def update_pred_mdl(self):

        Phi, G, Phi_cons, G_cons = super().update_pred_mdl()

        for s in range(0, (self.N - 1) * 18, 18):

            G_cons[30 + s:36 + s] += G_cons[12 + s:18 + s]

            Phi_cons[30 + s:36 + s] += Phi_cons[12 + s:18 + s]

        return Phi, G, Phi_cons, G_cons

    def build_reference(self):
        pass
