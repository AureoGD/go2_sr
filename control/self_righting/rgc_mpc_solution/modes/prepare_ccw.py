import numpy as np
import pinocchio as pin
from scipy.linalg import block_diag

from control.self_righting.rgc_mpc_solution.rgc_base_controller import BaseRGCController
from control.self_righting.rgc_mpc_solution.constraints.plane_colision import PlaneConstraint
from control.self_righting.rgc_mpc_solution.constraints.self_collision import self_collision_constraints



class PrepareCCW(BaseRGCController):

    def __init__(self, robot_states, **kwargs):
        super().__init__(robot_states, **kwargs)

        self.phase = 2

        # Predic and control horizons and sampe time
        self.N = 20
        self.M = 10
        self.ts = 0.01

        # Number of states, inputs, outputs and constarints
        self.nx = 12  # q (12, 1), q_ant (12, 1)
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

        R = 200 * block_diag(dqrWfr, dqrWfl, dqrWrr, dqrWrl)
        self.R = block_diag(*[R] * self.M)

        # ----------------------------------------
        # Reference
        # ----------------------------------------
        qr = np.array([[0.00, 1.00, -2.60,
                        1.05, 1.50, -2.30, 
                        1.02, 4.15, -2.20,
                        1.05, 1.50, -2.30]]).transpose()  

        self.ref = np.tile(qr, (self.N, 1))

        # ----------------------------------------
        # Controller specific variables and objects
        # ----------------------------------------
        self.ncs = 9
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

        self.p_offset_local = np.array([0.0, 0.0, 0.06755])
        R_b_plane = self.roty(-5)
        self.n_local = R_b_plane @ np.array([0.0, 0.0, 1.0])

        self.colision_cons = PlaneConstraint(self.n_local, self.p_offset_local, self.d_safe, np.zeros((3, 1)))
        self.wcs_plane = 2.5

        self.wcs_grf = 2

        wcs = np.vstack((self.wcs_collision * np.ones((6, 1)), np.array([[self.wcs_plane]]), self.wcs_grf * np.ones(
            (2, 1))))
        self.wcs = np.tile(wcs, (self.N, 1))

        self.first_int = True

        self.fr_count = 0

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
        Ccs[0:6, 0:12] = Jrow

        nJq0 = (Jrow @ self.x[0:12]).reshape(-1, 1)

        Rb = self.pin_engine.get_base_rot_mtx()
        n_w, lb, ub, p_offset_w = self.colision_cons.update(p_base=self.rs.b_pos, R_b=Rb)
        p_k0 = self.pin_engine.frame_pos('RR', 'calf')
        Jk = self.pin_engine.linear_leg_jacobian('RR', 'calf')
        g = n_w @ Jk
        Ccs[6, 9:12] = g
        knee_lower = lb - n_w @ p_k0 + g @ self.x[6:9]

        f_cte_l = np.array([-np.inf, -np.inf])
        f_cte_u = np.array([np.inf, np.inf])

        J = self.pin_engine.linear_leg_jacobian("FL", "thigh")
        J_inv = np.linalg.pinv(J).T
        self.rs.force_shoulder[1] = -J_inv @ self.cs.tau[3:6]
        if np.linalg.norm(self.rs.force_shoulder[1]) > 30:
            self.fr_count += 1

        if self.fr_count > 5:
            F = -J_inv @ self.Cch[15:18, :]
            Ccs[7, :] = F[2, :].copy()
            f_cte_l[0] = 30
            f_cte_u[0] = 35
        else:
            Ccs[7, :] = 0

        J = self.pin_engine.linear_leg_jacobian("RL", "thigh")
        J_inv = np.linalg.pinv(J).T
        self.rs.force_shoulder[3] = -J_inv @ self.cs.tau[9:]
        if np.linalg.norm(self.rs.force_shoulder[3]) > 30:
            F = -J_inv @ self.Cch[21:24, :]
            Ccs[8, :] = F[2, :].copy()
            f_cte_l[1] = 30
            f_cte_u[1] = 35
        else:
            Ccs[8, :] = 0

        l = np.vstack((dist + nJq0, knee_lower, f_cte_l.reshape(-1, 1)))
        u = np.vstack((np.full((7, 1), np.inf), f_cte_u.reshape(-1, 1)))
        self.lcs = np.tile(l, (self.N, 1))
        self.ucs = np.tile(u, (self.N, 1))

        Phi_cons[:self.ncs, :] = Ccs @ self.Aa
        aux_cons = Ccs @ self.Ba

        return aux_cons, Phi_cons

    def build_reference(self):
        pass

    def roty(self, theta):
        theta = np.pi * theta / 180
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
