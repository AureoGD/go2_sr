import numpy as np
import pinocchio as pin
from scipy.linalg import block_diag
from environment.strategies.rgc_mpc.base_controller import BaseRGC


class PrepareCW(BaseRGC):

    TASK_NAME = "prepare_cw"
    TASK_LEVEL = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not self.runtime:
            return

        self.N = 20
        self.M = 10
        self.ts = 0.01
        self.convergence_threshold = 0.5
        self.ws = 15
        self._update_detector()

        self.nx = 15
        self.nu = 12
        self.ny = 12
        self.nc = 13
        self.A = np.zeros((self.nx, self.nx), dtype=np.float32)
        self.B = np.zeros((self.nx, self.nu), dtype=np.float32)

        self.Aa = np.zeros((self.nx + self.nu, self.nx + self.nu), dtype=np.float32)
        self.Ba = np.zeros((self.nx + self.nu, self.nu), dtype=np.float32)
        self.Ca = np.zeros((self.ny, self.nx + self.nu), dtype=np.float32)

        self.C_cons = np.zeros((self.nc, self.nx + self.nu), dtype=np.float32)

        self.Aa[self.nx:, self.nx:] = np.identity(self.nu)
        self.Ba[self.nx:, :] = np.identity(self.nu)

        # Joint position
        self.Ca[0:9, 0:9] = np.identity(9)
        self.Ca[9:12, 12:15] = np.eye(3)

        # Joint reference
        self.C_cons[0:12, 15:] = np.identity(12)
        self.C_cons[12, 14] = 1

        M = np.diag([0.02, 0.011, 0.005, 0.011, 0.011, 0.005, 0.011, 0.011, 0.005, 0.011, 0.011, 0.005])

        M_diag = np.diag(M)  # shape (12,)
        Kp_diag = self.kp  # scalar or shape (12,)

        self.lambda_vec = np.sqrt(Kp_diag / M_diag)

        self.alpha = np.eye(12) - self.ts * np.diag(self.lambda_vec)

        Qq = np.array([0.01, 0.01, 0.01])
        Qq = np.diag(Qq)

        Qf = np.array([10, 10, 1])
        Qf = np.diag(Qf)

        Q = block_diag(Qq, Qq, Qq, Qf)
        self.Q = block_diag(*[Q] * self.N)

        Rdqr = np.array([1, 1, 1])
        Rdqr = np.diag(Rdqr)

        R = block_diag(10 * Rdqr, 10 * Rdqr, 10 * Rdqr, Rdqr)
        self.R = block_diag(*[R] * self.M)

        qr = np.array([[-0.6, 1.5, -2.0, -0.8, 1.0, -2.6, -0.6, 1.5, -2.0]]).transpose()
        pfoot = np.array([[-0.2, -0.15, 0.1]]).transpose()
        ref = np.vstack((qr, pfoot))
        self.ref = np.tile(ref, (self.N, 1))

        # FR, FL, RR, RL
        self.leg_idx = [3, 0, 9, 6]

        self.contact_ids = [
            self.model.getFrameId('FR_foot'),
            self.model.getFrameId('FL_foot'),
            self.model.getFrameId('RR_foot'),
            self.model.getFrameId('RL_foot'),
        ]

        self.n_local = np.array([0.0, 0.0, 1.0])
        self.p_offset_local = np.array([0.0, 0.0, 0.06755])
        self.d_safe = 0.01

        self.first_int = True

    def update_model(self):
        self._q, dq = self.ordering_joints()
        q = self._q

        pin.forwardKinematics(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data)

        pin.crba(self.model, self.data, q)
        pin.computeCoriolisMatrix(self.model, self.data, q, dq)

        pin.ccrba(self.model, self.data, q, dq)

        r = self.data.com[0]
        dr = self.data.vcom[0]

        M = block_diag(self.data.M[9:12, 9:12], self.data.M[6:9, 6:9], self.data.M[12:, 12:])

        Jf = pin.computeFrameJacobian(self.model, self.data, q, self.model.getFrameId('RL_calf_joint'),
                                      pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[0:3, 12:15]

        M_diag = M.diagonal()

        self.lambda_vec = np.sqrt(self.kp / M_diag)

        self.alpha = np.eye(12) - self.ts * np.diag(self.lambda_vec)

        self.robot_states.r_vel = dr.reshape(3, 1)
        self.robot_states.r_pos = r.reshape(3, 1)

        self.Aa[0:12, 0:12] = self.alpha
        self.Aa[0:12, 15:] = np.eye(12) - self.alpha
        self.Aa[12:15, 9:12] = -self.ts * np.identity(3) @ Jf
        self.Aa[12:15, 12:15] = np.identity(3)
        self.Aa[12:15, 24:] = self.ts * np.identity(3) @ Jf

        rl_foot = self.data.oMf[self.model.getFrameId('RL_calf_joint')].translation
        rl_foot = rl_foot.reshape(3, 1)

        self.x = np.vstack((self.robot_states.q, rl_foot, self.robot_states.qr))

    def define_constraints_matrices(self):
        Phi_cons = np.zeros((self.nc * self.N, self.nx + self.nu))
        aux_cons = np.zeros((self.nc, self.nu))

        Phi_cons[:self.nc, :] = self.C_cons @ self.Aa
        aux_cons = self.C_cons @ self.Ba

        base_id = self.model.getFrameId("base_link")
        R_base = self.data.oMf[base_id].rotation
        p_base = self.data.oMf[base_id].translation
        n_world = R_base @ -self.n_local
        p0_world = p_base + (R_base @ self.p_offset_local)
        p_knee = self.data.oMf[self.model.getFrameId("RL_calf_joint")].translation
        dist = np.dot(n_world, p_knee - p0_world)

        if self.first_int:
            l = np.vstack((self.qr_l, self.d_safe - dist))
            u = np.vstack((self.qr_u, np.inf))

            self.l = np.tile(l, (self.N, 1))
            self.u = np.tile(u, (self.N, 1))
            self.first_int = True

        return aux_cons, Phi_cons
