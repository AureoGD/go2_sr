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
            # Metadata-only: nothing else to do
            return

        self.N = 20
        self.M = 15
        self.ts = 0.01
        self.convergence_threshold = 0.1
        self.ws = 30
        self._update_detector()

        self.nx = 27
        self.nu = 12
        # self.ny = 15
        self.ny = 14

        self.nc = 14
        self.A = np.zeros((self.nx, self.nx), dtype=np.float32)
        self.B = np.zeros((self.nx, self.nu), dtype=np.float32)

        self.Aa = np.zeros((self.nx + self.nu, self.nx + self.nu), dtype=np.float32)
        self.Ba = np.zeros((self.nx + self.nu, self.nu), dtype=np.float32)
        self.Ca = np.zeros((self.ny, self.nx + self.nu), dtype=np.float32)

        self.C_cons = np.zeros((self.nc, self.nx + self.nu), dtype=np.float32)

        self.A[12:24, 0:12] = np.identity(12)
        self.A[24:27, 24:27] = np.identity(3)

        self.Aa[27:, 27:] = np.identity(self.nu)
        self.Ba[27:, :] = np.identity(self.nu)

        # 12 joints + min pz
        # self.Ca[0:12, 12:24] = np.identity(12)
        # self.Ca[12:, 24:27] = np.identity(3)

        # 11 joints + min pz
        self.Ca[0:9, 12:21] = np.identity(9)
        self.Ca[9:11, 22:24] = np.identity(2)
        self.Ca[11:, 24:27] = np.identity(3)

        # rx and rz
        self.C_cons[0:12, 12:24] = np.identity(12)
        self.C_cons[12, 10] = 1
        self.contacts = np.zeros((4, 3), dtype=np.float32)

        self.Is = np.concatenate((np.identity(3), np.identity(3), np.identity(3), np.identity(3)), axis=1)

        self.n_local = np.array([0.0, 0.0, 1.0])
        self.p_offset_local = np.array([0.0, 0.0, 0.05])

        Qq = 0.1 * np.eye(11)
        Qq[0, 0] = 1
        Qq[6, 6] = 1
        Qpc = block_diag(0.001, 0.001, 4)

        Q = block_diag(Qq, Qpc)

        self.Q = block_diag(*[Q] * self.N)

        dqrWeight = 10 * np.array([1, 1, 1])
        Rdqr = np.diag(dqrWeight)
        dqrWeight = np.array([2, 2, 2])
        # dqrWeight = np.array([2, 2])
        Rdqr_rl = np.diag(dqrWeight)
        R = block_diag(Rdqr, Rdqr, Rdqr, Rdqr_rl)
        self.R = block_diag(*[R] * self.M)

        # qr = np.array([[-0.6, 1.5, -2.0, -0.8, 1.0, -2.6, -0.6, 1.25, -2.0, -0.9, 4.45, -2.5]]).transpose()

        qr = np.array([[-0.6, 1.5, -2.0, -0.8, 1.0, -2.6, -0.6, 1.5, -2.0, 4.2, -2.25]]).transpose()
        # qr = np.array([-0.6, 1.5, -2.0, -0.8, 1.0, -2.6, -0.6, 1.5, -2.0, -0.5, 4.2, -2.25]).reshape(12, 1)

        ref = np.vstack((qr, np.zeros((3, 1))))
        self.ref = np.tile(ref, (self.N, 1))

        qr_l = np.array([
            -1.0472, -1.5708, -2.7227, -1.0472, -1.5708, -2.7227, -1.0472, -0.5236, -2.7227, -1.0472, -0.5236, -2.7227
        ])
        qr_u = np.array(
            [1.0472, 3.4907, -0.83776, 1.0472, 3.4907, -0.83776, 1.0472, 4.5379, -0.83776, 1.0472, 4.5379, -0.83776])

        self.d_safe = 0.01
        # Stack for all 4 feet
        self.qr_l = qr_l.reshape(12, 1)
        self.qr_u = qr_u.reshape(12, 1)

        # FR, FL, RR, RL
        self.leg_idx = [3, 0, 9, 6]

        self.J_contact_foot = None

        self.first_int = True

    def update_model(self):
        self._q, dq = self.ordering_joints()
        q = self._q

        # 1. COMPUTE ALL KINEMATICS (Positions and Velocities)
        # Do this once at the beginning.
        pin.forwardKinematics(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data)

        # 2. COMPUTE JOINT-SPACE MATRICES
        pin.crba(self.model, self.data, q)
        pin.computeCoriolisMatrix(self.model, self.data, q, dq)

        pin.jacobianCenterOfMass(self.model, self.data, q)

        # 3. COMPUTE ALL CENTROIDAL QUANTITIES
        pin.ccrba(self.model, self.data, q, dq)

        M = block_diag(self.data.M[9:12, 9:12], self.data.M[6:9, 6:9], self.data.M[15:18, 15:18], self.data.M[12:15,
                                                                                                              12:15])
        C = block_diag(self.data.C[9:12, 9:12], self.data.C[6:9, 6:9], self.data.C[15:18, 15:18], self.data.C[12:15,
                                                                                                              12:15])
        r = self.data.com[0]
        dr = self.data.vcom[0]

        J_contact_foot = pin.computeFrameJacobian(self.model, self.data, q, self.model.getFrameId('RL_foot'),
                                                  pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[0:6, 12:15]
        # Save states
        self.robot_states.r_vel = dr.reshape(3, 1)
        self.robot_states.r_pos = r.reshape(3, 1)

        M_inv = np.linalg.inv(M)

        self.A[0:12, 0:12] = -M_inv @ (C + self.kd / 10 * np.identity(12))
        self.A[0:12, 12:24] = -self.kp * M_inv @ np.identity(12)
        self.A[24:27, 9:12] = J_contact_foot[0:3, :]

        self.B[0:12, :] = self.kp * M_inv @ np.identity(12)

        self.Aa[0:self.nx, 0:self.nx] = np.identity(self.nx) + self.ts * self.A
        self.Aa[0:self.nx, self.nx:] = self.ts * self.B

        self.Ba[0:self.nx, :] = self.ts * self.B

        rl_foot = (self.data.oMf[self.model.getFrameId('RL_foot')].translation).reshape(3, 1)

        self.x = np.vstack((self.robot_states.dq, self.robot_states.q, rl_foot, self.robot_states.qr))

    def define_constraints_matrices(self):

        q = self._q

        base_id = self.model.getFrameId("base_link")
        R_base = self.data.oMf[base_id].rotation
        p_base = self.data.oMf[base_id].translation
        n_world = R_base @ -self.n_local
        p0_world = p_base + (R_base @ self.p_offset_local)
        p_knee = self.data.oMf[self.model.getFrameId("RL_calf_joint")].translation
        dist = np.dot(n_world, p_knee - p0_world)

        J_knee = pin.computeFrameJacobian(self.model, self.data, q, self.model.getFrameId("RL_calf_joint"),
                                          pin.LOCAL_WORLD_ALIGNED)[:3, 12:15]

        self.C_cons[13, 9:12] = self.ts * (n_world.T @ J_knee)

        Phi_cons = np.zeros((self.nc * self.N, self.nx + self.nu))
        aux_cons = np.zeros((self.nc, self.nu))

        Phi_cons[:self.nc, :] = self.C_cons @ self.Aa
        aux_cons = self.C_cons @ self.Ba
        if self.first_int:
            l = np.vstack((self.qr_l, 0, self.d_safe - dist))
            u = np.vstack((self.qr_u, 30, np.inf))

            self.l = np.tile(l, (self.N, 1))
            self.u = np.tile(u, (self.N, 1))

        pc = self.data.oMf[self.model.getFrameId("RL_thigh_joint")].translation
        pc[2] = 0.02

        self.ref.reshape(self.N, 14)[:, 11:] = pc
        return aux_cons, Phi_cons

    def update_pred_mdl(self):
        G = np.zeros((self.ny * self.N, self.nu * self.M))
        Phi = np.zeros((self.ny * self.N, self.nx + self.nu))
        aux = np.zeros((self.ny, self.nu))
        aux[:, :] = self.Ca @ self.Ba
        Phi[0:self.ny, :] = self.Ca @ self.Aa

        G_cons = np.zeros((self.nc * self.N, self.nu * self.M))
        aux_cons, Phi_cons = self.define_constraints_matrices()

        for i in range(self.N):
            j = 0
            if i != 0:
                Phi[i * self.ny:(i + 1) * self.ny, :] = Phi[(i - 1) * self.ny:i * self.ny, :] @ self.Aa
                aux[:, :] = Phi[(i - 1) * self.ny:i * self.ny, :] @ self.Ba

                Phi_cons[i * self.nc:(i + 1) * self.nc, :] = Phi_cons[(i - 1) * self.nc:i * self.nc, :] @ self.Aa
                aux_cons[:, :] = Phi_cons[(i - 1) * self.nc:i * self.nc, :] @ self.Ba
            while (j < self.M) and (i + j < self.N):
                G[(i + j) * self.ny:(i + j + 1) * self.ny, j * (self.nu):(j + 1) * (self.nu)] = aux[:, :]
                G_cons[(i + j) * self.nc:(i + j + 1) * self.nc, j * (self.nu):(j + 1) * (self.nu)] = aux_cons[:, :]
                j += 1

        # first index that the "variable block appears"
        var_idx = 13

        for s in range(0, (self.N - 1) * self.nc, self.nc):
            G_cons[var_idx + s + self.nc] += G_cons[var_idx + s]
            Phi_cons[var_idx + s + self.nc] += Phi_cons[var_idx + s]

        return Phi, G, Phi_cons, G_cons
