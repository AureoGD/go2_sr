import numpy as np
import pinocchio as pin
from scipy.linalg import block_diag
from environment.strategies.rgc_mpc.base_controller import BaseRGC


class LandingCW(BaseRGC):

    TASK_NAME = "landing_cw"
    TASK_LEVEL = 4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not self.runtime:
            # Metadata-only: nothing else to do
            return

        self.N = 20
        self.M = 10
        self.ts = 0.01
        self.ws = 10
        self.convergence_threshold = 0.05
        self.sigma = 0

        self._update_detector()

        self.nx = 29
        self.nu = 12
        self.ny = 16
        self.nc = 12  # max joint pos, GRF z component

        self.A = np.zeros((self.nx, self.nx), dtype=np.float32)
        self.B = np.zeros((self.nx, self.nu), dtype=np.float32)

        self.Aa = np.zeros((self.nx + self.nu, self.nx + self.nu), dtype=np.float32)
        self.Ba = np.zeros((self.nx + self.nu, self.nu), dtype=np.float32)
        self.Ca = np.zeros((self.ny, self.nx + self.nu), dtype=np.float32)

        self.C_cons = np.zeros((self.nc, self.nx + self.nu), dtype=np.float32)

        self.Aa[self.nx:, self.nx:] = np.identity(self.nu)
        self.Ba[self.nx:, :] = np.identity(self.nu)

        # Body orientation
        self.Ca[:3, 3:6] = np.eye(3)  # FR joints
        self.Ca[3:6, 9:12] = np.eye(3)  # RR joints
        self.Ca[6:10, 18:22] = np.eye(4)  # Quaternions
        self.Ca[10:13, 23:26] = np.eye(3)  # FL foot (P1)
        self.Ca[13:16, 26:29] = np.eye(3)  # RL foot (P2)

        self.C_cons[0:12, self.nx:] = np.identity(12)

        self.Is = np.concatenate((np.identity(3), np.identity(3), np.identity(3), np.identity(3)), axis=1)

        Qr = np.diag(np.array([1, 1, 1]))  # FR and RR joints
        Qeps = np.diag(np.array([1, 1, 1, 1]))  # Quaternions
        Qposfl = np.diag(np.array([5, 3, 5]))  # FL foot (P1)
        Qposrl = np.diag(np.array([8, 8, 4]))  # RL foot (P2)

        Q = block_diag(Qr, Qr, Qeps, Qposfl, Qposrl)
        self.Q = block_diag(*[Q] * self.N)

        # use latter to update self.Q
        self.single_output_dim = 19  # = 19
        self.idx_RL1 = slice(13, 16)  # = 13:16
        self.idx_RL2 = slice(16, 19)  # = 16:19

        # Update control action weight matrix
        Rdqrfr = 750 * np.diag(np.array([1, 10, 10]))
        Rdqrfl = 0.9 * np.diag(np.array([0.9, 1, 1]))
        Rdqrr = 750 * np.diag(np.array([1, 10, 10]))
        Rdqrl = 0.9 * np.diag(np.array([1, 1, 1]))

        R = block_diag(Rdqrfr, Rdqrfl, Rdqrr, Rdqrl)
        self.R = block_diag(*[R] * self.M)

        qr = np.array([0.4, 1.5, -2.0, 0.4, 1.5, -2.0]).reshape(6, 1)
        epsr = np.array([0, 0, 0, 1]).reshape(4, 1)
        pos_foot = np.array([0, 0, 0]).reshape(3, 1)
        ref = np.vstack((qr, epsr, pos_foot, pos_foot))
        self.ref = np.tile(ref, (self.N, 1))

        qr_l = np.array([
            -1.0472, -1.5708, -2.7227, -1.0472, -1.5708, -2.7227, -1.0472, -0.5236, -2.7227, -1.0472, -0.5236, -2.7227
        ])
        qr_u = np.array(
            [1.0472, 3.4907, -0.83776, 1.0472, 3.4907, -0.83776, 1.0472, 4.5379, -0.83776, 1.0472, 4.5379, -0.83776])

        self.qr_l = qr_l.reshape(12, 1)
        self.qr_u = qr_u.reshape(12, 1)

        self.Jinv = np.zeros((12, 12), dtype=np.float32)

        # (FR=3, FL=0, RR=9, RL=6)
        # Contact at front-right foot, rear-right foot and rear-left foot
        self.leg_idx = [3, 9]

        self.contact_ids = [
            self.model.getFrameId('FR_thigh_joint'),
            self.model.getFrameId('RR_thigh_joint'),
        ]

        self.contacts = np.zeros((4, 3), dtype=np.float32)

        self.first_int = True

        self.active = 1
        self.safe_side = False

        self.P2 = None
        self.P3 = None

    def update_model(self):

        q, dq = self.ordering_joints()

        # 1. COMPUTE ALL KINEMATICS (Positions and Velocities)
        # Do this once at the beginning.
        pin.forwardKinematics(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data)

        # 2. COMPUTE JOINT-SPACE MATRICES
        pin.crba(self.model, self.data, q)
        pin.computeCoriolisMatrix(self.model, self.data, q, dq)

        # 3. COMPUTE ALL CENTROIDAL QUANTITIES
        pin.ccrba(self.model, self.data, q, dq)

        J_com_full = pin.jacobianCenterOfMass(self.model, self.data, q, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:, 6:]

        J_com = np.hstack([
            J_com_full[:, 3:6],  # front-right leg
            J_com_full[:, 0:3],  # front-left
            J_com_full[:, 9:12],  # rear-right leg
            J_com_full[:, 6:9]  # rear-left leg
        ])

        r = self.data.com[0]
        dr = self.data.vcom[0]

        # Save states
        self.robot_states.r_vel = dr.reshape(3, 1)
        self.robot_states.r_pos = r.reshape(3, 1)

        # Update angular velocity to quaternions matrix

        x, y, z, w = self.robot_states.epsilon
        T = 0.5 * np.array([[w, z, -y], [-z, w, x], [y, -x, w], [-x, -y, -z]])

        pivot_fr = self.data.oMf[self.model.getFrameId('FR_hip_joint')].translation
        pivot_rr = self.data.oMf[self.model.getFrameId('RR_hip_joint')].translation
        mean_pivot = (pivot_fr + pivot_rr) / 2

        Jc = np.zeros((12, 12))
        Jc[0:3, 0:3] = pin.computeFrameJacobian(self.model, self.data, q, self.model.getFrameId('FR_foot'),
                                                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[0:3, 9:12]
        foot_fr = self.data.oMf[self.model.getFrameId('FR_foot')].translation
        cross_fr = self.skew_symmetric_matrix(foot_fr - pivot_fr)

        Jc[3:6:, 3:6] = pin.computeFrameJacobian(self.model, self.data, q, self.model.getFrameId('FL_foot'),
                                                 pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[0:3, 6:9]
        foot_fl = self.data.oMf[self.model.getFrameId('FL_foot')].translation

        Jc[6:9, 6:9] = pin.computeFrameJacobian(self.model, self.data, q, self.model.getFrameId('RR_foot'),
                                                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[0:3, 15:18]
        foot_rr = self.data.oMf[self.model.getFrameId('RR_foot')].translation
        cross_rr = self.skew_symmetric_matrix(foot_rr - pivot_rr)

        Jc[9:12, 9:12] = pin.computeFrameJacobian(self.model, self.data, q, self.model.getFrameId('RL_foot'),
                                                  pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[0:3, 12:15]
        foot_rl = self.data.oMf[self.model.getFrameId('RL_foot')].translation

        self.contacts[0, :] = foot_fr
        self.contacts[1, :] = self.data.oMf[self.model.getFrameId('FR_thigh_joint')].translation
        self.contacts[2, :] = foot_rr
        self.contacts[3, :] = self.data.oMf[self.model.getFrameId('RR_thigh_joint')].translation

        cross_fl = np.zeros((3, 3))
        cross_rl = np.zeros((3, 3))
        Sa = np.vstack((cross_fr, cross_fl, cross_rr, cross_rl))

        gamma = Jc.copy()
        gamma[3:6, :] = np.hstack((np.zeros((3, 3)), (np.eye(3)), np.zeros((3, 6))))
        gamma[9:12, :] = np.hstack((np.zeros((3, 9)), (np.eye(3))))

        gamma_a_star = np.linalg.inv(gamma) @ Sa

        self.Jinv = np.linalg.inv(Jc.T)

        I_com = self.data.Ig.inertia
        mass = self.data.mass[0]

        c_pivot = (pivot_fr + pivot_rr) / 2
        lever = self.data.com[0] - c_pivot
        S = self.skew_symmetric_matrix(lever)

        I_pivot = I_com + mass * (S.T @ S)
        I_inv = np.linalg.inv(I_pivot)

        comp_grav = S @ np.array([0, 0, mass])
        term_grav = I_inv @ comp_grav

        k3 = self.kp * I_inv @ gamma_a_star.T
        k4 = self.kd * I_inv @ gamma_a_star.T

        self.A[0:3, 0:3] = -k4 @ gamma_a_star
        self.A[0:3, 3:15] = -k3
        self.A[0:3, 22] = term_grav

        self.A[3:15, 0:3] = gamma_a_star

        self.A[15:18, 0:3] = -S + J_com @ gamma_a_star

        self.A[18:22, 0:3] = T.reshape(4, 3)

        self.B[0:3, :] = k3

        self.Aa[0:self.nx, 0:self.nx] = np.identity(self.nx) + self.ts * self.A
        self.Aa[0:self.nx, self.nx:] = self.ts * self.B

        self.Ba[0:self.nx, :] = self.ts * self.B

        self.Ba[6:9, 3:6] = self.ts * np.eye(3)
        self.Ba[12:15, 9:12] = self.ts * np.eye(3)

        self.Ba[15:18, 3:6] = J_com[:, 3:6]
        self.Ba[15:18, 9:12] = J_com[:, 9:]
        self.Ba[23:26,
                3:6] = self.ts * pin.computeFrameJacobian(self.model, self.data, q, self.model.getFrameId('FL_foot'),
                                                          pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[0:3, 6:9]
        self.Ba[26:29,
                9:12] = self.ts * pin.computeFrameJacobian(self.model, self.data, q, self.model.getFrameId('RL_foot'),
                                                           pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[0:3, 12:15]

        self.world_M_base = self.data.oMf[self.model.getFrameId('base_link')].copy()

        self.x = np.vstack(
            (self.robot_states.omega, self.robot_states.q, self.robot_states.r_pos, self.robot_states.epsilon, -9.81,
             foot_fl.reshape(3, 1), foot_rl.reshape(3, 1), self.robot_states.qr))

    def define_constraints_matrices(self):

        if self.first_int:
            n, _ = self._plane_normal(self.contacts)
            current_yaw = self.robot_states.r_pos[2, 0]
            quat_ref, R_ref = self.eps_reference(current_yaw=current_yaw, plane_normal=n)
            self.ref.reshape(self.N, self.ny)[:, 6:10] = quat_ref

            fl_ref, rl_ref1, rl_ref2 = self.feet_references(n=n, R=0.27)

            self.ref.reshape(self.N, self.ny)[:, 10:13] = fl_ref

            self.P2 = rl_ref1
            self.P3 = rl_ref2

            l = self.qr_l
            u = self.qr_u
            self.l = np.tile(l, (self.N, 1))
            self.u = np.tile(u, (self.N, 1))

            self.first_int = False

        sigma = self.compute_sigma()
        traking_point = (1 - sigma) * self.P2 + sigma * self.P3

        self.robot_states.pc_debug[1, :] = traking_point

        self.ref.reshape(self.N, self.ny)[:, 13:16] = traking_point

        Phi_cons = np.zeros((self.nc * self.N, self.nx + self.nu))
        aux_cons = np.zeros((self.nc, self.nu))

        Phi_cons[0:self.nc, :] = self.C_cons @ self.Aa
        aux_cons = self.C_cons @ self.Ba

        return aux_cons, Phi_cons

    def compute_sigma(self, s0=0.45, lookahead=0.2):
        """
            s0: distance threshold (meters) to consider 'near P2'
            lookahead: distance (meters) the target leads ahead of the robot projection
            """
        P2 = self.P2.flatten()
        P3 = self.P3.flatten()
        x = self.x[26:29].flatten()

        dist_to_P2 = np.linalg.norm(x[:2] - P2[:2])

        if self.sigma == 0.0 and dist_to_P2 > s0:
            return 0.0

        d_xy = P3[:2] - P2[:2]
        L = np.linalg.norm(d_xy)

        if L < 1e-6:
            return 1.0

        u_vec = d_xy / L

        s_robot = float(u_vec @ (x[:2] - P2[:2]))
        s_target = s_robot + lookahead

        # Normalize to 0..1 range
        sigma_raw = np.clip(s_target / L, 0.0, 1.0)

        self.sigma = max(self.sigma, sigma_raw)

        return self.sigma
