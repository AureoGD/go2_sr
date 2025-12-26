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

        self._update_detector()

        self.nx = 23
        self.nu = 12
        self.ny = 4 + 3 + 3 + 1  # orientation, q FL pos, q FR pos, romega_x
        self.nc = 13  # max joint pos, GRF pivot

        self.A = np.zeros((self.nx, self.nx), dtype=np.float32)
        self.B = np.zeros((self.nx, self.nu), dtype=np.float32)

        self.Aa = np.zeros((self.nx + self.nu, self.nx + self.nu), dtype=np.float32)
        self.Ba = np.zeros((self.nx + self.nu, self.nu), dtype=np.float32)
        self.Ca = np.zeros((self.ny, self.nx + self.nu), dtype=np.float32)

        self.C_cons = np.zeros((self.nc, self.nx + self.nu), dtype=np.float32)

        self.Aa[self.nx:, self.nx:] = np.identity(self.nu)
        self.Ba[self.nx:, :] = np.identity(self.nu)

        # Body orientation
        self.Ca[0:4, 18:22] = np.eye(4)
        self.Ca[4:7, 6:9] = np.eye(3)
        self.Ca[7:10, 12:15] = np.eye(3)
        self.Ca[10, 0] = 1
        # self.Ca[12, 16] = 1

        self.C_cons[0:12, 3:15] = np.identity(12)
        # self.C_cons[:, 23:] = np.identity(12)

        self.Is = np.concatenate((np.identity(3), np.identity(3), np.identity(3), np.identity(3)), axis=1)

        Qeps = 5 * np.diag(np.array([1, 1, 1, 1]))
        Qfl = np.diag(np.array([0.0001, 0.0001, 0.0001]))
        Qrl = np.diag(np.array([0.0001, 0.0001, 0.0001]))
        Qroll = 0.1

        Q = block_diag(Qeps, Qfl, Qrl, Qroll)

        self.Q1 = block_diag(*[Q] * self.N)

        Qfl = np.diag(np.array([0.1, 0.1, 0.1]))
        Qrl = np.diag(np.array([0.1, 0.1, 0.1]))

        Q = block_diag(Qeps, Qfl, Qrl, Qroll)

        self.Q2 = block_diag(*[Q] * self.N)

        self.Q = self.Q1

        # Update control action weight matrix
        Rdqrfr = np.diag(np.array([5, 1000, 1000]))
        Rdqrfl = 8 * np.diag(np.array([4, 4, 4]))
        Rdqrr = np.diag(np.array([2, 1000, 1000]))
        Rdqrl = 8 * np.diag(np.array([4, 4, 4]))

        R = block_diag(Rdqrfr, Rdqrfl, Rdqrr, Rdqrl)
        self.R = block_diag(*[R] * self.M)

        eps_ref = np.array([0, 0, 0, 1]).reshape(4, 1)
        qref = np.array([0.2, 1.0, -1.5, 0.2, 1.0, -1.5]).reshape(6, 1)

        ref = np.vstack((eps_ref, qref, 0))

        self.ref = np.tile(ref, (self.N, 1))

        qr_l = np.array([
            -1.0472, -1.5708, -2.7227, -1.0472, -1.5708, -2.7227, -1.0472, -0.5236, -2.7227, -1.0472, -0.5236, -2.7227
        ])
        qr_u = np.array(
            [1.0472, 3.4907, -0.83776, 1.0472, 3.4907, -0.83776, 1.0472, 4.5379, -0.83776, 1.0472, 4.5379, -0.83776])

        self.qr_l = qr_l.reshape(12, 1)
        self.qr_u = qr_u.reshape(12, 1)

        foot_l = np.array([-np.inf, -np.inf, 0, 0, 0])
        foot_u = np.array([0, 0, np.inf, np.inf, np.inf])

        # Stack for all 4 feet
        self.f_l = np.tile(foot_l.reshape(-1, 1), (1, 1))  # Shape: (5, 1)
        self.f_u = np.tile(foot_u.reshape(-1, 1), (1, 1))  # Shape: (5, 1)

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

    def update_model(self):
        if self.safe_side:
            self.active = 0
            self.Q = self.Q2

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

        c_fr = self.data.oMf[self.contact_ids[0]].translation
        c_rr = self.data.oMf[self.contact_ids[1]].translation

        c_pivot = (c_fr + c_rr) / 2

        J_p1 = np.zeros((3, 12))
        J_p1[:, 0:3] = pin.computeFrameJacobian(self.model, self.data, q, self.contact_ids[0],
                                                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[0:3, 9:12]

        J_p2 = np.zeros((3, 12))
        J_p2[:, 6:9] = pin.computeFrameJacobian(self.model, self.data, q, self.contact_ids[1],
                                                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[0:3, 15:18]

        J_pivot = (J_p1 + J_p2) / 2

        Jc = np.zeros((12, 12))
        Jc[0:3, 0:3] = pin.computeFrameJacobian(self.model, self.data, q, self.model.getFrameId('FR_foot'),
                                                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[0:3, 9:12]
        pc_fr = self.data.oMf[self.model.getFrameId('FR_foot')].translation
        cross_fr = self.skew_symmetric_matrix(pc_fr - c_pivot)

        Jc[3:6:, 3:6] = pin.computeFrameJacobian(self.model, self.data, q, self.model.getFrameId('FL_foot'),
                                                 pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[0:3, 6:9]
        pc_fl = self.data.oMf[self.model.getFrameId('FL_foot')].translation
        cross_fl = self.skew_symmetric_matrix(pc_fl - pc_fl)

        Jc[6:9, 6:9] = pin.computeFrameJacobian(self.model, self.data, q, self.model.getFrameId('RR_foot'),
                                                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[0:3, 15:18]
        pc_rr = self.data.oMf[self.model.getFrameId('RR_foot')].translation
        cross_rr = self.skew_symmetric_matrix(pc_rr - c_pivot)

        Jc[9:12, 9:12] = pin.computeFrameJacobian(self.model, self.data, q, self.model.getFrameId('RL_foot'),
                                                  pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[0:3, 12:15]
        pc_rl = self.data.oMf[self.model.getFrameId('RL_foot')].translation

        if self.active == 1:
            cross_rl = self.skew_symmetric_matrix(pc_rl - c_pivot)
        else:
            cross_rl = self.skew_symmetric_matrix(pc_rl - pc_rl)

        self.contacts[0, :] = pc_fr
        self.contacts[1, :] = pc_rr
        self.contacts[2, :] = c_fr
        self.contacts[3, :] = c_rr

        J_pivot_stacked = np.vstack((J_pivot, J_pivot, J_pivot, J_pivot))
        gamma = Jc - J_pivot_stacked
        gamma[3:6, :] = np.hstack((np.zeros((3, 3)), (np.eye(3)), np.zeros((3, 6))))
        # gamma[9:, :] = np.hstack((np.zeros((3, 9)), (np.eye(3))))
        Sa = np.vstack((cross_fr, cross_fl, cross_rr, cross_rl))

        gamma_a_star = np.linalg.inv(gamma) @ Sa

        I_com = self.data.Ig.inertia
        mass = self.data.mass[0]

        lever = r.flatten() - c_pivot  # Vector from Pivot to CoM
        S = self.skew_symmetric_matrix(lever)

        I_pivot = I_com + mass * (S.T @ S)
        Iinv = np.linalg.inv(I_pivot)

        comp_grav = S @ np.array((0, 0, mass))

        self.Jinv = np.linalg.inv(Jc)

        k3 = self.kp * Iinv @ Sa.T @ self.Jinv
        k4 = self.kd * Iinv @ Sa.T @ self.Jinv

        self.A[0:3, 0:3] = k4 @ gamma_a_star
        self.A[0:3, 3:15] = k3

        self.A[3:15, 0:3] = gamma_a_star

        self.A[15:18, 0:3] = -S + J_com @ gamma_a_star

        self.A[18:22, 0:3] = T.reshape(4, 3)

        self.B[0:3, 0:12] = -k3

        self.Aa[0:23, 0:23] = np.identity(self.nx) + self.ts * self.A
        self.Aa[0:23, 23:] = self.ts * self.B

        self.Ba[0:23, :] = self.ts * self.B

        self.Ba[6:9, 3:6] = np.eye(3)
        self.Ba[12:15, 9:12] = np.eye(3) * (1 - self.active)

        self.Ba[15:18, 3:6] = J_com[:, 3:6]
        self.Ba[15:18, 9:12] = J_com[:, 9:] * (1 - self.active)

        self.x = np.vstack((self.robot_states.omega, self.robot_states.q, self.robot_states.r_pos,
                            self.robot_states.epsilon, -9.81, self.robot_states.qr))

    def define_constraints_matrices(self):

        # --- PART 1: DEFINE STRUCTURE (Once) ---
        if self.first_int:
            plane_normal, _ = self.get_best_fit_normal(self.contacts)

            pivot_vec = self.contacts[2] - self.contacts[3]
            pivot_dir = pivot_vec / np.linalg.norm(pivot_vec)

            vec_vertical = np.array([0, 0, 1])

            n_stab = np.cross(vec_vertical, pivot_vec)
            n_stab = n_stab / np.linalg.norm(n_stab)

            self.C_cons[12, 15:18] = n_stab

            self.pivot_offset = np.dot(n_stab, self.contacts[3])

            q_ref, R_ref = self.eps_reference(plane_normal=plane_normal, pivot_direction=pivot_dir)
            self.ref.reshape(self.N, self.ny)[:, 0:4] = q_ref

            l = np.vstack((self.qr_l, -np.inf))
            u = np.vstack((self.qr_u, np.inf))
            self.l = np.tile(l, (self.N, 1))
            self.u = np.tile(u, (self.N, 1))

            self.first_int = False

        curr_com = self.robot_states.r_pos.flatten()

        curr_dist = np.dot(self.C_cons[12, 15:18], curr_com) - self.pivot_offset

        if curr_dist < 0:
            stability_lb = -np.inf
        else:
            self.safe_side = True
            stability_lb = 0.0

        l_reshaped = self.l.reshape(self.N, self.nc)
        l_reshaped[:, 12] = stability_lb + self.pivot_offset
        # self.l = l_reshaped.flatten()

        Phi_cons = np.zeros((self.nc * self.N, self.nx + self.nu))
        aux_cons = np.zeros((self.nc, self.nu))

        Phi_cons[0:self.nc, :] = self.C_cons @ self.Aa
        aux_cons = self.C_cons @ self.Ba

        return aux_cons, Phi_cons

    # def define_constraints_matrices(self):

    #     if self.first_int:
    #         plane_normal, _ = self.get_best_fit_normal(self.contacts)
    #         pivot_vec = self.contacts[2] - self.contacts[3]
    #         pivot_dir = pivot_vec / np.linalg.norm(pivot_vec)

    #         q_ref, R_ref = self.eps_reference(plane_normal=plane_normal, pivot_direction=pivot_dir)

    #         self.ref.reshape(self.N, self.ny)[:, 0:4] = q_ref
    #         self.l = np.tile(self.qr_l, (self.N, 1))
    #         self.u = np.tile(self.qr_u, (self.N, 1))
    #         self.first_int = False

    #     Phi_cons = np.zeros((self.nc * self.N, self.nx + self.nu))
    #     aux_cons = np.zeros((self.nc, self.nu))

    #     Phi_cons[0:self.nc, :] = self.C_cons @ self.Aa
    #     aux_cons = self.C_cons @ self.Ba

    #     return aux_cons, Phi_cons

    def center_of_mass_constraint(self):
        l = (self.robot_states.r_pos[0:2, 0]).reshape(2, 1) - 0.1 * np.ones((2, 1))
        u = (self.robot_states.r_pos[0:2, 0]).reshape(2, 1) + 0.1 * np.ones((2, 1))

        return l, u

    def cont_surfaces(self, c1, c2, c3):
        v1 = c2 - c1
        v2 = c3 - c1
        n = np.cross(v1, v2)
        if n[2] < 0:
            n = -n
        n = n / np.linalg.norm(n)
        t1 = v1 / np.linalg.norm(v1)
        t2 = np.cross(n, t1)

        n = np.array([0, 0, 1])
        t1 = np.array([1, 0, 0])
        t2 = np.array([0, 1, 0])

        return n, t1, t2

    def cf_matrix(self, n, t1, t2, mu):
        Cf = np.vstack([-mu * n + t1, -mu * n + t2, mu * n + t2, mu * n + t1, n])

        return Cf

    def create_com_constraint(self, margin=0.05):
        """
        Generates A, b for constraints based on the explicit order 
        of self.contacts (0->1, 1->2, 2->0...).
        """
        contacts_2d = np.array([contact[:2] for contact in self.contacts])
        n_vertices = len(contacts_2d)
        centroid = np.mean(contacts_2d, axis=0)

        A = []
        b = []

        for i in range(n_vertices):
            v1 = contacts_2d[i]
            v2 = contacts_2d[(i + 1) % n_vertices]

            edge_vec = v2 - v1

            normal = np.array([edge_vec[1], -edge_vec[0]])

            normal_norm = np.linalg.norm(normal)
            if normal_norm > 1e-10:
                normal_unit = normal / normal_norm
            else:
                continue  # Skip zero-length edges

            center_to_edge = v1 - centroid

            if np.dot(normal_unit, center_to_edge) < 0:
                normal_unit = -normal_unit

            A.append(normal_unit)
            b.append(np.dot(normal_unit, v1) - margin)

        return np.array(A), np.array(b)

    def get_best_fit_normal(self, points):
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        u, s, vh = np.linalg.svd(centered)
        normal = vh[2, :]
        if normal[2] < 0:
            normal = -normal
        return normal, centroid
