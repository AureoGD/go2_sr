import numpy as np
import pinocchio as pin
from scipy.linalg import block_diag
from sr_strategies.rgc.base_controller import BaseRGC
from scipy.spatial import ConvexHull
import faulthandler
import traceback
import signal
import sys


class RollCW(BaseRGC):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.N = 20
        self.M = 15
        self.ts = 0.01

        self.nx = 29
        self.nu = 12
        self.ny = 12
        self.nc = 5  #GRF

        self.A = np.zeros((self.nx, self.nx), dtype=np.float32)
        self.B = np.zeros((self.nx, self.nu), dtype=np.float32)

        self.Aa = np.zeros((self.nx + self.nu, self.nx + self.nu), dtype=np.float32)
        self.Ba = np.zeros((self.nx + self.nu, self.nu), dtype=np.float32)
        self.Ca = np.zeros((self.ny, self.nx + self.nu), dtype=np.float32)

        self.C_cons = np.zeros((self.nc, self.nx + self.nu), dtype=np.float32)

        self.A[18:21, 0:3] = np.identity(3)
        self.A[2, 25] = 1

        self.Aa[self.nx:, self.nx:] = np.identity(self.nu)
        self.Ba[self.nx:, :] = np.identity(self.nu)

        # Body orientation
        self.Ca[:, 29:] = np.identity(12)

        self.contacts = np.zeros((3, 3), dtype=np.float32)

        self.Is = np.concatenate((np.identity(3), np.identity(3), np.identity(3)), axis=1)

        Qqr = 0.5 * np.eye(12)

        Q = block_diag(Qqr)

        self.Q = block_diag(*[Q] * self.N)

        # Update control action weight matrix
        Rdqrfr = np.diag(np.array([1, 1, 1]))
        Rdqrfl = np.diag(np.array([1, 1, 1]))
        Rdqrr = np.diag(np.array([1, 1, 1]))
        Rdqrl = np.diag(np.array([1, 1, 1]))

        R = block_diag(Rdqrfr, Rdqrfl, Rdqrr, Rdqrl)
        self.R = 10 * block_diag(*[R] * self.M)

        # -0.6, 1.5, -2.0, -0.8, 1.0, -2.6, -0.6, 1.25, -2.0

        # qrRef = np.array([-0.25, 0.90, -2.85, -0.85, 0.85, -1.3, -0.25, 0.90, -2.85, 0.6, 3.75, -1.5]).reshape(12, 1)
        qrRef = np.array([-0.6, 1.5, -2.0, -0.8, 1.0, -2.6, -0.6, 1.5, -2.0, 0.6, 3.75, -1.5]).reshape(12, 1)

        ref = np.vstack((qrRef))

        self.ref = np.tile(ref, (self.N, 1))

        self.L = np.zeros((3, self.nx + self.nu))
        self.L[0:3, 15:18] = -self.kp * np.identity(3)
        self.L[0:3, 38:] = self.kp * np.identity(3)

        # Constraints for one foot
        foot_l = np.array([-np.inf, -np.inf, 0, 0, 40])
        foot_u = np.array([0, 0, np.inf, np.inf, 150])

        # Only RL foot
        self.f_l = np.tile(foot_l.reshape(-1, 1), (1, 1))  # Shape: (15, 1)
        self.f_u = np.tile(foot_u.reshape(-1, 1), (1, 1))  # Shape: (15, 1)

        self.Jinv = np.zeros((9, 9), dtype=np.float32)

        # (FR=3, FL=0, RR=9, RL=6)
        # Contact at front-right foot, rear-right foot and rear-left foot
        self.leg_idx = [3, 9, 6]

        self.contact_ids = [
            self.model.getFrameId('FR_foot'),
            self.model.getFrameId('RR_foot'),
            self.model.getFrameId('RL_foot'),
        ]

        self.first_int = True

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

        M = self.data.M[6:9, 6:9]
        C = self.data.C[6:9, 6:9]
        r = self.data.com[0]
        dr = self.data.vcom[0]

        # Save states
        self.robot_states.r_vel = dr.reshape(3, 1)
        self.robot_states.r_pos = r.reshape(3, 1)

        # Update angular velocity to quaternions matrix
        x, y, z, w = self.robot_states.epsilon
        T = 0.5 * np.array([[w, z, -y], [-z, w, x], [y, -x, w], [-x, -y, -z]])

        # Get Centroidal Inertia's 3x3 rotational part
        I = self.data.Ig.inertia
        Iinv = np.linalg.inv(I)

        # Get CoM Jacobian (from step 3)
        J_com_full = self.data.Jcom[0:3, 6:]

        J_com = np.hstack([
            J_com_full[:, self.leg_idx[0]:self.leg_idx[0] + 3],  # front-lef leg
            J_com_full[:, self.leg_idx[1]:self.leg_idx[1] + 3],  # rind-right leg
            J_com_full[:, self.leg_idx[2]:self.leg_idx[2] + 3]  # rind-left leg
        ])

        J_com_stacked = np.vstack([J_com, J_com, J_com])

        J_fl_com = np.vstack([J_com_full[:, 0:3], J_com_full[:, 0:3], J_com_full[:, 0:3]])

        # --- 3. Build the 9x9 Gamma (Gamma) and 9x3 Sa ---
        Gamma = np.zeros((9, 9), dtype=np.float32)
        Jc = np.zeros((9, 9), dtype=np.float32)
        Sa = np.zeros((9, 3), dtype=np.float32)

        for i in range(3):
            contact_id = self.contact_ids[i]
            leg_q_start_idx = self.leg_idx[i]

            Jc_full = pin.computeFrameJacobian(self.model, self.data, q, contact_id,
                                               pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[0:3, 6 + leg_q_start_idx:9 +
                                                                                       leg_q_start_idx]

            Jc[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = Jc_full

            contact_pos = self.data.oMf[contact_id].translation
            Sa[i * 3:(i + 1) * 3, :] = self.skew_symmetric_matrix(contact_pos - r)

            self.contacts[i, :] = contact_pos

        # Compute the 9x9 singular Gamma matrix
        Gamma = J_com_stacked - Jc

        # --- 4. Build the 3x9 Loop Constraint (J_loop) ---
        J_loop = np.zeros((3, 9), dtype=np.float32)

        J_task_plus = np.linalg.pinv(Gamma)  # 9x12 pseudoinverse

        # --- 6. Build the Right-Hand-Side Mappings ---
        # Build the 12x3 linear mapping vector
        I_stack = np.zeros((9, 3), dtype=np.float32)
        I_stack[0:9, :] = np.vstack([np.eye(3), np.eye(3), np.eye(3)])  # 9x3 part

        # Build the 12x3 angular mapping vector
        S_stack = np.zeros((9, 3), dtype=np.float32)
        S_stack[0:9, :] = Sa

        # --- 7. Calculate final gamma_l and gamma_a ---
        # These are the final 9x3 mapping matrices for your 9x1 dq vector
        gamma_l_star = J_task_plus @ I_stack
        gamma_a_star = J_task_plus @ S_stack
        gamma_q_star = J_task_plus @ J_fl_com

        # self.Jinv = np.linalg.pinv(Jc.T)

        self.Jinv[1, 0] = 0.9 / 0.1
        self.Jinv[2, 0] = 1 / 0.1

        self.Jinv[4, 3] = 0.9 / 0.1
        self.Jinv[5, 3] = 1 / 0.1

        J_rl = Jc[6:, 6:]
        self.Jinv[6:, 6:] = np.linalg.inv(J_rl).T

        k1 = (self.kp / self.total_mass) * self.Is @ self.Jinv
        k2 = (self.kd / self.total_mass) * self.Is @ self.Jinv
        k3 = self.kp * Iinv @ -Sa.T @ self.Jinv
        k4 = self.kd * Iinv @ -Sa.T @ self.Jinv

        self.A[0:3, 0:3] = k2 @ gamma_l_star
        self.A[0:3, 3:6] = -k2 @ gamma_a_star
        self.A[0:3, 6:9] = k1[:, 0:3]
        self.A[0:3, 12:18] = k1[:, 3:]
        self.A[0:3, 26:] = -k2 @ gamma_q_star

        self.A[3:6, 0:3] = k4 @ gamma_l_star
        self.A[3:6, 3:6] = -k4 @ gamma_a_star
        self.A[3:6, 6:9] = k3[:, 0:3]
        self.A[3:6, 12:18] = k3[:, 3:]
        self.A[3:6, 26:] = -k4 @ gamma_q_star

        self.A[6:9, 0:3] = gamma_l_star[0:3, :]
        self.A[6:9, 3:6] = -gamma_a_star[0:3, :]
        self.A[6:9, 26:] = -gamma_q_star[0:3, :]

        # q_fl is junst the integral of dq_fl
        self.A[9:12, 26:] = np.eye(3)

        self.A[12:18, 0:3] = gamma_l_star[3:, :]
        self.A[12:18, 3:6] = -gamma_a_star[3:, :]
        self.A[12:18, 26:] = -gamma_q_star[3:, :]

        self.A[21:25, 3:6] = T.reshape(4, 3)

        M_inv = np.linalg.inv(M)

        self.A[26:, 9:12] = -M_inv * self.kp
        self.A[26:, 26:] = -M_inv @ (C + self.kd / 10 * np.eye(3))

        self.B[0:3, 0:3] = -k1[:, 0:3]
        self.B[0:3, 6:] = -k1[:, 3:]

        self.B[3:6, 0:3] = -k3[:, 0:3]
        self.B[3:6, 6:] = -k3[:, 3:]

        self.B[26:, 3:6] = self.kp * M_inv

        self.Aa[0:self.nx, 0:self.nx] = np.identity(self.nx) + self.ts * self.A
        self.Aa[0:self.nx, self.nx:] = self.ts * self.B

        self.Ba[0:self.nx, :] = self.ts * self.B

        self.x = np.vstack(
            (self.robot_states.r_vel, self.robot_states.omega, self.robot_states.q, self.robot_states.r_pos,
             self.robot_states.epsilon, -9.81, self.robot_states.dq[3:6], self.robot_states.qr))

        # RL
        self.L[:, 0:3] = -self.kd * gamma_l_star[6:9, :]
        self.L[:, 3:6] = self.kd * gamma_a_star[6:9, :]
        self.L[:, 26:29] = self.kd * gamma_q_star[6:9, :]

    def define_constraints_matrices(self):

        if self.first_int:
            self.robot_states.contacts[0:3, :] = self.contacts
            # Ac, bc = self.create_com_constraint()
            # l = np.vstack((-np.inf, self.f_l))
            # u = np.vstack((bc[0], self.f_u))

            self.l = np.tile(self.f_l, (self.N, 1))
            self.u = np.tile(self.f_u, (self.N, 1))

            # self.C_cons[0, 18:20] = Ac[0, :]

            self.first_int = False

        Phi_cons = np.zeros((self.nc * self.N, self.nx + self.nu))
        aux_cons = np.zeros((self.nc, self.nu))

        n_rl, t1_rl, t2_rl = self.cont_surfaces(self.contacts[1, :], self.contacts[2, :], self.contacts[0, :])
        mu = 0.7 / np.sqrt(2)

        # 5x3
        Cf_rl = self.cf_matrix(n_rl, t1_rl, t2_rl, mu)

        Cf = block_diag(Cf_rl)

        Fc_max = -Cf @ self.Jinv[6:9, 6:9]

        self.C_cons[0:, :] = Fc_max @ self.L
        Phi_cons[0:self.nc, :] = self.C_cons @ self.Aa
        aux_cons = self.C_cons @ self.Ba

        return aux_cons, Phi_cons

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
