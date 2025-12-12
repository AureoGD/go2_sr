import numpy as np
import pinocchio as pin
from scipy.linalg import block_diag
from sr_strategies.rgc.base_controller import BaseRGC

import faulthandler
import traceback
import signal
import sys


class RollCCW(BaseRGC):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.N = 5
        self.M = 1
        self.ts = 0.01

        self.nx = 29  #
        self.nu = 12
        # self.ny = 13  # Epsilon, qr_fl, F_fr, F_rr
        self.ny = 7  # Epsilon, qr_fl
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
        self.Ca[0:4, 21:25] = np.identity(4)
        # FR joint
        self.Ca[4:7, 29:32] = np.identity(3)

        self.contacts = np.zeros((3, 3), dtype=np.float32)

        self.Is = np.concatenate((np.identity(3), np.identity(3), np.identity(3)), axis=1)

        Qeps = 0.5 * np.eye(4)
        Qqrlf = 0.06 * np.eye(3)

        # Qeps = 0.01 * np.eye(4)
        # Qqrlf = 0.05 * np.eye(3)

        # Qfgrf = np.diag((0.00001, 0.0000001, 0.0000001, 0.00001, 0.0000001, 0.0000001))
        # Qfgrf = np.diag((0.000001, 0.00000001, 0.00000001, 0.000001, 0.00000001, 0.00000001))

        Q = block_diag(Qeps, Qqrlf)

        self.Q = block_diag(*[Q] * self.N)

        # Update control action weight matrix
        dqrWeight_fl = np.array([5, 5, 2])
        dqrWeight = np.array([1, 1, 1])
        Rdqr_fr = np.diag(dqrWeight_fl)
        Rdqr = np.diag(dqrWeight)
        R = block_diag(Rdqr_fr, Rdqr, Rdqr, Rdqr)
        self.R = block_diag(*[R] * self.M)

        epsRef = np.array([0, 0, 0, 1]).reshape(4, 1)
        qflRef = np.array([0, 0, 1.0]).reshape(3, 1)

        ref = np.vstack((epsRef, qflRef))

        self.ref = np.tile(ref, (self.N, 1))

        # GRF vector:
        # Constraints for one foot
        foot_l = np.array([-np.inf, -np.inf, 0, 0, 20])
        foot_u = np.array([0, 0, np.inf, np.inf, 150])

        # Only RL foot
        self.f_l = np.tile(foot_l.reshape(-1, 1), (1, 1))  # Shape: (5, 1)
        self.f_u = np.tile(foot_u.reshape(-1, 1), (1, 1))  # Shape: (5, 1)

        self.Jinv = np.zeros((9, 9), dtype=np.float32)

        # (FR=3, FL=0, RR=9, RL=6)
        # Contact at front-left shoulder, rind-right foot and rind-left shoulder
        self.leg_idx = [0, 9, 6]

        self.contact_ids = [
            self.model.getFrameId('FL_shoulder'),
            self.model.getFrameId('RR_foot'),
            self.model.getFrameId('RL_shoulder'),
        ]

        # number of the contacts x 3
        self.L = np.zeros((9, self.nx + self.nu), dtype=np.float32)
        self.L[:, 9:18] = -self.kp * np.identity(9)
        self.L[:, 32:] = self.kp * np.identity(9)

        self.aux = np.zeros((6, 9), dtype=np.float32)
        self.aux[0:3, 0:3] = np.eye(3)
        self.aux[3:, 6:] = np.eye(3)

        self.first_int = True

    def update_model(self):
        q, dq = self.ordering_joints()

        # 1. COMPUTE ALL KINEMATICS (Positions and Velocities)
        # Do this once at the beginning.
        pin.forwardKinematics(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data)

        # 2. COMPUTE JOINT-SPACE MATRICES
        # This computes data.M
        pin.crba(self.model, self.data, q)
        # This computes data.C
        pin.computeCoriolisMatrix(self.model, self.data, q, dq)

        # 3. COMPUTE ALL CENTROIDAL QUANTITIES
        # This one function call computes:
        # - CoM position (data.com[0])
        # - CoM velocity (data.vcom[0])
        # - Centroidal Inertia 6x6 (data.Ig)
        # - CoM Jacobian 6xnv (data.Jcom)
        # - Centroidal Momentum (data.hg)
        pin.ccrba(self.model, self.data, q, dq)

        # --- 4. Get all data from the 'data' struct ---
        # All calculations are now done and are consistent.

        # Get M and C (from step 2)
        M = self.data.M[9:12, 9:12]
        C = self.data.C[9:12, 9:12]

        # Get CoM state (from step 3)
        r = self.data.com[0]
        dr = self.data.vcom[0]

        # Save states
        self.robot_states.r_vel = dr.reshape(3, 1)
        self.robot_states.r_pos = r.reshape(3, 1)

        x, y, z, w = self.robot_states.epsilon
        T = 0.5 * np.array([[w, z, -y], [-z, w, x], [y, -x, w], [-x, -y, -z]])

        # Get Centroidal Inertia's 3x3 rotational part (from step 3)
        # data.Ig is the 6x6 spatial inertia. .inertia gets the 3x3 block
        I = self.data.Ig.inertia
        Iinv = np.linalg.inv(I)

        # Get CoM Jacobian (from step 3)
        J_com_full = self.data.Jcom[0:3, 6:]

        # ... rest of your code is correct ...
        J_com = np.hstack([
            J_com_full[:, self.leg_idx[0]:self.leg_idx[0] + 3],  # front-lef leg
            J_com_full[:, self.leg_idx[1]:self.leg_idx[1] + 3],  # rind-right leg
            J_com_full[:, self.leg_idx[2]:self.leg_idx[2] + 3]  # rind-left leg
        ])

        J_com_stacked = np.vstack([J_com, J_com, J_com])

        J_com_fr = np.vstack([J_com_full[:, 3:6], J_com_full[:, 3:6], J_com_full[:, 3:6]])

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

            # Store contact position (from your original code)
            self.contacts[i, :] = contact_pos

        # Compute the 9x9 singular Gamma matrix
        Gamma = J_com_stacked - Jc

        # --- 4. Build the 3x9 Loop Constraint (J_loop) ---
        J_loop = np.zeros((3, 9), dtype=np.float32)

        # Get 3x3 FL foot jacobian
        J_fl_foot = pin.computeFrameJacobian(self.model, self.data, q, self.foot_ids[1],
                                             pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[0:3, 6 + self.leg_idx[0]:9 +
                                                                                     self.leg_idx[0]]

        # Get 3x3 RL foot jacobian
        J_rl_foot = pin.computeFrameJacobian(self.model, self.data, q, self.foot_ids[3],
                                             pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[0:3, 6 + self.leg_idx[2]:9 +
                                                                                     self.leg_idx[2]]

        # Build J_loop = [J_fl_foot| 0 | -J_rl_foot]
        J_loop[:, 0:3] = J_fl_foot  # Columns for FL leg
        # Columns 3-5 are already zero (for RL leg)
        J_loop[:, 6:9] = -J_rl_foot  # Columns for RL leg

        # --- 5. Build and Solve the Stacked System ---

        J_task = np.vstack([Gamma, J_loop])  # 12x9 matrix
        J_task_plus = np.linalg.pinv(J_task)  # 9x12 pseudoinverse

        # --- 6. Build the Right-Hand-Side Mappings ---

        # Build the 12x3 linear mapping vector
        I_stack = np.zeros((12, 3), dtype=np.float32)
        I_stack[0:9, :] = np.vstack([np.eye(3), np.eye(3), np.eye(3)])  # 9x3 part
        # The last 3x3 block is zero (for J_loop's 0 target)

        # Build the 12x3 angular mapping vector
        S_stack = np.zeros((12, 3), dtype=np.float32)
        S_stack[0:9, :] = Sa  # 9x3 part
        # The last 3x3 block is zero (for J_loop's 0 target)0

        # Build the 12x3 FR joint mapping vector
        J_fr_stack = np.zeros((12, 3), dtype=np.float32)
        J_fr_stack[0:9, :] = J_com_fr  # 9x3 part
        # The last 3x3 block is zero (for J_loop's 0 target)0

        # --- 7. Calculate final gamma_l and gamma_a ---
        # These are the final 9x3 mapping matrices for your 9x1 dq vector
        gamma_l_star = J_task_plus @ I_stack
        gamma_a_star = J_task_plus @ S_stack
        gamma_q_star = J_task_plus @ J_fr_stack

        self.Jinv = np.linalg.pinv(Jc.T)

        k1 = (self.kp / self.total_mass) * self.Is @ self.Jinv
        k2 = (self.kd / self.total_mass) * self.Is @ self.Jinv
        k3 = self.kp * Iinv @ -Sa.T @ self.Jinv
        k4 = self.kd * Iinv @ -Sa.T @ self.Jinv

        self.A[0:3, 0:3] = k2 @ gamma_l_star
        self.A[0:3, 3:6] = -k2 @ gamma_a_star
        self.A[0:3, 9:18] = k1
        self.A[0:3, 26:] = -k2 @ gamma_q_star

        self.A[3:6, 0:3] = k4 @ gamma_l_star
        self.A[3:6, 3:6] = -k4 @ gamma_a_star
        self.A[3:6, 9:18] = k3
        self.A[3:6, 26:] = -k4 @ gamma_q_star

        self.A[6:9, 26:] = np.eye(3)
        self.A[9:18, 0:3] = gamma_l_star
        self.A[9:18, 3:6] = -gamma_a_star
        self.A[9:18, 26:] = -gamma_q_star

        self.A[21:25, 3:6] = T.reshape(4, 3)

        M_inv = np.linalg.inv(M)

        self.A[26:, 6:9] = -self.kp * M_inv
        self.A[26:, 26:] = -M_inv @ (C + self.kd / 10 * np.eye(3))

        self.B[0:3, 3:12] = -k1
        self.B[3:6, 3:12] = -k3
        self.B[26:, 0:3] = self.kp * M_inv

        self.Aa[0:self.nx, 0:self.nx] = np.identity(self.nx) + self.ts * self.A
        self.Aa[0:self.nx, self.nx:] = self.ts * self.B

        self.Ba[0:self.nx, :] = self.ts * self.B

        # dr, omega, q, r, eps, qr, g
        self.x = np.vstack(
            (self.robot_states.r_vel, self.robot_states.omega, self.robot_states.q, self.robot_states.r_pos,
             self.robot_states.epsilon, -9.81, self.robot_states.dq[0:3], self.robot_states.qr))

        # GRF FL leg
        self.L[:, 0:3] = -self.kd * gamma_l_star
        self.L[:, 3:6] = self.kd * gamma_a_star
        self.L[:, 26:29] = self.kd * gamma_q_star

    def define_constraints_matrices(self):

        Phi_cons = np.zeros((self.nc * self.N, self.nx + self.nu))
        aux_cons = np.zeros((self.nc, self.nu))

        n_rl, t1_rl, t2_rl = self.cont_surfaces(self.contacts[1, :], self.contacts[2, :], self.contacts[0, :])
        mu = 0.9 / np.sqrt(2)

        # 5x3
        Cf_rl = self.cf_matrix(n_rl, t1_rl, t2_rl, mu)

        Cf = block_diag(Cf_rl)

        Fc_max = -Cf @ self.Jinv[3:6, 3:6]

        self.C_cons = Fc_max @ self.L[3:6, :]
        Phi_cons[0:self.nc, :] = self.C_cons @ self.Aa
        aux_cons = self.C_cons @ self.Ba
        if self.first_int:
            self.l = np.tile(self.f_l, (self.N, 1))
            self.u = np.tile(self.f_u, (self.N, 1))
            self.first_int = False

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
