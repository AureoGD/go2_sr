import numpy as np
import pinocchio as pin
from scipy.linalg import block_diag
from sr_strategies.rgc.base_controller import BaseRGC

import faulthandler
import traceback
import signal
import sys


class StandUpPhase(BaseRGC):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.N = 15
        self.M = 5
        self.ts = 0.01

        self.nx = 26
        self.nu = 12
        # self.ny = 16
        self.ny = 5
        # self.nc = 2  # constraint rx and ry
        self.nc = 22  # rx, ry, GRF

        self.A = np.zeros((self.nx, self.nx), dtype=np.float32)
        self.B = np.zeros((self.nx, self.nu), dtype=np.float32)

        self.Aa = np.zeros((self.nx + self.nu, self.nx + self.nu), dtype=np.float32)
        self.Ba = np.zeros((self.nx + self.nu, self.nu), dtype=np.float32)
        self.Ca = np.zeros((self.ny, self.nx + self.nu), dtype=np.float32)

        self.C_cons = np.zeros((self.nc, self.nx + self.nu), dtype=np.float32)

        self.A[18:21, 0:3] = np.identity(3)
        self.A[2, 25] = 1

        self.Aa[26:, 26:] = np.identity(self.nu)
        self.Ba[26:, :] = np.identity(self.nu)

        #  joint references and body orientation
        # self.Ca[0:12, 6:18] = np.identity(12)  # q
        # self.Ca[12:, 21:25] = np.identity(4)  # epsilon

        # CoM z position and body orientatio
        self.Ca[0, 20] = 1
        self.Ca[1:, 21:25] = np.identity(4)

        # self.C_cons[:, 18:20] = np.identity(2)  # rx and rz
        self.C_cons[0:2, 18:20] = np.identity(2)

        self.contacts = np.zeros((4, 3), dtype=np.float32)

        self.Is = np.concatenate((np.identity(3), np.identity(3), np.identity(3), np.identity(3)), axis=1)

        self.L = np.zeros((12, 38), dtype=np.float32)
        self.L[:, 6:18] = -self.kp * np.identity(12)
        self.L[:, 26:] = self.kp * np.identity(12)

        #  Update output weight matrices
        # jointWeight = np.array([0.05, 0.05, 0.05])
        # Qjoint = np.diag(jointWeight)
        # Qeps = 1 * np.eye(4)
        # Q = block_diag(Qjoint, Qjoint, Qjoint, Qjoint, Qeps)

        Qrz = np.array([1])
        # Qjoint = np.diag(jointWeight)
        Qeps = 0.5 * np.eye(4)
        Q = block_diag(Qrz, Qeps)

        self.Q = block_diag(*[Q] * self.N)

        self.Q = block_diag(*[Q] * self.N)

        # Update control action weight matrix
        dqrWeight = np.array([1, 1, 1])
        Rdqr = np.diag(dqrWeight)
        R = block_diag(Rdqr, Rdqr, Rdqr, Rdqr)
        self.R = block_diag(*[R] * self.M)

        # Update the reference vector:
        # qRef = np.array([[0, 0.5, -1.05, 0, 0.5, -1.05, 0, 0.5, -1.05, 0, 0.5, -1.05]]).reshape(12, 1)
        # epsRef = np.array([0, 0, 0, 1]).reshape(4, 1)
        # ref = np.vstack((qRef, epsRef))
        # self.ref = np.tile(ref, (self.N, 1))

        rzRef = np.array([[0.25]]).reshape(1, 1)
        epsRef = np.array([0, 0, 0, 1]).reshape(4, 1)
        ref = np.vstack((rzRef, epsRef))
        self.ref = np.tile(ref, (self.N, 1))

        # GRF vector:
        # Constraints for one foot
        foot_l = np.array([-np.inf, -np.inf, 0, 0, 30])
        foot_u = np.array([0, 0, np.inf, np.inf, 75])

        # Stack for all 4 feet
        self.f_l = np.tile(foot_l.reshape(-1, 1), (4, 1))  # Shape: (20, 1)
        self.f_u = np.tile(foot_u.reshape(-1, 1), (4, 1))  # Shape: (20, 1)

        self.Jinv = None

        self.first_int = True

    def update_model(self):

        q = np.vstack((self.robot_states.b_pos, self.robot_states.epsilon, self.robot_states.q))
        dq = np.vstack((self.robot_states.b_vel, self.robot_states.omega, self.robot_states.dq))

        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pin.ccrba(self.model, self.data, q, np.zeros(self.model.nv))
        r = pin.centerOfMass(self.model, self.data, q)

        #  Evaluate the CoM velocity
        centroidal_momentum = pin.computeCentroidalMomentum(self.model, self.data, q, dq)
        dr = centroidal_momentum.linear / self.data.mass[0]

        # Save states
        self.robot_states.r_vel = dr.reshape(3, 1)
        self.robot_states.r_pos = r.reshape(3, 1)

        x, y, z, w = self.robot_states.epsilon

        T = 0.5 * np.array([[w, z, -y], [-z, w, x], [y, -x, w], [-x, -y, -z]])

        I = self.data.Ig.inertia
        Iinv = np.linalg.inv(I)

        J_com = pin.jacobianCenterOfMass(self.model, self.data, q, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:, 6:]

        Jc = np.zeros((12, 12), dtype=np.float32)
        gamma = np.zeros((12, 12), dtype=np.float32)
        Sa = np.zeros((3, 12), dtype=np.float32)

        for i, _ in enumerate(self.legs):
            Jc[i * 3:(i + 1) * 3, :] = pin.computeFrameJacobian(self.model, self.data, q, self.foot_ids[i],
                                                                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[0:3, 6:]
            gamma[i * 3:(i + 1) * 3, :] = J_com - Jc[i * 3:(i + 1) * 3, :]
            foot_pos = self.data.oMf[self.foot_ids[i]].translation
            Sa[:, i * 3:(i + 1) * 3] = self.skew_symmetric_matrix(foot_pos - r)
            self.contacts[i, :] = foot_pos

        self.Jinv = (np.linalg.inv(Jc.T))

        k1 = (self.kp / self.total_mass) * self.Is @ self.Jinv
        k2 = (self.kd / self.total_mass) * self.Is @ self.Jinv
        k3 = self.kp * Iinv @ Sa @ self.Jinv
        k4 = self.kd * Iinv @ Sa @ self.Jinv

        gamma = np.linalg.inv(gamma)
        gamma_l_star = np.zeros((12, 3), dtype=np.float32)
        gamma_a_star = np.zeros((12, 3), dtype=np.float32)

        for i in range(4):
            gamma_l_star += gamma[:, 3 * i:3 * (i + 1)]
            gamma_a_star += gamma[:, 3 * i:3 * (i + 1)] @ Sa[:, 3 * i:3 * (i + 1)]

        self.A[0:3, 0:3] = k2 @ gamma_l_star
        self.A[0:3, 3:6] = -k2 @ gamma_a_star
        self.A[0:3, 6:18] = k1

        self.A[3:6, 0:3] = k4 @ gamma_l_star
        self.A[3:6, 3:6] = -k4 @ gamma_a_star
        self.A[3:6, 6:18] = k3

        self.A[6:18, 0:3] = gamma_l_star
        self.A[6:18, 3:6] = -gamma_a_star

        self.A[21:25, 3:6] = T.reshape(4, 3)

        self.B[0:3, 0:12] = -k1
        self.B[3:6, 0:12] = -k3

        self.Aa[0:26, 0:26] = np.identity(self.nx) + self.ts * self.A
        self.Aa[0:26, 26:] = self.ts * self.B

        self.Ba[0:26, :] = self.ts * self.B

        # dr, omega, q, r, eps, qr, g
        self.x = np.vstack((self.robot_states.r_vel, self.robot_states.omega, self.robot_states.q,
                            self.robot_states.r_pos, self.robot_states.epsilon, -9.81, self.robot_states.qr))

        self.L[:, 0:3] = -self.kd * gamma_l_star
        self.L[:, 3:6] = self.kp * gamma_a_star

        # self.define_constraints_matrices(Jinv)s

    def define_constraints_matrices(self):

        Phi_cons = np.zeros((self.nc * self.N, self.nx + self.nu))
        aux_cons = np.zeros((self.nc, self.nu))

        n_fl, t1_fl, t2_fl = self.cont_surfaces(self.contacts[0, :], self.contacts[1, :], self.contacts[2, :])
        n_fr, t1_fr, t2_fr = self.cont_surfaces(self.contacts[1, :], self.contacts[2, :], self.contacts[3, :])
        n_rl, t1_rl, t2_rl = self.cont_surfaces(self.contacts[2, :], self.contacts[3, :], self.contacts[0, :])
        n_rr, t1_rr, t2_rr = self.cont_surfaces(self.contacts[3, :], self.contacts[0, :], self.contacts[1, :])
        mu = 0.9 / np.sqrt(2)
        # 5x3
        Cf_fl = self.cf_matrix(n_fl, t1_fl, t2_fl, mu)
        Cf_fr = self.cf_matrix(n_fr, t1_fr, t2_fr, mu)
        Cf_rl = self.cf_matrix(n_rl, t1_rl, t2_rl, mu)
        Cf_rr = self.cf_matrix(n_rr, t1_rr, t2_rr, mu)
        # # 20 x 12
        Cf = block_diag(Cf_fl, Cf_fr, Cf_rl, Cf_rr)
        Fc_mtx = -Cf @ self.Jinv
        aux_cons[0:2, :] = self.C_cons[0:2, :] @ self.Ba
        aux_cons[2:, :] = self.kp * Fc_mtx
        self.C_cons[2:, :] = Fc_mtx @ self.L
        Phi_cons[0:self.nc, :] = self.C_cons @ self.Aa
        if self.first_int:
            l, u = self.center_of_mass_constraint()
            l = np.vstack((l, self.f_l))
            u = np.vstack((u, self.f_u))
            self.l = np.tile(l, (self.N, 1))
            self.u = np.tile(u, (self.N, 1))
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
