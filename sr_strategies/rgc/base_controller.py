from scipy import sparse
import pinocchio as pin

import numpy as np
import osqp


class BaseRGC():

    def __init__(self, **kwargs):

        self.model = kwargs.get('model')
        self.data = kwargs.get('data')
        self.geo_model = kwargs.get('geo_model')
        self.geo_data = kwargs.get('geo_data')
        self.links = kwargs.get('links')  # links sequence
        self.legs = kwargs.get('legs')  # legs sequence
        self.links_ids = kwargs.get('links_ids')
        self.foot_ids = kwargs.get('foot_ids')
        self.kp = kwargs.get('kp')
        self.kd = kwargs.get('kd')
        self.robot_states = kwargs.get('robot_states')
        self.total_mass = self.get_total_mass()
        self.i = 0

        self.N = None
        self.M = None
        self.ts = None
        self.A = None
        self.B = None
        self.Aa = None
        self.Ba = None
        self.Ca = None
        self.C_cons = None
        self.ref = None
        self.nu = None
        self.nx = None
        self.ny = None
        self.nc = None
        self.Q = None
        self.R = None
        self.x = None
        self.l = None
        self.u = None
        self.prob = osqp.OSQP()
        self.op_init = False
        self.task_finish = False

    def update_dqr(self):
        try:
            self.update_model()
            Phi, G, Phi_cons, G_cons = self.update_pred_mdl()

            H_dense = G.T @ self.Q @ G + self.R
            F_dense = 2 * (((Phi @ self.x) - self.ref).T) @ self.Q @ G

            # Convert to sparse
            H = 2 * sparse.csc_matrix(H_dense)
            G_cons_sparse = sparse.csc_matrix(G_cons)
            F = F_dense  # Keep as dense for now

            self.prob = osqp.OSQP()

            # Setup OSQP with error handling
            import io
            import sys
            from contextlib import redirect_stderr

            error_occurred = False
            stderr_capture = io.StringIO()

            with redirect_stderr(stderr_capture):
                try:
                    self.prob.setup(H,
                                    F.T,
                                    A=G_cons_sparse,
                                    l=self.l - Phi_cons @ self.x,
                                    u=self.u - Phi_cons @ self.x,
                                    verbose=False,
                                    warm_start=True)
                except Exception as e:
                    error_occurred = True

            # Check for OSQP errors in stderr
            captured_errors = stderr_capture.getvalue()
            if "ERROR in osqp_setup:" in captured_errors:
                error_occurred = True
                # print(f"OSQP setup error: {captured_errors.strip()}")

            if error_occurred:
                self.robot_states.mpc_fail = True
                self.robot_states.critical_mpc_fail = True  # New critical flag
                self.dqr = np.zeros((self.nu,), dtype=np.float32)
                return self.dqr

            # Try to solve
            res = self.prob.solve()

            if res.info.status != "solved":
                self.robot_states.mpc_fail = True
                self.dqr = np.zeros((self.nu,), dtype=np.float32)
            else:
                if abs(res.info.obj_val) < 0.0001:
                    self.robot_states.subtask_succes = True
                self.robot_states.mpc_obj_val = res.info.obj_val
                self.robot_states.mpc_fail = False
                self.robot_states.critical_mpc_fail = False
                self.dqr = res.x[0:self.nu]
                # print(res.info.solve_time)
            return self.dqr

        except Exception as e:
            self.robot_states.mpc_fail = True
            self.robot_states.critical_mpc_fail = True
            self.dqr = np.zeros((self.nu,), dtype=np.float32)
            return self.dqr

    def update_model(self):
        raise NotImplementedError("Subclasses must implement update_model() method")

    def define_constraints_matrices(self):
        raise NotImplementedError("Subclasses must implement define_constraints_matrices() method")

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

        return Phi, G, Phi_cons, G_cons

    def get_total_mass(self):
        total_mass = 0.0
        for inertia in self.model.inertias:
            total_mass += inertia.mass
        return total_mass

    def skew_symmetric_matrix(self, vector):
        v1, v2, v3 = vector
        matrix = np.array([[0, -v3, v2], [v3, 0, -v1], [-v2, v1, 0]])
        return matrix

    def ordering_joints(self):
        "This method ordering the robot joint postions and velocities accoding to the pinocchio's framework"
        q = np.vstack((self.robot_states.b_pos, self.robot_states.epsilon, self.robot_states.q[3:6],
                       self.robot_states.q[0:3], self.robot_states.q[9:12], self.robot_states.q[6:9]))

        dq = np.vstack((self.robot_states.b_vel, self.robot_states.omega, self.robot_states.dq[3:6],
                        self.robot_states.dq[0:3], self.robot_states.dq[9:12], self.robot_states.dq[6:9]))

        return q, dq

    def com_quatities(self):
        q, dq = self.ordering_joints()
        pin.ccrba(self.model, self.data, q, dq)

        r = self.data.com[0]
        dr = self.data.vcom[0]

        # Save states
        self.robot_states.r_vel = dr.reshape(3, 1)
        self.robot_states.r_pos = r.reshape(3, 1)

    def reset_controller(self):
        self.task_finish = False
