from scipy import sparse
import pinocchio as pin
import numpy as np
import osqp


class BaseRGC():

    def __init__(self, **kwargs):

        self.model = kwargs.get('model')
        self.data = kwargs.get('data')
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

    def solve_rgc(self):
        self.update_model()
        Phi, G, Phi_cons, G_cons = self.update_pred_mdl()

        H_dense = G.T @ self.Q @ G + self.R
        F_dense = 2 * (((Phi @ self.x) - self.ref).T) @ self.Q @ G

        # Convert to sparse safely
        H = 2 * sparse.csc_matrix(H_dense)
        G_cons_sparse = sparse.csc_matrix(G_cons)
        F = F_dense  # Keep as dense for now

        self.prob = osqp.OSQP()

        self.prob.setup(H,
                        F.T,
                        A=G_cons_sparse,
                        l=self.l - Phi_cons @ self.x,
                        u=self.u - Phi_cons @ self.x,
                        verbose=False)

        # if not self.op_init:
        #     self.prob.setup(H,
        #                     F.T.flatten(),
        #                     A=sparse.csc_matrix(G_cons_sparse),
        #                     l=self.l - Phi_cons @ self.x,
        #                     u=self.u - Phi_cons @ self.x,
        #                     verbose=False)
        #     self.op_init = True
        # else:
        #     self.prob.update(q=F.T, l=self.l - Phi_cons @ self.x, u=self.u - Phi_cons @ self.x)
        #     self.prob.update(Px=H, A=G_cons_sparse)

        res = self.prob.solve()
        if res.info.status != "solved":
            self.dqr = np.zeros((12, 1), dtype=np.float32)
        else:
            self.dqr = res.x[0:self.nu]
        # self.dqr = np.zeros((12, 1), dtype=np.float32)
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
        # Phi_cons = np.zeros((self.nc * self.N, self.nx + self.nu))
        # aux_cons = np.zeros((self.nc, self.nu))

        aux_cons, Phi_cons = self.define_constraints_matrices()
        # print("Cons. Mtx")

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
        """
        Get the total mass of the robot model
        """
        total_mass = 0.0
        for inertia in self.model.inertias:
            total_mass += inertia.mass
        return total_mass

    def skew_symmetric_matrix(self, vector):
        v1, v2, v3 = vector
        matrix = np.array([[0, -v3, v2], [v3, 0, -v1], [-v2, v1, 0]])
        return matrix
