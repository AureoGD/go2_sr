import osqp
import numpy as np
from scipy import sparse
from abc import ABC, abstractmethod

from control.self_righting.rgc_mpc_solution.utils.suppress_output import suppress_output


class BaseRGCController(ABC):

    def __init__(self, robot_states, **kwargs):

        self.rs = robot_states.robot
        self.cs = robot_states.low_level
        self.dg = robot_states.debug
        self.pin_engine = kwargs.get("pin_engine")
        self.task_state = kwargs.get("task_state")
        self.kp = kwargs.get("kp", 50.0)
        self.kd = kwargs.get("kd", 3.0)

        self.phase = None

        # Predic and control horizons and sampe time
        self.N = None
        self.M = None
        self.ts = None

        # Number of states, inputs, outputs and constarints
        self.nx = None
        self.nu = None
        self.ny = None
        self.nc = None

        # Dynamic matrices
        self.A = None
        self.B = None

        # Aumented matrices
        self.Aa = None
        self.Ba = None
        self.Ca = None

        # Constraint matrics
        self.Cc = None

        # Referece vector
        self.ref = None

        # Weight matrices
        self.Q = None
        self.R = None

        # States, ower and upper constarint vector
        self.x = None
        self.l = None
        self.u = None

        # Solver
        self.prob = osqp.OSQP()

        self.Phi_y = None
        self.G_y = None

        self.Phi_cy = None
        self.G_cy = None

        self.Phi_cu = None
        self.G_cu = None

        self.lu = None
        self.uu = None

        self.H = None
        self.F = None

        self.Gc_sparse = None

        self.lc = None
        self.uc = None

        # Robot constants

        self.leg_names = ["FR", "FL", "RR", "RL"]

        self.total_mass = self.pin_engine.robot_mass

        joints_lim = self.pin_engine.get_joint_limits()
        torque_lim = self.pin_engine.get_torque_limits()
        self.q_max = joints_lim[:, 1]
        self.q_min = joints_lim[:, 0]

        self.tau_max = torque_lim
        self.tau_min = -torque_lim

    def update_dqr(self):

        try:

            with suppress_output():

                self.update_model()

                self.build_reference()

                self.build_mpc_problem()

                self.build_cost_function()

                self.build_constraint_problem()

                if not self.validate_state():
                    return self._handle_mpc_failure(critical=True)

                self.setup_solver()

                res = self.solve_qp()

                self.update_solver_diagnostics(res)

                if not self.validate_solver_solution(res):
                    return self._handle_mpc_failure(critical=False)

                return self._handle_mpc_success(res)

        except Exception:
            return self._handle_mpc_failure(critical=True)

    def build_mpc_problem(self):
        self.G_y = np.zeros((self.ny * self.N, self.nu * self.M))
        self.Phi_y = np.zeros((self.ny * self.N, self.nx + self.nu))
        aux = np.zeros((self.ny, self.nu))

        # Calculate initial blocks
        aux[:, :] = self.Ca @ self.Ba
        self.Phi_y[0:self.ny, :] = self.Ca @ self.Aa

        self.G_cy = np.zeros((self.nc * self.N, self.nu * self.M))
        aux_cons, self.Phi_cy = self.build_output_constraint_matrices()
        self.G_cu, self.Phi_cu = self.build_input_constraint_matrices()

        for i in range(self.N):
            j = 0
            if i != 0:
                # Prediction propagation
                self.Phi_y[i * self.ny:(i + 1) * self.ny, :] = self.Phi_y[(i - 1) * self.ny:i * self.ny, :] @ self.Aa
                aux[:, :] = self.Phi_y[(i - 1) * self.ny:i * self.ny, :] @ self.Ba

                self.Phi_cy[i * self.nc:(i + 1) * self.nc, :] = self.Phi_cy[(i - 1) * self.nc:i * self.nc, :] @ self.Aa
                aux_cons[:, :] = self.Phi_cy[(i - 1) * self.nc:i * self.nc, :] @ self.Ba

            while (j < self.M) and (i + j < self.N):
                self.G_y[(i + j) * self.ny:(i + j + 1) * self.ny, j * (self.nu):(j + 1) * (self.nu)] = aux[:, :]
                self.G_cy[(i + j) * self.nc:(i + j + 1) * self.nc, j * (self.nu):(j + 1) * (self.nu)] = aux_cons[:, :]
                j += 1

    def build_cost_function(self):

        H_dense = self.G_y.T @ self.Q @ self.G_y + self.R

        F_dense = 2 * (((self.Phi_y @ self.x) - self.ref).T) @ self.Q @ self.G_y

        self.H = 2 * sparse.csc_matrix(H_dense)

        self.F = F_dense

    def build_constraint_problem(self):

        if self.G_cu is not None:
            G = sparse.csc_matrix(np.vstack((self.G_cy, self.G_cu)))
            l = np.vstack((self.l - self.Phi_cy @ self.x, self.lu))
            u = np.vstack((self.u - self.Phi_cy @ self.x, self.uu))
        else:
            G = self.G_cy
            l = self.l - self.Phi_cy @ self.x
            u = self.u - self.Phi_cy @ self.x

        self.Gc_sparse = sparse.csc_matrix(G)
        self.lc = l
        self.uc = u

    def setup_solver(self):

        self.prob = osqp.OSQP()

        self.prob.setup(self.H,
                        self.F.T,
                        A=self.Gc_sparse,
                        l=self.lc,
                        u=self.uc,
                        verbose=False,
                        warm_start=True,
                        max_iter=2000,
                        check_termination=10)

    def solve_qp(self):
        return self.prob.solve()

    def update_solver_diagnostics(self, res):

        if res.y is not None:
            self.task_state.lambda_max = np.max(np.abs(res.y))
        else:
            self.task_state.lambda_max = 0.0

        self.task_state.primal_res = res.info.prim_res

        self.task_state.dual_res = res.info.dual_res

    def validate_state(self):

        if self.x is None:
            return False

        if not np.isfinite(self.x).all():
            return False

        if self.ref is None:
            return False

        if self.l is None or self.u is None:
            return False

        if np.any(self.l > self.u):
            return False

        return True

    def _handle_mpc_failure(self, critical=True):
        self.task_state.mpc_fail = True
        self.task_state.mpc_critical_fail = critical
        self.dqr = np.zeros((self.nu,), dtype=np.float32)
        return self.dqr

    def _handle_mpc_success(self, res):
        self.dqr = res.x[0:self.nu]

        self.task_state.mpc_obj_val = res.info.obj_val

        self.task_state.mpc_fail = False

        self.task_state.mpc_critical_fail = False

        return self.dqr

    def validate_solver_solution(self, res):

        if res is None:
            return False

        if res.x is None:
            return False

        if res.info.status != "solved":
            return False

        if not np.isfinite(res.x).all():
            return False

        return True

    def skew_symmetric_matrix(self, vector):
        v1, v2, v3 = vector
        matrix = np.array([[0, -v3, v2], [v3, 0, -v1], [-v2, v1, 0]])
        return matrix

    def reset_controller(self):
        pass

    def get_gains(self):
        return self.Kp_vec, self.Kd_vec

    @abstractmethod
    def build_output_constraint_matrices(self):
        pass

    @abstractmethod
    def update_model(self):
        pass

    @abstractmethod
    def build_reference(self):
        pass

    def build_input_constraint_matrices(self):
        return None, None
