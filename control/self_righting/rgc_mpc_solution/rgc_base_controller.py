import osqp
import numpy as np
from scipy import sparse
from abc import ABC, abstractmethod

from control.self_righting.rgc_mpc_solution.utils.suppress_output import suppress_output

NCS_MAX = 20


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

        # Number of states, inputs, outputs
        self.nx = None
        self.nu = None
        self.ny = None

        #   nch -> number of hard state/output constraints
        #   ncs -> number of soft constraints (with slack variables)
        #   ncu -> number of hard input constraints (e.g., rate limits on delta q_r)
        self.nch = None
        self.ncs = None
        self.ncu = None

        # Dynamic matrices
        self.A = None
        self.B = None

        # Aumented matrices
        self.Aa = None
        self.Ba = None

        #   Cy  -> tracked outputs (cost function)
        #   Cch -> hard state/output constraints
        #   Ccs -> soft constraints (with slack variables)
        #   Ccu -> hard input constraints (e.g., rate limits on delta q_r)
        self.Cy = None
        self.Cch = None
        self.Ccs = None
        self.Ccu = None

        # Referece vector
        self.ref = None

        # Weight matrices
        self.Q = None
        self.R = None

        # States, ower and upper constarint vector
        self.x = None
        self.lc = None
        self.uc = None

        # Solver
        self.prob = osqp.OSQP()

        self.Phi_y = None
        self.G_y = None

        self.Phi_ch = None
        self.G_ch = None
        self.lch = None
        self.uch = None

        self.Phi_cs = None
        self.G_cs = None
        self.lcs = None
        self.ucs = None

        self.Phi_cu = None
        self.G_cu = None
        self.lcu = None
        self.ucu = None

        self.H = None
        self.F = None

        self.Gc_sparse = None

        self.wcs = None
        self.eps_soft_reg = 1e-6

        # Robot constants

        self.leg_names = ["FR", "FL", "RR", "RL"]

        self.total_mass = self.pin_engine.robot_mass

        joints_lim = self.pin_engine.get_joint_limits()
        torque_lim = self.pin_engine.get_torque_limits()
        self.q_max = joints_lim[:, 1]
        self.q_min = joints_lim[:, 0]

        self.tau_max = torque_lim
        self.tau_min = -torque_lim

        self.Kp_vec = None
        self.Kd_vec = None

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
        aux[:, :] = self.Cy @ self.Ba
        self.Phi_y[0:self.ny, :] = self.Cy @ self.Aa

        aux_ch = None
        aux_cs = None
        aux_cu = None

        if self.nch is not None:
            self.G_ch = np.zeros((self.nch * self.N, self.nu * self.M))
            aux_ch, self.Phi_ch = self.build_hard_constraint_matrices()

        if self.ncs is not None:
            self.G_cs = np.zeros((self.ncs * self.N, self.nu * self.M))
            aux_cs, self.Phi_cs = self.build_soft_constraint_matrices()

        if self.ncu is not None:
            self.G_cu = np.zeros((self.ncu * self.N, self.nu * self.M))
            aux_cu, self.Phi_cu = self.build_input_constraint_matrices()

        for i in range(self.N):
            j = 0
            if i != 0:
                self.Phi_y[i * self.ny:(i + 1) * self.ny, :] = self.Phi_y[(i - 1) * self.ny:i * self.ny, :] @ self.Aa
                aux[:, :] = self.Phi_y[(i - 1) * self.ny:i * self.ny, :] @ self.Ba

                if aux_ch is not None:
                    self.Phi_ch[i * self.nch:(i + 1) *
                                self.nch, :] = self.Phi_ch[(i - 1) * self.nch:i * self.nch, :] @ self.Aa
                    aux_ch[:, :] = self.Phi_ch[(i - 1) * self.nch:i * self.nch, :] @ self.Ba

                if aux_cs is not None:
                    self.Phi_cs[i * self.ncs:(i + 1) *
                                self.ncs, :] = self.Phi_cs[(i - 1) * self.ncs:i * self.ncs, :] @ self.Aa
                    aux_cs[:, :] = self.Phi_cs[(i - 1) * self.ncs:i * self.ncs, :] @ self.Ba

                if aux_cu is not None:
                    self.Phi_cu[i * self.ncu:(i + 1) *
                                self.ncu, :] = self.Phi_cu[(i - 1) * self.ncu:i * self.ncu, :] @ self.Aa
                    aux_cu[:, :] = self.Phi_cu[(i - 1) * self.ncu:i * self.ncu, :] @ self.Ba

            while (j < self.M) and (i + j < self.N):
                self.G_y[(i + j) * self.ny:(i + j + 1) * self.ny, j * (self.nu):(j + 1) * (self.nu)] = aux[:, :]

                if aux_ch is not None:
                    self.G_ch[(i + j) * self.nch:(i + j + 1) * self.nch,
                              j * (self.nu):(j + 1) * (self.nu)] = aux_ch[:, :]

                if aux_cs is not None:
                    self.G_cs[(i + j) * self.ncs:(i + j + 1) * self.ncs,
                              j * (self.nu):(j + 1) * (self.nu)] = aux_cs[:, :]

                if aux_cu is not None:
                    self.G_cu[(i + j) * self.ncu:(i + j + 1) * self.ncu,
                              j * (self.nu):(j + 1) * (self.nu)] = aux_cu[:, :]

                j += 1

    def build_cost_function(self):

        H_dense = self.G_y.T @ self.Q @ self.G_y + self.R

        F_dense = 2 * (((self.Phi_y @ self.x) - self.ref).T) @ self.Q @ self.G_y

        if self.ncs is not None:
            n_eps = self.ncs * self.N
            n_z = self.nu * self.M + n_eps

            H_aug = np.zeros((n_z, n_z))
            H_aug[0:self.nu * self.M, 0:self.nu * self.M] = H_dense
            H_aug[self.nu * self.M:, self.nu * self.M:] = self.eps_soft_reg * np.eye(n_eps)
            H_dense = H_aug

            F_dense = np.hstack((F_dense, self.wcs.reshape(1, -1)))

        self.H = 2 * sparse.csc_matrix(H_dense)

        self.F = F_dense

    def build_constraint_problem(self):

        n_u = self.nu * self.M

        G_blocks = []
        l_blocks = []
        u_blocks = []

        if self.G_ch is not None:
            G_blocks.append(self.G_ch)
            l_blocks.append(self.lch - self.Phi_ch @ self.x)
            u_blocks.append(self.uch - self.Phi_ch @ self.x)

        if self.G_cu is not None:
            G_blocks.append(self.G_cu)
            l_blocks.append(self.lcu - self.Phi_cu @ self.x)
            u_blocks.append(self.ucu - self.Phi_cu @ self.x)

        G = np.vstack(G_blocks)
        l = np.vstack(l_blocks)
        u = np.vstack(u_blocks)

        if self.ncs is not None:
            n_eps = self.ncs * self.N
            inf = np.inf * np.ones((n_eps, 1))
            ident = np.eye(n_eps)

            # Pad hard/input rowcs with zero columns for the slacks
            G = np.hstack((G, np.zeros((G.shape[0], n_eps))))

            ls_shift = self.lcs - self.Phi_cs @ self.x
            us_shift = self.ucs - self.Phi_cs @ self.x

            # Lower side:  G_cs U + eps >= ls_shift
            G_soft_low = np.hstack((self.G_cs, ident))
            # Upper side:  G_cs U - eps <= us_shift
            G_soft_up = np.hstack((self.G_cs, -ident))
            # Slack positivity: eps >= 0
            G_eps = np.hstack((np.zeros((n_eps, n_u)), ident))

            G = np.vstack((G, G_soft_low, G_soft_up, G_eps))
            l = np.vstack((l, ls_shift, -inf, np.zeros((n_eps, 1))))
            u = np.vstack((u, inf, us_shift, inf))

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

        per_row = np.full(NCS_MAX, np.nan)
        if self.ncs is not None and res.x is not None:
            eps = res.x[self.nu * self.M:].reshape(self.N, self.ncs)
            self.task_state.slack_max = np.max(eps)
            per_row[0:self.ncs] = np.max(eps, axis=0)
        else:
            self.task_state.slack_max = 0.0
        self.task_state.slack_max_per_row = per_row

        self.task_state.primal_res = res.info.prim_res

        self.task_state.dual_res = res.info.dual_res

        self.task_state.solver_status = res.info.status_val

    def validate_state(self):

        if self.x is None:
            return False

        if not np.isfinite(self.x).all():
            return False

        if self.ref is None:
            return False

        if self.nch is not None:
            if self.lch is None or self.uch is None:
                return False
            if np.any(self.lch > self.uch):
                return False

        if self.ncs is not None:
            if self.lcs is None or self.ucs is None or self.wcs is None:
                return False
            if np.any(self.lcs > self.ucs):
                return False

        if self.ncu is not None:
            if self.lcu is None or self.ucu is None:
                return False
            if np.any(self.lcu > self.ucu):
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
        if res is None or res.x is None:
            return False
        if res.info.status not in ("solved", "solved inaccurate"):
            return False
        if not np.isfinite(res.x).all():
            return False
        return True

    # def validate_solver_solution(self, res):

    #     if res is None:
    #         return False

    #     if res.x is None:
    #         return False

    #     if res.info.status != "solved":
    #         return False

    #     if not np.isfinite(res.x).all():
    #         return False

    #     return True

    def skew_symmetric_matrix(self, vector):
        v1, v2, v3 = vector
        matrix = np.array([[0, -v3, v2], [v3, 0, -v1], [-v2, v1, 0]])
        return matrix

    def reset_controller(self):
        pass

    def get_gains(self):
        return self.Kp_vec, self.Kd_vec

    def build_hard_constraint_matrices(self):
        return None, None

    def build_soft_constraint_matrices(self):
        return None, None

    def build_input_constraint_matrices(self):
        return None, None

    @abstractmethod
    def update_model(self):
        pass

    @abstractmethod
    def build_reference(self):
        pass
