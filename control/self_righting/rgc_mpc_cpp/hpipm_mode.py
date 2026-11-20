"""HPIPM (C++) solver for the Python RGC modes: step 1 of the mode-by-mode port.

`make_hpipm_mode(PythonModeClass)` returns a subclass of the Python mode in which ONLY the QP
solve runs in C++ (rgc_mpc_cpp/_rgc_mpc_cpp: stage-wise QP + HPIPM). Everything else is the
unchanged Python mode: model update, references, constraint matrices, diagnostics.

The condensed problem the mode builds for OSQP,
    min_z  z' H z + F z     s.t.  l <= G z <= u,      z = [U (nu*M) ; eps (ncs*N)]
is handed to HPIPM in its stage-wise (sparse) form: same cost, same constraints, same soft
constraints. The result is returned in the OSQP result layout (res.x, res.y, res.info), so
update_solver_diagnostics / validate_solver_solution / _handle_mpc_success work as before.
"""
import time
from types import SimpleNamespace

import numpy as np

try:
    from control.self_righting.rgc_mpc_cpp._rgc_mpc_cpp import HpipmStageQpSolver
except ImportError as e:  # pragma: no cover - build hint
    raise ImportError("rgc_mpc_cpp C++ module not built: run control/self_righting/rgc_mpc_cpp/build.sh") from e

# OSQP-compatible status codes (osqp: 1 solved, 2 solved inaccurate, -2 max iter, -10 unsolved)
_SOLVED = (1, "solved")
_INACCURATE = (2, "solved inaccurate")
_MAX_ITER = (-2, "maximum iterations reached")
_UNSOLVED = (-10, "unsolved")


class HpipmSolverMixin:
    """Mixin placed before a BaseRGCController subclass in the MRO."""

    hpipm_tol = 1e-6
    hpipm_iter_max = 100
    hpipm_mode = "speed_abs"
    # HPIPM "max iter" / "min step" results with residuals below this are accepted as
    # "solved inaccurate" (OSQP's "solved inaccurate" was accepted too)
    hpipm_accept_res = 1e-4

    def __init__(self, robot_states, **kwargs):
        super().__init__(robot_states, **kwargs)
        self.warm_start_enabled = False  # interior point: the OSQP warm start does not apply
        self._hp_solver = None
        self._hp_key = None
        self._hp_args = None
        self._hp_const = 0.0
        self.last_tracking_cost = np.nan  # sum_k |Cy s_{k+1} - ref_k|^2_Q at the solution
        self.hpipm_stats = {"calls": 0, "fails": 0, "iterations": 0, "solve_time": 0.0, "build_time": 0.0}

    # ------------------------------------------------------------------
    # The dense condensed cost / constraint matrices are only needed by OSQP
    # ------------------------------------------------------------------
    def build_cost_function(self):
        pass

    def build_constraint_problem(self):
        pass

    # ------------------------------------------------------------------
    def setup_solver(self):
        t0 = time.perf_counter()
        self._hp_args, key = self._build_stage_problem()
        if self._hp_key != key:
            N, nx, nu, ng, soft_rows = key
            self._hp_solver = HpipmStageQpSolver(N, nx, nu, ng, list(soft_rows), tol=self.hpipm_tol,
                                                 iter_max=self.hpipm_iter_max, mode=self.hpipm_mode)
            self._hp_key = key
        self.hpipm_stats["build_time"] += time.perf_counter() - t0

    def solve_qp(self):
        sol = self._hp_solver.solve(**self._hp_args)
        return self._to_osqp_result(sol)

    # ------------------------------------------------------------------
    # condensed (Python) -> stage-wise (C++)
    # ------------------------------------------------------------------
    def _build_stage_problem(self):
        N, M, nu, ny = self.N, self.M, self.nu, self.ny
        if M != N:
            raise NotImplementedError("HPIPM path: control horizon M must equal N")
        if self.terminal or self.G_term is not None:
            raise NotImplementedError("HPIPM path: terminal constraints not implemented")

        A = np.asarray(self.Aa, dtype=np.float64)
        B = np.asarray(self.Ba, dtype=np.float64)
        Cy = np.asarray(self.Cy, dtype=np.float64)
        Q = np.asarray(self.Q, dtype=np.float64)
        R = np.asarray(self.R, dtype=np.float64)
        ref = np.asarray(self.ref, dtype=np.float64).reshape(N, ny)
        x0 = np.asarray(self.x, dtype=np.float64).reshape(-1)

        Qy = np.vstack([Q[k * ny:(k + 1) * ny, k * ny:(k + 1) * ny] for k in range(N)])
        Rb = np.vstack([R[k * nu:(k + 1) * nu, k * nu:(k + 1) * nu] for k in range(N)])

        # Constraint rows at stage k act on s_{k+1}: C = C_out A, D = C_out B are the first
        # blocks of the condensed matrices (Phi_c[0], G_c[0]) built by build_mpc_problem.
        C_rows, D_rows, lg_rows, ug_rows = [], [], [], []
        for n, Phi, G, lo, up in ((self.nch, self.Phi_ch, self.G_ch, self.lch, self.uch),
                                  (self.ncu, self.Phi_cu, self.G_cu, self.lcu, self.ucu)):
            if n is None or G is None:
                continue
            C_rows.append(Phi[:n])
            D_rows.append(G[:n, :nu])
            lg_rows.append(np.asarray(lo, dtype=np.float64).reshape(N, n))
            ug_rows.append(np.asarray(up, dtype=np.float64).reshape(N, n))

        soft_rows = ()
        zl = np.zeros((N, 0))
        if self.ncs is not None:
            n_hard = sum(c.shape[0] for c in C_rows)
            soft_rows = tuple(range(n_hard, n_hard + self.ncs))
            C_rows.append(self.Phi_cs[:self.ncs])
            D_rows.append(self.G_cs[:self.ncs, :nu])
            lg_rows.append(np.asarray(self.lcs, dtype=np.float64).reshape(N, self.ncs))
            ug_rows.append(np.asarray(self.ucs, dtype=np.float64).reshape(N, self.ncs))
            zl = np.asarray(self.wcs, dtype=np.float64).reshape(N, self.ncs)

        nx = A.shape[0]
        if C_rows:
            C = np.vstack(C_rows)
            D = np.vstack(D_rows)
            lg = np.hstack(lg_rows)
            ug = np.hstack(ug_rows)
        else:
            C, D, lg, ug = np.zeros((0, nx)), np.zeros((0, nu)), np.zeros((N, 0)), np.zeros((N, 0))

        # constant of the condensed objective (OSQP obj_val omits it): |Phi_y x - ref|^2_Q
        e0 = (self.Phi_y @ self.x - self.ref).reshape(-1)
        self._hp_const = float(e0 @ Q @ e0)

        args = dict(A=A, B=B, Cy=Cy, Qy=Qy, ref=ref, R=Rb, C=C, D=D, lg=lg, ug=ug, zl=zl,
                    z_quad=float(self.eps_soft_reg), x0=x0)
        key = (N, nx, nu, C.shape[0], soft_rows)
        return args, key

    # ------------------------------------------------------------------
    # stage-wise solution -> OSQP result layout
    # ------------------------------------------------------------------
    def _to_osqp_result(self, sol):
        U = sol["U"]
        S = sol["S"]
        x = np.concatenate([U.reshape(-1), S.reshape(-1)])  # [U (nu*M) ; eps (ncs*N)], step-major

        max_res = max(sol["res_stat"], sol["res_eq"], sol["res_ineq"], sol["res_comp"])
        st = sol["status"]
        if st == 0:
            status_val, status = _SOLVED
        elif st in (1, 2) and max_res < self.hpipm_accept_res:
            status_val, status = _INACCURATE
        elif st == 1:
            status_val, status = _MAX_ITER
        else:
            status_val, status = _UNSOLVED

        # same definition as OSQP's obj_val (z'Hz + Fz, constant term omitted)
        obj_val = sol["tracking_cost"] + sol["effort_cost"] + sol["slack_cost"] - self._hp_const
        self.last_tracking_cost = sol["tracking_cost"]

        stats = self.hpipm_stats
        stats["calls"] += 1
        stats["iterations"] += sol["iterations"]
        stats["solve_time"] += sol["solve_time"]
        if status_val < 0:
            stats["fails"] += 1

        info = SimpleNamespace(obj_val=obj_val, status=status, status_val=status_val,
                               prim_res=max(sol["res_eq"], sol["res_ineq"]), dual_res=sol["res_stat"],
                               iter=sol["iterations"], solve_time=sol["solve_time"], run_time=sol["solve_time"],
                               hpipm_status=st, res_comp=sol["res_comp"])
        return SimpleNamespace(x=x, y=np.array([sol["lam_max"]]), info=info, hpipm=sol)


_CACHE = {}


def make_hpipm_mode(mode_cls):
    """Subclass of a Python RGC mode whose QP is solved by HPIPM (C++)."""
    if mode_cls not in _CACHE:
        _CACHE[mode_cls] = type(f"{mode_cls.__name__}Hpipm", (HpipmSolverMixin, mode_cls),
                                {"__module__": __name__, "python_mode": mode_cls})
    return _CACHE[mode_cls]
