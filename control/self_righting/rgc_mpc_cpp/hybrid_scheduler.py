"""Hybrid RGC scheduler: each mode runs either fully in Python (OSQP) or with its QP in C++ (HPIPM).

Same interface, same mode order (CONTROLLER_CLASSES) and same task_state as SchedulerRGCMPC,
so the env, the tasks, the mode-graph mask and every training script work unchanged:

    EnvConfig(controller_class=SchedulerRGCMPCHpipm, ...)          # every MPC mode on HPIPM
    SchedulerRGCMPCHybrid(pin_engine=..., cpp_modes=["ProneCW"])  # only some modes
    SchedulerRGCMPCHybrid(pin_engine=..., cpp_modes=())           # pure Python (== SchedulerRGCMPC)

Dependency rule: rgc_mpc_cpp imports rgc_mpc_solution, never the other way around.
"""
from control.self_righting.rgc_mpc_solution.rgc_scheduler import CONTROLLER_CLASSES, SchedulerRGCMPC
from control.self_righting.rgc_mpc_solution.modes.hold_position import HoldPosition
from control.self_righting.rgc_mpc_cpp.hpipm_mode import make_hpipm_mode

# modes without an MPC (nothing to port)
_NO_MPC = {HoldPosition}


class SchedulerRGCMPCHybrid(SchedulerRGCMPC):

    def __init__(self, cpp_modes="all", **kwargs):
        super().__init__(**kwargs)
        if cpp_modes == "all":
            self.cpp_modes = {c.__name__ for c in CONTROLLER_CLASSES if c not in _NO_MPC}
        else:
            self.cpp_modes = set(cpp_modes or ())
        unknown = self.cpp_modes - {c.__name__ for c in CONTROLLER_CLASSES}
        if unknown:
            raise ValueError(f"unknown modes in cpp_modes: {sorted(unknown)}")

    def mode_class(self, cls):
        """Class actually instantiated for a Python mode class."""
        if cls in _NO_MPC or cls.__name__ not in self.cpp_modes:
            return cls
        return make_hpipm_mode(cls)

    def _instantiate_controllers(self, state):
        self.controllers = [self.mode_class(cls)(robot_states=state, **self._base_kwargs) for cls in CONTROLLER_CLASSES]
        self.n_controllers = len(self.controllers)
        self._controllers_initialized = True

        # keep solver statistics across episodes (controllers are re-created on reset_phase)
        if not hasattr(self, "_stats_total"):
            self._stats_total = {}

    def reset_phase(self):
        self._accumulate_stats()
        super().reset_phase()

    def _accumulate_stats(self):
        if not self.controllers:
            return
        if not hasattr(self, "_stats_total"):
            self._stats_total = {}
        for ctrl in self.controllers:
            st = getattr(ctrl, "hpipm_stats", None)
            if not st or st["calls"] == 0:
                continue
            name = type(ctrl).python_mode.__name__
            acc = self._stats_total.setdefault(name, {k: 0 for k in st})
            for k, v in st.items():
                acc[k] += v
            for k in st:
                st[k] = 0

    def solver_stats(self):
        """Per mode: calls, fails, mean iterations, mean HPIPM solve time [ms], mean Python build time [ms]."""
        self._accumulate_stats()
        out = {}
        for name, s in getattr(self, "_stats_total", {}).items():
            n = max(1, s["calls"])
            out[name] = {"calls": s["calls"], "fails": s["fails"], "iterations": s["iterations"] / n,
                         "solve_ms": 1e3 * s["solve_time"] / n, "build_ms": 1e3 * s["build_time"] / n}
        return out


class SchedulerRGCMPCHpipm(SchedulerRGCMPCHybrid):
    """Every MPC mode solved with HPIPM (for EnvConfig, which only passes pin_engine)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("cpp_modes", "all")
        super().__init__(**kwargs)
