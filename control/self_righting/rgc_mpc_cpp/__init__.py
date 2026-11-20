"""C++ port of the RGC-MPC controller (mode by mode).

Step 1 (current): the QP of every MPC mode can be solved by HPIPM in C++ (stage-wise QP),
while the mode logic stays in Python (control/self_righting/rgc_mpc_solution, unchanged).

Build once:  control/self_righting/rgc_mpc_cpp/build.sh
Use:         from control.self_righting.rgc_mpc_cpp import SchedulerRGCMPCHpipm, SchedulerRGCMPCHybrid
"""
from control.self_righting.rgc_mpc_cpp.hybrid_scheduler import SchedulerRGCMPCHpipm, SchedulerRGCMPCHybrid
from control.self_righting.rgc_mpc_cpp.hpipm_mode import HpipmSolverMixin, make_hpipm_mode

__all__ = ["SchedulerRGCMPCHpipm", "SchedulerRGCMPCHybrid", "HpipmSolverMixin", "make_hpipm_mode"]
