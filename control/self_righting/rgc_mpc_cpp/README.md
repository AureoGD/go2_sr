# rgc_mpc_cpp — C++ port of the RGC-MPC controller (mode by mode)

The Python controller in `../rgc_mpc_solution` stays the reference and is never modified or
imported back from here (dependency goes one way: `rgc_mpc_cpp` → `rgc_mpc_solution`).

## Status

| Step | What runs in C++ | State |
|---|---|---|
| 1 | QP of every MPC mode (stage-wise QP + HPIPM) | **done** |
| 2 | Base controller: prediction model / constraint matrices | next |
| 3 | Modes, one at a time (references, constraints, kinematics via pinocchio C++) | — |
| 4 | Scheduler, state processing, observation builder (robot deployment) | — |

## Build

```bash
control/self_righting/rgc_mpc_cpp/build.sh          # BLASFEO 0.1.4.3 + HPIPM 0.1.4 (pinned) + module
BLASFEO_TARGET=ARMV8A_ARM_CORTEX_A57 HPIPM_TARGET=GENERIC ./build.sh   # ARM (robot)
CLEAN=1 ./build.sh                                   # rebuild everything
```
Produces `_rgc_mpc_cpp*.so` in this folder. Needs cmake ≥ 3.18, a C/C++17 compiler, Eigen 3,
git, and `pybind11` in the project venv.

## Use

```python
from control.self_righting.rgc_mpc_cpp import SchedulerRGCMPCHpipm, SchedulerRGCMPCHybrid

EnvConfig(controller_class=SchedulerRGCMPCHpipm, ...)            # all MPC modes on HPIPM
SchedulerRGCMPCHybrid(pin_engine=pe, cpp_modes=["ProneCW"])     # only some modes
SchedulerRGCMPCHybrid(pin_engine=pe, cpp_modes=())              # pure Python (OSQP)
```
Debug run: `python3 -m scripts.debug_control_rgc --solver hpipm [--cpp-modes ProneCW,StandUp] [--no-viewer]`

## Layout

- `include/rgc_mpc/stage_qp.hpp`, `src/hpipm_stage_qp_solver.cpp` — C++ core (no Python)
- `bindings/module.cpp` — pybind11 module `_rgc_mpc_cpp` (numpy in / out, GIL released while solving)
- `hpipm_mode.py` — `make_hpipm_mode(PythonMode)`: the Python mode with its QP solved in C++
- `hybrid_scheduler.py` — `SchedulerRGCMPCHybrid` / `SchedulerRGCMPCHpipm`
- `tests/test_mode_equivalence.py` — OSQP vs HPIPM on identical QPs along an FSM episode
