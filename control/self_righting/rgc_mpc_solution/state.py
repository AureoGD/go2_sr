from dataclasses import dataclass, field
import numpy as np

NCS_MAX = 8


@dataclass
class RGCState:
    controller_index: int = -1

    last_controller: int = -1

    controller_evolution: float = 0.0

    action_group: int = -1

    # ---------------------------------
    # MPC status
    # ---------------------------------

    mpc_obj_val: float = 0.0

    mpc_fail: bool = False

    mpc_critical_fail: bool = False

    lambda_max: float = 0.0

    primal_res: float = 0.0

    dual_res: float = 0.0

    slack_max: float = 0.0

    solver_status: float = 0.0

    # ---------------------------------
    # MPC status
    # ---------------------------------

    slack_max_per_row: np.ndarray = field(default_factory=lambda: np.full(NCS_MAX, np.nan))

    swing_foot_error: np.ndarray = field(default_factory=lambda: np.ones(6) * np.inf)

    cp_trig_signal: bool = False

    com_trig_signal: bool = False

    prone_final_stage: bool = False

