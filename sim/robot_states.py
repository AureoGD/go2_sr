import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RobotStates:
    b_pos: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    b_vel: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    r_pos: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    r_vel: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    omega: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))

    epsilon: np.ndarray = field(default_factory=lambda: np.zeros((4, 1), dtype=np.float64))
    rpy: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))

    q: np.ndarray = field(default_factory=lambda: np.zeros((12, 1), dtype=np.float64))
    dq: np.ndarray = field(default_factory=lambda: np.zeros((12, 1), dtype=np.float64))
    qr: np.ndarray = field(default_factory=lambda: np.zeros((12, 1), dtype=np.float64))
    dqr: np.ndarray = field(default_factory=lambda: np.zeros((12, 1), dtype=np.float64))
    qrh: np.ndarray = field(default_factory=lambda: np.zeros((12, 1), dtype=np.float64))
    tau_pd: np.ndarray = field(default_factory=lambda: np.zeros((12, 1), dtype=np.float64))
    tau_g: np.ndarray = field(default_factory=lambda: np.zeros((12, 1), dtype=np.float64))
    contacts: np.ndarray = field(default_factory=lambda: np.zeros((4, 3), dtype=np.float64))

    ext_contact_force: np.ndarray = field(default_factory=lambda: np.zeros((12, 1), dtype=np.float64))

    pc_debug: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=np.float64))

    mpc_obj_val: float = field(default=0)
    mpc_fail: bool = field(default=False)
    mpc_critical_fail: bool = field(default=False)

    # robot phases: holding, safe, prepared_to_roll, end_roll, landed, end_proning, end_standin
    sr_mode_completed: List[bool] = field(default_factory=lambda: [False] * 7)
    sr_controller_sucess_percent: float = field(default=0)
    sr_current_controller: float = field(default=0)

    lambda_max: float = field(default=0)
    primal_res: float = field(default=0)
    dual_res: float = field(default=0)

    subtask_succes: bool = field(default=False)
    current_sucess_mode: float = field(default=0)
