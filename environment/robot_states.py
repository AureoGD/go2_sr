import numpy as np
from dataclasses import dataclass, field
from typing import Dict


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

    mpc_fail: bool = field(default=False)
    critical_mpc_fail: bool = field(default=False)
    subtask_succes: bool = field(default=False)
    mpc_obj_val: float = field(default=0)

    sr_mode_completed: Dict[int, bool] = field(default_factory=lambda: {
        0: False,  # go_safe
        1: False,  # prepare_cw
        2: False,  # roll_cw
        3: False,  # landing_cw
        4: False  # stand_up
    })
