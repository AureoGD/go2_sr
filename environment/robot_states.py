import numpy as np
from dataclasses import dataclass, field


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
    tau: np.ndarray = field(default_factory=lambda: np.zeros((12, 1), dtype=np.float64))
