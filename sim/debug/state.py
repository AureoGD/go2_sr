from dataclasses import dataclass, field
import numpy as np


@dataclass
class DebugState:
    dr: np.ndarray = field(default_factory=lambda: np.zeros(3))
    r: np.ndarray = field(default_factory=lambda: np.zeros(3))
    foot_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    n: np.ndarray = field(default_factory=lambda: np.zeros((2, 3)))
    p_knee: np.ndarray = field(default_factory=lambda: np.zeros((2, 3)))
    p_pivot: np.ndarray = field(default_factory=lambda: np.zeros((2, 3)))
    f_pivot: np.ndarray = field(default_factory=lambda: np.zeros((2, 3)))
    sw_foot_data: np.ndarray = field(default_factory=lambda: np.zeros((7, 3)))

    Sa: np.ndarray = field(default_factory=lambda: np.zeros((12, 3)))
    trigger: np.ndarray = field(default_factory=lambda: np.zeros(1))
    cp_signal: np.ndarray = field(default_factory=lambda: np.zeros(1))
    com_signal: np.ndarray = field(default_factory=lambda: np.zeros(1))
    cp: np.ndarray = field(default_factory=lambda: np.zeros(1))
    cp_val: np.ndarray = field(default_factory=lambda: np.zeros(1))

    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))
    q: np.ndarray = field(default_factory=lambda: np.zeros(12))

    obj: np.ndarray = field(default_factory=lambda: np.zeros(3,))
    eps_ref: np.ndarray = field(default_factory=lambda: np.zeros(4))
    cv: np.ndarray = field(default_factory=lambda: np.zeros(1))