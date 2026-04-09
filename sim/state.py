from dataclasses import dataclass, field
import numpy as np


@dataclass
class Go2State:
    b_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    b_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))

    epsilon: np.ndarray = field(default_factory=lambda: np.zeros(4))  # (x,y,z,w)
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))

    q: np.ndarray = field(default_factory=lambda: np.zeros(12))
    dq: np.ndarray = field(default_factory=lambda: np.zeros(12))

    r_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    r_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))

    foot_contacts: np.ndarray = field(default_factory=lambda: np.zeros(12))
    foot_forces: np.ndarray = field(default_factory=lambda: np.zeros(12))


@dataclass
class LowLevelState:

    # --------------------------------------
    # LOW-LEVEL CONTROL
    # --------------------------------------
    Kp: np.ndarray = field(default_factory=lambda: np.ones(12) * 50.0)
    Kd: np.ndarray = field(default_factory=lambda: np.ones(12) * 2.0)

    qr: np.ndarray = field(default_factory=lambda: np.zeros(12))
    dqr: np.ndarray = field(default_factory=lambda: np.zeros(12))

    tau_pd: np.ndarray = field(default_factory=lambda: np.zeros(12))
    tau_g: np.ndarray = field(default_factory=lambda: np.zeros(12))
    tau: np.ndarray = field(default_factory=lambda: np.zeros(12))


@dataclass
class SystemState:
    robot: Go2State = field(default_factory=Go2State)
    low_level: LowLevelState = field(default_factory=LowLevelState)
