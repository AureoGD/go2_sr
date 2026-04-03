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

    contacts: np.ndarray = field(default_factory=lambda: np.zeros((4, 3)))
    contact_forces: np.ndarray = field(default_factory=lambda: np.zeros(12))

    qr: np.ndarray = field(default_factory=lambda: np.zeros(12))
    dqr: np.ndarray = field(default_factory=lambda: np.zeros(12))


@dataclass
class ControllerState:

    # --------------------------------------
    # LOW-LEVEL CONTROL
    # --------------------------------------
    Kp: np.ndarray = field(default_factory=lambda: np.ones(12) * 50.0)
    Kd: np.ndarray = field(default_factory=lambda: np.ones(12) * 2.0)

    tau_pd: np.ndarray = field(default_factory=lambda: np.zeros(12))
    tau_g: np.ndarray = field(default_factory=lambda: np.zeros(12))
    tau: np.ndarray = field(default_factory=lambda: np.zeros(12))

    # --------------------------------------
    # Scheduller
    # --------------------------------------
    controller_index: int = -1
    controller_evolution: float = 0.0
    sr_semantics: int = -1

    # --------------------------------------
    # MPC / OPTIMIZATION
    # --------------------------------------
    mpc_obj_val: float = 0.0
    mpc_fail: bool = False
    mpc_critical_fail: bool = False

    lambda_max: float = 0.0
    primal_res: float = 0.0
    dual_res: float = 0.0

    # --------------------------------------
    # DEBUG
    # --------------------------------------
    pc_debug: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))


@dataclass
class TaskState:
    probs: np.ndarray = field(default_factory=lambda: np.zeros(4))


@dataclass
class SystemState:
    robot: Go2State = field(default_factory=Go2State)
    controller: ControllerState = field(default_factory=ControllerState)
    tpe: TaskState = field(default_factory=TaskState)
