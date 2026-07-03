from enum import Enum, auto, IntEnum
from dataclasses import dataclass
from typing import Optional
import numpy as np
from env.normalizer import StateNormalizer
from scipy.spatial.transform import Rotation


class Controller(IntEnum):
    HOLD = 0
    GO_SAFE = 1
    PREPARE_CW = 2
    ROLL_CW = 3
    SWING_LEG_CW = 4
    SETTLE_CW = 5
    PRONE_CW = 6
    STAND_UP = 7
    PREPARE_CCW = 8
    ROLL_CCW = 9
    SWING_LEG_CCW = 10
    SETTLE_CCW = 11
    PRONE_CCW = 12


class RobotStatus:

    SAFE_POSITION = np.array([0.7, 1.4, -2.6, -0.7, 1.4, -2.6, 0.7, 1.4, -2.6, -0.7, 1.4, -2.6])
    PRONE_POSITION = np.array([0.0, 1.4, -2.7, 0.0, 1.4, -2.7, 0.0, 1.4, -2.7, 0.0, 1.4, -2.7])
    PREPARED_CW = np.array([-0.6, 1.5, -2.0, -0.8, 1.0, -2.6, -0.6, 1.5, -2.0, -1.025, 4.15, -2.2])
    PREPARED_CCW = np.array([0.8, 1.0, -2.6, 0.6, 1.5, -2.0, 1.025, 4.15, -2.2, 0.6, 1.5, -2.0])
    SAFE_THRESHOLD = 0.05  # radians tolerance
    PHASE_TIMEOUT = 10.0  # seconds

    def __init__(self, robot_state, controller_state, torque_lim, joint_lin):
        self.rs = robot_state.robot
        self.lcs = robot_state.low_level
        self.hcs = controller_state
        self.normalizer = StateNormalizer(torque_limits=torque_lim, joint_limits=joint_lin)
        self.bz0 = None

    def set_initial_bz(self):
        self.bz0 = self.rs.b_pos[2]

    @property
    def is_upside_down(self):
        alpha = self.normalizer.compute_alpha(self.rs.epsilon)
        return alpha < -0.1

    @property
    def roll_cw(self):
        rpy = self.rs.rpy
        R_wb = Rotation.from_euler("xyz", [rpy[0], rpy[1], rpy[2]]).as_matrix()
        g_body = R_wb.T @ np.array([0, 0, -1])
        gy = g_body[1]

        return gy < 0

    @property
    def roll_ccw(self):
        rpy = self.rs.rpy
        R_wb = Rotation.from_euler("xyz", [rpy[0], rpy[1], rpy[2]]).as_matrix()
        g_body = R_wb.T @ np.array([0, 0, -1])
        gy = g_body[1]

        return gy >= 0

    @property
    def is_upside(self):
        alpha = self.normalizer.compute_alpha(self.rs.epsilon)
        return alpha > 0.9

    @property
    def joints_at_safe_position(self):
        return all(abs(self.rs.q - self.SAFE_POSITION) < self.SAFE_THRESHOLD)

    @property
    def is_robot_prepared_cw(self):
        return all(abs(self.rs.q - self.PREPARED_CW) < 0.12) or (np.linalg.norm(self.lcs.dqr) < 0.001)

    @property
    def is_robot_prepared_ccw(self):
        return all(abs(self.rs.q - self.PREPARED_CCW) < 0.12) or (np.linalg.norm(self.lcs.dqr) < 0.001)

    @property
    def left_rear_foot_touching(self):
        return bool(self.rs.foot_touching[3])

    @property
    def all_foot_touching(self):
        return bool(np.all(self.rs.foot_touching) == 1)

    @property
    def right_side_feet_touching(self):
        front = bool(self.rs.foot_touching[0])
        rear = bool(self.rs.foot_touching[2])

        if not np.any(np.isinf(self.hcs.swing_foot_error)):
            front |= bool(np.linalg.norm(self.hcs.swing_foot_error[0:3]) < 0.05)
            rear |= bool(np.linalg.norm(self.hcs.swing_foot_error[3:6]) < 0.05)

        return front and rear

    @property
    def left_side_feet_touching(self):
        front = bool(self.rs.foot_touching[1])
        rear = bool(self.rs.foot_touching[3])

        if not np.any(np.isinf(self.hcs.swing_foot_error)):
            front |= bool(np.linalg.norm(self.hcs.swing_foot_error[0:3]) < 0.05)
            rear |= bool(np.linalg.norm(self.hcs.swing_foot_error[3:6]) < 0.05)

        return front and rear

    @property
    def robot_proned(self):
        return all(abs(self.rs.q - self.PRONE_POSITION) < 0.25)

    @property
    def stand_finish(self):
        return self.rs.b_pos[2] - self.bz0 > 0.14


class SelfRightingFSM:

    def __init__(self, default=None):
        if default == 'CW':
            self.default = Controller.PREPARE_CW
        elif default == 'CCW':
            self.default = Controller.PREPARE_CCW
        else:
            self.default = None

        self._state = Controller.HOLD

    def update(self, status: RobotStatus) -> int:
        """Evaluate transitions and return the active controller index."""
        self._state = self._next_state(status)
        return int(self._state)

    def reset(self):
        self._state = Controller.HOLD

    def _next_state(self, s: RobotStatus) -> Controller:
        match self._state:

            case Controller.HOLD:
                if s.is_upside_down:
                    return Controller.GO_SAFE

            case Controller.GO_SAFE:
                if s.joints_at_safe_position:
                    if self.default is not None:
                        return self.default
                    else:
                        if s.roll_cw:
                            return Controller.PREPARE_CW
                        else:
                            return Controller.PREPARE_CCW

            case Controller.PREPARE_CW:
                if s.is_robot_prepared_cw:
                    return Controller.ROLL_CW

            case Controller.PREPARE_CCW:
                if s.is_robot_prepared_ccw:
                    return Controller.ROLL_CCW

            case Controller.ROLL_CW:
                if s.right_side_feet_touching:
                    return Controller.SWING_LEG_CW

            case Controller.ROLL_CCW:
                if s.left_side_feet_touching:
                    return Controller.SWING_LEG_CCW

            case Controller.SWING_LEG_CW:
                if s.left_side_feet_touching:
                    return Controller.SETTLE_CW

            case Controller.SWING_LEG_CCW:
                if s.right_side_feet_touching:
                    return Controller.SETTLE_CCW

            case Controller.SETTLE_CW:
                if s.is_upside:
                    return Controller.PRONE_CW

            case Controller.SETTLE_CCW:
                if s.is_upside:
                    return Controller.PRONE_CCW

            case Controller.PRONE_CW:
                if s.robot_proned:
                    s.set_initial_bz()
                    return Controller.STAND_UP

            case Controller.PRONE_CCW:
                if s.robot_proned:
                    s.set_initial_bz()
                    return Controller.STAND_UP

            case Controller.STAND_UP:
                if s.stand_finish:
                    self.reset()
                    return Controller.HOLD

        return self._state
