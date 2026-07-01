from scipy.spatial.transform import Rotation
import numpy as np

class SelfRightingTSM:
    # base sequence, direction-agnostic
    # slots 2,3,4,5 get resolved to CW or CCW variants at runtime
    BASE_SEQUENCE = ["Hold", "GoSafe", "Prepare", "Roll", "Landing", "Prone", "StandUp"]

    CW_ACTIONS  = {"Prepare": 2, "Roll": 3, "Landing": 4, "Prone": 5}
    CCW_ACTIONS = {"Prepare": 7, "Roll": 8, "Landing": 9, "Prone": 10}
    FIXED_ACTIONS = {"Hold": 0, "GoSafe": 1, "StandUp": 6}

    def __init__(self, dt=1):
        # durations[i] corresponds to BASE_SEQUENCE[i]
        self.durations = [50, 100, 300, 350, 250, 200, float('inf')]       
        self.bounds = []
        total = 0
        for d in self.durations:
            total += d
            self.bounds.append(total)
        self.dt = dt
        self.direction = None
        self.elapsed = 0.0
        self._started = False

    def _decide_direction(self,rpy):
        R_wb = Rotation.from_euler("xyz", [rpy[0], rpy[1], rpy[2]]).as_matrix()
        g_body = R_wb.T @ np.array([0, 0, -1])
        gy = g_body[1]
        if gy<0:
            self.direction = "cw"
        else:
            self.direction = "ccw"

    def step(self, rpy):
        if not self._started:
            self._decide_direction(rpy)
            self._started = True

        action = self.action
        self.elapsed += self.dt
        return action

    @property
    def action(self):
        table = self.CW_ACTIONS if self.direction == "cw" else self.CCW_ACTIONS
        for bound, name in zip(self.bounds, self.BASE_SEQUENCE):
            if self.elapsed < bound:
                return self.FIXED_ACTIONS.get(name, table.get(name))
        return self.FIXED_ACTIONS["StandUp"]

    def reset(self):
        self.elapsed = 0.0
        self.direction = None
        self._started = False