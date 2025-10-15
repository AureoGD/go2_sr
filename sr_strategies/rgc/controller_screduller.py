import numpy as np
import pinocchio as pin
from sr_strategies.rgc.stand_up import StandUpPhase


class ControlScheduller():

    def __init__(self, **kwargs):
        self.stand_up = StandUpPhase(**kwargs)
        self.robot_states = kwargs.get('robot_states', [])

    def update(self, mode):
        if mode == 1:
            delta_qr = self.stand_up.solve_rgc().reshape(12, 1)
        else:
            delta_qr = np.zeros((12, 1), dtype=np.float32)
        # delta_qr = np.zeros((12, 1), dtype=np.float32)
        return self.robot_states.qr + delta_qr
