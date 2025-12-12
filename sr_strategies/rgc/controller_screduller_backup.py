import numpy as np
import pinocchio as pin
from sr_strategies.rgc.stand_up import StandUpPhase
from sr_strategies.tsm.smooth_filter import SmoothFilter
from sr_strategies.rgc.roll_cw import RollCW
from sr_strategies.rgc.go_safe import GoSafe
from sr_strategies.rgc.prepare_cw import PrepareCW


class ControlScheduler():

    def __init__(self, **kwargs):
        self.stand_up = StandUpPhase(**kwargs)
        self.roll_cw = RollCW(**kwargs)
        self.robot_states = kwargs.get('robot_states', [])
        self.go_safe = GoSafe(**kwargs)
        self.prepare_cw = PrepareCW(**kwargs)

        self.kp = kwargs.get('kp')
        self.kd = kwargs.get('kd')

        self.KP = self.kp * np.eye(12)
        self.KD = self.kd * np.eye(12)
        self.delta_qr = np.zeros((12, 1))

    def update(self, mode):

        self.delta_qr = self.prepare_cw.update_dqr().reshape(12, 1)

        return self.delta_qr.reshape(12), self.KP, self.KD / 10
