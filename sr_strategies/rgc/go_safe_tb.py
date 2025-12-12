import numpy as np
import pinocchio as pin
from sr_strategies.tsm.smooth_filter import SmoothFilter
from sr_strategies.rgc.roll_cw import RollCW
from sr_strategies.rgc.stand_up import StandUpPhase


class GoSafe():

    def __init__(self, **kwargs):
        self.robot_states = kwargs.get('robot_states', [])
        # configure "start_pos" smooth filer
        qr = np.array([[0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7]]).transpose()
        go_safe_kwargd = {'robot_states': self.robot_states, 'settling_time': 1, 't_cont': 0.01, 'qHL': qr}
        self.go_safe = SmoothFilter(**go_safe_kwargd)

        self.reset_controller()

    def update_dqr(self):
        if self.tick * 0.01 <= 1.5:
            delta_qr = self.go_safe.smooth_reference()
        else:
            self.task_finish = True
            delta_qr = np.zeros((12, 1))
        self.tick += 1

        return delta_qr

    def reset_controller(self):
        self.task_finish = False
        self.tick = 0
