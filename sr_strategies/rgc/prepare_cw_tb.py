import numpy as np
import pinocchio as pin
from sr_strategies.tsm.smooth_filter import SmoothFilter
from sr_strategies.rgc.roll_cw import RollCW
from sr_strategies.rgc.stand_up import StandUpPhase


class PrepareCW():

    def __init__(self, **kwargs):
        self.robot_states = kwargs.get('robot_states', [])
        qr = np.array([[0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.3, -2.7, -1.1, 1.4, -2.8]]).transpose()
        prepare_to_rool1 = {'robot_states': self.robot_states, 'settling_time': 1, 't_cont': 0.01, 'qHL': qr}
        self.prepare_to_rool1 = SmoothFilter(**prepare_to_rool1)

        qr = np.array([[0, 1.4, -2.7, -0.50, 1.0, -2.7, 0, 1.3, -2.7, -0.75, 4, -2.8]]).transpose()
        prepare_to_rool2 = {'robot_states': self.robot_states, 'settling_time': 1, 't_cont': 0.01, 'qHL': qr}
        self.prepare_to_rool2 = SmoothFilter(**prepare_to_rool2)

        qr = np.array([[-0.6, 0, -2.8, -0.8, 1.0, -2.8, -0.6, 0, -2.8, -0.5, 4.45, -2.5]]).transpose()
        prepare_to_rool3 = {'robot_states': self.robot_states, 'settling_time': 0.5, 't_cont': 0.01, 'qHL': qr}
        self.prepare_to_rool3 = SmoothFilter(**prepare_to_rool3)

        self.reset_controller()

    def update_dqr(self):
        if self.tick * 0.01 <= 1:
            delta_qr = self.prepare_to_rool1.smooth_reference().reshape(12, 1)
        elif self.tick * 0.01 > 1 and self.tick * 0.01 <= 2:
            delta_qr = self.prepare_to_rool2.smooth_reference().reshape(12, 1)
        else:
            self.task_finish = True
            delta_qr = self.prepare_to_rool3.smooth_reference().reshape(12, 1)

        self.tick += 1
        return delta_qr

    def reset_controller(self):
        self.task_finish = False
        self.tick = 0
