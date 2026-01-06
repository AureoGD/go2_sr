import numpy as np
import pinocchio as pin
from environment.strategies.rgc_mpc.smooth_filter import SmoothFilter


class PrepareCW():

    def __init__(self, **kwargs):
        self.task_name = "prepare_cw"
        self.task_level = 2

        self.runtime = "robot_states" in kwargs

        if not self.runtime:
            return

        self.robot_states = kwargs.get('robot_states', [])
        qr = np.array([[0.2, 1.4, -2.6, -0.8, 1.4, -2.6, 0.2, 1.4, -2.6, -0.8, 4.45, -2.5]]).transpose()
        prepare_to_rool1 = {'robot_states': self.robot_states, 'settling_time': 1, 't_cont': 0.01, 'qHL': qr}
        self.prepare_to_rool1 = SmoothFilter(**prepare_to_rool1)
        qr = np.array([-0.6, 1.5, -2.0, -0.8, 1.0, -2.6, -0.6, 1.5, -2.0, -0.5, 4.2, -2.25]).reshape(12, 1)
        prepare_to_rool2 = {'robot_states': self.robot_states, 'settling_time': 2, 't_cont': 0.01, 'qHL': qr}
        self.prepare_to_rool2 = SmoothFilter(**prepare_to_rool2)

        self.total_time_task = 1 + 2 + 0.1

        self.reset_controller()

    def update_dqr(self):
        if self.runtime:
            self.robot_states.subtask_succes = False

            spend_time = self.tick * 0.01

            if spend_time <= 1:
                delta_qr = self.prepare_to_rool1.smooth_reference().reshape(12, 1)
            elif spend_time <= 3:
                delta_qr = self.prepare_to_rool2.smooth_reference().reshape(12, 1)
            else:
                delta_qr = np.zeros((12, 1))
                self.robot_states.subtask_succes = True

            self.tick += 1

            self.percent_task = np.clip(spend_time / self.total_time_task, 0, 1)

            return delta_qr
        return np.zeros(12)

    def reset_controller(self):
        self.task_finish = False
        self.tick = 0
        self.percent_task = 0
