import numpy as np
import pinocchio as pin
from environment.strategies.rgc_mpc.smooth_filter import SmoothFilter


class RollCW():

    def __init__(self, **kwargs):
        self.task_name = "roll_cw"
        self.task_level = 3

        self.runtime = "robot_states" in kwargs

        if not self.runtime:
            return

        self.robot_states = kwargs.get('robot_states', [])
        qr = np.array([[-0.6, 1.5, -2.0, -0.6, 1.3, -2.6, -0.6, 1.5, -2.0, 0.4, 3.75, -1.5]]).transpose()
        roll1 = {'robot_states': self.robot_states, 'settling_time': 1, 't_cont': 0.01, 'qHL': qr}
        self.roll1 = SmoothFilter(**roll1)
        qr = np.array([[-0.4, 1.5, -2.0, -0.6, 1.3, -2.6, -0.4, 1.5, -2.0, 0.5, 3.75, -1.5]]).transpose()
        roll2 = {'robot_states': self.robot_states, 'settling_time': 1.5, 't_cont': 0.01, 'qHL': qr}
        self.roll2 = SmoothFilter(**roll2)
        qr = np.array([[0.3, 1.5, -2.0, -0.6, 1.3, -2.6, 0.3, 1.5, -2.0, 0.9, 3.75, -1.5]]).transpose()
        roll3 = {'robot_states': self.robot_states, 'settling_time': 1.0, 't_cont': 0.01, 'qHL': qr}
        self.roll3 = SmoothFilter(**roll3)

        self.total_time_task = 1 + 1.5 + 1 + 0.1

        self.reset_controller()

    def update_dqr(self):
        if self.runtime:
            self.robot_states.subtask_succes = False

            spend_time = self.tick * 0.01

            if spend_time <= 1:
                delta_qr = self.roll1.smooth_reference().reshape(12, 1)
            if spend_time <= 2.5:
                delta_qr = self.roll2.smooth_reference().reshape(12, 1)
            if spend_time <= 3.5:
                delta_qr = self.roll3.smooth_reference().reshape(12, 1)
            else:
                delta_qr = np.zeros((12, 1))
                self.robot_states.subtask_succes = True

            self.tick += 1

            self.percent_task = np.clip(spend_time / self.total_time_task, 0, 1)
            self.robot_states.sr_controller_sucess_percent = self.percent_task

            return delta_qr
        return np.zeros(12)

    def reset_controller(self):
        self.task_finish = False
        self.tick = 0
        self.percent_task = 0
