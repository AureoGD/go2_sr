import numpy as np
from environment.strategies.rgc_mpc.smooth_filter import SmoothFilter


class ProneCW():

    def __init__(self, **kwargs):
        self.task_name = "prone_cw"
        self.task_level = 5

        self.runtime = "robot_states" in kwargs

        if not self.runtime:
            return

        self.robot_states = kwargs.get('robot_states', [])

        qr = np.array([[1.05, 1.5, -2.7, -0.85, 1.4, -2.7, 1.05, 1.5, -2.7, -0.9, 1.4, -2.7]]).transpose()
        landing_stage_1 = {'robot_states': self.robot_states, 'settling_time': 0.5, 't_cont': 0.01, 'qHL': qr}
        self.landing_stage_1 = SmoothFilter(**landing_stage_1)

        qr = np.array([[1.05, 1.5, -2.7, -0.85, 1.4, -2.7, 0, 1.55, -2.7, -0.9, 1.4, -2.7]]).transpose()
        landing_stage_2 = {'robot_states': self.robot_states, 'settling_time': 0.5, 't_cont': 0.01, 'qHL': qr}
        self.landing_stage_2 = SmoothFilter(**landing_stage_2)

        qr = np.array([[0, 1.5, -2.7, -0.85, 1.4, -2.7, 0, 1.55, -2.7, -0.9, 1.4, -2.7]]).transpose()
        landing_stage_3 = {'robot_states': self.robot_states, 'settling_time': 0.5, 't_cont': 0.01, 'qHL': qr}
        self.landing_stage_3 = SmoothFilter(**landing_stage_3)

        qr = np.array([[0.2, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7]]).transpose()
        landing_stage_4 = {'robot_states': self.robot_states, 'settling_time': 0.5, 't_cont': 0.01, 'qHL': qr}
        self.landing_stage_4 = SmoothFilter(**landing_stage_4)

        self.total_time_task = 0.5 + 0.5 + 0.5 + 0.5 + 0.1

        self.reset_controller()

    def update_dqr(self):
        if self.runtime:
            self.robot_states.subtask_succes = False

            spend_time = self.tick * 0.01
            if spend_time <= 0.5:
                delta_qr = self.landing_stage_1.smooth_reference().reshape(12, 1)
            elif spend_time <= 1.0:
                delta_qr = self.landing_stage_2.smooth_reference().reshape(12, 1)
            elif spend_time <= 1.5:
                delta_qr = self.landing_stage_3.smooth_reference().reshape(12, 1)
            elif spend_time <= 2.0:
                delta_qr = self.landing_stage_4.smooth_reference().reshape(12, 1)
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
        self.percent_task = 0
        self.tick = 0
