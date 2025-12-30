import numpy as np
from environment.strategies.rgc_mpc.smooth_filter import SmoothFilter


class LandingCW():

    def __init__(self, **kwargs):
        self.task_name = "landing_cw"
        self.task_level = 4

        self.runtime = "robot_states" in kwargs

        if not self.runtime:
            return

        self.robot_states = kwargs.get('robot_states', [])

        # qr = np.array([[-0.25, 1.5, -2.0, -0.85, 0.85, -1.3, -0.25, 1.5, -2.2, 0.6, 3.75, -1.5]]).transpose()
        # landing_stage_1 = {'robot_states': self.robot_states, 'settling_time': 1, 't_cont': 0.01, 'qHL': qr}
        # self.landing_stage_1 = SmoothFilter(**landing_stage_1)

        # qr = np.array([[-0.25, 1.5, -2.0, -0.85, 0.85, -1.3, -0.25, 1.5, -2.2, -0.9, 0.9, -1.5]]).transpose()
        # landing_stage_2 = {'robot_states': self.robot_states, 'settling_time': 1, 't_cont': 0.01, 'qHL': qr}
        # self.landing_stage_2 = SmoothFilter(**landing_stage_2)

        qr = np.array([[0.5, 1.5, -2.0, -0.85, 0.85, -2.0, 0.5, 1.5, -2.2, -0.9, 0.9, -2.0]]).transpose()
        landing_stage_1 = {'robot_states': self.robot_states, 'settling_time': 1.5, 't_cont': 0.01, 'qHL': qr}
        self.landing_stage_1 = SmoothFilter(**landing_stage_1)

        qr = np.array([[0.5, 1.5, -2.7, -0.85, 0.5, -1.7, 0.5, 1.5, -2.7, -0.9, 0.5, -1.7]]).transpose()
        landing_stage_2 = {'robot_states': self.robot_states, 'settling_time': 0.5, 't_cont': 0.01, 'qHL': qr}
        self.landing_stage_2 = SmoothFilter(**landing_stage_2)

        self.reset_controller()

    def update_dqr(self):
        if self.runtime:
            self.robot_states.subtask_succes = False
            if self.tick * 0.01 <= 1.5:
                delta_qr = self.landing_stage_1.smooth_reference().reshape(12, 1)
            elif self.tick * 0.01 <= 2.0:
                delta_qr = self.landing_stage_2.smooth_reference().reshape(12, 1)
            else:
                delta_qr = np.zeros((12, 1))
                self.robot_states.subtask_succes = True
            self.tick += 1

            return delta_qr
        return np.zeros(12)

    def reset_controller(self):
        self.task_finish = False
        self.tick = 0
