import numpy as np
import pinocchio as pin
from environment.strategies.rgc_mpc.smooth_filter import SmoothFilter


class StandUpPhase():

    def __init__(self, **kwargs):
        self.task_name = "standing_up"
        self.task_level = 6

        self.runtime = "robot_states" in kwargs

        if not self.runtime:
            return

        self.robot_states = kwargs.get('robot_states', [])
        qr = np.array([[0.0, 1.0, -2.0, 0, 1.0, -2.0, 0, 1.0, -2.0, 0, 1.0, -2.0]]).transpose()
        standing = {'robot_states': self.robot_states, 'settling_time': 2, 't_cont': 0.01, 'qHL': qr}
        self.standing = SmoothFilter(**standing)

        self.total_time_task = 2 + 0.1

        self.reset_controller()

    def update_dqr(self):
        if self.runtime:
            self.robot_states.subtask_succes = False

            spend_time = self.tick * 0.01

            if spend_time <= 2:
                delta_qr = self.standing.smooth_reference().reshape(12, 1)
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
