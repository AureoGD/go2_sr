import numpy as np
from environment.strategies.rgc_mpc.smooth_filter import SmoothFilter


class HoldPosition():

    def __init__(self, **kwargs):
        self.task_name = "hold"
        self.task_level = 0

        self.runtime = "robot_states" in kwargs

        if not self.runtime:
            return

        self.robot_states = kwargs.get('robot_states', [])

        self.reset_controller()

    def update_dqr(self):
        if self.runtime:
            self.robot_states.subtask_succes = True
            self.percent_task = 1
            return np.zeros(12)

        return np.zeros(12)

    def reset_controller(self):
        self.task_finish = False
        self.percent_task = 1
        self.tick = 0
