from control.time_based_stochastic.time_base_controller import BaseTimeController
from control.rgc_mpc.smooth_filter import SmoothFilter
import numpy as np


class Hold(BaseTimeController):

    def __init__(self, **kwargs):
        state = kwargs.get("state")
        super().__init__(state, references=[], settling_times=[])

        self.task_level = 0

    def update_dqr(self):

        return np.zeros(12)
