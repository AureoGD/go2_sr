from environment.strategies.time_based_stochastic.time_base_controller import BaseTimeController
from environment.strategies.rgc_mpc.smooth_filter import SmoothFilter
import numpy as np


class Hold(BaseTimeController):

    def __init__(self, **kwargs):
        robot_states = kwargs.get("robot_states")
        super().__init__(robot_states, references=[], settling_times=[])

    def update_dqr(self):

        return np.zeros((12, 1))
