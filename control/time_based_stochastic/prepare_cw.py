from control.time_based_stochastic.time_base_controller import BaseTimeController
from control.rgc_mpc.smooth_filter import SmoothFilter
import numpy as np


class PrepareCW(BaseTimeController):

    def __init__(self, **kwargs):

        state = kwargs.get("state")
        seed = kwargs.get("seed", None)
        stochastic = kwargs.get("stochastic", False)

        references = [
            np.array([[0.2, 1.4, -2.6, -0.8, 1.4, -2.6, 0.2, 1.4, -2.6, -0.8, 4.45, -2.5]]),
            np.array([[-0.6, 1.5, -2.0, -0.8, 1.0, -2.6, -0.6, 1.5, -2.0, -0.5, 4.2, -2.25]])
        ]

        settling_times = [1.0, 2.0]

        delta_times = [0.1, 0.2] if stochastic else None

        super().__init__(state, references, settling_times, delta_times=delta_times, seed=seed)
        self.task_level = 2
