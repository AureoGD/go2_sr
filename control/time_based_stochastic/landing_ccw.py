from control.time_based_stochastic.time_base_controller import BaseTimeController
from control.rgc_mpc.smooth_filter import SmoothFilter
import numpy as np


class LandingCCW(BaseTimeController):

    def __init__(self, **kwargs):

        state = kwargs.get("state")
        seed = kwargs.get("seed", None)
        stochastic = kwargs.get("stochastic", False)

        references = [
            np.array([[0.6, 1.3, -2.6, -0.3, 1.5, -2.0, 0, 1.4, -1.5, -0.3, 1.5, -2.0]]),
            np.array([[0.6, 0.9, -1.5, -0.3, 1.5, -2.0, 0.6, 0.9, -1.5, -0.3, 1.5, -2.0]])
        ]

        settling_times = [1.5, 1.0]

        delta_times = [0.3, 0.1] if stochastic else None

        super().__init__(state, references, settling_times, delta_times=delta_times, seed=seed)
        self.task_level = 4
