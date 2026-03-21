from environment.strategies.time_based_stochastic.time_base_controller import BaseTimeController
from environment.strategies.rgc_mpc.smooth_filter import SmoothFilter
import numpy as np


class GoSafe(BaseTimeController):

    def __init__(self, **kwargs):

        robot_states = kwargs.get("robot_states")
        seed = kwargs.get("seed", None)
        stochastic = kwargs.get("stochastic", False)

        references = [
            np.array([[0.7, 1.4, -2.6, -0.7, 1.4, -2.6, 0.7, 1.4, -2.6, -0.7, 1.4, -2.6]]).T,
        ]

        settling_times = [1.0]

        delta_times = [0.25] if stochastic else None

        super().__init__(robot_states, references, settling_times, delta_times=delta_times, seed=seed)
