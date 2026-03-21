from environment.strategies.time_based_stochastic.time_base_controller import BaseTimeController
from environment.strategies.rgc_mpc.smooth_filter import SmoothFilter
import numpy as np


class ProneCCW(BaseTimeController):

    def __init__(self, **kwargs):

        robot_states = kwargs.get("robot_states")
        seed = kwargs.get("seed", None)
        stochastic = kwargs.get("stochastic", False)

        # np.array([[0.6, 0.9, -1.5, 0.3, 1.5, -2.0, 0.6, 0.9, -1.5, 0.3, 1.5, -2.0]]).T

        references = [
            np.array([[0.85, 1.4, -2.7, -1.05, 1.5, -2.7, 0.9, 1.4, -2.7, -1.05, 1.50, -2.7]]).T,
            np.array([[0.85, 1.4, -2.7, -1.05, 1.5, -2.7, 0.9, 1.4, -2.7, 0.00, 1.55, -2.7]]).T,
            np.array([[0.85, 1.4, -2.7, 0.00, 1.5, -2.7, 0.9, 1.4, -2.7, 0.00, 1.55, -2.7]]).T,
            np.array([[0.00, 1.4, -2.7, 0.00, 1.4, -2.7, -0.0, 1.4, -2.7, 0.00, 1.40, -2.7]]).T
        ]

        settling_times = [0.5, 0.5, 0.5, 0.5]

        delta_times = [0.1, 0.1, 0.1, 0.1] if stochastic else None

        super().__init__(robot_states, references, settling_times, delta_times=delta_times, seed=seed)
