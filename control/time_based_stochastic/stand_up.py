from environment.strategies.time_based_stochastic.time_base_controller import BaseTimeController
from environment.strategies.rgc_mpc.smooth_filter import SmoothFilter
import numpy as np


class StandUp(BaseTimeController):

    def __init__(self, **kwargs):

        robot_states = kwargs.get("robot_states")
        seed = kwargs.get("seed", None)
        stochastic = kwargs.get("stochastic", False)

        references = [np.array([[0.0, 1.0, -2.0, 0, 1.0, -2.0, 0, 1.0, -2.0, 0, 1.0, -2.0]]).T]

        settling_times = [2.0]

        delta_times = [0.25] if stochastic else None

        super().__init__(robot_states, references, settling_times, delta_times=delta_times, seed=seed)
