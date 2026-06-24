from control.self_righting.time_based_solution.tb_base_controller import BaseTimeController
import numpy as np


class ProneCW(BaseTimeController):

    def __init__(self, **kwargs):

        state = kwargs.get("state")
        seed = kwargs.get("seed", None)
        stochastic = kwargs.get("stochastic", False)

        references = [
            np.array([[1.05, 1.5, -2.7, -0.85, 1.4, -2.7, 1.05, 1.50, -2.7, -0.9, 1.4, -2.7]]),
            np.array([[1.05, 1.5, -2.7, -0.85, 1.4, -2.7, 0.00, 1.55, -2.7, -0.9, 1.4, -2.7]]),
            np.array([[0.00, 1.5, -2.7, -0.85, 1.4, -2.7, 0.00, 1.55, -2.7, -0.9, 1.4, -2.7]]),
            np.array([[0.00, 1.4, -2.7, -0.00, 1.4, -2.7, 0.00, 1.40, -2.7, -0.0, 1.4, -2.7]])
        ]

        settling_times = [0.5, 0.5, 0.5, 0.5]

        delta_times = [0.1, 0.1, 0.1, 0.1] if stochastic else None

        super().__init__(state, references, settling_times, delta_times=delta_times, seed=seed)
        self.phase = 5
