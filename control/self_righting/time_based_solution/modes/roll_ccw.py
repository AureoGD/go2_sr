from control.self_righting.time_based_solution.tb_base_controller import BaseTimeController
import numpy as np


class RollCCW(BaseTimeController):

    def __init__(self, **kwargs):

        state = kwargs.get("state")
        seed = kwargs.get("seed", None)
        stochastic = kwargs.get("stochastic", False)

        references = [
            np.array([[0.6, 1.3, -2.6, 0.6, 1.5, -2.0, -0.4, 3.75, -1.5, 0.6, 1.5, -2.0]]),
            np.array([[0.6, 1.3, -2.6, 0.4, 1.5, -2.0, -0.5, 3.75, -1.5, 0.4, 1.5, -2.0]]),
            np.array([[0.6, 1.3, -2.6, -0.3, 1.5, -2.0, -0.9, 3.75, -1.5, -0.3, 1.5, -2.0]])
        ]

        settling_times = [1.0, 1.5, 1.0]

        delta_times = [0.1, 0.3, 0.15] if stochastic else None

        super().__init__(state, references, settling_times, delta_times=delta_times, seed=seed)
        self.action_group = 3
