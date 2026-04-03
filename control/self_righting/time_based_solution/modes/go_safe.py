from control.self_righting.time_based_solution.tb_base_controller import BaseTimeController
import numpy as np


class GoSafe(BaseTimeController):

    def __init__(self, **kwargs):

        state = kwargs.get("state")
        seed = kwargs.get("seed", None)
        stochastic = kwargs.get("stochastic", False)

        references = [
            np.array([0.7, 1.4, -2.6, -0.7, 1.4, -2.6, 0.7, 1.4, -2.6, -0.7, 1.4, -2.6]),
        ]

        settling_times = [1.0]

        delta_times = [0.25] if stochastic else None

        super().__init__(state, references, settling_times, delta_times=delta_times, seed=seed)

        self.task_level = 1
