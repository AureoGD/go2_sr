from control.self_righting.time_based_solution.tb_base_controller import BaseTimeController
import numpy as np


class StandUp(BaseTimeController):

    def __init__(self, **kwargs):

        state = kwargs.get("state")
        seed = kwargs.get("seed", None)
        stochastic = kwargs.get("stochastic", False)

        references = [np.array([[0.0, 1.0, -2.0, 0, 1.0, -2.0, 0, 1.0, -2.0, 0, 1.0, -2.0]])]

        settling_times = [2.0]

        delta_times = [0.25] if stochastic else None

        super().__init__(state, references, settling_times, delta_times=delta_times, seed=seed)
        self.phase = 6
