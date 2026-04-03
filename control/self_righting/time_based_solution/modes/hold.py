from control.self_righting.time_based_solution.tb_base_controller import BaseTimeController
import numpy as np


class Hold(BaseTimeController):

    def __init__(self, **kwargs):
        state = kwargs.get("state")
        super().__init__(state, references=[], settling_times=[])

        self.task_level = 0

    def update_dqr(self):

        return np.zeros(12)
