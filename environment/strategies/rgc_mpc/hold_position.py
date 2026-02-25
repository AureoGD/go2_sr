import numpy as np
from environment.strategies.rgc_mpc.base_controller import BaseRGC


class HoldPosition(BaseRGC):
    TASK_NAME = "hold"
    TASK_LEVEL = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not self.runtime:
            # Metadata-only: nothing else to do
            return

    def update_dqr(self):
        # HOLD is always valid: clear any failure flags
        if self.runtime:
            self.robot_states.mpc_fail = False
            self.robot_states.mpc_critical_fail = False

        return np.zeros(12)
