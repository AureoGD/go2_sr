import numpy as np
from control.self_righting.rgc_mpc_solution.rgc_base_controller import BaseRGCController


class HoldPosition(BaseRGCController):

    def __init__(self, robot_states, **kwargs):
        super().__init__(robot_states, **kwargs)
        self.phase = 0

        # ----------------------------------------
        # Low-level mode controller gains
        # ----------------------------------------

        self.Kp_vec = np.ones(12) * self.kp
        self.Kd_vec = np.ones(12) * self.kd

    def update_dqr(self):
        return np.zeros(12)

    def build_output_constraint_matrices(self):
        pass

    def update_model(self):
        pass

    def build_reference(self):
        pass
