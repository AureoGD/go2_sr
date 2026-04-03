import numpy as np
from control.base_self_righting import BaseSelfRighting

# -------------------------------------------------
# Import stochastic time-based controllers
# -------------------------------------------------
from control.time_based_stochastic.hold import Hold
from control.time_based_stochastic.go_safe import GoSafe
from control.time_based_stochastic.prepare_cw import PrepareCW
from control.time_based_stochastic.roll_cw import RollCW
from control.time_based_stochastic.landing_cw import LandingCW
from control.time_based_stochastic.prone_cw import ProneCW
from control.time_based_stochastic.stand_up import StandUp
from control.time_based_stochastic.prepare_ccw import PrepareCCW
from control.time_based_stochastic.roll_ccw import RollCCW
from control.time_based_stochastic.landing_ccw import LandingCCW
from control.time_based_stochastic.prone_ccw import ProneCCW

CONTROLLER_CLASSES = [
    Hold, GoSafe, PrepareCW, RollCW, LandingCW, ProneCW, StandUp, PrepareCCW, RollCCW, LandingCCW, ProneCCW
]


class SchedulerTB(BaseSelfRighting):

    # ======================================================
    # INIT
    # ======================================================
    def __init__(self, **kwargs):
        super().__init__()

        self._base_kwargs = {k: v for k, v in kwargs.items() if k != "robot_states"}

        kp = kwargs.get("kp", 50.0)
        kd = kwargs.get("kd", 3.0)

        self.Kp_vec = np.ones(12) * kp
        self.Kd_vec = np.ones(12) * kd

        self.controllers = None
        self.n_controllers = 0
        self._controllers_initialized = False

        self.last_controller = None
        self.delta_qr = np.zeros(12)

    def get_num_modes(self):
        self.num_modes = len(CONTROLLER_CLASSES)
        return self.num_modes

    def _instantiate_controllers(self, state):

        self.controllers = [cls(state=state, **self._base_kwargs) for cls in CONTROLLER_CLASSES]

        self.n_controllers = len(self.controllers)
        self._controllers_initialized = True

    # ======================================================
    # RESET
    # ======================================================
    def reset_phase(self):

        self.last_controller = None
        self.delta_qr[:] = 0.0
        self._controllers_initialized = False

    # ======================================================
    # MAIN LOGIC
    # ======================================================
    def compute_action(self, state, action):

        if not self._controllers_initialized:
            self._instantiate_controllers(state)

        if action is None:
            controller_idx = 0
        else:
            controller_idx = int(action)

        if controller_idx < 0 or controller_idx >= self.n_controllers:
            return np.zeros(12), self.Kp_vec, self.Kd_vec

        if self.last_controller != controller_idx:
            self.controllers[controller_idx].reset_controller()
            self.delta_qr[:] = 0.0
            self.last_controller = controller_idx

        active_ctrl = self.controllers[controller_idx]

        self.delta_qr = active_ctrl.update_dqr().reshape(12)

        percent_task = np.clip(active_ctrl.get_elapsed_time() / active_ctrl.get_total_time(), 0, 1)
        state.controller.controller_evolution = percent_task
        state.controller.sr_semantics = active_ctrl.task_level
        state.controller.controller_index = action

        return self.delta_qr, self.Kp_vec, self.Kd_vec

    # ======================================================
    # OPTIONAL
    # ======================================================
    def get_phase_mapping(self):
        return None, None
