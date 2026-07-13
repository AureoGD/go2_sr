import numpy as np
from control.self_righting.base_self_righting_controller import BaseSelfRighting
from control.self_righting.time_based_solution.state import TimeBasedState
# -------------------------------------------------
# Import time-based controllers
# -------------------------------------------------
from control.self_righting.time_based_solution.modes import (
    Hold,
    GoSafe,
    PrepareCW,
    RollCW,
    LandingCW,
    ProneCW,
    StandUp,
    PrepareCCW,
    RollCCW,
    LandingCCW,
    ProneCCW,
)

CONTROLLER_CLASSES = [
    Hold, GoSafe, PrepareCW, RollCW, LandingCW, ProneCW, StandUp, PrepareCCW, RollCCW, LandingCCW, ProneCCW
]


class SchedulerTB(BaseSelfRighting):

    # ======================================================
    # INIT
    # ======================================================
    def __init__(self, stochastic=False, **kwargs):
        super().__init__()
        self.comp_grav = True
        self.stochastic = stochastic

        self.task_state = TimeBasedState()

        self._base_kwargs = {k: v for k, v in kwargs.items() if k not in ["robot_states", "seed"]}

        kp = kwargs.get("kp", 50.0)
        kd = kwargs.get("kd", 3.0)

        self.Kp_vec = np.ones(12) * kp
        self.Kd_vec = np.ones(12) * kd

        self.controllers = None
        self.n_controllers = 0
        self._controllers_initialized = False

        self.task_state.last_controller = None
        self.delta_qr = np.zeros(12)

    def get_num_modes(self):
        self.num_modes = len(CONTROLLER_CLASSES)
        return self.num_modes

    def _instantiate_controllers(self, state):

        self.controllers = [
            cls(state=state, seed=np.random.randint(0, 1_000_000), stochastic=self.stochastic, **self._base_kwargs)
            for cls in CONTROLLER_CLASSES
        ]

        self.n_controllers = len(self.controllers)
        self._controllers_initialized = True

    # ======================================================
    # RESET
    # ======================================================
    def reset_phase(self):

        self.task_state.last_controller = None
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

        if self.task_state.last_controller != controller_idx:
            self.controllers[controller_idx].reset_controller()
            self.delta_qr[:] = 0.0
            self.task_state.last_controller = controller_idx

        active_ctrl = self.controllers[controller_idx]

        self.delta_qr = active_ctrl.update_dqr().reshape(12)

        percent_task = np.clip(active_ctrl.get_elapsed_time() / active_ctrl.get_total_time(), 0, 1)

        self.task_state.controller_index = controller_idx
        self.task_state.action_group = active_ctrl.action_group
        self.task_state.controller_evolution = percent_task

        return self.delta_qr, self.Kp_vec, self.Kd_vec

    # ======================================================
    # OPTIONAL
    # ======================================================
    def get_phase_mapping(self):
        return None, None
