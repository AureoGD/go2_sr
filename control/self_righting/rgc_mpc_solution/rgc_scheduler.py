import numpy as np
from control.self_righting.base_self_righting_controller import BaseSelfRighting
from control.self_righting.rgc_mpc_solution.state import RGCState

from control.self_righting.rgc_mpc_solution.modes.hold_position import HoldPosition
from control.self_righting.rgc_mpc_solution.modes.go_safe import GoSafe
from control.self_righting.rgc_mpc_solution.modes.prepare_cw import PrepareCW
from control.self_righting.rgc_mpc_solution.modes.roll_cw import RollCW
from control.self_righting.rgc_mpc_solution.modes.landing_cw import LandingCW
from control.self_righting.rgc_mpc_solution.modes.prone_cw import ProneCW
from control.self_righting.rgc_mpc_solution.modes.prone_final import prone_final
from control.self_righting.rgc_mpc_solution.modes.stand_up import StandUp

CONTROLLER_CLASSES = [HoldPosition, GoSafe, PrepareCW, RollCW, LandingCW, ProneCW, prone_final, StandUp]


class SchedulerRGCMPC(BaseSelfRighting):

    def __init__(self, **kwargs):

        kp = kwargs.get("kp", 50.0)
        kd = kwargs.get("kd", 3.0)

        self.Kp_vec = np.ones(12) * kp
        self.Kd_vec = np.ones(12) * kd

        self.controllers = None
        self.n_controllers = 0
        self._controllers_initialized = False

        self.delta_qr = np.zeros(12)

        self.task_state = RGCState()
        self.task_state.last_controller = None

        self._base_kwargs = {"pin_engine": kwargs.get("pin_engine"), "task_state": self.task_state, "kp": kp, "kd": kd}

    def _instantiate_controllers(self, state):
        self.controllers = []

        for cls in CONTROLLER_CLASSES:

            ctrl = cls(robot_states=state, **self._base_kwargs)

            self.controllers.append(ctrl)

        self.n_controllers = len(self.controllers)
        self._controllers_initialized = True

    def get_num_modes(self):
        self.num_modes = len(CONTROLLER_CLASSES)
        return self.num_modes

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

        self.task_state.controller_index = controller_idx
        self.task_state.action_group = active_ctrl.action_group

        Kp_vec, Kd_vec = active_ctrl.get_gains()

        return self.delta_qr, Kp_vec, Kd_vec

    def reset_phase(self):
        self.task_state.last_controller = None
        self.delta_qr[:] = 0.0
        self._controllers_initialized = False
