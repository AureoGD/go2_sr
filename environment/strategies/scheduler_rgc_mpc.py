import numpy as np
from environment.strategies.base_self_righting import BaseSelfRighting

# Import specific phases from the subfolder
from environment.strategies.rgc_mpc.go_safe import GoSafe
from environment.strategies.rgc_mpc.prepare_cw import PrepareCW
from environment.strategies.rgc_mpc.landing_cw_tb_backup import LandingCW
from environment.strategies.rgc_mpc.roll_cw import RollCW
from environment.strategies.rgc_mpc.roll_ccw import RollCCW
from environment.strategies.rgc_mpc.stand_up import StandUpPhase


class SchedulerRGCMPC(BaseSelfRighting):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.kp = kwargs.get('kp', 50)
        self.kd = kwargs.get('kd', 3)

        # Initialize sub-controllers
        # We pass kwargs down so they get robot_states too
        self.go_safe = GoSafe(**kwargs)
        self.prepare_cw = PrepareCW(**kwargs)
        self.landing_cw = LandingCW(**kwargs)
        self.stand_up = StandUpPhase(**kwargs)
        self.roll_cw = RollCW(**kwargs)
        self.roll_ccw = RollCCW(**kwargs)

        self.controllers = [
            self.go_safe,  # 0
            self.prepare_cw,  # 1
            self.roll_cw,  # 2
            self.landing_cw,  # 3
            self.stand_up,  # 4]
        ]

        self.modes = len(self.controllers)

        self.last_mode = None
        self.delta_qr = np.zeros((12, 1))

    def update(self, mode=None):
        """
        Args:
            mode (int): CRITICAL here. Determined by the Neural Network.
        """
        # Safety fallback
        if mode is None:
            mode = 0

        if mode < 0 or mode >= len(self.controllers):
            return np.zeros(12), self.KP, self.KD

        # Handle Transitions
        if self.last_mode != mode:
            if self.last_mode is not None:
                self.controllers[mode].reset_controller()
            self.last_mode = mode

        # Execute Active Controller
        active_ctrl = self.controllers[mode]
        self.delta_qr = active_ctrl.update_dqr()

        # Sync success state if needed
        if hasattr(active_ctrl, 'task_finish') and hasattr(self.robot_states, 'subtask_succes'):
            active_ctrl.task_finish = self.robot_states.subtask_succes

        # Dynamic Gains Logic (RGC specific)
        if mode in [1, 2, 3]:
            self.KD = self.kd * np.eye(12)
            self.KD[3:6, 3:6] = (self.kd / 10.0) * np.eye(3)
        elif mode == 0:
            self.KD = (self.kd / 10.0) * np.eye(12)
        else:
            self.KD = self.kd * np.eye(12)

        return self.delta_qr.reshape(12), self.KP, self.KD

    def reset_phase(self):
        self.last_mode = None
        for ctrl in self.controllers:
            if hasattr(ctrl, 'reset_controller'):
                ctrl.reset_controller()
