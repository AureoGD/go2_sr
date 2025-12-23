import numpy as np
import pinocchio as pin
from environment.strategies.rgc_mpc.go_safe import GoSafe
from environment.strategies.rgc_mpc.prepare_cw import PrepareCW
from environment.strategies.rgc_mpc.landing_cw_tb_backup import LandingCW
from environment.strategies.rgc_mpc.roll_cw import RollCW
from environment.strategies.rgc_mpc.roll_ccw import RollCCW
from environment.strategies.rgc_mpc.stand_up import StandUpPhase


class ControlScheduler():

    def __init__(self, **kwargs):
        self.robot_states = kwargs.get('robot_states', [])

        # Initialize controllers
        self.go_safe = GoSafe(**kwargs)
        self.landing_cw = LandingCW(**kwargs)
        self.prepare_cw = PrepareCW(**kwargs)
        self.stand_up = StandUpPhase(**kwargs)
        self.roll_ccw = RollCCW(**kwargs)
        self.roll_cw = RollCW(**kwargs)

        self.kp = kwargs.get('kp')
        self.kd = kwargs.get('kd')

        # Default values
        self.KP = self.kp * np.eye(12)
        self.KD = self.kd * np.eye(12)
        self.delta_qr = np.zeros((12, 1))

        # Map modes to controllers - ensure alignment with your mode definitions
        self.controller_map = {
            0: self.go_safe,  # mode 0 -> go_safe
            1: self.prepare_cw,  # mode 1 -> prepare_cw
            2: self.roll_cw,  # mode 2 -> roll_cw
            3: self.landing_cw,  # mode 4 -> landing_cw
            4: self.stand_up,  # mode 5 -> stand_up
        }

        # Alternative if you prefer list (make sure indices match modes)
        self.controllers = [
            self.go_safe,  # mode 0
            self.prepare_cw,  # mode 1
            self.roll_cw,  # mode 2
            self.landing_cw,  # mode 3
            self.stand_up  # mode 4
        ]

        self.using_mpc = False
        self.modes = 5
        self.last_mode = None

    def update(self, mode):
        # Reset MPC flag
        self.robot_states.mpc_fail = False

        # Check for invalid mode first
        if mode not in [0, 1, 2, 3, 4]:
            self.delta_qr = np.zeros((12, 1))
            self.KD = self.kd * np.eye(12)
            self.using_mpc = False
            return self.delta_qr.reshape(12), self.KP, self.KD

        # Handle mode transition
        if self.last_mode is None:
            self.last_mode = mode
        elif self.last_mode != mode:
            if self.last_mode < len(self.controllers):
                self.controllers[self.last_mode].reset_controller()
            self.last_mode = mode
        # Update delta_qr from current controller
        if mode < len(self.controllers):
            self.delta_qr = self.controllers[mode].update_dqr()
            self.controllers[mode].task_finish = self.robot_states.subtask_succes
        else:
            self.delta_qr = np.zeros((12, 1))
            self.robot_states.subtask_success = False

        if mode in [1, 2, 3]:
            self.KD = self.kd * np.eye(12)
            self.KD[3:6, 3:6] = self.kd / 10 * np.eye(3)
        elif mode == 0:
            self.KD = self.kd / 10 * np.eye(12)
        else:
            self.KD = self.kd * np.eye(12)

        self.using_mpc = (mode in [0, 1, 2, 4])

        # if mode == 0:
        #     print(f"{self.robot_states.mpc_obj_val},")

        return self.delta_qr.reshape(12), self.KP, self.KD

    def reset_controller(self):
        for i in range(len(self.controllers)):
            self.controllers[i].reset_controller()
