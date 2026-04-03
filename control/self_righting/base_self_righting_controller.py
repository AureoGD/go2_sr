import numpy as np
from abc import abstractmethod
import gymnasium as gym

from control.base_controller import BaseController


class BaseSelfRighting(BaseController):

    def __init__(self):
        super().__init__()

        # --------------------------------------
        # INTERNAL MEMORY
        # --------------------------------------
        self.dqr = np.zeros(12)

    # ======================================================
    # ACTION SPACE (default: discrete modes)
    # ======================================================
    def get_action_space(self):
        """
        Default self-righting uses discrete modes.
        Override if needed.
        """
        self.get_num_modes()
        return gym.spaces.Discrete(self.num_modes)

    @abstractmethod
    def get_num_modes(self):
        pass

    # ======================================================
    # BEFORE STEP (MAIN ENTRY POINT)
    # ======================================================
    def before_step(self, state, action):

        cs = state.low_level

        # --------------------------------------
        # 1. APPLY PREVIOUS DELTA (MEMORY)
        # --------------------------------------
        cs.qr = cs.qr + self.dqr

        # --------------------------------------
        # 2. COMPUTE NEW DELTA
        # --------------------------------------
        self.dqr, Kp, Kd = self.compute_action(state, action)

        # --------------------------------------
        # 3. WRITE BACK
        # --------------------------------------
        cs.dqr = self.dqr
        cs.Kp = Kp
        cs.Kd = Kd

    # ======================================================
    # CORE LOGIC (TO IMPLEMENT)
    # ======================================================
    @abstractmethod
    def compute_action(self, state, action):
        """
        Must return:
            qr (12,)
            Kp (12,)
            Kd (12,)
        """
        pass

    # ======================================================
    # RESET
    # ======================================================
    def reset(self):
        self.dqr = np.zeros(12)
        self.reset_phase()

    @abstractmethod
    def reset_phase(self):
        pass

    # ======================================================
    # OPTIONAL
    # ======================================================
    def after_step(self, state):
        pass
