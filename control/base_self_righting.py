import numpy as np
from abc import ABC, abstractmethod


class BaseSelfRighting(ABC):

    def __init__(self, **kwargs):
        """
        Base constructor.
        Expects 'robot_states' in kwargs.
        """
        self.robot_states = kwargs.get('robot_states')

        # Default returns if things go wrong
        self.qr_ant = np.zeros((12, 1))
        self.KP = np.eye(12)
        self.KD = np.eye(12)

    @abstractmethod
    def update(self, mode=None):
        """
        Computes the next command.
        Args:
            mode (int, optional): The strategy ID provided by the Neural Network.
                                  Unitree logic might ignore this.
                                  RGC logic requires this.
        Returns:
            (q_target, KP, KD)
        """
        pass

    @abstractmethod
    def reset_phase(self):
        """
        Resets internal timers or state machines.
        """
        pass
