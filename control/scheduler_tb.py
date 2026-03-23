import numpy as np
from environment.strategies.base_self_righting import BaseSelfRighting

# -------------------------------------------------
# Import stochastic time-based controllers
# -------------------------------------------------
from environment.strategies.time_based_stochastic.hold import Hold
from environment.strategies.time_based_stochastic.go_safe import GoSafe
from environment.strategies.time_based_stochastic.prepare_cw import PrepareCW
from environment.strategies.time_based_stochastic.roll_cw import RollCW
from environment.strategies.time_based_stochastic.landing_cw import LandingCW
from environment.strategies.time_based_stochastic.prone_cw import ProneCW
from environment.strategies.time_based_stochastic.stand_up import StandUp
from environment.strategies.time_based_stochastic.prepare_ccw import PrepareCCW
from environment.strategies.time_based_stochastic.roll_ccw import RollCCW
from environment.strategies.time_based_stochastic.landing_ccw import LandingCCW
from environment.strategies.time_based_stochastic.prone_ccw import ProneCCW

# -------------------------------------------------
# Controller ordering (IMPORTANT)
# -------------------------------------------------
# Index -> Controller
#
# 0  -> Hold
# 1  -> GoSafe
# 2  -> PrepareCW
# 3  -> RollCW
# 4  -> LandingCW
# 5  -> ProneCW
# 6  -> StandUp
# 7  -> PrepareCCW
# 8  -> RollCCW
# 9  -> LandingCCW
# 10 -> ProneCCW
#
# This mapping MUST stay consistent with NN action space.

CONTROLLER_CLASSES = [
    Hold, GoSafe, PrepareCW, RollCW, LandingCW, ProneCW, StandUp, PrepareCCW, RollCCW, LandingCCW, ProneCCW
]


class SchedulerTB(BaseSelfRighting):
    """
    Time-based self-righting scheduler.
    Neural network selects controller index directly.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # -------------------------------------------------
        # Store base kwargs (needed for stochastic reset)
        # -------------------------------------------------
        self._base_kwargs = kwargs.copy()

        # -------------------------------------------------
        # Gains
        # -------------------------------------------------
        self.kp = kwargs.get("kp", 50)
        self.kd = kwargs.get("kd", 3)

        self.KP = self.kp * np.eye(12)
        self.KD = self.kd * np.eye(12)

        # -------------------------------------------------
        # Instantiate controllers
        # -------------------------------------------------
        self._instantiate_controllers()

        # -------------------------------------------------
        # Runtime state
        # -------------------------------------------------
        self.last_controller = None
        self.delta_qr = np.zeros(12)

    # -------------------------------------------------
    # Controller instantiation helper
    # -------------------------------------------------
    def _instantiate_controllers(self):
        """
        Instantiate controllers.
        Used at initialization and episode reset.
        """
        self.controllers = [cls(**self._base_kwargs) for cls in CONTROLLER_CLASSES]
        self.n_controllers = len(self.controllers)

    # -------------------------------------------------
    # Episode reset (IMPORTANT for stochastic timing)
    # -------------------------------------------------
    def reset_phase(self):
        """
        Reset scheduler at episode start.
        Re-instantiates controllers to resample
        stochastic timing if enabled.
        """
        self.last_controller = None
        self.delta_qr = np.zeros(12)

        # Recreate controllers (important for stochastic)
        self._instantiate_controllers()

    # -------------------------------------------------
    # Main update entry point
    # -------------------------------------------------
    def update(self, controller=None):
        """
        Args:
            controller (int): Selected by neural network.
        Returns:
            delta_qr (np.ndarray): Joint reference increment
            KP (np.ndarray): Proportional gains
            KD (np.ndarray): Derivative gains
        """

        # Safety fallback
        if controller is None:
            controller = 0

        # Validate index
        if controller < 0 or controller >= self.n_controllers:
            return np.zeros(12), self.KP, self.KD

        # If controller changed → reset it
        if self.last_controller != controller:
            self.controllers[controller].reset_controller()
            self.delta_qr = np.zeros(12)
            self.last_controller = controller

        active_ctrl = self.controllers[controller]

        # Update controller
        self.delta_qr = active_ctrl.update_dqr().reshape(12)

        return self.delta_qr, self.KP, self.KD

    def get_phase_mapping(self):
        return None, None
