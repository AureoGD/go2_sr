import numpy as np
from environment.strategies.base_self_righting import BaseSelfRighting

# -------------------------------------------------
# Import controllers
# -------------------------------------------------
from environment.strategies.rgc_mpc.hold_position import HoldPosition
from environment.strategies.rgc_mpc.go_safe import GoSafe
from environment.strategies.rgc_mpc.prepare_cw import PrepareCW
from environment.strategies.rgc_mpc.roll_cw import RollCW
# from environment.strategies.rgc_mpc.landing_cw_tb_backup import LandingCW
from environment.strategies.rgc_mpc.landing_cw_rgc import LandingCW
from environment.strategies.rgc_mpc.end_landing_cw import EndLandingCW
from environment.strategies.rgc_mpc.stand_up import StandUpPhase

CONTROLLER_CLASSES = [HoldPosition, GoSafe, PrepareCW, RollCW, LandingCW, EndLandingCW, StandUpPhase]


class SchedulerRGCMPC(BaseSelfRighting):
    """
    Mode-based RGC-MPC scheduler.

    Mode semantics (index = NN action):
        0 -> HOLD
        1 -> GoSafe
        2 -> PrepareCW
        3 -> RollCW
        4 -> LandingCW
        5 -> StandUp
        6 -> RollCCW (optional / future)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # -------------------------------------------------
        # Default gains
        # -------------------------------------------------
        self.kp = kwargs.get("kp", 50)
        self.kd = kwargs.get("kd", 3)

        # -------------------------------------------------
        # Instantiate controllers (safe for metadata-only)
        # -------------------------------------------------
        self.controllers = [cls(**kwargs) for cls in CONTROLLER_CLASSES]
        self.modes = len(self.controllers)

        # -------------------------------------------------
        # Runtime state
        # -------------------------------------------------
        self.last_mode = None
        self.delta_qr = np.zeros(12)

        # -------------------------------------------------
        # Explicit gains initialization
        # -------------------------------------------------
        self.KP = self.kp * np.eye(12)
        self.KD = self.kd * np.eye(12)

    # -------------------------------------------------
    # Main update entry point
    # -------------------------------------------------
    def update(self, mode=None):
        """
        Args:
            mode (int): Selected by the neural network.
        Returns:
            delta_qr (np.ndarray): Joint reference increment
            KP (np.ndarray): Proportional gains
            KD (np.ndarray): Derivative gains
        """
        # Safety fallback
        if mode is None:
            mode = 0

        # Validate mode
        if mode < 0 or mode >= self.modes:
            return np.zeros(12), self.KP, self.KD
        # Ensure that the mpc_fail flag is false before any MPC being solved
        self.robot_states.mpc_fail = False

        # -------------------------------------------------
        # Handle transitions
        # Reset the NEW controller on entry (by design)
        # -------------------------------------------------
        if self.last_mode != mode:
            self.controllers[mode].reset_controller()
            self.last_mode = mode

        # -------------------------------------------------
        # Execute active controller
        # -------------------------------------------------
        active_ctrl = self.controllers[mode]
        self.delta_qr = active_ctrl.update_dqr().reshape(12)

        # -------------------------------------------------
        # Dynamic gains logic (RGC specific)
        # -------------------------------------------------
        if mode == 0:
            # HOLD / SAFE
            self.KP = self.kp * np.eye(12)
            self.KD = self.kd * np.eye(12)

        elif mode == 1:
            # GoSafe: softer damping
            self.KP = self.kp * np.eye(12)
            self.KD = (self.kd / 10.0) * np.eye(12)

        elif mode in [2, 3, 4]:
            # Prepare / Roll / Landing CW
            self.KP = self.kp * np.eye(12)
            self.KD = self.kd * np.eye(12)
            self.KD[3:6, 3:6] = (self.kd / 10.0) * np.eye(3)

        else:
            # StandUp or future controllers
            self.KP = self.kp * np.eye(12)
            self.KD = self.kd * np.eye(12)

        return self.delta_qr, self.KP, self.KD

    # -------------------------------------------------
    # Phase metadata (for logging / analysis only)
    # -------------------------------------------------
    def get_phase_mapping(self):
        """
        Returns:
            dict: {task_name: task_level}

        Enforces that every controller defines valid
        semantic metadata.
        """
        phase_mapping = {}

        for ctrl in self.controllers:
            if ctrl is None:
                continue

            if not hasattr(ctrl, "task_name") or not hasattr(ctrl, "task_level"):
                raise ValueError(f"Controller {ctrl.__class__.__name__} "
                                 "does not define task_name/task_level")

            if ctrl.task_name in (None, "undefined"):
                raise ValueError(f"Controller {ctrl.__class__.__name__} "
                                 "has invalid task_name")

            if ctrl.task_level is None or ctrl.task_level < 0:
                raise ValueError(f"Controller {ctrl.__class__.__name__} "
                                 f"has invalid task_level={ctrl.task_level}")

            if ctrl.task_name in phase_mapping:
                raise ValueError(f"Duplicate task_name '{ctrl.task_name}' detected")

            phase_mapping[ctrl.task_name] = ctrl.task_level

        return phase_mapping

    # -------------------------------------------------
    # Full reset (episode boundary)
    # -------------------------------------------------
    def reset_phase(self):
        self.last_mode = None
        for ctrl in self.controllers:
            if hasattr(ctrl, "reset_controller"):
                ctrl.reset_controller()


if __name__ == "__main__":
    print("=== Testing SchedulerRGCMPC ===")

    # Create dummy scheduler (no robot_states)
    scheduler = SchedulerRGCMPC()

    print("\nControllers list:")
    for i, ctrl in enumerate(scheduler.controllers):
        print(f"  Mode {i}: "
              f"class={ctrl.__class__.__name__}, "
              f"task_name={getattr(ctrl, 'task_name', None)}, "
              f"task_level={getattr(ctrl, 'task_level', None)}")

    print("\nPhase mapping:")
    phase_mapping = scheduler.get_phase_mapping()
    for name, level in phase_mapping.items():
        print(f"  {name:15s} -> level {level}")

    print("\nTest update() calls:")
    for mode in range(len(scheduler.controllers)):
        dq, KP, KD = scheduler.update(mode)
        print(f"  Mode {mode}: "
              f"delta_qr_norm={np.linalg.norm(dq):.3f}, "
              f"KP_shape={KP.shape}, "
              f"KD_shape={KD.shape}")

    print("\nAll tests completed successfully.")
