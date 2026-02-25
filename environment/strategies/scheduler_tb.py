import numpy as np
from environment.strategies.base_self_righting import BaseSelfRighting

# -------------------------------------------------
# Import controllers
# -------------------------------------------------
from environment.strategies.time_based.hold_position import HoldPosition
from environment.strategies.time_based.go_safe_tb import GoSafe
from environment.strategies.time_based.prepare_cw_tb import PrepareCW
from environment.strategies.time_based.roll_cw_tb import RollCW
from environment.strategies.time_based.landing_cw_tb import LandingCW
from environment.strategies.time_based.prone_cw_tb import ProneCW
from environment.strategies.time_based.standin_up_tb import StandUpPhase

CONTROLLER_CLASSES = [HoldPosition, GoSafe, PrepareCW, RollCW, LandingCW, ProneCW, StandUpPhase]


class SchedulerTB(BaseSelfRighting):
    """
    Mode-based RGC-MPC scheduler.

    Mode semantics (index = NN action):
        0 -> HOLD
        1 -> GoSafe
        2 -> PrepareCW
        3 -> RollCW
        4 -> LandingCW
        5 -> StandUp
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
        self.n_controllers = len(self.controllers)

        # -------------------------------------------------
        # Runtime state
        # -------------------------------------------------
        self.last_controller = None
        self.delta_qr = np.zeros(12)

        # -------------------------------------------------
        # Explicit gains initialization
        # -------------------------------------------------
        self.KP = self.kp * np.eye(12)
        self.KD = self.kd * np.eye(12)

    # -------------------------------------------------
    # Main update entry point
    # -------------------------------------------------
    def update(self, controller=None):
        """
        Args:
            controller (int): Selected by the neural network.
        Returns:
            delta_qr (np.ndarray): Joint reference increment
            KP (np.ndarray): Proportional gains
            KD (np.ndarray): Derivative gains
        """
        # Safety fallback
        if controller is None:
            controller = 0

        # Validate controller
        if controller < 0 or controller >= self.n_controllers:
            return np.zeros(12), self.KP, self.KD

        if self.last_controller != controller:
            self.controllers[controller].reset_controller()
            self.last_controller = controller

        active_ctrl = self.controllers[controller]
        self.delta_qr = active_ctrl.update_dqr().reshape(12)

        return self.delta_qr, self.KP, self.KD

    # -------------------------------------------------
    # Phase metadata (for logging / analysis only)
    # -------------------------------------------------
    def get_phase_mapping(self):
        """
        Returns:
            dict:  {task_name: task_level}
            list:  [task_level per controller_index]
        """
        phase_mapping = {}  # name → level
        phase_index_list = [0] * len(self.controllers)  # index → level

        for idx, ctrl in enumerate(self.controllers):
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

            # populate structures
            phase_mapping[ctrl.task_name] = ctrl.task_level
            phase_index_list[idx] = ctrl.task_level

        return phase_mapping, phase_index_list

    # -------------------------------------------------
    # Full reset (episode boundary)
    # -------------------------------------------------
    def reset_phase(self):
        self.last_controller = None
        for ctrl in self.controllers:
            if hasattr(ctrl, "reset_controller"):
                ctrl.reset_controller()


if __name__ == "__main__":
    print("=== Testing SchedulerRGCMPC ===")

    # Create dummy scheduler (no robot_states)
    scheduler = SchedulerTB()

    print("\nControllers list:")
    for i, ctrl in enumerate(scheduler.controllers):
        print(f"  Mode {i}: "
              f"class={ctrl.__class__.__name__}, "
              f"task_name={getattr(ctrl, 'task_name', None)}, "
              f"task_level={getattr(ctrl, 'task_level', None)}")

    print("\nPhase mapping:")
    phase_mapping, _ = scheduler.get_phase_mapping()
    for name, level in phase_mapping.items():
        print(f"  {name:15s} -> level {level}")

    # print("\nTest update() calls:")
    # for controller in range(len(scheduler.controllers)):
    #     dq, KP, KD = scheduler.update(controller)
    #     print(f"  Mode {controller}: "
    #           f"delta_qr_norm={np.linalg.norm(dq):.3f}, "
    #           f"KP_shape={KP.shape}, "
    #           f"KD_shape={KD.shape}")

    print("\nAll tests completed successfully.")
