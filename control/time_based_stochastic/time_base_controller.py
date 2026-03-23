import numpy as np
from environment.strategies.rgc_mpc.smooth_filter import SmoothFilter


class BaseTimeController:
    """
    Generic time-based controller.

    Handles:
        - Multiple motion stages
        - Optional stochastic settling times
        - Optional stochastic final padding
        - Time-dependent update
        - Safe handling of zero-stage controllers (e.g., Hold)
    """

    def __init__(self,
                 robot_states,
                 references,
                 settling_times,
                 t_cont=0.01,
                 delta_times=None,
                 final_padding=0.1,
                 delta_padding=None,
                 seed=None):

        self.robot_states = robot_states
        self.t_cont = t_cont
        self.tick = 0

        rng = np.random.default_rng(seed)

        # -------------------------------------------------
        # Base stage times
        # -------------------------------------------------
        self.references = references
        self.base_times = np.array(settling_times)

        # -------------------------------------------------
        # Randomize stage times (optional)
        # -------------------------------------------------
        if delta_times is not None and len(self.base_times) > 0:
            delta_times = np.array(delta_times)
            noise = rng.uniform(-delta_times, delta_times)
            self.stage_times = self.base_times + noise
            self.stage_times = np.maximum(self.stage_times, 0.05)
        else:
            self.stage_times = self.base_times.copy()

        # -------------------------------------------------
        # Randomize final padding (optional)
        # -------------------------------------------------
        if delta_padding is not None:
            pad_noise = rng.uniform(-delta_padding, delta_padding)
            self.final_padding = max(final_padding + pad_noise, 0.01)
        else:
            self.final_padding = final_padding

        # -------------------------------------------------
        # Create motion stages (SmoothFilters)
        # -------------------------------------------------
        self.stages = []

        for qr, st in zip(self.references, self.stage_times):
            config = {"robot_states": self.robot_states, "settling_time": st, "t_cont": self.t_cont, "qHL": qr}
            self.stages.append(SmoothFilter(**config))

        # -------------------------------------------------
        # Compute cumulative timing
        # -------------------------------------------------
        if len(self.stage_times) > 0:
            self.cumulative_times = np.cumsum(self.stage_times)
            self.total_time_task = self.cumulative_times[-1] + self.final_padding
        else:
            # For controllers like Hold (no motion stages)
            self.cumulative_times = np.array([])
            self.total_time_task = self.final_padding

    # -------------------------------------------------
    # Main update
    # -------------------------------------------------
    def update_dqr(self):
        spend_time = self.tick * self.t_cont

        if len(self.cumulative_times) > 0:
            idx = np.searchsorted(self.cumulative_times, spend_time)

            if idx < len(self.stages):
                delta_qr = self.stages[idx].smooth_reference().reshape(12, 1)
            else:
                delta_qr = np.zeros((12, 1))
        else:
            delta_qr = np.zeros((12, 1))

        self.tick += 1
        return delta_qr

    # -------------------------------------------------
    # Reset controller state
    # -------------------------------------------------
    def reset_controller(self):
        self.tick = 0

        # Reset SmoothFilters if they have reset method
        for stage in self.stages:
            if hasattr(stage, "reset"):
                stage.reset()

    # -------------------------------------------------
    # Expose total expected duration
    # -------------------------------------------------
    def get_total_time(self):
        return self.total_time_task

    # -------------------------------------------------
    # Expose elapsed time
    # -------------------------------------------------
    def get_elapsed_time(self):
        return self.tick * self.t_cont
