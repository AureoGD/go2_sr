import numpy as np
from control.self_righting.time_based_solution.smooth_filter import SmoothFilter


class BaseTimeController:
    """
    Generic time-based controller.
    """

    def __init__(self,
                 state,
                 references,
                 settling_times,
                 t_cont=0.01,
                 delta_times=None,
                 final_padding=0.1,
                 delta_padding=None,
                 seed=None):

        self.action_group = None
        self.state = state
        self.t_cont = t_cont
        self.tick = 0

        rng = np.random.default_rng(seed)

        # -------------------------------------------------
        # Base stage times
        # -------------------------------------------------
        self.references = references
        self.base_times = np.array(settling_times)

        if delta_times is not None:
            assert len(delta_times) == len(settling_times), \
                "delta_times must match settling_times length"

        if delta_times is not None and len(self.base_times) > 0:
            delta_times = np.array(delta_times)
            noise = rng.uniform(-delta_times, delta_times)
            self.stage_times = self.base_times + noise
            self.stage_times = np.maximum(self.stage_times, 0.05)
        else:
            self.stage_times = self.base_times.copy()

        # -------------------------------------------------
        # Randomize final padding
        # -------------------------------------------------
        if delta_padding is not None:
            pad_noise = rng.uniform(-delta_padding, delta_padding)
            self.final_padding = max(final_padding + pad_noise, 0.01)
        else:
            self.final_padding = final_padding

        # -------------------------------------------------
        # Create motion stages
        # -------------------------------------------------
        self.stages = []

        for qr, st in zip(self.references, self.stage_times):
            config = {"state": self.state, "settling_time": st, "t_cont": self.t_cont, "qHL": qr}
            self.stages.append(SmoothFilter(**config))

        # -------------------------------------------------
        # Compute cumulative timing
        # -------------------------------------------------
        if len(self.stage_times) > 0:
            self.cumulative_times = np.cumsum(self.stage_times)
            self.total_time_task = self.cumulative_times[-1] + self.final_padding
        else:
            self.cumulative_times = np.array([])
            self.total_time_task = self.final_padding

    # -------------------------------------------------
    def update_dqr(self):
        spend_time = self.tick * self.t_cont

        if len(self.cumulative_times) > 0:
            idx = np.searchsorted(self.cumulative_times, spend_time)

            if idx < len(self.stages):
                delta_qr = self.stages[idx].smooth_reference()
            else:
                delta_qr = np.zeros(12)
        else:
            delta_qr = np.zeros(12)

        self.tick += 1
        return delta_qr

    # -------------------------------------------------
    def reset_controller(self):
        self.tick = 0

        for stage in self.stages:
            if hasattr(stage, "reset"):
                stage.reset()

    # -------------------------------------------------
    def get_total_time(self):
        return self.total_time_task

    # -------------------------------------------------
    def get_elapsed_time(self):
        return self.tick * self.t_cont
