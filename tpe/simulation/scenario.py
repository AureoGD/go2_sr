import numpy as np
import random


class TimeSchedulingScenario:

    def __init__(self):

        self.q0_base = np.array([0.7, 1.0, -2.1, 0, 1.4, -1.75, 0, 1.4, -1.8, 0, 1.4, -2.0])

        # scheduling
        self.base_durations = np.array([25, 110, 310, 360, 260, 210, 210])
        self.delta = np.array([15, 20, 50, 30, 30, 30, 30])

        self.tick = 0
        self.current_durations = None
        self.current_tick = None
        self.direction = None

    # ======================================================
    # SAMPLE EPISODE
    # ======================================================
    def reset(self):

        self.tick = 0

        # ------------------------
        # initial state
        # ------------------------
        noise = np.random.uniform(-0.5, 0.5, 12)
        self.q0 = self.q0_base + noise

        yaw = random.uniform(-np.pi, np.pi)
        self.rpy = np.array([np.pi, 0, yaw])

        x0 = random.uniform(-1, 2.75)
        self.b0 = np.array([x0, 0, 0.3])

        # ------------------------
        # durations
        # ------------------------
        durations = self.base_durations + np.random.uniform(-self.delta, self.delta)
        durations = np.maximum(durations, 1)
        durations = np.round(durations).astype(int)

        self.current_durations = durations
        self.current_tick = np.cumsum(durations)

        # ------------------------
        # direction
        # ------------------------
        self.direction = np.random.choice(["cw", "ccw"])

    # ======================================================
    # ACTION LOGIC
    # ======================================================
    def _map_action(self, logical_idx):

        if self.direction == "cw":
            mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
        else:
            mapping = {0: 0, 1: 1, 2: 7, 3: 8, 4: 9, 5: 10, 6: 6}

        return mapping[logical_idx]

    def get_action(self):

        self.tick += 1

        idx = np.searchsorted(self.current_tick, self.tick)

        if idx < len(self.current_tick):
            return self._map_action(idx)

        return 0

    # ======================================================
    # HELPERS
    # ======================================================
    def get_initial_state(self):
        return self.q0, self.rpy, self.b0

    def get_total_steps(self):
        return int(self.current_tick[-1])
