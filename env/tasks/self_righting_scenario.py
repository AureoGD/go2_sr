import numpy as np
from typing import List
from env.tasks.base_task_scenario import BaseTaskScenario


class SelfRightingScenario(BaseTaskScenario):

    def __init__(self, config=None):
        self.config = config or {}

        self.q0_base = np.array([-0.2, 2, -1.65, -0.6, 1.86, -1.65, -0.5, 1.06, -1.0, 0.25, 1.36, -1.05])

        self.q0_ranges = np.array([[-0.25, 0.25], [-0.5, 0.5], [-0.75, 0.75], [-0.25, 0.25], [-0.5, 0.5], [-0.75, 0.75],
                                   [-0.25, 0.25], [-0.5, 0.5], [-0.75, 0.75], [-0.25, 0.25], [-0.5, 0.5], [-0.75,
                                                                                                           0.75]])

        self.r0_base = np.array([np.pi, 0.0, 0.0])
        self.r0_yaw_range = [-np.pi, np.pi]

        self.b0_base = np.array([0.0, 0.0, 0.2])
        self.b0_range = np.array([[-1.0, 1.0], [-1.0, 1.0], [0.0, 0.1]])

    def sample(self):
        return {"q0": self._generate_q0(), "r0": self._generate_r0(), "b0": self._generate_b0()}

    def apply(self, env, scenario):
        return env.reset(q0=scenario["q0"], r0=scenario["r0"], b0=scenario["b0"])

    def _generate_q0(self) -> np.ndarray:
        """Generate q0 with random variation within defined ranges."""
        mins = self.q0_base + self.q0_ranges[:, 0]
        maxs = self.q0_base + self.q0_ranges[:, 1]
        return np.random.uniform(mins, maxs)

    def _generate_r0(self) -> np.ndarray:
        """
        Generate base orientation:
        - Roll and pitch around base (with small/no variation)
        - Yaw fully random
        """
        r0 = self.r0_base.copy()

        # Apenas yaw aleatório
        r0[2] = np.random.uniform(self.r0_yaw_range[0], self.r0_yaw_range[1])

        return r0

    def _generate_b0(self) -> np.ndarray:
        """Generate base position with variation."""
        mins = self.b0_base + self.b0_range[:, 0]
        maxs = self.b0_base + self.b0_range[:, 1]
        return np.random.uniform(mins, maxs)
