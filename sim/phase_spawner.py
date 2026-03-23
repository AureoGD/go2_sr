from typing import Tuple
import numpy as np


class PhaseSpawner():

    def __init__(self, phase_library, prob_phase_spawn=0.4):
        self.lib = phase_library
        self.p = prob_phase_spawn

    def apply(self):
        """
        scenario = (q0, r0, b0) from LearningPhases
        returns same OR replaced by PhaseLibrary pose
        """
        # if np.random.random() > self.p:
        #     return scenario  # leave environment spawn unchanged

        phase_idx = np.random.choice(self.lib.available_indices())
        q, b, r = self.lib.get(phase_idx, add_noise=True)
        return (q.tolist(), r.tolist(), b.tolist(), phase_idx)
