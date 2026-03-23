from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


# ============================================================
# PHASE STATE
# ============================================================
@dataclass
class PhaseState:
    """
    Represents a recorded terminal configuration of a semantic phase.
    """
    name: str
    index: int  # semantic phase index (1..N)
    q: np.ndarray  # joint positions (12,)
    b: np.ndarray  # base position (3,)
    r: np.ndarray  # base orientation rpy (3,)

    def sample(self,
               noise: bool = True,
               q_noise: float = 0.02,
               b_noise: float = 0.005,
               r_noise: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return a (q, b, r) sample, optionally with small noise.
        """
        q = self.q.copy()
        b = self.b.copy()
        r = self.r.copy()

        if noise:
            q += np.random.uniform(-q_noise, q_noise, size=q.shape)
            b += np.random.uniform(-b_noise, b_noise, size=b.shape)
            r += np.random.uniform(-r_noise, r_noise, size=r.shape)

        return q, b, r


# ============================================================
# PHASE LIBRARY
# ============================================================
class PhaseLibrary:
    """
    Holds a set of semantic phase terminal configurations.
    Used only for spawning, never for reward logic.
    """

    def __init__(self):
        self._phases: Dict[int, PhaseState] = {}
        self._load_defaults()

    # --------------------------------------------------------
    # LOAD DEFAULT RECORDED PHASES
    # --------------------------------------------------------
    def _load_defaults(self):
        """
        Load recorded self-righting phases.
        """

        # ===== go_safe =====
        self.add(
            PhaseState(name="go_safe",
                       index=1,
                       b=np.array([-0.004, -0.004, 0.098]),
                       r=np.array([3.142, -0.078, -1.596]),
                       q=np.array(
                           [0.676, 1.391, -2.559, -0.686, 1.400, -2.566, 0.705, 1.382, -2.545, -0.706, 1.386, -2.547])))

        # ===== prepare_cw =====
        self.add(
            PhaseState(name="prepare_cw",
                       index=2,
                       b=np.array([-0.006, -0.005, 0.102]),
                       r=np.array([-3.037, -0.061, -1.632]),
                       q=np.array(
                           [-0.546, 1.614, -2.332, -0.813, 1.028, -2.602, -0.350, 1.635, -2.391, -1.048, 2.932,
                            -2.327])))

        # ===== roll_cw =====
        self.add(
            PhaseState(name="roll_cw",
                       index=3,
                       b=np.array([0.150, -0.011, 0.181]),
                       r=np.array([1.471, 0.003, -1.503]),
                       q=np.array(
                           [0.129, 1.584, -2.192, -0.806, 1.009, -2.605, 0.191, 1.637, -2.281, 0.980, 3.990, -1.658])))

        # ===== landing_cw =====
        self.add(
            PhaseState(name="landing_cw",
                       index=4,
                       b=np.array([0.147, -0.012, 0.180]),
                       r=np.array([1.329, -0.000356, -1.457]),
                       q=np.array(
                           [0.488, 1.496, -2.660, -0.818, 0.509, -1.741, 0.503, 1.505, -2.670, -0.831, 0.540, -1.767])))

        # ===== prone_cw =====
        self.add(
            PhaseState(name="prone_cw",
                       index=5,
                       b=np.array([0.0, 0.0, 0.20]),
                       r=np.array([0.0, 0.0, 0.0]),
                       q=np.array([0.0, 1.4, -2.7, 0.0, 1.4, -2.7, 0.0, 1.4, -2.7, 0.0, 1.4, -2.7])))

    # --------------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------------
    def add(self, state: PhaseState):
        if state.index in self._phases:
            raise ValueError(f"Duplicate phase index {state.index}")
        self._phases[state.index] = state

    def available_indices(self) -> List[int]:
        return sorted(self._phases.keys())

    def get(self, idx: int, add_noise: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if idx not in self._phases:
            raise KeyError(f"Phase index {idx} not found in PhaseLibrary")
        return self._phases[idx].sample(noise=add_noise)

    def has_phases(self) -> bool:
        return len(self._phases) > 0
