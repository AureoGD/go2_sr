from dataclasses import dataclass, field
import numpy as np


@dataclass
class TPEState:
    phase_probs: np.ndarray = field(default_factory=lambda: np.zeros(4))
    phase: int = -1
    valid: bool = False
