from dataclasses import dataclass, field
import numpy as np


@dataclass
class DebugState:
    sw_foot_data: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
