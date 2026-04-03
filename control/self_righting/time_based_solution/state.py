from dataclasses import dataclass, field
import numpy as np


@dataclass
class TimeBasedState:
    controller_index: int = -1
    last_controller: int = -1
    controller_evolution: float = 0.0
    action_group: int = -1
