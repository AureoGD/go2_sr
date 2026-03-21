from enum import Enum


class Phase(Enum):
    PREPARING = 0
    ROLLING = 1
    STANDING = 2
    FAIL = 3
