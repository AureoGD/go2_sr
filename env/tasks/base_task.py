import numpy as np
from abc import ABC, abstractmethod


class BaseTask(ABC):

    def __init__(self, step_limit=1000, normalizer=None):
        self.step_limit = step_limit
        self.normalizer = normalizer
        self._features = None
        self.state_copy = None
        self.obs_dim = -1
        self.difficulty = 1

    def set_difficulty(self, difficulty):
        self.difficulty = difficulty

    def gen_info(self):
        return {}

    def reset(self):
        self._features = None

    @abstractmethod
    def compute_features(self, state):
        pass

    @abstractmethod
    def get_obs(self):
        pass

    @abstractmethod
    def evaluate_reward(self):
        pass

    @abstractmethod
    def check_termination(self, current_step):
        pass
