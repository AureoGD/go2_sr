import numpy as np
from typing import List, Tuple, Optional
import random


class SelfAdaptingCurriculum:

    def __init__(self,
                 min_difficulty: float = 0.1,
                 max_difficulty: float = 1.0,
                 cooldown_period: int = 20,
                 variance_threshold: float = 0.001,
                 ema_weight: float = 0.95,
                 slope_threshold: float = 5e-2,
                 warmup_generations: int = 20,
                 use_ema_variance: bool = True,
                 slope_stable_period: int = 5,
                 require_positive_slope: bool = True):
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.current_difficulty = min_difficulty

        self.generation_count = 0
        self.cooldown_counter = 0
        self.stable_slope_counter = 0

        self.ema_weight = float(ema_weight)
        self.slope_threshold = float(slope_threshold)
        self.warmup_generations = int(warmup_generations)
        self.use_ema_variance = bool(use_ema_variance)
        self.slope_stable_period = int(slope_stable_period)
        self.require_positive_slope = bool(require_positive_slope)
        self.slope = 0.0

        self.ema_mean = None
        self.prev_ema_mean = None
        self.ema_variance = None

        self.variance_threshold = variance_threshold
        self.cooldown_period = cooldown_period

        # Define base values for q0 and r0
        self.q0_base = np.array([-0.2, 2, -1.65, -0.6, 1.86, -1.65, -0.5, 1.06, -1.0, 0.25, 1.36, -1.05])
        self.r0_base = np.array([np.pi, 0, 0])  # Third element will be randomized

        # Define ranges for randomization
        self.q0_ranges = np.array([
            [-0.25, 0.25],  # q0[0] range
            [-0.5, 0.5],  # q0[1] range
            [-0.75, 0.75],  # q0[2] range
            [-0.25, 0.25],  # q0[3] range
            [-0.5, 0.5],  # q0[4] range
            [-0.75, 0.75],  # q0[5] range
            [-0.25, 0.25],  # q0[6] range
            [-0.5, 0.5],  # q0[7] range
            [-0.75, 0.75],  # q0[8] range
            [-0.25, 0.25],  # q0[9] range
            [-0.5, 0.5],  # q0[10] range
            [-0.75, 0.75]  # q0[11] range
        ])

        # r0[2] will be randomized between these bounds
        self.r0_z_range = [-np.pi, np.pi]

        # b0[2] will be randomized between these bounds (always positive)
        self.b0_z_range = [0.2, 0.4]

        # Base value for b0 (x=0, y=0, z=0.2 as midpoint)
        self.b0_base = np.array([0.0, 0.0, 0.2])

    def get_initial_conditions(self,
                               num_conditions: int,
                               seed: Optional[int] = None) -> List[Tuple[List[float], List[float], List[float]]]:
        """
        Generate initial conditions for reset method.
        
        Returns:
            List of tuples: [(q0_1, r0_1, b0_1), (q0_2, r0_2, b0_2), ...]
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        conditions = []

        for _ in range(num_conditions):
            # Generate q0 with random variation
            q0 = self._generate_q0()

            # Generate r0 with random z-component
            r0 = self._generate_r0()

            # Generate b0 with random z-component between 0.1 and 0.3
            b0 = self._generate_b0()

            # Return as (q0, r0, b0)
            conditions.append((q0, r0, b0))

        return conditions

    def _generate_q0(self) -> List[float]:
        """Generate q0 with random variation based on current difficulty."""
        q0_result = []

        for i in range(len(self.q0_base)):
            # Get the range for this element
            min_val, max_val = self.q0_ranges[i]

            # Scale the randomization range by current difficulty
            # At min_difficulty: near base value, at max_difficulty: full range
            range_scale = self.current_difficulty
            center = self.q0_base[i]

            # Calculate actual min/max based on difficulty
            actual_min = center + (min_val - center) * range_scale
            actual_max = center + (max_val - center) * range_scale

            # Generate random value
            rand_val = np.random.uniform(actual_min, actual_max)
            q0_result.append(float(rand_val))

        return q0_result

    def _generate_r0(self) -> List[float]:
        """Generate r0 with random z-component."""
        # r0[0] is always pi, r0[1] is always 0
        # r0[2] is random within range scaled by difficulty
        min_z, max_z = self.r0_z_range
        range_scale = self.current_difficulty

        # Calculate actual z range based on difficulty
        # At min_difficulty: near 0, at max_difficulty: full [-pi, pi]
        actual_min = min_z * range_scale
        actual_max = max_z * range_scale

        z_component = np.random.uniform(actual_min, actual_max)

        return [float(np.pi), 0.0, float(z_component)]

    def _generate_b0(self) -> List[float]:
        """Generate b0 with random z-component between 0.1 and 0.3."""
        # b0[0] is always 0, b0[1] is always 0
        # b0[2] is random between 0.1 and 0.3

        # Apply difficulty scaling to the b0 range
        min_z, max_z = self.b0_z_range

        if self.current_difficulty < 0.5:
            # At lower difficulties, bias toward lower values
            # Scale the range to be [0.1, 0.1 + (current_difficulty * 0.4)]
            scaled_max = 0.1 + (self.current_difficulty * 0.4)
            actual_min, actual_max = 0.1, min(scaled_max, 0.3)
        else:
            # At higher difficulties, use full range
            actual_min, actual_max = min_z, max_z

        z_component = np.random.uniform(actual_min, actual_max)

        return [0.0, 0.0, float(z_component)]

    # Keep your existing methods for future difficulty scaling
    def _ema_update(self, prev_ema: float | None, value: float) -> float:
        if prev_ema is None:
            return float(value)
        w = self.ema_weight
        return float(w * prev_ema + (1.0 - w) * value)

    def update_difficulty(self, evaluated_population, extra_metrics=None):
        self.generation_count += 1
        # Your difficulty update logic here
        return 1

    def get_difficulty(self) -> float:
        return self.current_difficulty

    def set_difficulty(self, difficulty: float):
        """Manually set the current difficulty level."""
        self.current_difficulty = np.clip(difficulty, self.min_difficulty, self.max_difficulty)
