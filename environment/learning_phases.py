import numpy as np
import random
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class ScenarioConfig:
    """
    Defines the environment and spawn parameters for a specific learning phase.
    """
    name: str
    difficulty_id: int

    # --- Environment Settings ---
    # The center point where the robot should spawn (on average)
    terrain_origin: np.ndarray
    # The angle of the floor [Roll, Pitch, Yaw]
    terrain_angle_rad: np.ndarray

    # --- Randomization Constraints ---
    x_noise_range: Tuple[float, float] = (-0.75, 0.75)
    y_noise_range: Tuple[float, float] = (-0.75, 0.75)
    z_noise_range: Tuple[float, float] = (0.2, 0.35)


class LearningPhases:
    """
    Manages the initial conditions (q0, r0, b0) for different difficulty stages.
    Decoupled from the curriculum progression logic.
    """

    def __init__(self):

        self.scenarios = [
            # STAGE 0: Flat Ground (Easy)
            ScenarioConfig(
                name="Flat_Easy",
                difficulty_id=0,
                terrain_origin=np.array([0.0, 0.0, 0.0]),
                terrain_angle_rad=np.array([0.0, 0.0, 0.0]),
            ),

            # # STAGE 1: Flat HeightField
            # ScenarioConfig(name="Flat_HF",
            #                difficulty_id=1,
            #                terrain_origin=np.array([0.0, 3.0, 0.0]),
            #                terrain_angle_rad=np.array([0.0, 0.0, 0.0]),
            #                z_noise_range=(0.5, 0.8)),

            # # STAGE 2: Slope 5°
            # # Note: Origin Z is pre-calculated to be "on the slope" at X=3.25
            # ScenarioConfig(
            #     name="Slope_5",
            #     difficulty_id=2,
            #     terrain_origin=np.array([3.25, 0.0, 0.0]),  # Z will be handled by logic or pre-calc if needed
            #     terrain_angle_rad=np.array([0.0, np.deg2rad(5), 0.0]),
            # ),

            # # STAGE 3: HeightField Slope 5°
            # ScenarioConfig(name="Slope_5_HF",
            #                difficulty_id=3,
            #                terrain_origin=np.array([3.25, 3.0, 0.0]),
            #                terrain_angle_rad=np.array([0.0, np.deg2rad(5), 0.0]),
            #                z_noise_range=(0.5, 0.8))
        ]

        self.q0_base = np.array([-0.2, 2, -1.65, -0.6, 1.86, -1.65, -0.5, 1.06, -1.0, 0.25, 1.36, -1.05])

        self.r0_base = np.array([np.pi, 0.0, 0.0])

        self.r0_yaw_range = [-np.pi, np.pi]

        self.q0_ranges = np.array([[-0.25, 0.25], [-0.5, 0.5], [-0.75, 0.75], [-0.25, 0.25], [-0.5, 0.5], [-0.75, 0.75],
                                   [-0.25, 0.25], [-0.5, 0.5], [-0.75, 0.75], [-0.25, 0.25], [-0.5, 0.5], [-0.75,
                                                                                                           0.75]])

        self.q0_upside = np.array([0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7])
        self.b0_upside = np.array([0, 0, 0.2])
        self.r0_upside = np.array([0, 0, 0])

    def get_initial_conditions(self,
                               num_conditions: int,
                               max_difficulty_id: int,
                               seed: Optional[int] = None) -> List[Tuple[List[float], List[float], List[float]]]:

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        ceiling_idx = min(max_difficulty_id, len(self.scenarios) - 1)
        active_indices = list(range(0, ceiling_idx + 1))

        if len(active_indices) == 1:
            probs = [1.0]
        else:
            frontier_prob = 0.5
            history_prob = (1.0 - frontier_prob) / (len(active_indices) - 1)
            probs = [history_prob] * (len(active_indices) - 1) + [frontier_prob]

        conditions = []

        for _ in range(num_conditions):
            val = np.random.random()
            if val < 0.6:
                selected_idx = np.random.choice(active_indices, p=probs)
                config = self.scenarios[selected_idx]

                q0 = self._generate_q0()
                r0 = self._generate_r0(config)
                b0 = self._generate_b0(config)
            else:
                q0 = self.q0_base.tolist()
                b0 = self.b0_upside.tolist()
                r0 = self.r0_upside
                r0[2] = np.random.uniform(self.r0_yaw_range[0], self.r0_yaw_range[1])
                r0 = r0.tolist()

            conditions.append((q0, r0, b0))

        return conditions, 0

    def _generate_q0(self) -> List[float]:
        """Generate q0 with random variation within defined ranges."""
        mins = self.q0_base + self.q0_ranges[:, 0]
        maxs = self.q0_base + self.q0_ranges[:, 1]

        q0_result = np.random.uniform(mins, maxs)
        return q0_result.tolist()

    def _generate_r0(self, config: ScenarioConfig) -> List[float]:
        """
        Generate orientation (r0).
        Logic: Base(UpsideDown) + SlopeAlignment + RandomYaw
        """
        r0 = self.r0_base.copy()

        r0[0] += config.terrain_angle_rad[0]  # Roll
        r0[1] += config.terrain_angle_rad[1]  # Pitch

        random_yaw = np.random.uniform(self.r0_yaw_range[0], self.r0_yaw_range[1])
        r0[2] = random_yaw

        return r0.tolist()

    def _generate_b0(self, config: ScenarioConfig) -> List[float]:
        """
        Generate base position (b0).
        Logic: 
        1. Randomize X/Y around the origin.
        2. Calculate geometric Z of the floor at that new X/Y.
        3. Add Z-drop noise.
        """
        x_noise = np.random.uniform(config.x_noise_range[0], config.x_noise_range[1])
        y_noise = np.random.uniform(config.y_noise_range[0], config.y_noise_range[1])

        x_final = config.terrain_origin[0] + x_noise
        y_final = config.terrain_origin[1] + y_noise

        pitch = config.terrain_angle_rad[1]
        roll = config.terrain_angle_rad[0]

        z_geometric = (x_final * np.tan(pitch)) + (y_final * np.tan(roll))

        z_drop = np.random.uniform(config.z_noise_range[0], config.z_noise_range[1])

        z_final = z_geometric + z_drop

        if z_final < 0:
            print("ERROR")

        return [float(x_final), float(y_final), float(z_final)]
