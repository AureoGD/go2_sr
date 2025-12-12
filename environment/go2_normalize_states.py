import numpy as np


class RunningNormalizer:

    def __init__(self, shape, clip_range=(-5, 5)):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 1e-4
        self.clip_range = clip_range

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        total_count = self.count + batch_count
        delta = batch_mean - self.mean

        self.mean += delta * batch_count / total_count
        self.var = (self.count * self.var + batch_count * batch_var +
                    np.square(delta) * self.count * batch_count / total_count) / total_count
        self.count = total_count

    def normalize(self, x):
        std = np.sqrt(self.var) + 1e-8
        normalized = (x - self.mean) / std
        return np.clip(normalized, self.clip_range[0], self.clip_range[1])

    def save(self, filepath):
        np.savez(filepath, mean=self.mean, var=self.var, count=self.count)

    def load(self, filepath):
        data = np.load(filepath)
        self.mean = data['mean']
        self.var = data['var']
        self.count = data['count']


class Go2StateNormalizer:

    def __init__(self, box_size=0.5):
        # State indices
        self.POS_START, self.POS_END = 0, 3
        self.RVEL_START, self.RVEL_END = 3, 6
        self.EPSILON_START, self.EPSILON_END = 6, 10
        self.OMEGA_START, self.OMEGA_END = 10, 13
        self.Q_START, self.Q_END = 13, 25
        self.DQ_START, self.DQ_END = 25, 37
        self.QR_START, self.QR_END = 37, 49
        self.TAU_START, self.TAU_END = 49, 61
        self.MODE = 61
        self.MPC_FAIL = 62

        # === 1. Main Normalizer (The "Actor") ===
        # Holds Global History. Used for normalizing inputs during inference.
        # This should remain static during an episode.
        self.vel_normalizer = RunningNormalizer(shape=(18,))  # r_vel(3) + omega(3) + dq(12)

        # === 2. Shadow Normalizer (The "Collector") ===
        # Holds ONLY new samples from the current run.
        # Used to calculate updates for the next generation.
        self.shadow_normalizer = RunningNormalizer(shape=(18,))

        # Position normalization settings
        self.box_size = box_size
        self.current_center = None
        self.is_out_of_box = False
        self.out_of_box_distance = 0.0

        # === GO2 SPECIFIC JOINT LIMITS ===
        self.joint_limits = np.array([
            [-1.0472, 1.0472],  # FR_abduction
            [-1.5708, 3.4907],  # FR_hip 
            [-2.7227, -0.83776],  # FR_knee
            [-1.0472, 1.0472],  # FL_abduction
            [-1.5708, 3.4907],  # FL_hip 
            [-2.7227, -0.83776],  # FL_knee
            [-1.0472, 1.0472],  # RR_abduction  
            [-0.5236, 4.5379],  # RR_hip 
            [-2.7227, -0.83776],  # RR_knee
            [-1.0472, 1.0472],  # RL_abduction
            [-0.5236, 4.5379],  # RL_hip 
            [-2.7227, -0.83776],  # RL_knee
        ])

        # === GO2 SPECIFIC TORQUE LIMITS ===
        self.torque_limits = np.array([
            23.7,
            23.7,
            45.43,  # FR
            23.7,
            23.7,
            45.43,  # FL
            23.7,
            23.7,
            45.43,  # RR
            23.7,
            23.7,
            45.43,  # RL
        ])

        self.num_modes = 5

    def set_reference_position(self, start_pos):
        """Set the reference position (typically at episode start)"""
        self.current_center = start_pos.copy()
        self.is_out_of_box = False
        self.out_of_box_distance = 0.0

    def normalize_position(self, pos):
        """Normalize position relative to current center with fixed box size"""
        if self.current_center is None:
            self.current_center = pos.copy()
            print(f"Auto-set reference position: {self.current_center}")

        relative_pos = pos - self.current_center
        normalized = relative_pos / (self.box_size / 2)

        self.is_out_of_box = np.any(np.abs(normalized) > 1.0)
        clipped_normalized = np.clip(normalized, -1.0, 1.0)
        self.out_of_box_distance = np.linalg.norm(normalized - clipped_normalized)

        return clipped_normalized

    def denormalize_position(self, normalized_pos):
        if self.current_center is None:
            raise ValueError("No reference position set!")
        return self.current_center + normalized_pos * (self.box_size / 2)

    def get_box_info(self):
        if self.current_center is None:
            return {'is_out_of_box': False, 'out_of_box_distance': 0.0, 'box_size': self.box_size}

        half_size = self.box_size / 2
        box_bounds = {
            'x_min': self.current_center[0] - half_size,
            'x_max': self.current_center[0] + half_size,
            'y_min': self.current_center[1] - half_size,
            'y_max': self.current_center[1] + half_size,
            'z_min': self.current_center[2] - half_size,
            'z_max': self.current_center[2] + half_size
        }

        return {
            'is_out_of_box': self.is_out_of_box,
            'out_of_box_distance': self.out_of_box_distance,
            'box_center': self.current_center.copy(),
            'box_bounds': box_bounds,
            'box_size': self.box_size
        }

    def get_out_of_box_direction(self, pos):
        if self.current_center is None or not self.is_out_of_box:
            return None
        relative_pos = pos - self.current_center
        normalized = relative_pos / (self.box_size / 2)
        directions = []
        axis_names = ['x', 'y', 'z']
        for i in range(3):
            if normalized[i] > 1.0:
                directions.append(f"{axis_names[i]}+")
            elif normalized[i] < -1.0:
                directions.append(f"{axis_names[i]}-")
        return directions

    def normalize(self, state):
        """Normalize full state vector using hybrid strategy"""
        normalized = np.zeros_like(state)

        # 1. Positions
        if self.current_center is None:
            self.set_reference_position(state[self.POS_START:self.POS_END])
        normalized[self.POS_START:self.POS_END] = self.normalize_position(state[self.POS_START:self.POS_END])

        # 2. Velocities: Use self.vel_normalizer (The frozen Global History)
        velocities = np.concatenate([
            state[self.RVEL_START:self.RVEL_END], state[self.OMEGA_START:self.OMEGA_END],
            state[self.DQ_START:self.DQ_END]
        ])

        # NOTE: We normalize using the Main Normalizer (Policy Stability)
        normalized_vels = self.vel_normalizer.normalize(velocities.reshape(1, -1)).flatten()

        normalized[self.RVEL_START:self.RVEL_END] = normalized_vels[0:3]
        normalized[self.OMEGA_START:self.OMEGA_END] = normalized_vels[3:6]
        normalized[self.DQ_START:self.DQ_END] = normalized_vels[6:18]

        # 3. Quaternions
        normalized[self.EPSILON_START:self.EPSILON_END] = np.clip(state[self.EPSILON_START:self.EPSILON_END], -1.0, 1.0)

        # 4. Joint positions & Reference
        normalized[self.Q_START:self.Q_END] = self._normalize_with_limits(state[self.Q_START:self.Q_END],
                                                                          self.joint_limits)
        normalized[self.QR_START:self.QR_END] = self._normalize_with_limits(state[self.QR_START:self.QR_END],
                                                                            self.joint_limits)

        # 5. Torques & Mode
        normalized[self.TAU_START:self.TAU_END] = state[self.TAU_START:self.TAU_END] / self.torque_limits
        normalized[self.MODE] = state[self.MODE] / self.num_modes
        normalized[self.MPC_FAIL] = state[self.MPC_FAIL]
        return normalized

    def update(self, state):
        """
        Update running statistics.
        CRITICAL: We update the SHADOW normalizer, not the main one.
        The main one is static during the episode.
        """
        velocities = np.concatenate([
            state[self.RVEL_START:self.RVEL_END], state[self.OMEGA_START:self.OMEGA_END],
            state[self.DQ_START:self.DQ_END]
        ])
        # Update the collector
        self.shadow_normalizer.update(velocities.reshape(1, -1))

    def reset_shadow(self):
        """
        Resets the shadow normalizer. 
        Call this at the beginning of run_worker.
        """
        self.shadow_normalizer = RunningNormalizer(shape=(18,))

    def _normalize_with_limits(self, values, limits):
        normalized = 2.0 * (values - limits[:, 0]) / (limits[:, 1] - limits[:, 0]) - 1.0
        return np.clip(normalized, -1.0, 1.0)

    def save_weights(self, filepath):
        """Save the Main Normalizer weights (the ones we actually use)"""
        self.vel_normalizer.save(filepath)
        print(f"Saved normalizer weights to: {filepath}")

    def load_weights(self, filepath):
        self.vel_normalizer.load(filepath)
        print(f"Loaded normalizer weights from: {filepath}")

    def reset_reference(self):
        self.current_center = None
        self.is_out_of_box = False
        self.out_of_box_distance = 0.0

    def get_joint_info(self):
        leg_names = ['FR', 'FL', 'RR', 'RL']
        joint_names = ['abduction', 'hip', 'knee']
        print("\n=== Go2 Joint Configuration ===")
        for i in range(12):
            leg_idx = i // 3
            joint_idx = i % 3
            print(
                f"{leg_names[leg_idx]}_{joint_names[joint_idx]}: [{self.joint_limits[i, 0]:.3f}, {self.joint_limits[i, 1]:.3f}] rad"
            )

    def get_state_info(self):
        return {
            'position_reference_set': self.current_center is not None,
            'is_out_of_box': self.is_out_of_box,
            'out_of_box_distance': self.out_of_box_distance,
            'velocity_normalizer_stats': {
                'mean': self.vel_normalizer.mean,
                'std': np.sqrt(self.vel_normalizer.var),
                'count': self.vel_normalizer.count
            }
        }
