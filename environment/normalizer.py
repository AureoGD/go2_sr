import numpy as np
from typing import Tuple, List, Optional, Dict, Union
from dataclasses import dataclass


@dataclass
class NormalizerStats:
    """
    Data transfer object for sharing stats between processes.
    NOTE: Data is stored as Lists/Floats during transfer to prevent
    Multiprocessing 'Double Free' crashes caused by pickling NumPy arrays.
    """
    count: float
    mean: Union[np.ndarray, List[float]]
    var: Union[np.ndarray, List[float]]


class RunningNormalizer:
    """
    Standardizes data using running mean and variance.
    Supports parallel aggregation using Chan's Algorithm.
    """

    def __init__(self, shape: Tuple[int, ...], clip_range: Tuple[float, float] = (-5.0, 5.0)):
        self.shape = shape
        self.clip_range = clip_range
        # Use float64 for stability during accumulation
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray):
        """Updates stats with a batch of new data."""
        if x.shape[0] == 0:
            return

        batch_mean = np.mean(x, axis=0, dtype=np.float64)
        batch_var = np.var(x, axis=0, dtype=np.float64)
        batch_count = float(x.shape[0])
        self._merge_stats(batch_count, batch_mean, batch_var)

    def _merge_stats(self, b_count: float, b_mean: np.ndarray, b_var: np.ndarray):
        """Internal implementation of Chan's algorithm for merging."""
        delta = b_mean - self.mean
        tot_count = self.count + b_count

        new_mean = self.mean + delta * b_count / tot_count
        m_a = self.var * self.count
        m_b = b_var * b_count
        m_2 = m_a + m_b + np.square(delta) * self.count * b_count / tot_count

        self.var = m_2 / tot_count
        self.mean = new_mean
        self.count = tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        # Convert to float32 only at the very end for the network
        mean_32 = self.mean.astype(np.float32)
        var_32 = self.var.astype(np.float32)

        std = np.sqrt(var_32) + 1e-8
        normalized = (x - mean_32) / std
        return np.clip(normalized, self.clip_range[0], self.clip_range[1])

    def get_stats(self) -> NormalizerStats:
        """
        Export current stats SAFE for Multiprocessing.
        Converts NumPy arrays to standard Python Lists.
        """
        return NormalizerStats(
            count=float(self.count),
            mean=self.mean.tolist(),  # Convert to list to avoid Double Free
            var=self.var.tolist()  # Convert to list to avoid Double Free
        )

    def set_stats(self, stats: NormalizerStats):
        """Import stats (handles both Lists and Arrays)."""
        self.count = stats.count
        self.mean = np.array(stats.mean, dtype=np.float64)
        self.var = np.array(stats.var, dtype=np.float64)

    @staticmethod
    def aggregate(stats_list: List[Optional[NormalizerStats]]) -> Optional[NormalizerStats]:
        """
        Merges a list of NormalizerStats using Chan's parallel algorithm.
        Safe for both Arrays and Lists.
        """
        if not stats_list:
            return None

        # --- STEP 1: PARSE & CONVERT BACK TO NUMPY ---
        valid_stats = []
        for s in stats_list:
            if s is None:
                continue

            try:
                # Reconstruct NumPy arrays from lists if necessary
                s_mean = np.array(s.mean, dtype=np.float64)
                s_var = np.array(s.var, dtype=np.float64)
                s_count = float(s.count)

                if np.isfinite(s_count) and np.all(np.isfinite(s_mean)):
                    # Store as a temporary simple object for processing
                    valid_stats.append({'count': s_count, 'mean': s_mean, 'var': s_var})
            except (TypeError, ValueError):
                continue

        if not valid_stats:
            return None

        # --- STEP 2: AGGREGATE ---
        total_n = valid_stats[0]['count']
        grand_mean = valid_stats[0]['mean']
        # Recover M2 from Var (M2 = Var * N)
        grand_m2 = valid_stats[0]['var'] * total_n

        for s in valid_stats[1:]:
            n_b = s['count']
            mu_b = s['mean']
            m2_b = s['var'] * n_b  # Recover M2

            delta = mu_b - grand_mean
            n_new = total_n + n_b

            new_mean = grand_mean + delta * (n_b / n_new)
            new_m2 = grand_m2 + m2_b + (delta**2) * (total_n * n_b / n_new)

            total_n = n_new
            grand_mean = new_mean
            grand_m2 = new_m2

        final_var = grand_m2 / total_n

        # Return as Lists (Safe for transport if needed)
        return NormalizerStats(total_n, grand_mean.tolist(), final_var.tolist())


class Go2StateNormalizer:
    """
    Handles domain-specific normalization for the Unitree Go2 robot.
    """
    IDX = {
        'POS': slice(0, 3),
        'RVEL': slice(3, 6),
        'EPSILON': slice(6, 10),
        'OMEGA': slice(10, 13),
        'Q': slice(13, 25),
        'DQ': slice(25, 37),
        'QR': slice(37, 49),
        'TAU': slice(49, 61),
        'MODE': 61,
        'MPC_FAIL': 62,
        'CURRENT_STATE': 63,
        'SUCCESS_FLAG': 64,
    }

    def __init__(self, box_size=0.5):
        self.vel_normalizer = RunningNormalizer(shape=(18,))
        self.shadow_normalizer = RunningNormalizer(shape=(18,))
        self.box_size = box_size
        self.current_center = None
        self.is_out_of_box = False
        self.out_of_box_distance = 0.0

        self.joint_limits = np.array([
            [-1.0472, 1.0472],
            [-1.5708, 3.4907],
            [-2.7227, -0.83776],
            [-1.0472, 1.0472],
            [-1.5708, 3.4907],
            [-2.7227, -0.83776],
            [-1.0472, 1.0472],
            [-0.5236, 4.5379],
            [-2.7227, -0.83776],
            [-1.0472, 1.0472],
            [-0.5236, 4.5379],
            [-2.7227, -0.83776],
        ])

        self.torque_limits = np.array([23.7, 23.7, 45.43] * 4)
        self.num_modes = 7
        self.num_states = 6  # safe, end_prep, end_roll, end_landing, end_prone, end_stand_up

    def reset_shadow(self):
        """Resets the shadow normalizer."""
        self.shadow_normalizer = RunningNormalizer(shape=(18,))

    def normalize(self, state: np.ndarray, update_stats: bool = False) -> np.ndarray:
        if update_stats:
            vels = self._extract_velocities(state)
            self.shadow_normalizer.update(vels.reshape(1, -1))

        normalized = np.zeros_like(state)

        normalized[self.IDX['POS']] = self._norm_pos(state[self.IDX['POS']])

        vels = self._extract_velocities(state)
        norm_vels = self.vel_normalizer.normalize(vels.reshape(1, -1)).flatten()

        normalized[self.IDX['RVEL']] = norm_vels[0:3]
        normalized[self.IDX['OMEGA']] = norm_vels[3:6]
        normalized[self.IDX['DQ']] = norm_vels[6:18]

        normalized[self.IDX['EPSILON']] = np.clip(state[self.IDX['EPSILON']], -1.0, 1.0)
        normalized[self.IDX['Q']] = self._norm_limits(state[self.IDX['Q']], self.joint_limits)
        normalized[self.IDX['QR']] = self._norm_limits(state[self.IDX['QR']], self.joint_limits)
        normalized[self.IDX['TAU']] = state[self.IDX['TAU']] / self.torque_limits
        normalized[self.IDX['MODE']] = state[self.IDX['MODE']] / self.num_modes
        normalized[self.IDX['MPC_FAIL']] = state[self.IDX['MPC_FAIL']]
        normalized[self.IDX['CURRENT_STATE']] = state[self.IDX['CURRENT_STATE']] / self.num_states
        normalized[self.IDX['SUCCESS_FLAG']] = state[self.IDX['SUCCESS_FLAG']]

        return normalized

    def get_shadow_stats(self) -> NormalizerStats:
        return self.shadow_normalizer.get_stats()

    def sync_global_stats(self, global_stats: NormalizerStats):
        # Handle cases where global_stats might be None on first run
        if global_stats is not None:
            self.vel_normalizer.set_stats(global_stats)
        self.reset_shadow()

    def _extract_velocities(self, state):
        return np.concatenate([state[self.IDX['RVEL']], state[self.IDX['OMEGA']], state[self.IDX['DQ']]])

    def _norm_pos(self, pos):
        if self.current_center is None:
            self.current_center = pos.copy()
        relative = (pos - self.current_center) / (self.box_size / 2)
        self.is_out_of_box = np.any(np.abs(relative) > 1.0)
        clipped = np.clip(relative, -1.0, 1.0)
        self.out_of_box_distance = np.linalg.norm(relative - clipped)
        return clipped

    def _norm_limits(self, val, limits):
        return np.clip(2.0 * (val - limits[:, 0]) / (limits[:, 1] - limits[:, 0]) - 1.0, -1.0, 1.0)

    def reset_reference(self):
        self.current_center = None
        self.is_out_of_box = False
