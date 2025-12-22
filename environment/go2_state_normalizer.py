import numpy as np
from typing import Tuple, List, Optional, Dict, Union
from dataclasses import dataclass


@dataclass
class NormalizerStats:
    """Data transfer object for sharing stats between processes."""
    count: float
    mean: np.ndarray
    var: np.ndarray


class RunningNormalizer:
    """
    Standardizes data using running mean and variance.
    Supports parallel aggregation using Chan's Algorithm.
    """

    def __init__(self, shape: Tuple[int, ...], clip_range: Tuple[float, float] = (-5.0, 5.0)):
        self.shape = shape
        self.clip_range = clip_range
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 1e-4

    def update(self, x: np.ndarray):
        """Updates stats with a batch of new data."""
        if x.shape[0] == 0:
            return

        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
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
        std = np.sqrt(self.var) + 1e-8
        normalized = (x - self.mean) / std
        return np.clip(normalized, self.clip_range[0], self.clip_range[1])

    def get_stats(self) -> NormalizerStats:
        """Export current stats."""
        return NormalizerStats(self.count, self.mean.copy(), self.var.copy())

    def set_stats(self, stats: NormalizerStats):
        """Import stats (e.g., from global history)."""
        self.count = stats.count
        self.mean = stats.mean
        self.var = stats.var

    @staticmethod
    def aggregate(stats_list: List[Optional[NormalizerStats]]) -> Optional[NormalizerStats]:
        """
        Merges a list of NormalizerStats using Chan's parallel algorithm.
        Robust to None values and invalid types.
        """
        if not stats_list:
            return None

        # --- FIX 2: ROBUST FILTERING ---
        valid_stats = []
        for s in stats_list:
            # 1. Check if object exists
            if s is None:
                continue
            # 2. Check if attributes exist (in case of bad return object)
            if not hasattr(s, 'count') or not hasattr(s, 'mean'):
                continue

            # 3. Check values safely
            try:
                # Explicit float conversion check to catch 'ufunc' errors
                if np.isfinite(s.count) and np.all(np.isfinite(s.mean)):
                    valid_stats.append(s)
            except (TypeError, ValueError):
                continue

        if not valid_stats:
            return None

        # Initialize with first valid
        total_n = valid_stats[0].count
        grand_mean = valid_stats[0].mean
        # Recover M2 from Var (M2 = Var * N)
        grand_m2 = valid_stats[0].var * valid_stats[0].count

        for s in valid_stats[1:]:
            n_b = s.count
            mu_b = s.mean
            m2_b = s.var * n_b  # Recover M2

            delta = mu_b - grand_mean
            n_new = total_n + n_b

            new_mean = grand_mean + delta * (n_b / n_new)
            new_m2 = grand_m2 + m2_b + (delta**2) * (total_n * n_b / n_new)

            total_n = n_new
            grand_mean = new_mean
            grand_m2 = new_m2

        final_var = grand_m2 / total_n
        return NormalizerStats(total_n, grand_mean, final_var)


class Go2StateNormalizer:
    """
    Handles domain-specific normalization for the Unitree Go2 robot.
    Manages both hard-limits (Joints) and soft-stats (Velocities).
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
        'MPC_FAIL': 62
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
        self.num_modes = 5

    # --- FIX 1: ADDED MISSING METHOD ---
    def reset_shadow(self):
        """Resets the shadow normalizer. Essential for worker initialization."""
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

        return normalized

    def get_shadow_stats(self) -> NormalizerStats:
        return self.shadow_normalizer.get_stats()

    def sync_global_stats(self, global_stats: NormalizerStats):
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
