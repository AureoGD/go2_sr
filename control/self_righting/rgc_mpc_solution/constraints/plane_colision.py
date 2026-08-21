import numpy as np


class PlaneConstraint:

    def __init__(self, n_local, p_offset_local, d_safe, p_safe_local):
        self.n_local = n_local / np.linalg.norm(n_local)
        self.d_safe = d_safe
        self.p_offset_local = p_offset_local
        # Fix sign once at construction using a stable safe-side reference
        p_rel = p_safe_local.flatten() - p_offset_local.flatten()
        if self.n_local @ p_rel < 0:
            self.n_local *= -1

    def update(self, R_b, p_base):
        n_w = R_b @ self.n_local
        p_off_w = p_base.flatten() + R_b @ self.p_offset_local.flatten()

        lb = self.d_safe + n_w @ p_off_w
        ub = np.inf

        return n_w, lb, ub, p_off_w
