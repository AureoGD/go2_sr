import numpy as np


class SwingFootPlanner:

    def __init__(self, s0=0.45, lookahead=0.2):
        self.sigma = 0.0

        # Path parameters
        self.s0 = s0
        self.lookahead = lookahead

        # Geometry
        self.p_front = None
        self.p_rear_final = None
        self.p_mid_inter = None

    # ----------------------------------------
    # Internal sigma update
    # ----------------------------------------
    def _compute_sigma(self, foot_pos):
        if self.p_mid_inter is None or self.p_rear_final is None:
            raise RuntimeError("Geometry not initialized.")

        P2 = self.p_mid_inter.flatten()
        P3 = self.p_rear_final.flatten()
        x = foot_pos.flatten()

        dist_to_P2 = np.linalg.norm(x[:2] - P2[:2])

        if self.sigma == 0.0 and dist_to_P2 > self.s0:
            return self.sigma

        d_xy = P3[:2] - P2[:2]
        L = np.linalg.norm(d_xy)

        if L < 1e-6:
            self.sigma = 1.0
            return self.sigma

        u_vec = d_xy / L

        s_robot = float(u_vec @ (x[:2] - P2[:2]))
        s_target = s_robot + self.lookahead

        sigma_raw = np.clip(s_target / L, 0.0, 1.0)

        self.sigma = max(self.sigma, sigma_raw)

        return self.sigma

    # ----------------------------------------
    # Geometry update
    # ----------------------------------------
    def update_geometry(self, foot_pos, contacts, n, R=0.05):
        pc1, pc2, pc3, pc4 = contacts

        n = n / np.linalg.norm(n)

        # Rotation axis
        d_rot = pc4 - pc2
        d_rot = d_rot / np.linalg.norm(d_rot)

        # Perpendicular direction
        d_perp = np.cross(n, d_rot)
        d_perp = d_perp / np.linalg.norm(d_perp)

        # Ensure outward direction
        if (np.dot(d_perp, pc1 - pc2) < 0) and (np.dot(d_perp, pc3 - pc2) < 0):
            d_perp = -d_perp

        foot_world = foot_pos.flatten()

        self.p_front = pc2 + R * d_perp

        self.p_rear_final = pc4 + R * d_perp

        self.p_mid_inter = pc4 + d_rot * R / 1.5
        self.p_mid_inter[2] = foot_world[2] * 1.8

        return self.p_front, self.p_mid_inter

    # ----------------------------------------
    # Trajectory evaluation (MAIN ENTRY)
    # ----------------------------------------
    def evaluate(self, foot_pos):
        if self.p_front is None or self.p_rear_final is None:
            raise RuntimeError("Geometry not initialized. Call update_geometry first.")

        sigma = self._compute_sigma(foot_pos)

        # Planar interpolation
        p = (1 - sigma) * self.p_mid_inter + sigma * self.p_rear_final

        return p, sigma

    # ----------------------------------------
    # Reset
    # ----------------------------------------
    def reset(self):
        self.sigma = 0.0
        self.p_front = None
        self.p_rear_final = None
        self.p_mid_inter = None
