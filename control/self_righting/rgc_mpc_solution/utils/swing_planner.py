import numpy as np


class SwingFootPlanner:

    def __init__(self, bezier_mode=False, s0=0.08, lookahead=0.05):
        self.bezier_mode = bezier_mode
        self.clearance = 0.08
        self.sigma = 0.0

        # Path parameters
        self.s0 = s0
        self.lookahead = lookahead

        # Geometry
        self.p_front = None
        self.p_rear_final = None
        self.p_mid_inter = None
        self.p_ctrl1 = None
        self.p_ctrl2 = None
        self.terrain_fn = None
        self.p_swing_start = None

    # ----------------------------------------
    # Geometry update
    # ----------------------------------------
    def update_geometry(self, foot_pos_front, foot_pos_rear, contacts, n, R=0.05):

        self.foot_pos_front = foot_pos_front

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

        foot_world = foot_pos_rear.flatten()

        self.p_front = pc2 + R * d_perp

        self.p_rear_final = pc4 + R * d_perp

        self.p_mid_inter = pc4 + d_rot * 0.5
        self.p_mid_inter[2] = 0.35

        self.p_swing_start = foot_world.copy()

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

        if not self.bezier_mode:
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

    def get_front_reference(self, foot_pos):
        return self.p_front

    def get_rear_reference(self, foot_pos):
        if self.bezier_mode:
            if self.p_ctrl1 is None:
                self._compute_bezier_controls()
            p, sigma, _ = self.evaluate_bezier(foot_pos)
        else:
            p, sigma = self.evaluate(foot_pos)

        return p, sigma

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

    def _compute_bezier_controls(self, t1=0.6, t3=0.6):
        P0 = self.p_swing_start
        P4 = self.p_rear_final.flatten()
        self.p_ctrl2 = self.p_mid_inter.flatten()

        # P1 along the line P0 → P2
        self.p_ctrl1 = P0 + t1 * (self.p_ctrl2 - P0)
        self.p_ctrl1[2] = self.p_ctrl2[2] * 0.75

        # P3 along the line P2 → P4
        self.p_ctrl3 = self.p_ctrl2 + t3 * (P4 - self.p_ctrl2)
        self.p_ctrl3[2] = self.p_ctrl2[2] * 0.75

    # NEW: Bézier trajectory evaluation
    # ----------------------------------------
    def evaluate_bezier(self, foot_pos):
        if self.p_ctrl1 is None:
            raise RuntimeError("Bézier geometry not initialized.")

        sigma = self._compute_sigma_bezier(foot_pos)
        t = sigma
        u = 1 - t

        P0 = self.p_swing_start
        P1 = self.p_ctrl1
        P2 = self.p_ctrl2  # p_mid_inter
        P3 = self.p_ctrl3
        P4 = self.p_rear_final.flatten()

        p = (u**4 * P0 + 4 * u**3 * t * P1 + 6 * u**2 * t**2 * P2 + 4 * u * t**3 * P3 + t**4 * P4)

        vel = 4 * (u**3 * (P1 - P0) + 3 * u**2 * t * (P2 - P1) + 3 * u * t**2 * (P3 - P2) + t**3 * (P4 - P3))

        return p, sigma, vel

    def _compute_sigma_bezier(self, foot_pos):
        P0 = self.p_swing_start
        P5 = self.p_rear_final.flatten()
        x = foot_pos.flatten()

        d_xy = P5[:2] - P0[:2]
        L = np.linalg.norm(d_xy)

        if L < 1e-6:
            self.sigma = 1.0
            return self.sigma

        u_vec = d_xy / L
        s_robot = float(u_vec @ (x[:2] - P0[:2]))
        s_target = s_robot + self.lookahead

        sigma_raw = np.clip(s_target / L, 0.0, 1.0)
        self.sigma = max(self.sigma, sigma_raw)

        return self.sigma

    # ----------------------------------------
    # Reset
    # ----------------------------------------
    def reset(self):
        self.sigma = 0.0
        self.p_front = None
        self.p_rear_final = None
        self.p_mid_inter = None

        self.p_ctrl1 = None  # NEW
        self.p_ctrl2 = None  # NEW
