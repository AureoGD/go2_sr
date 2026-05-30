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

        self.p_mid_inter = pc4 + d_rot * 0.5
        self.p_mid_inter[2] = foot_world[2] * 3

        self.p_swing_start = foot_world.copy()

    def get_front_reference(self):
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

    def _compute_bezier_controls(self, lift_dz=0.04, land_dz=0.04):
        P0 = self.p_swing_start
        P5 = self.p_rear_final.flatten()
        Pm = self.p_mid_inter.flatten()
        z_peak = Pm[2]

        chord_mid_xy = 0.5 * (P0[:2] + P5[:2])
        lateral = Pm[:2] - chord_mid_xy
        lateral_norm = lateral / np.linalg.norm(lateral)

        reach_P1 = np.dot(P0[:2] - chord_mid_xy, lateral_norm)
        reach_P4 = np.dot(P5[:2] - chord_mid_xy, lateral_norm)

        # P1: lift-off — stay near P0 xy, just rise slightly
        xy1 = P0[:2] + max(reach_P1, 0) * lateral_norm
        self.p_ctrl1 = np.array([xy1[0], xy1[1], P0[2] + lift_dz])  # ← fixed

        # P2: full lateral clearance, peak height
        self.p_ctrl2 = Pm.copy()

        # P3: midpoint xy between Pm and P5, still at peak
        xy3 = 0.5 * (Pm[:2] + P5[:2])
        self.p_ctrl3 = np.array([xy3[0], xy3[1], z_peak])

        # P4: pre-landing — stay near P5 xy, just above ground
        xy4 = P5[:2] + max(reach_P4, 0) * lateral_norm
        self.p_ctrl4 = np.array([xy4[0], xy4[1], P5[2] + land_dz])  # ← fixed
        # ----------------------------------------

    # NEW: Bézier trajectory evaluation
    # ----------------------------------------
    def evaluate_bezier(self, foot_pos):
        if self.p_ctrl1 is None:
            raise RuntimeError("Bézier geometry not initialized.")

        sigma = self._compute_sigma_bezier(foot_pos)  # <-- use this instead
        t = sigma
        u = 1 - t

        P0 = self.p_swing_start
        P1 = self.p_ctrl1
        P2 = self.p_ctrl2
        P3 = self.p_ctrl3
        P4 = self.p_ctrl4
        P5 = self.p_rear_final.flatten()

        p = (u**5 * P0 + 5 * u**4 * t * P1 + 10 * u**3 * t**2 * P2 + 10 * u**2 * t**3 * P3 + 5 * u * t**4 * P4 +
             t**5 * P5)

        vel = 5 * (u**4 * (P1 - P0) + 4 * u**3 * t * (P2 - P1) + 6 * u**2 * t**2 * (P3 - P2) + 4 * u * t**3 *
                   (P4 - P3) + t**4 * (P5 - P4))

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
