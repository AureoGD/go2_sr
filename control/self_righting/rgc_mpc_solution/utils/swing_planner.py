import numpy as np
from control.self_righting.rgc_mpc_solution.utils.geometry import plane_normal


class SwingFootPlanner:

    def __init__(self, N, dt, lookahead=0.1):
        self.R = 0.35
        self.lookahead = lookahead
        self.N = N
        self.delta = dt
        self.reset()

    # ----------------------------------------
    # Geometry update
    # ----------------------------------------
    def update_geometry(self, foot_pos_front, foot_pos_rear, contacts):
        pc1, pc2, pc3, pc4 = contacts  #foot_f, pivot_f, pivot_r, foot_f

        nf, _ = plane_normal(contacts[0:3])  # foot_f, pivot_f, pivot_r
        nr, _ = plane_normal(contacts[1:4])  # pivot_f, pivot_r, foot_r

        # Rotation axis
        d_rot = pc3 - pc2
        d_rot = d_rot / np.linalg.norm(d_rot)

        # Perpendicular direction

        d_perp_f = self._d_perpendicular(nf, d_rot, pc1, pc2)
        d_perp_r = self._d_perpendicular(nr, d_rot, pc4, pc3)

        self.ref_front = pc2 + self.R * d_perp_f

        self.ref_rear = pc3 + self.R * d_perp_r

        # Mid points
        self.p_mid_f = pc2 + (pc3 - pc2) * 0.75
        self.p_mid_f[2] = foot_pos_front[2]

        self.p_mid_r = pc3 + d_rot * 0.5
        self.p_mid_r[2] = 0.35

        # Swing start positions (captured once at activation)
        self.swing_start_f = foot_pos_front.flatten().copy()
        self.swing_start_r = foot_pos_rear.flatten().copy()

        self.cp_front = self._compute_bezier_controls(swing_start=self.swing_start_f,
                                                      p_mid=self.p_mid_f,
                                                      ref_final=self.ref_front,
                                                      t1=0.35,
                                                      t3=0.65)

        self.cp_rear = self._compute_bezier_controls(swing_start=self.swing_start_r,
                                                     p_mid=self.p_mid_r,
                                                     ref_final=self.ref_rear)

    def _d_perpendicular(self, n, d_rot, p_ref, p_pivot):
        d_perp = np.cross(n, d_rot)
        d_perp = d_perp / np.linalg.norm(d_perp)
        if (np.dot(d_perp, p_ref - p_pivot) < 0):
            d_perp = -d_perp
        return d_perp

    def _compute_bezier_controls(self, swing_start, p_mid, ref_final, t1=0.6, t3=0.6):
        cp0 = swing_start.flatten().copy()
        cp2 = p_mid.flatten().copy()
        cp4 = ref_final.flatten().copy()

        cp1 = cp0 + t1 * (cp2 - cp0)
        cp1[2] = cp2[2] * 0.75

        cp3 = cp2 + t3 * (cp4 - cp2)
        cp3[2] = cp2[2] * 0.75

        return np.array([cp0, cp1, cp2, cp3, cp4])  # (5,3)

    def _update_sigma(self, foot_pos, sigma, cp):
        x = foot_pos.flatten()
        p_start = cp[0]
        p_end = cp[4]

        d_xy = p_end[:2] - p_start[:2]
        L = np.linalg.norm(d_xy)

        if L < 1e-6:
            return 1.0

        u_vec = d_xy / L
        s_robot = float(u_vec @ (x[:2] - p_start[:2]))
        s_target = s_robot + self.lookahead

        sigma_raw = np.clip(s_target / L, 0.0, 1.0)
        return max(sigma, sigma_raw)

    def evaluate_at(self, t, cp):
        """
        Vectorized quartic Bezier evaluation.

        Parameters
        ----------
        t  : float or array-like (N,)  -- curve parameter in [0,1]
        cp : ndarray (5,3)             -- control points

        Returns
        -------
        p   : ndarray (N,3) -- positions
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))[:, None]  # (N,1)
        u = 1.0 - t

        p = (u**4 * cp[0] + 4 * u**3 * t * cp[1] + 6 * u**2 * t**2 * cp[2] + 4 * u * t**3 * cp[3] + t**4 * cp[4])

        return p

    def get_front_ref(self, foot_pos):
        N = self.N
        self.sigma_f = self._update_sigma(foot_pos, self.sigma_f, self.cp_front)
        t = np.linspace(self.sigma_f, min(1.0, self.sigma_f + N * self.delta), N)
        p = self.evaluate_at(t, self.cp_front)
        return p

    def get_rear_ref(self, foot_pos):
        N = self.N
        self.sigma_r = self._update_sigma(foot_pos, self.sigma_r, self.cp_rear)
        t = np.linspace(self.sigma_r, min(1.0, self.sigma_r + N * self.delta), N)
        p = self.evaluate_at(t, self.cp_rear)
        return p

    def reset(self):
        self.sigma_f = 0.0
        self.sigma_r = 0.0

        self.ref_front = None
        self.ref_rear = None
        self.p_mid_f = None
        self.p_mid_r = None
        self.swing_start_f = None
        self.swing_start_r = None
        self.cp_front = None
        self.cp_rear = None
