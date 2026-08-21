import numpy as np
from control.self_righting.rgc_mpc_solution.utils.geometry import plane_normal


def _quartic_bezier(t, cp):
    """Vectorized quartic Bezier evaluation. t: (N,) in [0,1], cp: (5,3) -> (N,3)."""
    t = np.atleast_1d(np.asarray(t, dtype=float))[:, None]  # (N,1)
    u = 1.0 - t
    return (u**4 * cp[0] + 4 * u**3 * t * cp[1] + 6 * u**2 * t**2 * cp[2] + 4 * u * t**3 * cp[3] + t**4 * cp[4])


class _SwingLeg:
    """Per-leg swing state: owns its control points, its parametrization
    progress (sigma), and its shape params (t1, t3). The planner sets up the
    geometry; the leg handles per-tick sampling and endpoint refresh."""

    def __init__(self, t1, t3):
        self.t1 = t1
        self.t3 = t3
        self.cp = None  # (5,3)
        self.sigma = 0.0

    # -- setup ------------------------------------------------------------
    def build(self, swing_start, p_mid, ref_final):
        cp0 = np.asarray(swing_start).flatten().copy()
        cp2 = np.asarray(p_mid).flatten().copy()
        cp4 = np.asarray(ref_final).flatten().copy()

        cp1 = cp0 + self.t1 * (cp2 - cp0)
        # cp1[2] = cp2[2] * 0.75

        cp3 = self._descent_tangent(cp2, cp4)

        self.cp = np.array([cp0, cp1, cp2, cp3, cp4])  # (5,3)
        self.sigma = 0.0

    # -- per-tick endpoint refresh ---------------------------------------
    def rebuild_endpoint(self, ref_final):
        """Move cp4 (endpoint) and cp3 (descent tangent) to a new target,
        leaving cp0/cp1/cp2 (start + apex) frozen."""
        self.cp[4] = np.asarray(ref_final).flatten()
        self.cp[3] = self._descent_tangent(self.cp[2], self.cp[4])

    def _descent_tangent(self, apex, endpoint):
        cp3 = apex + self.t3 * (endpoint - apex)
        cp3[2] = apex[2] * 0.75
        return cp3

    # -- per-tick sampling ------------------------------------------------
    def advance(self, foot_pos, N, dt, lookahead):
        """Advance the parametrization and sample the reference horizon.

        Returns
        -------
        ref   : ndarray (N,3) -- reference positions over the horizon
        error : ndarray (3,)  -- terminal error (ref_final - current foot pos)
        """
        x = np.asarray(foot_pos).flatten()
        self.sigma = self._update_sigma(x, lookahead)
        t = np.linspace(self.sigma, min(1.0, self.sigma + N * dt), N)
        ref = _quartic_bezier(t, self.cp)
        error = self.cp[4] - x
        return ref, error

    def _update_sigma(self, x, lookahead):
        p_start, p_end = self.cp[0], self.cp[4]

        d_xy = p_end[:2] - p_start[:2]
        L = np.linalg.norm(d_xy)
        if L < 1e-6:
            return 1.0

        u_vec = d_xy / L
        s_robot = float(u_vec @ (x[:2] - p_start[:2]))
        s_target = s_robot + lookahead

        return max(self.sigma, float(np.clip(s_target / L, 0.0, 1.0)))

    # -- accessors --------------------------------------------------------
    @property
    def ref_final(self):
        """Current endpoint == cp4. None until built."""
        return None if self.cp is None else self.cp[4]

    def reset(self):
        self.cp = None
        self.sigma = 0.0


class SwingFootPlanner:

    def __init__(self, N, dt, R=0.35, lookahead=0.1):
        self.R = R
        self.lookahead = lookahead
        self.N = N
        self.delta = dt

        self.front = _SwingLeg(t1=0.2, t3=0.5)
        self.rear = _SwingLeg(t1=0.6, t3=0.6)

    # ----------------------------------------
    # Setup: called once at activation
    # ----------------------------------------
    def update_geometry(self, foot_pos_front, foot_pos_rear, contacts):
        pc1, pc2, pc3, pc4 = contacts
        g = self._terminal_geometry(contacts)

        # Mid points (leg-specific formulas; note p_mid_f uses the rear perp)
        # p_mid_f = pc2 + (pc3 - pc2) * 0.75 + self.R * g["d_perp_r"] / 2
        # p_mid_f[2] = pc2[2] + 0.4
        # p_mid_f = 0.8 * (g["ref_front"] + foot_pos_front.flatten())
        p_mid_f = foot_pos_front.flatten() + 0.5 * (g["ref_front"] - foot_pos_front.flatten())
        # p_mid_f[2] = foot_pos_front[2]

        p_mid_r = pc3 + g["d_rot"] * 0.5
        p_mid_r[2] += 0.3

        swing_start_f = foot_pos_front.flatten().copy()
        swing_start_r = foot_pos_rear.flatten().copy()

        self.front.build(swing_start_f, p_mid_f, g["ref_front"])
        self.rear.build(swing_start_r, p_mid_r, g["ref_rear"])

    # ----------------------------------------
    # Per-tick: refresh only the terminal points
    # ----------------------------------------
    def update_endpoints(self, contacts):
        g = self._terminal_geometry(contacts)
        self.front.rebuild_endpoint(g["ref_front"])
        self.rear.rebuild_endpoint(g["ref_rear"])

    # ----------------------------------------
    # Terminal geometry -- shared by setup and per-tick update
    # ----------------------------------------
    def _terminal_geometry(self, contacts):
        pc1, pc2, pc3, pc4 = contacts

        nf, _ = plane_normal(contacts[0:3])  # calf_f, pivot_f, pivot_r
        nr, _ = plane_normal(contacts[1:4])  # pivot_f, pivot_r, calf_r

        d_rot = pc3 - pc2
        d_rot = d_rot / np.linalg.norm(d_rot)

        d_perp_f = self._d_perpendicular(nf, d_rot, pc1, pc2)
        d_perp_r = self._d_perpendicular(nr, d_rot, pc4, pc3)

        return {
            "ref_front": pc2 + self.R * d_perp_f,
            "ref_rear": pc3 + self.R * d_perp_r,
            "d_rot": d_rot,
            "d_perp_f": d_perp_f,
            "d_perp_r": d_perp_r,
        }

    def _d_perpendicular(self, n, d_rot, p_ref, p_pivot):
        d_perp = np.cross(n, d_rot)
        d_perp = d_perp / np.linalg.norm(d_perp)
        if (np.dot(d_perp, p_ref - p_pivot) < 0):
            d_perp = -d_perp
        return d_perp

    # ----------------------------------------
    # Per-tick reference sampling -> (ref (N,3), terminal_error (3,))
    # ----------------------------------------
    def get_front_ref(self, foot_pos):
        return self.front.advance(foot_pos, self.N, self.delta, self.lookahead)

    def get_rear_ref(self, foot_pos):
        return self.rear.advance(foot_pos, self.N, self.delta, self.lookahead)

    # Kept for external curve plotting / debugging (same signature as before)
    @staticmethod
    def evaluate_at(t, cp):
        return _quartic_bezier(t, cp)

    def reset(self):
        self.front.reset()
        self.rear.reset()
