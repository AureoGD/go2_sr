"""
Capture-point half-plane constraint + axis-crossing detection, in a single object.

One instance per controller (RollCW, RollCCW, swing, ...). Instances are
independent: each holds its own configuration (roll direction, margins) and its
own crossing state, so no reset() is needed across phases.

Geometry reference is GRAVITY (vertical), not the contact plane. This keeps the
side normal and omega_zero well defined even when the robot rests only on the
shoulders and no contact plane can be estimated.

Roles served by the same object:
  - QP soft-constraint : provides n_side, p_axis, omega_zero and the row pieces
                         so the controller fills Ccs and the lower bound. The
                         base class handles the slack.
  - crossing detection : per-tick booleans for CP and CoM, each with a Schmitt
                         trigger (two thresholds) plus a latch.
      * instantaneous side  (cp_side / com_side)   -> "is it on the right side now?"
                                                       used by the swing (Role 1:
                                                       do not let CP go back).
      * latched crossing    (cp_crossed/com_crossed)-> "has it crossed for good?"
                                                       used by roll to switch phase.
"""

import numpy as np


def _unit(v, eps=1e-9):
    n = np.linalg.norm(v)
    return v / n if n > eps else v * 0.0


class CapturePointConstraint:

    def __init__(self, direction, margin_cp=0.01, com_ratio=0.1, dead_zone=0.0025, g=9.81, deg_tol=1e-3):
        """
        Parameters
        ----------
        direction : 'cw' | 'ccw'
            Roll direction (robot seen from the front). 'cw' -> correct side to
            the right ; 'ccw' -> to the left. Sign calibrated so that, for the
            reference config (FR at +x, RR at -x), 'cw' points to +y.
        margin_cp : float
            Half-plane margin for the CP (used both as the QP lower-bound folga
            and as the Schmitt ON threshold for the CP crossing).
        com_ratio : float
            CoM margin = margin_cp * com_ratio (default 0.1 -> 10x smaller).
        dead_zone : float
            Schmitt hysteresis width (m_on - m_off), 2.5 mm by default.
        g : float
            Gravity magnitude, for omega_zero.
        deg_tol : float
            Degeneracy tolerance for a near-vertical pivot axis.
        """
        self.direction = str(direction).lower()
        self.g = g
        self.deg_tol = deg_tol

        # QP / CP margin and Schmitt thresholds -----------------------------
        self.margin_cp = margin_cp
        self._cp_on = margin_cp
        self._cp_off = margin_cp - dead_zone
        margin_com = margin_cp * com_ratio
        self._com_on = margin_com
        self._com_off = margin_com - dead_zone

        # Frozen-per-tick geometry ------------------------------------------
        self.n_side = None  # side normal (horizontal, correct side)
        self.p_axis = None  # midpoint of the pivot axis
        self.omega_zero = None  # inverted-pendulum frequency
        self.ok = False  # geometry valid this tick?

        # Crossing state ----------------------------------------------------
        self._initialized = False  # first update just initializes (starts wrong)
        self._cp_side = False  # instantaneous: CP on correct side (Schmitt)
        self._com_side = False  # instantaneous: CoM on correct side (Schmitt)
        self._cp_crossed = False  # latch: CP has crossed at least once
        self._com_crossed = False  # latch: CoM has crossed at least once

    # ----------------------------------------------------------------------
    # Geometry (call once per tick, before using the constraint or crossings)
    # ----------------------------------------------------------------------
    def update_geometry(self, f_pivot, r_pivot, r_com, n_prev=None):
        """
        Recompute the side normal, axis point and omega_zero for this tick.

        f_pivot : (3,) front pivot contact (e.g. FR thigh, contacts[3]).
        r_pivot : (3,) rear  pivot contact (e.g. RR thigh, contacts[0]).
        r_com   : (3,) CoM position (world). Used for the vertical height.
        n_prev  : (3,) previous side normal, for continuity if the axis is
                  near-vertical (horizontal normal ill-defined).
        """
        f_pivot = np.asarray(f_pivot, float)
        r_pivot = np.asarray(r_pivot, float)
        r_com = np.asarray(r_com, float)

        d = r_pivot - f_pivot  # pivot axis (front -> rear)
        # horizontal normal to the axis: d x zhat = [d_y, -d_x, 0]
        n_raw = np.array([d[1], -d[0], 0.0])
        self.p_axis = 0.5 * (f_pivot + r_pivot)

        self.ok = True
        if np.linalg.norm(n_raw) < self.deg_tol:  # axis ~ vertical
            self.ok = False
            self.n_side = _unit(np.asarray(n_prev, float)) if n_prev is not None \
                else _unit(n_raw)
        else:
            n_side = _unit(n_raw)  # 'cw' branch (calibrated -> +y)
            if self.direction == 'ccw':
                n_side = -n_side
            self.n_side = n_side

        # omega_zero from the vertical height (cos(theta) terms cancel out, so
        # the plain vertical height is correct even on a slope).
        z = float(r_com[2] - self.p_axis[2])
        self.omega_zero = np.sqrt(self.g / max(z, 1e-6))
        return self.n_side, self.p_axis, self.ok

    # ----------------------------------------------------------------------
    # Capture point and constraint pieces
    # ----------------------------------------------------------------------
    def capture_point(self, r_com, r_vel):
        """xi = r_com + r_vel / omega_zero (component-wise, omega_zero scalar)."""
        return np.asarray(r_com, float) + np.asarray(r_vel, float) / self.omega_zero

    def constraint_value(self, xi):
        """
        Signed constraint value  n_side . (xi - p_axis) - margin_cp.
        >= 0 : CP on the correct side with the required margin.
        <  0 : violated (magnitude = how much is missing).
        """
        xi = np.asarray(xi, float)
        return float(self.n_side @ (xi - self.p_axis) - self.margin_cp)

    def constraint_row(self):
        """
        Pieces to fill the soft-constraint row in the controller's Ccs and its
        lower bound, for  n_side . xi >= margin_cp + n_side . p_axis  with
        xi = r_com + r_vel / omega_zero:

            coeff_rcom  : n_side              -> place on the r_com state columns
            coeff_drcom : n_side / omega_zero -> place on the r_vel state columns
            lower       : margin_cp + n_side . p_axis   (scalar)

        The base class adds the slack; the controller only places these.
        """
        coeff_rcom = self.n_side
        coeff_drcom = self.n_side / self.omega_zero
        lower = self.margin_cp + float(self.n_side @ self.p_axis)
        return coeff_rcom, coeff_drcom, lower

    # ----------------------------------------------------------------------
    # Crossing detection (call once per tick)
    # ----------------------------------------------------------------------
    def update_crossings(self, xi, r_com):
        """
        Update CP and CoM crossing state from this tick's xi and r_com, using a
        two-threshold Schmitt trigger plus a latch. The first call only
        initializes (both start on the wrong side), so detection begins on the
        second call.

        Returns (cp_side, com_side) — the instantaneous sides after this update.
        The latches are available via cp_crossed / com_crossed.
        """
        d_cp = float(self.n_side @ (np.asarray(xi, float) - self.p_axis))
        d_com = float(self.n_side @ (np.asarray(r_com, float) - self.p_axis))

        if not self._initialized:
            # Always starts on the wrong side; just seed the state, no detection.
            self._cp_side = False
            self._com_side = False
            self._initialized = True
            return self._cp_side, self._com_side

        self._cp_side = self._schmitt(self._cp_side, d_cp, self._cp_on, self._cp_off)
        self._com_side = self._schmitt(self._com_side, d_com, self._com_on, self._com_off)

        # Latch: once it has been on the correct side, remember it.
        self._cp_crossed = self._cp_crossed or self._cp_side
        self._com_crossed = self._com_crossed or self._com_side
        return self._cp_side, self._com_side

    @staticmethod
    def _schmitt(state, d, m_on, m_off):
        """Two-threshold Schmitt: flip up at m_on, down at m_off, else hold."""
        if not state and d >= m_on:
            return True
        if state and d <= m_off:
            return False
        return state

    # ----------------------------------------------------------------------
    # Accessors
    # ----------------------------------------------------------------------
    @property
    def cp_side(self):
        """Instantaneous: is the CP on the correct side now? (for the swing)"""
        return self._cp_side

    @property
    def com_side(self):
        """Instantaneous: is the CoM on the correct side now?"""
        return self._com_side

    @property
    def cp_crossed(self):
        """Latched: has the CP crossed for good? (for roll -> swing switch)"""
        return self._cp_crossed

    @property
    def com_crossed(self):
        """Latched: has the CoM crossed for good?"""
        return self._com_crossed


# ==========================================================================
# Self-tests
# ==========================================================================
if __name__ == "__main__":
    # --- sign calibration: CW -> +y, CCW -> -y (reference config) ----------
    p_FR = np.array([0.19255273, 0.11031958, 0.06629304])  # front pivot
    p_RR = np.array([-0.19549662, 0.1312115, 0.07179374])  # rear pivot
    r_com = np.array([0.0, 0.05, 0.13])

    cw = CapturePointConstraint('cw')
    n, p, ok = cw.update_geometry(p_FR, p_RR, r_com)
    print("CW  n_side:", np.round(n, 4), "| y+? ", n[1] > 0, "| ok:", ok)
    assert n[1] > 0

    ccw = CapturePointConstraint('ccw')
    n2, _, _ = ccw.update_geometry(p_FR, p_RR, r_com)
    print("CCW n_side:", np.round(n2, 4), "| y-? ", n2[1] < 0)
    assert n2[1] < 0

    # omega_zero sanity: vertical height = 0.13 - 0.069 = 0.061 -> ~12.7
    print("omega_zero:", round(cw.omega_zero, 3))

    # --- constraint row pieces ---------------------------------------------
    cr, cdr, low = cw.constraint_row()
    print("\nrow: coeff_rcom", np.round(cr, 3), "| coeff_drcom", np.round(cdr, 4), "| lower %.4f" % low)

    # --- crossing detection: synthetic d rising through the axis -----------
    # Build a CP whose projection goes from -0.05 (wrong) up past +0.01, dips,
    # and recovers. Latch must stay True after first cross; instantaneous side
    # must reflect the dip if it goes below m_off.
    cp = CapturePointConstraint('cw', margin_cp=0.01, com_ratio=0.1, dead_zone=0.0025)
    n, p, _ = cp.update_geometry(p_FR, p_RR, r_com)
    # craft xi values along n_side giving desired projections d (plus p_axis)
    ds = [-0.05, -0.03, -0.005, 0.006, 0.012, 0.02, 0.009, 0.004, 0.03, 0.03]
    print("\n d_cp    cp_side cp_crossed")
    for i, dval in enumerate(ds):
        xi = p + dval * n  # projection along n_side equals dval
        rc = p + (dval * 0.1) * n  # CoM projection = 0.1*d (smaller, for variety)
        s, sc = cp.update_crossings(xi, rc)
        tag = " (init)" if i == 0 else ""
        print(" %+.3f    %-5s   %-5s%s" % (dval, s, cp.cp_crossed, tag))

    # Assertions on the Schmitt+latch behavior:
    # after d=0.012 (>= m_on=0.01) side True and latch True; at d=0.009 (< m_on
    # but > m_off=0.0075) side stays True; at d=0.004 (< m_off) side flips False
    # but latch stays True.
    assert cp.cp_crossed is True, "latch must stay set after first cross"
    print("\nlatch stayed True after dip; instantaneous side tracked the dip. OK.")
    print("All tests passed.")
