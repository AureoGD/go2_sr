import numpy as np

# ---------------------------------------------------------------------------
# Leg modes
# ---------------------------------------------------------------------------
# FOOT      : foot in full contact. Uses the foot linear Jacobian and foot
#             contact point. All three joints active. Contributes a CoM row
#             block (J_com - Jc_foot) and an Sa block.
# PIVOT     : shoulder pivot. Uses the THIGH linear Jacobian and thigh contact
#             point. Only the HIP joint is active; thigh + calf are frozen.
#             Contributes a CoM row block (J_com - Jc_thigh) and an Sa block.
# REFERENCE : leg driven by dq = Lambda (qr - q). Its own row is [0 .. I .. 0];
#             its columns are KEPT in the contact rows so its CoM back-reaction
#             is captured. Contributes to gamma_e_star.
# FROZEN    : dq = 0. Columns zeroed everywhere; own row is [0 .. I .. 0] with
#             zero forcing. (FROZEN == REFERENCE with Lambda = 0.)
# ---------------------------------------------------------------------------
FOOT = "FOOT"
PIVOT = "PIVOT"
REFERENCE = "REFERENCE"
FROZEN = "FROZEN"

# Column / row order of the joints. Leg i occupies columns [3*i : 3*i+3],
# and within each leg the order is (hip, thigh, calf).
LEGS = ["FR", "FL", "RR", "RL"]
LEG_INDEX = {leg: i for i, leg in enumerate(LEGS)}

# Which frame each contact mode pivots / stands on.
CONTACT_FRAME = {FOOT: "foot", PIVOT: "thigh"}


def skew_symmetric_matrix(vector):
    v1, v2, v3 = vector
    return np.array([[0.0, -v3, v2], [v3, 0.0, -v1], [-v2, v1, 0.0]])


class GammaBuilder:
    """Builds the reference-generation gain matrices for a single controller
    phase. The leg configuration (stance_config) is fixed at construction, so
    each controller (roll_cw, swing_cw, ...) owns its own GammaBuilder.

    build() returns, for the relation
        dq = gamma_l_star @ dr  -  gamma_a_star @ omega  +  gamma_e_star @ (qr - q)
    the three gain matrices plus the assembled active-contact Jacobian.

    Shapes:
        gamma_l_star : (12, 3)
        gamma_a_star : (12, 3)
        gamma_e_star : (12, 12)   # acts on full joint error (qr - q)
        contact_jacobians : (12, 12)
    """

    def __init__(self, pin_engine, stance_config, Kp_vec, Kd_vec, d):
        self.pin = pin_engine
        self.stance_config = dict(stance_config)
        self.Kp_vec = np.asarray(Kp_vec, dtype=float)
        self.Kd_vec = np.asarray(Kd_vec, dtype=float)
        self.d = d  # scalar for now; a length-12 vector will also broadcast

        self._validate_config()

        # Lambda = Kp / (Kd + d), per joint. Used only on REFERENCE legs.
        self.Lambda_vec = self.Kp_vec / (self.Kd_vec + self.d)

        # Pre-resolve per-leg column bookkeeping that never changes tick to tick.
        self._resolve_masks()

    # ------------------------------------------------------------------
    # init-time bookkeeping
    # ------------------------------------------------------------------
    def _validate_config(self):
        missing = [leg for leg in LEGS if leg not in self.stance_config]
        if missing:
            raise ValueError(f"stance_config missing legs: {missing}")
        for leg, mode in self.stance_config.items():
            if leg not in LEG_INDEX:
                raise ValueError(f"unknown leg '{leg}'")
            if mode not in (FOOT, PIVOT, REFERENCE, FROZEN):
                raise ValueError(f"leg '{leg}' has unknown mode '{mode}'")

    def _cols(self, leg):
        i = LEG_INDEX[leg]
        return slice(3 * i, 3 * i + 3)

    def _rows(self, leg):
        return self._cols(leg)  # row-block index == leg index

    def _resolve_masks(self):
        # Columns that are frozen (removed from the unknown set):
        #   PIVOT  -> thigh + calf of that leg
        #   FROZEN -> all three joints of that leg
        frozen_cols = []
        for leg, mode in self.stance_config.items():
            base = 3 * LEG_INDEX[leg]
            if mode == PIVOT:
                frozen_cols += [base + 1, base + 2]  # thigh, calf
            elif mode == FROZEN:
                frozen_cols += [base + 0, base + 1, base + 2]
        self.frozen_cols = np.array(sorted(frozen_cols), dtype=int)

        # Active joint indices (everything not frozen). Used for the rank guard.
        self.active_cols = np.array([c for c in range(12) if c not in set(self.frozen_cols)], dtype=int)

        self.contact_legs = [lg for lg, m in self.stance_config.items() if m in (FOOT, PIVOT)]
        self.reference_legs = [lg for lg, m in self.stance_config.items() if m == REFERENCE]

    # ------------------------------------------------------------------
    # per-tick build
    # ------------------------------------------------------------------
    def build(self, r, use_gamma_e=True):
        J_com = self.pin.com_jacobian()  # (3, 12)

        gamma = np.vstack([J_com, J_com, J_com, J_com])  # (12, 12)

        # Forcing selectors:
        I_sel = np.zeros((12, 3))  # coefficient of dr   (M in the derivation)
        Sa = np.zeros((12, 3))  # coefficient of omega
        E = np.zeros((12, 12))  # coefficient of (qr - q)
        contact_jacobians = np.zeros((12, 12))

        for leg in LEGS:
            mode = self.stance_config[leg]
            rows = self._rows(leg)
            cols = self._cols(leg)

            if mode in (FOOT, PIVOT):
                frame = CONTACT_FRAME[mode]
                Jc = self.pin.linear_leg_jacobian(leg, frame)  # (3, 3)
                pc = self.pin.frame_pos(leg, frame)  # (3,)
                # pc[1]-=3

                # Contact CoM equation:  (J_com - Jc) dq = M dr - Sa omega
                gamma[rows, cols] -= Jc
                I_sel[rows, :] = np.eye(3)
                Sa[rows, :] = skew_symmetric_matrix(pc - r)

                # Active-contact Jacobian block (returned for downstream use).
                contact_jacobians[rows, cols] = Jc

            else:  # REFERENCE or FROZEN -> own row is [0 .. I .. 0]
                gamma[rows, :] = 0.0
                gamma[rows, cols] = np.eye(3)
                if mode == REFERENCE:
                    # dq_leg = Lambda (qr - q)_leg.  Sign is +Lambda: this is the
                    # convention that made the FL estimate track (corr +1.00).
                    E[rows, cols] = np.diag(self.Lambda_vec[cols])
                # FROZEN: forcing stays 0 -> dq_leg = 0.

        # Freeze columns: PIVOT thigh/calf and whole FROZEN legs are removed
        # from the unknowns by zeroing their columns everywhere. (REFERENCE
        # columns are intentionally kept so the CoM back-reaction survives.)
        if self.frozen_cols.size:
            gamma[:, self.frozen_cols] = 0.0

        gamma_inv = self._safe_pinv(gamma)

        gamma_l_star = gamma_inv @ I_sel  # (12, 3)
        gamma_a_star = gamma_inv @ Sa  # (12, 3)
        if use_gamma_e:
            gamma_e_star = gamma_inv @ E  # (12, 12)
        else:
            gamma_e_star = np.zeros((12, 12))

        return gamma_l_star, gamma_a_star, gamma_e_star, Sa, contact_jacobians

    # ------------------------------------------------------------------
    # pseudo-inverse with a rank guard
    # ------------------------------------------------------------------
    def _safe_pinv(self, gamma, rcond=1e-6):
        # The system should have rank == number of active joints. A drop below
        # that means a contact went (near-)degenerate this tick -- flag it
        # rather than let the pinv silently invent a huge dq.
        s = np.linalg.svd(gamma, compute_uv=False)
        tol = rcond * s[0] if s[0] > 0 else 0.0
        rank = int(np.sum(s > tol))
        if rank != self.active_cols.size:
            raise np.linalg.LinAlgError(f"gamma rank {rank} != expected {self.active_cols.size} "
                                        f"(smallest sv {s[-1]:.2e}); a contact may be degenerate.")
        return np.linalg.pinv(gamma, rcond=rcond)


# ---------------------------------------------------------------------------
# Named phase configurations. A controller picks the one for its phase.
# ---------------------------------------------------------------------------
CONFIGS = {
    "stand_up": {
        "FR": FOOT,
        "FL": FOOT,
        "RR": FOOT,
        "RL": FOOT
    },
    "roll_cw": {
        "FR": PIVOT,
        "FL": FROZEN,
        "RR": PIVOT,
        "RL": FOOT
    },
    "roll_ccw": {
        "FR": FROZEN,
        "FL": PIVOT,
        "RR": FOOT,
        "RL": PIVOT,
    },
    "swing_cw": {
        "FR": PIVOT,
        "FL": REFERENCE,
        "RR": PIVOT,
        "RL": REFERENCE,

    },
    "swing_ccw": {
        "FR": REFERENCE,
        "FL": PIVOT,
        "RR": REFERENCE,
        "RL": PIVOT,
    },
    "settle_cw": {
        "FR": PIVOT,
        "FL": FOOT,
        "RR": PIVOT,
        "RL": FOOT
    },
}