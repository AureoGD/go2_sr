import numpy as np
import json
from sim.utils.orientation import compute_alpha


class StateNormalizer:

    def __init__(self,
                 joint_limits,
                 torque_limits,
                 stats_path="tpe/data/processed/normalization_stats.json",
                 box_size=0.5):

        # ======================================================
        # LOAD NORMALIZATION STATS (TPE)
        # ======================================================
        with open(stats_path, "r") as f:
            stats = json.load(f)

        self.v_scale = stats.get("v_norm", 1.0)
        self.omega_scale = stats.get("omega_norm", 1.0)

        # ======================================================
        # ROBOT LIMITS (FROM SIM)
        # ======================================================
        self.joint_limits = joint_limits
        self.torque_limits = torque_limits

        # ======================================================
        # POSITION NORMALIZATION (LOCAL FRAME)
        # ======================================================
        self.box_size = box_size
        self.current_center = None
        self.is_out_of_box = False
        self.out_of_box_distance = 0.0

        # ======================================================
        # DISCRETE NORMALIZATION
        # ======================================================
        self.num_modes = 7
        self.num_states = 6

    def get_obs_dim(self):
        return 37

    # ======================================================
    # MAIN ENCODE
    # ======================================================
    def encode(self, state):

        rs = state.robot
        cs = state.controller

        # --------------------------------------
        # POSITION (local box)
        # --------------------------------------
        pos = self._norm_pos(rs.b_pos)

        # --------------------------------------
        # ORIENTATION (gravity alignment)
        # --------------------------------------
        alpha = compute_alpha(rs.epsilon)

        # --------------------------------------
        # VELOCITY (CoM)
        # --------------------------------------
        v = rs.r_vel
        v_norm = np.linalg.norm(v)

        dir_v = v / (v_norm + 1e-8)
        v_abs = np.tanh(v_norm / (self.v_scale + 1e-8))

        # --------------------------------------
        # ANGULAR VELOCITY
        # --------------------------------------
        wx = np.tanh(rs.omega[0] / (self.omega_scale + 1e-8))

        # --------------------------------------
        # JOINT POSITIONS
        # --------------------------------------
        q_norm = self._norm_limits(rs.q, self.joint_limits)

        # --------------------------------------
        # TORQUE (REAL APPLIED)
        # --------------------------------------
        tau = cs.tau / (self.torque_limits + 1e-8)

        # --------------------------------------
        # FLAGS / LOGIC
        # --------------------------------------
        mode = getattr(rs, "mode", 0) / self.num_modes
        mpc_fail = float(cs.mpc_fail)

        current_state = getattr(rs, "current_state", 0) / self.num_states
        success_flag = float(getattr(rs, "subtask_succes", 0))

        # --------------------------------------
        # FINAL OBS VECTOR
        # --------------------------------------
        obs = np.concatenate([
            [alpha],  # 1
            pos,  # 3
            dir_v,  # 3
            [v_abs],  # 1
            [wx],  # 1
            q_norm,  # 12
            tau,  # 12
            [mode],  # 1
            [mpc_fail],  # 1
            [current_state],  # 1
            [success_flag],  # 1
        ])

        return obs.astype(np.float32)

    # ======================================================
    # POSITION NORMALIZATION
    # ======================================================
    def _norm_pos(self, pos):

        if self.current_center is None:
            self.current_center = pos.copy()

        relative = (pos - self.current_center) / (self.box_size / 2)

        self.is_out_of_box = np.any(np.abs(relative) > 1.0)

        clipped = np.clip(relative, -1.0, 1.0)

        self.out_of_box_distance = np.linalg.norm(relative - clipped)

        return clipped

    # ======================================================
    # NORMALIZE WITH LIMITS [-1, 1]
    # ======================================================
    def _norm_limits(self, val, limits):

        return np.clip(2.0 * (val - limits[:, 0]) / (limits[:, 1] - limits[:, 0] + 1e-8) - 1.0, -1.0, 1.0)

    # ======================================================
    # RESET (CALL ON ENV RESET)
    # ======================================================
    def reset_reference(self):

        self.current_center = None
        self.is_out_of_box = False
        self.out_of_box_distance = 0.0
