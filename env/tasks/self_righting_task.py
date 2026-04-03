import numpy as np
from env.tasks.base_task import BaseTask
from collections import deque


class SelfRightingTask(BaseTask):

    def __init__(self, normalizer, tpe=None):
        super().__init__(normalizer=normalizer)

        self.tpe = tpe
        self.tpe_probs = None

        self.obs_dim = self.tpe.probs_dim + 43

        self.n_sr_semantics = 6
        self.min_upright_height = 0.1

        self.MIN_DWELL_TICKS = 100
        self.MAX_PHASE_COUNT = 15

        self.WEIGHT_ORIENTATION = 0.01
        self.WEIGHT_MODE_HOLD = 2.5
        self.WEIGHT_END_TASK = 10.0
        self.WEIGHT_STAGNATION = 0.002
        self.WEIGHT_STAGNATION = 0.0005
        self.WEIGHT_MPC_FAIL = 1.0
        self.WEIGHT_BAD_ORIENTATION = 1
        self.WEIGHT_IDLE = 0.1

        self.WINDOW = 20
        self.hq = deque(maxlen=self.WINDOW)
        self.hz = deque(maxlen=self.WINDOW)
        self.hth = deque(maxlen=self.WINDOW)

    def get_obs_dim(self):
        return self.obs_dim

    def get_tpe_probs(self):
        return self.tpe_probs

    def set_step_limit(self, step_limit):
        self.step_limit = step_limit

    def compute_features(self, state):

        self.state_copy = state

        rs = self.state_copy.robot
        cs = self.state_copy.controller

        dir_v, v_abs = self.normalizer.normalize_velocity(rs.r_vel)

        omega = self.normalizer.normalize_omega(rs.omega)

        alpha = self.normalizer.compute_alpha(state)

        q_norm = self.normalizer.normalize_q(rs.q)

        qr_norm = self.normalizer.normalize_q(rs.qr)

        dq_norm = self.normalizer.compute_dq_norm(rs.dq)

        dq_abs = self.normalizer.compute_dq_norm(rs.dq)

        tau = self.normalizer.normalize_tau(cs.tau)

        controll_index_norm = np.array([cs.controller_index / self.normalizer.n_actions])

        n_sr_semantics_norm = np.array([cs.sr_semantics / self.n_sr_semantics])

        controller_evolution = np.array([cs.controller_evolution])

        self._features = {
            "dir_v": dir_v,
            "v_abs": v_abs,
            "alpha": alpha,
            "omega": omega,
            "q": q_norm,
            "qr_norm": qr_norm,
            "dq_norm": dq_norm,
            "dq_abs": dq_abs,
            "tau": tau,
            "controll_index_norm": controll_index_norm,
            "n_sr_semantics_norm": n_sr_semantics_norm,
            "controller_evolution": controller_evolution,
        }

    def get_obs(self):

        f = self._features
        obs = np.concatenate([[f["alpha"]], f["dir_v"], [f["v_abs"]], [f["omega"][0]], [f["dq_abs"]], f["q"],
                              f["qr_norm"], f["tau"], f["controll_index_norm"], f["n_sr_semantics_norm"],
                              f["controller_evolution"]])

        tpe_features = obs[:19]

        probs = self.tpe.predict(tpe_features)

        if probs is None:
            probs = np.zeros(self.tpe.probs_dim)

        self.tpe_probs = probs

        obs = np.concatenate([obs, probs])

        return obs.astype(np.float32)

    def evaluate_reward(self):
        r = 0
        f = self._features
        state = self.state_copy
        rs = state.robot
        cs = state.controller

        alpha = f["alpha"]
        current_controller_idx = cs.controller_index

        if current_controller_idx != self.last_controller_idx:
            self.joint_stagnation_counter = 0
            self.stagnation_counter = 0
            self.current_controller_tick = 0
            self.total_controller_idx_changes += 1

            # ✔ keep exactly
            if current_controller_idx == 6:
                self.bz_initial = rs.b_pos[2].copy()

            self.hq.clear()
            self.hth.clear()
            self.hz.clear()

        if self.current_controller_tick == self.MIN_DWELL_TICKS:
            r += self.WEIGHT_MODE_HOLD

        r -= self.WEIGHT_ORIENTATION * (1.0 - alpha)

        self.current_controller_tick += 1

        if current_controller_idx == 0:
            r -= self.WEIGHT_IDLE

        return float(r)

    # ======================================================
    # TERMINATION
    # ======================================================
    def check_termination(self, current_step):

        terminated = False
        truncated = False

        f = self._features
        state = self.state_copy
        rs = state.robot
        cs = state.controller

        alpha = f["alpha"]

        truncated = current_step >= self.step_limit

        return terminated, truncated

    def reset(self):
        self.bz_initial = 0
        self.last_controller_idx = -1
        self.joint_stagnation_counter = 0
        self.stagnation_counter = 0
        self.current_controller_tick = 0
        self.total_controller_idx_changes = 0

        self.hq.clear()
        self.hth.clear()
        self.hz.clear()

        return super().reset()

    def compute_initial_obs(self, state):
        self.compute_features(state)
        return self.get_obs()
