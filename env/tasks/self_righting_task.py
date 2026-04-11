import numpy as np
from env.tasks.base_task import BaseTask
from collections import deque


class SelfRightingTask(BaseTask):

    def __init__(self, normalizer, tpe=None):
        super().__init__(normalizer=normalizer)

        self.tpe = tpe
        self.tpe_probs = None

        self.obs_dim = self.tpe.probs_dim + 49

        self.n_action_group = 6
        self.min_upright_height = 0.1

        self.success = False

        self.MIN_DWELL_TICKS = 100
        self.MAX_PHASE_COUNT = 15
        self.MAX_SWITCHES = 50

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

        self.map_control_index_acion_group = [0, 1, 2, 3, 4, 5, 6, 2, 3, 4, 5]
        self.BODY_PROGRESS_ACTION_GROUP = {3, 5, 6}
        self.JOINT_PROGRESS_ACTION_GROUP = {1, 2, 4}

    def get_obs_dim(self):
        return self.obs_dim

    def get_tpe_probs(self):
        return self.tpe_probs

    def set_step_limit(self, step_limit):
        self.step_limit = step_limit

    def compute_features(self, state):

        self.state_copy = state

        rs = self.state_copy.robot
        cs = self.state_copy.low_level
        ts = self.state_copy.task_state

        dir_v, v_abs = self.normalizer.normalize_velocity(rs.r_vel)

        dir_omega, omega_abs = self.normalizer.normalize_omega(rs.omega)

        alpha = self.normalizer.compute_alpha(rs.epsilon)

        q_norm = self.normalizer.normalize_q(rs.q)

        qr_norm = self.normalizer.normalize_q(cs.qr)

        dq_norm = self.normalizer.compute_dq_norm(rs.dq)

        dq_abs = self.normalizer.compute_dq_norm(rs.dq)

        tau = self.normalizer.normalize_tau(cs.tau)

        control_index_norm = np.array([ts.controller_index / self.normalizer.n_actions])

        action_group_norm = np.array([ts.action_group / self.n_action_group])

        controller_evolution = np.array([ts.controller_evolution])

        self._features = {
            "dir_v": dir_v,
            "v_abs": v_abs,
            "alpha": alpha,
            "dir_omega": dir_omega,
            "omega_abs": omega_abs,
            "q_norm": q_norm,
            "qr_norm": qr_norm,
            "dq_norm": dq_norm,
            "dq_abs": dq_abs,
            "tau": tau,
            "control_index_norm": control_index_norm,
            "action_group_norm": action_group_norm,
            "controller_evolution": controller_evolution,
        }

    def get_obs(self):

        f = self._features
        obs = np.concatenate([[f["alpha"]], f["dir_v"], [f["v_abs"]], f["dir_omega"], [f["omega_abs"]], f["q_norm"],
                              [f["dq_abs"]], f["qr_norm"], f["tau"], f["control_index_norm"], f["action_group_norm"],
                              f["controller_evolution"]])

        tpe_features = obs[:22]

        self.tpe.predict(tpe_features)

        if self.tpe.state.valid:
            probs = self.tpe.state.phase_probs
        else:
            probs = np.zeros(self.tpe.probs_dim)

        self.tpe_probs = probs

        obs = np.concatenate([obs, probs])

        return obs.astype(np.float32)

    def evaluate_reward(self):
        r = 0
        f = self._features
        state = self.state_copy
        rs = state.robot
        cs = state.low_level
        ts = state.task_state
        ps = state.tpe

        alpha = f["alpha"]
        current_controller_idx = ts.controller_index

        if current_controller_idx != self.last_controller_idx:
            self.last_controller_idx = current_controller_idx
            self.joint_stagnation_counter = 0
            self.stagnation_counter = 0
            self.current_controller_tick = 0
            self.total_controller_idx_changes += 1

            if current_controller_idx == 6:
                self.bz_initial = rs.b_pos[2].copy()

            self.hq.clear()
            self.hth.clear()
            self.hz.clear()

        if self.current_controller_tick == self.MIN_DWELL_TICKS:
            r += self.WEIGHT_MODE_HOLD

        r -= self.WEIGHT_ORIENTATION * (1.0 - alpha)

        if alpha > 0.8:
            r += self.WEIGHT_ORIENTATION * 8

        current_action_group = self.map_control_index_acion_group[current_controller_idx]

        # if alpha < 0 and current_action_group in [4, 5, 6]:
        #     r -= self.WEIGHT_BAD_ORIENTATION

        # if alpha > 0 and current_action_group in [1, 2]:
        #     r -= self.WEIGHT_BAD_ORIENTATION

        phase = ps.phase

        if phase == 0 and current_action_group not in [1, 2]:
            r -= self.WEIGHT_BAD_ORIENTATION

        if phase == 1 and current_action_group not in [3, 4]:
            r -= self.WEIGHT_BAD_ORIENTATION

        if phase == 2 and current_action_group not in [5, 6]:
            r -= self.WEIGHT_BAD_ORIENTATION

        if current_controller_idx == 0:
            r -= self.WEIGHT_IDLE

        self.current_controller_tick += 1

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
        cs = state.low_level
        ts = state.task_state

        if self.bz_initial is not None:
            z = rs.b_pos[2] - self.bz_initial
        else:
            z = 0

        alpha = f["alpha"]
        if alpha > 0.95 and z > 0.1:
            self.success = True

        too_many_switches = self.total_controller_idx_changes > self.MAX_SWITCHES

        terminated = (self.success or too_many_switches)

        truncated = current_step >= self.step_limit

        return terminated, truncated

    def reset(self):
        self.bz_initial = None
        self.last_controller_idx = -1
        self.joint_stagnation_counter = 0
        self.stagnation_counter = 0
        self.current_controller_tick = 0
        self.total_controller_idx_changes = 0
        self.success = False

        self.hq.clear()
        self.hth.clear()
        self.hz.clear()

        return super().reset()

    def compute_initial_obs(self, state):
        self.compute_features(state)
        return self.get_obs()

    def gen_info(self):
        info = {"sucess_flag": self.success, "sucess_extra_reward": 20}

        return info
