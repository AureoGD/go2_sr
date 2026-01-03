import gymnasium as gym
import numpy as np
import psutil
import os
from environment.go2_sim import Go2ModelSimMuJoCo
from environment.normalizer import Go2StateNormalizer
from collections import deque


class Go2Env(gym.Env):

    def __init__(self, env_id=None, max_step=1500, rendering=False, **kwargs):
        super().__init__()
        self.env_id = env_id
        self.rendering = rendering

        self.internal_reset_count = 0
        self.pid = os.getpid()
        self.process = psutil.Process(self.pid)

        self.max_step_limit = max_step
        self.current_step_limit = 300
        self.time_extension = 350

        strategy_name = kwargs.get("strategy", "rgc")
        self.robot_sim = Go2ModelSimMuJoCo(render=self.rendering, strategy_name=strategy_name, **kwargs)

        self.n_states = 64
        self.action_space = gym.spaces.Discrete(int(self.robot_sim.task_control.n_controllers))
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(self.n_states,),
            dtype=np.float32,
        )

        self.scale_factor = 1.0
        self.ep_reward = 0.0
        self.current_step = 0
        self.ep = 0
        self.st = np.zeros((self.n_states,), dtype=np.float32)

        self.normalizer = Go2StateNormalizer(box_size=0.75)

        self.current_controller_idx = 0
        self.last_controller_idx = 0
        self.total_controller_idx_changes = 0
        self.current_controller_tick = 0

        _, self.control_index_to_level = (self.robot_sim.task_control.get_phase_mapping())

        # ✔ correct
        self.end_phase_count = [0] * len(self.control_index_to_level)
        # =====================================================
        # Stagnation metrics
        # =====================================================

        self.MIN_DWELL_TICKS = 20
        self.MAX_PHASE_COUNT = 15

        self.BODY_PROGRESS_MODES = {3, 5, 6}
        self.JOINT_PROGRESS_MODES = {1, 2, 4}
        self.EPS_Q = 1e-3
        self.EPS_Z = 1e-3
        self.EPS_TH = 5e-4

        # =====================================================
        # Reward weights (naming consistency)
        # =====================================================
        self.WEIGHT_ORIENTATION = 0.01
        self.WEIGHT_MODE_HOLD = 2.5
        self.WEIGHT_END_TASK = 5.0
        self.WEIGHT_STAGNATION = 0.002
        self.WEIGHT_MPC_FAIL = 1.0
        self.WEIGHT_BAD_ORIENTATION = 0.01
        self.WEIGHT_IDLE = 0.1

        self.min_upright_height = 0.1
        self.stagnation_counter = 0
        self.joint_stagnation_counter = 0

        self.WINDOW = 20
        self.hq = deque(maxlen=self.WINDOW)
        self.hz = deque(maxlen=self.WINDOW)
        self.hth = deque(maxlen=self.WINDOW)

    def set_difficulty(self, value: float):
        self.scale_factor = float(value)

    def step(self, action):
        self.last_controller_idx = self.current_controller_idx
        self.current_controller_idx = int(action)
        self.current_step += 1

        self.robot_sim.control_loop(self.current_controller_idx)
        self._norm()

        reward = self._reward()
        self.ep_reward += reward

        terminated, truncated, success, ext_penalty = self._termination()
        info = {}

        if success:
            reward += 20.0
            self.ep_reward += 20.0
            info["is_success"] = True

        if terminated:
            reward -= ext_penalty
            self.ep_reward -= ext_penalty

        if truncated and not terminated:
            info["TimeLimit.truncated"] = True
            info["terminal_observation"] = self.st.copy()

        return self.st, reward, terminated, truncated, info

    def _norm(self):
        rs = self.robot_sim.robot_states
        self.st[0:3] = rs.r_pos.reshape(3)
        self.st[3:6] = rs.r_vel.reshape(3)
        self.st[6:10] = rs.epsilon.reshape(4)
        self.st[10:13] = rs.omega.reshape(3)
        self.st[13:25] = rs.q.reshape(12)
        self.st[25:37] = rs.dq.reshape(12)
        self.st[37:49] = rs.qr.reshape(12)
        self.st[49:61] = (rs.tau_pd + rs.tau_g).reshape(12)
        self.st[61] = float(self.current_controller_idx)
        self.st[62] = float(rs.mpc_fail)
        self.st[63] = float(rs.current_sucess_mode)
        self.st = self.normalizer.normalize(self.st)

    def reset(self, *, seed=None, q0=None, b0=None, r0=None, mode=None):
        super().reset(seed=seed)

        self.internal_reset_count += 1

        if q0 is None:
            q0 = [-0.2, 2, -1.65, -0.6, 1.86, -1.65, -0.5, 1.06, -1.0, 0.25, 1.36, -1.05]
        if b0 is None:
            b0 = [0, 0, 0.085]
        if r0 is None:
            r0 = [np.pi, 0, 0]

        self.robot_sim.reset_robot_pose(q0=q0, b0=b0, r0=r0)

        self.current_step_limit = 300

        if self.robot_sim.task_control is not None:
            self.robot_sim.task_control.reset_phase()

        self.ep += 1
        ep_r = self.ep_reward

        self.current_step = 0
        self.ep_reward = 0.0
        self.total_controller_idx_changes = 0
        self.current_controller_tick = 0
        self.stagnation_counter = 0
        self.joint_stagnation_counter = 0
        self.end_phase_count[:] = [0] * len(self.end_phase_count)
        self.current_controller_idx = 0
        self.last_controller_idx = 0
        self.hq.clear()
        self.hth.clear()
        self.hz.clear()
        self.bz_initial = 0

        # ✔ critical reset
        self.last_ended_controller_idx_mode = -1

        rs = self.robot_sim.robot_states
        rs.critical_mpc_fail = False
        rs.subtask_succes = False
        rs.mpc_fail = False

        start_pos = rs.r_pos
        self.normalizer._norm_pos(start_pos.reshape(3))
        self.normalizer.reset_reference()

        if mode is not None:
            self.var_mode_conf(mode=mode)

        self._norm()
        return self.st, {"Episode": self.ep, "Episode reward": ep_r}

    def _reward(self):
        r = 0.0
        rs = self.robot_sim.robot_states

        if self.current_controller_idx != self.last_controller_idx:
            self.joint_stagnation_counter = 0
            self.stagnation_counter = 0
            self.current_controller_tick = 0
            self.total_controller_idx_changes += 1

            # ✔ keep exactly
            if self.current_controller_idx == 6:
                self.bz_initial = rs.b_pos[2].copy()

            self.hq.clear()
            self.hth.clear()
            self.hz.clear()

        if self.current_controller_tick == self.MIN_DWELL_TICKS:
            r += self.WEIGHT_MODE_HOLD

        roll = rs.rpy[0]
        abs_roll = abs(roll)

        r -= self.WEIGHT_ORIENTATION * (abs_roll / np.pi)

        if abs_roll < np.pi * 30 / 180:
            r += self.WEIGHT_ORIENTATION

        if abs_roll > np.pi / 2 and self.current_controller_idx in [4, 5, 6]:
            r -= self.WEIGHT_BAD_ORIENTATION

        if self.current_controller_idx == 0:
            r -= self.WEIGHT_IDLE

        if rs.subtask_succes and self.last_ended_controller_idx_mode != self.current_controller_idx:
            self.last_ended_controller_idx_mode = self.current_controller_idx
            level = self.control_index_to_level[self.current_controller_idx]

            for i in range(level + 1, len(rs.sr_mode_completed)):
                rs.sr_mode_completed[i] = False

            rs.sr_mode_completed[level] = True

            count = min(self.end_phase_count[self.current_controller_idx], self.MAX_PHASE_COUNT)
            r += self.WEIGHT_END_TASK / (2**count)

            decayed_time = self.time_extension / (2**count)
            self.current_step_limit = min(self.current_step_limit + decayed_time, self.max_step_limit)

            self.end_phase_count[self.current_controller_idx] += 1

            idx = np.where(rs.sr_mode_completed)[0]
            valid = idx[idx > 0]
            rs.current_sucess_mode = float(valid[-1]) if valid.size > 0 else 0.0

        stagnation_metric = max(self.joint_stagnation_counter, self.stagnation_counter)
        if stagnation_metric > 50:
            r -= self.WEIGHT_STAGNATION * stagnation_metric

        if rs.mpc_fail:
            r -= self.WEIGHT_MPC_FAIL

        self.hq.append(rs.q.copy().ravel())
        self.hz.append(float(rs.r_pos[2]))
        self.hth.append(float(abs_roll))

        self.current_controller_tick += 1
        return float(r)

    def _termination(self):
        rs = self.robot_sim.robot_states

        roll = abs(rs.rpy[0])
        pitch = abs(rs.rpy[1])
        z = abs(self.bz_initial - rs.b_pos[2])

        physically_upright = (roll < np.pi * 15 / 180 and pitch < np.pi * 15 / 180 and z > self.min_upright_height)

        sequence_completed = all(rs.sr_mode_completed[1:])

        success = physically_upright and sequence_completed

        mpc_crash = rs.critical_mpc_fail
        too_many_switches = self.total_controller_idx_changes > 50
        stagnated = self.stagnation_counter > 300
        joint_stagnated = self.joint_stagnation_counter > 300

        terminated = (success or mpc_crash or stagnated or too_many_switches or joint_stagnated)

        truncated = self.current_step >= self.current_step_limit

        ext_penalty = 10 if mpc_crash else 0

        return terminated, truncated, success, ext_penalty

    def save_normalizer(self, filepath):
        try:
            self.normalizer.save_weights(filepath)
            return True
        except Exception:
            return False

    def load_normalizer(self, filepath):
        try:
            self.normalizer.load_weights(filepath)
            return True
        except Exception:
            return False

    def var_mode_conf(self, mode):
        rs = self.robot_sim.robot_states

        rs.sr_mode_completed[:] = [False] * len(rs.sr_mode_completed)

        for lvl in range(1, mode + 1):
            rs.sr_mode_completed[lvl] = True

        rs.current_success_mode = float(mode)
