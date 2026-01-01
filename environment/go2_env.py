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
        self.action_space = gym.spaces.Discrete(n=int(self.robot_sim.task_control.modes))
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(self.n_states,), dtype=np.float32)

        self.scale_factor = 1.0
        self.ep_reward = 0.0
        self.current_step = 0
        self.ep = 0
        self.st = np.zeros((self.n_states,), dtype=np.float32)

        self.normalizer = Go2StateNormalizer(box_size=0.75)

        self.current_mode = 0
        self.last_mode = 0
        self.total_mode_changes = 0
        self.current_mode_tick = 0

        self.end_phase_count = [0] * len(self.robot_sim.robot_states.sr_mode_completed)

        self.end_task_weight = 60.0
        self.ori_weight = 0.02
        self.heigh_weight = 0.1

        self.MIN_DWELL_TICKS = 20
        self.MAX_PHASE_COUNT = 15

        self.PROGRESS_MODES = {3, 5, 6}
        self.JOINT_PROGRESS_MODES = {1, 2, 4}
        self.EPS_Q = 1e-3
        self.EPS_Z = 1e-3
        self.EPS_TH = 5e-4

        self.min_upright_height = 0.2

        self.stagnation_counter = 0
        self.joint_stagnation_counter = 0

        self.WINDOW = 20
        self.hq = deque(maxlen=self.WINDOW)  # history of the joint pos
        self.hz = deque(maxlen=self.WINDOW)  # history of the height
        self.hth = deque(maxlen=self.WINDOW)  # history of the orietation

    def set_difficulty(self, value: float):
        self.scale_factor = float(value)

    def step(self, action):
        self.last_mode = self.current_mode
        self.current_mode = int(action)
        self.current_step += 1

        self.robot_sim.control_loop(self.current_mode)

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

    def reset(self, *, seed=None, q0=None, b0=None, r0=None):
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
        self.total_mode_changes = 0
        self.current_mode_tick = 0
        self.end_phase_count[:] = [0] * len(self.end_phase_count)

        self.current_mode = 0
        self.last_mode = 0

        rs = self.robot_sim.robot_states
        rs.critical_mpc_fail = False
        rs.subtask_succes = False
        rs.mpc_fail = False
        rs.sr_mode_completed[:] = [False] * len(rs.sr_mode_completed)
        rs.current_sucess_mode = 0.0

        start_pos = rs.r_pos
        self.normalizer._norm_pos(start_pos.reshape(3))
        self.normalizer.reset_reference()
        self._norm()

        self.hq.clear()
        self.hth.clear()
        self.hz.clear()

        self.stagnation_counter = 0
        self.joint_stagnation_counter = 0
        self.standup_end_flag = False

        return self.st, {"Episode": self.ep, "Episode reward": ep_r}

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
        self.st[61] = float(self.current_mode)
        self.st[62] = float(rs.mpc_fail)
        self.st[63] = float(rs.current_sucess_mode)
        self.st = self.normalizer.normalize(self.st)

    def _reward(self):
        r = 0.0

        rs = self.robot_sim.robot_states
        roll = rs.rpy[0]
        abs_roll = abs(roll)
        z = rs.r_pos[2]

        r -= self.ori_weight * (abs_roll / np.pi)

        if abs_roll < np.pi * 30 / 180:
            r += self.ori_weight

        if abs_roll > np.pi / 2 and self.current_mode in [5, 6, 4]:
            r -= 1.0

        if self.current_mode == 0:
            r -= 0.1

        if self.current_mode != self.last_mode:
            if self.current_mode_tick < self.MIN_DWELL_TICKS:
                r -= 5.0
            else:
                r -= 0.05
            self.joint_stagnation_counter = 0
            self.stagnation_counter = 0
            self.current_mode_tick = 0
            self.total_mode_changes += 1
            self.hq.clear()
            self.hth.clear()
            self.hz.clear()

        if rs.subtask_succes and not rs.sr_mode_completed[self.current_mode]:
            rs.sr_mode_completed[:] = [False] * len(rs.sr_mode_completed)
            rs.sr_mode_completed[self.current_mode] = True
            count = min(self.end_phase_count[self.current_mode], self.MAX_PHASE_COUNT)

            r += self.end_task_weight / (2**count)

            decayed_time = self.time_extension / (2**count)
            self.current_step_limit = min(self.current_step_limit + decayed_time, self.max_step_limit)

            self.end_phase_count[self.current_mode] += 1

            if self.current_mode == 6:
                self.standup_end_flag = True

        idx = np.where(rs.sr_mode_completed)[0]
        rs.current_sucess_mode = float(idx[0]) if idx.size > 0 else 0

        self.hq.append(rs.q.copy().ravel())
        self.hz.append(float(rs.r_pos[2]))
        self.hth.append(float(abs_roll))

        if len(self.hq) >= self.WINDOW:
            dq_diff = np.diff(np.array(self.hq), axis=0)
            dq = np.abs(dq_diff).mean() if dq_diff.size > 0 else 0.0

            hz_diff = np.diff(np.array(self.hz))
            dz = abs(hz_diff[-1]) if hz_diff.size > 0 else 0.0

            hth_diff = np.diff(np.array(self.hth))
            dth = abs(hth_diff[-1]) if hth_diff.size > 0 else 0.0

            if self.current_mode in self.PROGRESS_MODES:
                if dth < self.EPS_TH and dz < self.EPS_Z:
                    self.stagnation_counter += 1
                else:
                    self.stagnation_counter = 0
            else:
                self.stagnation_counter = 0

            if self.current_mode in self.JOINT_PROGRESS_MODES:
                if dq < self.EPS_Q:
                    self.joint_stagnation_counter += 1
                else:
                    self.joint_stagnation_counter = 0
            else:
                self.joint_stagnation_counter = 0

        if self.joint_stagnation_counter > 50:
            r -= 0.002 * self.joint_stagnation_counter

        if rs.mpc_fail:
            r -= 1.0

        self.current_mode_tick += 1

        return float(r)

    def _termination(self):
        rs = self.robot_sim.robot_states

        roll = abs(rs.rpy[0])
        pitch = abs(rs.rpy[1])
        z = rs.r_pos[2]

        physically_upright = (roll < np.pi * 15 / 180 and pitch < np.pi * 15 / 180 and z > self.min_upright_height)

        REQUIRED_PHASES = [1, 2, 3, 4, 5]
        sequence_completed = all(self.end_phase_count[p] > 0 for p in REQUIRED_PHASES)

        stand_completed = self.standup_end_flag

        success = physically_upright and stand_completed and sequence_completed

        mpc_crash = rs.critical_mpc_fail
        too_many_switches = self.total_mode_changes > 50
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
