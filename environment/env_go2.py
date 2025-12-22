import gymnasium as gym
import numpy as np
import math
from environment.go2_model import Go2ModelSimMuJoCo
from environment.go2_state_normalizer import Go2StateNormalizer


class Go2Env(gym.Env):

    def __init__(self, env_id=None, max_step=1500, rendering=False):
        super().__init__()
        self.env_id = env_id
        self.rendering = rendering

        # --- DYNAMIC TIME HORIZON CONFIG ---
        self.max_step_limit = max_step  # Absolute hard limit
        self.current_step_limit = 300  # Start short (3s)
        self.time_extension = 300  # Base time purchase (3s)

        self.robot_sim = Go2ModelSimMuJoCo(render=self.rendering)

        self.n_states = 63
        self.action_space = gym.spaces.Discrete(n=self.robot_sim.task_control.modes)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(self.n_states,), dtype=np.float32)

        self.scale_factor = 1.0
        self.ep_reward = 0
        self.current_step = 0
        self.ep = 0
        self.st = np.zeros((self.n_states,), dtype=np.float32)

        self.normalizer = Go2StateNormalizer(box_size=0.75)

        self.current_mode = 0
        self.last_mode = 0
        self.total_mode_changes = 0
        self.current_mode_success_tick = 0

        self.end_phase_count = [0] * len(self.robot_sim.robot_states.sr_mode_completed)

        # --- REWARDS ---
        self.end_task_weight = 60.0
        self.ori_weight = 0.02
        self.heigh_weight = 0.1

        self.MIN_DWELL_TICKS = 20

    def set_difficulty(self, value: float):
        self.scale_factor = float(value)

    def step(self, action):
        self.last_mode = self.current_mode
        self.current_mode = int(action)
        self.current_step += 1

        # 1. Physics
        self.robot_sim.control_loop(self.current_mode)

        # 2. Observation
        self._norm()

        # 3. Reward (Handles Time Extension)
        reward = self._reward()

        # 4. Termination Logic
        truncated = self.current_step >= self.current_step_limit
        too_poor_performance = self.ep_reward < -100

        mpc_crash = self.robot_sim.robot_states.critical_mpc_fail
        excessive_switching = self.total_mode_changes > 50
        is_standup = self.robot_sim.robot_states.sr_mode_completed.get(4, False)

        terminated = mpc_crash or excessive_switching or is_standup or too_poor_performance

        if is_standup:
            self.ep_reward += reward + 20
        else:
            self.ep_reward += reward

        info = {}
        if truncated and not terminated:
            info["TimeLimit.truncated"] = True
            info["terminal_observation"] = self.st.copy()

        return self.st, reward, terminated, truncated, info

    def reset(self, *, seed=None, q0=None, b0=None, r0=None):
        super().reset(seed=seed)

        if q0 is None:
            q0 = [-0.2, 2, -1.65, -0.6, 1.86, -1.65, -0.5, 1.06, -1.0, 0.25, 1.36, -1.05]
        if b0 is None:
            b0 = [0, 0, 0.085]
        if r0 is None:
            r0 = [np.pi, 0, 0]

        self.robot_sim.reset_robot_pose(q0=q0, b0=b0, r0=r0)

        # Reset Time Horizon
        self.current_step_limit = 300

        if self.robot_sim.task_control is not None:
            self.robot_sim.task_control.reset_controller()

        self.ep += 1
        ep_r = self.ep_reward
        self.current_step = 0
        self.ep_reward = 0
        self.total_mode_changes = 0
        self.current_mode_success_tick = 0
        self.end_phase_count[:] = [0] * len(self.end_phase_count)

        self.current_mode = 0
        self.last_mode = 0

        self.robot_sim.robot_states.critical_mpc_fail = False
        self.robot_sim.robot_states.subtask_succes = False
        self.robot_sim.robot_states.mpc_fail = False
        for mode_key in self.robot_sim.robot_states.sr_mode_completed.keys():
            self.robot_sim.robot_states.sr_mode_completed[mode_key] = False

        start_pos = self.robot_sim.robot_states.r_pos
        self.normalizer._norm_pos(start_pos.reshape(3))
        self.normalizer.reset_reference()
        self._norm()

        return self.st, {"Episode": self.ep, "Episode reward": ep_r}

    def _norm(self):
        self.st[0:3] = self.robot_sim.robot_states.r_pos.reshape(3)
        self.st[3:6] = self.robot_sim.robot_states.r_vel.reshape(3)
        self.st[6:10] = self.robot_sim.robot_states.epsilon.reshape(4)
        self.st[10:13] = self.robot_sim.robot_states.omega.reshape(3)
        self.st[13:25] = self.robot_sim.robot_states.q.reshape(12)
        self.st[25:37] = self.robot_sim.robot_states.dq.reshape(12)
        self.st[37:49] = self.robot_sim.robot_states.qr.reshape(12)
        self.st[49:61] = (self.robot_sim.robot_states.tau_pd + self.robot_sim.robot_states.tau_g).reshape(12)
        self.st[61] = float(self.current_mode if self.current_mode is not None else 0.0)
        self.st[62] = float(self.robot_sim.robot_states.mpc_fail)
        self.st = self.normalizer.normalize(self.st)

    def _reward(self):
        r = 0.0
        roll = self.robot_sim.robot_states.rpy[0]

        r -= self.ori_weight * (abs(roll) / np.pi)
        if abs(roll) > np.pi / 2 and self.current_mode in [4]:
            r -= 2
        if abs(roll) < np.pi * 30 / 180:
            r += self.ori_weight

        # Switching Logic
        if self.current_mode != self.last_mode:
            if self.current_mode_success_tick < self.MIN_DWELL_TICKS:
                r -= 2.0
            else:
                r -= 0.05
            self.current_mode_success_tick = 0
            self.total_mode_changes += 1

        if self.robot_sim.robot_states.r_pos[2] > 0.15 and abs(roll) < np.pi * 15 / 180:
            r += self.heigh_weight

        # --- TASK COMPLETION ---
        if bool(self.robot_sim.robot_states.subtask_succes) is True and \
           self.robot_sim.robot_states.sr_mode_completed.get(self.current_mode, False) is False:

            # Reset flags
            for mode_key in self.robot_sim.robot_states.sr_mode_completed.keys():
                self.robot_sim.robot_states.sr_mode_completed[mode_key] = False
            self.robot_sim.robot_states.sr_mode_completed[self.current_mode] = True

            # 1. Decay Reward
            count = self.end_phase_count[self.current_mode]
            r += self.end_task_weight / (2**count)

            # 2. Decay Time Extension (Prevent Infinite Farming)
            decayed_time = self.time_extension / (2**count)
            self.current_step_limit = min(self.current_step_limit + decayed_time, self.max_step_limit)

            # Increment counter
            self.end_phase_count[self.current_mode] += 1

        self.current_mode_success_tick += 1
        if self.current_mode_success_tick >= 200:
            r -= 0.1
        if self.robot_sim.robot_states.mpc_fail:
            r -= 1.0

        return float(r)

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
