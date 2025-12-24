import gymnasium as gym
import numpy as np
import psutil
import os
from environment.go2_sim import Go2ModelSimMuJoCo
from environment.normalizer import Go2StateNormalizer

# TODO: clamp long-horizon counters to avoid rare int->float overflow


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
        self.time_extension = 300

        strategy_name = kwargs.get('strategy', 'rgc')
        self.robot_sim = Go2ModelSimMuJoCo(render=self.rendering, strategy_name=strategy_name, **kwargs)

        self.n_states = 63
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
        self.current_mode_success_tick = 0

        self.end_phase_count = [0] * len(self.robot_sim.robot_states.sr_mode_completed)

        self.end_task_weight = 60.0
        self.ori_weight = 0.02
        self.heigh_weight = 0.1

        self.MIN_DWELL_TICKS = 20

        self.PROGRESS_MODES = {3, 4, 5, 7, 8}

        self.min_upright_height = 0.15

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

        terminated, truncated, success = self._termination()

        if success:
            reward += 20.0
            self.ep_reward += 20.0

        info = {}
        if success:
            info["is_success"] = True

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
        self.current_mode_success_tick = 0
        self.end_phase_count[:] = [0] * len(self.end_phase_count)

        self.current_mode = 0
        self.last_mode = 0

        self.robot_sim.robot_states.critical_mpc_fail = False
        self.robot_sim.robot_states.subtask_succes = False
        self.robot_sim.robot_states.mpc_fail = False
        self.robot_sim.robot_states.sr_mode_completed[:] = \
            [False] * len(self.robot_sim.robot_states.sr_mode_completed)

        start_pos = self.robot_sim.robot_states.r_pos
        self.normalizer._norm_pos(start_pos.reshape(3))
        self.normalizer.reset_reference()
        self._norm()

        self.start_roll = abs(self.robot_sim.robot_states.rpy[0])
        self.start_height = self.robot_sim.robot_states.r_pos[2]

        self.stagnation_counter = 0

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
        abs_roll = abs(roll)
        z = self.robot_sim.robot_states.r_pos[2]

        r -= self.ori_weight * (abs_roll / np.pi)

        if abs_roll < np.pi * 30 / 180:
            r += self.ori_weight

        if abs_roll > np.pi / 2 and self.current_mode == 5:
            r -= 1.0

        if self.current_mode == 0:
            r -= 0.01

        if self.current_mode != self.last_mode:
            if self.current_mode in self.PROGRESS_MODES:
                self.start_roll = abs(self.robot_sim.robot_states.rpy[0])
                self.start_height = z

            if self.current_mode_success_tick < self.MIN_DWELL_TICKS:
                r -= 2.0
            else:
                r -= 0.05

            self.current_mode_success_tick = 0
            self.total_mode_changes += 1

        if (self.robot_sim.robot_states.subtask_succes and
                not self.robot_sim.robot_states.sr_mode_completed[self.current_mode]):

            self.robot_sim.robot_states.sr_mode_completed[:] = \
                [False] * len(self.robot_sim.robot_states.sr_mode_completed)

            count = self.end_phase_count[self.current_mode]
            r += self.end_task_weight / (2**count)

            decayed_time = self.time_extension / (2**count)
            self.current_step_limit = min(self.current_step_limit + decayed_time, self.max_step_limit)

            self.end_phase_count[self.current_mode] += 1

        if self.current_mode in self.PROGRESS_MODES:
            roll_progress = self.start_roll - abs_roll
            r += 0.5 * roll_progress

            if roll_progress < 0.01:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
        else:
            self.stagnation_counter = 0

        self.current_mode_success_tick += 1
        r -= 0.001 * self.current_mode_success_tick

        if self.robot_sim.robot_states.mpc_fail:
            r -= 1.0

        return float(r)

    def _termination(self):
        roll = abs(self.robot_sim.robot_states.rpy[0])
        pitch = abs(self.robot_sim.robot_states.rpy[1])
        z = self.robot_sim.robot_states.r_pos[2]

        physically_upright = (roll < np.pi * 15 / 180 and pitch < np.pi * 15 / 180 and z > self.min_upright_height)

        REQUIRED_PHASES = [1, 2, 3, 4, 5]
        sequence_completed = all(self.end_phase_count[p] > 0 for p in REQUIRED_PHASES)

        stand_completed = self.robot_sim.robot_states.sr_mode_completed[5]

        success = physically_upright and stand_completed and sequence_completed

        mpc_crash = self.robot_sim.robot_states.critical_mpc_fail
        too_many_switches = self.total_mode_changes > 50
        stagnated = self.stagnation_counter > 300

        terminated = success or mpc_crash or stagnated or too_many_switches
        truncated = self.current_step >= self.current_step_limit

        return terminated, truncated, success

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
