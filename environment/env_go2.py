import gymnasium as gym
import numpy as np
from environment.go2_model import Go2ModelSimMuJoCo
from environment.go2_normalize_states import Go2StateNormalizer
import math


class SchedullerRule():

    def __init__(self):
        self.n_controllers = 1

    def update_control_action(self, controller_index, state):

        return 0


class Go2Env(gym.Env):

    def __init__(self, env_id=None, max_step=2000, rendering=False):
        super().__init__()
        self.env_id = env_id
        self.rendering = rendering
        self.max_step = max_step

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

        self.current_mode = None
        self.last_mode = None
        self.total_mode_changes = 0

        # reward weights
        self.ori_weight = 0.01
        self.heigh_weight = 0.1

    def set_difficulty(self, value: float):
        self.scale_factor = float(value)

    def step(self, action):
        self.last_mode = self.current_mode
        self.current_mode = action
        self.current_step += 1

        self.robot_sim.control_loop(self.current_mode)
        self._norm()

        reward = self._reward()
        self.ep_reward += reward

        terminated = self.normalizer.is_out_of_box or self.robot_sim.robot_states.critical_mpc_fail or self.total_mode_changes > 100 or self.ep_reward < -50

        truncated = self.current_step >= self.max_step

        info = {}
        if truncated and not terminated:
            info["TimeLimit.truncated"] = True
            info["terminal_observation"] = self.st

        return self.st, reward, terminated, truncated, info

    def reset(self, *, seed=None, q0=None, b0=None, r0=None):

        if q0 is None:
            q0 = [-0.2, 2, -1.65, -0.6, 1.86, -1.65, -0.5, 1.06, -1.0, 0.25, 1.36, -1.05]

        if b0 is None:
            b0 = [0, 0, 0.085]

        if r0 is None:
            r0 = [np.pi, 0, 0]

        self.robot_sim.reset_robot_pose(q0=q0, b0=b0, r0=r0)

        self.total_mode_changes = 0

        super().reset(seed=seed)
        self.ep += 1
        ep_r = self.ep_reward
        self.current_step = 0
        self.ep_reward = 0
        self.last_mode = None
        start_pos = self.robot_sim.robot_states.r_pos
        self.normalizer.set_reference_position(start_pos.reshape(3))
        self.normalizer.reset_reference()
        self._norm()
        self.robot_sim.robot_states.critical_mpc_fail = False
        for mode_key in self.robot_sim.robot_states.sr_mode_completed.keys():
            self.robot_sim.robot_states.sr_mode_completed[mode_key] = False

        return self.st, {"Episode": self.ep, "Episode reward": ep_r}

    def _done(self):
        if self.current_step >= self.max_step:
            return True
        return False

    def _norm(self):
        self.st[0:3] = self.robot_sim.robot_states.r_pos.reshape(3)
        self.st[3:6] = self.robot_sim.robot_states.r_vel.reshape(3)
        self.st[6:10] = self.robot_sim.robot_states.epsilon.reshape(4)
        self.st[10:13] = self.robot_sim.robot_states.omega.reshape(3)
        self.st[13:25] = self.robot_sim.robot_states.q.reshape(12)
        self.st[25:37] = self.robot_sim.robot_states.dq.reshape(12)
        self.st[37:49] = self.robot_sim.robot_states.qr.reshape(12)
        self.st[49:61] = (self.robot_sim.robot_states.tau_pd + self.robot_sim.robot_states.tau_g).reshape(12)
        self.st[61] = self.current_mode
        self.st[62] = float(self.robot_sim.robot_states.mpc_fail)
        self.st = self.normalizer.normalize(self.st)

    def _reward(self):
        r = 0
        roll, _, _ = self.quaternion_to_rpy(self.robot_sim.robot_states.epsilon)

        r = -self.ori_weight * (abs(roll) / np.pi)

        # change to self.current_mode is in [3,4]
        if abs(roll) > np.pi / 2 and self.current_mode in [3, 4]:
            r -= 0.5

        if self.current_mode != self.last_mode:
            r -= 1
            self.total_mode_changes += 1

        if self.robot_sim.robot_states.r_pos[2] > 0.15:
            r += self.heigh_weight

        if self.robot_sim.robot_states.subtask_success is True and self.robot_sim.robot_states.sr_mode_completed[
                self.current_mode] is False:
            self.robot_sim.robot_states.sr_mode_completed[self.current_mode] = True
            r += 10

        r += -float(self.robot_sim.robot_states.mpc_fail)
        return float(r)

    def quaternion_to_rpy(self, quaternion):
        """        
        Parameters:
        quaternion: numpy array of shape (4, 1) with [[x], [y], [z], [w]] components
        
        Returns:
        roll, pitch, yaw: Euler angles in radians
        """
        x = quaternion[0, 0]
        y = quaternion[1, 0]
        z = quaternion[2, 0]
        w = quaternion[3, 0]

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            # Use 90 degrees if out of range
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def save_normalizer(self, filepath):
        """Save normalizer weights to file"""
        try:
            self.normalizer.save_weights(filepath)
            return True
        except Exception as e:
            print(f"Error saving normalizer: {e}")
            return False

    def load_normalizer(self, filepath):
        """Load normalizer weights from file"""
        try:
            self.normalizer.load_weights(filepath)
            return True
        except Exception as e:
            print(f"Error loading normalizer: {e}")
            return False
