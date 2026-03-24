import gymnasium as gym
import numpy as np
from sim.go2_sim import Go2Sim
from env.normalizer import StateNormalizer


class Go2Env(gym.Env):

    def __init__(self, env_id=None, max_step=1500, **kwargs):

        super().__init__()

        controller = kwargs.get("controller", None)

        if controller is None:
            raise ValueError("Controller must be provided to Go2Env")

        self.controller = controller

        # --------------------------------------
        # SIMULATION
        # --------------------------------------
        self.sim = Go2Sim(
            urdf_path=kwargs.get("urdf_path"),
            mj_model=kwargs.get("mj_model"),
            mj_data=kwargs.get("mj_data"),
            controller=self.controller,
            con_dt=kwargs.get("con_dt", 0.01),
            dyn_dt=kwargs.get("dyn_dt", 0.001),
            viewer=kwargs.get("viewer", None),
        )

        # --------------------------------------
        # NORMALIZER
        # --------------------------------------
        self.normalizer = StateNormalizer(joint_limits=self.sim.joint_limits, torque_limits=self.sim.torque_limits)

        # --------------------------------------
        # ENV STATE
        # --------------------------------------
        self.max_step = max_step
        self.current_step = 0
        self.ep_reward = 0.0

        # --------------------------------------
        # SPACES
        # --------------------------------------
        self.action_space = self.controller.get_action_space()

        obs_dim = self.normalizer.get_obs_dim()

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # --------------------------------------
        # META
        # --------------------------------------
        self.env_id = env_id

    def reset(self, *, seed=None, q0=None, b0=None, r0=None):

        super().reset(seed=seed)

        self.sim.reset_robot_pose(q0=q0, b0=b0, r0=r0)

        self.current_step = 0
        self.ep_reward = 0.0

        state = self.sim.state
        obs = self.normalizer.encode(state)

        return obs, {}

    def step(self, action):

        self.current_step += 1

        self.sim.simulation_loop(action)

        state = self.sim.state
        obs = self.normalizer.encode(state)

        reward = self._reward()
        self.ep_reward += reward

        terminated, truncated = self._termination()

        info = {}

        return obs, reward, terminated, truncated, info

    def _reward(self):
        pass

    def _termination(self):
        pass
