import gymnasium as gym
import numpy as np
from sim.go2_sim import Go2Sim


class Go2Env(gym.Env):

    def __init__(self, env_id=None, max_step=2000, **kwargs):

        super().__init__()

        controller = kwargs.get("controller", None)
        task = kwargs.get("task", None)

        if controller is None:
            raise ValueError("Controller must be provided to Go2Env")

        if task is None:
            raise ValueError("Task must be provided to Go2Env")

        self.controller = controller
        self.task = task

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

        self.task.normalizer.set_robot_limits(joint_limits=self.sim.joint_limits, torque_limits=self.sim.torque_limits)
        n_modes = self.controller.get_num_modes()
        self.task.normalizer.set_n_actions(n_modes)

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

        obs_dim = self.task.get_obs_dim()

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # --------------------------------------
        # META
        # --------------------------------------
        self.env_id = env_id

    def step(self, action):

        self.current_step += 1

        self.sim.simulation_loop(action)

        state = self.sim.state
        state.task_state = self.sim.controller.task_state
        state.tpe = self.task.tpe.state

        # --------------------------------------
        # Task block
        # --------------------------------------

        self.task.compute_features(state)

        obs = self.task.get_obs()

        reward = self.task.evaluate_reward()

        terminated, truncated = self.task.check_termination(self.current_step)

        info = self.task.gen_info()

        if info["sucess_flag"] is True:
            reward += info["sucess_extra_reward"]

        # --------------------------------------

        self.ep_reward += reward

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, q0=None, b0=None, r0=None):

        if seed is not None:
            np.random.seed(seed)

        self.sim.reset_robot_pose(q0=q0, b0=b0, r0=r0)

        state = self.sim.state
        state.task_state = self.sim.controller.task_state

        # reset normalizer via task
        if self.task.normalizer is not None:
            self.task.normalizer.reset_reference()

        # forma correta (encapsulada)
        obs = self.task.compute_initial_obs(state)
        self.task.reset()
        self.task.set_step_limit(self.max_step)

        self.current_step = 0
        self.ep_reward = 0.0

        return obs, {}

    def close(self):

        if hasattr(self, "viewer") and self.viewer is not None:
            try:
                self.viewer.close()
            except:
                pass

        # liberar referências
        self.sim = None
        self.mj_model = None
        self.mj_data = None
