import numpy as np
import gymnasium as gym

from control.base_controller import BaseController


class ActionAdapter(BaseController):
    """
    Direct joint controller.

    Modes:
        - incremental=True  → action = delta_q
        - incremental=False → action = absolute q
    """

    def __init__(self, action_dim=12, kp=50.0, kd=3.0, incremental=False, action_scale=1.0):

        super().__init__()

        self.action_dim = action_dim
        self.incremental = incremental
        self.action_scale = action_scale

        self.Kp_vec = np.ones(action_dim) * kp
        self.Kd_vec = np.ones(action_dim) * kd

    # ======================================================
    # ACTION SPACE
    # ======================================================
    def get_action_space(self):

        # Normalmente RL usa [-1, 1]
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

    # ======================================================
    # BEFORE STEP
    # ======================================================
    def before_step(self, state, action):

        if action is None:
            action = np.zeros(self.action_dim)

        action = np.asarray(action).reshape(self.action_dim)

        # Escala (útil para RL)
        action = action * self.action_scale

        q_current = state.robot.q  # ⚠️ ajuste se seu nome for diferente

        if self.incremental:
            qr = q_current + action
        else:
            qr = action

        # Atualiza estado (seguindo seu BaseController)
        state.controller.qr = qr
        state.controller.Kp = self.Kp_vec
        state.controller.Kd = self.Kd_vec

    # ======================================================
    def reset(self):
        pass
