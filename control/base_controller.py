from abc import ABC, abstractmethod
import gymnasium as gym


class BaseController(ABC):

    def __init__(self):
        self.num_modes = 0

    # ======================================================
    # ACTION SPACE (obrigatório)
    # ======================================================
    @abstractmethod
    def get_action_space(self) -> gym.Space:
        """
        Define o espaço de ação usado pelo agente.
        """
        pass

    # ======================================================
    # BEFORE STEP (obrigatório)
    # ======================================================
    @abstractmethod
    def before_step(self, state, action):
        """
        Atualiza referências de controle com base na ação.

        - state: SystemState
        - action: saída do agente

        Deve atualizar:
            state.robot.qr
            state.controller.Kp
            state.controller.Kd
        """
        pass

    # ======================================================
    # AFTER STEP (opcional)
    # ======================================================
    def after_step(self, state):
        """
        Executado após o loop de física.
        Pode ser usado para logs, estados internos, etc.
        """
        pass

    # ======================================================
    # RESET (opcional)
    # ======================================================
    def reset(self):
        """
        Reset interno do controller (timers, buffers, etc.)
        """
        pass
