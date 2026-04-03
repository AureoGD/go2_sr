import numpy as np
import json
from sim.utils.orientation import compute_alpha


class StateNormalizer:

    def __init__(self,
                 joint_limits,
                 torque_limits,
                 stats_path="tpe/data/processed/normalization_stats.json",
                 box_size=0.5):

        # ======================================================
        # LOAD NORMALIZATION STATS (TPE / DATASET)
        # ======================================================
        # Estes valores vêm do dataset usado para treinar a TPE.
        # Garantem consistência entre treino e simulação online.
        with open(stats_path, "r") as f:
            stats = json.load(f)

        self.v_scale = stats.get("v_norm", 1.0)
        self.omega_scale = stats.get("omega_norm", 1.0)
        self.dq_scale = stats.get("dq_scale", 1.0)

        # ======================================================
        # ROBOT LIMITS
        # ======================================================
        # Limites físicos do robô (vindos da simulação)
        self.joint_limits = joint_limits  # shape (12,2)
        self.torque_limits = torque_limits  # shape (12,)

        # ======================================================
        # POSITION NORMALIZATION (LOCAL FRAME)
        # ======================================================
        # Usado para manter o robô dentro de uma "caixa" local
        self.box_size = box_size
        self.current_center = None
        self.is_out_of_box = False
        self.out_of_box_distance = 0.0

        self.eps = 1e-8

        self.n_actions = 0

    def set_n_actions(self, n_actions):
        self.n_actions = n_actions

    def set_robot_limits(self, joint_limits, torque_limits):
        self.joint_limits = joint_limits
        self.torque_limits = torque_limits

    # ======================================================
    # ---------------- FEATURE METHODS ----------------------
    # ======================================================

    def compute_alpha(self, state):
        """
        Mede o alinhamento do robô com a gravidade.

        Retorna:
            escalar ∈ [-1, 1]

        Interpretação:
            +1 → robô em pé
             0 → lateral
            -1 → de cabeça para baixo

        Usado em:
            - reward
            - observação
            - TPE
        """
        return compute_alpha(state.robot.epsilon)

    def normalize_q(self, q):
        """
        Normaliza as posições das juntas para o intervalo [-1, 1].

        Entrada:
            q → vetor (12,)

        Saída:
            vetor (12,) normalizado

        Usado em:
            - observação do RL
        """
        q = np.asarray(q).reshape(-1)
        return np.clip(
            2.0 * (q - self.joint_limits[:, 0]) / (self.joint_limits[:, 1] - self.joint_limits[:, 0] + self.eps) - 1.0,
            -1.0, 1.0)

    def normalize_dq(self, dq):
        """
        Normaliza as velocidades das juntas usando tanh.

        Entrada:
            dq → vetor (12,)

        Saída:
            vetor (12,)

        Interpretação:
            captura a velocidade de cada junta individualmente.

        Usado em:
            - observação do RL
        """
        dq = np.asarray(dq).reshape(-1)
        return np.tanh(dq / (self.dq_scale + self.eps))

    def normalize_tau(self, tau):
        """
        Normaliza os torques aplicados nas juntas.

        Entrada:
            tau → vetor (12,)

        Saída:
            vetor (12,) em escala aproximada [-1, 1]

        Usado em:
            - observação
            - análise de esforço
        """
        tau = np.asarray(tau).reshape(-1)
        return tau / (self.torque_limits + self.eps)

    def compute_dq_norm(self, dq):
        """
        Mede a intensidade global de movimento do robô.

        Entrada:
            dq → vetor (12,)

        Saída:
            escalar ∈ [0, 1]

        Interpretação:
            ||dq|| → quão rápido o robô está se movendo no geral.

        Usado em:
            - TPE
            - detecção de estagnação
            - reward
        """
        return np.tanh(np.linalg.norm(dq) / (self.dq_scale + self.eps))

    def normalize_velocity(self, v):
        """
        Normaliza a velocidade linear do robô.

        Entrada:
            v → vetor (3,)

        Saída:
            dir_v → direção unitária (3,)
            v_abs → magnitude normalizada (escalar)

        Usado em:
            - observação
            - análise de movimento global
        """
        v = np.asarray(v).reshape(-1)
        v_norm = np.linalg.norm(v)

        dir_v = v / (v_norm + self.eps)
        v_abs = np.tanh(v_norm / (self.v_scale + self.eps))

        return dir_v, v_abs

    def normalize_omega(self, omega):
        """
        Normaliza a velocidade angular completa do robô.

        Entrada:
            omega → vetor (3,)

        Saída:
            vetor (3,)

        Interpretação:
            [wx, wy, wz] → rotação completa do corpo

        Usado em:
            - observação
            - estabilidade
        """
        omega = np.asarray(omega).reshape(-1)
        return np.tanh(omega / (self.omega_scale + self.eps))

    # ======================================================
    # POSITION NORMALIZATION
    # ======================================================
    def normalize_position(self, pos):
        """
        Normaliza posição da base em relação a uma referência local.

        Mantém o robô dentro de uma "caixa" centrada.

        Também calcula:
            - se saiu da caixa
            - distância fora da caixa

        Usado em:
            - observação
            - termination
        """

        if self.current_center is None:
            self.current_center = pos.copy()

        relative = (pos - self.current_center) / (self.box_size / 2)

        self.is_out_of_box = np.any(np.abs(relative) > 1.0)

        clipped = np.clip(relative, -1.0, 1.0)

        self.out_of_box_distance = np.linalg.norm(relative - clipped)

        return clipped

    def normalize_action(self, action, lower, upper):

        action = np.asarray(action)

        return np.clip(2.0 * (action - lower) / (upper - lower + self.eps) - 1.0, -1.0, 1.0)

    def normalize_state_id(self, state_id):
        return state_id / self.num_states

    # ======================================================
    # ---------------- LEGACY ENCODE ------------------------
    # ======================================================
    def encode(self, state):
        """
        Gera vetor de observação completo.

        IMPORTANTE:
            Usa as funções modulares acima → mantém consistência.

        Saída:
            vetor de observação (obs)
        """

        rs = state.robot
        cs = state.controller

        alpha = self.compute_alpha(state)
        pos = self.normalize_position(rs.b_pos)

        dir_v, v_abs = self.normalize_velocity(rs.r_vel)
        omega = self.normalize_omega(rs.omega)

        q_norm = self.normalize_q(rs.q)
        tau_norm = self.normalize_tau(cs.tau)

        mode = self.normalize_mode(getattr(rs, "mode", 0))
        mpc_fail = float(cs.mpc_fail)

        current_state = self.normalize_state_id(getattr(rs, "current_state", 0))
        success_flag = float(getattr(rs, "subtask_succes", 0))

        obs = np.concatenate([
            [alpha],
            pos,
            dir_v,
            [v_abs],
            omega,
            q_norm,
            tau_norm,
            [mode],
            [mpc_fail],
            [current_state],
            [success_flag],
        ])

        return obs.astype(np.float32)

    # ======================================================
    # OBS DIM
    # ======================================================
    def get_obs_dim(self):
        """
        Retorna dimensão do vetor de observação.
        """
        return 1 + 3 + 3 + 1 + 3 + 12 + 12 + 1 + 1 + 1 + 1

    # ======================================================
    # RESET
    # ======================================================
    def reset_reference(self):
        """
        Reseta referência da normalização de posição.
        Deve ser chamado no reset do ambiente.
        """
        self.current_center = None
        self.is_out_of_box = False
        self.out_of_box_distance = 0.0
