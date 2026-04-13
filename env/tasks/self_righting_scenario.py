import numpy as np
from typing import List
from env.tasks.base_task_scenario import BaseTaskScenario


class SelfRightingScenario(BaseTaskScenario):

    def __init__(self, config=None):
        self.config = config or {}

        # Configurações originais (Caso 0)
        self.q0_base = np.array([-0.2, 2.0, -1.65, -0.6, 1.86, -1.65, -0.5, 1.06, -1.0, 0.25, 1.36, -1.05])
        self.q0_ranges = np.array([[-0.25, 0.25], [-0.5, 0.5], [-0.75, 0.75], [-0.25, 0.25], [-0.5, 0.5], [-0.75, 0.75],
                                   [-0.25, 0.25], [-0.5, 0.5], [-0.75, 0.75], [-0.25, 0.25], [-0.5, 0.5], [-0.75,
                                                                                                           0.75]])
        self.r0_base = np.array([3.14, 0.0, 0.0])
        self.r0_yaw_range = [-3.14, 3.14]
        self.b0_base = np.array([0.0, 0.0, 0.2])
        self.b0_range = np.array([[-1.0, 1.0], [-1.0, 1.0], [0.0, 0.1]])

        # Definição dos Ends (1 a 9) com valores arredondados para 2 casas decimais
        self.ends_data = {
            1: {
                "b_z": 0.10,
                "rpy": np.array([-3.14, -0.08, 0.0]),
                "q": np.array([0.66, 1.40, -2.61, -0.66, 1.40, -2.61, 0.66, 1.40, -2.61, -0.66, 1.40, -2.61])
            },
            2: {
                "b_z": 0.11,
                "rpy": np.array([-3.13, -0.01, -0.05]),
                "q": np.array([-0.51, 1.50, -2.01, -0.80, 1.01, -2.60, -0.48, 1.50, -2.01, -0.69, 4.29, -2.29])
            },
            3: {
                "b_z": 0.18,
                "rpy": np.array([1.54, 0.0, -0.01]),
                "q": np.array([0.29, 1.49, -2.00, -0.60, 1.30, -2.60, 0.30, 1.51, -2.00, 0.89, 3.75, -1.50])
            },
            4: {
                "b_z": 0.18,
                "rpy": np.array([1.24, 0.0, 0.0]),
                "q": np.array([0.33, 1.52, -2.01, -0.58, 0.90, -1.54, 0.33, 1.50, -2.01, -0.57, 0.93, -1.50])
            },
            5: {
                "b_z": 0.13,
                "rpy": np.array([0.01, -0.02, -0.02]),
                "q": np.array([-0.0, 1.41, -2.72, -0.02, 1.40, -2.72, -0.02, 1.44, -2.73, -0.01, 1.39, -2.73])
            },
            7: {
                "b_z": 0.16,
                "rpy": np.array([-2.68, -0.0, 0.01]),
                "q": np.array([0.80, 1.01, -2.60, 0.57, 1.50, -2.01, -0.35, 4.16, -2.40, 0.52, 1.50, -2.01])
            },
            8: {
                "b_z": 0.18,
                "rpy": np.array([-1.54, 0.0, 0.02]),
                "q": np.array([0.60, 1.30, -2.60, -0.29, 1.49, -2.00, -0.89, 3.75, -1.50, -0.30, 1.51, -2.00])
            },
            9: {
                "b_z": 0.18,
                "rpy": np.array([-1.24, 0.0, 0.02]),
                "q": np.array([0.58, 0.90, -1.54, -0.33, 1.50, -2.00, 0.57, 0.93, -1.50, -0.33, 1.50, -2.01])
            }
        }

    def sample(self, ch=None):
        if ch is None:
            choice = np.random.randint(0, 10)
        else:
            choice = ch

        # Caso 0 ou se o índice não estiver definido (ex: 6)
        if choice == 0 or choice not in self.ends_data:
            return {"q0": self._generate_q0(), "r0": self._generate_r0(), "b0": self._generate_b0()}

        data = self.ends_data[choice]

        # b0: XY aleatório entre -1 e 1, Z fixo + offset de 0.05
        b0 = np.array([
            np.round(np.random.uniform(-1.0, 1.0), 2),
            np.round(np.random.uniform(-1.0, 1.0), 2),
            np.round(data["b_z"] + 0.05, 2)
        ])

        # r0: Roll/Pitch fixos, Yaw aleatório entre -pi e pi
        r0 = data["rpy"].copy()
        r0[2] = np.round(np.random.uniform(-np.pi, np.pi), 2)

        # q0: Valores fixos do End
        q0 = data["q"].copy()

        return {"q0": q0, "r0": r0, "b0": b0}

    def apply(self, env, scenario):
        return env.reset(q0=scenario["q0"], r0=scenario["r0"], b0=scenario["b0"])

    def _generate_q0(self) -> np.ndarray:
        mins = self.q0_base + self.q0_ranges[:, 0]
        maxs = self.q0_base + self.q0_ranges[:, 1]
        return np.round(np.random.uniform(mins, maxs), 2)

    def _generate_r0(self) -> np.ndarray:
        r0 = self.r0_base.copy()
        r0[2] = np.round(np.random.uniform(self.r0_yaw_range[0], self.r0_yaw_range[1]), 2)
        return r0

    def _generate_b0(self) -> np.ndarray:
        mins = self.b0_base + self.b0_range[:, 0]
        maxs = self.b0_base + self.b0_range[:, 1]
        return np.round(np.random.uniform(mins, maxs), 2)
