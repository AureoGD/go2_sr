import numpy as np
from control.self_righting.base_self_righting_controller import BaseSelfRighting
from control.self_righting.rgc_mpc_solution.utils.build_gamma_star import GammaBuilder, CONFIGS

# from control.self_righting.rgc_mpc_solution.utils.build_gamma_star import build_gamma_star


class ModelValidation(BaseSelfRighting):

    def __init__(self, **kwargs):
        super().__init__()

        self.pin_engine = kwargs.get("pin_engine")

        self.comp_grav = True

        self.KP = kwargs.get("kp", 50.0)
        self.KD = kwargs.get("kd", 3.0)

        # self.qr = np.array([0, 0.67, -1.3, 0, 0.67, -1.3, 0, 0.67, -1.3, 0, 0.67, -1.3])
        # self.qr = np.array([-0.9, 0, -2.8, 0, 1.26, -2.8, -0.5, 0, -2.8, 0.9, 3.75, -1.5])
        self.qr = np.array([-0.9, 0, -2.8, 0, 1.26, -2.8, -0.9, 0, -2.8, 0.9, 3.75, -1.5])
        # self.qr = np.array([0.1, 2.4, -2.7, -0.5, 1.4, -2.7, 0.1, 2.4, -2.7, 0.0, 1.4, -2.7])

        self.Kp_vec = np.ones(12) * 50
        self.Kd_vec = np.ones(12) * 3

        self.gamma_builder = GammaBuilder(self.pin_engine, CONFIGS["roll_cw"], self.Kp_vec, self.Kd_vec, 2)

        # Gains per phase
        # self.kp = np.array([50] * 12)
        # self.kd = np.array([3] * 12)

        # self.KP = np.array([50] * 12)
        # self.KD = np.array([3] * 12)

        self.ramp_duration = 100
        self.phase_duration = 300

        self.qr_ant = np.zeros((12, 1))
        self.increment_per_step = np.zeros((12, 1))

        self.iterations = 0
        self.phase_iterations = 0
        self.phase_now = 0

        self.b_ref = np.array([0, 0.0, 0.27])
        self.ep_ref = np.array([0, 0, 0, 1])

        self.l_star = {
            "FR": None,
            "FL": None,
            "RR": None,
            "RL": None,
        }

        self.r_ref = None

        self.first_int = True
        self.last_r_vel = None
        self.b = np.zeros(5)
        self.L = 0.025 * np.ones(5)
        self.step = False

    def before_step(self, state, action):
        cs = state.low_level
        rs = state.robot
        dg = state.debug

        if action == 2:
            dr = rs.r_vel.flatten()
            omega = rs.omega.flatten()
            db = rs.b_vel.flatten()
            J_com = self.pin_engine.com_jacobian()
            r = rs.r_pos.flatten()

            gl, ga, ge, Jc = self.gamma_builder.build(r, use_gamma_e=True)
            dg.dqe = gl @ dr - ga @ omega + 0 * ge @ (cs.qr - rs.q)

            dg.dr = dr
            dg.db = db
            dg.omega = omega
            dg.dq = rs.dq

            # gamma_l, gamma_a, gamma_e = build_gamma_star(rs.r_pos.flatten(), self.pin_engine, 1)
            # dg.dqe = gamma_l @ dr - gamma_a @ omega - 0 * gamma_e @ (cs.qr[3:6] - rs.q[3:6])

        self.dqr, Kp, Kd = self.compute_action(state, action)

        cs.qr += self.dqr
        cs.dqr = self.dqr

        dg.qr = cs.qr[0:3]
        dg.q = rs.q[0:3]

        cs.Kp = Kp
        cs.Kd = Kd

        # dg.dq = rs.dq[9:]
        # dg.q = rs.q[9:]
        # dg.qr = cs.qr[9:]
        # dg.dqr = cs.dqr[9:]

    def compute_action(self, state, action=None):
        dqr = np.zeros((12, 1))
        if action == 1:
            self.KP = self.Kp
            self.KD = self.Kd
            if not self.step:
                dqr = np.array([-0.2, -0.4, 0.4, 0.2, -0.4, 0.4, -0.2, -0.4, 0.4, 0.2, -0.4, 0.4])
                self.step = True

        if action == 2:
            if self.phase_now in [0] and self.phase_iterations == 0:
                state.low_level.qr = state.robot.q.copy()
            if self.iterations < self.phase_duration:
                if self.phase_iterations == 0:
                    if self.phase_iterations == 0 and state is not None:
                        self.qr_ant = state.low_level.qr.reshape(12, 1)

                    target = self.qr.reshape(12, 1)
                    duration = self.ramp_duration

                    self.increment_per_step = (target - self.qr_ant.reshape(12, 1)) / duration
                    self.KP = self.Kp
                    self.KD = self.Kd

                # Interpolate
                if self.phase_iterations < self.ramp_duration:
                    dqr = self.increment_per_step

                self.phase_iterations += 1

                # Advance Phase
                if self.phase_iterations >= self.phase_duration:
                    self.phase_iterations = 0
                    self.phase_now += 1

            self.iterations += 1
        return dqr.reshape(12), self.KP, self.KD

    def reset_phase(self, kp=None, kd=None):
        self.iterations = 0
        self.phase_iterations = 0
        self.phase_now = 0
        self.step = False
        if kp is None:
            self.Kp = np.array([50] * 12)
        else:
            self.Kp = np.array([kp] * 12)

        if kd is None:
            self.Kd = np.array([3] * 12)
        else:
            self.Kd = np.array([kd] * 12)

    def get_num_modes(self):
        pass

    def skew_symmetric_matrix(self, vector):
        v1, v2, v3 = vector
        matrix = np.array([[0, -v3, v2], [v3, 0, -v1], [-v2, v1, 0]])
        return matrix
