import numpy as np
from control.self_righting.base_self_righting_controller import BaseSelfRighting


class UnitreeSelfRighting(BaseSelfRighting):

    def __init__(self, **kwargs):
        super().__init__()

        self.comp_grav = False

        self.kp = kwargs.get('kp', 80)
        self.kd = kwargs.get('kd', 5)

        self.KP = kwargs.get("kp", 50.0)
        self.KD = kwargs.get("kd", 3.0)

        # Hardcoded Unitree Sequences
        self.q_phase_0 = np.array([0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7])
        self.q_phase_1 = np.array([0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7])
        self.q_phase_2 = np.array([-0.8, 3.3, -2.77, 0.6, 2.2, -2.77, -0.8, 3.3, -2.77, 0.6, 2.2, -2.77])
        self.q_phase_3 = np.array([0.8, 1.1, -2.2, -0.8, 2.97, -2.77, 0.8, 1.1, -2.2, -0.8, 2.97, -2.77])
        self.q_phase_4 = np.array([-0.045, 1.26, -2.8, 0.5, 1.26, -2.8, -0.31, 1.295, -2.8, 0.31, 1.295, -2.8])
        self.q_phase_5 = np.array([0, 0.67, -1.3, 0, 0.67, -1.3, 0, 0.67, -1.3, 0, 0.67, -1.3])


        # Gains per phase
        self.kp_mode_1 = np.array([80] * 12)
        self.kp_mode_1[[2, 5, 8, 11]] = 0
        self.kp_mode_2 = np.array([80] * 12)
        self.kd_mode_1 = np.array([1] * 12)
        self.kd_mode_1[[2, 5, 8, 11]] = 0
        self.kd_mode_2 = np.array([1] * 12)

        self.kp_per_phase = [self.kp_mode_1, self.kp_mode_2, self.kp_mode_2, self.kp_mode_2,self.kp_mode_1,self.kp_mode_2]
        self.kd_per_phase = [self.kd_mode_1, self.kd_mode_2, self.kd_mode_2, self.kd_mode_2, self.kd_mode_1, self.kd_mode_2]

        self.sr_q_refs = [self.q_phase_0, self.q_phase_1, self.q_phase_2, self.q_phase_3, self.q_phase_4, self.q_phase_5]
        self.ramp_duration = [65, 65, 120, 120, 50, 50]
        self.phase_duration = [65, 65, 140, 145, 70, 75]

        self.qr_ant = np.zeros((12, 1))
        self.increment_per_step = np.zeros((12, 1))

        self.iterations = 0
        self.phase_iterations = 0
        self.phase_now = 0

        self.KP = np.array([80] * 12)
        self.KD = np.array([1] * 12)

    def before_step(self, state, action):
        cs = state.low_level

        self.dqr, Kp, Kd = self.compute_action(state, action)
     
        cs.qr += self.dqr
        cs.dqr = self.dqr

        cs.Kp = Kp
        cs.Kd = Kd

    def compute_action(self, state, action=None):
        # Unitree strategy IGNORES the 'mode' input from NN
        # because it follows a strict time schedule.
        dqr = np.zeros((12,1))
        if action == 1:
            if self.phase_now in [0, 1] and self.phase_now == 0:
                state.low_level.qr = state.robot.q.copy()
            if self.iterations < sum(self.phase_duration):
                # Init Phase
                if self.phase_iterations == 0:
                    if self.phase_iterations == 0 and state is not None:
                        self.qr_ant = state.low_level.qr.reshape(12, 1)

                    target = self.sr_q_refs[self.phase_now].reshape(12, 1)
                    duration = self.ramp_duration[self.phase_now]

                    self.increment_per_step = (target - self.qr_ant.reshape(12, 1)) / duration
                    self.KP = self.kp_per_phase[self.phase_now]
                    self.KD = self.kd_per_phase[self.phase_now]

                # Interpolate
                if self.phase_iterations < self.ramp_duration[self.phase_now]:
                    dqr = self.increment_per_step


                self.phase_iterations += 1

                # Advance Phase
                if self.phase_iterations >= self.phase_duration[self.phase_now]:
                    self.phase_iterations = 0
                    self.phase_now += 1

            self.iterations += 1
        return dqr.reshape(12), self.KP, self.KD

    def reset_phase(self):
        self.iterations = 0
        self.phase_iterations = 0
        self.phase_now = 0
        self.KP = np.array([80] * 12)
        self.KD = np.array([1] * 12)

    def get_num_modes(self):
        pass
