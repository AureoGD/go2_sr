import numpy as np


class UnitreeSR():

    def __init__(self, **kwargs):
        self.robot_states = kwargs.get('robot_states')

        self.kp = 80
        self.kd = 5

        self.KP = self.kp * np.eye(12)
        self.KD = self.kd * np.eye(12)
        self.q_phase_0 = np.array([0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7])
        self.q_phase_1 = np.array([0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7, -1.0, 1.4, -2.8])
        self.q_phase_2 = np.array([0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7, -1.0, 4.3, -2.8])
        self.q_phase_3 = np.array([-0.6, 1.55, -2.8, -0.21, 1.6, -2.8, -0.55, 1.6, -2.8, -0.16, 4.3, -2.8])

        self.kp_mode_1 = np.array([80, 80, 0, 80, 80, 0, 80, 80, 0, 80, 80, 0])
        self.kp_mode_2 = np.array([80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80])
        self.kd_mode_1 = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0])
        self.kd_mode_2 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        self.kp_per_phase = [
            self.kp_mode_1, self.kp_mode_2, self.kp_mode_2, self.kp_mode_2, self.kp_mode_2, self.kp_mode_2
        ]
        self.kd_per_phase = [
            self.kd_mode_1, self.kd_mode_2, self.kd_mode_2, self.kd_mode_2, self.kd_mode_2, self.kd_mode_2
        ]

        self.sr_q_refs = [self.q_phase_0, self.q_phase_1, self.q_phase_2, self.q_phase_3]

        self.ramp_duration = [65, 75, 75, 25]
        self.phase_duration = [65, 100, 100, 40]

        self.qr_ant = np.zeros((12, 1))
        # self.qr_ant = np.zeros((12, 1))
        self.increment_per_step = np.zeros((12, 1))

        self.iterations = 0
        self.phase_iterations = 0
        self.phase_now = 0

    def update(self, mode=None):

        if self.iterations < sum(self.phase_duration):
            if self.phase_iterations == 0:
                self.qr_ant = self.robot_states.q.copy()
                self.increment_per_step = (
                    (self.sr_q_refs[self.phase_now]).reshape(12, 1) - self.qr_ant) / self.ramp_duration[self.phase_now]
                self.KP = np.diag(self.kp_per_phase[self.phase_now])
                self.KD = np.diag(self.kd_per_phase[self.phase_now])
            if self.phase_iterations < self.ramp_duration[self.phase_now]:
                self.qr_ant += self.increment_per_step
            self.phase_iterations += 1
            if self.phase_iterations >= self.phase_duration[self.phase_now]:
                self.phase_iterations = 0
                self.phase_now += 1

        self.iterations += 1
        return self.qr_ant.reshape(12), self.KP, self.KD

    def reset_phase(self):
        self.iterations = 0
        self.phase_iterations = 0
        self.phase_now = 0
        self.KP = self.kp * np.eye(12)
        self.KD = self.kd * np.eye(12)
