import numpy as np


class UnitreeSR():

    def __init__(self, **kwargs):
        self.robot_states = kwargs.get('robot_states')

        self.kp = 100
        self.kd = 5

        self.KP = self.kp * np.eye(12)
        self.KD = self.kd * np.eye(12)
        self.q_phase_0 = np.array([0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7])
        self.q_phase_1 = np.array([0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7])
        self.q_phase_2 = np.array([-0.8, 3.3, -2.77, 0.6, 2.2, -2.77, -0.8, 3.3, -2.77, 0.6, 2.2, -2.77])
        self.q_phase_3 = np.array([0.8, 1.1, -2.2, -0.8, 2.97, -2.77, 0.8, 1.1, -2.2, -0.8, 2.97, -2.77])
        self.q_phase_4 = np.array([-0.045, 1.26, -2.8, 0.5, 1.26, -2.8, -0.31, 1.295, -2.8, 0.31, 1.295, -2.8])
        self.q_phase_5 = np.array([0, 0.67, -1.3, 0, 0.67, -1.3, 0, 0.67, -1.3, 0, 0.67, -1.3])


        self.kp_mode_1 = np.array([80, 80, 0, 80, 80, 0, 80, 80, 0, 80, 80, 0])
        self.kp_mode_2 = np.array([80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80])
        self.kd_mode_1 = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0])
        self.kd_mode_2 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        self.kp_per_phase = [
            self.kp_mode_1, self.kp_mode_2, self.kp_mode_2, self.kp_mode_2, self.kp_mode_1, self.kp_mode_2
        ]
        self.kd_per_phase = [
            self.kd_mode_1, self.kd_mode_2, self.kd_mode_2, self.kd_mode_2, self.kd_mode_1, self.kd_mode_2
        ]

        self.sr_q_refs = [
            self.q_phase_0, self.q_phase_1, self.q_phase_2, self.q_phase_3, self.q_phase_4, self.q_phase_5
        ]

        self.ramp_duration = [65, 65, 120, 120, 50, 50]
        self.phase_duration = [65, 65, 140, 145, 70, 75]

        self.qr_ant = np.zeros((12, 1))
        # self.qr_ant = np.zeros((12, 1))
        self.increment_per_step = np.zeros((12, 1))

        self.iterations = 0
        self.phase_iterations = 0
        self.phase_now = 0

    def update(self, mode=None):

        if self.iterations < 560:
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
        return self.qr_ant.reshape(12,), self.KP, self.KD

    def reset_phase(self):
        self.iterations = 0
        self.phase_iterations = 0
        self.phase_now = 0
        self.KP = self.kp * np.eye(12)
        self.KD = self.kd * np.eye(12)
