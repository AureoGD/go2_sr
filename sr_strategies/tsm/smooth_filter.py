import numpy as np


class SmoothFilter():

    def __init__(self, **kwargs):
        """ Smooth exponential filer """
        self.robot_states = kwargs.get('robot_states')
        self.settling_time = kwargs.get('settling_time')
        self.t_cont = kwargs.get('t_cont')
        self.qHL = kwargs.get('qHL')

        time_constant = self.settling_time / 4.0

        self.alpha = np.exp(-self.t_cont / time_constant)
        self.beta = 1 - self.alpha

    def smooth_reference(self):
        error = self.qHL - self.robot_states.qr
        return self.beta * error
