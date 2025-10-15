import numpy as np


class SmoothFilter():

    def __init__(self):
        self.qr = np.zeros((12, 1))

    def update(self, mode=None):

        return self.qr
