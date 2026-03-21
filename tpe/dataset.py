# tpe/dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np


class TPEDataset(Dataset):

    def __init__(self, data_array, window_size):
        """
        data_array shape:
            (T, feature_dim + 1)

        Last column = phase label
        """
        self.window_size = window_size

        self.features = data_array[:, :-1]
        self.labels = data_array[:, -1].astype(int)

        self.T = len(self.features)

    def __len__(self):
        return self.T - self.window_size

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.window_size]
        y = self.labels[idx + self.window_size - 1]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return x, y
