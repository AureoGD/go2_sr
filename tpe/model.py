# tpe/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class TPE(nn.Module):

    def __init__(self, input_dim, window_size, num_classes=3):
        super().__init__()

        self.input_dim = input_dim
        self.window_size = window_size
        self.num_classes = num_classes

        # 1D CNN over temporal dimension
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, padding=2)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        x shape:
            (batch_size, window_size, input_dim)

        We transpose to:
            (batch_size, input_dim, window_size)
        """
        x = x.transpose(1, 2)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = torch.mean(x, dim=2)

        logits = self.fc(x)

        return logits

    def predict(self, x):
        """
        Returns softmax probabilities.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
        return probs
