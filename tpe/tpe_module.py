import numpy as np
import torch
from collections import deque
import json

from tpe.model import TPE
from tpe.state import TPEState


class TPEModule:

    def __init__(self, model_path, window=15, stats_path="tpe/data/processed/normalization_stats.json", device=None):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        self.window_size = window
        self.input_dim = 19

        self.state = TPEState()

        self.model = TPE(self.input_dim, self.window_size, num_classes=4)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.probs_dim = self.model.num_classes

        self.buffer = deque(maxlen=self.window_size)

        with open(stats_path, "r") as f:
            stats = json.load(f)
        self.dq_scale = stats.get("dq_scale", 1.0)

        self.EPS = 1e-8

    def reset(self):
        self.buffer.clear()
        self.state = TPEState()

    def predict(self, features):

        self.buffer.append(features)

        # -----------------------------------
        # Ainda não tem histórico suficiente
        # -----------------------------------
        if len(self.buffer) < self.window_size:
            self.state.valid = False
            return None

        # -----------------------------------
        # Monta input
        # -----------------------------------
        x = np.array(self.buffer)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)

        # -----------------------------------
        # Forward
        # -----------------------------------
        with torch.no_grad():
            logits = self.model(x)

            probs = torch.softmax(logits, dim=1)
            probs = probs.cpu().numpy().flatten()

        # -----------------------------------
        # Atualiza estado interno
        # -----------------------------------
        self.state.phase_probs = probs
        self.state.phase = int(np.argmax(probs))
        self.state.valid = True

        return probs
