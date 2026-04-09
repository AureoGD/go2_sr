import numpy as np
import torch
from collections import deque
import json
import os

from tpe.core.model import TPE
from tpe.core.state import TPEState


class TPEModule:

    def __init__(self, model_path, window=15, stats_path="tpe/data/processed/normalization_stats.json", device=None):

        # --------------------------------------------------
        # DEVICE
        # --------------------------------------------------
        self.device = torch.device("cpu") if device is None else device

        # --------------------------------------------------
        # LOAD CONFIG  (ESSENCIAL)
        # --------------------------------------------------
        config_path = model_path.replace("best_model.pt", "config.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config = json.load(f)

        self.input_dim = config["input_dim"]
        self.num_classes = config["num_classes"]

        # --------------------------------------------------
        # WINDOW
        # --------------------------------------------------
        self.window_size = window

        # --------------------------------------------------
        # STATE
        # --------------------------------------------------
        self.state = TPEState()

        # --------------------------------------------------
        # MODEL
        # --------------------------------------------------
        self.model = TPE(self.input_dim, self.window_size, num_classes=self.num_classes)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.probs_dim = self.model.num_classes

        # --------------------------------------------------
        # BUFFER
        # --------------------------------------------------
        self.buffer = deque(maxlen=self.window_size)

        # --------------------------------------------------
        # NORMALIZATION STATS
        # --------------------------------------------------
        with open(stats_path, "r") as f:
            stats = json.load(f)

        self.dq_scale = stats.get("dq_scale", 1.0)

        self.EPS = 1e-8

        print("\n TPEModule initialized:")
        print(f"   input_dim: {self.input_dim}")
        print(f"   num_classes: {self.num_classes}")
        print(f"   window_size: {self.window_size}")

    # ======================================================
    # RESET
    # ======================================================
    def reset(self):
        self.buffer.clear()
        self.state = TPEState()

    # ======================================================
    # PREDICT
    # ======================================================
    def predict(self, features):

        # Sanity check (MUITO útil)
        if len(features) != self.input_dim:
            raise ValueError(f"Feature size mismatch: got {len(features)}, expected {self.input_dim}")

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
        x = np.array(self.buffer, dtype=np.float32)
        x = torch.tensor(x).unsqueeze(0).to(self.device)

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
