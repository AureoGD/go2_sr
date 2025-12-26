import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):

    def __init__(self, observation_dim: int, output_dim: int, **model_cfg):
        super().__init__()

        self.discrete = bool(model_cfg.pop("discrete", False))

        hidden_dims = model_cfg.pop("hidden_dims", None)
        activation_name = model_cfg.pop("activation", "relu")

        action_low = model_cfg.pop("action_low", None)
        action_high = model_cfg.pop("action_high", None)

        if hidden_dims is None:
            fc1_dim = int(model_cfg.pop("fc1_dim", 64))
            fc2_dim = int(model_cfg.pop("fc2_dim", 64))
            hidden_dims = [fc1_dim, fc2_dim]

        if activation_name == "relu":
            self.activation = nn.ReLU()
        elif activation_name == "tanh":
            self.activation = nn.Tanh()
        elif activation_name == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unsupported activation: {activation_name}")

        layers = []
        in_dim = observation_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(self.activation)
            in_dim = h

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

        self._low = None
        self._high = None
        if action_low is not None and action_high is not None:
            low = np.asarray(action_low, dtype=np.float32).ravel()
            high = np.asarray(action_high, dtype=np.float32).ravel()
            if low.shape != high.shape or low.size != output_dim:
                raise ValueError("action_low/high must match output_dim")
            self._low = torch.tensor(low, dtype=torch.float32)
            self._high = torch.tensor(high, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.net(x)

    @torch.no_grad()
    def predict(self, state, device="cpu", deterministic=True):
        x = torch.from_numpy(state).float() if isinstance(state, np.ndarray) else state.float()

        single = False
        if x.ndim == 1:
            x = x.unsqueeze(0)
            single = True

        x = x.to(device)
        out = self.forward(x)

        if self.discrete:
            probs = F.softmax(out, dim=-1)
            if deterministic:
                a = probs.argmax(dim=-1)
            else:
                a = torch.multinomial(probs, 1).squeeze(-1)

            a = a.cpu().numpy()
            return int(a[0]) if single else a, state

        a = torch.tanh(out)

        if self._low is not None and self._high is not None:
            low = self._low.to(a.device)
            high = self._high.to(a.device)
            a = low + (a + 1.0) * 0.5 * (high - low)

        a = a.cpu().numpy().astype(np.float32)
        return a[0] if single else a, state
