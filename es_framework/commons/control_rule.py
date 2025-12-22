import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ControlRule(nn.Module):

    def __init__(self, observation_dim: int, output_dim: int, **model_cfg):
        super().__init__()

        self.discrete = bool(model_cfg.pop("discrete", False))
        fc1_dim = int(model_cfg.pop("fc1_dim", 64))
        fc2_dim = int(model_cfg.pop("fc2_dim", 64))
        action_low = model_cfg.pop("action_low", None)
        action_high = model_cfg.pop("action_high", None)

        # layers
        self.fc1 = nn.Linear(observation_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc_out = nn.Linear(fc2_dim, output_dim)

        # optional continuous-action scaling
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
        # RAW outputs: logits (discrete) or unsquashed (continuous)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return self.fc_out(h)

    @torch.no_grad()
    def predict(self, state, device="cpu", deterministic=False):
        x = torch.from_numpy(state).float() if isinstance(state, np.ndarray) else state.float()
        single = False
        if x.ndim == 1:
            x = x.unsqueeze(0)
            single = True
        x = x.to(device)

        out = self.forward(x)
        if self.discrete:
            a = F.softmax(out, dim=-1).argmax(dim=-1).cpu().numpy()
            return int(a[0]) if single else a.astype(np.int64), state

        a = torch.tanh(out)
        if self._low is not None and self._high is not None:
            low = self._low.to(a.device)
            high = self._high.to(a.device)
            a = low + (a + 1.0) * 0.5 * (high - low)
        a = a.cpu().numpy().astype(np.float32)
        return a[0] if single else a, state

    def reset_parameters(self, ortho: bool = True, last_layer_std: float = 0.01, seed: int | None = None):
        if seed is not None:
            torch.manual_seed(seed)

        if ortho:
            gain = nn.init.calculate_gain("relu")
            nn.init.orthogonal_(self.fc1.weight, gain)
            nn.init.constant_(self.fc1.bias, 0.0)
            nn.init.orthogonal_(self.fc2.weight, gain)
            nn.init.constant_(self.fc2.bias, 0.0)
            nn.init.orthogonal_(self.fc_out.weight, last_layer_std)
            nn.init.constant_(self.fc_out.bias, 0.0)
        else:
            nn.init.kaiming_uniform_(self.fc1.weight, a=0.0, nonlinearity="relu")
            nn.init.constant_(self.fc1.bias, 0.0)
            nn.init.kaiming_uniform_(self.fc2.weight, a=0.0, nonlinearity="relu")
            nn.init.constant_(self.fc2.bias, 0.0)
            nn.init.xavier_uniform_(self.fc_out.weight)
            nn.init.constant_(self.fc_out.bias, 0.0)
        return self
