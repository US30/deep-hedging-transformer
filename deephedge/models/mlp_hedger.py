"""Per-step MLP hedger. No memory — processes each time step independently."""

import torch
import torch.nn as nn


class MLPHedger(nn.Module):
    """
    State: (log_S, log_v, t, prev_delta)  → hedge ratio ∈ [-2, 2]

    Baseline with no temporal memory.
    """
    def __init__(self, hidden_dim: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(4, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        S: torch.Tensor,         # (n_paths,)
        v: torch.Tensor,
        t: torch.Tensor,
        prev_delta: torch.Tensor,
    ) -> torch.Tensor:
        log_S = S.log()
        log_v = v.clamp(min=1e-8).log()
        x = torch.stack([log_S, log_v, t.expand_as(S), prev_delta], dim=1)  # (P, 4)
        return self.net(x).squeeze(-1)  # (P,)
