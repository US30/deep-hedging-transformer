"""
LSTM hedger — Buehler 2019 style recurrent network.

Processes the full path sequentially. Hidden state carries memory.
State at each step: (log_S, log_v, t, prev_delta).
"""

import torch
import torch.nn as nn


class LSTMHedger(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (n_paths, n_steps, 4)  — (log_S, log_v, t, prev_delta) at each step
        Returns:  (n_paths, n_steps)     — hedge ratio per step
        """
        out, _ = self.lstm(features)            # (P, N, H)
        return self.head(out).squeeze(-1)       # (P, N)
