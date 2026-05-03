"""
Mamba (SSM) hedger — week-14 ablation.

Requires: pip install mamba-ssm  (CUDA only, needs libcusolver)
Falls back gracefully with ImportError if not available.

Mamba processes sequences in O(N) vs Transformer O(N^2), potentially
better for long-horizon paths (T=1yr, daily = 252 steps).

Reference: Gu & Dao (2023) "Mamba: Linear-Time Sequence Modeling
with Selective State Spaces".
"""

import torch
import torch.nn as nn


def _try_import_mamba():
    try:
        from mamba_ssm import Mamba
        return Mamba
    except ImportError:
        return None


class MambaBlock(nn.Module):
    """Single Mamba layer with residual + LayerNorm. Falls back to GRU if Mamba unavailable."""
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        Mamba = _try_import_mamba()
        if Mamba is not None:
            self.ssm = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self._use_mamba = True
        else:
            self.ssm = nn.GRU(d_model, d_model, batch_first=True)
            self._use_mamba = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_mamba:
            return x + self.ssm(self.norm(x))
        else:
            out, _ = self.ssm(self.norm(x))
            return x + out


class MambaHedger(nn.Module):
    """
    Mamba SSM hedger.

    Interface identical to TransformerHedger:
      features: (n_paths, n_steps, 4)  →  hedge: (n_paths, n_steps)

    If mamba-ssm not installed, silently uses stacked GRU layers
    (still a valid ablation against LSTM).
    """
    def __init__(
        self,
        input_dim: int = 4,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, T, 4) → (B, T)"""
        x = self.input_proj(features)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x)).squeeze(-1)
