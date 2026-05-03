"""
Transformer hedger — main contribution.

Causal (decoder-only) transformer with RoPE positional encoding.
Processes path features autoregressively; at step k sees only (0..k).

Architecture:
  - Input projection: 4 → d_model
  - N causal transformer blocks: MultiheadAttention (with causal mask) + FFN
  - Output head: d_model → 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _rotary_emb(dim: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE sin/cos cache. Returns (seq_len, dim//2) each."""
    half = dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)        # (seq, half)
    return freqs.sin(), freqs.cos()


def _apply_rope(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """
    x: (B, heads, seq, head_dim)
    sin, cos: (seq, head_dim//2)
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    sin = sin[:x.shape[2]].unsqueeze(0).unsqueeze(0)   # (1,1,seq,h/2)
    cos = cos[:x.shape[2]].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, H, T, head_dim)
        q = _apply_rope(q, sin, cos)
        k = _apply_rope(k, sin, cos)
        # Flash attention / scaled dot-product with causal mask
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), sin, cos)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerHedger(nn.Module):
    """
    Causal Transformer hedger.

    Input features per step: (log_S, log_v, t, prev_delta) — 4-dim.
    Output: hedge ratio per step in [-2, 2] (unconstrained linear head; clipping in eval).
    """
    def __init__(
        self,
        input_dim: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_mult, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

        # Pre-compute RoPE cache; lazy-extend if seq_len > max_seq_len
        sin, cos = _rotary_emb(d_model // n_heads, max_seq_len, device=torch.device("cpu"))
        self.register_buffer("rope_sin", sin)
        self.register_buffer("rope_cos", cos)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _get_rope(self, seq_len: int, device: torch.device):
        if seq_len > self.rope_sin.shape[0]:
            sin, cos = _rotary_emb(self.d_model // (self.head.in_features // self.d_model if False else 4), seq_len, device)
            self.rope_sin = sin
            self.rope_cos = cos
        return self.rope_sin[:seq_len].to(device), self.rope_cos[:seq_len].to(device)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (n_paths, n_steps, 4)
        Returns:  (n_paths, n_steps)
        """
        B, T, _ = features.shape
        x = self.input_proj(features)               # (B, T, d_model)
        sin, cos = self._get_rope(T, features.device)
        for block in self.blocks:
            x = block(x, sin, cos)
        x = self.norm(x)
        return self.head(x).squeeze(-1)             # (B, T)
