"""Black-Scholes delta baseline (numerical, works for any payoff approximated as vanilla)."""

import torch
import math
from scipy.stats import norm
import numpy as np


def bs_call_delta(S: torch.Tensor, K: float, T_remaining: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Vectorised BS call delta.
    S, T_remaining, sigma: (n_paths,)
    Returns delta: (n_paths,)
    """
    eps = 1e-8
    S_np = S.cpu().double().numpy()
    T_np = T_remaining.cpu().double().numpy()
    sig_np = sigma.cpu().double().numpy()

    T_np = np.maximum(T_np, eps)
    sig_np = np.maximum(sig_np, eps)

    d1 = (np.log(S_np / K) + 0.5 * sig_np ** 2 * T_np) / (sig_np * np.sqrt(T_np))
    delta_np = norm.cdf(d1)

    return torch.tensor(delta_np, dtype=S.dtype, device=S.device)


def bs_put_delta(S: torch.Tensor, K: float, T_remaining: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return bs_call_delta(S, K, T_remaining, sigma) - 1.0


class BSDeltaHedger:
    """
    Wraps BS delta as a stateless hedger compatible with the training loop interface.
    Uses ATM implied vol (sqrt of current v as proxy).
    """
    def __init__(self, K: float = 1.0, payoff_type: str = "call"):
        self.K = K
        self.payoff_type = payoff_type

    def __call__(
        self,
        S: torch.Tensor,
        v: torch.Tensor,
        t_remaining: torch.Tensor,
        prev_delta: torch.Tensor,
    ) -> torch.Tensor:
        sigma = v.sqrt()
        if self.payoff_type == "call":
            return bs_call_delta(S, self.K, t_remaining, sigma)
        return bs_put_delta(S, self.K, t_remaining, sigma)
