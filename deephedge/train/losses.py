"""
Risk measures for deep hedging.

Each loss(pnl) takes PnL tensor (n_paths,) and returns scalar loss.
Training maximises E[-payoff] + hedge_gain; equivalently minimises risk of net PnL.
"""

import torch


def quadratic_loss(pnl: torch.Tensor) -> torch.Tensor:
    """E[PnL^2]  — penalises variance, symmetric."""
    return pnl.pow(2).mean()


def entropic_loss(pnl: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    """
    (1/lam) * log E[exp(-lam * PnL)]
    Exponential utility. lam = risk-aversion coefficient.
    Numerically stable via log-sum-exp.
    """
    return torch.logsumexp(-lam * pnl, dim=0) - torch.log(torch.tensor(pnl.shape[0], dtype=pnl.dtype, device=pnl.device))


def cvar_loss(pnl: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
    """
    CVaR_{alpha}(−PnL) = ES of worst (1-alpha) fraction of losses.
    Rockafellar-Uryasev dual:  CVaR = min_{z} { z + (1/alpha) * E[ max(-PnL - z, 0) ] }
    We use the learnable z trick: z is a scalar parameter passed externally,
    or approximated as the empirical alpha-quantile per batch.

    This function uses the sorted approximation (differentiable).
    """
    loss = -pnl  # convert to loss variable
    n = pnl.shape[0]
    k = max(1, int(n * alpha))
    # Sort losses descending and average top-k
    sorted_loss, _ = loss.sort(descending=True)
    return sorted_loss[:k].mean()


def cvar_rockafellar(pnl: torch.Tensor, z: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
    """
    CVaR dual form with explicit threshold z (trainable scalar).
    Use this for joint optimisation of hedger + z.
    """
    excess = (-pnl - z).clamp(min=0.0)
    return z + excess.mean() / alpha


def make_loss(name: str, **kwargs):
    if name == "quadratic":
        return quadratic_loss
    if name == "entropic":
        lam = kwargs.get("lam", 1.0)
        return lambda pnl: entropic_loss(pnl, lam)
    if name == "cvar":
        alpha = kwargs.get("alpha", 0.05)
        return lambda pnl: cvar_loss(pnl, alpha)
    raise ValueError(f"Unknown loss: {name}")
