"""Payoff functions. All take S: (n_paths, n_steps+1) and return (n_paths,)."""

import torch
from typing import Optional


def european_call(S: torch.Tensor, K: float = 1.0) -> torch.Tensor:
    return (S[:, -1] - K).clamp(min=0.0)


def european_put(S: torch.Tensor, K: float = 1.0) -> torch.Tensor:
    return (K - S[:, -1]).clamp(min=0.0)


def digital_call(S: torch.Tensor, K: float = 1.0) -> torch.Tensor:
    """Pays 1 if S_T > K. Delta blows up near K — hard for BS-delta."""
    return (S[:, -1] > K).float()


def down_and_out_call(
    S: torch.Tensor,
    K: float = 1.0,
    B: float = 0.8,
) -> torch.Tensor:
    """Call knocked out if S_t <= B at any time. Path-dependent."""
    alive = (S.min(dim=1).values > B).float()
    return alive * (S[:, -1] - K).clamp(min=0.0)


def autocall(
    S: torch.Tensor,
    K: float = 1.0,
    barriers: Optional[torch.Tensor] = None,
    coupon: float = 0.08,
    t_grid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Simplified annual autocall:
    - Observation at each integer year: if S_t / S_0 >= autocall_barrier → redeem at 1 + coupon*t
    - At maturity: if no early redemption → pay max(S_T/S_0, 0) (capital at risk)

    S: (n_paths, n_steps+1)
    barriers: autocall barrier levels at each observation (default 1.0)
    t_grid: (n_steps+1,) time in years
    """
    n_paths, n_steps_p1 = S.shape
    n_steps = n_steps_p1 - 1

    if t_grid is None:
        t_grid = torch.linspace(0, 1.0, n_steps_p1, device=S.device)

    S0 = S[:, 0:1]
    ratio = S / S0  # (n_paths, n_steps+1)

    obs_mask = ((t_grid % 1.0) < 1e-8) & (t_grid > 1e-8)  # annual observations
    obs_indices = obs_mask.nonzero(as_tuple=True)[0]

    if len(obs_indices) == 0:
        return ratio[:, -1].clamp(min=0.0)

    autocall_level = 1.0 if barriers is None else barriers
    pv = torch.zeros(n_paths, device=S.device)
    redeemed = torch.zeros(n_paths, dtype=torch.bool, device=S.device)

    for i, idx in enumerate(obs_indices):
        t_yr = t_grid[idx].item()
        at_obs = ratio[:, idx]
        trigger = (~redeemed) & (at_obs >= autocall_level)
        pv[trigger] = 1.0 + coupon * t_yr
        redeemed |= trigger

    # surviving paths at maturity: full capital at risk (floored at 0)
    maturity_pv = ratio[:, -1].clamp(min=0.0)
    pv[~redeemed] = maturity_pv[~redeemed]

    return pv
