"""Transaction cost models."""

import torch


def proportional_cost(
    delta: torch.Tensor,
    prev_delta: torch.Tensor,
    S: torch.Tensor,
    epsilon: float = 0.001,
) -> torch.Tensor:
    """
    Proportional cost: epsilon * S_t * |delta_t - delta_{t-1}|

    delta:      (n_paths,) hedge ratio at t
    prev_delta: (n_paths,) hedge ratio at t-1
    S:          (n_paths,) spot at t
    Returns:    (n_paths,) cost at step t
    """
    return epsilon * S * (delta - prev_delta).abs()


def fixed_proportional_cost(
    delta: torch.Tensor,
    prev_delta: torch.Tensor,
    S: torch.Tensor,
    epsilon: float = 0.001,
    fixed: float = 0.0,
) -> torch.Tensor:
    """Proportional + fixed (paid when trading occurs)."""
    trade = (delta - prev_delta).abs()
    return epsilon * S * trade + fixed * (trade > 0).float()
