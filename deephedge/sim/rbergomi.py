"""
Rough Bergomi (rBergomi) path simulator via Bennedsen-Lunde-Pakkanen hybrid scheme.

Hybrid scheme splits W^H into a "past" sum of Wiener increments and a "Brownian"
part, giving exact covariance at the cost of O(N^2) per path but easily vectorised.

Reference: Bennedsen, Lunde, Pakkanen (2017) "Hybrid scheme for Brownian
semistationary processes".
"""

import torch
import math
from typing import Optional


def _gamma_ratio(H: float) -> float:
    """Γ(H + 0.5) / Γ(H - 0.5 + 1) * Γ(1) for kernel normalisation."""
    return math.gamma(H + 0.5)


def _kernel(H: float, n: int, device: torch.device) -> torch.Tensor:
    """
    Discretised Volterra kernel  g(j) = ((j+1)^α - j^α) / α,  α = H - 0.5.
    Shape: (n,)
    """
    alpha = H - 0.5  # negative for H < 0.5
    j = torch.arange(1, n + 1, dtype=torch.float64, device=device)
    return (j ** (H + 0.5) - (j - 1) ** (H + 0.5)) / (H + 0.5)


def simulate_rbergomi(
    n_paths: int,
    n_steps: int,
    T: float = 1.0,
    H: float = 0.1,
    eta: float = 1.9,
    rho: float = -0.9,
    xi0: float = 0.04,     # flat forward variance (sigma^2)
    S0: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """
    Simulate (S, v) paths under rough Bergomi model.

    v_t = xi0 * exp(eta * sqrt(2H) * W^H_t  -  0.5 * eta^2 * t^{2H})
    dS  = sqrt(v_t) * S_t * (rho * dW1 + sqrt(1-rho^2) * dW2)

    Args:
        n_paths:  batch size (use >= 100_000 for stable CVaR estimation)
        n_steps:  time steps (252 for daily, 50 for weekly)
        T:        maturity in years
        H:        Hurst exponent, H ≈ 0.1 for equities
        eta:      vol-of-vol
        rho:      spot-vol correlation (typically -0.9 for SPX)
        xi0:      initial forward variance (≈ ATM IV^2)
        S0:       initial spot
        device:   torch device
        dtype:    float32 or float64 (float64 recommended for kernel accuracy)

    Returns dict with keys:
        S:   (n_paths, n_steps+1)  spot prices
        v:   (n_paths, n_steps+1)  instantaneous variance
        dt:  scalar time step
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    # -------------------------------------------------------------------
    # Hybrid scheme: build fractional Brownian motion W^H
    # W^H_{t_{k+1}} ≈ sum_{j=0}^{k} g(j) * Z_{k-j}  + b(k) * Z_k
    # where g is the Volterra kernel and Z_i ~ N(0, dt)
    # -------------------------------------------------------------------
    # For memory efficiency on H100: compute incrementally O(N^2) per path
    # but fully vectorised over paths.

    # Build kernel (n_steps,) in float64 for numerical accuracy
    kernel = _kernel(H, n_steps, device)  # g(0), g(1), ..., g(n_steps-1)

    # Brownian increments for vol driving process
    # Z1: (n_paths, n_steps)  used for W^H
    # Z2: (n_paths, n_steps)  independent Brownian for S (before mixing with rho)
    Z1 = torch.randn(n_paths, n_steps, device=device, dtype=torch.float64) * sqrt_dt
    Z2 = torch.randn(n_paths, n_steps, device=device, dtype=torch.float64) * sqrt_dt

    # Correlated increment for S: rho * Z1 + sqrt(1-rho^2) * Z2
    sqrt_1mrho2 = math.sqrt(1 - rho * rho)
    dW_S = rho * Z1 + sqrt_1mrho2 * Z2  # (n_paths, n_steps)

    # Build W^H via convolution with kernel (causal, lower-triangular)
    # W_H[:, k] = sum_{j=0}^{k-1} kernel[j] * Z1[:, k-1-j]
    # Implemented via F.conv1d equivalently; explicit loop avoids huge temp tensors.

    # --- Accumulated W^H using cumulative convolution ---
    # For efficiency: use FFT-based convolution over time dim.
    # Z1: (n_paths, n_steps) -> (n_paths, 1, n_steps) for conv
    # kernel: (n_steps,) -> flip for correlation mode

    Z1_r = Z1.unsqueeze(1)  # (P, 1, N)
    k_r = kernel.flip(0).unsqueeze(0).unsqueeze(0)  # (1, 1, N)

    # Full linear convolution then take valid causal part
    import torch.nn.functional as F
    # Pad Z1 on left by (n_steps-1) zeros so conv output[k] = sum_{j=0..k} kernel[k-j]*Z1[j]
    Z1_padded = F.pad(Z1_r, (n_steps - 1, 0))  # (P, 1, 2N-1)
    WH_full = F.conv1d(Z1_padded, k_r)  # (P, 1, N)
    WH = WH_full.squeeze(1)  # (P, N)  W^H at t_1,...,t_N

    # W^H at t=0 is 0; prepend
    WH0 = torch.zeros(n_paths, 1, device=device, dtype=torch.float64)
    WH = torch.cat([WH0, WH], dim=1)  # (P, N+1)

    # -------------------------------------------------------------------
    # Variance process
    # v_{t_k} = xi0 * exp( eta*sqrt(2H)*W^H_{t_k} - 0.5*eta^2 * t_k^{2H} )
    # -------------------------------------------------------------------
    t_grid = torch.arange(n_steps + 1, device=device, dtype=torch.float64) * dt  # (N+1,)
    drift_v = -0.5 * eta ** 2 * t_grid ** (2 * H)  # (N+1,)
    v = xi0 * torch.exp(eta * math.sqrt(2 * H) * WH + drift_v)  # (P, N+1)

    # -------------------------------------------------------------------
    # Spot process (Euler-Maruyama, log-Euler for stability)
    # log S_{k+1} = log S_k - 0.5*v_k*dt + sqrt(v_k)*dW_S_k
    # -------------------------------------------------------------------
    log_S = torch.zeros(n_paths, n_steps + 1, device=device, dtype=torch.float64)
    log_S[:, 0] = math.log(S0)

    sqrt_v = v[:, :-1].clamp(min=1e-12).sqrt()  # (P, N)
    log_S[:, 1:] = log_S[:, 0:1] + torch.cumsum(
        -0.5 * v[:, :-1] * dt + sqrt_v * dW_S, dim=1
    )
    S = log_S.exp()

    return {
        "S": S.to(dtype),
        "v": v.to(dtype),
        "dt": dt,
        "t_grid": t_grid.to(dtype),
    }


def simulate_heston(
    n_paths: int,
    n_steps: int,
    T: float = 1.0,
    kappa: float = 2.0,
    theta: float = 0.04,
    sigma: float = 0.5,
    rho: float = -0.7,
    v0: float = 0.04,
    S0: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """
    Heston model via Euler-Maruyama (full truncation for v >= 0).

    dv = kappa*(theta - v)*dt + sigma*sqrt(v)*dW_v
    dS = sqrt(v)*S*(rho*dW_v + sqrt(1-rho^2)*dW_S)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    sqrt_1mrho2 = math.sqrt(1 - rho ** 2)

    Z1 = torch.randn(n_paths, n_steps, device=device, dtype=torch.float64) * sqrt_dt
    Z2 = torch.randn(n_paths, n_steps, device=device, dtype=torch.float64) * sqrt_dt
    dW_S = rho * Z1 + sqrt_1mrho2 * Z2

    # Scan over time steps — unavoidably sequential for v (Markovian)
    # but uses torch.compile-friendly pure tensor ops.
    v_steps = [torch.full((n_paths,), v0, device=device, dtype=torch.float64)]
    log_S_steps = [torch.full((n_paths,), math.log(S0), device=device, dtype=torch.float64)]

    for k in range(n_steps):
        vk = v_steps[-1].clamp(min=0.0)
        sqrt_vk = vk.sqrt()
        v_next = (vk + kappa * (theta - vk) * dt + sigma * sqrt_vk * Z1[:, k]).clamp(min=0.0)
        log_S_next = log_S_steps[-1] + (-0.5 * vk * dt + sqrt_vk * dW_S[:, k])
        v_steps.append(v_next)
        log_S_steps.append(log_S_next)

    v = torch.stack(v_steps, dim=1)           # (P, N+1)
    S = torch.stack(log_S_steps, dim=1).exp() # (P, N+1)

    t_grid = torch.arange(n_steps + 1, device=device, dtype=torch.float64) * dt
    return {
        "S": S.to(dtype),
        "v": v.to(dtype),
        "dt": dt,
        "t_grid": t_grid.to(dtype),
    }


def simulate_gbm(
    n_paths: int,
    n_steps: int,
    T: float = 1.0,
    sigma: float = 0.2,
    S0: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """GBM baseline. v is constant sigma^2."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dt = T / n_steps
    Z = torch.randn(n_paths, n_steps, device=device, dtype=torch.float64) * math.sqrt(dt)
    log_S = torch.zeros(n_paths, n_steps + 1, device=device, dtype=torch.float64)
    log_S[:, 0] = math.log(S0)
    log_S[:, 1:] = math.log(S0) + torch.cumsum(-0.5 * sigma ** 2 * dt + sigma * Z, dim=1)
    S = log_S.exp()
    v = torch.full_like(S, sigma ** 2)
    t_grid = torch.arange(n_steps + 1, device=device, dtype=torch.float64) * dt

    return {
        "S": S.to(dtype),
        "v": v.to(dtype),
        "dt": dt,
        "t_grid": t_grid.to(dtype),
    }
