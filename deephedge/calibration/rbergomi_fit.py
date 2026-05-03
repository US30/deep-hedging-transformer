"""
Calibrate rBergomi parameters (H, eta, rho, xi0) to:
  1. VIX term structure (forward variance xi0(T))
  2. Realized vol roughness (H)

Strategy:
  - H: estimated directly from realized vol structure function (Gatheral-Jaisson-Rosenbaum)
  - xi0: flat level set to average realized variance (can be extended to term structure)
  - eta, rho: match ATM implied vol term structure slope and skew via grid search / scipy.optimize

This is a practical calibration; for research-grade use add NelderMead or gradient-based calibration
via differentiable simulator.
"""

import numpy as np
import torch
from scipy.optimize import minimize
from typing import Optional

from deephedge.calibration.data import realized_roughness, compute_realized_vol
from deephedge.sim.rbergomi import simulate_rbergomi


def estimate_H(spx_prices) -> float:
    """Estimate Hurst exponent from SPX realized vol structure function."""
    results = realized_roughness(spx_prices, q_vals=[1, 2])
    H_estimates = [v["H_estimate"] for v in results.values()]
    H = float(np.mean(H_estimates))
    return max(0.05, min(0.45, H))  # clamp to valid rough vol range


def atm_iv_from_paths(paths: dict, K: float, T: float) -> float:
    """
    Approximate ATM implied vol by matching call price to BS formula.
    Uses average forward price as ATM proxy.
    """
    from scipy.stats import norm
    S = paths["S"][:, -1].float().cpu().numpy()
    payoff = np.maximum(S - K, 0.0)
    C_mc = float(payoff.mean())

    # BS inversion via Newton for sigma
    F = paths["S"][:, 0].mean().item()  # forward ≈ S0

    def bs_call(sigma):
        if sigma < 1e-4:
            return max(F - K, 0.0)
        d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return F * norm.cdf(d1) - K * norm.cdf(d2)

    from scipy.optimize import brentq
    try:
        iv = brentq(lambda s: bs_call(s) - C_mc, 1e-4, 5.0)
    except Exception:
        iv = float(np.sqrt(paths["v"][:, 0].mean().item()))
    return iv


def calibrate(
    spx_prices,
    vix_df=None,
    n_paths: int = 50_000,
    n_steps: int = 252,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Returns calibrated params dict: {H, eta, rho, xi0}.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: estimate H
    H = estimate_H(spx_prices)
    print(f"Estimated H = {H:.4f}")

    # Step 2: xi0 from realized vol level
    rv = compute_realized_vol(spx_prices)
    xi0 = float(rv.iloc[-252:].mean() ** 2)  # last year avg realized var
    print(f"xi0 (forward var) = {xi0:.4f}")

    # Step 3: calibrate eta and rho jointly via Monte Carlo
    # Objective: match (ATM IV for 1M, slope of skew)
    # Proxy: match 1M ATM IV using rho=-0.9 (typical SPX), eta as free param.
    target_iv_1m = float(rv.iloc[-21:].mean())  # recent 1M realized vol as proxy target

    def objective(params):
        eta, rho = params
        eta = max(0.1, min(4.0, eta))
        rho = max(-0.99, min(-0.01, rho))
        paths = simulate_rbergomi(
            n_paths=n_paths,
            n_steps=21,
            T=1.0 / 12,
            H=H, eta=eta, rho=rho, xi0=xi0,
            device=device, dtype=torch.float32,
        )
        iv = atm_iv_from_paths(paths, K=1.0, T=1.0 / 12)
        return (iv - target_iv_1m) ** 2

    result = minimize(
        objective,
        x0=[1.9, -0.9],
        method="Nelder-Mead",
        options={"maxiter": 30, "xatol": 0.01, "fatol": 0.0001},
    )
    eta, rho = result.x
    eta = max(0.1, min(4.0, eta))
    rho = max(-0.99, min(-0.01, rho))

    params = {"H": H, "eta": eta, "rho": rho, "xi0": xi0}
    print(f"Calibrated: {params}")
    return params
