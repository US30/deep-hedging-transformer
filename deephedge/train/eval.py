"""
Evaluation utilities: OOS metrics, comparison table, PnL distribution plots.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any


def bootstrap_cvar(pnl: torch.Tensor, alpha: float = 0.05, n_boot: int = 1000) -> tuple[float, float, float]:
    """Return (mean, lower_95CI, upper_95CI) of CVaR_alpha via bootstrap."""
    losses = -pnl.cpu().numpy()
    n = len(losses)
    k = max(1, int(n * alpha))
    boot_cvars = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, n)
        sample = losses[idx]
        sample.sort()
        boot_cvars.append(sample[-k:].mean())
    boot_cvars = np.array(boot_cvars)
    cvar_mean = losses[np.argsort(losses)[-k:]].mean()
    return cvar_mean, np.percentile(boot_cvars, 2.5), np.percentile(boot_cvars, 97.5)


def pnl_metrics(pnl: torch.Tensor, alpha: float = 0.05) -> Dict[str, float]:
    """Full metric suite for a PnL vector."""
    pnl_np = pnl.cpu().float().numpy()
    losses = -pnl_np
    k = max(1, int(len(losses) * alpha))
    sorted_losses = np.sort(losses)[::-1]
    cvar = sorted_losses[:k].mean()
    var = np.percentile(losses, (1 - alpha) * 100)

    return {
        "pnl_mean": float(pnl_np.mean()),
        "pnl_std":  float(pnl_np.std()),
        "pnl_skew": float(_skew(pnl_np)),
        "pnl_kurt": float(_kurt(pnl_np)),
        f"var_{int(100*(1-alpha))}%": float(var),
        f"cvar_{int(100*alpha)}%": float(cvar),
        "sharpe": float(pnl_np.mean() / (pnl_np.std() + 1e-9)),
    }


def _skew(x: np.ndarray) -> float:
    m = x - x.mean()
    return (m ** 3).mean() / ((m ** 2).mean() ** 1.5 + 1e-12)


def _kurt(x: np.ndarray) -> float:
    m = x - x.mean()
    return (m ** 4).mean() / ((m ** 2).mean() ** 2 + 1e-12) - 3


def compare_hedgers(results: Dict[str, torch.Tensor], alpha: float = 0.05) -> None:
    """Print comparison table across hedger methods."""
    header = f"{'Method':<20} {'PnL mean':>10} {'PnL std':>10} {'VaR 95%':>10} {'CVaR 5%':>10} {'Sharpe':>8}"
    print(header)
    print("-" * len(header))
    for name, pnl in results.items():
        m = pnl_metrics(pnl, alpha)
        print(
            f"{name:<20} "
            f"{m['pnl_mean']:>10.4f} "
            f"{m['pnl_std']:>10.4f} "
            f"{m[f'var_95%']:>10.4f} "
            f"{m[f'cvar_5%']:>10.4f} "
            f"{m['sharpe']:>8.3f}"
        )


def plot_pnl_distributions(results: Dict[str, torch.Tensor], save_path: str = None):
    """Histogram overlay of PnL distributions across methods."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skip plots")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, pnl in results.items():
        pnl_np = pnl.cpu().float().numpy()
        ax.hist(pnl_np, bins=100, alpha=0.5, label=name, density=True)
    ax.set_xlabel("PnL")
    ax.set_ylabel("Density")
    ax.set_title("PnL Distribution Comparison")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot: {save_path}")
    else:
        plt.show()
    plt.close()
