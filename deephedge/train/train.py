"""
Deep Hedging training loop.

Differentiable Monte Carlo:
  1. Simulate (S, v) paths from simulator
  2. Roll hedger across time steps to get hedge ratios
  3. Compute PnL = -payoff + sum(delta_k * dS_k) - sum(costs)
  4. Minimise risk(PnL) via backprop

Usage:
    python -m deephedge.train.train --model transformer --loss cvar --sim rbergomi
"""

import argparse
import math
import time
import torch
import torch.optim as optim
from pathlib import Path

from deephedge.sim import simulate_rbergomi, simulate_heston, simulate_gbm
from deephedge.sim.payoffs import european_call, european_put, digital_call, down_and_out_call
from deephedge.sim.costs import proportional_cost
from deephedge.models import MLPHedger, LSTMHedger, TransformerHedger
from deephedge.models.bs_delta import BSDeltaHedger
from deephedge.train.losses import make_loss


def build_features(S: torch.Tensor, v: torch.Tensor, t_grid: torch.Tensor, prev_deltas: torch.Tensor) -> torch.Tensor:
    """
    Build (n_paths, n_steps, 4) feature tensor.
    Features at step k: (log_S_k, log_v_k, t_k, prev_delta_k)
    """
    log_S = S[:, :-1].log()                    # (P, N)
    log_v = v[:, :-1].clamp(min=1e-9).log()   # (P, N)
    t = t_grid[:-1].unsqueeze(0).expand(S.shape[0], -1)  # (P, N)
    return torch.stack([log_S, log_v, t, prev_deltas], dim=2)  # (P, N, 4)


def compute_pnl(
    S: torch.Tensor,
    v: torch.Tensor,
    t_grid: torch.Tensor,
    deltas: torch.Tensor,
    payoff_fn,
    cost_fn=None,
    epsilon: float = 0.001,
) -> torch.Tensor:
    """
    PnL = -payoff(S) + sum_k delta_k * (S_{k+1} - S_k) - sum_k cost_k

    S:      (P, N+1)
    deltas: (P, N)   hedge ratio at each step (applied before observing next price)
    Returns (P,)
    """
    payoff = payoff_fn(S)                           # (P,)
    dS = S[:, 1:] - S[:, :-1]                       # (P, N)
    hedge_pnl = (deltas * dS).sum(dim=1)            # (P,)

    if cost_fn is not None and epsilon > 0:
        prev = torch.zeros_like(deltas[:, 0])
        total_cost = torch.zeros(S.shape[0], device=S.device)
        for k in range(deltas.shape[1]):
            c = cost_fn(deltas[:, k], prev, S[:, k], epsilon)
            total_cost = total_cost + c
            prev = deltas[:, k]
    else:
        total_cost = torch.zeros(S.shape[0], device=S.device)

    return -payoff + hedge_pnl - total_cost


def run_hedger(model, features: torch.Tensor) -> torch.Tensor:
    """
    Works for both sequential (LSTM, Transformer) and stateless (MLP) hedgers.
    Returns deltas: (P, N).
    """
    return model(features)


def simulate_batch(sim_name: str, n_paths: int, n_steps: int, T: float, device: torch.device, cfg: dict):
    if sim_name == "rbergomi":
        return simulate_rbergomi(n_paths, n_steps, T=T, device=device, **cfg)
    if sim_name == "heston":
        return simulate_heston(n_paths, n_steps, T=T, device=device, **cfg)
    return simulate_gbm(n_paths, n_steps, T=T, device=device, **cfg)


def evaluate(model, loss_fn, sim_name, sim_cfg, payoff_fn, n_paths, n_steps, T, epsilon, device):
    model.eval()
    with torch.no_grad():
        batch = simulate_batch(sim_name, n_paths, n_steps, T, device, sim_cfg)
        S, v, t_grid = batch["S"], batch["v"], batch["t_grid"]
        init_deltas = torch.zeros(n_paths, n_steps, device=device)
        features = build_features(S, v, t_grid, init_deltas)
        deltas = run_hedger(model, features)
        pnl = compute_pnl(S, v, t_grid, deltas, payoff_fn, proportional_cost, epsilon)
        loss_val = loss_fn(pnl)
        pnl_std = pnl.std().item()
        # CVaR 5% of losses
        cvar5 = (-pnl).sort(descending=True).values[:max(1, int(0.05 * n_paths))].mean().item()
    model.train()
    return {"loss": loss_val.item(), "pnl_std": pnl_std, "cvar5%": cvar5, "pnl_mean": pnl.mean().item()}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model
    if args.model == "mlp":
        model = MLPHedger().to(device)
    elif args.model == "lstm":
        model = LSTMHedger().to(device)
    elif args.model == "transformer":
        model = TransformerHedger(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} | params: {n_params:,}")

    # Loss
    loss_fn = make_loss(args.loss, lam=args.lam, alpha=args.alpha)

    # Payoff
    payoff_map = {
        "call": lambda S: european_call(S, args.K),
        "put": lambda S: european_put(S, args.K),
        "digital": lambda S: digital_call(S, args.K),
        "barrier": lambda S: down_and_out_call(S, args.K, args.barrier),
    }
    payoff_fn = payoff_map[args.payoff]

    # Simulator config
    sim_cfg_map = {
        "rbergomi": {"H": args.H, "eta": args.eta, "rho": args.rho, "xi0": args.xi0},
        "heston": {"kappa": 2.0, "theta": args.xi0, "sigma": 0.5, "rho": args.rho, "v0": args.xi0},
        "gbm": {"sigma": math.sqrt(args.xi0)},
    }
    sim_cfg = sim_cfg_map[args.sim]

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    print(f"\nTraining | sim={args.sim} | payoff={args.payoff} | loss={args.loss} | eps={args.epsilon}")
    print(f"Paths/batch={args.n_paths} | steps={args.n_steps} | T={args.T} | epochs={args.epochs}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        batch = simulate_batch(args.sim, args.n_paths, args.n_steps, args.T, device, sim_cfg)
        S, v, t_grid = batch["S"], batch["v"], batch["t_grid"]

        init_deltas = torch.zeros(args.n_paths, args.n_steps, device=device)
        features = build_features(S, v, t_grid, init_deltas)

        deltas = run_hedger(model, features)
        pnl = compute_pnl(S, v, t_grid, deltas, payoff_fn, proportional_cost, args.epsilon)
        loss = loss_fn(pnl)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        elapsed = time.time() - t0

        if epoch % args.log_every == 0:
            metrics = evaluate(model, loss_fn, args.sim, sim_cfg, payoff_fn,
                               args.n_paths, args.n_steps, args.T, args.epsilon, device)
            print(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"train_loss={loss.item():.4f} | "
                f"val_loss={metrics['loss']:.4f} | "
                f"pnl_std={metrics['pnl_std']:.4f} | "
                f"cvar5%={metrics['cvar5%']:.4f} | "
                f"t={elapsed:.1f}s"
            )
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                torch.save(model.state_dict(), save_dir / f"{args.model}_best.pt")

    print(f"\nDone. Best val loss: {best_loss:.4f}. Saved to {save_dir}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="transformer", choices=["mlp", "lstm", "transformer"])
    p.add_argument("--sim", default="rbergomi", choices=["rbergomi", "heston", "gbm"])
    p.add_argument("--payoff", default="call", choices=["call", "put", "digital", "barrier"])
    p.add_argument("--loss", default="cvar", choices=["quadratic", "entropic", "cvar"])
    p.add_argument("--n_paths", type=int, default=100_000)
    p.add_argument("--n_steps", type=int, default=30)
    p.add_argument("--T", type=float, default=1.0 / 12)   # 1 month
    p.add_argument("--K", type=float, default=1.0)
    p.add_argument("--barrier", type=float, default=0.8)
    p.add_argument("--epsilon", type=float, default=0.001)
    p.add_argument("--H", type=float, default=0.1)
    p.add_argument("--eta", type=float, default=1.9)
    p.add_argument("--rho", type=float, default=-0.9)
    p.add_argument("--xi0", type=float, default=0.04)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--save_dir", default="checkpoints")
    return p.parse_args()


if __name__ == "__main__":
    train(get_args())
