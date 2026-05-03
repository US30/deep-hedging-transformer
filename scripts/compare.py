"""
Final comparison table: load all checkpoints and produce paper-ready results.

Usage:
    python scripts/compare.py --payoff call --sim rbergomi
"""

import sys, os, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from deephedge.sim import simulate_rbergomi, simulate_heston, simulate_gbm
from deephedge.sim.payoffs import european_call, european_put, digital_call, down_and_out_call
from deephedge.sim.costs import proportional_cost
from deephedge.models import MLPHedger, LSTMHedger, TransformerHedger
from deephedge.models.bs_delta import BSDeltaHedger
from deephedge.train.train import build_features, compute_pnl
from deephedge.train.eval import compare_hedgers, bootstrap_cvar, plot_pnl_distributions


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P_EVAL = 200_000
    N_STEPS = 30
    T = 1.0 / 12
    K = 1.0
    EPSILON = 0.001

    sim_cfg = {
        "rbergomi": dict(H=0.1, eta=1.9, rho=-0.9, xi0=0.04),
        "heston":   dict(kappa=2.0, theta=0.04, sigma=0.5, rho=-0.7, v0=0.04),
        "gbm":      dict(sigma=0.2),
    }[args.sim]

    payoff_map = {
        "call":    lambda s: european_call(s, K),
        "put":     lambda s: european_put(s, K),
        "digital": lambda s: digital_call(s, K),
        "barrier": lambda s: down_and_out_call(s, K, 0.8),
    }
    payoff_fn = payoff_map[args.payoff]

    # Simulate eval paths
    if args.sim == "rbergomi":
        paths = simulate_rbergomi(P_EVAL, N_STEPS, T=T, device=device, **sim_cfg)
    elif args.sim == "heston":
        paths = simulate_heston(P_EVAL, N_STEPS, T=T, device=device, **sim_cfg)
    else:
        paths = simulate_gbm(P_EVAL, N_STEPS, T=T, sigma=0.2, device=device)

    S, v, tg = paths["S"], paths["v"], paths["t_grid"]
    feat = build_features(S, v, tg, torch.zeros(P_EVAL, N_STEPS, device=device))

    results = {}

    # BS-delta
    bs = BSDeltaHedger(K=K, payoff_type=args.payoff if args.payoff in ("call", "put") else "call")
    with torch.no_grad():
        deltas_list = [
            bs(S[:, k], v[:, k], torch.full((P_EVAL,), T - k*T/N_STEPS, device=device),
               torch.zeros(P_EVAL, device=device))
            for k in range(N_STEPS)
        ]
        deltas_bs = torch.stack(deltas_list, dim=1)
        results["BS-delta"] = compute_pnl(S, v, tg, deltas_bs, payoff_fn, proportional_cost, EPSILON)

    # Load neural models
    ckpt_dir = "checkpoints"
    model_specs = [
        ("MLP",         MLPHedger(),                                    f"{ckpt_dir}/mlp_{args.sim}_{args.payoff}/mlp_best.pt"),
        ("LSTM",        LSTMHedger(),                                   f"{ckpt_dir}/lstm_{args.sim}_{args.payoff}/lstm_best.pt"),
        ("Transformer", TransformerHedger(d_model=128, n_heads=4, n_layers=4), f"{ckpt_dir}/tfm_{args.sim}_{args.payoff}/transformer_best.pt"),
    ]

    for name, model, ckpt_path in model_specs:
        if not os.path.exists(ckpt_path):
            print(f"  Skipping {name}: checkpoint not found at {ckpt_path}")
            continue
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model = model.to(device).eval()
        with torch.no_grad():
            deltas = model(feat)
            pnl = compute_pnl(S, v, tg, deltas, payoff_fn, proportional_cost, EPSILON)
        results[name] = pnl

    print(f"\n{'='*60}")
    print(f"Results: sim={args.sim}  payoff={args.payoff}  epsilon={EPSILON}")
    print(f"{'='*60}")
    compare_hedgers(results)

    print("\nBootstrap CVaR 5% (95% CI, n_boot=2000):")
    for name, pnl in results.items():
        mean, lo, hi = bootstrap_cvar(pnl, alpha=0.05, n_boot=2000)
        print(f"  {name:<20} {mean:.4f}  [{lo:.4f}, {hi:.4f}]")

    if args.plot:
        os.makedirs("results/figs", exist_ok=True)
        plot_pnl_distributions(results, save_path=f"results/figs/{args.sim}_{args.payoff}.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sim",     default="rbergomi", choices=["rbergomi", "heston", "gbm"])
    p.add_argument("--payoff",  default="call",     choices=["call", "put", "digital", "barrier"])
    p.add_argument("--plot",    action="store_true")
    main(p.parse_args())
