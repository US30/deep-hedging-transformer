"""
Sanity checks — run with:  python -m pytest deephedge/tests/ -v
or just:  python deephedge/tests/test_sanity.py
"""

import math
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from deephedge.sim import simulate_rbergomi, simulate_heston, simulate_gbm
from deephedge.sim.payoffs import european_call, european_put, digital_call, down_and_out_call
from deephedge.sim.costs import proportional_cost
from deephedge.models import MLPHedger, LSTMHedger, TransformerHedger
from deephedge.train.losses import quadratic_loss, entropic_loss, cvar_loss
from deephedge.train.train import build_features, compute_pnl


DEVICE = torch.device("cpu")
P, N = 500, 30   # small for CPU sanity


def test_rbergomi_paths():
    paths = simulate_rbergomi(P, N, T=1/12, H=0.1, eta=1.9, rho=-0.9, xi0=0.04, device=DEVICE)
    S, v = paths["S"], paths["v"]
    assert S.shape == (P, N + 1)
    assert v.shape == (P, N + 1)
    assert (S > 0).all(), "Negative spot prices"
    assert (v > 0).all(), "Negative variance"
    # S0 should be ~1.0
    assert abs(S[:, 0].mean().item() - 1.0) < 0.01
    print(f"rBergomi OK | S mean={S[:,-1].mean():.4f} v mean={v[:,0].mean():.4f}")


def test_heston_paths():
    paths = simulate_heston(P, N, T=1/12, device=DEVICE)
    S = paths["S"]
    assert S.shape == (P, N + 1)
    assert (S > 0).all()
    print(f"Heston OK | S mean={S[:,-1].mean():.4f}")


def test_gbm_paths():
    paths = simulate_gbm(P, N, T=1/12, sigma=0.2, device=DEVICE)
    S = paths["S"]
    assert S.shape == (P, N + 1)
    print(f"GBM OK | S mean={S[:,-1].mean():.4f}")


def test_gbm_bs_convergence():
    """
    Zero-cost GBM + quadratic loss: MLP should converge near BS delta.
    We don't train here; just verify PnL computation is correct direction.
    """
    from deephedge.models.bs_delta import BSDeltaHedger
    paths = simulate_gbm(5_000, N, T=1/12, sigma=0.2, device=DEVICE)
    S, v, t_grid = paths["S"], paths["v"], paths["t_grid"]
    T = 1/12
    K = 1.0
    # Build deltas from BS formula
    bs = BSDeltaHedger(K=K, payoff_type="call")
    delta_list = []
    for k in range(N):
        t_rem = torch.tensor(T - k * T / N, dtype=torch.float32)
        d = bs(S[:, k], v[:, k], t_rem.expand(S.shape[0]), torch.zeros(S.shape[0]))
        delta_list.append(d)
    deltas = torch.stack(delta_list, dim=1)  # (P, N)
    pnl = compute_pnl(S, v, t_grid, deltas, lambda s: european_call(s, K), None, 0.0)
    # BS delta hedge PnL std should be smaller than unhedged payoff std
    payoff_std = european_call(S, K).std().item()
    pnl_std = pnl.std().item()
    print(f"BS hedge PnL std={pnl_std:.4f} vs unhedged payoff std={payoff_std:.4f}")
    assert pnl_std < payoff_std, "BS delta hedge did not reduce PnL std"


def test_payoffs():
    S = simulate_gbm(P, N, T=1/12, device=DEVICE)["S"]
    call = european_call(S)
    put = european_put(S)
    dig = digital_call(S)
    bar = down_and_out_call(S)
    for name, val in [("call", call), ("put", put), ("digital", dig), ("barrier", bar)]:
        assert (val >= 0).all(), f"{name} payoff negative"
    print(f"Payoffs OK | call={call.mean():.4f} put={put.mean():.4f} dig={dig.mean():.4f}")


def test_costs():
    delta = torch.ones(P) * 0.5
    prev = torch.zeros(P)
    S = torch.ones(P)
    c = proportional_cost(delta, prev, S, epsilon=0.001)
    expected = 0.001 * 0.5
    assert abs(c.mean().item() - expected) < 1e-5
    print(f"Costs OK | cost={c.mean():.6f}")


def test_losses():
    pnl = torch.randn(P)
    q = quadratic_loss(pnl)
    e = entropic_loss(pnl, lam=1.0)
    c = cvar_loss(pnl, alpha=0.05)
    assert q.item() > 0
    assert e.item() > -100
    assert c.item() > -100
    print(f"Losses OK | quad={q:.4f} entrop={e:.4f} cvar={c:.4f}")


def test_mlp_forward():
    model = MLPHedger()
    paths = simulate_gbm(P, N, T=1/12, device=DEVICE)
    S, v, t_grid = paths["S"], paths["v"], paths["t_grid"]
    init_deltas = torch.zeros(P, N)
    features = build_features(S, v, t_grid, init_deltas)
    # MLP: process step by step
    deltas_list = []
    for k in range(N):
        d = model(S[:, k], v[:, k], t_grid[k:k+1], torch.zeros(P))
        deltas_list.append(d)
    deltas = torch.stack(deltas_list, dim=1)
    assert deltas.shape == (P, N)
    print(f"MLP OK | delta mean={deltas.mean():.4f}")


def test_lstm_forward():
    model = LSTMHedger()
    paths = simulate_gbm(P, N, T=1/12, device=DEVICE)
    S, v, t_grid = paths["S"], paths["v"], paths["t_grid"]
    init_deltas = torch.zeros(P, N)
    features = build_features(S, v, t_grid, init_deltas)
    deltas = model(features)
    assert deltas.shape == (P, N)
    print(f"LSTM OK | delta mean={deltas.mean():.4f}")


def test_transformer_forward():
    model = TransformerHedger(d_model=64, n_heads=4, n_layers=2)
    paths = simulate_gbm(P, N, T=1/12, device=DEVICE)
    S, v, t_grid = paths["S"], paths["v"], paths["t_grid"]
    init_deltas = torch.zeros(P, N)
    features = build_features(S, v, t_grid, init_deltas)
    deltas = model(features)
    assert deltas.shape == (P, N)
    print(f"Transformer OK | delta mean={deltas.mean():.4f}")


def test_end_to_end_loss():
    """Full forward+backward pass through Transformer hedger."""
    model = TransformerHedger(d_model=32, n_heads=2, n_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    paths = simulate_gbm(200, N, T=1/12, device=DEVICE)
    S, v, t_grid = paths["S"], paths["v"], paths["t_grid"]
    init_deltas = torch.zeros(200, N)
    features = build_features(S, v, t_grid, init_deltas)
    deltas = model(features)
    pnl = compute_pnl(S, v, t_grid, deltas, lambda s: european_call(s, 1.0), None, 0.0)
    loss = quadratic_loss(pnl)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert not torch.isnan(loss), "NaN loss in e2e test"
    print(f"E2E backward OK | loss={loss.item():.4f}")


if __name__ == "__main__":
    tests = [
        test_rbergomi_paths,
        test_heston_paths,
        test_gbm_paths,
        test_gbm_bs_convergence,
        test_payoffs,
        test_costs,
        test_losses,
        test_mlp_forward,
        test_lstm_forward,
        test_transformer_forward,
        test_end_to_end_loss,
    ]
    passed = 0
    for t in tests:
        try:
            print(f"\n--- {t.__name__} ---")
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {e}")
    print(f"\n{'='*40}")
    print(f"Passed: {passed}/{len(tests)}")
