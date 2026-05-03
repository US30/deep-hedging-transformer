# %% [markdown]
# # Notebook 3 — Transformer Hedger
# Main contribution: causal Transformer with RoPE vs LSTM vs BS-delta.
# Experiments:
#   1. Vanilla call under rBergomi (main result table)
#   2. Digital call  (BS blows up near K)
#   3. Down-and-out barrier (path-dependent, sequence model wins)
#   4. Ablations: depth, d_model, n_heads

# %%
import sys; sys.path.insert(0, "../..")
import torch, numpy as np, matplotlib.pyplot as plt
from deephedge.sim import simulate_rbergomi
from deephedge.sim.payoffs import european_call, digital_call, down_and_out_call
from deephedge.sim.costs import proportional_cost
from deephedge.models import LSTMHedger, TransformerHedger
from deephedge.models.bs_delta import BSDeltaHedger
from deephedge.train.losses import cvar_loss
from deephedge.train.train import build_features, compute_pnl
from deephedge.train.eval import pnl_metrics, compare_hedgers, plot_pnl_distributions, bootstrap_cvar
import os; os.makedirs("../../checkpoints", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

P_TRAIN = 500_000   # use full H100 capacity
P_EVAL  = 200_000
N_STEPS = 30
T       = 1.0 / 12
K       = 1.0
EPSILON = 0.001
LR      = 3e-4
EPOCHS  = 300

RB_CFG = dict(H=0.1, eta=1.9, rho=-0.9, xi0=0.04)

# %%
def train_model(model, payoff_fn, epochs=EPOCHS, label=""):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=LR*0.01)
    log = []
    for ep in range(1, epochs + 1):
        paths = simulate_rbergomi(P_TRAIN, N_STEPS, T=T, device=DEVICE, **RB_CFG)
        S, v, tg = paths["S"], paths["v"], paths["t_grid"]
        feat = build_features(S, v, tg, torch.zeros(P_TRAIN, N_STEPS, device=DEVICE))
        deltas = model(feat)
        pnl = compute_pnl(S, v, tg, deltas, payoff_fn, proportional_cost, EPSILON)
        loss = cvar_loss(pnl)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); sched.step()
        log.append(loss.item())
        if ep % 50 == 0:
            print(f"  [{label}] ep {ep}/{epochs}  CVaR={loss.item():.4f}")
    return log

def eval_model(model, payoff_fn):
    model.eval()
    with torch.no_grad():
        paths = simulate_rbergomi(P_EVAL, N_STEPS, T=T, device=DEVICE, **RB_CFG)
        S, v, tg = paths["S"], paths["v"], paths["t_grid"]
        feat = build_features(S, v, tg, torch.zeros(P_EVAL, N_STEPS, device=DEVICE))
        deltas = model(feat)
        pnl = compute_pnl(S, v, tg, deltas, payoff_fn, proportional_cost, EPSILON)
    model.train()
    return pnl

def bs_eval(payoff_fn, payoff_type="call"):
    bs = BSDeltaHedger(K=K, payoff_type=payoff_type)
    with torch.no_grad():
        paths = simulate_rbergomi(P_EVAL, N_STEPS, T=T, device=DEVICE, **RB_CFG)
        S, v, tg = paths["S"], paths["v"], paths["t_grid"]
        deltas_list = []
        for k in range(N_STEPS):
            t_rem = torch.full((P_EVAL,), T - k * T / N_STEPS, device=DEVICE)
            deltas_list.append(bs(S[:, k], v[:, k], t_rem, torch.zeros(P_EVAL, device=DEVICE)))
        deltas = torch.stack(deltas_list, dim=1)
        pnl = compute_pnl(S, v, tg, deltas, payoff_fn, proportional_cost, EPSILON)
    return pnl

# %% [markdown]
# ## Experiment 1: Vanilla Call under rBergomi

# %%
payoff_call = lambda s: european_call(s, K)

print("Training LSTM (vanilla call, rBergomi)")
lstm_call = LSTMHedger().to(DEVICE)
log_lstm_call = train_model(lstm_call, payoff_call, label="LSTM-call")
torch.save(lstm_call.state_dict(), "../../checkpoints/lstm_rbergomi_call.pt")

print("\nTraining Transformer (vanilla call, rBergomi)")
tfm_call = TransformerHedger(d_model=128, n_heads=4, n_layers=4).to(DEVICE)
log_tfm_call = train_model(tfm_call, payoff_call, label="TFM-call")
torch.save(tfm_call.state_dict(), "../../checkpoints/transformer_rbergomi_call.pt")

# Evaluate
pnl_bs    = bs_eval(payoff_call, "call")
pnl_lstm_c = eval_model(lstm_call,  payoff_call)
pnl_tfm_c  = eval_model(tfm_call,   payoff_call)

print("\n--- Vanilla Call Results (rBergomi) ---")
compare_hedgers({"BS-delta": pnl_bs, "LSTM": pnl_lstm_c, "Transformer": pnl_tfm_c})

# Bootstrap CVaR CIs
for name, pnl in [("BS-delta", pnl_bs), ("LSTM", pnl_lstm_c), ("Transformer", pnl_tfm_c)]:
    mean, lo, hi = bootstrap_cvar(pnl)
    print(f"{name:<20} CVaR 5%: {mean:.4f}  95% CI: [{lo:.4f}, {hi:.4f}]")

# %% [markdown]
# ## Experiment 2: Digital Call

# %%
payoff_dig = lambda s: digital_call(s, K)

print("\nTraining LSTM (digital call)")
lstm_dig = LSTMHedger().to(DEVICE)
train_model(lstm_dig, payoff_dig, label="LSTM-digital")

print("Training Transformer (digital call)")
tfm_dig = TransformerHedger(d_model=128, n_heads=4, n_layers=4).to(DEVICE)
train_model(tfm_dig, payoff_dig, label="TFM-digital")

pnl_bs_d   = bs_eval(payoff_dig, "call")  # BS delta used as proxy (misspecified)
pnl_lstm_d = eval_model(lstm_dig, payoff_dig)
pnl_tfm_d  = eval_model(tfm_dig,  payoff_dig)

print("\n--- Digital Call Results ---")
compare_hedgers({"BS-delta (misspec)": pnl_bs_d, "LSTM": pnl_lstm_d, "Transformer": pnl_tfm_d})

# %% [markdown]
# ## Experiment 3: Down-and-Out Barrier Call

# %%
payoff_bar = lambda s: down_and_out_call(s, K, B=0.8)

print("\nTraining LSTM (barrier)")
lstm_bar = LSTMHedger(hidden_dim=128).to(DEVICE)
train_model(lstm_bar, payoff_bar, label="LSTM-barrier")

print("Training Transformer (barrier)")
tfm_bar = TransformerHedger(d_model=128, n_heads=4, n_layers=4).to(DEVICE)
train_model(tfm_bar, payoff_bar, label="TFM-barrier")

pnl_bs_b   = bs_eval(payoff_bar, "call")
pnl_lstm_b = eval_model(lstm_bar, payoff_bar)
pnl_tfm_b  = eval_model(tfm_bar,  payoff_bar)

print("\n--- Barrier Call Results ---")
compare_hedgers({"BS-delta (misspec)": pnl_bs_b, "LSTM": pnl_lstm_b, "Transformer": pnl_tfm_b})

# %% [markdown]
# ## Experiment 4: Ablations (Transformer depth / width)

# %%
ablation_configs = [
    dict(d_model=64,  n_heads=2, n_layers=2, label="TFM-S"),
    dict(d_model=128, n_heads=4, n_layers=4, label="TFM-M"),
    dict(d_model=256, n_heads=8, n_layers=6, label="TFM-L"),
]

ablation_results = {}
for cfg in ablation_configs:
    label = cfg.pop("label")
    m = TransformerHedger(**cfg).to(DEVICE)
    n_p = sum(p.numel() for p in m.parameters())
    print(f"\nTraining {label} ({n_p:,} params)")
    train_model(m, payoff_call, epochs=200, label=label)
    ablation_results[f"{label} ({n_p//1000}K)"] = eval_model(m, payoff_call)

compare_hedgers(ablation_results)

# %% [markdown]
# ## Summary Plots

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
pairs = [
    ("Vanilla Call", {"BS-delta": pnl_bs, "LSTM": pnl_lstm_c, "Transformer": pnl_tfm_c}),
    ("Digital Call", {"BS (misspec)": pnl_bs_d, "LSTM": pnl_lstm_d, "Transformer": pnl_tfm_d}),
    ("Barrier Call", {"BS (misspec)": pnl_bs_b, "LSTM": pnl_lstm_b, "Transformer": pnl_tfm_b}),
]
for ax, (title, results) in zip(axes, pairs):
    for name, pnl in results.items():
        pnl_np = pnl.cpu().float().numpy()
        ax.hist(pnl_np, bins=100, alpha=0.5, density=True, label=name)
    ax.set_title(title); ax.set_xlabel("PnL"); ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig("../notebooks/figs/03_pnl_distributions.png", dpi=150)
plt.show()
