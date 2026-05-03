# %% [markdown]
# # Notebook 2 — Baseline Hedgers
# Train MLP and LSTM hedgers, compare to BS-delta.
# Reproduce Buehler 2019 style results on Heston.
# Sanity check: zero-cost GBM + quadratic loss → hedger converges to BS delta.

# %%
import sys; sys.path.insert(0, "../..")
import torch, math, numpy as np, matplotlib.pyplot as plt
from deephedge.sim import simulate_gbm, simulate_heston
from deephedge.sim.payoffs import european_call
from deephedge.sim.costs import proportional_cost
from deephedge.models import MLPHedger, LSTMHedger
from deephedge.models.bs_delta import BSDeltaHedger
from deephedge.train.losses import quadratic_loss, cvar_loss
from deephedge.train.train import build_features, compute_pnl
from deephedge.train.eval import pnl_metrics, compare_hedgers

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# %%
# Hyperparams
P_TRAIN = 200_000
P_EVAL  = 100_000
N_STEPS = 30         # ~1 month daily hedging
T       = 1.0 / 12
K       = 1.0
EPSILON = 0.001
LR      = 3e-4
EPOCHS  = 300

SIM_CFG = dict(kappa=2.0, theta=0.04, sigma=0.5, rho=-0.7, v0=0.04)  # Heston

# %% [markdown]
# ## 1. Sanity: BS delta on GBM with zero cost

# %%
def bs_delta_rollout(S, v, t_grid, K, T):
    from deephedge.models.bs_delta import BSDeltaHedger
    bs = BSDeltaHedger(K=K, payoff_type="call")
    P_loc = S.shape[0]
    deltas = []
    for k in range(N_STEPS):
        t_rem = torch.tensor(T - k * T / N_STEPS, dtype=torch.float32).expand(P_loc)
        d = bs(S[:, k], v[:, k], t_rem, torch.zeros(P_loc, device=S.device))
        deltas.append(d)
    return torch.stack(deltas, dim=1)

paths_gbm = simulate_gbm(P_EVAL, N_STEPS, T=T, sigma=0.2, device=DEVICE)
S_g, v_g, tg = paths_gbm["S"], paths_gbm["v"], paths_gbm["t_grid"]

# BS delta, no cost
deltas_bs = bs_delta_rollout(S_g, v_g, tg, K, T)
pnl_bs_nocost = compute_pnl(S_g, v_g, tg, deltas_bs, lambda s: european_call(s, K), None, 0.0)
print(f"BS delta no-cost PnL std = {pnl_bs_nocost.std():.4f}  (should be very small)")

# Unhedged
pnl_unhedged = -european_call(S_g, K)
print(f"Unhedged PnL std         = {pnl_unhedged.std():.4f}")

# %% [markdown]
# ## 2. Train MLP on Heston

# %%
def train_loop(model, epochs, sim_cfg, loss_fn_name="cvar"):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=LR*0.01)
    loss_fn = cvar_loss if loss_fn_name == "cvar" else quadratic_loss
    log = []
    for ep in range(1, epochs + 1):
        paths = simulate_heston(P_TRAIN, N_STEPS, T=T, device=DEVICE, **sim_cfg)
        S, v, tg = paths["S"], paths["v"], paths["t_grid"]
        init_d = torch.zeros(P_TRAIN, N_STEPS, device=DEVICE)
        feat = build_features(S, v, tg, init_d)

        if isinstance(model, MLPHedger):
            deltas_list = []
            for k in range(N_STEPS):
                d = model(S[:, k], v[:, k], tg[k:k+1], torch.zeros(P_TRAIN, device=DEVICE))
                deltas_list.append(d)
            deltas = torch.stack(deltas_list, dim=1)
        else:
            deltas = model(feat)

        pnl = compute_pnl(S, v, tg, deltas, lambda s: european_call(s, K), proportional_cost, EPSILON)
        loss = loss_fn(pnl)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); sched.step()
        if ep % 50 == 0:
            print(f"  ep {ep}/{epochs}  loss={loss.item():.4f}  pnl_std={pnl.std():.4f}")
        log.append(loss.item())
    return log

print("=== Training MLP ===")
mlp = MLPHedger().to(DEVICE)
log_mlp = train_loop(mlp, EPOCHS, SIM_CFG)
torch.save(mlp.state_dict(), "../../checkpoints/mlp.pt")

print("\n=== Training LSTM ===")
lstm = LSTMHedger().to(DEVICE)
log_lstm = train_loop(lstm, EPOCHS, SIM_CFG)
torch.save(lstm.state_dict(), "../../checkpoints/lstm.pt")

# %% [markdown]
# ## 3. Evaluate on held-out Heston paths

# %%
def eval_model(model, is_mlp=False):
    model.eval()
    with torch.no_grad():
        paths = simulate_heston(P_EVAL, N_STEPS, T=T, device=DEVICE, **SIM_CFG)
        S, v, tg = paths["S"], paths["v"], paths["t_grid"]
        if is_mlp:
            deltas_list = []
            for k in range(N_STEPS):
                d = model(S[:, k], v[:, k], tg[k:k+1], torch.zeros(P_EVAL, device=DEVICE))
                deltas_list.append(d)
            deltas = torch.stack(deltas_list, dim=1)
        else:
            feat = build_features(S, v, tg, torch.zeros(P_EVAL, N_STEPS, device=DEVICE))
            deltas = model(feat)
        pnl = compute_pnl(S, v, tg, deltas, lambda s: european_call(s, K), proportional_cost, EPSILON)
    model.train()
    return pnl

pnl_mlp  = eval_model(mlp, is_mlp=True)
pnl_lstm = eval_model(lstm)

# BS on Heston
paths_h = simulate_heston(P_EVAL, N_STEPS, T=T, device=DEVICE, **SIM_CFG)
S_h, v_h, tg_h = paths_h["S"], paths_h["v"], paths_h["t_grid"]
deltas_bs_h = bs_delta_rollout(S_h, v_h, tg_h, K, T)
pnl_bs_h = compute_pnl(S_h, v_h, tg_h, deltas_bs_h, lambda s: european_call(s, K), proportional_cost, EPSILON)

compare_hedgers({"BS-delta": pnl_bs_h, "MLP": pnl_mlp, "LSTM": pnl_lstm})

# %% [markdown]
# ## 4. Training curves

# %%
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(log_mlp,  label="MLP")
ax.plot(log_lstm, label="LSTM")
ax.set_xlabel("Epoch"); ax.set_ylabel("CVaR Loss")
ax.set_title("Baseline Training Curves (Heston, vanilla call)")
ax.legend()
plt.tight_layout()
plt.savefig("../notebooks/figs/02_training_curves.png", dpi=150)
plt.show()
