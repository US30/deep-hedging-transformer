# %% [markdown]
# # Notebook 4 — Real SPX Out-of-Sample Evaluation
# 1. Estimate H and calibrate rBergomi to SPX/VIX history
# 2. Roll a 1-month ATM call hedging program on real SPX (2020–present)
# 3. Stress-test: 2020 COVID crash, 2022 drawdown
# 4. Compare BS-delta vs LSTM vs Transformer on real paths

# %%
import sys; sys.path.insert(0, "../..")
import torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from deephedge.calibration.data import load_spx, compute_realized_vol, realized_roughness
from deephedge.calibration.rbergomi_fit import estimate_H, calibrate
from deephedge.sim import simulate_rbergomi
from deephedge.sim.payoffs import european_call
from deephedge.sim.costs import proportional_cost
from deephedge.models import LSTMHedger, TransformerHedger
from deephedge.models.bs_delta import BSDeltaHedger
from deephedge.train.train import build_features, compute_pnl
from deephedge.train.eval import pnl_metrics, compare_hedgers, bootstrap_cvar

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## 1. Load SPX and estimate roughness

# %%
print("Loading SPX data...")
spx = load_spx(start="2000-01-01")
log_ret = np.log(spx / spx.shift(1)).dropna()
rv = compute_realized_vol(spx)

print(f"SPX loaded: {len(spx)} days  ({spx.index[0].date()} – {spx.index[-1].date()})")
print(f"Mean realized vol: {rv.mean():.4f}")

# Structure function H estimate
results = realized_roughness(spx)
for q, v in results.items():
    print(f"q={q}: H_est = {v['H_estimate']:.4f}")

# %% [markdown]
# ## 2. Calibrate rBergomi parameters

# %%
# Use training period: 2000-01-01 to 2021-12-31; hold out 2022-present for OOS
TRAIN_END = "2021-12-31"
spx_train = spx[:TRAIN_END]

print("Calibrating rBergomi to training period...")
params = calibrate(spx_train, n_paths=100_000, n_steps=252, device=DEVICE)
print(f"Calibrated params: {params}")

# %% [markdown]
# ## 3. Load pre-trained hedger models
# (trained in notebook 03 or via train.py script)

# %%
N_STEPS = 21   # ~1 month of trading days
T_MONTH = 21 / 252
K = 1.0
EPSILON = 0.001
P_EVAL = 200_000

# Reload models
lstm_model = LSTMHedger().to(DEVICE)
lstm_model.load_state_dict(torch.load("../../checkpoints/lstm_rbergomi_call.pt", map_location=DEVICE))
lstm_model.eval()

tfm_model = TransformerHedger(d_model=128, n_heads=4, n_layers=4).to(DEVICE)
tfm_model.load_state_dict(torch.load("../../checkpoints/transformer_rbergomi_call.pt", map_location=DEVICE))
tfm_model.eval()

bs_hedger = BSDeltaHedger(K=K, payoff_type="call")
print("Models loaded.")

# %% [markdown]
# ## 4. Real-path rolling hedging simulation
# Construct "real" test paths: take consecutive 21-day windows from SPX OOS period.
# Normalise each window by starting price. Compare hedger PnL across windows.

# %%
spx_oos = spx["2022-01-01":]
log_ret_oos = np.log(spx_oos / spx_oos.shift(1)).dropna().values

def build_real_paths(log_returns: np.ndarray, n_steps: int = 21) -> torch.Tensor:
    """Slide a window of n_steps over log_returns. Returns (n_windows, n_steps+1) spot paths."""
    windows = []
    for i in range(len(log_returns) - n_steps):
        increments = log_returns[i: i + n_steps]
        log_path = np.concatenate([[0.0], np.cumsum(increments)])
        windows.append(np.exp(log_path))   # normalised: S_0 = 1
    return torch.tensor(np.stack(windows), dtype=torch.float32, device=DEVICE)

S_real = build_real_paths(log_ret_oos, N_STEPS)
print(f"Real OOS paths: {S_real.shape}  ({S_real.shape[0]} windows)")

# Use calibrated params vol as proxy for v (constant per window = recent RV)
rv_oos = compute_realized_vol(spx_oos, window=21).values
v_scalar = float(np.nanmean(rv_oos[:len(S_real)]) ** 2)
v_real = torch.full_like(S_real, v_scalar)

t_grid_real = torch.linspace(0, T_MONTH, N_STEPS + 1, device=DEVICE)
payoff_fn = lambda s: european_call(s, K)

def hedger_pnl(model_fn, S, v, tg, is_bs=False):
    n = S.shape[0]
    if is_bs:
        deltas_list = []
        for k in range(N_STEPS):
            t_rem = torch.full((n,), T_MONTH - k * T_MONTH / N_STEPS, device=DEVICE)
            d = bs_hedger(S[:, k], v[:, k], t_rem, torch.zeros(n, device=DEVICE))
            deltas_list.append(d)
        deltas = torch.stack(deltas_list, dim=1)
    else:
        feat = build_features(S, v, tg, torch.zeros(n, N_STEPS, device=DEVICE))
        with torch.no_grad():
            deltas = model_fn(feat)
    return compute_pnl(S, v, tg, deltas, payoff_fn, proportional_cost, EPSILON)

pnl_bs_real   = hedger_pnl(None,       S_real, v_real, t_grid_real, is_bs=True)
pnl_lstm_real = hedger_pnl(lstm_model,  S_real, v_real, t_grid_real)
pnl_tfm_real  = hedger_pnl(tfm_model,  S_real, v_real, t_grid_real)

print("\n--- OOS Real SPX (2022–present) ---")
compare_hedgers({"BS-delta": pnl_bs_real, "LSTM": pnl_lstm_real, "Transformer": pnl_tfm_real})

# Bootstrap CIs
for name, pnl in [("BS-delta", pnl_bs_real), ("LSTM", pnl_lstm_real), ("Transformer", pnl_tfm_real)]:
    mean, lo, hi = bootstrap_cvar(pnl, n_boot=2000)
    print(f"{name:<20} CVaR 5%: {mean:.4f}  95% CI: [{lo:.4f}, {hi:.4f}]")

# %% [markdown]
# ## 5. Stress period: 2020 COVID crash

# %%
spx_covid = spx["2020-01-01":"2020-06-30"]
log_ret_covid = np.log(spx_covid / spx_covid.shift(1)).dropna().values

S_covid = build_real_paths(log_ret_covid, N_STEPS)
if len(S_covid) > 10:
    v_covid = torch.full_like(S_covid, float(compute_realized_vol(spx_covid).mean() ** 2))
    pnl_bs_c   = hedger_pnl(None,      S_covid, v_covid, t_grid_real, is_bs=True)
    pnl_lstm_c = hedger_pnl(lstm_model, S_covid, v_covid, t_grid_real)
    pnl_tfm_c  = hedger_pnl(tfm_model,  S_covid, v_covid, t_grid_real)
    print("\n--- Stress: 2020 COVID Crash ---")
    compare_hedgers({"BS-delta": pnl_bs_c, "LSTM": pnl_lstm_c, "Transformer": pnl_tfm_c})

# %% [markdown]
# ## 6. Synthetic vs Real: rBergomi with calibrated params

# %%
print("\nEvaluating with calibrated rBergomi params...")
calib_paths = simulate_rbergomi(
    P_EVAL, N_STEPS, T=T_MONTH, device=DEVICE,
    H=params["H"], eta=params["eta"], rho=params["rho"], xi0=params["xi0"]
)
S_c, v_c, tg_c = calib_paths["S"], calib_paths["v"], calib_paths["t_grid"]

pnl_bs_sc   = hedger_pnl(None,       S_c, v_c, tg_c, is_bs=True)
pnl_lstm_sc = hedger_pnl(lstm_model,  S_c, v_c, tg_c)
pnl_tfm_sc  = hedger_pnl(tfm_model,  S_c, v_c, tg_c)

print("\n--- Calibrated rBergomi OOS ---")
compare_hedgers({"BS-delta": pnl_bs_sc, "LSTM": pnl_lstm_sc, "Transformer": pnl_tfm_sc})

# %% [markdown]
# ## Summary plot

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (title, results) in zip(axes, [
    ("Real SPX OOS (2022–)", {"BS-delta": pnl_bs_real, "LSTM": pnl_lstm_real, "Transformer": pnl_tfm_real}),
    ("Calibrated rBergomi OOS", {"BS-delta": pnl_bs_sc, "LSTM": pnl_lstm_sc, "Transformer": pnl_tfm_sc}),
]):
    for name, pnl in results.items():
        pnl_np = pnl.cpu().float().numpy()
        ax.hist(pnl_np, bins=100, alpha=0.5, density=True, label=name)
    ax.set_title(title); ax.set_xlabel("PnL"); ax.legend()

plt.tight_layout()
plt.savefig("../notebooks/figs/04_real_data_pnl.png", dpi=150)
plt.show()
