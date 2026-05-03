# %% [markdown]
# # Notebook 1 — Path Simulator Sanity Checks
# Verify rBergomi, Heston, GBM paths reproduce stylized facts:
# 1. Spot prices > 0, S0 ≈ 1
# 2. Variance positive, mean ≈ xi0
# 3. rBergomi ATM vol skew term-structure ∝ T^(H - 0.5)
# 4. Realized roughness exponent H ≈ 0.1

# %%
import sys; sys.path.insert(0, "../..")
import torch, math, numpy as np, matplotlib.pyplot as plt
from deephedge.sim import simulate_rbergomi, simulate_heston, simulate_gbm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# %% [markdown]
# ## 1. Shape and positivity

# %%
P, N = 100_000, 252
paths_rb = simulate_rbergomi(P, N, T=1.0, H=0.1, eta=1.9, rho=-0.9, xi0=0.04, device=DEVICE)
S_rb, v_rb = paths_rb["S"], paths_rb["v"]

print(f"S shape: {S_rb.shape}")
print(f"S0 mean: {S_rb[:,0].mean():.4f} (expect 1.0)")
print(f"v0 mean: {v_rb[:,0].mean():.6f} (expect 0.04)")
print(f"S > 0: {(S_rb > 0).all()}")
print(f"v > 0: {(v_rb > 0).all()}")

# %% [markdown]
# ## 2. Vol-of-vol structure function — estimate H

# %%
# log v increments: slope of E[|log v(t+Δ) - log v(t)|^q] vs Δ should give q*H
log_v = v_rb.cpu().float().log()  # (P, N+1)

lags = [1, 2, 5, 10, 21, 42, 63]
q_vals = [1, 2]
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

for q in q_vals:
    m_q = []
    for lag in lags:
        diff = (log_v[:, lag:] - log_v[:, :-lag]).abs().pow(q).mean().item()
        m_q.append(diff)
    log_lags = np.log(lags)
    log_mq = np.log(m_q)
    coef = np.polyfit(log_lags, log_mq, 1)
    H_est = coef[0] / q
    ax.plot(log_lags, log_mq, "o-", label=f"q={q}, H_est={H_est:.3f}")

ax.set_xlabel("log(Δ)")
ax.set_ylabel("log m(q,Δ)")
ax.set_title("Structure Function Scaling (rBergomi H=0.1)")
ax.legend()
plt.tight_layout()
plt.savefig("../notebooks/figs/01_structure_function.png", dpi=150)
plt.show()

# %% [markdown]
# ## 3. ATM vol skew term-structure: slope ∝ T^(H - 0.5)

# %%
from scipy.stats import norm as sp_norm

def atm_call_mc(S, K=1.0):
    return (S[:, -1] - K).clamp(min=0).cpu().float().mean().item()

def bs_iv(price, K, T, F=1.0):
    from scipy.optimize import brentq
    def bs(sig):
        if sig < 1e-5: return max(F - K, 0)
        d1 = (math.log(F/K) + 0.5*sig**2*T) / (sig*math.sqrt(T))
        d2 = d1 - sig*math.sqrt(T)
        return F*sp_norm.cdf(d1) - K*sp_norm.cdf(d2)
    try: return brentq(lambda s: bs(s) - price, 1e-5, 5.0)
    except: return float("nan")

maturities = [1/52, 1/12, 3/12, 6/12, 1.0]  # weeks to 1yr
ivs = []
for T in maturities:
    pp = simulate_rbergomi(50_000, max(5, int(T*252)), T=T, H=0.1, eta=1.9, rho=-0.9, xi0=0.04, device=DEVICE)
    price = atm_call_mc(pp["S"])
    iv = bs_iv(price, K=1.0, T=T)
    ivs.append(iv)
    print(f"T={T:.4f}  ATM IV={iv:.4f}")

# Plot: log(IV) vs log(T) — slope ≈ H - 0.5 = -0.4
log_T = np.log(maturities)
log_IV = np.log(ivs)
coef = np.polyfit(log_T, log_IV, 1)
print(f"\nSlope of log(IV) vs log(T) = {coef[0]:.3f}  (expect ~{0.1 - 0.5:.3f})")

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(log_T, log_IV, "o-", label="MC IV")
ax.plot(log_T, np.polyval(coef, log_T), "--", label=f"slope={coef[0]:.3f}")
ax.set_xlabel("log(T)")
ax.set_ylabel("log(ATM IV)")
ax.set_title("Vol Term Structure Scaling")
ax.legend()
plt.tight_layout()
plt.savefig("../notebooks/figs/01_vol_termstruct.png", dpi=150)
plt.show()

# %% [markdown]
# ## 4. Compare simulators side by side

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
titles = ["rBergomi (H=0.1)", "Heston", "GBM"]
sims = [
    simulate_rbergomi(5_000, 252, T=1.0, device=DEVICE),
    simulate_heston(5_000, 252, T=1.0, device=DEVICE),
    simulate_gbm(5_000, 252, T=1.0, sigma=0.2, device=DEVICE),
]

for ax, paths, title in zip(axes, sims, titles):
    S = paths["S"][:20].cpu().float().numpy()  # 20 sample paths
    ax.plot(S.T, alpha=0.4, lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("Day")
    ax.set_ylabel("S_t")

plt.tight_layout()
plt.savefig("../notebooks/figs/01_sample_paths.png", dpi=150)
plt.show()
