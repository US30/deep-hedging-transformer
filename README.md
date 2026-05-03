# Deep Hedging with Transformer Policy under Rough Volatility

A research implementation extending [Buehler et al. (2019) "Deep Hedging"](https://arxiv.org/abs/1802.03042) with two contributions:

1. **Rough Bergomi simulator** — paths generated from the empirically validated rough volatility model ($H \approx 0.1$), instead of Heston/GBM
2. **Causal Transformer policy** (with RoPE) — replaces the LSTM hedger, capturing long-range path dependencies more effectively

---

## Problem Statement

A derivatives dealer sells an option (e.g., a 1-month ATM call on SPX) and must hedge the resulting exposure by dynamically trading the underlying. Under **Black-Scholes** assumptions (continuous trading, no costs, GBM dynamics), the hedge is the closed-form delta. In reality:

- Volatility is **not constant** — it follows a rough, mean-reverting process with Hurst exponent $H \approx 0.1$ (much rougher than standard Brownian motion)
- Trading is **discrete** — rebalancing happens once a day, not continuously
- Trading has **costs** — proportional bid-ask spread, typically 5–10 bps for liquid indices
- Payoffs can be **exotic** — path-dependent (barriers, autocalls) where BS-delta is undefined or blows up

**Deep Hedging** solves this as a reinforcement learning / stochastic control problem: train a neural network policy that minimises a **risk measure** (CVaR, entropic utility) of the net hedging P&L over a Monte Carlo ensemble of paths.

---

## What This Project Does

### 1. Path Simulation (`deephedge/sim/`)

Three simulators, all GPU-batched via PyTorch:

**Rough Bergomi (rBergomi)** — main simulator, captures empirical equity vol stylized facts:

$$v_t = \xi_0(t) \cdot \exp\!\left( \eta\sqrt{2H} \cdot W^H_t \;-\; \tfrac{1}{2}\eta^2 t^{2H} \right)$$

$$dS_t = \sqrt{v_t}\; S_t \left( \rho\, dW_t + \sqrt{1-\rho^2}\, dW^\perp_t \right)$$

where $W^H$ is fractional Brownian motion with Hurst exponent $H \in (0, 0.5)$.

| Parameter | Symbol | Default | Meaning |
|-----------|--------|---------|---------|
| Hurst exponent | $H$ | 0.1 | Roughness of vol path; $H = 0.5$ = Brownian, $H < 0.5$ = rough |
| Vol-of-vol | $\eta$ | 1.9 | Controls amplitude of vol fluctuations |
| Spot-vol correlation | $\rho$ | −0.9 | Leverage effect: spot down → vol up |
| Forward variance | $\xi_0$ | 0.04 | Initial variance level ($\approx 20\%$ ATM vol) |

Implemented via the **Bennedsen–Lunde–Pakkanen hybrid scheme** — exact covariance structure via FFT-based convolution with the Volterra kernel:

$$g(j) = \frac{(j+1)^{\alpha} - j^{\alpha}}{\alpha}, \quad \alpha = H - \tfrac{1}{2}$$

**Heston** — classical affine stochastic vol, used to reproduce Buehler 2019 benchmark numbers:

$$dv_t = \kappa(\theta - v_t)\,dt + \sigma\sqrt{v_t}\,dW^v_t$$

where $\kappa$ = mean-reversion speed, $\theta$ = long-run variance, $\sigma$ = vol-of-vol.

**GBM** — constant vol $\sigma$ baseline for sanity checks (BS-delta should be near-optimal here).

---

### 2. Hedger Architectures (`deephedge/models/`)

At each time step $k$, the hedger observes state $(\log S_k,\, \log v_k,\, t_k,\, \delta_{k-1})$ and outputs hedge ratio $\delta_k$ (shares to hold).

| Model | Architecture | Memory | Params |
|-------|-------------|--------|--------|
| **BS-delta** | Closed-form Black-Scholes | None | 0 |
| **MLP** | 3-layer feedforward, Tanh | None | ~8K |
| **LSTM** | 2-layer LSTM (Buehler 2019) | Hidden state | ~50K |
| **Transformer** ⭐ | Causal decoder, 4L × 128d, RoPE | Full path attention | ~800K |
| **Mamba** | SSM (selective state space) | $O(N)$ recurrence | ~500K |

The **Transformer** is the main contribution. It uses:
- **Causal (masked) self-attention** — at step $k$ attends only to steps $0 \ldots k$ (no lookahead)
- **RoPE positional encoding** — relative position encoding, better than sinusoidal for financial time series
- **PyTorch `scaled_dot_product_attention`** — uses Flash Attention on H100 automatically via `is_causal=True`

The Transformer's ability to attend to the entire path history (not just the most recent hidden state) is expected to be especially advantageous for **path-dependent exotic payoffs** like barriers and autocalls.

---

### 3. Training Objective (`deephedge/train/`)

Net P&L of the hedging strategy over one episode:

$$\text{PnL}_T = -\text{Payoff}(S_T) + \sum_{k=0}^{N-1} \delta_k \,(S_{k+1} - S_k) - \sum_{k=0}^{N-1} \varepsilon\, S_k\, |\delta_k - \delta_{k-1}|$$

The three terms are: **short option** position, **delta hedge gain**, **transaction costs** ($\varepsilon = 10$ bps).

Three **risk measures** (all differentiable — backprop flows through the Monte Carlo expectation):

| Loss | Formula | Use |
|------|---------|-----|
| **Quadratic** | $\mathbb{E}[\text{PnL}^2]$ | Variance minimisation, fast convergence |
| **Entropic** | $\frac{1}{\lambda}\log \mathbb{E}\!\left[e^{-\lambda\,\text{PnL}}\right]$ | Exponential utility; $\lambda$ = risk aversion |
| **CVaR** ⭐ | $\mathbb{E}\left[-\text{PnL} \mid -\text{PnL} \geq \text{VaR}_\alpha\right]$ | Tail risk; $\alpha = 5\%$ default |

CVaR (Conditional Value at Risk) is the primary metric — it penalises the worst $\alpha$-fraction of outcomes, directly relevant to what a risk manager monitors.

---

### 4. Payoffs (`deephedge/sim/payoffs.py`)

| Payoff | Formula | Challenge for BS-delta |
|--------|---------|----------------------|
| European Call | $\max(S_T - K,\; 0)$ | Works under GBM, breaks under rough vol |
| European Put | $\max(K - S_T,\; 0)$ | Same |
| Digital Call | $\mathbf{1}_{\{S_T > K\}}$ | $\Delta \to \infty$ near $K$ at expiry |
| Down-and-Out Barrier | $\max(S_T - K, 0)\cdot\mathbf{1}_{\{\min_t S_t > B\}}$ | Path-dependent; BS-delta ignores path history |
| Autocall | Redeem at $1 + c \cdot t$ if $S_t \geq B$ at annual observation | Multi-period path dependency |

where $K$ = strike, $B$ = barrier level, $c$ = coupon rate.

---

### 5. Calibration (`deephedge/calibration/`)

Fits rBergomi parameters $(H,\, \eta,\, \rho,\, \xi_0)$ to real SPX data:

**Step 1 — Estimate $H$** from the structure function of realised vol. For fractional processes:

$$m(q, \Delta) = \mathbb{E}\!\left[\left|\log v_{t+\Delta} - \log v_t\right|^q\right] \;\sim\; C_q \cdot \Delta^{qH}$$

Slope of $\log m(q,\Delta)$ vs $\log \Delta$ gives $qH$. Average over $q = 1, 2$ for robustness. (Gatheral–Jaisson–Rosenbaum 2018)

**Step 2 — Set $\xi_0$** from recent 1-year realised variance average.

**Step 3 — Calibrate $(\eta, \rho)$** by matching Monte Carlo ATM implied vol to observed VIX level via Nelder-Mead optimisation.

---

### 6. Out-of-Sample Evaluation (`deephedge/notebooks/04_real_data.py`)

- **Training period**: 2000–2021 — calibration + model training on simulated paths
- **OOS period**: 2022–present — real SPX daily returns
- **Evaluation**: roll 1-month hedging windows over real SPX; compare $\text{CVaR}_{5}$ with **bootstrap 95% confidence intervals** ($n = 2000$ resamples)
- **Stress tests**: 2020 COVID crash, 2022 rate-hike drawdown

---

## Experiments

| # | Experiment | Simulator | Payoff | Primary finding |
|---|-----------|-----------|--------|----------------|
| 1 | Reproduce Buehler 2019 | Heston | Vanilla call | LSTM baseline validation |
| 2 | rBergomi main result | rBergomi | Vanilla call | Transformer vs LSTM under rough vol |
| 3 | Exotic payoffs | rBergomi | Digital, Barrier | Transformer wins by larger margin |
| 4 | Ablations | rBergomi | Vanilla call | depth / width / attention window |
| 5 | Real SPX OOS | Real + calibrated rBergomi | Vanilla call | Generalisation test |
| 6 | Stress periods | Real | Vanilla call | COVID crash, 2022 drawdown |

---

## Project Structure

```
deephedge/
├── sim/
│   ├── rbergomi.py          Rough Bergomi (hybrid scheme), Heston, GBM — GPU batched
│   ├── payoffs.py           Vanilla, digital, barrier, autocall payoff functions
│   ├── costs.py             Proportional + fixed transaction cost models
│   └── __init__.py
├── models/
│   ├── bs_delta.py          Black-Scholes delta (closed-form, scipy)
│   ├── mlp_hedger.py        Per-step MLP — no temporal memory
│   ├── lstm_hedger.py       2-layer LSTM (Buehler 2019 baseline)
│   ├── transformer_hedger.py  Causal Transformer + RoPE  ← main contribution
│   ├── mamba_hedger.py      Mamba SSM (week-14 ablation)
│   └── __init__.py
├── train/
│   ├── losses.py            Quadratic, entropic, CVaR risk measures
│   ├── train.py             CLI training loop (differentiable Monte Carlo)
│   └── eval.py              Metrics, bootstrap CI, PnL distribution plots
├── calibration/
│   ├── data.py              SPX/VIX data loaders (yfinance, FRED), roughness estimation
│   └── rbergomi_fit.py      Parameter fitting: H from structure function, (η, ρ) from IV match
├── notebooks/
│   ├── 01_paths_sanity.py   Verify rBergomi stylized facts, H estimation, vol term structure
│   ├── 02_baselines.py      Train MLP + LSTM on Heston, compare to BS-delta
│   ├── 03_transformer.py    Main experiments: all payoffs, ablations, result tables
│   └── 04_real_data.py      Calibration + OOS SPX evaluation + stress tests
└── tests/
    └── test_sanity.py       Unit tests: shapes, positivity, BS convergence, forward pass

scripts/
├── train_all.sh             Run all 6 training experiments in sequence
├── compare.py               Load checkpoints → final tables + bootstrap CI
└── calibrate.py             Fit rBergomi to SPX, save params.json

setup.py
requirements.txt
```

---

## Skill Mapping

| Data Science | Quant Finance |
|-------------|--------------|
| Transformer / Mamba sequence models | Stochastic volatility (Heston, rough Bergomi) |
| Differentiable simulation, autodiff training | Derivatives hedging, Greeks, delta neutrality |
| GPU-batched Monte Carlo (PyTorch) | Risk measures: CVaR, entropic utility |
| Calibration via Nelder-Mead / gradient descent | Vol surface, VIX term structure |
| Bootstrap confidence intervals, OOS evaluation | PnL attribution, transaction cost modelling |
| Ablation studies, hyperparameter search | Exotic payoffs: barrier, autocall |

---

## References

- Buehler, Gonon, Teichmann, Wood (2019). *Deep Hedging*. Quantitative Finance.
- Bayer, Friz, Gatheral (2016). *Pricing under rough volatility*. Quantitative Finance.
- Bennedsen, Lunde, Pakkanen (2017). *Hybrid scheme for Brownian semistationary processes*.
- Gatheral, Jaisson, Rosenbaum (2018). *Volatility is rough*. Quantitative Finance.
- Horvath, Muguruza, Tomas (2021). *Deep learning volatility*.
- Rockafellar, Uryasev (2000). *Optimization of Conditional Value-at-Risk*.
- Gu & Dao (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*.
