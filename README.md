# Deep Hedging with Transformer Policy under Rough Volatility

Extends [Buehler et al. 2019](https://arxiv.org/abs/1802.03042) Deep Hedging by:
1. **Causal Transformer** (RoPE) policy replacing LSTM
2. **Rough Bergomi** path simulator (Bennedsen–Lunde–Pakkanen hybrid scheme, H ≈ 0.1)
3. **CVaR risk measure** for tail-aware hedging
4. **Exotic payoffs**: digital, down-and-out barrier, autocall
5. **Real SPX OOS** evaluation with calibrated rBergomi parameters

MTech Data Science capstone project — runs on H100 GPU.

---

## Structure

```
deephedge/
  sim/          rBergomi, Heston, GBM simulators + payoffs + costs
  models/       BS-delta, MLP, LSTM, Transformer, Mamba hedgers
  train/        differentiable MC training loop, CVaR/entropic/quadratic losses
  calibration/  SPX/VIX data loaders, rBergomi parameter fitting
  notebooks/    experiment notebooks (01–04)
  tests/        sanity checks
scripts/
  train_all.sh  run all 6 experiments
  compare.py    load checkpoints → result tables + bootstrap CI
  calibrate.py  fit rBergomi to historical SPX
```

## Quickstart

```bash
pip install -e .

# Train transformer hedger (vanilla call, rBergomi, CVaR loss)
python -m deephedge.train.train \
    --model transformer --sim rbergomi --payoff call \
    --n_paths 500000 --epochs 300

# Run all experiments
bash scripts/train_all.sh

# Final comparison table
python scripts/compare.py --sim rbergomi --payoff call --plot
```

## Key results (after training)

| Hedger      | CVaR 5% (vanilla) | CVaR 5% (barrier) |
|-------------|------------------|------------------|
| BS-delta    | —                | —                |
| LSTM        | —                | —                |
| Transformer | —                | —                |

*Fill after running experiments.*

## References

- Buehler, Gonon, Teichmann, Wood (2019) — Deep Hedging
- Bayer, Friz, Gatheral (2016) — Pricing under rough volatility
- Bennedsen, Lunde, Pakkanen (2017) — Hybrid scheme for BSS processes
- Gatheral, Jaisson, Rosenbaum (2018) — Volatility is rough
- Rockafellar, Uryasev (2000) — CVaR optimization
