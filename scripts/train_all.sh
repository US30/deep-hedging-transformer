#!/usr/bin/env bash
# Train all hedger variants in sequence.
# Run from repo root:  bash scripts/train_all.sh
# Expected total runtime on H100 40GB: ~8-12 hours for all experiments.

set -e
cd "$(dirname "$0")/.."

mkdir -p checkpoints logs

COMMON="--n_paths 500000 --n_steps 30 --T 0.0833 --K 1.0 --epsilon 0.001 \
        --epochs 300 --log_every 10 --loss cvar --alpha 0.05 \
        --H 0.1 --eta 1.9 --rho -0.9 --xi0 0.04"

echo "=== [1/6] MLP  | rBergomi | vanilla call ==="
python -m deephedge.train.train $COMMON \
    --model mlp --sim rbergomi --payoff call \
    --save_dir checkpoints/mlp_rbergomi_call \
    2>&1 | tee logs/mlp_rbergomi_call.log

echo "=== [2/6] LSTM | rBergomi | vanilla call ==="
python -m deephedge.train.train $COMMON \
    --model lstm --sim rbergomi --payoff call \
    --save_dir checkpoints/lstm_rbergomi_call \
    2>&1 | tee logs/lstm_rbergomi_call.log

echo "=== [3/6] Transformer | rBergomi | vanilla call ==="
python -m deephedge.train.train $COMMON \
    --model transformer --d_model 128 --n_heads 4 --n_layers 4 \
    --sim rbergomi --payoff call \
    --save_dir checkpoints/tfm_rbergomi_call \
    2>&1 | tee logs/tfm_rbergomi_call.log

echo "=== [4/6] Transformer | rBergomi | digital call ==="
python -m deephedge.train.train $COMMON \
    --model transformer --d_model 128 --n_heads 4 --n_layers 4 \
    --sim rbergomi --payoff digital \
    --save_dir checkpoints/tfm_rbergomi_digital \
    2>&1 | tee logs/tfm_rbergomi_digital.log

echo "=== [5/6] Transformer | rBergomi | barrier call ==="
python -m deephedge.train.train $COMMON \
    --model transformer --d_model 128 --n_heads 4 --n_layers 4 \
    --sim rbergomi --payoff barrier --barrier 0.8 \
    --save_dir checkpoints/tfm_rbergomi_barrier \
    2>&1 | tee logs/tfm_rbergomi_barrier.log

echo "=== [6/6] Transformer | Heston | vanilla call (reproduce Buehler 2019) ==="
python -m deephedge.train.train \
    --n_paths 500000 --n_steps 30 --T 0.0833 --K 1.0 --epsilon 0.001 \
    --epochs 300 --log_every 10 --loss cvar --alpha 0.05 \
    --model transformer --d_model 128 --n_heads 4 --n_layers 4 \
    --sim heston --payoff call \
    --save_dir checkpoints/tfm_heston_call \
    2>&1 | tee logs/tfm_heston_call.log

echo "All training complete. Checkpoints in ./checkpoints/"
