#!/usr/bin/env bash
set -euo pipefail # prevent if the code is wrong, stop immediately and print error message

DATA_DIR="${DATA_DIR:-/data/CPE_487-587/ACCDataset}"
K="${K:-10}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
DROPOUT="${DROPOUT:-0.2}"
ETA="${ETA:-0.001}"
EPOCH="${EPOCH:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
OPTIMIZER="${OPTIMIZER:-ADAM}"
LOSS_FN="${LOSS_FN:-dice}"
DEVICE="${DEVICE:-cuda}"
OUTDIR="${OUTDIR:-results_acc}"
KEYWORD="${KEYWORD:-hw03_q7}"
ONNX_NAME="${ONNX_NAME:-acc_model.onnx}"
SPLIT_RATIO="${SPLIT_RATIO:-0.8}"
RANDOM_SEED="${RANDOM_SEED:-42}"

echo "Running acc_impl.py"
python scripts/acc_impl.py \
  --data_dir "$DATA_DIR" \
  --k "$K" \
  --hidden_dim "$HIDDEN_DIM" \
  --dropout "$DROPOUT" \
  --eta "$ETA" \
  --epoch "$EPOCH" \
  --batch_size "$BATCH_SIZE" \
  --optimizer "$OPTIMIZER" \
  --loss "$LOSS_FN" \
  --device "$DEVICE" \
  --outdir "$OUTDIR" \
  --keyword "$KEYWORD" \
  --save_onnx \
  --onnx_name "$ONNX_NAME"
echo "Finishing acc_impl.py"