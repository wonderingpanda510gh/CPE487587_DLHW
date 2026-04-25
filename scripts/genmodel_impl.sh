#!/usr/bin/env bash
set -euo pipefail # prevent if the code is wrong, stop immediately and print error message

ZIP_PATH="${ZIP_PATH:-/data/CPE_487-587/img_align_celeba.zip}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-128}"
ONNX_INTERVAL="${ONNX_INTERVAL:-5}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
SAVE_DIR="${SAVE_DIR:-./results_genmodel}"
DEVICE="${DEVICE:-cuda}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
MODEL=("Diffusion")

mkdir -p "${SAVE_DIR}" # output directory
mkdir -p logs
LOG_FILE="logs/train_genmodel.log"

echo "Running genmodel_impl.py"
echo "Start Time: $(date)" | tee -a "$LOG_FILE"

for MODEL_TYPE in "${MODEL[@]}"; do
    echo "------------------------------------------------" | tee -a "$LOG_FILE"
    echo "Now Training: $MODEL_TYPE" | tee -a "$LOG_FILE"
    echo "------------------------------------------------" | tee -a "$LOG_FILE"
    
    python scripts/genmodel_impl.py \
        --zip_path "$ZIP_PATH" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --onnx_interval "$ONNX_INTERVAL" \
        --train_ratio "$TRAIN_RATIO" \
        --save_dir "$SAVE_DIR" \
        --device "$DEVICE" \
        --model_type "$MODEL_TYPE" \
        --learning_rate "$LEARNING_RATE" \
        2>&1 | tee -a "$LOG_FILE"
        
    echo "Finished $MODEL_TYPE at $(date)" | tee -a "$LOG_FILE"
done
echo "Finishing the all genmodel_impl.py"