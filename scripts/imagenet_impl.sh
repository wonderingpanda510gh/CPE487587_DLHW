#!/usr/bin/env bash
set -euo pipefail # prevent if the code is wrong, stop immediately and print error message


KEYWORD="hw03"
ETA="${ETA:-0.01}"
EPOCH="${EPOCH:-10000}"
BATCH_SIZE="${BATCH_SIZE:-64}" 
INPUT_CHANNELS="${INPUT_CHANNELS:-3}"
OPTIMIZER="${OPTIMIZER:-SGD}"
DEVICE="${DEVICE:-cuda}"
OUTDIR="${OUTDIR:-results}"
ONNX_NAME="${ONNX_NAME:-imagenet_model.onnx}"

mkdir -p "${OUTDIR}" # output directory

echo "Running imagenet_impl.py"
python scripts/imagenet_impl.py \
    --eta "${ETA}" \
    --epoch "${EPOCH}" \
    --batch_size "${BATCH_SIZE}" \
    --input_channels "${INPUT_CHANNELS}" \
    --optimizer "${OPTIMIZER}" \
    --device "${DEVICE}" \
    --keyword "${KEYWORD}" \
    --outdir "${OUTDIR}" \
    --onnx_name "${ONNX_NAME}"

echo "Finished everything"