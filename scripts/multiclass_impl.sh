#!/usr/bin/env bash
set -euo pipefail # prevent if the code is wrong, stop immediately and print error message


KEYWORD="hw02" # the keyword, we set to hw02
DATA="${DATA:-data/Android_Malware.csv}" # this is all the default values, and we change them mannually
ETA="${ETA:-0.01}"
EPOCH="${EPOCH:-5000}"
TEST_SIZE="${TEST_SIZE:-0.2}"
SEED="${SEED:-42}"
OPTIMIZER="${OPTIMIZER:-adam}"
DEVICE="${DEVICE:-cuda}"     
OUTDIR="${OUTDIR:-results}"
ONNX_NAME="${ONNX_NAME:-multiclass_model.onnx}"
STAMP=$(date +"%Y%m%d%H%M%S")


mkdir -p "${OUTDIR}" # output directory

echo "We will run 5 times multiclass_impl.py"
for i in 1; do
  echo "This is the ${i} run of multiclass_impl.py"
  python multiclass_impl.py \
    --data "${DATA}" \
    --eta "${ETA}" \
    --epoch "${EPOCH}" \
    --test_size "${TEST_SIZE}" \
    --seed "${SEED}" \
    --optimizer "${OPTIMIZER}" \
    --device "${DEVICE}" \
    --keyword "${KEYWORD}" \
    --outdir "${OUTDIR}" \
    --save_onnx \
    --onnx_name "${ONNX_NAME}" \
    --stamp "${STAMP}"
done

echo "Finish running multiclass_impl.py 5 times, now we will run multiclass_eval.py to aggregate the results and plot the boxplot."
python multiclass_eval.py \
  --keyword "${KEYWORD}" \
  --outdir "${OUTDIR}" \
  --stamp "${STAMP}" \
  --recursive

echo "Finished everything"
