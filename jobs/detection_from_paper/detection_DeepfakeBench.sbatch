#!/bin/bash
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026
#SBATCH --qos=all_qos_dbg
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=/work/cvcs2026/resnet_gang/logs/%j_%x.out
#SBATCH --error=/work/cvcs2026/resnet_gang/logs/%j_%x.err

set -euo pipefail

: "${DETECTOR:?DETECTOR non impostato}"
: "${CONFIG_PATH:?CONFIG_PATH non impostato}"
: "${WEIGHTS_PATH:?WEIGHTS_PATH non impostato}"

PYTHON_SCRIPT="/homes/mlusvarghi/cvcs2026/scripts/detection_from_paper/detection_DeepfakeBench.py"

DEEPFAKEBENCH_ROOT="${DEEPFAKEBENCH_ROOT:-/work/cvcs2026/resnet_gang/external/DeepfakeBench}"

JSONL_PATH="${JSONL_PATH:-/work/cvcs2026/resnet_gang/datasets/json_standardized/json/openfake.jsonl}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/work/cvcs2026/resnet_gang/outputs/detectors_from_deepfakebench/detection_from_paper}"

BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
THRESHOLD="${THRESHOLD:-0.5}"
USE_AMP="${USE_AMP:-0}"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p /work/cvcs2026/resnet_gang/logs
mkdir -p "${OUTPUT_ROOT}"

cd /homes/mlusvarghi/cvcs2026

module purge
source /work/cvcs2026/resnet_gang/env.sh

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/work/cvcs2026/resnet_gang/.cache/huggingface
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TORCH_HOME=/work/cvcs2026/resnet_gang/.cache/torch

ARGS=(
    --detector "${DETECTOR}"
    --config "${CONFIG_PATH}"
    --weights "${WEIGHTS_PATH}"
    --deepfakebench-root "${DEEPFAKEBENCH_ROOT}"
    --jsonl "${JSONL_PATH}"
    --output-root "${OUTPUT_ROOT}"
    --batch-size "${BATCH_SIZE}"
    --workers "${NUM_WORKERS}"
    --threshold "${THRESHOLD}"
)

if [[ "${USE_AMP}" == "1" ]]; then
    ARGS+=(--amp)
fi

if [[ "${OVERWRITE}" == "1" ]]; then
    ARGS+=(--overwrite)
fi

echo "============================================================"
echo "DETECTOR           = ${DETECTOR}"
echo "CONFIG_PATH        = ${CONFIG_PATH}"
echo "WEIGHTS_PATH       = ${WEIGHTS_PATH}"
echo "DEEPFAKEBENCH_ROOT = ${DEEPFAKEBENCH_ROOT}"
echo "JSONL_PATH         = ${JSONL_PATH}"
echo "OUTPUT_ROOT        = ${OUTPUT_ROOT}"
echo "BATCH_SIZE         = ${BATCH_SIZE}"
echo "NUM_WORKERS        = ${NUM_WORKERS}"
echo "THRESHOLD          = ${THRESHOLD}"
echo "USE_AMP            = ${USE_AMP}"
echo "OVERWRITE          = ${OVERWRITE}"
echo "============================================================"

python "${PYTHON_SCRIPT}" "${ARGS[@]}"
