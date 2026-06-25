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

if [[ $# -ne 3 ]]; then
    echo "Uso:"
    echo "sbatch detection_DeepfakeBench.sbatch DETECTOR CONFIG_PATH WEIGHTS_PATH"
    exit 1
fi

DETECTOR="$1"
CONFIG_PATH="$2"
WEIGHTS_PATH="$3"

case "${DETECTOR}" in
    effort|clip|cnn_aug|xception|efficientnetb4|f3net|spsl|srm)
        ;;
    *)
        echo "Detector non valido: ${DETECTOR}"
        echo "Valori ammessi:"
        echo "effort, clip, cnn_aug, xception, efficientnetb4, f3net, spsl, srm"
        exit 1
        ;;
esac

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Config non trovato: ${CONFIG_PATH}"
    exit 1
fi

if [[ ! -f "${WEIGHTS_PATH}" ]]; then
    echo "Checkpoint non trovato: ${WEIGHTS_PATH}"
    exit 1
fi

SCRIPT_NAME="detection_DeepfakeBench.py"

DEEPFAKEBENCH_ROOT="/work/cvcs2026/resnet_gang/external/DeepfakeBench"

JSONL_PATH="/work/cvcs2026/resnet_gang/datasets/json_standardized/json/openfake.jsonl"

OUTPUT_ROOT="/work/cvcs2026/resnet_gang/outputs/detectors_from_deepfakebench/detection_from_paper"

BATCH_SIZE=32
NUM_WORKERS=8
THRESHOLD=0.5

cd /work/cvcs2026/resnet_gang

mkdir -p logs
mkdir -p "${OUTPUT_ROOT}"

cd /homes/mlusvarghi/cvcs2026

module load python/3.10.16-gcc-11.4.0

source /work/cvcs2026/resnet_gang/env.sh

export HF_HOME=/work/cvcs2026/resnet_gang/.cache/huggingface
export TRANSFORMERS_CACHE=/work/cvcs2026/resnet_gang/.cache/huggingface
export TORCH_HOME=/work/cvcs2026/resnet_gang/.cache/torch
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "START:               $(date --iso-8601=seconds)"
echo "JOB ID:              ${SLURM_JOB_ID}"
echo "JOB NAME:            ${SLURM_JOB_NAME}"
echo "HOST:                $(hostname)"
echo "DETECTOR:            ${DETECTOR}"
echo "CONFIG:              ${CONFIG_PATH}"
echo "WEIGHTS:             ${WEIGHTS_PATH}"
echo "DEEPFAKEBENCH ROOT:  ${DEEPFAKEBENCH_ROOT}"
echo "JSONL:               ${JSONL_PATH}"
echo "OUTPUT:              ${OUTPUT_ROOT}"
echo "SCRIPT:              scripts/detection_from_paper/${SCRIPT_NAME}"
echo "============================================================"

python -u "scripts/detection_from_paper/${SCRIPT_NAME}" \
    --detector "${DETECTOR}" \
    --config "${CONFIG_PATH}" \
    --weights "${WEIGHTS_PATH}" \
    --deepfakebench-root "${DEEPFAKEBENCH_ROOT}" \
    --jsonl "${JSONL_PATH}" \
    --output-root "${OUTPUT_ROOT}" \
    --batch-size "${BATCH_SIZE}" \
    --workers "${NUM_WORKERS}" \
    --threshold "${THRESHOLD}"

echo "END: $(date --iso-8601=seconds)"