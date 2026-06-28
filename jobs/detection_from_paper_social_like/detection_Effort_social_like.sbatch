#!/bin/bash
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs2026
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

#SBATCH --output=/work/cvcs2026/resnet_gang/logs/%j_%x.out
#SBATCH --error=/work/cvcs2026/resnet_gang/logs/%j_%x.err

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Uso: sbatch detection_Effort_social_like.sbatch EFFORT_VARIANT WEIGHTS_PATH"
    echo "EFFORT_VARIANT: sdv14 | chameleon"
    exit 1
fi

EFFORT_VARIANT="$1"
WEIGHTS_PATH="$2"

case "${EFFORT_VARIANT}" in
    sdv14|chameleon)
        ;;
    *)
        echo "EFFORT_VARIANT non valida: ${EFFORT_VARIANT}"
        echo "Valori ammessi: sdv14, chameleon"
        exit 1
        ;;
esac

if [[ -z "${WEIGHTS_PATH}" ]]; then
    echo "WEIGHTS_PATH vuoto"
    exit 1
fi

if [[ ! -f "${WEIGHTS_PATH}" ]]; then
    echo "Checkpoint non trovato: ${WEIGHTS_PATH}"
    exit 1
fi

SCRIPT_NAME="detection_Effort_social_like.py"
SCRIPT_DIR="/homes/mlusvarghi/cvcs2026/scripts/detection_from_paper_social_like"

DEEPFAKEBENCH_ROOT="/work/cvcs2026/resnet_gang/external/Effort-AIGI-Detection/DeepfakeBench"
CONFIG_PATH="${DEEPFAKEBENCH_ROOT}/training/config/detector/effort.yaml"
JSONL_PATH="/work/cvcs2026/resnet_gang/datasets/json_standardized/json/openfake.jsonl"
OUTPUT_ROOT="/work/cvcs2026/resnet_gang/outputs/Effort/detection_from_paper_social_like"
DETECTOR_ID="effort_${EFFORT_VARIANT}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Config non trovato: ${CONFIG_PATH}"
    exit 1
fi

if [[ ! -f "${JSONL_PATH}" ]]; then
    echo "JSONL non trovato: ${JSONL_PATH}"
    exit 1
fi

if [[ ! -f "${SCRIPT_DIR}/${SCRIPT_NAME}" ]]; then
    echo "Script Python non trovato: ${SCRIPT_DIR}/${SCRIPT_NAME}"
    exit 1
fi

if [[ ! -f "${SCRIPT_DIR}/degradation_social_like.py" ]]; then
    echo "Modulo social-like non trovato: ${SCRIPT_DIR}/degradation_social_like.py"
    exit 1
fi

cd /work/cvcs2026/resnet_gang
mkdir -p logs
mkdir -p "${OUTPUT_ROOT}"

cd /homes/mlusvarghi/cvcs2026

module load python/3.10.16-gcc-11.4.0
source /work/cvcs2026/resnet_gang/env.sh

export HF_HOME=/work/cvcs2026/resnet_gang/.cache/huggingface
export TRANSFORMERS_CACHE=/work/cvcs2026/resnet_gang/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${DEEPFAKEBENCH_ROOT}/training:${DEEPFAKEBENCH_ROOT}:${SCRIPT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "START:              $(date --iso-8601=seconds)"
echo "JOB ID:             ${SLURM_JOB_ID}"
echo "JOB NAME:           ${SLURM_JOB_NAME}"
echo "HOST:               $(hostname)"
echo "EFFORT VARIANT:     ${EFFORT_VARIANT}"
echo "DETECTOR ID:        ${DETECTOR_ID}"
echo "WEIGHTS:            ${WEIGHTS_PATH}"
echo "CONFIG:             ${CONFIG_PATH}"
echo "OPENFAKE JSONL:     ${JSONL_PATH}"
echo "OUTPUT ROOT:        ${OUTPUT_ROOT}"
echo "SCRIPT:             ${SCRIPT_DIR}/${SCRIPT_NAME}"
echo "============================================================"

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("CUDA disponibile:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA non disponibile nel job")

print("GPU:", torch.cuda.get_device_name(0))
print("CUDA build:", torch.version.cuda)
PY

python -u "${SCRIPT_DIR}/${SCRIPT_NAME}" \
    --detector-id "${DETECTOR_ID}" \
    --config "${CONFIG_PATH}" \
    --weights "${WEIGHTS_PATH}" \
    --deepfakebench-root "${DEEPFAKEBENCH_ROOT}" \
    --jsonl "${JSONL_PATH}" \
    --output-root "${OUTPUT_ROOT}" \
    --batch-size 8 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --threshold 0.5 \
    --seed 1024 \
    --amp

echo "END: $(date --iso-8601=seconds)"